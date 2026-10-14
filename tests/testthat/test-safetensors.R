test_that("safetensors round-trips at f32, f16 and bf16", {
  withr::local_seed(80)
  p <- zl_params()
  zl_param(p, "a", c(3, 4), init = zl_init_normal(1))
  zl_param(p, "b", 5, init = zl_init_uniform(2))
  zl_param(p, "c", c(2, 1, 3), init = zl_init_normal(1))
  for (type in c("f32", "f16", "bf16")) {
    path <- withr::local_tempfile(fileext = ".safetensors")
    zl_save_safetensors(p, path, type = type, metadata = list(by = "zulm"))
    st <- zl_safetensors_read(path)
    expect_named(st, c("a", "b", "c"))
    expect_identical(attr(st, "metadata"), list(by = "zulm"))
    ref <- zl_params()
    for (nm in zl_param_names(p)) {
      d <- dim(zl_param_get(p, nm))
      zl_param(ref, nm, if (is.null(d)) length(zl_param_get(p, nm)) else d, type = type, init = zl_init_zeros())
      zl_param_set(ref, nm, zl_param_get(p, nm))
    }
    for (nm in zl_param_names(p)) {
      want <- zl_param_get(ref, nm)
      expect_identical(as.vector(st[[nm]]), as.vector(want))
    }
    expect_identical(dim(st$a), c(3L, 4L))
    q <- zl_params()
    zl_param(q, "a", c(3, 4), init = zl_init_zeros(), type = type)
    filled <- zl_load_safetensors(q, path)
    expect_identical(filled, "a")
    expect_identical(zl_param_get(q, "a"), zl_param_get(ref, "a"))
  }
})

test_that("a PyTorch file loads with [out, in] as c(in, out), no transposition", {
  fx <- zl_load(test_path("fixtures", "block.gguf"))
  st <- zl_safetensors_read(test_path("fixtures", "block.safetensors"))
  # nn.Linear's weight [out, in] is zulm's c(in, out), the same bytes
  expect_identical(st[["blk.attn.q.weight"]], zl_param_get(fx, "blk.attn.q.weight"))
  m <- attr(st, "metadata")
  p <- zl_params()
  blk <- zl_block(p, "blk", dim = as.integer(m$dim), heads = as.integer(m$heads), hidden = as.integer(m$hidden))
  filled <- zl_load_safetensors(p, test_path("fixtures", "block.safetensors"))
  expect_setequal(filled, zl_param_names(p))
  x <- st[["input.x"]]
  out <- zl_forward(zl_graph(function(i) blk(i$x), p, list(x = dim(x))), list(x = x))$output
  expect_equal(out, st[["output.y"]], tolerance = 1e-5)
})

test_that("a name map fills differently named parameters", {
  p <- zl_params()
  zl_param(p, "q", c(16, 16), init = zl_init_zeros())
  filled <- zl_load_safetensors(p, test_path("fixtures", "block.safetensors"), map = c(blk.attn.q.weight = "q"))
  expect_identical(filled, "q")
  expect_identical(zl_param_get(p, "q"), zl_safetensors_read(test_path("fixtures", "block.safetensors"), "blk.attn.q.weight")[[1]])
  expect_error(zl_load_safetensors(p, test_path("fixtures", "block.safetensors"), map = c(blk.attn.q.bias = "q")), class = "zulm_shape_error")
  expect_error(zl_load_safetensors(p, test_path("fixtures", "block.safetensors"), map = c(nope = "q")), class = "zulm_invalid_argument")
})

test_that("the Toto 2.0 4M header from the Hub parses and validates", {
  path <- test_path("fixtures", "toto-2.0-4m-header.safetensors")
  bytes <- readBin(path, "raw", file.size(path))
  n <- readBin(bytes[1:8], "double", size = 8)
  n <- sum(as.double(as.integer(bytes[1:8])) * 256^(0:7))
  json <- rawToChar(bytes[9:(8 + n)])
  ends <- vapply(zujson::json_parse(json), function(e) if (is.list(e) && !is.null(e$data_offsets)) e$data_offsets[2] else 0, numeric(1))
  h <- zulm:::zlm_st_check(json, max(ends))
  expect_identical(nrow(h$tensors), 48L)
  expect_true(all(h$tensors$dtype == "F32"))
  expect_true("output_head.param_projection.proj.linear1.weight" %in% h$tensors$name)
  # the file holds no data, so reading it is a file error
  expect_error(zl_safetensors_read(path), class = "zulm_file_error")
})

test_that("malformed safetensors files are zulm_file_error", {
  dir <- withr::local_tempdir()
  write_st <- function(name, json, data = raw(), len = NULL) {
    j <- charToRaw(json)
    n <- if (is.null(len)) length(j) else len
    head <- as.raw(as.integer((n %/% 256^(0:7)) %% 256))
    path <- file.path(dir, name)
    writeBin(c(head, j, data), path)
    path
  }
  good <- '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
  expect_length(zl_safetensors_read(write_st("good", good, raw(8)))$a, 2L)
  bad <- list(
    short = "{",
    notjson = "{nope",
    array = "[1,2]",
    badtype = '{"a":{"dtype":"F99","shape":[2],"data_offsets":[0,8]}}',
    badsize = '{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}',
    reversed = '{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}',
    gap = '{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
    overlap = '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
    extra = '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"x":1}}',
    negshape = '{"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}',
    meta = '{"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
  )
  for (nm in names(bad)) {
    expect_error(zl_safetensors_read(write_st(nm, bad[[nm]], raw(8))), class = "zulm_file_error", label = nm)
  }
  expect_error(zl_safetensors_read(write_st("huge", good, raw(8), len = 2^40)), class = "zulm_file_error")
  expect_error(zl_safetensors_read(write_st("trunc", good, raw(4))), class = "zulm_file_error")
  expect_error(zl_safetensors_read(file.path(dir, "missing")), class = "zulm_file_error")
  tiny <- file.path(dir, "tiny")
  writeBin(as.raw(1:3), tiny)
  expect_error(zl_safetensors_read(tiny), class = "zulm_file_error")
  # unsupported types are read errors, not file errors
  u8 <- write_st("u8", '{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}', raw(8))
  expect_error(zl_safetensors_read(u8), class = "zulm_type_error")
})

test_that("F64 rounds to f32, and integers read as integers", {
  dir <- withr::local_tempdir()
  path <- file.path(dir, "mixed.safetensors")
  json <- '{"d":{"dtype":"F64","shape":[2],"data_offsets":[0,16]},"i":{"dtype":"I64","shape":[2],"data_offsets":[16,32]},"j":{"dtype":"I32","shape":[2],"data_offsets":[32,40]}}'
  j <- charToRaw(json)
  head <- as.raw(as.integer((length(j) %/% 256^(0:7)) %% 256))
  body <- c(writeBin(c(1 / 3, 1e40), raw(), size = 8), writeBin(c(5L, 0L, -7L, -1L), raw(), size = 4), writeBin(c(9L, -2L), raw(), size = 4))
  writeBin(c(head, j, body), path)
  expect_warning(st <- zl_safetensors_read(path), class = "zulm_range_warning")
  expect_identical(st$d, c(f32(1 / 3), Inf))
  expect_identical(st$i, c(5L, -7L))
  expect_identical(st$j, c(9L, -2L))
})

test_that("half-precision weights stay in their type, and cost the documented precision", {
  withr::local_seed(81)
  path <- withr::local_tempfile(fileext = ".gguf")
  p <- zl_params()
  zl_param(p, "w", c(256, 128), init = zl_init_normal(1 / 16))
  zl_save(p, path, type = "f16")
  expect_identical(zl_param_names(zl_load(path), types = TRUE), c(w = "f32"))
  kept <- zl_load(path, keep_types = TRUE)
  expect_identical(zl_param_names(kept, types = TRUE), c(w = "f16"))
  x <- matrix(rnorm(256 * 64), 256)
  w <- zl_param_get(p, "w")
  prod_with <- function(type) {
    q <- zl_params()
    zl_param(q, "w", dim(w), type = type, init = zl_init_zeros())
    zl_param_set(q, "w", w)
    zl_forward(zl_graph(function(i) zl_crossprod(zl_param_tensor(q, "w"), i$x), q, list(x = dim(x))), list(x = x))$output
  }
  ref <- prod_with("f32")
  rel <- function(a) max(abs(a - ref)) / max(abs(ref))
  # the bounds the documentation states (design §6.9): if a re-vendoring
  # changes them, the documentation must change too
  expect_gt(rel(prod_with("f16")), 2e-4)
  expect_lt(rel(prod_with("f16")), 5e-3)
  expect_gt(rel(prod_with("bf16")), 5e-4)
  expect_lt(rel(prod_with("bf16")), 1e-2)
})

test_that("mutated headers give a zulm error or a valid read, never anything else", {
  withr::local_seed(82)
  dir <- withr::local_tempdir()
  json <- '{"__metadata__":{"k":"v"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"F16","shape":[4],"data_offsets":[24,32]}}'
  data <- as.raw(sample(0:255, 32, replace = TRUE))
  alphabet <- strsplit('{}[]",:0123456789-.eE FBIU_abdtyhpesoflm', "")[[1]]
  n <- if (identical(Sys.getenv("NOT_CRAN"), "true")) 600L else 150L
  outcomes <- character(n)
  for (k in seq_len(n)) {
    j <- strsplit(json, "")[[1]]
    for (m in seq_len(sample(1:4, 1))) {
      at <- sample(length(j), 1)
      switch(sample(4, 1),
        j[at] <- sample(alphabet, 1),
        j <- append(j, sample(alphabet, 1), after = at),
        j <- j[-at],
        j <- j[seq_len(at)]
      )
    }
    body <- charToRaw(paste(j, collapse = ""))
    len <- length(body)
    if (k %% 7 == 0) len <- len + sample(-3:3, 1)
    if (len < 0) len <- 0
    head <- as.raw(as.integer((len %/% 256^(0:7)) %% 256))
    path <- file.path(dir, sprintf("m%03d", k))
    writeBin(c(head, body, data[seq_len(sample(0:32, 1))]), path)
    outcomes[[k]] <- tryCatch({
      zl_safetensors_read(path)
      "read"
    }, zulm_error = function(e) "zulm_error", error = function(e) paste("other:", conditionMessage(e)))
  }
  expect_true(all(outcomes %in% c("read", "zulm_error")), info = paste(unique(outcomes[!outcomes %in% c("read", "zulm_error")]), collapse = "; "))
  expect_true(any(outcomes == "zulm_error"))
})
