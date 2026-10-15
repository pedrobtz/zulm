test_that("GGUF round-trips values, shapes and order at f32", {
  p <- zl_params()
  zl_param(p, "m", c(3, 1), init = zl_init_normal(1))
  zl_param(p, "t", c(2, 3, 4), init = zl_init_uniform(2))
  zl_param(p, "q", c(1, 2, 1, 2), init = zl_init_normal(1))
  zl_param(p, "v", 7, init = zl_init_normal(1))
  path <- withr::local_tempfile(fileext = ".gguf")
  expect_identical(zl_save(p, path), path)
  q <- zl_load(path)
  expect_identical(zl_param_names(q), zl_param_names(p))
  for (n in zl_param_names(p)) {
    expect_identical(zl_param_get(q, n), zl_param_get(p, n))
  }
  expect_identical(dim(zl_param_get(q, "m")), c(3L, 1L))
  expect_identical(dim(zl_param_get(q, "q")), c(1L, 2L, 1L, 2L))
})

test_that("saving as f16 or bf16 rounds, and loading widens to f32", {
  withr::local_seed(3)
  p <- zl_params()
  zl_param(p, "w", c(8, 5))
  zl_param_set(p, "w", matrix(runif(40, -3, 3), 8, 5))
  for (type in c("f16", "bf16")) {
    path <- withr::local_tempfile(fileext = ".gguf")
    zl_save(p, path, type = type)
    q <- zl_load(path)
    h <- zl_params()
    zl_param(h, "w", c(8, 5), type = type)
    zl_param_set(h, "w", zl_param_get(p, "w"))
    expect_identical(zl_param_get(q, "w"), zl_param_get(h, "w"))
    expect_identical(zl_param_tensor(q, "w")$type, "f32")
  }
})

test_that("half-precision stores save at f32 exactly", {
  p <- zl_params()
  zl_param(p, "h", c(4, 4), type = "f16")
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, path)
  expect_identical(zl_param_get(zl_load(path), "h"), zl_param_get(p, "h"))
})

test_that("metadata round-trips", {
  p <- make_store()
  meta <- list(
    name = "tiny", tags = c("a", "b"), width = 64L, sizes = c(1L, 2L, 3L),
    lr = 0.001, grid = c(0.5, 1.5), trained = TRUE, flags = c(TRUE, FALSE)
  )
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, path, metadata = meta)
  expect_identical(attr(zl_load(path), "metadata"), meta)
  zl_save(p, path)
  expect_identical(attr(zl_load(path), "metadata"), structure(list(), names = character()))
})

test_that("an empty store round-trips", {
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(zl_params(), path)
  expect_identical(zl_param_names(zl_load(path)), character())
})

test_that("bad metadata is refused", {
  p <- make_store()
  path <- withr::local_tempfile(fileext = ".gguf")
  expect_error(zl_save(p, path, metadata = list(1)), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, metadata = list(a = NA)), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, metadata = list(a = list(1))), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, metadata = list(a = character())), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, metadata = list(zulm.x = 1)), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, metadata = "x"), class = "zulm_invalid_argument")
  expect_error(zl_save(p, path, type = "q4"), class = "zulm_type_error")
  expect_false(file.exists(path))
})

test_that("unreadable files are zulm_file_error", {
  dir <- withr::local_tempdir()
  expect_error(zl_load(file.path(dir, "missing.gguf")), class = "zulm_file_error")
  expect_error(zl_load(dir), class = "zulm_file_error")
  junk <- file.path(dir, "junk.gguf")
  writeBin(as.raw(0:255), junk)
  expect_error(zl_load(junk), class = "zulm_file_error")
  good <- file.path(dir, "good.gguf")
  p <- zl_params()
  zl_param(p, "w", c(50, 20))
  zl_save(p, good)
  bytes <- readBin(good, "raw", file.size(good))
  # the last 100 bytes always include tensor data: padding is under 32
  for (n in c(4L, 30L, 100L, length(bytes) - 100L)) {
    cut <- file.path(dir, sprintf("cut%d.gguf", n))
    writeBin(bytes[seq_len(n)], cut)
    expect_error(zl_load(cut), class = "zulm_file_error")
  }
  expect_error(zl_save(make_store(), file.path(dir, "no", "such", "dir.gguf")), class = "zulm_file_error")
})

test_that("a file whose tensors exceed the memory cap is refused", {
  p <- zl_params()
  zl_param(p, "w", c(100, 100))
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, path)
  withr::local_options(zulm.max_bytes = 1000)
  expect_error(zl_load(path), class = "zulm_memory_error")
})

test_that("a metadata type outside GGUF's range is zulm_file_error (patch P11)", {
  p <- zl_params()
  zl_param(p, "w", 2)
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, path, metadata = list(zzzz = 1))
  bytes <- readBin(path, "raw", file.size(path))
  # key-value pair: key length (u64) and key, then the value's type (i32):
  # put 999 where the type is
  key <- c(as.raw(c(4, 0, 0, 0, 0, 0, 0, 0)), charToRaw("zzzz"))
  at <- which(vapply(seq_len(length(bytes) - length(key)), function(i) {
    identical(bytes[i:(i + length(key) - 1L)], key)
  }, logical(1)))
  expect_length(at, 1L)
  type_at <- at + length(key)
  bytes[type_at:(type_at + 3L)] <- writeBin(999L, raw(), size = 4, endian = "little")
  bad <- withr::local_tempfile(fileext = ".gguf")
  writeBin(bytes, bad)
  expect_error(zl_load(bad), class = "zulm_file_error")
})

test_that("a tensor type outside ggml's range is zulm_file_error", {
  p <- zl_params()
  zl_param(p, "zzzz", c(2, 2))
  path <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, path)
  bytes <- readBin(path, "raw", file.size(path))
  # tensor info: name length (u64) and name, n_dims (u32), ne (n_dims x
  # i64), type (i32): put 999 where the type is
  key <- c(as.raw(c(4, 0, 0, 0, 0, 0, 0, 0)), charToRaw("zzzz"))
  at <- which(vapply(seq_len(length(bytes) - length(key)), function(i) {
    identical(bytes[i:(i + length(key) - 1L)], key)
  }, logical(1)))
  expect_length(at, 1L)
  dims_at <- at + length(key)
  n_dims <- readBin(bytes[dims_at:(dims_at + 3L)], "integer", size = 4, endian = "little")
  type_at <- dims_at + 4L + 8L * n_dims
  bytes[type_at:(type_at + 3L)] <- writeBin(999L, raw(), size = 4, endian = "little")
  bad <- withr::local_tempfile(fileext = ".gguf")
  writeBin(bytes, bad)
  expect_error(zl_load(bad), class = "zulm_file_error")
})
