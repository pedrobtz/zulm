# Batches from Arrow streams (design §7.3, §15 criterion 12).

zlm_call <- function(...) getFromNamespace("zlm_call", "zulm")(...)

# A stream of `batches` record batches of `size` rows of one float32 column
# "x" counting from 0, whose releases are counted in `counts`.
counting_stream <- function(batches, size, counts) {
  zlm_call("zlm_arrow_test_stream", as.integer(batches), as.integer(size), counts)
}

# Every batch of an iterator, as a list of lists.
drain <- function(it) {
  out <- list()
  while (!is.null(b <- zl_next(it))) out[[length(out) + 1L]] <- b
  out
}

# nanoarrow converts R vectors to int8, uint8 and int16 only through the 'arrow'
# package, which zulm does not suggest: those columns are built from their
# bytes, and every other column by nanoarrow.
small_int_array <- function(v, type, width) {
  a <- nanoarrow::nanoarrow_array_init(type)
  bytes <- writeBin(as.integer(v), raw(), size = width, endian = "little")
  nanoarrow::nanoarrow_array_modify(a, list(
    length = length(v), null_count = 0,
    buffers = list(NULL, nanoarrow::as_nanoarrow_buffer(bytes))
  ))
}

batch_array <- function(df, schema) {
  children <- lapply(names(schema$children), function(nm) {
    type <- schema$children[[nm]]
    width <- c(c = 1L, C = 1L, s = 2L)[type$format]
    if (!is.na(width)) {
      small_int_array(df[[nm]], type, width)
    } else {
      nanoarrow::as_nanoarrow_array(df[[nm]], schema = type)
    }
  })
  names(children) <- names(schema$children)
  nanoarrow::nanoarrow_array_modify(nanoarrow::nanoarrow_array_init(schema), list(
    length = nrow(df), null_count = 0, children = children
  ))
}

na_stream <- function(batches, schema) {
  arrays <- lapply(batches, batch_array, schema = schema)
  nanoarrow::basic_array_stream(arrays, schema = schema)
}

test_that("record batches of any size become batches of exactly `rows`", {
  for (size in c(3, 4, 5, 9)) {
    counts <- integer(2)
    it <- zl_arrow_batches(counting_stream(5, size, counts), list(x = "x"), rows = 4)
    got <- drain(it)
    n <- 5 * size
    expect_length(got, n %/% 4)
    expect_identical(unname(unlist(lapply(got, `[[`, "x"))), as.double(seq_len(4 * (n %/% 4)) - 1))
    expect_identical(dim(got[[1]]$x), c(1L, 4L))
    expect_identical(counts, c(1L, 5L))
    expect_null(zl_next(it))
  }
  # padded: the remainder comes as a last batch under a row mask
  counts <- integer(2)
  it <- zl_arrow_batches(counting_stream(2, 3, counts), list(x = "x"), rows = 4, last = "pad")
  got <- drain(it)
  expect_length(got, 2L)
  expect_identical(as.vector(got[[2]]$x), c(4, 5, 0, 0))
  expect_identical(got[[2]]$row_mask, c(1, 1, 0, 0))
  expect_identical(got[[1]]$row_mask, c(1, 1, 1, 1))
  # an exact multiple adds no padded batch
  it <- zl_arrow_batches(counting_stream(2, 2, integer(2)), list(x = "x"), rows = 4, last = "pad")
  expect_length(drain(it), 1L)
})

test_that("the stream and every array are released exactly once", {
  counts <- integer(2)
  s <- counting_stream(4, 3, counts)
  it <- zl_arrow_batches(s, list(x = "x"), rows = 2)
  zl_next(it)                                        # two batches of three read
  zl_next(it)
  expect_identical(counts, c(0L, 1L))
  expect_error(zl_arrow_batches(s, list(x = "x"), rows = 2), "consumed", class = "zulm_invalid_argument")
  rm(it, s)
  gc()
  expect_identical(counts, c(1L, 2L))
  # read to the end, then collected
  counts <- integer(2)
  it <- zl_arrow_batches(counting_stream(3, 2, counts), list(x = "x"), rows = 6)
  drain(it)
  expect_identical(counts, c(1L, 3L))
  rm(it)
  gc()
  expect_identical(counts, c(1L, 3L))
  # an iterator made and never read
  counts <- integer(2)
  it <- zl_arrow_batches(counting_stream(3, 2, counts), list(x = "x"), rows = 6)
  rm(it)
  gc()
  expect_identical(counts, c(1L, 0L))
})

test_that("an iterator is checked against its map", {
  s <- function() counting_stream(1, 2, integer(2))
  expect_error(zl_arrow_batches(s(), list(y = "nope"), rows = 2), "no column 'nope'", class = "zulm_arrow_error")
  expect_error(zl_arrow_batches(s(), list(x = zl_int("x")), rows = 2), "floating-point", class = "zulm_arrow_error")
  expect_error(zl_arrow_batches(s(), list("x"), rows = 2), class = "zulm_invalid_argument")
  expect_error(zl_arrow_batches(s(), list(x = "x", x_mask = "x"), rows = 2, na = "mask"), "clash", class = "zulm_invalid_argument")
  expect_error(zl_arrow_batches(s(), list(row_mask = "x"), rows = 2, last = "pad"), "clash", class = "zulm_invalid_argument")
  expect_error(zl_arrow_batches(s(), list(x = character()), rows = 2), class = "zulm_invalid_argument")
  expect_error(zl_arrow_batches(s(), list(x = "x"), rows = 0), class = "zulm_invalid_argument")
  expect_error(zl_arrow_batches(s(), list(x = "x"), rows = 2.5), class = "zulm_invalid_argument")
  expect_error(zl_next(list()), class = "zulm_invalid_argument")
  expect_output(print(zl_arrow_batches(s(), list(x = "x"), rows = 2)), "2 rows per batch")
})

test_that("without nanoarrow, other objects are refused with a companion error", {
  local_mocked_bindings(zlm_has_nanoarrow = function() FALSE)
  expect_error(
    zl_arrow_batches(data.frame(x = 1), list(x = "x"), rows = 1),
    "nanoarrow", class = "zulm_companion_error"
  )
  # a nanoarrow_array_stream itself needs no nanoarrow
  it <- zl_arrow_batches(counting_stream(1, 2, integer(2)), list(x = "x"), rows = 2)
  expect_identical(as.vector(zl_next(it)$x), c(0, 1))
})

test_that("every supported type is read, and others are refused", {
  skip_if_not_installed("nanoarrow")
  schema <- nanoarrow::na_struct(list(
    f32 = nanoarrow::na_float(), f64 = nanoarrow::na_double(),
    i8 = nanoarrow::na_int8(), i16 = nanoarrow::na_int16(), i32 = nanoarrow::na_int32(),
    i64 = nanoarrow::na_int64(), b = nanoarrow::na_bool(),
    d = nanoarrow::na_dictionary(nanoarrow::na_string(), nanoarrow::na_int32())
  ))
  lv <- c("lo", "mid", "hi")
  b1 <- data.frame(
    f32 = c(0.1, -2), f64 = c(1 / 3, 1e40), i8 = c(-128, 127), i16 = c(300L, -300L),
    i32 = c(2147483647L, -5L), i64 = c(2^40, -7), b = c(TRUE, FALSE),
    d = factor(c("hi", "lo"), levels = lv)
  )
  b2 <- data.frame(
    f32 = 3, f64 = -1, i8 = 1, i16 = 2L, i32 = 3L, i64 = 4, b = TRUE, d = factor("mid", levels = lv)
  )
  num <- c("f32", "f64", "i8", "i16", "i32", "i64", "b")
  it <- zl_arrow_batches(na_stream(list(b1, b2), schema), list(x = num, d = zl_int("d")), rows = 3)
  got <- zl_next(it)
  want <- rbind(
    f32 = f32(c(0.1, -2, 3)), f64 = f32(c(1 / 3, Inf, -1)), i8 = c(-128, 127, 1),
    i16 = c(300, -300, 2), i32 = f32(c(2147483647, -5, 3)), i64 = f32(c(2^40, -7, 4)),
    b = c(1, 0, 1)
  )
  expect_identical(got$x, unname(want))
  expect_identical(got$d, c(2L, 0L, 1L))
  # integer inputs from integer and boolean columns; int64 within range
  it <- zl_arrow_batches(na_stream(list(b2), schema), list(i = zl_int("i8", "i16", "i64", "b")), rows = 1)
  expect_identical(zl_next(it)$i, matrix(c(1L, 2L, 4L, 1L), 4, 1))
  # int64 beyond the 32-bit range in an integer input
  it <- zl_arrow_batches(na_stream(list(b1), schema), list(i = zl_int("i64")), rows = 2)
  err <- expect_error(zl_next(it), "32-bit", class = "zulm_arrow_error")
  expect_identical(err$column, "i64")
  expect_identical(err$row, 1)
  # a dictionary cannot fill an f32 input; unsupported types are refused
  expect_error(zl_arrow_batches(na_stream(list(b1), schema), list(x = "d"), rows = 1), "zl_int", class = "zulm_arrow_error")
  s2 <- nanoarrow::na_struct(list(u = nanoarrow::na_uint8(), s = nanoarrow::na_string()))
  df2 <- data.frame(u = 1, s = "a")
  expect_error(zl_arrow_batches(na_stream(list(df2), s2), list(x = "u"), rows = 1), "does not read", class = "zulm_arrow_error")
  expect_error(zl_arrow_batches(na_stream(list(df2), s2), list(x = "s"), rows = 1), "does not read", class = "zulm_arrow_error")
})

test_that("missing values are errors with column and row, or masked", {
  skip_if_not_installed("nanoarrow")
  schema <- nanoarrow::na_struct(list(a = nanoarrow::na_double(), k = nanoarrow::na_int32()))
  b1 <- data.frame(a = c(1, 2, 3), k = c(1L, 2L, 3L))
  b2 <- data.frame(a = c(4, NA, 6), k = c(4L, 5L, NA))
  it <- zl_arrow_batches(na_stream(list(b1, b2), schema), list(x = "a", k = zl_int("k")), rows = 2)
  zl_next(it)
  zl_next(it)
  err <- expect_error(zl_next(it), "Column 'a' is missing at row 5", class = "zulm_arrow_error")
  expect_identical(err$column, "a")
  expect_identical(err$row, 5)
  it <- zl_arrow_batches(na_stream(list(b1, b2), schema), list(x = "a", k = zl_int("k")), rows = 3, na = "mask")
  zl_next(it)
  got <- zl_next(it)
  expect_identical(as.vector(got$x), c(4, 0, 6))
  expect_identical(as.vector(got$x_mask), c(1, 0, 1))
  expect_identical(got$k, c(4L, 5L, 0L))
  expect_identical(got$k_mask, c(1, 1, 0))
})

test_that("a dictionary that changes between record batches is an error", {
  skip_if_not_installed("nanoarrow")
  schema <- nanoarrow::na_struct(list(d = nanoarrow::na_dictionary(nanoarrow::na_string())))
  b1 <- data.frame(d = factor(c("a", "b")))
  same <- data.frame(d = factor(c("b", "b"), levels = c("a", "b")))
  other <- data.frame(d = factor(c("c", "a")))
  it <- zl_arrow_batches(na_stream(list(b1, same), schema), list(d = zl_int("d")), rows = 4)
  expect_identical(zl_next(it)$d, c(0L, 1L, 1L, 1L))
  it <- zl_arrow_batches(na_stream(list(b1, other), schema), list(d = zl_int("d")), rows = 4)
  err <- expect_error(zl_next(it), "changes at row 3", class = "zulm_arrow_error")
  expect_identical(err$column, "d")
})

test_that("array offsets are honoured", {
  skip_if_not_installed("nanoarrow")
  schema <- nanoarrow::na_struct(list(a = nanoarrow::na_double()))
  arr <- nanoarrow::as_nanoarrow_array(data.frame(a = c(10, 11, 12, 13, 14)), schema = schema)
  sliced <- nanoarrow::nanoarrow_array_modify(arr, list(offset = 2, length = 3))
  it <- zl_arrow_batches(nanoarrow::basic_array_stream(list(sliced), schema = schema), list(x = "a"), rows = 3)
  expect_identical(as.vector(zl_next(it)$x), c(12, 13, 14))
})

test_that("training from an iterator equals training from the same rows as arrays", {
  skip_if_not_installed("nanoarrow")
  set.seed(5)
  n <- 40
  df <- data.frame(a = stats::rnorm(n), b = stats::rnorm(n), y = stats::rnorm(n), k = sample(0:2, n, TRUE))
  setup <- function() {
    p <- zl_params()
    zl_param(p, "w", c(2, 1), init = zl_init_normal(sd = 0.1))
    zl_param(p, "e", c(1, 3), init = zl_init_normal(sd = 0.1))
    g <- zl_graph(function(x) {
      pred <- zl_crossprod(zl_param_tensor(p, "w"), x$x) + zl_get_rows(zl_param_tensor(p, "e"), x$k)
      list(loss = zl_mse(pred, x$y))
    }, p, list(x = c(2, 8), y = c(1, 8), k = zl_int(8)), loss = "loss")
    list(p = p, tr = zl_trainer(g, zl_adamw(lr = 0.01)))
  }
  a <- setup()
  # record batches of 5, against batches of 8
  schema <- nanoarrow::na_struct(list(a = nanoarrow::na_double(), b = nanoarrow::na_double(),
                                      y = nanoarrow::na_double(), k = nanoarrow::na_int32()))
  chunks <- split(df, rep(seq_len(n / 5), each = 5))
  df$k <- as.integer(df$k)
  chunks <- lapply(chunks, function(d) transform(d, k = as.integer(k)))
  it <- zl_arrow_batches(na_stream(chunks, schema), list(x = c("a", "b"), y = "y", k = zl_int("k")), rows = 8)
  la <- c()
  while (!is.null(l <- zl_step(a$tr, it))) la <- c(la, l)
  expect_null(zl_step(a$tr, it))
  b <- setup()
  lb <- c()
  for (i in seq_len(n / 8)) {
    r <- (i - 1) * 8 + 1:8
    lb <- c(lb, zl_step(b$tr, list(x = rbind(df$a[r], df$b[r]), y = matrix(df$y[r], 1), k = df$k[r])))
  }
  expect_identical(la, lb)
  expect_identical(zl_param_get(a$p, "w"), zl_param_get(b$p, "w"))
  expect_identical(zl_param_get(a$p, "e"), zl_param_get(b$p, "e"))
  # and zl_forward() from an iterator
  it <- zl_arrow_batches(na_stream(chunks, schema), list(x = c("a", "b"), y = "y", k = zl_int("k")), rows = 8)
  f1 <- zl_forward(a$tr$graph, it)
  expect_identical(f1$loss, zl_forward(a$tr$graph, list(x = rbind(df$a[1:8], df$b[1:8]), y = matrix(df$y[1:8], 1), k = df$k[1:8]))$loss)
})

test_that("an iterator must match the graph it fills", {
  p <- zl_params()
  g <- zl_graph(function(x) zl_sum(x$x), p, list(x = c(1, 4)))
  expect_false(is.null(zl_forward(g, zl_arrow_batches(counting_stream(1, 4, integer(2)), list(x = "x"), rows = 4))))
  expect_error(zl_forward(g, zl_arrow_batches(counting_stream(1, 4, integer(2)), list(x = "x"), rows = 2)), class = "zulm_shape_error")
  expect_error(zl_forward(g, zl_arrow_batches(counting_stream(1, 4, integer(2)), list(z = "x"), rows = 4)), "no input 'z'", class = "zulm_invalid_argument")
  expect_error(zl_forward(g, zl_arrow_batches(counting_stream(1, 4, integer(2)), list(x = "x"), rows = 4, last = "pad")),
               "inputs", class = "zulm_invalid_argument")
  gc <- zl_graph_cache(function(x) zl_sum(x$x), p)
  expect_error(zl_forward(gc, zl_arrow_batches(counting_stream(1, 4, integer(2)), list(x = "x"), rows = 4)), class = "zulm_invalid_argument")
})
