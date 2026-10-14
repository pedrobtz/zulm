# Hostile input (design §11): every case is a documented zulm condition and
# none reaches a ggml assertion.

test_that("store arguments are checked", {
  p <- make_store()
  expect_error(zl_param_get(list(), "w"), class = "zulm_invalid_argument")
  expect_error(zl_param_get(p, "nope"), class = "zulm_invalid_argument")
  expect_error(zl_param_get(p, NA_character_), class = "zulm_invalid_argument")
  expect_error(zl_param_get(p, c("w", "b")), class = "zulm_invalid_argument")
  expect_error(zl_param(p, "w", 2), class = "zulm_invalid_argument")
  expect_error(zl_param(p, strrep("x", 64), 2), class = "zulm_invalid_argument")
  expect_error(zl_param(p, "", 2), class = "zulm_invalid_argument")
  expect_error(zl_param(p, "x", 2, init = "normal"), class = "zulm_invalid_argument")
  expect_error(zl_param(p, "x", 2, type = "f64"), class = "zulm_type_error")
  expect_error(zl_init_normal(sd = -1), class = "zulm_invalid_argument")
  expect_error(zl_init_uniform(a = NA), class = "zulm_invalid_argument")
})

test_that("dimensions are checked", {
  p <- zl_params()
  expect_error(zl_param(p, "x", c(1, 2, 3, 4, 5)), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", c(2, 0)), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", integer()), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", c(2, NA)), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", 2.5), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", -1), class = "zulm_shape_error")
  expect_error(zl_param(p, "x", "3"), class = "zulm_shape_error")
  expect_identical(zl_param_names(p), character())
})

test_that("values are checked", {
  p <- make_store()
  expect_error(zl_param_set(p, "w", matrix("a", 3, 2)), class = "zulm_type_error")
  expect_error(zl_param_set(p, "w", list(1)), class = "zulm_type_error")
  expect_error(zl_param_set(p, "w", matrix(NA_real_, 3, 2)), class = "zulm_invalid_argument")
  expect_error(zl_param_set(p, "w", matrix(NA_integer_, 3, 2)), class = "zulm_invalid_argument")
  expect_error(zl_param_set(p, "w", 1:6), class = "zulm_shape_error")
  expect_error(zl_param_set(p, "w", matrix(0, 2, 3)), class = "zulm_shape_error")
  expect_error(zl_param_set(p, "b", 1:3), class = "zulm_shape_error")
  expect_error(zl_param_set(p, "b", numeric()), class = "zulm_shape_error")
  expect_identical(zl_param_get(p, "b"), c(1, 1))
})

test_that("a store restored from saveRDS() is stale", {
  p <- make_store()
  path <- withr::local_tempfile(fileext = ".rds")
  saveRDS(p, path)
  q <- readRDS(path)
  expect_error(zl_param_names(q), class = "zulm_stale_error")
  expect_error(zl_param_get(q, "w"), class = "zulm_stale_error")
  expect_error(zl_save(q, withr::local_tempfile()), class = "zulm_stale_error")
  expect_output(print(q), "invalid")
})

test_that("a freed store is stale", {
  p <- make_store()
  zulm:::zlm_call("zlm_store_free_", p$ptr)
  expect_error(zl_param_get(p, "w"), class = "zulm_stale_error")
  expect_error(zl_param(p, "x", 2), class = "zulm_stale_error")
})

test_that("the memory cap is enforced with both byte counts", {
  withr::local_options(zulm.max_bytes = 1000)
  p <- zl_params()
  err <- expect_error(zl_param(p, "big", c(100, 100)), class = "zulm_memory_error")
  expect_gt(err$requested, 40000)
  expect_identical(err$allowed, 1000)
  expect_identical(zl_param_names(p), character())
  withr::local_options(zulm.max_bytes = -1)
  expect_error(zl_param(p, "x", 2), class = "zulm_invalid_argument")
})

test_that("every condition class inherits its root", {
  for (cls in zulm:::zlm_error_classes) {
    err <- tryCatch(zulm:::zlm_abort("x", cls), error = identity)
    expect_s3_class(err, cls)
    expect_s3_class(err, "zulm_error")
  }
  for (cls in zulm:::zlm_warning_classes) {
    w <- tryCatch(zulm:::zlm_warn("x", cls), warning = identity)
    expect_s3_class(w, cls)
    expect_s3_class(w, "zulm_warning")
  }
  intr <- tryCatch(zulm:::zlm_abort("x", "zulm_interrupt"), error = identity)
  expect_s3_class(intr, "interrupt")
})

test_that("index values convert to int32 and refuse NA", {
  rt <- function(x) zulm:::zlm_call("zlm_i32_roundtrip", x)
  expect_identical(rt(c(0L, 5L, -3L)), c(0L, 5L, -3L))
  expect_identical(rt(c(1, 2^31 - 1)), c(1L, .Machine$integer.max))
  expect_identical(rt(integer()), integer())
  expect_error(rt(c(1L, NA)), class = "zulm_invalid_argument")
  expect_error(rt(c(1, NaN)), class = "zulm_invalid_argument")
  expect_error(rt(1.5), class = "zulm_type_error")
  expect_error(rt(2^31), class = "zulm_type_error")
  expect_error(rt("1"), class = "zulm_type_error")
})
