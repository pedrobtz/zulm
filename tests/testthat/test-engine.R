# Design §13.2, layer 2: an assertion during graph construction is an R
# error, and the process carries on.

test_that("a ggml assertion during construction is zulm_engine_error", {
  err <- expect_error(zulm:::zlm_engine_check(), class = "zulm_engine_error")
  expect_s3_class(err, "zulm_error")
  expect_match(err$ggml_message, "GGML_ASSERT")
})

test_that("the session is usable after repeated assertions", {
  for (i in 1:3) {
    expect_error(zulm:::zlm_engine_check(), class = "zulm_engine_error")
  }
  expect_invisible(zulm:::zlm_engine_check(valid = TRUE))
  before <- zulm:::zlm_call("zlm_bench_mul_mat", 64L, 1L, 2L)[[3L]]
  expect_error(zulm:::zlm_engine_check(), class = "zulm_engine_error")
  after <- zulm:::zlm_call("zlm_bench_mul_mat", 64L, 1L, 2L)[[3L]]
  expect_identical(before, after)
})

test_that("ggml's writes to stdout and stderr reach the log, not the console", {
  out <- zulm:::zlm_call("zlm_log_drain")
  expect_type(out, "character")
  expect_length(out, 1L)
})
