test_that("a graph runs any number of times with rewritten inputs", {
  withr::local_seed(30)
  p <- zl_params()
  zl_param(p, "w", c(6, 4), init = zl_init_normal(0.5))
  build <- function(x) {
    h <- zl_silu(zl_crossprod(zl_param_tensor(p, "w"), x$x))
    list(y = zl_softmax(h), s = zl_sum(h * h))
  }
  g <- zl_graph(build, p, inputs = list(x = c(6, 3)))
  n <- if (identical(Sys.getenv("NOT_CRAN"), "true")) 1000L else 100L
  for (i in seq_len(n)) {
    x <- rand_array(c(6, 3))
    got <- zl_forward(g, list(x = x))
    if (i %% 100L == 1L || i == n) {
      fresh <- zl_forward(zl_graph(build, p, inputs = list(x = c(6, 3))), list(x = x))
      expect_identical(got, fresh)
    }
  }
})

test_that("outputs can be chosen and come back in their order", {
  g <- zl_graph(function(x) list(a = x$x + 1, b = x$x * 2), zl_params(), list(x = 3))
  out <- zl_forward(g, list(x = 1:3), outputs = c("b", "a"))
  expect_named(out, c("b", "a"))
  expect_identical(out$b, c(2, 4, 6))
  g2 <- zl_graph(function(x) list(a = x$x + 1, b = x$x * 2), zl_params(), list(x = 3), outputs = "b")
  expect_named(zl_forward(g2, list(x = 1:3)), "b")
})

test_that("a ggml assertion during construction is zulm_engine_error and the session goes on", {
  p <- zl_params()
  bad <- function(x) zulm:::zlm_op("unchecked:add", x$a, x$b)
  for (i in 1:3) {
    err <- expect_error(
      zl_graph(bad, p, list(a = c(4, 3), b = c(5, 2))),
      class = "zulm_engine_error"
    )
    expect_match(err$ggml_message, "GGML_ASSERT")
  }
  g <- zl_graph(function(x) x$a * 2, p, list(a = c(2, 2)))
  expect_identical(zl_forward(g, list(a = matrix(1, 2, 2)))$output, matrix(2, 2, 2))
})

test_that("zl_graph_info() describes the graph", {
  g <- zl_graph(function(x) list(y = zl_relu(x$x), loss = zl_sum(x$x)), zl_params(), list(x = c(3, 2)), loss = "loss")
  info <- zl_graph_info(g)
  expect_gt(info$nodes, 0L)
  expect_true("RELU" %in% names(info$ops) || "UNARY" %in% names(info$ops))
  expect_identical(info$outputs, c("y", "loss"))
  expect_identical(info$loss, "loss")
  expect_output(print(g), "zl_graph")
})

test_that("ggml's log reaches R as zulm_message at the chosen level", {
  withr::local_options(zulm.log = "warn")
  expect_message(zulm:::zlm_call("zlm_log_emit", 3L, "a warning\n"), class = "zulm_message")
  expect_silent(zulm:::zlm_call("zlm_log_emit", 2L, "some info\n"))
  withr::local_options(zulm.log = "info")
  expect_message(zulm:::zlm_call("zlm_log_emit", 2L, "some info\n"), "some info")
  withr::local_options(zulm.log = "none")
  expect_silent(zulm:::zlm_call("zlm_log_emit", 4L, "an error\n"))
})
