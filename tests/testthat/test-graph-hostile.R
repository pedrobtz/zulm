# Hostile input for graph construction (design §11): documented classes,
# never a ggml assertion.

test_that("zl_graph() checks its arguments", {
  p <- zl_params()
  expect_error(zl_graph("f", p, list(x = 2)), class = "zulm_invalid_argument")
  expect_error(zl_graph(identity, list(), list(x = 2)), class = "zulm_invalid_argument")
  expect_error(zl_graph(identity, p, list()), class = "zulm_invalid_argument")
  expect_error(zl_graph(identity, p, list(2)), class = "zulm_invalid_argument")
  expect_error(zl_graph(identity, p, list(x = 2, x = 3)), class = "zulm_invalid_argument")
  expect_error(zl_graph(identity, p, list(x = c(1, 2, 3, 4, 5))), class = "zulm_shape_error")
  expect_error(zl_graph(identity, p, list(x = 0)), class = "zulm_shape_error")
  expect_error(zl_graph(identity, p, list(x = NA)), class = "zulm_shape_error")
  expect_error(zl_graph(function(x) 1, p, list(x = 2)), class = "zulm_invalid_argument")
  expect_error(zl_graph(function(x) list(x$x), p, list(x = 2)), class = "zulm_invalid_argument")
  expect_error(zl_graph(function(x) x$x, p, list(x = 2), loss = "output"), class = "zulm_shape_error")
  expect_error(zl_graph(function(x) x$x, p, list(x = 2), loss = "nope"), class = "zulm_invalid_argument")
  expect_error(zl_graph(function(x) x$x, p, list(x = 2), outputs = "nope"), class = "zulm_invalid_argument")
})

test_that("operations refuse shapes ggml would assert on", {
  p <- zl_params()
  shape <- function(build, inputs) {
    expect_error(zl_graph(build, p, inputs), class = "zulm_shape_error")
  }
  err <- expect_error(zl_graph(function(x) x$a + x$b, p, list(a = c(4, 3), b = c(3, 4))), class = "zulm_shape_error")
  expect_identical(err$op, "add")
  expect_true(is.call(err$call))
  shape(function(x) x$a * x$b, list(a = c(4, 3), b = c(3, 1)))
  shape(function(x) zl_crossprod(x$a, x$b), list(a = c(4, 3), b = c(5, 3)))
  shape(function(x) zl_crossprod(x$a, x$b), list(a = c(4, 3, 2), b = c(4, 3, 3)))
  shape(function(x) zl_swiglu(x$a, x$b), list(a = c(4, 3), b = c(4, 1)))
  shape(function(x) zl_reshape(x$a, c(5, 2)), list(a = c(4, 3)))
  shape(function(x) zl_view(x$a, c(4, 3), offset = 1), list(a = c(4, 3)))
  shape(function(x) zl_concat(x$a, x$b, 1), list(a = c(4, 3), b = c(4, 2)))
  shape(function(x) zl_tri(x$a), list(a = c(4, 3)))
  shape(function(x) zl_softmax(x$a, x$m), list(a = c(4, 3), m = c(3, 3)))
  shape(function(x) zl_softmax(x$a, x$m), list(a = c(4, 3, 2), m = c(4, 3, 3)))
  shape(function(x) zl_cross_entropy(x$a, x$b), list(a = c(4, 3), b = c(4, 2)))
  shape(function(x) zl_get_rows(x$a, x$i), list(a = c(4, 3, 2), i = zl_int(2)))
  shape(function(x) zl_rope(x$a, x$i, 4), list(a = c(8, 2, 3), i = zl_int(4)))
  shape(function(x) zl_rope(x$a, x$i, 5), list(a = c(8, 2, 3), i = zl_int(3)))
  shape(function(x) zl_rope(x$a, x$i, 10), list(a = c(8, 2, 3), i = zl_int(3)))
  expect_error(zl_graph(function(x) zl_t(x$a), p, list(a = c(2, 2, 2))), class = "zulm_shape_error")
  expect_error(zl_graph(function(x) zl_reshape(x$a, c(1, 2, 3, 4, 5)), p, list(a = 120)), class = "zulm_shape_error")
})

test_that("operations refuse types ggml would assert on", {
  p <- zl_params()
  zl_param(p, "h", c(4, 3), type = "f16")
  type <- function(build, inputs) {
    expect_error(zl_graph(build, p, inputs), class = "zulm_type_error")
  }
  type(function(x) x$i + x$a, list(i = zl_int(4), a = 4))
  type(function(x) zl_exp(x$i), list(i = zl_int(4)))
  type(function(x) zl_param_tensor(p, "h") + x$a, list(a = c(4, 3)))
  type(function(x) zl_crossprod(x$a, zl_param_tensor(p, "h")), list(a = c(4, 3)))
  type(function(x) zl_get_rows(x$a, x$b), list(a = c(4, 3), b = 2))
  type(function(x) zl_rope(x$a, x$b, 2), list(a = c(4, 2, 3), b = 3))
  type(function(x) zl_cast(x$a, "f16"), list(a = 4))
  expect_error(zl_graph(function(x) zl_cast(x$a, "q4"), p, list(a = 4)), class = "zulm_type_error")
})

test_that("arguments of operations are checked", {
  p <- zl_params()
  inv <- function(build, inputs = list(a = c(4, 3))) {
    expect_error(zl_graph(build, p, inputs), class = "zulm_invalid_argument")
  }
  inv(function(x) x$a + "1")
  inv(function(x) x$a + c(1, 2))
  inv(function(x) x$a + NA)
  inv(function(x) x$a^2)
  inv(function(x) x$a > 0)
  inv(function(x) zl_scale(x$a, Inf))
  inv(function(x) zl_permute(x$a, c(1, 1)))
  inv(function(x) zl_permute(x$a, 1))
  inv(function(x) zl_concat(x$a, x$a, 5))
  inv(function(x) zl_pad(x$a, -1))
  inv(function(x) zl_view(x$a, c(2, 2), offset = -1))
  inv(function(x) zl_clamp(x$a, 2, 1))
  inv(function(x) zl_rms_norm(x$a, 0))
  inv(function(x) zl_softmax(x$a, scale = NA))
  inv(function(x) zl_sum(1))
  inv(function(x) zl_rope(x$a, x$i, 2, base = 0), list(a = c(4, 2, 3), i = zl_int(3)))
})

test_that("tensors are bound to their graph and to build()", {
  p <- zl_params()
  q <- zl_params()
  zl_param(q, "w", 4)
  keep <- NULL
  g <- zl_graph(function(x) {
    keep <<- x$a
    x$a + 1
  }, p, list(a = 4))
  expect_error(zl_relu(keep), class = "zulm_invalid_argument")
  expect_error(
    zl_graph(function(x) x$a + keep, p, list(a = 4)),
    class = "zulm_invalid_argument"
  )
  expect_error(
    zl_graph(function(x) x$a + zl_param_tensor(q, "w"), p, list(a = 4)),
    class = "zulm_invalid_argument"
  )
  expect_error(
    zl_graph(function(x) zl_graph(identity, p, list(b = 1)), p, list(a = 4)),
    class = "zulm_invalid_argument"
  )
})

test_that("an R error inside build() leaves no graph and no state behind", {
  p <- zl_params()
  expect_error(zl_graph(function(x) stop("boom"), p, list(a = 4)), "boom")
  g <- zl_graph(function(x) x$a * 3, p, list(a = 2))
  expect_identical(zl_forward(g, list(a = 1:2))$output, c(3, 6))
})

test_that("zl_forward() checks its data", {
  p <- zl_params()
  g <- zl_graph(function(x) zl_get_rows(x$t, x$i) + 0, p, list(t = c(2, 5), i = zl_int(3)))
  ok <- list(t = matrix(1, 2, 5), i = c(0L, 4L, 2L))
  expect_identical(dim(zl_forward(g, ok)$output), c(2L, 3L))
  expect_error(zl_forward(g, list(t = matrix(1, 2, 5))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, c(ok, list(z = 1))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, modifyList(ok, list(t = matrix(1, 5, 2)))), class = "zulm_shape_error")
  expect_error(zl_forward(g, modifyList(ok, list(t = rep(1, 10)))), class = "zulm_shape_error")
  expect_error(zl_forward(g, modifyList(ok, list(t = matrix("a", 2, 5)))), class = "zulm_type_error")
  expect_error(zl_forward(g, modifyList(ok, list(t = matrix(NA_real_, 2, 5)))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, modifyList(ok, list(i = c(0L, NA, 1L)))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, modifyList(ok, list(i = c(0L, 5L, 1L)))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, modifyList(ok, list(i = c(-1L, 0L, 1L)))), class = "zulm_invalid_argument")
  expect_error(zl_forward(g, modifyList(ok, list(i = c(0.5, 1, 1)))), class = "zulm_type_error")
  expect_error(zl_forward(g, ok, outputs = "nope"), class = "zulm_invalid_argument")
  expect_error(zl_forward(list(), ok), class = "zulm_invalid_argument")
  expect_identical(dim(zl_forward(g, ok)$output), c(2L, 3L))
})

test_that("a graph restored from saveRDS() is stale", {
  p <- zl_params()
  g <- zl_graph(function(x) x$a + 1, p, list(a = 2))
  path <- withr::local_tempfile(fileext = ".rds")
  saveRDS(g, path)
  h <- readRDS(path)
  expect_error(zl_forward(h, list(a = 1:2)), class = "zulm_stale_error")
})

test_that("a graph beyond the memory cap is refused", {
  p <- zl_params()
  withr::local_options(zulm.max_bytes = 4000)
  expect_error(zl_graph(function(x) x$a + 1, p, list(a = c(100, 100))), class = "zulm_memory_error")
  withr::local_options(zulm.max_bytes = 2e5)
  expect_error(
    zl_graph(function(x) zl_crossprod(x$a, x$a) + 1, p, list(a = c(10, 400))),
    class = "zulm_memory_error"
  )
})
