# What the fuzz targets (tools/fuzz/, design §11) found, pinned: each case
# reached an unclassed error, a ggml assertion or undefined behaviour
# before it was fixed.

test_that("sizes of 2^31 and NA are refused, not passed to ggml as -2^31", {
  p <- zl_params()
  run <- function(f) zl_graph(function(x) f(x$a), p, inputs = list(a = c(2, 3)))
  expect_error(run(function(a) zl_pad(a, after = 2^31)), class = "zulm_invalid_argument")
  expect_error(run(function(a) zl_rope(a, a, n_dims = NA)), class = "zulm_invalid_argument")
  expect_error(run(function(a) zl_time_embedding(a, dim = 2^31)), class = "zulm_invalid_argument")
})

test_that("a choice outside the allowed values is zulm_invalid_argument", {
  p <- zl_params()
  run <- function(f) zl_graph(function(x) f(x$a), p, inputs = list(a = c(2, 2)))
  expect_error(run(function(a) zl_tri(a, "x")), class = "zulm_invalid_argument")
  expect_error(run(function(a) zl_gelu(a, approximate = "x")), class = "zulm_invalid_argument")
  expect_error(run(function(a) zl_tri(a, NA)), class = "zulm_invalid_argument")
  # partial matching still works, as with match.arg()
  expect_s3_class(run(function(a) zl_tri(a, "upper_s")), "zl_graph")
  expect_error(
    zl_mlp(p, "m", 2, 4, act = "x"),
    class = "zulm_invalid_argument"
  )
})

test_that("a parameter larger than the memory cap is refused before R allocates it", {
  p <- zl_params()
  err <- expect_error(zl_param(p, "w", c(1e5, 1e5), init = zl_init_ones()), class = "zulm_memory_error")
  expect_identical(err$requested, 4e10)
  expect_identical(zl_param_names(p), character())
})

test_that("an operation on a permuted view with unit dimensions copies it first", {
  p <- zl_params()
  g <- zl_graph(
    function(x) zl_scale(zl_permute(x$a, c(4, 3, 2, 1)), 2),
    p,
    inputs = list(a = c(1, 1, 1, 4))
  )
  out <- zl_forward(g, list(a = array(1:4, c(1, 1, 1, 4))))$output
  expect_identical(as.vector(out), c(2, 4, 6, 8))
})

test_that("gradients ggml cannot carry back through a permutation are refused", {
  # a copy of a permuted gradient: ggml's CONT backward asserts contiguity
  expect_error(
    train_once(function(w) zl_permute(zl_cont(w), c(4, 3, 2, 1)), dim = c(3, 1, 8, 2)),
    class = "zulm_unsupported_backward"
  )
  # a broadcast operand's gradient, reduced from rows that are not dense
  expect_error(
    train_once(function(w) zl_permute(w + zl_sum_rows(w), c(2, 1, 3, 4))),
    class = "zulm_unsupported_backward"
  )
  # a parameter handed two gradients, the first with rows not dense
  expect_error(
    train_once(function(w) zl_permute(w + w, c(2, 1, 3, 4))),
    class = "zulm_unsupported_backward"
  )
  # a node handed two gradients, one with rows not dense
  expect_error(
    train_once(function(w) {
      h <- w + zl_sqr(w)
      zl_permute(h, c(2, 1, 3, 4)) + h
    }),
    class = "zulm_unsupported_backward"
  )
})

test_that("permutations whose gradients ggml can carry still train", {
  # the first axis stays first: rows stay dense
  expect_true(is.finite(train_once(function(w) zl_permute(w * w, c(1, 3, 2, 4)))))
  # a parameter permuted once, its gradient copied by ggml-opt
  expect_true(is.finite(train_once(function(w) zl_permute(w, c(2, 1, 3, 4)))))
  expect_true(is.finite(train_once(function(w) zl_permute(zl_reshape(w, c(16, 16)), c(2, 1)))))
})

test_that("clipping reads a parameter's permuted gradient through dense rows (patch P7)", {
  loss <- train_once(function(w) zl_permute(w, c(4, 3, 2, 1)), dim = c(1, 1, 1, 4), clip_norm = 1)
  expect_identical(loss, 4)
})

test_that("labels that depend on a parameter are refused in training", {
  p <- zl_params()
  zl_param(p, "w", c(3, 2), init = zl_init_ones())
  g <- zl_graph(
    function(x) list(loss = zl_cross_entropy(x$logits, zl_softmax(zl_param_tensor(p, "w")))),
    p,
    inputs = list(logits = c(3, 2)),
    loss = "loss"
  )
  expect_error(zl_trainer(g, zl_sgd()), class = "zulm_unsupported_backward")
})
