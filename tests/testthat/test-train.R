linear_graph <- function(p, n = 8) {
  zl_graph(
    function(x) list(
      pred = zl_crossprod(zl_param_tensor(p, "w"), x$x),
      loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), x$x), x$y)
    ),
    p,
    inputs = list(x = c(3, n), y = c(1, n)),
    loss = "loss"
  )
}

linear_data <- function(n = 8) {
  x <- rand_array(c(3, n))
  list(x = x, y = matrix(f32(c(1, -2, 0.5) %*% x), 1, n))
}

test_that("AdamW and SGD fit a linear model", {
  withr::local_seed(50)
  d <- linear_data()
  for (opt in list(zl_adamw(lr = 0.05), zl_sgd(lr = 0.2))) {
    p <- zl_params()
    zl_param(p, "w", c(3, 1), init = zl_init_zeros())
    tr <- zl_trainer(linear_graph(p), opt)
    first <- zl_step(tr, d)
    for (i in 1:300) last <- zl_step(tr, d)
    expect_lt(last, first * 1e-3)
    expect_equal(as.vector(zl_param_get(p, "w")), c(1, -2, 0.5), tolerance = 1e-2)
  }
})

test_that("an SGD step moves parameters by -lr times the gradient", {
  withr::local_seed(51)
  d <- linear_data()
  p <- zl_params()
  zl_param(p, "w", c(3, 1), init = zl_init_uniform(1))
  tr <- zl_trainer(linear_graph(p), zl_sgd(lr = 0.1))
  zl_forward(linear_graph(p), d)
  w0 <- zl_param_get(p, "w")
  zlm_write <- zulm:::zlm_write_inputs
  zlm_write(tr$graph, d)
  gr <- zl_grad(tr, "w")
  zl_step(tr, d, lr = 0.1)
  expect_equal(zl_param_get(p, "w"), w0 - 0.1 * gr, tolerance = 1e-6)
})

test_that("clip_norm scales every gradient by min(1, clip / norm)", {
  withr::local_seed(52)
  d <- linear_data()
  p <- zl_params()
  zl_param(p, "w", c(3, 1), init = zl_init_uniform(2))
  zl_param(p, "b", c(1, 1), init = zl_init_uniform(2))
  g <- zl_graph(
    function(x) list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), x$x) + zl_param_tensor(p, "b"), x$y)),
    p, inputs = list(x = c(3, 8), y = c(1, 8)), loss = "loss"
  )
  tr <- zl_trainer(g, zl_sgd(lr = 0.1), clip_norm = 0.5)
  zulm:::zlm_write_inputs(g, d)
  gw <- zl_grad(tr, "w")
  gb <- zl_grad(tr, "b")
  norm <- sqrt(sum(gw^2) + sum(gb^2))
  expect_gt(norm, 0.5)
  scale <- min(1, 0.5 / norm)
  w0 <- zl_param_get(p, "w")
  b0 <- zl_param_get(p, "b")
  zl_step(tr, d)
  expect_equal(zl_param_get(p, "w"), w0 - 0.1 * scale * gw, tolerance = 1e-5)
  expect_equal(zl_param_get(p, "b"), b0 - 0.1 * scale * gb, tolerance = 1e-5)
  # a large clip leaves the step unchanged
  q <- zl_params()
  zl_param(q, "w", c(3, 1), init = zl_init_uniform(2))
  tq <- zl_trainer(linear_graph(q), zl_sgd(lr = 0.1), clip_norm = 1e6)
  zulm:::zlm_write_inputs(tq$graph, d)
  gq <- zl_grad(tq, "w")
  wq <- zl_param_get(q, "w")
  zl_step(tq, d)
  expect_equal(zl_param_get(q, "w"), wq - 0.1 * gq, tolerance = 1e-5)
})

test_that("accumulate sums gradients and updates every k batches", {
  withr::local_seed(53)
  d1 <- linear_data()
  d2 <- linear_data()
  p <- zl_params()
  zl_param(p, "w", c(3, 1), init = zl_init_uniform(1))
  tr <- zl_trainer(linear_graph(p), zl_sgd(lr = 0.1), accumulate = 2L)
  w0 <- zl_param_get(p, "w")
  zulm:::zlm_write_inputs(tr$graph, d1)
  g1 <- zl_grad(tr, "w")
  zulm:::zlm_write_inputs(tr$graph, d2)
  g2 <- zl_grad(tr, "w")
  zl_step(tr, d1)
  expect_identical(zl_param_get(p, "w"), w0)
  zl_step(tr, d2)
  expect_equal(zl_param_get(p, "w"), w0 - 0.1 * (g1 + g2), tolerance = 1e-5)
})

test_that("zl_reset() restarts AdamW from fresh state", {
  withr::local_seed(54)
  d <- linear_data()
  run <- function(reset) {
    p <- zl_params()
    zl_param(p, "w", c(3, 1), init = zl_init_zeros())
    tr <- zl_trainer(linear_graph(p), zl_adamw(lr = 0.01))
    for (i in 1:5) zl_step(tr, d)
    zl_param_set(p, "w", matrix(0, 3, 1))
    if (reset) zl_reset(tr)
    zl_step(tr, d)
    zl_param_get(p, "w")
  }
  fresh <- {
    p <- zl_params()
    zl_param(p, "w", c(3, 1), init = zl_init_zeros())
    tr <- zl_trainer(linear_graph(p), zl_adamw(lr = 0.01))
    zl_step(tr, d)
    zl_param_get(p, "w")
  }
  expect_identical(run(TRUE), fresh)
  expect_false(identical(run(FALSE), fresh))
})

test_that("a non-finite loss warns", {
  p <- zl_params()
  zl_param(p, "w", 2, init = zl_init_ones())
  g <- zl_graph(function(x) list(loss = zl_sum(zl_log(zl_param_tensor(p, "w") * x$x))), p, list(x = 2), loss = "loss")
  tr <- zl_trainer(g, zl_sgd(lr = 0.1))
  expect_warning(zl_step(tr, list(x = c(-1, -1))), class = "zulm_nonfinite_warning")
})

test_that("operations without a gradient are refused before ggml sees them", {
  p <- zl_params()
  zl_param(p, "w", c(4, 3))
  refuse <- function(f, op) {
    g <- zl_graph(function(x) list(loss = zl_sum(f(zl_param_tensor(p, "w")) * x$x)), p, list(x = c(4, 3)), loss = "loss")
    err <- expect_error(zl_trainer(g, zl_sgd()), class = "zulm_unsupported_backward")
    expect_match(err$op, op)
    expect_true(is.call(err$call))
    err
  }
  err <- refuse(function(w) zl_gelu(w), "GELU")
  expect_identical(err$call[[1]], quote(zl_gelu))
  refuse(function(w) zl_tanh(w), "TANH")
  refuse(function(w) zl_sigmoid(w), "SIGMOID")
  refuse(function(w) zl_clamp(w, -1, 1), "CLAMP")
  refuse(function(w) zl_cumsum(w), "CUMSUM")
  refuse(function(w) zl_reshape(zl_tri(zl_reshape(w, c(2, 2, 3))), c(4, 3)), "TRI")
  refuse(function(w) zl_view(zl_concat(w, w, 1), c(4, 3)), "CONCAT")
  refuse(function(w) zl_view(zl_pad(w, 1), c(4, 3)), "PAD")
  refuse(function(w) zl_cast(zl_cast(w, "f16"), "f32"), "CPY")
})

test_that("a weight broadcast over b's fourth dimension is refused", {
  p <- zl_params()
  zl_param(p, "a", c(4, 3, 2))
  g <- zl_graph(function(x) list(loss = zl_sum(zl_crossprod(zl_param_tensor(p, "a"), x$b))),
                p, list(b = c(4, 5, 2, 3)), loss = "loss")
  err <- expect_error(zl_trainer(g, zl_sgd()), class = "zulm_unsupported_backward")
  expect_match(err$op, "MUL_MAT")
})

test_that("a mask on a gradient path and half-precision parameters are refused", {
  p <- zl_params()
  zl_param(p, "w", c(4, 3))
  zl_param(p, "h", c(4, 3), type = "f16")
  g <- zl_graph(function(x) list(loss = zl_sum(zl_softmax(x$s, zl_param_tensor(p, "w")))), p, list(s = c(4, 3)), loss = "loss")
  expect_error(zl_trainer(g, zl_sgd()), class = "zulm_unsupported_backward")
  g2 <- zl_graph(function(x) list(loss = zl_sum(zl_crossprod(zl_param_tensor(p, "h"), x$x))), p, list(x = c(4, 2)), loss = "loss")
  expect_error(zl_trainer(g2, zl_sgd()), class = "zulm_type_error")
})

test_that("trainer arguments are checked", {
  p <- zl_params()
  zl_param(p, "w", c(3, 1))
  g <- linear_graph(p)
  expect_error(zl_trainer(g, "adam"), class = "zulm_invalid_argument")
  expect_error(zl_trainer(g, zl_sgd(), accumulate = 0), class = "zulm_invalid_argument")
  expect_error(zl_trainer(g, zl_sgd(), clip_norm = -1), class = "zulm_invalid_argument")
  expect_error(zl_trainer(zl_graph(function(x) x$a + 1, p, list(a = 2)), zl_sgd()), class = "zulm_invalid_argument")
  expect_error(zl_adamw(lr = 0), class = "zulm_invalid_argument")
  expect_error(zl_adamw(beta1 = 1.5), class = "zulm_invalid_argument")
  expect_error(zl_sgd(weight_decay = 2), class = "zulm_invalid_argument")
  q <- zl_params()
  zl_param(q, "unused", 2)
  gq <- zl_graph(function(x) list(loss = zl_sum(x$a)), q, list(a = 2), loss = "loss")
  expect_error(zl_trainer(gq, zl_sgd()), class = "zulm_invalid_argument")
  tr <- zl_trainer(g, zl_sgd())
  expect_error(zl_step(tr, list(x = matrix(0, 3, 8))), class = "zulm_invalid_argument")
  expect_error(zl_step(tr, linear_data(), lr = -1), class = "zulm_invalid_argument")
  expect_error(zl_grad(tr, "nope"), class = "zulm_invalid_argument")
  expect_output(print(tr), "zl_trainer")
  expect_output(print(zl_adamw()), "AdamW")
})

test_that("patches P1, P2 and P7 are vendored, and their cases train", {
  expect_true(all(c(
    "p01-mean-backward-accumulates.patch", "p02-sub-div-broadcast-backward.patch",
    "p07-opt-global-norm-clipping.patch", "p08-c23-static-assert.patch",
    "p09-gguf-type-range.patch", "p10-function-pointer-types.patch"
  ) %in% zulm_info()$ggml_patches))
  withr::local_seed(55)
  # P1: ggml_mean() whose input has a second consumer (a LayerNorm written
  # with ggml_mean aborted in ggml_opt_init before the patch)
  p <- zl_params()
  zl_param(p, "w", c(4, 3), init = zl_init_uniform(1))
  g <- zl_graph(function(x) {
    w <- zl_param_tensor(p, "w")
    m <- zulm:::zlm_op("unchecked:mean", zulm:::zlm_as_tensor(w), rank = 2L)
    list(loss = zl_sum(zl_sqr(w - m) * x$x))
  }, p, list(x = c(4, 3)), loss = "loss")
  tr <- zl_trainer(g, zl_sgd(lr = 0.1))
  expect_true(is.finite(zl_step(tr, list(x = matrix(1, 4, 3)))))
  # P2: a 4 x 3 minus a 1 x 3, with a gradient on the broadcast operand
  q <- zl_params()
  zl_param(q, "a", c(4, 3), init = zl_init_uniform(1))
  zl_param(q, "b", c(1, 3), init = zl_init_uniform(1))
  g2 <- zl_graph(function(x) {
    d <- zulm:::zlm_op("unchecked:sub_raw", zulm:::zlm_as_tensor(zl_param_tensor(q, "a")),
                       zulm:::zlm_as_tensor(zl_param_tensor(q, "b")))
    list(loss = zl_sum(zl_sqr(d) * x$x))
  }, q, list(x = c(4, 3)), loss = "loss")
  tr2 <- zl_trainer(g2, zl_sgd(lr = 0.1))
  expect_true(is.finite(zl_step(tr2, list(x = matrix(1, 4, 3)))))
  zulm:::zlm_write_inputs(g2, list(x = matrix(1, 4, 3)))
  a <- zl_param_get(q, "a")
  b <- zl_param_get(q, "b")
  expect_equal(as.vector(zl_grad(tr2, "b")), -2 * colSums(sweep(a, 2, b)), tolerance = 1e-5)
})

test_that("a step through a transposed or permuted parameter updates the right elements", {
  withr::local_seed(56)
  for (perm in list(NULL, c(2, 1, 3))) {
    p <- zl_params()
    if (is.null(perm)) {
      zl_param(p, "w", c(3, 2), init = zl_init_uniform(1))
      use <- function(w) zl_t(w)
      wd <- c(2, 3)
    } else {
      zl_param(p, "w", c(3, 2, 2), init = zl_init_uniform(1))
      use <- function(w) zl_permute(w, perm)
      wd <- c(2, 3, 2)
    }
    g <- zl_graph(function(x) list(loss = zl_sum(use(zl_param_tensor(p, "w")) * x$x)), p, list(x = wd), loss = "loss")
    tr <- zl_trainer(g, zl_sgd(lr = 0.5))
    x <- rand_array(wd)
    zulm:::zlm_write_inputs(g, list(x = x))
    gr <- zl_grad(tr, "w")
    expect_equal(gr, if (is.null(perm)) t(x) else aperm(x, perm), tolerance = 1e-6)
    w0 <- zl_param_get(p, "w")
    zl_step(tr, list(x = x))
    expect_equal(zl_param_get(p, "w"), w0 - 0.5 * gr, tolerance = 1e-6)
  }
})

test_that("training is bit-identical at one, two and four threads", {
  withr::local_seed(57)
  x <- rand_array(c(16, 32))
  y <- matrix(f32(rnorm(32)), 1, 32)
  w1 <- rand_array(c(16, 24), -0.3, 0.3)
  w2 <- rand_array(c(24, 1), -0.3, 0.3)
  run <- function(threads) {
    withr::local_options(zulm.threads = threads)
    p <- zl_params()
    zl_param(p, "w1", c(16, 24), init = zl_init_zeros())
    zl_param(p, "w2", c(24, 1), init = zl_init_zeros())
    zl_param_set(p, "w1", w1)
    zl_param_set(p, "w2", w2)
    g <- zl_graph(function(x) {
      h <- zl_silu(zl_crossprod(zl_param_tensor(p, "w1"), x$x))
      list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w2"), h), x$y))
    }, p, list(x = c(16, 32), y = c(1, 32)), loss = "loss")
    tr <- zl_trainer(g, zl_adamw(lr = 1e-2), clip_norm = 1)
    losses <- vapply(1:20, function(i) zl_step(tr, list(x = x, y = y)), numeric(1))
    list(losses = losses, w1 = zl_param_get(p, "w1"), w2 = zl_param_get(p, "w2"))
  }
  one <- run(1L)
  expect_lt(one$losses[20], one$losses[1])
  expect_identical(run(2L), one)
  expect_identical(run(4L), one)
})
