test_that("zl_seed() sets and returns the session key", {
  old <- zl_seed(11)
  on.exit(zl_seed(old), add = TRUE)
  expect_s3_class(zl_seed(), "rng_key")
  a <- zl_rng_normal(5, "s")
  zl_seed(12)
  b <- zl_rng_normal(5, "s")
  zl_seed(11)
  expect_identical(zl_rng_normal(5, "s"), a)
  expect_false(identical(a, b))
  k <- zurand::rng_key(99L, engine = "philox4x64")
  zl_seed(k)
  expect_identical(zl_seed(), k)
  expect_error(zl_seed("a"), class = "zulm_invalid_argument")
  expect_error(zl_seed(1.5), class = "zulm_invalid_argument")
  expect_error(zl_seed(c(k, k)), class = "zulm_invalid_argument")
})

test_that("R draws are zurand's, rounded by zulm's rule", {
  old <- zl_seed(21)
  on.exit(zl_seed(old), add = TRUE)
  key <- zurand::rng_fold(zurand::rng_fold(zl_seed(), "w"), 3L)
  expect_identical(
    zl_rng_uniform(c(4, 5), "w", step = 3, offset = 7),
    array(f32_toward_zero(zurand::rng_uniform(key, 20, offset = 7)), c(4, 5))
  )
  expect_identical(
    zl_rng_normal(30, "w", step = 3),
    f32(zurand::rng_normal(key, 30))
  )
  u <- zl_rng_uniform(1e5, "edge")
  expect_true(all(u > 0 & u < 1))
})

test_that("in-graph draws equal the R draws, at every thread count and split", {
  old <- zl_seed(31)
  on.exit(zl_seed(old), add = TRUE)
  draws <- function(threads, dims = c(7, 9), offset = 0) {
    local_threads(threads)
    g <- zl_graph(function(x) list(n = zl_noise_normal(dims, "eps", offset), u = zl_noise_uniform(dims, "tau", offset)),
                  zl_params(), list(a = 1))
    zl_forward(g, list(a = 1))
  }
  one <- draws(1L)
  expect_identical(one$n, zl_rng_normal(c(7, 9), "eps"))
  expect_identical(one$u, zl_rng_uniform(c(7, 9), "tau"))
  expect_identical(draws(2L), one)
  expect_identical(draws(4L), one)
  # a batch of 9 columns split into 4 + 5, the second starting at 7 * 4
  first <- draws(2L, c(7, 4), 0)
  second <- draws(2L, c(7, 5), 28)
  expect_identical(cbind(first$n, second$n), one$n)
  expect_identical(cbind(first$u, second$u), one$u)
  expect_true(all(one$u > 0 & one$u < 1))
})

test_that("noise is drawn anew at each training step", {
  old <- zl_seed(41)
  on.exit(zl_seed(old), add = TRUE)
  p <- zl_params()
  zl_param(p, "w", 3, init = zl_init_zeros())
  g <- zl_graph(function(x) list(
    loss = zl_sum(zl_param_tensor(p, "w") * zl_noise_normal(3, "eps")),
    eps = zl_noise_normal(3, "eps")
  ), p, list(a = 1), loss = "loss")
  tr <- zl_trainer(g, zl_sgd(lr = 1))
  expect_identical(zl_forward(g, list(a = 1))$eps, zl_rng_normal(3, "eps", step = 0))
  zl_step(tr, list(a = 1))
  # the step's gradient is the step's noise
  expect_identical(zl_param_get(p, "w"), -zl_rng_normal(3, "eps", step = 1))
  zl_step(tr, list(a = 1))
  expect_equal(zl_param_get(p, "w"), -zl_rng_normal(3, "eps", step = 1) - zl_rng_normal(3, "eps", step = 2), tolerance = 1e-6)
  expect_identical(zl_forward(g, list(a = 1))$eps, zl_rng_normal(3, "eps", step = 2))
})

test_that("reported losses are bit-identical at one, two and four threads", {
  withr::local_seed(51)
  l <- matrix(f32(rnorm(32 * 256)), 32)
  pr <- matrix(runif(32 * 256), 32)
  pr <- array(f32(sweep(pr, 2, colSums(pr), "/")), dim(pr))
  report <- function(threads) {
    local_threads(threads)
    g <- zl_graph(
      function(x) list(ce = zl_cross_entropy(x$l, x$p), mse = zl_mse(x$l, x$p), mae = zl_mae(x$l, x$p)),
      zl_params(), list(l = c(32, 256), p = c(32, 256))
    )
    zl_forward(g, list(l = l, p = pr))
  }
  one <- report(1L)
  expect_identical(report(2L), one)
  expect_identical(report(4L), one)
  # and they are the loss, summed in double
  logsm <- log(softmax_cols(l))
  expect_equal(one$ce, -sum(pr * logsm) / 256, tolerance = 1e-6)
  expect_equal(one$mse, mean((l - pr)^2), tolerance = 1e-6)
  expect_equal(one$mae, mean(abs(l - pr)), tolerance = 1e-6)
})

test_that("zl_step() reports the per-row loss, the same at any thread count", {
  withr::local_seed(52)
  x <- matrix(f32(rnorm(8 * 64)), 8)
  y <- array(f32(diag(4)[, sample(4, 64, replace = TRUE)]), c(4, 64))
  w0 <- matrix(f32(rnorm(32, sd = 0.3)), 8)
  run <- function(threads) {
    local_threads(threads)
    p <- zl_params()
    zl_param(p, "w", c(8, 4), init = zl_init_zeros())
    zl_param_set(p, "w", w0)
    g <- zl_graph(function(i) list(loss = zl_cross_entropy(zl_crossprod(zl_param_tensor(p, "w"), i$x), i$y)),
                  p, list(x = c(8, 64), y = c(4, 64)), loss = "loss")
    tr <- zl_trainer(g, zl_adamw(lr = 0.05))
    vapply(1:10, function(k) zl_step(tr, list(x = x, y = y)), numeric(1))
  }
  one <- run(1L)
  expect_lt(one[10], one[1])
  expect_identical(run(2L), one)
  expect_identical(run(4L), one)
})

test_that("draws are checked", {
  expect_error(zl_rng_uniform(0, "a"), class = "zulm_shape_error")
  expect_error(zl_rng_uniform(2, ""), class = "zulm_invalid_argument")
  expect_error(zl_rng_uniform(2, "a", step = -1), class = "zulm_invalid_argument")
  expect_error(zl_rng_normal(2, "a", offset = 0.5), class = "zulm_invalid_argument")
  expect_error(zl_noise_normal(2, "a"), class = "zulm_invalid_argument")
})

test_that("a zurand below table version 3 is refused", {
  expect_error(zulm:::zlm_check_zurand(2L), class = "zulm_companion_error")
  expect_invisible(zulm:::zlm_check_zurand(3L))
  expect_identical(zulm_info()$zurand_table_version >= 3L, TRUE)
})
