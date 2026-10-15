# Helpers of test-threads.R.

train_small <- function() {
  withr::local_seed(70)
  x <- matrix(f32(rnorm(16 * 32)), 16)
  y <- matrix(f32(rnorm(32)), 1)
  p <- zl_params()
  zl_param(p, "w1", c(16, 24), init = zl_init_zeros())
  zl_param(p, "w2", c(24, 1), init = zl_init_zeros())
  zl_param_set(p, "w1", matrix(f32(rnorm(16 * 24, sd = 0.2)), 16))
  zl_param_set(p, "w2", matrix(f32(rnorm(24, sd = 0.2)), 24))
  g <- zl_graph(function(i) list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w2"), zl_silu(zl_crossprod(zl_param_tensor(p, "w1"), i$x))), i$y)),
                p, list(x = c(16, 32), y = c(1, 32)), loss = "loss")
  tr <- zl_trainer(g, zl_adamw(lr = 0.01))
  losses <- vapply(1:20, function(k) zl_step(tr, list(x = x, y = y)), numeric(1))
  list(losses = losses, w1 = zl_param_get(p, "w1"), w2 = zl_param_get(p, "w2"))
}

fixture_forward <- function(name, threads) {
  local_threads(threads)
  fx <- zl_load(test_path("fixtures", name))
  p <- zl_params()
  m <- attr(fx, "metadata")
  blk <- zl_block(p, "blk", dim = m$dim, heads = m$heads, hidden = m$hidden)
  for (nm in zl_param_names(p)) zl_param_set(p, nm, zl_param_get(fx, nm))
  x <- zl_param_get(fx, "input.x")
  zl_forward(zl_graph(function(i) blk(i$x), p, list(x = dim(x))), list(x = x))$output
}
