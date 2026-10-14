# Central finite differences against ggml's gradients for every operation
# and composition with a backward pass (design §11; Stage 4 exit criterion).

test_that("arithmetic gradients agree, with broadcasts either way", {
  withr::local_seed(40)
  for (rep in 1:6) {
    da <- rand_dims(sample(1:3, 1), 4)
    db <- bcast_dims(da)
    if (rep %% 2 == 0) {
      tmp <- da
      da <- db
      db <- tmp
    }
    a <- rand_array(da, 0.5, 2)
    b <- rand_array(db, 0.5, 2)
    grad_check(function(x, p) zl_param_tensor(p, "a") + zl_param_tensor(p, "b"), list(a = a, b = b))
    grad_check(function(x, p) zl_param_tensor(p, "a") - zl_param_tensor(p, "b"), list(a = a, b = b))
    grad_check(function(x, p) zl_param_tensor(p, "a") * zl_param_tensor(p, "b"), list(a = a, b = b))
    grad_check(function(x, p) zl_param_tensor(p, "a") / zl_param_tensor(p, "b"), list(a = a, b = b))
  }
})

test_that("scales, constants and negation have gradients", {
  withr::local_seed(41)
  a <- rand_array(c(3, 2), 0.5, 2)
  grad_check(function(x, p) {
    w <- zl_param_tensor(p, "a")
    list2 <- 2 * w + 1
    3 / w - list2 * 0.5 + zl_scale(-w, 1.5) - (w - 4)
  }, list(a = a))
})

test_that("reductions and elementwise functions have gradients", {
  withr::local_seed(42)
  a <- rand_array(c(4, 3), 0.3, 1.5)
  s <- rand_array(c(4, 3), -1.5, 1.5)
  fs <- list(
    sum = function(w) zl_sum(w), rows = function(w) zl_sum_rows(w), mean = function(w) zl_mean_rows(w),
    sqr = zl_sqr, sqrt = zl_sqrt, log = zl_log, exp = zl_exp, sin = zl_sin, cos = zl_cos
  )
  for (nm in names(fs)) {
    f <- fs[[nm]]
    grad_check(function(x, p) f(zl_param_tensor(p, "a")), list(a = a))
  }
  for (f in list(zl_abs, zl_relu, zl_silu, zl_softplus)) {
    grad_check(function(x, p) f(zl_param_tensor(p, "s")), list(s = s))
  }
  grad_check(function(x, p) zl_swiglu(zl_param_tensor(p, "g"), zl_param_tensor(p, "u")),
             list(g = s, u = a))
})

test_that("products and shapes have gradients", {
  withr::local_seed(43)
  # the weight broadcasts over the third dimension of b
  a <- rand_array(c(4, 3, 2))
  b <- rand_array(c(4, 5, 6))
  grad_check(function(x, p) zl_crossprod(zl_param_tensor(p, "a"), zl_param_tensor(p, "b")), list(a = a, b = b))
  grad_check(function(x, p) zl_crossprod(zl_param_tensor(p, "a"), zl_param_tensor(p, "b")),
             list(a = rand_array(c(4, 3, 2, 3)), b = rand_array(c(4, 5, 2, 3))))
  grad_check(function(x, p) zl_crossprod(zl_param_tensor(p, "w"), x$x),
             list(w = rand_array(c(4, 3))), list(x = c(4, 6)), list(x = rand_array(c(4, 6))))
  x <- rand_array(c(2, 3, 4))
  grad_check(function(x, p) {
    w <- zl_param_tensor(p, "x")
    zl_reshape(w, c(6, 4)) * 2
  }, list(x = x))
  grad_check(function(x, p) zl_permute(zl_param_tensor(p, "x"), c(3, 1, 2)) * 1, list(x = x))
  grad_check(function(x, p) zl_t(zl_reshape(zl_param_tensor(p, "x"), c(6, 4))), list(x = x))
  grad_check(function(x, p) zl_cont(zl_permute(zl_param_tensor(p, "x"), c(2, 1, 3))), list(x = x))
  grad_check(function(x, p) zl_view(zl_param_tensor(p, "x"), c(3, 2), offset = 5) * 1, list(x = x))
  tab <- rand_array(c(3, 5))
  grad_check(function(x, p) zl_get_rows(zl_param_tensor(p, "t"), x$i),
             list(t = tab), list(i = zl_int(4)), list(i = c(4L, 0L, 2L, 4L)))
})

test_that("attention pieces and norms have gradients", {
  withr::local_seed(44)
  s <- rand_array(c(5, 4, 2))
  mask <- array(0, c(5, 4))
  mask[5, ] <- -Inf
  grad_check(function(x, p) zl_softmax(zl_param_tensor(p, "s"), x$m, scale = 0.7),
             list(s = s), list(m = c(5, 4)), list(m = mask))
  sq <- rand_array(c(4, 4, 2))
  grad_check(function(x, p) zl_softmax(zl_causal_mask(zl_param_tensor(p, "s"))), list(s = sq))
  q <- rand_array(c(8, 2, 3, 1))
  for (mode in c("interleaved", "half")) {
    for (n_dims in c(8, 4)) {
      grad_check(function(x, p) zl_rope(zl_param_tensor(p, "q"), x$pos, n_dims, base = 100, mode = mode),
                 list(q = q), list(pos = zl_int(3)), list(pos = c(0L, 2L, 5L)))
    }
  }
  n <- rand_array(c(6, 3))
  grad_check(function(x, p) zl_rms_norm(zl_param_tensor(p, "n")), list(n = n))
  grad_check(function(x, p) zl_layer_norm(zl_param_tensor(p, "n")), list(n = n))
})

test_that("losses and zl_where() have gradients", {
  withr::local_seed(45)
  logits <- rand_array(c(5, 3))
  prob <- softmax_cols(rand_array(c(5, 3)))
  grad_check(function(x, p) zl_cross_entropy(zl_param_tensor(p, "l"), x$p),
             list(l = logits), list(p = c(5, 3)), list(p = array(f32(prob), dim(prob))))
  pred <- rand_array(c(4, 2))
  target <- rand_array(c(4, 2))
  grad_check(function(x, p) zl_mse(zl_param_tensor(p, "a"), x$t), list(a = pred), list(t = c(4, 2)), list(t = target))
  grad_check(function(x, p) zl_mae(zl_param_tensor(p, "a"), x$t), list(a = pred), list(t = c(4, 2)), list(t = target))
  m <- matrix(rbinom(8, 1, 0.5), 4, 2)
  grad_check(function(x, p) zl_where(x$m, zl_param_tensor(p, "a"), zl_param_tensor(p, "b")),
             list(a = pred, b = target), list(m = c(4, 2)), list(m = m))
})
