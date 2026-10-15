# Each module against a hand-written R version (design §11; Stage 6).

test_that("zl_linear() is crossprod(weight, x) + bias", {
  withr::local_seed(60)
  p <- zl_params()
  l1 <- zl_linear(p, "a", 5, 3)
  l2 <- zl_linear(p, "b", 5, 3, bias = FALSE)
  x <- rand_array(c(5, 4, 2))
  out <- run_graph(function(i) list(a = l1(i$x), b = l2(i$x)), list(x = c(5, 4, 2)), list(x = x), p = p)
  expect_f32(out$a, lin_r(p, "a", x))
  expect_f32(out$b, lin_r(p, "b", x, bias = FALSE))
  expect_false("b.bias" %in% zl_param_names(p))
  expect_lte(max(abs(zl_param_get(p, "a.weight"))), 1 / sqrt(5))
})

test_that("zl_embedding() takes rows by 0-based index, of any shape", {
  p <- zl_params()
  e <- zl_embedding(p, "e", n = 6, dim = 3)
  idx <- matrix(c(0L, 5L, 2L, 2L, 1L, 4L), 3, 2)
  out <- run_graph(function(i) e(i$i), list(i = zl_int(3, 2)), list(i = idx), p = p)
  w <- zl_param_get(p, "e.weight")
  expect_identical(out$output, array(w[, idx + 1L], c(3, 3, 2)))
  expect_error(
    zl_forward(zl_graph(function(i) e(i$i), p, list(i = zl_int(3, 2))), list(i = idx + 1L)),
    class = "zulm_invalid_argument"
  )
})

test_that("zl_norm() is layer or RMS normalisation with an affine map", {
  withr::local_seed(61)
  p <- zl_params()
  ln <- zl_norm(p, "ln", 6)
  rn <- zl_norm(p, "rn", 6, type = "rms")
  pl <- zl_norm(p, "pl", 6, affine = FALSE)
  zl_param_set(p, "ln.weight", rand_array(6))
  zl_param_set(p, "ln.bias", rand_array(6))
  zl_param_set(p, "rn.weight", rand_array(6))
  x <- rand_array(c(6, 4))
  out <- run_graph(function(i) list(ln = ln(i$x), rn = rn(i$x), pl = pl(i$x)), list(x = c(6, 4)), list(x = x), p = p)
  expect_f32(out$ln, layer_norm_r(x, zl_param_get(p, "ln.weight"), zl_param_get(p, "ln.bias")), tol = 1e-4)
  rms <- sweep(x, 2, sqrt(colMeans(x^2) + 1e-5), "/") * zl_param_get(p, "rn.weight")
  expect_f32(out$rn, rms, tol = 1e-4)
  expect_f32(out$pl, layer_norm_r(x), tol = 1e-4)
  expect_false(any(c("rn.bias", "pl.weight") %in% zl_param_names(p)))
})

test_that("zl_attention() is multi-head attention, causal or not", {
  withr::local_seed(62)
  for (causal in c(TRUE, FALSE)) {
    p <- zl_params()
    at <- zl_attention(p, "att", dim = 8, heads = 2, causal = causal)
    x <- rand_array(c(8, 5, 3))
    out <- run_graph(function(i) at(i$x), list(x = c(8, 5, 3)), list(x = x), p = p)
    expect_f32(out$output, attn_r(p, "att", x, heads = 2, causal = causal), tol = 1e-4)
  }
  p <- zl_params()
  at <- zl_attention(p, "att", dim = 8, heads = 4)
  x <- rand_array(c(8, 5))
  out <- run_graph(function(i) at(i$x), list(x = c(8, 5)), list(x = x), p = p)
  expect_f32(out$output, attn_r(p, "att", array(x, c(8, 5, 1)), heads = 4)[, , 1], tol = 1e-4)
  expect_error(zl_attention(zl_params(), "a", dim = 6, heads = 4), class = "zulm_invalid_argument")
})

test_that("zl_mlp() and zl_block() match R", {
  withr::local_seed(63)
  p <- zl_params()
  mlp <- zl_mlp(p, "m", 6, hidden = 10)
  sil <- zl_mlp(p, "s", 6, hidden = 10, act = "silu")
  blk <- zl_block(p, "b", dim = 6, heads = 2, hidden = 8)
  x <- rand_array(c(6, 4, 2))
  out <- run_graph(function(i) list(m = mlp(i$x), s = sil(i$x), b = blk(i$x)), list(x = c(6, 4, 2)), list(x = x), p = p)
  silu <- function(z) z / (1 + exp(-z))
  expect_f32(out$m, lin_r(p, "m.down", silu(lin_r(p, "m.gate", x)) * lin_r(p, "m.up", x)), tol = 1e-4)
  expect_f32(out$s, lin_r(p, "s.down", silu(lin_r(p, "s.up", x))), tol = 1e-4)
  ln <- function(nm, z) layer_norm_r(z, zl_param_get(p, paste0(nm, ".weight")), zl_param_get(p, paste0(nm, ".bias")))
  h <- x + attn_r(p, "b.attn", ln("b.norm1", x), heads = 2)
  hb <- ln("b.norm2", h)
  want <- h + lin_r(p, "b.mlp.down", silu(lin_r(p, "b.mlp.gate", hb)) * lin_r(p, "b.mlp.up", hb))
  expect_f32(out$b, want, tol = 1e-4)
})

test_that("zl_adaln() and zl_time_embedding() match R", {
  withr::local_seed(64)
  p <- zl_params()
  ada <- zl_adaln(p, "ada", dim = 4, cond_dim = 3)
  x <- rand_array(c(4, 5))
  cnd <- rand_array(c(3, 5))
  tt <- array(f32(runif(10)), c(1, 5, 2))
  out <- run_graph(
    function(i) list(a = ada(i$x, i$c), t = zl_time_embedding(i$t, 6, max_period = 100)),
    list(x = c(4, 5), c = c(3, 5), t = c(1, 5, 2)), list(x = x, c = cnd, t = tt), p = p
  )
  expect_f32(out$a, x * (1 + lin_r(p, "ada.scale", cnd)) + lin_r(p, "ada.shift", cnd), tol = 1e-4)
  f <- exp(-log(100) * (0:2) / 3)
  args <- outer(f, as.vector(tt))
  expect_f32(out$t, array(rbind(cos(args), sin(args)), c(6, 5, 2)), tol = 1e-4)
})

test_that("module arguments are checked", {
  p <- zl_params()
  expect_error(zl_linear(p, "x", 0, 3), class = "zulm_invalid_argument")
  expect_error(zl_linear(list(), "x", 2, 3), class = "zulm_invalid_argument")
  expect_error(zl_embedding(p, "e", 2.5, 3), class = "zulm_invalid_argument")
  zl_linear(p, "dup", 2, 2)
  expect_error(zl_linear(p, "dup", 2, 2), class = "zulm_invalid_argument")
  expect_error(zl_graph(function(i) zl_time_embedding(i$t, 5), p, list(t = c(1, 3))), class = "zulm_invalid_argument")
  expect_error(zl_graph(function(i) zl_time_embedding(i$t, 4), p, list(t = c(2, 3))), class = "zulm_shape_error")
})
