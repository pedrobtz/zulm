# PyTorch fixtures (design §11): written by tools/make-torch-reference.py.
# Each module is declared in a fresh store, given the fixture's weights,
# and compared with PyTorch's outputs and gradients.

fixture <- function(name) zl_load(test_path("fixtures", name))

# Copies every parameter the modules declared from the fixture's store.
take_weights <- function(p, fx) {
  for (nm in zl_param_names(p)) zl_param_set(p, nm, zl_param_get(fx, nm))
  invisible(p)
}

# Forward output and gradients of sum(y * w) for every parameter of p.
check_fixture <- function(fx, p, build, inputs, data, tol = 1e-5, gtol = 1e-4) {
  g <- zl_graph(function(x) {
    y <- build(x)
    list(y = y, loss = zl_sum(y * x$w))
  }, p, c(inputs, list(w = inputs$x)), loss = "loss")
  full <- c(data, list(w = zl_param_get(fx, "input.w")))
  out <- zl_forward(g, full)
  expect_equal(out$y, zl_param_get(fx, "output.y"), tolerance = tol)
  tr <- zl_trainer(g, zl_sgd())
  for (nm in zl_param_names(p)) {
    expect_equal(zl_grad(tr, nm), zl_param_get(fx, paste0("grad.", nm)), tolerance = gtol, label = nm)
  }
}

test_that("the transformer block matches PyTorch", {
  fx <- fixture("block.gguf")
  m <- attr(fx, "metadata")
  p <- zl_params()
  blk <- zl_block(p, "blk", dim = m$dim, heads = m$heads, hidden = m$hidden)
  take_weights(p, fx)
  x <- zl_param_get(fx, "input.x")
  check_fixture(fx, p, function(i) blk(i$x), list(x = dim(x)), list(x = x))
})

test_that("AdaLN matches PyTorch", {
  fx <- fixture("adaln.gguf")
  m <- attr(fx, "metadata")
  p <- zl_params()
  ada <- zl_adaln(p, "ada", dim = m$dim, cond_dim = m$cond_dim)
  take_weights(p, fx)
  x <- zl_param_get(fx, "input.x")
  cc <- zl_param_get(fx, "input.c")
  check_fixture(fx, p, function(i) ada(i$x, i$c), list(x = dim(x), c = dim(cc)), list(x = x, c = cc))
})

test_that("the losses match PyTorch, values and gradients", {
  fx <- fixture("losses.gguf")
  p <- zl_params()
  zl_param(p, "logits", dim(zl_param_get(fx, "logits")), init = zl_init_zeros())
  zl_param(p, "pred", dim(zl_param_get(fx, "pred")), init = zl_init_zeros())
  take_weights(p, fx)
  d <- dim(zl_param_get(fx, "logits"))
  data <- list(prob = zl_param_get(fx, "input.prob"), target = zl_param_get(fx, "input.target"))
  for (loss in c("ce", "mse", "mae")) {
    g <- zl_graph(function(x) {
      l <- switch(loss,
        ce = zl_cross_entropy(zl_param_tensor(p, "logits"), x$prob),
        mse = zl_mse(zl_param_tensor(p, "pred"), x$target),
        mae = zl_mae(zl_param_tensor(p, "pred"), x$target)
      )
      list(loss = l)
    }, p, list(prob = d, target = d), loss = "loss")
    expect_equal(zl_forward(g, data)$loss, zl_param_get(fx, paste0("output.", loss)), tolerance = 1e-6)
    tr <- zl_trainer(g, zl_sgd())
    wrt <- if (loss == "ce") "logits" else "pred"
    expect_equal(zl_grad(tr, wrt), zl_param_get(fx, paste0("grad.", loss, ".", wrt)), tolerance = 1e-5)
  }
})

test_that("a TimesFM-style attention block matches PyTorch", {
  fx <- fixture("timesfm-attention.gguf")
  m <- attr(fx, "metadata")
  dim <- m$dim
  heads <- m$heads
  dh <- dim / heads
  p <- zl_params()
  lin <- lapply(c(q = "q", k = "k", v = "v", o = "o"), function(nm) zl_linear(p, paste0("tfm.", nm), dim, dim))
  zl_param(p, "tfm.q_norm", dh)
  zl_param(p, "tfm.k_norm", dh)
  zl_param(p, "tfm.q_scale", dh)
  take_weights(p, fx)
  x <- zl_param_get(fx, "input.x")
  n <- dim(x)[2]
  b <- dim(x)[3]
  mask <- zl_param_get(fx, "input.mask")
  build <- function(i) {
    heads4 <- function(t) zl_reshape(t, c(dh, heads, n, b))
    q <- zl_rope(heads4(lin$q(i$x)), i$pos, dh, base = 10000, mode = "half")
    q <- zl_rms_norm(q, eps = m$eps) * zl_param_tensor(p, "tfm.q_norm")
    q <- q * zl_scale(zl_softplus(zl_param_tensor(p, "tfm.q_scale")), 1.442695 / sqrt(dh))
    k <- zl_rope(heads4(lin$k(i$x)), i$pos, dh, base = 10000, mode = "half")
    k <- zl_rms_norm(k, eps = m$eps) * zl_param_tensor(p, "tfm.k_norm")
    v <- heads4(lin$v(i$x))
    per <- function(t) zl_permute(t, c(1, 3, 2, 4))
    s <- zl_crossprod(per(k), per(q))
    a <- zl_softmax(s, i$mask, scale = 1)
    o <- zl_crossprod(zl_permute(per(v), c(2, 1, 3, 4)), a)
    lin$o(zl_reshape(per(o), c(dim, n, b)))
  }
  check_fixture(
    fx, p, build,
    list(x = dim(x), pos = zl_int(n), mask = dim(mask)),
    list(x = x, pos = seq_len(n) - 1L, mask = mask),
    tol = 1e-5, gtol = 1e-4
  )
})

test_that("a Toto-style attention block matches PyTorch", {
  fx <- fixture("toto-attention.gguf")
  m <- attr(fx, "metadata")
  dim <- m$dim
  heads <- m$heads
  dh <- dim / heads
  p <- zl_params()
  lin <- lapply(c(q = "q", k = "k", v = "v", o = "o"), function(nm) {
    l <- zl_linear(p, paste0("toto.", nm), dim, dim, bias = FALSE)
    function(t) zl_scale(l(t), 1 / sqrt(dim))
  })
  take_weights(p, fx)
  x <- zl_param_get(fx, "input.x")
  n <- dim(x)[2]
  b <- dim(x)[3]
  build <- function(i) {
    heads4 <- function(t) zl_reshape(t, c(dh, heads, n, b))
    q <- zl_rope(heads4(lin$q(i$x)), i$pos, m$n_rot, base = 10000, mode = "interleaved") * i$fq
    k <- zl_rope(heads4(lin$k(i$x)), i$pos, m$n_rot, base = 10000, mode = "interleaved") * i$fk
    v <- heads4(lin$v(i$x))
    per <- function(t) zl_permute(t, c(1, 3, 2, 4))
    s <- zl_crossprod(per(k), per(q))
    a <- zl_softmax(zl_causal_mask(s), scale = 1 / dh)
    o <- zl_crossprod(zl_permute(per(v), c(2, 1, 3, 4)), a)
    lin$o(zl_reshape(per(o), c(dim, n, b)))
  }
  fq <- zl_param_get(fx, "input.fq")
  fk <- zl_param_get(fx, "input.fk")
  check_fixture(
    fx, p, build,
    list(x = dim(x), pos = zl_int(n), fq = dim(fq), fk = dim(fk)),
    list(x = x, pos = seq_len(n) - 1L, fq = fq, fk = fk)
  )
})
