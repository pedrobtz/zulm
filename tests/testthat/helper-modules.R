# Helpers of test-modules.R.

lin_r <- function(p, name, x, bias = TRUE) {
  w <- zl_param_get(p, paste0(name, ".weight"))
  d <- dim(x)
  y <- crossprod(w, matrix(x, d[1]))
  if (bias) y <- y + zl_param_get(p, paste0(name, ".bias"))
  array(y, c(ncol(w), d[-1]))
}

layer_norm_r <- function(x, w = NULL, b = NULL, eps = 1e-5) {
  d <- dim(x)
  m <- matrix(x, d[1])
  c <- sweep(m, 2, colMeans(m))
  y <- sweep(c, 2, sqrt(colMeans(c^2) + eps), "/")
  if (!is.null(w)) y <- y * w + b
  array(y, d)
}

attn_r <- function(p, name, x, heads, causal = TRUE) {
  d <- dim(x)
  dim <- d[1]
  n <- d[2]
  b <- d[3]
  dh <- dim / heads
  q <- lin_r(p, paste0(name, ".q"), x)
  k <- lin_r(p, paste0(name, ".k"), x)
  v <- lin_r(p, paste0(name, ".v"), x)
  out <- array(0, d)
  for (bi in 1:b) for (h in 1:heads) {
    rows <- ((h - 1) * dh + 1):(h * dh)
    qh <- q[rows, , bi]
    kh <- k[rows, , bi]
    vh <- v[rows, , bi]
    s <- crossprod(kh, qh) / sqrt(dh)    # n_k x n_q
    if (causal) s[row(s) > col(s)] <- -Inf
    a <- apply(s, 2, function(col) exp(col - max(col)) / sum(exp(col - max(col))))
    out[rows, , bi] <- vh %*% a
  }
  lin_r(p, paste0(name, ".o"), out)
}
