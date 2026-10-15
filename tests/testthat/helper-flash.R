# Helpers of test-flash.R.

# The plain form in the flash layout: q c(d, n, h, b), k c(d, nk, hk, b),
# v c(dv, nk, hk, b) -> c(dv, h, n, b), with hk dividing h.
plain_attention <- function(q, k, v, mask, scale) {
  s <- zl_crossprod(k, q)                                  # c(nk, n, h, b)
  a <- zl_softmax(s, mask = mask, scale = scale)
  o <- zl_crossprod(zl_permute(v, c(2, 1, 3, 4)), a)       # c(dv, n, h, b)
  zl_cont(zl_permute(o, c(1, 3, 2, 4)))
}

# R's attention in double precision for query i (1-based) of head h, batch b.
r_attention <- function(q, k, v, mask, scale, i, h, b = 1) {
  hk <- dim(k)[[3]]
  kh <- (h - 1) %/% (dim(q)[[3]] / hk) + 1
  s <- drop(crossprod(k[, , kh, b], q[, i, h, b])) * scale
  if (!is.null(mask)) s <- s + mask[, i]
  w <- exp(s - max(s))
  drop(v[, , kh, b] %*% (w / sum(w)))
}

flash_case <- function(d, n, nk, h, hk = h, b = 1, dv = d, mask = TRUE, mask_heads = 1) {
  dims <- list(q = c(d, n, h, b), k = c(d, nk, hk, b), v = c(dv, nk, hk, b))
  data <- lapply(dims, function(x) array(f32(stats::rnorm(prod(x))), x))
  m <- NULL
  if (mask) {
    m <- outer(seq_len(nk), seq_len(n), function(j, i) ifelse(j <= i + nk - n, 0, -Inf))
    if (mask_heads > 1) {
      m <- array(rep(m, mask_heads), c(nk, n, mask_heads))
      m[, , 2][m[, , 2] == 0] <- -0.5                       # differs by head
    }
    dims$m <- dim(m)
    data$m <- m
  }
  list(dims = dims, data = data, d = d)
}

run_flash <- function(cs) {
  p <- zl_params()
  g <- zl_graph(function(x) list(
    flash = zl_flash_attention(x$q, x$k, x$v, x$m, scale = 1 / sqrt(cs$d)),
    plain = plain_attention(x$q, x$k, x$v, x$m, 1 / sqrt(cs$d))
  ), p, cs$dims)
  zl_forward(g, cs$data)
}
