# Helpers of test-decode.R.

# Causal multi-head attention whose keys and values go through a cache of
# `cap` positions held in states "<name>.kc" and "<name>.vc": the new
# positions' keys and values are written at `rows`, and the queries attend
# over the whole capacity under the additive `mask`.
decode_attention <- function(p, name, dim, heads, cap) {
  dh <- dim %/% heads
  zl_state(p, paste0(name, ".kc"), c(dh, cap, heads))
  zl_state(p, paste0(name, ".vc"), c(dh, cap, heads))
  lin <- function(part, x) {
    zl_crossprod(zl_param_tensor(p, paste0(name, ".", part, ".weight")), x) +
      zl_param_tensor(p, paste0(name, ".", part, ".bias"))
  }
  function(x, rows, mask) {
    n <- dim(x)[[2L]]
    split <- function(t) zl_permute(zl_reshape(t, c(dh, heads, n)), c(1, 3, 2))
    q <- split(lin("q", x))
    k <- zl_state_write(zl_param_tensor(p, paste0(name, ".kc")), split(lin("k", x)), rows)
    v <- zl_state_write(zl_param_tensor(p, paste0(name, ".vc")), split(lin("v", x)), rows)
    s <- zl_crossprod(k, q)                                  # c(cap, n, heads)
    a <- zl_softmax(s, mask = mask, scale = 1 / sqrt(dh))
    o <- zl_crossprod(zl_permute(v, c(2, 1, 3)), a)          # c(dh, n, heads)
    lin("o", zl_reshape(zl_permute(o, c(1, 3, 2)), c(dim, n)))
  }
}
