# The §1.1 model of the design and its synthetic data: a marked process of
# event kinds and seven-dimensional values, an encoder block, a kind head
# and a rectified-flow head. Used by test-acceptance.R and by
# tools/acceptance/arrow-duckdb.R.

synth_batch <- function(step, n = 64L, b = 16L) {
  withr::local_seed(1000L + step)
  # kinds follow a Markov chain; targets an AR(1) around a mean per kind
  m <- matrix(c(
    .85, .05, .05, .05,
    .10, .80, .05, .05,
    .05, .10, .80, .05,
    .05, .05, .10, .80
  ), 4, byrow = TRUE)
  mu <- matrix(c(1, -1, 0.5, 0, 2, -0.5, 1, -1, 1, 0, -1, 0.5, 1, 0, 0.5, -1, 2, 0, 1, -1, 0, 1, -1, 1, 0, -1, 0.5, 2), 7, 4)
  kinds <- matrix(0L, n + 1L, b)
  y <- array(0, c(7, n + 1L, b))
  for (j in seq_len(b)) {
    k <- sample.int(4L, 1L)
    prev <- rep(0, 7)
    for (t in seq_len(n + 1L)) {
      k <- sample.int(4L, 1L, prob = m[k, ])
      prev <- 3 * mu[, k] + 0.5 * prev + 0.1 * rnorm(7)
      kinds[t, j] <- k - 1L
      y[, t, j] <- prev
    }
  }
  hist <- seq_len(n)
  nxt <- hist + 1L
  feat <- array(0, c(8, n, b))
  for (j in seq_len(b)) {
    feat[cbind(kinds[hist, j] + 1L, hist, j)] <- 1
    feat[5:8, , j] <- y[1:4, hist, j]
  }
  kind <- kinds[nxt, , drop = FALSE]
  list(
    feat = feat,
    kind = kind,
    kind_prob = array(diag(4)[, kind + 1L], c(4, n, b)),
    t = array(runif(n * b), c(1, n, b)),
    y = y[, nxt, , drop = FALSE]
  )
}

flow_model <- function() {
  p <- zl_params()
  m <- list(
    emb = zl_linear(p, "in", n_in = 8, n_out = 64),
    blk = zl_block(p, "enc1", dim = 64, heads = 4, causal = TRUE),
    ln = zl_norm(p, "enc.ln", 64),
    kh = zl_linear(p, "kind", 64, 4),
    ke = zl_embedding(p, "kind.emb", 4, 64),
    ada = zl_adaln(p, "flow.ada", dim = 64, cond_dim = 64),
    fin = zl_linear(p, "flow.in", 7, 64),
    fo = zl_linear(p, "flow.out", 64, 7)
  )
  list(p = p, m = m)
}

velocity <- function(m, z, kind, x_t, t) {
  cond <- z + m$ke(kind) + zl_time_embedding(t, 64)
  m$fo(zl_silu(m$ada(zl_rms_norm(m$fin(x_t)), cond)))
}
