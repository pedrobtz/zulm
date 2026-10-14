# Design §15 criterion 2 (Stage 6): the §1.1 model trains on a synthetic
# marked process. CI only (the `acceptance` workflow sets
# ZULM_ACCEPTANCE=true): two runs of 1 000 AdamW steps take minutes.

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

train_flow <- function(threads, steps = 1000L) {
  withr::local_options(zulm.threads = threads)
  old <- zl_seed(2026)
  on.exit(zl_seed(old), add = TRUE)
  fm <- flow_model()
  m <- fm$m
  build <- function(x) {
    z <- m$ln(m$blk(m$emb(x$feat)))
    ce <- zl_cross_entropy(m$kh(z), x$kind_prob)
    eps <- zl_noise_normal(c(7, 64, 16), "x0")
    x_t <- (1 - x$t) * eps + x$t * x$y
    v <- velocity(m, z, x$kind, x_t, x$t)
    list(loss = ce + zl_mse(v, x$y - eps), latent = z)
  }
  g <- zl_graph(build, fm$p,
    inputs = list(feat = c(8, 64, 16), kind = zl_int(64, 16), kind_prob = c(4, 64, 16), t = c(1, 64, 16), y = c(7, 64, 16)),
    loss = "loss"
  )
  tr <- zl_trainer(g, zl_adamw(lr = 1e-3))
  losses <- vapply(seq_len(steps), function(i) zl_step(tr, synth_batch(i)), numeric(1))
  list(p = fm$p, m = m, g = g, losses = losses,
       params = lapply(stats::setNames(nm = zl_param_names(fm$p)), function(nm) zl_param_get(fm$p, nm)))
}

test_that("the §1.1 model trains, reproducibly at one and two threads", {
  skip_if_not(identical(Sys.getenv("ZULM_ACCEPTANCE"), "true"), "acceptance runs on CI")
  one <- train_flow(1L)
  first <- mean(one$losses[1:100])
  last <- mean(one$losses[901:1000])
  message(sprintf("mean loss: steps 1-100 %.4f, steps 901-1000 %.4f (%.0f%%)", first, last, 100 * last / first))
  expect_lt(last, 0.6 * first)
  two <- train_flow(2L)
  expect_identical(two$losses, one$losses)
  expect_identical(two$params, one$params)

  # Euler sampling of the flow head on a forward-only graph, the latent fixed
  b <- synth_batch(5000L)
  z <- zl_forward(one$g, b, outputs = "latent")$latent
  head <- zl_graph(
    function(x) velocity(one$m, x$z, x$kind, x$x, x$t),
    one$p,
    inputs = list(z = c(64, 64, 16), kind = zl_int(64, 16), x = c(7, 64, 16), t = c(1, 64, 16))
  )
  x <- zl_rng_normal(c(7, 64, 16), "sample")
  n_steps <- 8L
  for (s in seq_len(n_steps)) {
    t <- array((s - 1) / n_steps, c(1, 64, 16))
    v <- zl_forward(head, list(z = z, kind = b$kind, x = x, t = t))$output
    x <- x + v / n_steps
  }
  expect_true(all(is.finite(x)))
  expect_lt(abs(mean(x) - mean(b$y)), 1)
})
