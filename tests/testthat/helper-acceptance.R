# Helpers of test-acceptance.R.

train_flow <- function(threads, steps = 1000L, clip_norm = NULL) {
  local_threads(threads)
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
  tr <- zl_trainer(g, zl_adamw(lr = 1e-3), clip_norm = clip_norm)
  losses <- vapply(seq_len(steps), function(i) zl_step(tr, synth_batch(i)), numeric(1))
  list(p = fm$p, m = m, g = g, losses = losses,
       params = lapply(stats::setNames(nm = zl_param_names(fm$p)), function(nm) zl_param_get(fm$p, nm)))
}
