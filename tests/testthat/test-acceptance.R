# Design §15 criterion 2 (Stage 6): the §1.1 model trains on a synthetic
# marked process. CI only (the `acceptance` workflow sets
# ZULM_ACCEPTANCE=true): two runs of 1 000 AdamW steps take minutes.

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

test_that("200 steps of the §1.1 model are bit-identical at one, two and four threads", {
  skip_if_not(identical(Sys.getenv("ZULM_ACCEPTANCE"), "true"), "acceptance runs on CI")
  for (clip in list(NULL, 1)) {
    one <- train_flow(1L, steps = 200L, clip_norm = clip)
    for (threads in c(2L, 4L)) {
      other <- train_flow(threads, steps = 200L, clip_norm = clip)
      expect_identical(other$losses, one$losses)
      expect_identical(other$params, one$params)
    }
  }
})
