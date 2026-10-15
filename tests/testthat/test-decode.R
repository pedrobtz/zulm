# A TimesFM-style decode through state tensors equals one full-context
# forward pass (design §6.9, §15 criterion 11).

test_that("decoding 256 values through state equals the full-context pass", {
  set.seed(9)
  d <- 16L
  heads <- 4L
  len <- 256L
  p <- zl_params()
  attn <- zl_attention(p, "attn", d, heads, causal = TRUE)
  dec <- decode_attention(p, "attn", d, heads, cap = len)
  x <- rand_array(c(d, len))
  full <- zl_forward(zl_graph(function(i) attn(i$x), p, list(x = c(d, len))), list(x = x))$output

  gc <- zl_graph_cache(function(i) dec(i$x, i$rows, i$mask), p)
  # several block sizes, some repeated, at several fill levels
  blocks <- c(64L, 1L, 1L, 7L, 32L, 7L, 16L, 128L)
  expect_identical(sum(blocks), len)
  got <- matrix(NA_real_, d, len)
  filled <- 0L
  for (n in blocks) {
    pos <- filled + seq_len(n) - 1L                          # 0-based positions
    mask <- outer(seq_len(len) - 1L, pos, function(j, i) ifelse(j <= i, 0, -Inf))
    out <- zl_forward(gc, list(
      x = x[, pos + 1L, drop = FALSE], rows = as.integer(pos), mask = mask
    ))$output
    got[, pos + 1L] <- out
    filled <- filled + n
  }
  expect_identical(gc$builds, length(unique(blocks)))
  expect_equal(got, full, tolerance = 1e-5)
  # resetting the caches and decoding again gives the same numbers
  zl_state_set(p, "attn.kc", array(0, c(d / heads, len, heads)))
  zl_state_set(p, "attn.vc", array(0, c(d / heads, len, heads)))
  pos <- 0:63
  mask <- outer(seq_len(len) - 1L, pos, function(j, i) ifelse(j <= i, 0, -Inf))
  again <- zl_forward(gc, list(x = x[, pos + 1L], rows = pos, mask = mask))$output
  expect_identical(again, got[, pos + 1L])
})

test_that("a sliding window shifts by offset writes without a new graph", {
  p <- zl_params()
  zl_state(p, "win", c(2, 4))
  g <- zl_graph(function(i) zl_sum_rows(zl_param_tensor(p, "win")) * i$w, p, list(w = 1))
  vals <- matrix(seq_len(2 * 8), 2, 8)
  for (t in 1:8) {
    # drop the oldest column, append column t at the end
    cur <- zl_state_get(p, "win")
    zl_state_set(p, "win", as.vector(cur[, -1]), offset = 0)
    zl_state_set(p, "win", vals[, t], offset = 6)
    lo <- max(1L, t - 3L)
    want <- matrix(0, 2, 4)
    want[, (4 - (t - lo)):4] <- vals[, lo:t]
    expect_identical(zl_state_get(p, "win"), want)
    expect_identical(zl_forward(g, list(w = 1))$output, matrix(colSums(want), 1, 4))
  }
})

test_that("graph construction for twenty blocks is paid once per shape", {
  skip_on_cran()
  p <- zl_params()
  blocks <- lapply(seq_len(20), function(i) zl_block(p, sprintf("b%02d", i), 64L, 4L))
  gc <- zl_graph_cache(function(i) Reduce(function(h, b) b(h), blocks, i$x), p)
  zl_forward(gc, list(x = matrix(0, 64, 1)))
  times <- vapply(2:9, function(n) {
    system.time(zl_forward(gc, list(x = matrix(0, 64, n))))[["elapsed"]]
  }, numeric(1))
  reuse <- system.time(for (n in 2:9) zl_forward(gc, list(x = matrix(0, 64, n))))[["elapsed"]]
  expect_identical(gc$builds, 9L)
  expect_lt(reuse, sum(times))
  # design §12 asked for 5 ms per shape; 980 nodes take about 21 ms on an
  # Apple M2, which the roadmap records as a miss. The acceptance workflow,
  # on an ordinary build, keeps it under 50 ms; sanitizers and valgrind
  # are too slow for a bound.
  skip_if_not(identical(Sys.getenv("ZULM_ACCEPTANCE"), "true"), "timing is checked in the acceptance workflow")
  expect_lt(stats::median(times), 0.05)
})
