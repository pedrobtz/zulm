# Rounds doubles to the nearest single-precision value, as zulm stores f32.
f32 <- function(x) {
  readBin(writeBin(as.double(x), raw(), size = 4), "double", n = length(x), size = 4)
}

# A store with a few parameters of known values.
make_store <- function() {
  p <- zl_params()
  zl_param(p, "w", c(3, 2), init = zl_init_zeros())
  zl_param_set(p, "w", matrix(c(1.5, -2, 0.25, 3, -0.125, 7), 3, 2))
  zl_param(p, "b", 2, init = zl_init_ones())
  p
}

# Runs build() once on inputs of the given dims (numeric, or zl_int()) and
# returns zl_forward()'s outputs for `data`.
run_graph <- function(build, inputs, data, p = zl_params()) {
  g <- zl_graph(build, p, inputs)
  zl_forward(g, data)
}

# Pads a dim vector with 1s to length 4.
dim4 <- function(d) c(d, rep(1L, 4L - length(d)))

# R's version of ggml's broadcast: b repeated (tiled) to dims d, where every
# dimension of b divides the matching one of d.
bcast <- function(b, d) {
  db <- dim4(if (is.null(dim(b))) length(b) else dim(b))
  d4 <- dim4(d)
  idx <- arrayInd(seq_len(prod(d4)), d4) - 1L
  bi <- sweep(idx, 2L, db, `%%`)
  lin <- bi[, 1] + db[1] * (bi[, 2] + db[2] * (bi[, 3] + db[3] * bi[, 4])) + 1
  array(as.vector(b)[lin], d)
}

# A random shape of rank 1 to 4 with small dimensions, and a random shape
# that broadcasts to it (each dimension 1, the full size, or a divisor).
rand_dims <- function(rank = sample(1:4, 1), max = 5) {
  as.integer(sample(1:max, rank, replace = TRUE))
}
bcast_dims <- function(d) {
  vapply(d, function(n) {
    divs <- which(n %% seq_len(n) == 0)
    as.integer(sample(c(1L, n, divs), 1))
  }, integer(1))
}

rand_array <- function(d, lo = -2, hi = 2) {
  x <- f32(stats::runif(prod(d), lo, hi))
  if (length(d) > 1L) array(x, d) else x
}

# Expected f32 accuracy: relative to the values' scale.
expect_f32 <- function(object, expected, tol = 1e-5) {
  expect_equal(object, expected, tolerance = tol, ignore_attr = FALSE)
}
