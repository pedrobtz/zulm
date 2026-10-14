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
