# Helpers of test-ops.R.

as_r <- function(x, d) if (length(d) > 1L) array(x, d) else as.vector(x)

rope_r <- function(x, pos, n_dims, base, mode) {
  d <- dim(x)
  out <- x
  half <- n_dims / 2
  for (t in seq_len(d[3])) {
    p <- pos[t]
    for (i in 0:(half - 1)) {
      theta <- p * base^(-2 * i / n_dims)
      if (mode == "interleaved") {
        j0 <- 2 * i + 1
        j1 <- 2 * i + 2
      } else {
        j0 <- i + 1
        j1 <- i + 1 + half
      }
      a <- x[j0, , t, ]
      b <- x[j1, , t, ]
      out[j0, , t, ] <- a * cos(theta) - b * sin(theta)
      out[j1, , t, ] <- a * sin(theta) + b * cos(theta)
    }
  }
  out
}
