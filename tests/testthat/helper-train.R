# Helpers of test-train.R.

linear_graph <- function(p, n = 8) {
  zl_graph(
    function(x) list(
      pred = zl_crossprod(zl_param_tensor(p, "w"), x$x),
      loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), x$x), x$y)
    ),
    p,
    inputs = list(x = c(3, n), y = c(1, n)),
    loss = "loss"
  )
}

linear_data <- function(n = 8) {
  x <- rand_array(c(3, n))
  list(x = x, y = matrix(f32(c(1, -2, 0.5) %*% x), 1, n))
}
