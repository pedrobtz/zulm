# Helpers of test-fixtures.R.

fixture <- function(name) zl_load(test_path("fixtures", name))

# Copies every parameter the modules declared from the fixture's store.
take_weights <- function(p, fx) {
  for (nm in zl_param_names(p)) zl_param_set(p, nm, zl_param_get(fx, nm))
  invisible(p)
}

# Forward output and gradients of sum(y * w) for every parameter of p.
check_fixture <- function(fx, p, build, inputs, data, tol = 1e-5, gtol = 1e-4) {
  g <- zl_graph(function(x) {
    y <- build(x)
    list(y = y, loss = zl_sum(y * x$w))
  }, p, c(inputs, list(w = inputs$x)), loss = "loss")
  full <- c(data, list(w = zl_param_get(fx, "input.w")))
  out <- zl_forward(g, full)
  expect_equal(out$y, zl_param_get(fx, "output.y"), tolerance = tol)
  tr <- zl_trainer(g, zl_sgd())
  for (nm in zl_param_names(p)) {
    expect_equal(zl_grad(tr, nm), zl_param_get(fx, paste0("grad.", nm)), tolerance = gtol, label = nm)
  }
}
