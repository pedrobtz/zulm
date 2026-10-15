# Helpers of test-fuzz-findings.R.

# A graph whose loss is the sum of build(w), trained one step, with w a
# 4 x 4 x 4 x 4 parameter of ones and one unused input.
train_once <- function(build, dim = c(4, 4, 4, 4), clip_norm = NULL) {
  p <- zl_params()
  zl_param(p, "w", dim, init = zl_init_ones())
  g <- zl_graph(
    function(x) list(loss = zl_sum(build(zl_param_tensor(p, "w"))) + 0 * zl_sum(x$x)),
    p,
    inputs = list(x = 1),
    loss = "loss"
  )
  tr <- zl_trainer(g, zl_sgd(lr = 0.1), clip_norm = clip_norm)
  zl_step(tr, list(x = 0))
}
