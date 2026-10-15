# Flash attention against the plain form (design §6.9, open question 16).

test_that("flash attention agrees with the plain form at several shapes", {
  set.seed(10)
  cases <- list(
    flash_case(16, 64, 64, 4),                       # square, causal
    flash_case(16, 1, 600, 4),                       # one query, long cache
    flash_case(64, 7, 300, 2),                       # a block of new positions
    flash_case(8, 33, 40, 4, hk = 2),                # grouped keys and values
    flash_case(8, 5, 9, 2, b = 3),                   # a batch
    flash_case(8, 6, 10, 2, dv = 4),                 # values of another width
    flash_case(8, 6, 10, 2, mask = FALSE),           # no mask
    flash_case(8, 6, 10, 2, mask_heads = 2)          # a mask per head
  )
  for (cs in cases) {
    out <- run_flash(cs)
    expect_identical(dim(out$flash), dim(out$plain))
    scale <- max(abs(out$plain))
    expect_lt(max(abs(out$flash - out$plain)) / scale, 1e-5)
  }
  # and with R in double precision, for a few queries
  cs <- cases[[4]]
  out <- run_flash(cs)
  for (i in c(1, 17, 33)) for (h in 1:4) {
    want <- r_attention(cs$data$q, cs$data$k, cs$data$v, cs$data$m, 1 / sqrt(8), i, h)
    expect_equal(out$flash[, h, i, 1], want, tolerance = 1e-5)
  }
})

test_that("flash attention gives the same bits at one, two and four threads", {
  set.seed(11)
  for (cs in list(flash_case(16, 1, 600, 4), flash_case(16, 100, 1000, 2), flash_case(8, 1, 513, 3, mask = FALSE))) {
    res <- lapply(c(1, 2, 4), function(t) {
      local_threads(t)
      run_flash(cs)$flash
    })
    expect_identical(res[[2]], res[[1]])
    expect_identical(res[[3]], res[[1]])
  }
})

test_that("flash attention has no gradient and checks its operands", {
  p <- zl_params()
  zl_param(p, "w", c(4, 4))
  g <- zl_graph(function(x) {
    q <- zl_reshape(zl_crossprod(zl_param_tensor(p, "w"), x$x), c(4, 3, 1))
    list(loss = zl_sum(zl_flash_attention(q, x$k, x$v)))
  }, p, list(x = c(4, 3), k = c(4, 5, 1), v = c(4, 5, 1)), loss = "loss")
  expect_error(zl_trainer(g, zl_sgd()), class = "zulm_unsupported_backward")
  bad <- function(inputs, build = function(x) zl_flash_attention(x$q, x$k, x$v, x$m)) {
    expect_error(zl_graph(build, zl_params(), inputs), class = "zulm_shape_error")
  }
  ok <- list(q = c(4, 3, 2), k = c(4, 5, 2), v = c(4, 5, 2), m = c(5, 3))
  expect_no_error(zl_graph(function(x) zl_flash_attention(x$q, x$k, x$v, x$m), zl_params(), ok))
  bad(utils::modifyList(ok, list(k = c(3, 5, 2))))   # key width
  bad(utils::modifyList(ok, list(v = c(4, 6, 2))))   # value count
  bad(utils::modifyList(ok, list(k = c(4, 5, 3), v = c(4, 5, 3))))   # heads
  bad(utils::modifyList(ok, list(m = c(4, 3))))      # mask rows
  bad(utils::modifyList(ok, list(m = c(5, 2))))      # mask columns
  bad(utils::modifyList(ok, list(q = c(4, 3, 2, 2))))  # batch
  expect_error(
    zl_graph(function(x) zl_flash_attention(x$q, x$k, x$v, scale = NA), zl_params(), ok[1:3]),
    class = "zulm_invalid_argument"
  )
  expect_error(
    zl_graph(function(x) zl_flash_attention(x$q, x$k, zl_cast(x$v, "f16")), zl_params(), ok[1:3]),
    class = "zulm_type_error"
  )
})

test_that("a 10 240-position context attends within the memory cap", {
  skip_if_not(identical(Sys.getenv("ZULM_ACCEPTANCE"), "true"), "run in the acceptance workflow")
  set.seed(12)
  n <- 10240
  d <- 16
  h <- 2
  dat <- list(
    q = array(f32(stats::rnorm(d * n * h)), c(d, n, h)),
    k = array(f32(stats::rnorm(d * n * h)), c(d, n, h)),
    v = array(f32(stats::rnorm(d * n * h)), c(d, n, h))
  )
  # the scores alone would take 10 240^2 x 2 heads x 4 bytes = 800 MB
  withr::local_options(zulm.max_bytes = 64 * 2^20)
  p <- zl_params()
  ins <- list(q = c(d, n, h), k = c(d, n, h), v = c(d, n, h))
  expect_error(
    zl_graph(function(x) plain_attention(x$q, x$k, x$v, NULL, 0.25), p, ins),
    class = "zulm_memory_error"
  )
  g <- zl_graph(function(x) zl_flash_attention(x$q, x$k, x$v, scale = 0.25), p, ins)
  expect_lt(sum(zl_graph_info(g)$bytes), 64 * 2^20)
  out <- zl_forward(g, dat)$output
  expect_identical(dim(out), c(16L, 2L, 10240L))
  for (i in c(1, 5000, 10240)) for (hh in 1:2) {
    want <- r_attention(array(dat$q, c(d, n, h, 1)), array(dat$k, c(d, n, h, 1)),
                        array(dat$v, c(d, n, h, 1)), NULL, 0.25, i, hh)
    expect_equal(out[, hh, i], want, tolerance = 1e-5)
  }
})
