test_that("zl_threads() sets, caps and restores the session's threads", {
  old <- zl_threads()
  on.exit(zl_threads(old), add = TRUE)
  expect_identical(zl_threads(1), old)
  expect_identical(zl_threads(), 1L)
  cores <- zulm:::zlm_call("zlm_cores")
  zl_threads(10000)
  expect_lte(zl_threads(), max(2L, cores))
  withr::local_envvar(`_R_CHECK_LIMIT_CORES_` = "TRUE")
  zl_threads(64)
  expect_lte(zl_threads(), 2L)
  expect_error(zl_threads(0), class = "zulm_invalid_argument")
  expect_error(zl_threads(1.5), class = "zulm_invalid_argument")
  expect_error(zl_threads("2"), class = "zulm_invalid_argument")
})

test_that("an interrupt stops a long graph with zulm_interrupt, and the session goes on", {
  skip_on_cran()
  skip_on_os("windows")
  local_threads(2)
  p <- zl_params()
  zl_param(p, "w", c(512, 512), init = zl_init_normal(0.04))
  # one computation of about a thousand nodes, long enough to interrupt
  # inside ggml rather than between two R calls
  layers <- 1000L
  g <- zl_graph(function(x) {
    h <- x$x
    for (i in seq_len(layers)) h <- zl_tanh(zl_crossprod(zl_param_tensor(p, "w"), h))
    h
  }, p, list(x = c(512, 256)))
  data <- list(x = matrix(0.1, 512, 256))
  t0 <- Sys.time()
  zl_forward(g, data)
  once <- as.numeric(Sys.time() - t0, units = "secs")
  skip_if(once < 0.5, "the graph runs too fast to interrupt reliably")
  sig <- tools::SIGINT
  pid <- Sys.getpid()
  # interrupt a fifth of the way in, from outside the process
  system(sprintf("(sleep %.2f; kill -%d %d) >/dev/null 2>&1 &", once / 5 + 0.1, sig, pid))
  got <- tryCatch(
    {
      zl_forward(g, data)
      "finished"
    },
    zulm_interrupt = function(e) "zulm_interrupt",
    interrupt = function(e) "interrupt"
  )
  expect_identical(got, "zulm_interrupt")
  small <- zl_graph(function(x) x$a * 2, zl_params(), list(a = 2))
  expect_identical(zl_forward(small, list(a = c(1, 2)))$output, c(2, 4))
  expect_identical(dim(zl_forward(g, data)$output), c(512L, 256L))
})

test_that("a forked child trains with two threads", {
  skip_on_cran()
  skip_on_os("windows")
  skip_if_not_installed("parallel")
  local_threads(2)
  parent <- train_small()
  job <- parallel::mcparallel(train_small())
  child <- parallel::mccollect(job)[[1]]
  expect_identical(child, parent)
  expect_identical(train_small(), parent)
})

test_that("forward outputs are bit-identical at one, two and four threads, in every variant", {
  on.exit(zulm:::zlm_load_variant(NULL), add = TRUE)
  for (v in zulm_info()$variants_usable) {
    zulm:::zlm_load_variant(v)
    one <- fixture_forward("block.gguf", 1L)
    expect_identical(fixture_forward("block.gguf", 2L), one)
    expect_identical(fixture_forward("block.gguf", 4L), one)
  }
})

test_that("the same seed gives the same training twice", {
  local_threads(2)
  a <- train_small()
  b <- train_small()
  expect_identical(a, b)
})

test_that("resizing the pool under live graphs and trainers is safe", {
  old <- zl_threads()
  on.exit(zl_threads(old), add = TRUE)
  p <- zl_params()
  zl_param(p, "w", c(8, 1), init = zl_init_normal(0.1))
  g <- zl_graph(function(i) list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), i$x), i$y)),
                p, list(x = c(8, 16), y = c(1, 16)), loss = "loss")
  tr <- zl_trainer(g, zl_sgd(lr = 0.1))
  d <- list(x = matrix(0.5, 8, 16), y = matrix(1, 1, 16))
  for (n in c(1, 2, 4, 3, 2, 1, 4)) {
    zl_threads(n)
    expect_true(is.finite(zl_step(tr, d)))
    expect_true(is.finite(zl_forward(g, d)$loss))
    expect_true(all(is.finite(zl_grad(tr, "w"))))
  }
  old_poll <- zulm:::zlm_call("zlm_poll_", 50L)
  expect_true(is.finite(zl_step(tr, d)))
  zulm:::zlm_call("zlm_poll_", old_poll)
  expect_true(is.finite(zl_step(tr, d)))
})
