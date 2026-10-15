# ZULM_GCTORTURE, the gctorture CI job's switch (R/load.R).

test_that("ZULM_GCTORTURE is read as a step and a per-call limit", {
  setup <- getFromNamespace("zlm_torture_setup", "zulm")
  zlm <- getFromNamespace(".zlm", "zulm")
  old <- zlm$torture
  withr::defer(assign("torture", old, envir = zlm))
  setup("1,3")
  expect_identical(zlm$torture$step, 1L)
  expect_identical(zlm$torture$limit, 3L)
  setup("20")
  expect_identical(zlm$torture$limit, 0L)
  setup("")
  expect_null(zlm$torture)
  setup("0")
  expect_identical(zlm$torture$step, 0L)
  for (bad in c("-1", "x", "1,-1", "1,2,3")) {
    expect_error(setup(bad), class = "zulm_invalid_argument")
  }
})

test_that("a tortured call returns what an untortured one does", {
  setup <- getFromNamespace("zlm_torture_setup", "zulm")
  zlm <- getFromNamespace(".zlm", "zulm")
  old <- zlm$torture
  withr::defer(assign("torture", old, envir = zlm))
  p <- zl_params()
  g <- zl_graph(function(x) zl_relu(x$a) * 2, p, inputs = list(a = 3))
  plain <- zl_forward(g, list(a = c(-1, 0.5, 2)))
  setup("1,2")
  tortured <- lapply(1:4, function(i) zl_forward(g, list(a = c(-1, 0.5, 2))))
  expect_identical(tortured, rep(list(plain), 4))
  expect_true(length(ls(zlm$torture$seen)) > 0)
  # every call is counted; only the first two of each kind were tortured
  expect_true(max(unlist(mget(ls(zlm$torture$seen), zlm$torture$seen))) >= 4L)
})
