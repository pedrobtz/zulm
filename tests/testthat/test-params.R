test_that("a store starts empty and grows in declaration order", {
  p <- zl_params()
  expect_s3_class(p, "zl_params")
  expect_identical(zl_param_names(p), character())
  expect_identical(zl_param_count(p), 0)
  zl_param(p, "a", c(2, 3))
  zl_param(p, "b", 4)
  zl_param(p, "c", c(1, 2, 3, 4))
  expect_identical(zl_param_names(p), c("a", "b", "c"))
  expect_identical(zl_param_count(p), 6 + 4 + 24)
})

test_that("f32 values round-trip exactly after rounding to single precision", {
  withr::local_seed(1)
  p <- zl_params()
  x <- array(rnorm(2 * 3 * 4 * 5) * 1e3, c(2, 3, 4, 5))
  zl_param(p, "x", dim(x))
  zl_param_set(p, "x", x)
  expect_identical(zl_param_get(p, "x"), array(f32(x), dim(x)))
  y <- zl_param_get(p, "x")
  zl_param_set(p, "x", y)
  expect_identical(zl_param_get(p, "x"), y)
})

test_that("one-dimensional parameters come back as plain vectors", {
  p <- zl_params()
  zl_param(p, "v", 5, init = zl_init_ones())
  expect_identical(zl_param_get(p, "v"), rep(1, 5))
  zl_param_set(p, "v", 1:5)
  expect_identical(zl_param_get(p, "v"), as.double(1:5))
})

test_that("f16 and bf16 values are within their rounding", {
  withr::local_seed(2)
  x <- matrix(runif(60, -100, 100), 6, 10)
  for (type in c("f16", "bf16")) {
    p <- zl_params()
    zl_param(p, "h", dim(x), type = type)
    zl_param_set(p, "h", x)
    got <- zl_param_get(p, "h")
    rel <- max(abs(got - x) / abs(x))
    bound <- if (type == "f16") 2^-11 else 2^-8
    expect_lte(rel, bound)
    expect_gt(rel, 0)
    # values already representable round-trip exactly
    zl_param_set(p, "h", got)
    expect_identical(zl_param_get(p, "h"), got)
  }
})

test_that("values beyond the type's range become infinite with a warning", {
  p <- zl_params()
  zl_param(p, "w", 3, init = zl_init_zeros())
  expect_warning(zl_param_set(p, "w", c(1e39, -1e39, 1)), class = "zulm_range_warning")
  expect_identical(zl_param_get(p, "w"), c(Inf, -Inf, 1))
  zl_param(p, "h", 2, type = "f16", init = zl_init_zeros())
  expect_warning(zl_param_set(p, "h", c(1e5, 1)), class = "zulm_range_warning")
  expect_identical(zl_param_get(p, "h"), c(Inf, 1))
  expect_no_warning(zl_param_set(p, "w", c(Inf, NaN, 0)))
  expect_true(is.nan(zl_param_get(p, "w")[2]))
})

test_that("initial values follow the counter layout of design §7.2", {
  p <- zl_params()
  zl_param(p, "layer1.w", c(4, 3), init = zl_init_normal(sd = 0.5))
  zl_param(p, "layer1.b", 3, init = zl_init_uniform(0.1))
  key <- zulm:::zlm_session_key()
  stream <- function(name) zurand::rng_fold(zurand::rng_fold(key, name), 0L)
  expect_identical(
    zl_param_get(p, "layer1.w"),
    matrix(f32(zurand::rng_normal(stream("layer1.w"), 12, sd = 0.5)), 4, 3)
  )
  expect_identical(
    zl_param_get(p, "layer1.b"),
    f32(zurand::rng_uniform(stream("layer1.b"), 3, min = -0.1, max = 0.1))
  )
})

test_that("a parameter's values do not depend on the others", {
  p <- zl_params()
  zl_param(p, "a", 5)
  zl_param(p, "b", 5)
  q <- zl_params()
  zl_param(q, "b", 5)
  expect_identical(zl_param_get(q, "b"), zl_param_get(p, "b"))
})

test_that("zeros and ones initialise exactly", {
  p <- zl_params()
  zl_param(p, "z", c(2, 2), init = zl_init_zeros(), type = "bf16")
  zl_param(p, "o", c(2, 2), init = zl_init_ones(), type = "f16")
  expect_identical(zl_param_get(p, "z"), matrix(0, 2, 2))
  expect_identical(zl_param_get(p, "o"), matrix(1, 2, 2))
})

test_that("zl_param_tensor() describes a parameter", {
  p <- make_store()
  h <- zl_param_tensor(p, "w")
  expect_s3_class(h, "zl_param_ref")
  expect_identical(h$dim, c(3, 2))
  expect_identical(h$type, "f32")
  expect_output(print(h), "zl_param_ref")
})

test_that("stores print a summary", {
  p <- make_store()
  expect_output(print(p), "2 tensors, 8 values")
  expect_output(print(zl_params()), "0 tensors")
})

test_that("tensors keep their addresses as the store grows", {
  p <- zl_params()
  zl_param(p, "first", 3, init = zl_init_ones())
  for (i in 1:50) zl_param(p, paste0("p", i), c(10, 10))
  expect_identical(zl_param_get(p, "first"), rep(1, 3))
  expect_length(zl_param_names(p), 51L)
})
