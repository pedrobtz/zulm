test_that("a variant the CPU supports is loaded", {
  info <- zulm_info()
  expect_true(info$variant %in% info$variants_usable)
  expect_true(info$variant %in% info$variants_installed)
  expect_identical(info$ggml_version, "0.26.0")
  expect_match(info$ggml_commit, "^[0-9a-f]{40}$")
})

test_that("the best usable variant is the default", {
  info <- zulm_info()
  expect_identical(info$variant, info$variants_usable[[1L]])
})

test_that("a native build loads its one variant", {
  info <- zulm_info()
  skip_if_not(info$variant == "native", "not a native build")
  expect_identical(info$variants_installed, "native")
  expect_identical(info$variants_usable, "native")
  expect_true(nzchar(info$native_cflags) || nzchar(info$native_cxxflags))
})

test_that("the architecture's variants are installed", {
  info <- zulm_info()
  skip_if(info$variant == "native", "native build")
  expected <- switch(
    R.version$arch,
    x86_64 = ,
    amd64 = c("baseline", "sse42", "avx2"),
    aarch64 = ,
    arm64 = "neon",
    "generic"
  )
  expect_setequal(info$variants_installed, expected)
})

test_that("options(zulm.variant =) forces each usable variant", {
  on.exit(zulm:::zlm_load_variant(NULL), add = TRUE)
  for (v in zulm_info()$variants_usable) {
    zulm:::zlm_load_variant(v)
    expect_identical(zulm_info()$variant, v)
    bench <- zulm:::zlm_call("zlm_bench_mul_mat", 64L, 1L, 1L)
    expect_true(is.finite(bench[[2L]]))
  }
})

test_that("a variant this CPU cannot run is refused", {
  on.exit(zulm:::zlm_load_variant(NULL), add = TRUE)
  expect_error(
    zulm:::zlm_load_variant("no-such-variant"),
    class = "zulm_invalid_argument"
  )
  expect_error(
    zulm:::zlm_load_variant(c("a", "b")),
    class = "zulm_invalid_argument"
  )
})

test_that("the product is identical at one, two and four threads", {
  h <- vapply(
    c(1L, 2L, 4L),
    function(t) zulm:::zlm_call("zlm_bench_mul_mat", 96L, 1L, t)[[3L]],
    numeric(1)
  )
  expect_identical(h[[1L]], h[[2L]])
  expect_identical(h[[1L]], h[[3L]])
})
