# The C table (design §8, inst/include/zulm.h), driven the way zuflow and
# zutsfm drive it: a fixture package, tests/testthat/zulmclient, is built
# against the header and calls the table from C. Every value it returns
# must equal what the R functions give.
#
# Not on CRAN: building a package takes seconds and needs a toolchain.
# Everywhere else a build failure is a test failure, not a skip.

test_that("the table has the header's size and reports ggml's version", {
  skip_on_cran()
  ns <- zulm_client()
  info <- ns$client_info()
  expect_identical(info[[1]], info[[2]])
  expect_true(startsWith(info[[3]], paste0(zulm_info()$ggml_version, "+p")))
})

test_that("a consumer built against a longer table is refused", {
  skip_on_cran()
  ns <- zulm_client()
  expect_no_error(ns$client_require(0L))
  expect_error(ns$client_require(8L), class = "simpleError")
})

test_that("a sampler loop in C equals zl_forward() on zl_rng_normal() draws", {
  skip_on_cran()
  withr::local_seed(11)
  old <- zl_seed(5)
  withr::defer(zl_seed(old))
  p <- zl_params()
  zl_param(p, "w", c(3, 2))
  zl_param(p, "e", c(2, 5))
  g <- zl_graph(
    function(x) {
      zl_tanh(zl_crossprod(zl_param_tensor(p, "w"), x$x)) +
        zl_sum(zl_get_rows(zl_param_tensor(p, "e"), x$index))
    },
    p,
    inputs = list(x = c(3, 4), index = zl_int(2))
  )
  ns <- zulm_client()
  index <- c(4L, 1L)
  got <- ns$client_sampler(g, "eps", "x", "output", 8, steps = 3, index = index)
  want <- vapply(
    1:3,
    function(s) {
      x <- zl_rng_normal(c(3, 4), "eps", step = s)
      as.vector(zl_forward(g, list(x = x, index = index))$output)
    },
    numeric(8)
  )
  expect_identical(got, want)
})

test_that("draws on worker threads equal zl_rng_uniform()", {
  skip_on_cran()
  old <- zl_seed(9)
  withr::defer(zl_seed(old))
  ns <- zulm_client()
  got <- ns$client_draw_threads("workers", step = 3, n = 250, threads = 4)
  expect_identical(got, zl_rng_uniform(1000, "workers", step = 3))
})

test_that("a new session key needs the streams registered again", {
  skip_on_cran()
  old <- zl_seed(1)
  withr::defer(zl_seed(old))
  ns <- zulm_client()
  first <- ns$client_draw_threads("again", step = 1, n = 10, threads = 1)
  expect_identical(
    ns$client_draw_threads("again", 1, 10, 1, register = FALSE),
    first
  )
  zl_seed(2)
  # ZULM_EUNKNOWN: the key the stream was registered under is gone
  expect_identical(
    ns$client_draw_threads("again", 1, 10, 1, register = FALSE),
    1L
  )
  expect_identical(
    ns$client_draw_threads("again", 1, 10, 1),
    zl_rng_uniform(10, "again", step = 1)
  )
})

test_that("weights averaged in C equal the average in R", {
  skip_on_cran()
  withr::local_seed(3)
  a <- zl_params()
  b <- zl_params()
  zl_param(a, "w", c(3, 2), init = zl_init_normal(sd = 1))
  zl_param(b, "w", c(3, 2), init = zl_init_normal(sd = 1))
  wa <- zl_param_get(a, "w")
  wb <- zl_param_get(b, "w")
  ns <- zulm_client()
  ns$client_average(a, b, "w", 6)
  expect_identical(zl_param_get(a, "w"), array(f32((wa + wb) / 2), dim(wa)))
  expect_identical(zl_param_get(b, "w"), wb)
})

test_that("a state write in C is the state zl_state_get() reads", {
  skip_on_cran()
  p <- zl_params()
  zl_state(p, "h", c(2, 4))
  ns <- zulm_client()
  got <- ns$client_state(p, "h", offset = 3, values = c(1.5, -2))
  expect_identical(got, c(0, 0, 0, 1.5, -2, 0, 0, 0))
  expect_identical(as.vector(zl_state_get(p, "h")), got)
})

test_that("batch_fill() runs a graph over an Arrow stream as zl_forward() does", {
  skip_on_cran()
  stream <- function(counts) {
    zlm_call("zlm_arrow_test_stream", 5L, 3L, counts)
  }
  p <- zl_params()
  g <- zl_graph(function(x) 2 * x$x, p, inputs = list(x = c(1, 4)))
  ns <- zulm_client()
  counts <- integer(2)
  got <- ns$client_batches(
    g,
    zl_arrow_batches(stream(counts), list(x = "x"), rows = 4),
    "output",
    4
  )
  it <- zl_arrow_batches(stream(integer(2)), list(x = "x"), rows = 4)
  want <- list()
  while (!is.null(out <- zl_forward(g, it))) {
    want[[length(want) + 1L]] <- as.vector(out$output)
  }
  expect_length(got, 3L)
  expect_identical(got, want)
  expect_identical(counts, c(1L, 5L))
})

test_that("an Arrow error in batch_fill() is a code and a message, not a jump", {
  skip_on_cran()
  p <- zl_params()
  g <- zl_graph(function(x) 2 * x$y, p, inputs = list(y = c(1, 4)))
  it <- zl_arrow_batches(
    zlm_call("zlm_arrow_test_stream", 2L, 4L, integer(2)),
    list(x = "x"),
    rows = 4
  )
  ns <- zulm_client()
  expect_error(ns$client_batches(g, it, "output", 4), class = "simpleError")
})

test_that("failures are return codes", {
  skip_on_cran()
  p <- zl_params()
  zl_param(p, "w", c(3, 2))
  zl_state(p, "h", c(2, 2))
  g <- zl_graph(function(x) x$x + 1, p, inputs = list(x = c(2, 2)))
  ns <- zulm_client()
  # see client_codes() for the calls
  expect_identical(
    ns$client_codes(g, p),
    c(1L, 2L, 2L, 3L, 7L, 1L, 1L, 2L, 1L, 1L, 3L, 1L, 7L, 7L, 7L, 7L)
  )
})

test_that("an index out of range stops compute() with a code", {
  skip_on_cran()
  p <- zl_params()
  zl_param(p, "e", c(2, 5))
  g <- zl_graph(
    function(x) x$x + zl_sum(zl_get_rows(zl_param_tensor(p, "e"), x$index)),
    p,
    inputs = list(x = 2, index = zl_int(1))
  )
  ns <- zulm_client()
  expect_error(
    ns$client_sampler(g, "s", "x", "output", 2, steps = 1, index = 7L),
    class = "simpleError"
  )
  expect_no_error(ns$client_sampler(
    g,
    "s",
    "x",
    "output",
    2,
    steps = 1,
    index = 4L
  ))
})

test_that("a graph restored from a saved session is a stale handle", {
  skip_on_cran()
  p <- zl_params()
  g <- zl_graph(function(x) x$x + 1, p, inputs = list(x = 2))
  stale <- unserialize(serialize(g, NULL))
  ns <- zulm_client()
  expect_error(
    ns$client_sampler(stale, "s", "x", "output", 2, steps = 1),
    class = "simpleError"
  )
})

test_that("zulm.h compiles on its own as C99 and as C++17", {
  skip_on_cran()
  inc <- system.file("include", package = "zulm")
  r <- file.path(R.home("bin"), "R")
  flags <- c(
    system2(r, c("CMD", "config", "--cppflags"), stdout = TRUE),
    paste0("-I", shQuote(normalizePath(inc, "/")))
  )
  for (lang in c("c", "c++")) {
    src <- withr::local_tempfile(fileext = if (lang == "c") ".c" else ".cpp")
    writeLines(
      c(
        "#include <zulm.h>",
        "const zulm_api *f(void) { return zulm_api_get(); }"
      ),
      src
    )
    # the compiler may come with arguments, as "clang -arch arm64"
    compiler <- strsplit(
      system2(
        r,
        c("CMD", "config", if (lang == "c") "CC" else "CXX17"),
        stdout = TRUE
      ),
      " +"
    )[[1]]
    std <- if (lang == "c") "-std=c99" else "-std=c++17"
    out <- suppressWarnings(system2(
      compiler[[1]],
      c(
        compiler[-1],
        std,
        "-Wall",
        "-Werror",
        "-fsyntax-only",
        flags,
        shQuote(src)
      ),
      stdout = TRUE,
      stderr = TRUE
    ))
    expect_null(
      attr(out, "status"),
      label = paste(lang, "compile:", paste(out, collapse = "\n"))
    )
  }
})
