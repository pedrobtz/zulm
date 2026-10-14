# Design §9.2: no diagnostic pragma may come back with a re-vendoring. The
# source tree is only reachable from a development checkout; CI also runs
# tools/verify-ggml on every pull request.

ggml_tree <- function() {
  src <- testthat::test_path("..", "..", "src", "ggml")
  if (dir.exists(src)) normalizePath(src) else ""
}

test_that("the vendored tree carries no diagnostic pragma", {
  tree <- ggml_tree()
  skip_if(!nzchar(tree), "source tree not available")
  files <- list.files(tree, pattern = "\\.(c|cpp|h)$", recursive = TRUE, full.names = TRUE)
  expect_gt(length(files), 40L)
  hits <- unlist(lapply(files, function(f) {
    grep("#[[:space:]]*pragma[[:space:]]+(GCC|clang)[[:space:]]+diagnostic", readLines(f, warn = FALSE), value = TRUE)
  }))
  expect_length(hits, 0L)
})

test_that("ggml never qualifies the stdio names zlm_compat.h redirects", {
  tree <- ggml_tree()
  skip_if(!nzchar(tree), "source tree not available")
  files <- list.files(tree, pattern = "\\.(c|cpp|h)$", recursive = TRUE, full.names = TRUE)
  pattern <- "std::(printf|vprintf|fprintf|vfprintf|sprintf|fputs|puts|fputc|putc|putchar|fflush|fwrite|abort|_Exit)\\b"
  hits <- unlist(lapply(files, function(f) grep(pattern, readLines(f, warn = FALSE), value = TRUE, perl = TRUE)))
  expect_length(hits, 0L)
})
