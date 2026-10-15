# Helpers of test-vendor.R.

ggml_tree <- function() {
  src <- testthat::test_path("..", "..", "src", "ggml")
  if (dir.exists(src)) normalizePath(src) else ""
}
