# Reads the files tools/interop/gguf-check.py wrote and checks every value,
# shape and the metadata (design §11, Stage 2).
#
#   Rscript tools/interop/gguf-read.R <dir>
library(zulm)
dir <- commandArgs(TRUE)[[1L]]
shapes <- list(v = 7, m = c(3, 4), t = c(2, 3, 4), q = c(2, 1, 3, 2))
for (type in c("f32", "f16", "bf16")) {
  p <- zl_load(file.path(dir, sprintf("py-%s.gguf", type)))
  stopifnot(identical(zl_param_names(p), names(shapes)))
  for (n in names(shapes)) {
    d <- shapes[[n]]
    x <- seq(0, prod(d) - 1) / 8 - 3
    want <- if (length(d) > 1L) array(x, d) else x
    got <- zl_param_get(p, n)
    # GGUF keeps no trailing unit dimensions, so q = c(2, 1, 3, 2) keeps
    # its rank and a written c(3, 1) would come back as c(3).
    stopifnot(identical(as.vector(got), as.vector(want)))
    stopifnot(identical(length(got), length(want)))
    if (length(d) > 1L) stopifnot(identical(dim(got), as.integer(d)))
  }
  meta <- attr(p, "metadata")
  stopifnot(identical(meta$source, "gguf-py"), identical(meta$width, 64))
  cat("ok  py-", type, ".gguf read by zl_load()\n", sep = "")
}
