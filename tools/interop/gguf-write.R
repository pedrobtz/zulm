# Writes GGUF files with zulm for tools/interop/gguf-check.py (design §11,
# Stage 2): one per storage type, each holding tensors whose element i (in
# R's order, first index fastest) is i / 8 - 3, exact in f16 and bf16.
#
#   Rscript tools/interop/gguf-write.R <dir>
library(zulm)
dir <- commandArgs(TRUE)[[1L]]
shapes <- list(v = 7, m = c(3, 4), t = c(2, 3, 4), q = c(2, 1, 3, 2))
for (type in c("f32", "f16", "bf16")) {
  p <- zl_params()
  for (n in names(shapes)) {
    d <- shapes[[n]]
    zl_param(p, n, d, init = zl_init_zeros())
    x <- seq(0, prod(d) - 1) / 8 - 3
    zl_param_set(p, n, if (length(d) > 1L) array(x, d) else x)
  }
  zl_save(p, file.path(dir, sprintf("zulm-%s.gguf", type)), type = type,
          metadata = list(source = "zulm", width = 64L, ratio = 0.5, ok = TRUE))
}
cat("wrote", length(shapes), "tensors in three files to", dir, "\n")
