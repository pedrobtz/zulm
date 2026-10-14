# Design §12 and §13.1 measurements: the 512^3 f32 product on every usable
# variant of the installed zulm, at one, two and four threads, with the hash
# of the result's bits.
#
#   Rscript tools/bench-variants.R <label> [n] [reps]
args <- commandArgs(TRUE)
label <- if (length(args) >= 1L) args[[1L]] else "default"
n <- if (length(args) >= 2L) as.integer(args[[2L]]) else 512L
reps <- if (length(args) >= 3L) as.integer(args[[3L]]) else 20L
library(zulm)
info <- zulm_info()
rows <- list()
for (v in info$variants_usable) {
  zulm:::zlm_load_variant(v)
  for (t in c(1L, 2L, 4L)) {
    best <- Inf
    hash <- NA_real_
    for (k in 1:3) {
      b <- zulm:::zlm_call("zlm_bench_mul_mat", n, reps, t)
      best <- min(best, b[[1L]])
      hash <- b[[3L]]
    }
    rows[[length(rows) + 1L]] <- data.frame(
      build = label, variant = v, threads = t,
      gflops = round(2 * n^3 / best / 1e9, 2),
      hash = sprintf("%08x", as.integer(hash %% 2^31)),
      llamafile = isTRUE(zulm_info()$cpu$llamafile)
    )
  }
}
out <- do.call(rbind, rows)
print(out, row.names = FALSE)
f <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(f)) {
  md <- c(
    sprintf("### %s (%s)", label, info$compiler),
    "",
    "| variant | threads | GFLOP/s | hash | llamafile |",
    "| --- | --- | --- | --- | --- |",
    sprintf("| %s | %d | %.2f | %s | %s |", out$variant, out$threads, out$gflops, out$hash, out$llamafile),
    ""
  )
  cat(md, file = f, sep = "\n", append = TRUE)
}
