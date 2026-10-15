# Chooses the smoke set, tests/testthat/_smoke.txt: the cheapest tests that
# together make every kind of call into zulm's C code (routine and argument
# types) the whole suite makes. CI's quick runs of slow instruments run
# only these (ZULM_TESTS=smoke, tests/testthat/helper-smoke.R).
#
#   Rscript tools/smoke-tests.R
#
# Run it from the package root after adding tests or C entry points; it
# installs nothing and runs the suite once, recording calls and times.
trace <- tempfile(fileext = ".tsv")
status <- system2(
  file.path(R.home("bin"), "Rscript"),
  c("-e", shQuote("testthat::test_local(reporter = 'silent', stop_on_failure = TRUE)")),
  env = c("ZULM_TESTS=trace", "ZULM_GCTORTURE=0", paste0("ZULM_TESTS_TRACE=", trace))
)
if (status != 0L) stop("the traced run of the suite failed")
rows <- strsplit(readLines(trace), "\t", fixed = TRUE)
desc <- vapply(rows, `[[`, "", 1L)
time <- as.numeric(vapply(rows, `[[`, "", 2L))
calls <- lapply(rows, function(r) if (length(r) >= 3L && nzchar(r[[3L]])) strsplit(r[[3L]], ";", fixed = TRUE)[[1L]] else character())
all_calls <- unique(unlist(calls))

# Greedy weighted set cover: the test with the most new kinds of call per
# second, until every kind is covered.
covered <- character()
chosen <- integer()
while (length(setdiff(all_calls, covered)) > 0L) {
  gain <- vapply(calls, function(k) length(setdiff(k, covered)), numeric(1))
  score <- gain / (time + 0.005)
  score[gain == 0] <- -Inf
  i <- which.max(score)
  chosen <- c(chosen, i)
  covered <- union(covered, calls[[i]])
}
chosen <- sort(chosen)
writeLines(
  c(
    "# Written by tools/smoke-tests.R; do not edit. One test description",
    sprintf("# per line: %d of %d tests, %.1f of %.1f seconds, all %d kinds of call.", length(chosen), length(desc), sum(time[chosen]), sum(time), length(all_calls)),
    desc[chosen]
  ),
  "tests/testthat/_smoke.txt"
)
message(sprintf(
  "%d of %d tests cover all %d kinds of call in %.1f of %.1f seconds",
  length(chosen), length(desc), length(all_calls), sum(time[chosen]), sum(time)
))
