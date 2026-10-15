# The smoke set, for CI's quick runs of slow instruments (sanitizers,
# gctorture): ZULM_TESTS=smoke runs only the tests listed in _smoke.txt,
# the cheapest set that makes every kind of call into zulm's C code
# (routine and argument types). tools/smoke-tests.R chooses it from a run
# with ZULM_TESTS=trace, which records each test's time and calls to the
# file ZULM_TESTS_TRACE names. Unset, every test runs as usual.
if (Sys.getenv("ZULM_TESTS") %in% c("smoke", "trace")) {
  test_that <- function(desc, code) {
    run <- sys.call()
    run[[1L]] <- quote(testthat::test_that)
    if (Sys.getenv("ZULM_TESTS") == "smoke") {
      if (!desc %in% readLines(test_path("_smoke.txt"))) {
        return(invisible(TRUE))
      }
      return(eval(run, parent.frame()))
    }
    zlm <- getFromNamespace(".zlm", "zulm")
    zlm$torture$seen <- new.env(parent = emptyenv())
    t <- system.time(res <- eval(run, parent.frame()))[["elapsed"]]
    calls <- sort(ls(zlm$torture$seen))
    cat(sprintf("%s\t%.4f\t%s\n", desc, t, paste(calls, collapse = ";")),
      file = Sys.getenv("ZULM_TESTS_TRACE"), append = TRUE
    )
    invisible(res)
  }
}
