#' Threads
#'
#' `zl_threads()` sets how many threads 'ggml' computes with, for every
#' graph, trainer and gradient of the session, and returns the previous
#' number; with no argument it returns the current one. It starts at 1. A
#' request is capped at the number of available cores and, while
#' `R CMD check` limits cores (`_R_CHECK_LIMIT_CORES_`), at 2: too many
#' threads on too few cores make 'ggml' many times slower.
#'
#' One pool of threads serves the session; it is recreated when the number
#' changes, and in a process forked from this one. Results are identical at
#' any number of threads (see the package documentation on determinism).
#' A computation can be interrupted between two of its operations; it then
#' signals a condition of class `zulm_interrupt`, which also inherits
#' `interrupt`.
#'
#' @param n `NULL`, or the number of threads: a positive whole number.
#' @return The number of threads; when setting, the previous number,
#'   invisibly.
#' @export
#' @examples
#' old <- zl_threads(2)
#' zl_threads()
#' zl_threads(old)
zl_threads <- function(n = NULL) {
  if (is.null(n)) {
    return(zlm_call("zlm_threads_", NULL))
  }
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1 || n != floor(n)) {
    zlm_abort("`n` must be a positive whole number.", "zulm_invalid_argument")
  }
  cap <- zlm_call("zlm_cores")
  lim <- Sys.getenv("_R_CHECK_LIMIT_CORES_", "")
  if (nzchar(lim) && !identical(tolower(lim), "false")) {
    cap <- min(cap, 2L)
  }
  invisible(zlm_call("zlm_threads_", as.integer(min(n, cap, 512L))))
}

# The session's number of threads.
zlm_threads <- function() zlm_call("zlm_threads_", NULL)
