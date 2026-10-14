# Conditions (design §10). Every error inherits zulm_error, every warning
# zulm_warning. Stage 2 completes the hierarchy.

zlm_abort <- function(message, class, ..., call = NULL) {
  stop(structure(
    class = c(class, "zulm_error", "error", "condition"),
    list(message = message, call = call, ...)
  ))
}

# Runs a ggml construction known to fail an assertion (design §13.2) and
# signals the caught assertion as zulm_engine_error. `valid = TRUE` runs a
# valid construction instead and returns invisibly.
zlm_engine_check <- function(valid = FALSE) {
  res <- zlm_call("zlm_engine_selftest", valid)
  zlm_check_status(res[[1L]], res[[2L]])
  invisible(TRUE)
}

zlm_check_status <- function(status, message) {
  switch(
    as.character(status),
    "0" = invisible(NULL),
    "1" = zlm_abort(
      paste0("ggml rejected the graph construction: ", message),
      "zulm_engine_error",
      ggml_message = message
    ),
    "2" = zlm_abort(
      "ggml ran out of memory during graph construction.",
      "zulm_memory_error"
    ),
    zlm_abort(
      paste0("Internal error in zulm's C++ shim: ", message),
      "zulm_internal_error"
    )
  )
}
