#' Conditions signalled by zulm
#'
#' Every error zulm signals inherits `zulm_error`, and every warning
#' `zulm_warning`, so a handler can catch them all or one class at a time
#' with [tryCatch()] or [withCallingHandlers()].
#'
#' Errors:
#' * `zulm_invalid_argument`: an argument of the wrong kind or value.
#' * `zulm_shape_error`: dimensions that do not fit; carries `op`, the
#'   shapes and the R call where an operation is involved.
#' * `zulm_type_error`: a tensor or value type that is not accepted.
#' * `zulm_unsupported_backward`: an operation without a gradient on a
#'   gradient path; carries `op`.
#' * `zulm_memory_error`: an allocation beyond
#'   `getOption("zulm.max_bytes")` (4 GiB by default); carries `requested`
#'   and `allowed`, in bytes.
#' * `zulm_engine_error`: an assertion of the ggml library caught while a
#'   graph was being built; carries `ggml_message`. The graph is discarded.
#' * `zulm_interrupt`: a computation stopped by a user interrupt; also
#'   inherits `interrupt`.
#' * `zulm_file_error`: a GGUF or safetensors file that cannot be read or
#'   written.
#' * `zulm_arrow_error`: an Arrow stream with an unsupported type, a null,
#'   a dictionary or a stream failure; carries `column` and `row`.
#' * `zulm_companion_error`: a companion package whose interface lacks
#'   something zulm needs.
#' * `zulm_stale_error`: an object that is no longer valid, because it was
#'   restored from a saved R session or created by a variant no longer in
#'   use.
#' * `zulm_internal_error`: a bug in zulm.
#'
#' Warnings:
#' * `zulm_range_warning`: values beyond the range of the storage type,
#'   which became infinite.
#' * `zulm_nonfinite_warning`: a training step whose loss is not finite.
#'
#' @name zulm-conditions
#' @examples
#' p <- zl_params()
#' tryCatch(
#'   zl_param_get(p, "missing"),
#'   zulm_invalid_argument = function(e) conditionMessage(e)
#' )
NULL

zlm_error_classes <- c(
  "zulm_invalid_argument", "zulm_shape_error", "zulm_type_error",
  "zulm_unsupported_backward", "zulm_memory_error", "zulm_engine_error",
  "zulm_interrupt", "zulm_file_error", "zulm_arrow_error",
  "zulm_companion_error", "zulm_stale_error", "zulm_internal_error"
)

zlm_warning_classes <- c("zulm_range_warning", "zulm_nonfinite_warning")

zlm_abort <- function(message, class, ..., call = NULL) {
  extra <- if (identical(class, "zulm_interrupt")) "interrupt" else character()
  stop(structure(
    class = c(class, extra, "zulm_error", "error", "condition"),
    list(message = message, call = call, ...)
  ))
}

zlm_warn <- function(message, class, ..., call = NULL) {
  warning(structure(
    class = c(class, "zulm_warning", "warning", "condition"),
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

# Argument checks shared by the R API.
zlm_check_string <- function(x, what) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    zlm_abort(
      sprintf("`%s` must be a single non-empty string.", what),
      "zulm_invalid_argument"
    )
  }
  invisible(x)
}

zlm_check_number <- function(x, what, lower = -Inf, finite = TRUE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) ||
    (finite && !is.finite(x)) || x < lower) {
    zlm_abort(
      sprintf("`%s` must be a single number of at least %s.", what, format(lower)),
      "zulm_invalid_argument"
    )
  }
  invisible(as.double(x))
}

zlm_max_bytes <- function() {
  cap <- getOption("zulm.max_bytes", 4 * 2^30)
  if (!is.numeric(cap) || length(cap) != 1L || is.na(cap) || cap <= 0) {
    zlm_abort(
      "`options(zulm.max_bytes =)` must be a single positive number of bytes.",
      "zulm_invalid_argument"
    )
  }
  as.double(cap)
}
