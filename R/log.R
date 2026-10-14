# ggml's log (design §6.8). The C side buffers ggml's log records and its
# writes to stdout and stderr, from any thread; after each native call the
# R side turns the buffered lines at or above options(zulm.log =) into
# messages of class zulm_message. Levels: "debug", "info", "warn" (the
# default), "error", "none".

zlm_log_levels <- c(debug = 1L, info = 2L, warn = 3L, error = 4L, none = 99L)

zlm_log_threshold <- function() {
  lv <- getOption("zulm.log", "warn")
  if (!is.character(lv) || length(lv) != 1L || !lv %in% names(zlm_log_levels)) {
    return(zlm_log_levels[["warn"]])
  }
  zlm_log_levels[[lv]]
}

zlm_flush_log <- function() {
  routines <- .zlm$routines
  if (is.null(routines$zlm_log_pending_) || !zlm_dot_call(routines$zlm_log_pending_)) {
    return(invisible(NULL))
  }
  rec <- zlm_dot_call(routines$zlm_log_drain)
  levels <- rec[[1L]]
  texts <- rec[[2L]]
  if (isTRUE(attr(rec, "dropped"))) {
    levels <- c(levels, 3L)
    texts <- c(texts, "zulm: ggml's log overflowed its buffer; some lines were dropped.\n")
  }
  keep <- levels >= zlm_log_threshold()
  if (!any(keep)) {
    return(invisible(NULL))
  }
  lines <- unlist(strsplit(paste(texts[keep], collapse = ""), "\n", fixed = TRUE))
  for (line in lines[nzchar(trimws(lines))]) {
    message(structure(
      class = c("zulm_message", "message", "condition"),
      list(message = paste0(line, "\n"), call = NULL)
    ))
  }
  invisible(NULL)
}
