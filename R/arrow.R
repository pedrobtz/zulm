# Training batches from Arrow streams (design §7.3).

#' Batches from Arrow streams
#'
#' `zl_arrow_batches()` reads record batches from an Arrow stream, such as
#' a 'DuckDB' query, a Parquet file or an 'arrow' table, and cuts them into
#' batches of exactly `rows` rows for [zl_step()] and [zl_forward()], which
#' take the iterator as `data`. `zl_next()` returns the next batch as R
#' arrays, for inspection. Each batch is one copy of the data: columns are
#' interleaved into feature-major arrays, `c(features, rows)`.
#'
#' `map` names the graph's inputs and gives each its columns: a character
#' vector for an f32 input of `c(length(columns), rows)`, or `zl_int(...)`
#' around the column names for an i32 input (a vector of `rows` for one
#' column). float32 columns are copied, float64 rounded to f32, int8 to
#' int64 and boolean (0 and 1) go to either kind of input, and
#' dictionary-encoded columns of strings go to i32 inputs as their codes,
#' counting from 0. int64 values beyond the 32-bit range are errors in an
#' i32 input. The dictionary must be the same in every record batch.
#' Other types are refused.
#'
#' Missing values are errors naming the column and row (`na = "error"`).
#' With `na = "mask"` they are written as 0, and every input `x` gets a
#' companion input `x_mask` of its shape holding 1 for a value and 0 for a
#' missing one. The rows left at the end of the stream are dropped
#' (`last = "drop"`), or padded with zeros to a full batch (`last =
#' "pad"`), in which case every batch carries a `row_mask` input of
#' `rows` values, 1 for a real row and 0 for padding. A graph that takes
#' the iterator declares these inputs too.
#'
#' The stream is taken over by the iterator: it is read on the main thread
#' only, as producers such as 'DuckDB' require, and released when it ends
#' or when the iterator is garbage collected. `zl_step()` and
#' `zl_forward()` return `NULL` once it has ended.
#'
#' A `nanoarrow_array_stream` is used directly; anything else is converted
#' with `nanoarrow::as_nanoarrow_array_stream()`, which needs the
#' 'nanoarrow' package (a `zulm_companion_error` without it).
#'
#' @param stream An Arrow stream: a `nanoarrow_array_stream`, or an object
#'   that 'nanoarrow' converts to one (a data frame, an 'arrow' table or
#'   reader).
#' @param map A named list from input names to column names, as above.
#' @param rows The rows per batch.
#' @param na What a missing value does: `"error"` or `"mask"`.
#' @param last What the final, partial batch does: `"drop"` or `"pad"`.
#' @param it An iterator from `zl_arrow_batches()`.
#' @param x An iterator, for `print()`.
#' @param ... Unused.
#' @return `zl_arrow_batches()` returns an iterator of class
#'   `zl_arrow_batches`; `zl_next()` a named list of arrays, or `NULL` at
#'   the end of the stream.
#' @export
#' @examples
#' if (requireNamespace("nanoarrow", quietly = TRUE)) {
#'   df <- data.frame(a = c(1, 2, 3, 4, 5), b = c(0.5, 1, 1.5, 2, 2.5),
#'                    kind = factor(c("x", "y", "x", "x", "y")))
#'   it <- zl_arrow_batches(df, list(x = c("a", "b"), kind = zl_int("kind")), rows = 2)
#'   zl_next(it)
#'   zl_next(it)
#'   zl_next(it)
#' }
zl_arrow_batches <- function(stream, map, rows, na = c("error", "mask"), last = c("drop", "pad")) {
  na <- match.arg(na)
  last <- match.arg(last)
  if (!inherits(stream, "nanoarrow_array_stream")) {
    if (!zlm_has_nanoarrow()) {
      zlm_abort(
        "Reading this object as an Arrow stream needs the 'nanoarrow' package; install it, or pass a nanoarrow_array_stream.",
        "zulm_companion_error"
      )
    }
    stream <- nanoarrow::as_nanoarrow_array_stream(stream)
  }
  if (!is.list(map) || length(map) == 0L || is.null(names(map)) || anyNA(names(map)) ||
    any(!nzchar(names(map))) || anyDuplicated(names(map))) {
    zlm_abort("`map` must be a non-empty list with unique input names.", "zulm_invalid_argument")
  }
  outs <- c(names(map), if (na == "mask") paste0(names(map), "_mask"), if (last == "pad") "row_mask")
  if (anyDuplicated(outs)) {
    zlm_abort(
      sprintf("The input names would clash with the masks: %s.", paste(unique(outs[duplicated(outs)]), collapse = ", ")),
      "zulm_invalid_argument"
    )
  }
  if (any(nchar(outs, type = "bytes") > 63L)) {
    zlm_abort("Input names, with their `_mask` suffixes, have at most 63 bytes.", "zulm_invalid_argument")
  }
  is_int <- vapply(map, inherits, logical(1), "zl_int_cols")
  cols <- lapply(map, function(m) if (inherits(m, "zl_int_cols")) m$cols else m)
  for (i in seq_along(cols)) {
    cv <- cols[[i]]
    if (!is.character(cv) || length(cv) == 0L || anyNA(cv) || any(!nzchar(cv))) {
      zlm_abort(
        sprintf("`map$%s` must name one or more columns, or be zl_int() around them.", names(map)[[i]]),
        "zulm_invalid_argument"
      )
    }
  }
  if (!is.numeric(rows) || length(rows) != 1L || is.na(rows) || rows < 1 || rows != floor(rows)) {
    zlm_abort("`rows` must be a whole number from 1.", "zulm_invalid_argument")
  }
  if (max(lengths(cols)) * rows > .Machine$integer.max) {
    zlm_abort("A batch would hold more than 2^31 - 1 values per input.", "zulm_invalid_argument")
  }
  ptr <- zlm_call(
    "zlm_arrow_open", stream, enc2utf8(names(map)), is_int,
    lapply(cols, enc2utf8), as.double(rows), na == "mask", last == "pad"
  )
  structure(
    list(ptr = ptr, inputs = names(map), outputs = outs, rows = rows, na = na, last = last),
    class = "zl_arrow_batches"
  )
}

#' @rdname zl_arrow_batches
#' @export
zl_next <- function(it) {
  if (!inherits(it, "zl_arrow_batches")) {
    zlm_abort("`it` must be an iterator from zl_arrow_batches().", "zulm_invalid_argument")
  }
  zlm_call("zlm_arrow_next", it$ptr)
}

#' @rdname zl_arrow_batches
#' @export
print.zl_arrow_batches <- function(x, ...) {
  st <- tryCatch(zlm_call("zlm_arrow_state", x$ptr), zulm_error = function(e) NULL)
  if (is.null(st)) {
    cat("<zl_arrow_batches> (invalid)\n")
    return(invisible(x))
  }
  cat(sprintf(
    "<zl_arrow_batches> %s rows per batch; inputs %s; %s rows read%s\n",
    format(x$rows), paste(x$outputs, collapse = ", "), format(st[[1L]], big.mark = ","),
    if (isTRUE(st[[2L]])) ", ended" else ""
  ))
  invisible(x)
}

zlm_has_nanoarrow <- function() requireNamespace("nanoarrow", quietly = TRUE)
