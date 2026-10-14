#' Save and load parameters as GGUF
#'
#' `zl_save()` writes every parameter of a store, and optional metadata,
#' to a GGUF file, the format of the 'ggml' library, which other tools
#' (llama.cpp, the 'gguf' Python package) read. `zl_load()` reads one into
#' a new store, with every tensor in single precision unless `keep_types`
#' is `TRUE`.
#'
#' Only parameters are saved: the state of an optimiser is not, so
#' training resumed from a loaded store starts with fresh optimiser state.
#'
#' Tensor dimensions follow R's order: a GGUF tensor of `ne = {d1, d2}`
#' is an R matrix of `dim = c(d1, d2)`, which NumPy reads as an array of
#' shape `(d2, d1)` holding the same bytes.
#'
#' @param p A parameter store.
#' @param path The file to write or read.
#' @param keep_types For `zl_load()`: `FALSE` widens every tensor to f32;
#'   `TRUE` keeps f16 and bf16 tensors in their type in memory (see
#'   [zl_params()] for the cost in precision).
#' @param type The type tensors are written as: `"f32"`, `"f16"` or
#'   `"bf16"`. Half-precision types halve the file and round each value.
#' @param metadata A named list of key-value pairs to store with the
#'   tensors: each value a character, integer, double or logical vector
#'   without `NA`. Keys starting with `zulm.` are reserved.
#' @return `zl_save()` returns `path` invisibly. `zl_load()` returns a
#'   parameter store whose `"metadata"` attribute is the file's metadata as
#'   a named list.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(3, 2))
#' path <- tempfile(fileext = ".gguf")
#' zl_save(p, path, metadata = list(model = "example", width = 3L))
#' q <- zl_load(path)
#' all.equal(zl_param_get(q, "w"), zl_param_get(p, "w"))
#' attr(q, "metadata")$model
#' unlink(path)
zl_save <- function(p, path, type = c("f32", "f16", "bf16"), metadata = list()) {
  zlm_check_params(p)
  zlm_check_string(path, "path")
  type <- zlm_match_type(type)
  metadata <- zlm_check_metadata(metadata)
  zlm_call(
    "zlm_store_save", p$ptr, enc2utf8(path.expand(path)), type, metadata,
    zlm_max_bytes()
  )
  invisible(path)
}

#' @rdname zl_save
#' @export
zl_load <- function(path, keep_types = FALSE) {
  zlm_check_string(path, "path")
  if (!isTRUE(keep_types) && !isFALSE(keep_types)) {
    zlm_abort("`keep_types` must be TRUE or FALSE.", "zulm_invalid_argument")
  }
  full <- path.expand(path)
  if (!file.exists(full) || dir.exists(full)) {
    zlm_abort(sprintf("There is no file '%s'.", path), "zulm_file_error")
  }
  res <- zlm_call("zlm_store_load", enc2utf8(full), zlm_max_bytes(), keep_types)
  zlm_new_params(res[[1L]], metadata = res[[2L]])
}

zlm_check_metadata <- function(metadata) {
  if (!is.list(metadata) || (length(metadata) && is.null(names(metadata)))) {
    zlm_abort("`metadata` must be a named list.", "zulm_invalid_argument")
  }
  if (length(metadata) == 0L) {
    return(structure(list(), names = character()))
  }
  keys <- names(metadata)
  if (anyNA(keys) || any(!nzchar(keys)) || anyDuplicated(keys)) {
    zlm_abort("`metadata` needs unique, non-empty names.", "zulm_invalid_argument")
  }
  if (any(startsWith(keys, "zulm."))) {
    zlm_abort("Metadata keys starting with 'zulm.' are reserved.", "zulm_invalid_argument")
  }
  for (k in keys) {
    v <- metadata[[k]]
    ok <- (is.character(v) || is.integer(v) || is.double(v) || is.logical(v)) &&
      is.null(dim(v)) && length(v) >= 1L && !anyNA(v)
    if (!ok) {
      zlm_abort(
        sprintf("Metadata '%s' must be a character, integer, double or logical vector without NA.", k),
        "zulm_invalid_argument"
      )
    }
    if (is.character(v)) {
      metadata[[k]] <- enc2utf8(v)
    }
    attributes(metadata[[k]]) <- NULL
  }
  names(metadata) <- enc2utf8(keys)
  metadata
}
