#' safetensors files
#'
#' safetensors is the format PyTorch-trained weights travel in.
#' `zl_safetensors_read()` returns a file's tensors as R arrays;
#' `zl_load_safetensors()` fills a store's parameters from one, reading
#' each tensor straight into the store; `zl_save_safetensors()` writes a
#' store.
#'
#' Shapes are row-major in the file, so a tensor of `shape = c(s1, ..., sk)`
#' is an R array of `dim = c(sk, ..., s1)` holding the same bytes:
#' PyTorch's `nn.Linear` weight of shape `[out, in]` is `c(in, out)`, the
#' layout of [zl_linear()], with no transposition.
#'
#' Types: F32, F16 and BF16 values are read as single precision; F64 is
#' rounded to single precision (with a `zulm_range_warning` if a value
#' overflows); I32 and I64 are read as integers by `zl_safetensors_read()`.
#' A parameter declared `type = "f16"` or `"bf16"` keeps a matching tensor
#' as it is and rounds others to its type. Other types are refused.
#'
#' The header is checked before any data is read: its length, every
#' tensor's offsets (in range, in order, without overlap, covering the
#' data) and every size against its type and shape. A failure is
#' `zulm_file_error`.
#'
#' @param path The file.
#' @param names Tensors to read; `NULL` for all.
#' @param p A parameter store.
#' @param map `NULL`, or a named character vector from the file's tensor
#'   names (the names) to the store's parameter names (the values).
#'   Without one, tensors fill parameters of the same name.
#' @param type The type to write: `"f32"`, `"f16"` or `"bf16"`.
#' @param metadata A named list of strings, written as the header's
#'   `__metadata__`.
#' @return `zl_safetensors_read()`: a named list of arrays, with the file's
#'   metadata as the `"metadata"` attribute. `zl_load_safetensors()`: the
#'   names of the parameters it filled, invisibly. `zl_save_safetensors()`:
#'   `path`, invisibly.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "lin.weight", c(3, 2))
#' path <- tempfile(fileext = ".safetensors")
#' zl_save_safetensors(p, path, metadata = list(source = "example"))
#' str(zl_safetensors_read(path))
#' q <- zl_params()
#' zl_param(q, "w", c(3, 2), init = zl_init_zeros())
#' zl_load_safetensors(q, path, map = c(lin.weight = "w"))
#' identical(zl_param_get(q, "w"), zl_param_get(p, "lin.weight"))
#' unlink(path)
zl_safetensors_read <- function(path, names = NULL) {
  h <- zlm_st_parse(path)
  sel <- h$tensors
  if (!is.null(names)) {
    if (!is.character(names) || anyNA(names) || !all(names %in% sel$name)) {
      zlm_abort("`names` must name tensors of the file.", "zulm_invalid_argument")
    }
    sel <- sel[match(names, sel$name), , drop = FALSE]
  }
  bad <- !sel$dtype %in% c("F32", "F16", "BF16", "F64", "I32", "I64")
  if (any(bad)) {
    zlm_abort(sprintf("Tensor '%s' has type %s, which zulm does not read.", sel$name[bad][1], sel$dtype[bad][1]), "zulm_type_error")
  }
  vals <- zlm_call("zlm_st_read", h$path, h$data_start, sel$begin, sel$end, sel$dtype)
  for (i in seq_along(vals)) {
    d <- rev(sel$shape[[i]])
    if (length(d) > 1L) dim(vals[[i]]) <- d
  }
  names(vals) <- sel$name
  attr(vals, "metadata") <- h$metadata
  vals
}

#' @rdname zl_safetensors_read
#' @export
zl_load_safetensors <- function(p, path, map = NULL) {
  zlm_check_params(p)
  h <- zlm_st_parse(path)
  tn <- h$tensors$name
  if (is.null(map)) {
    target <- tn
  } else {
    if (!is.character(map) || is.null(names(map)) || anyNA(map) || anyDuplicated(names(map))) {
      zlm_abort("`map` must be a named character vector from file names to parameter names.", "zulm_invalid_argument")
    }
    target <- unname(map[tn])
  }
  info <- zlm_call("zlm_store_info", p$ptr)
  use <- which(!is.na(target) & target %in% info$names[!info$state])
  if (length(use) == 0L) {
    zlm_abort("No tensor of the file matches a parameter of the store.", "zulm_invalid_argument")
  }
  sel <- h$tensors[use, , drop = FALSE]
  params <- target[use]
  for (i in seq_along(params)) {
    want <- info$dims[[match(params[[i]], info$names)]]
    have <- rev(sel$shape[[i]])
    trim <- function(d) { while (length(d) > 1L && d[length(d)] == 1) d <- d[-length(d)]; d }
    if (!identical(as.double(trim(have)), as.double(trim(want)))) {
      zlm_abort(
        sprintf("Tensor '%s' has dimensions %s; parameter '%s' has %s.", sel$name[[i]],
                zlm_format_dim(if (length(have)) have else 1), params[[i]], zlm_format_dim(want)),
        "zulm_shape_error"
      )
    }
    if (!sel$dtype[[i]] %in% c("F32", "F16", "BF16", "F64")) {
      zlm_abort(sprintf("Tensor '%s' has type %s, which cannot fill a parameter.", sel$name[[i]], sel$dtype[[i]]), "zulm_type_error")
    }
  }
  zlm_call("zlm_st_fill", p$ptr, h$path, h$data_start, enc2utf8(params), sel$begin, sel$end, sel$dtype)
  invisible(params)
}

#' @rdname zl_safetensors_read
#' @export
zl_save_safetensors <- function(p, path, type = c("f32", "f16", "bf16"), metadata = list()) {
  zlm_check_params(p)
  zlm_check_string(path, "path")
  type <- zlm_match_type(type)
  if (!is.list(metadata) || (length(metadata) && (is.null(names(metadata)) ||
    !all(vapply(metadata, function(v) is.character(v) && length(v) == 1L && !is.na(v), logical(1)))))) {
    zlm_abort("`metadata` must be a named list of single strings.", "zulm_invalid_argument")
  }
  info <- zlm_call("zlm_store_info", p$ptr)
  size <- c(f32 = 4, f16 = 2, bf16 = 2)[[type]]
  dtype <- c(f32 = "F32", f16 = "F16", bf16 = "BF16")[[type]]
  hdr <- list()
  if (length(metadata)) hdr[["__metadata__"]] <- lapply(metadata, enc2utf8)
  off <- 0
  for (i in which(!info$state)) {
    n <- prod(info$dims[[i]]) * size
    hdr[[info$names[[i]]]] <- list(dtype = dtype, shape = I(rev(info$dims[[i]])), data_offsets = I(c(off, off + n)))
    off <- off + n
  }
  if (length(hdr) == 0L) hdr <- structure(list(), names = character())
  json <- zujson::json_write(hdr)
  if (identical(json, "[]")) json <- "{}"
  # the data starts on an 8-byte boundary, as the format recommends
  pad <- (8 - (nchar(json, type = "bytes") + 8) %% 8) %% 8
  json <- paste0(json, strrep(" ", pad))
  zlm_call("zlm_st_write", p$ptr, enc2utf8(path.expand(path)), json, type)
  invisible(path)
}

# Reads and validates a safetensors header (design §7.4): a data frame of
# tensors in file order (name, dtype, shape, begin, end) and the metadata.
zlm_st_parse <- function(path) {
  zlm_check_string(path, "path")
  full <- path.expand(path)
  if (!file.exists(full) || dir.exists(full)) {
    zlm_abort(sprintf("There is no file '%s'.", path), "zulm_file_error")
  }
  raw <- zlm_call("zlm_st_header", enc2utf8(full))
  h <- zlm_st_check(raw[[1L]], raw[[3L]] - raw[[2L]], path)
  h$path <- enc2utf8(full)
  h$data_start <- raw[[2L]]
  h
}

zlm_st_sizes <- c(
  F64 = 8, I64 = 8, U64 = 8, F32 = 4, I32 = 4, U32 = 4, F16 = 2, BF16 = 2,
  I16 = 2, U16 = 2, I8 = 1, U8 = 1, BOOL = 1, F8_E4M3 = 1, F8_E5M2 = 1
)

# Validates a header's JSON against a data buffer of `data_size` bytes.
zlm_st_check <- function(json, data_size, path = "the file") {
  fail <- function(why) zlm_abort(sprintf("'%s' is not a valid safetensors file: %s", path, why), "zulm_file_error")
  if (!startsWith(json, "{")) fail("its header does not start with '{'.")
  h <- tryCatch(zujson::json_parse(json, simplify = TRUE), error = function(e) fail(conditionMessage(e)))
  if (!is.list(h) || (length(h) && is.null(names(h)))) fail("its header is not a JSON object.")
  if (anyDuplicated(names(h))) fail("its header repeats a key.")
  meta <- h[["__metadata__"]]
  h[["__metadata__"]] <- NULL
  if (!is.null(meta)) {
    if (!is.list(meta) || !all(vapply(meta, function(v) is.character(v) && length(v) == 1L, logical(1)))) {
      fail("its __metadata__ is not a map of strings.")
    }
  }
  n <- length(h)
  name <- names(h)
  dtype <- character(n)
  shape <- vector("list", n)
  begin <- end <- numeric(n)
  for (i in seq_len(n)) {
    e <- h[[i]]
    if (!is.list(e) || !setequal(names(e), c("dtype", "shape", "data_offsets"))) {
      fail(sprintf("tensor '%s' needs exactly dtype, shape and data_offsets.", name[[i]]))
    }
    if (!is.character(e$dtype) || length(e$dtype) != 1L || !e$dtype %in% names(zlm_st_sizes)) {
      fail(sprintf("tensor '%s' has an unknown dtype.", name[[i]]))
    }
    s <- e$shape
    if (is.logical(s) && length(s) == 0L) s <- numeric()
    if (!is.numeric(s) || anyNA(s) || any(s < 0) || any(s != floor(s))) {
      fail(sprintf("tensor '%s' has an invalid shape.", name[[i]]))
    }
    o <- e$data_offsets
    if (!is.numeric(o) || length(o) != 2L || anyNA(o) || any(o < 0) || any(o != floor(o)) || o[2] < o[1]) {
      fail(sprintf("tensor '%s' has invalid data_offsets.", name[[i]]))
    }
    if (prod(s) * zlm_st_sizes[[e$dtype]] != o[2] - o[1]) {
      fail(sprintf("tensor '%s' has %s bytes, not what its type and shape need.", name[[i]], format(o[2] - o[1])))
    }
    dtype[[i]] <- e$dtype
    shape[[i]] <- as.double(s)
    begin[[i]] <- o[1]
    end[[i]] <- o[2]
  }
  ord <- order(begin, end)
  if (n > 0L) {
    b <- begin[ord]
    e <- end[ord]
    if (b[1] != 0 || any(b[-1] != e[-n]) || e[n] != data_size) {
      fail("its tensors' offsets overlap, leave gaps or do not cover the data.")
    }
  } else if (data_size != 0) {
    fail("it holds data but no tensor.")
  }
  tensors <- data.frame(name = name, dtype = dtype, begin = begin, end = end, stringsAsFactors = FALSE)
  tensors$shape <- shape
  list(tensors = tensors[ord, , drop = FALSE], metadata = meta)
}
