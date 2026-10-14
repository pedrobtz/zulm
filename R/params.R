#' Parameter stores
#'
#' A parameter store holds named tensors of up to four dimensions. Its
#' memory belongs to the 'ggml' library, outside R's heap, and is freed
#' when the store is garbage collected. Stores cannot be saved with
#' [saveRDS()]: a restored store is invalid (`zulm_stale_error`); use
#' [zl_save()] and [zl_load()].
#'
#' `zl_param()` adds a parameter. Its values are stored as `type`: `"f32"`
#' (single precision, the default), or `"f16"` and `"bf16"` (half
#' precision, for weights that are only read). R doubles are rounded to
#' the nearest representable value; values beyond the type's range become
#' infinite with a `zulm_range_warning`.
#'
#' An R array of `dim = c(d1, d2, d3, d4)` is stored with the first index
#' fastest, as R stores it, so values move without reordering.
#'
#' @param p A parameter store from `zl_params()` or [zl_load()].
#' @param name The parameter's name: a string of 1 to 63 bytes, unique in
#'   the store.
#' @param dim Its dimensions: one to four positive whole numbers.
#' @param init How to fill it: see [zl_init_normal()].
#' @param type The storage type: `"f32"`, `"f16"` or `"bf16"`.
#' @param value An array of `dim(value)` equal to the parameter's
#'   dimensions, or a vector of the right length for a one-dimensional
#'   parameter. `NA` is refused.
#' @return `zl_params()` returns an empty store. `zl_param()` and
#'   `zl_param_set()` return the store invisibly. `zl_param_get()` returns
#'   a double vector or array; `zl_param_names()` the names in the order
#'   added; `zl_param_count()` the total number of values;
#'   `zl_param_tensor()` a handle for use when building a graph.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(3, 2))
#' zl_param(p, "b", 2, init = zl_init_zeros())
#' zl_param_names(p)
#' zl_param_count(p)
#' zl_param_set(p, "b", c(0.5, -0.5))
#' zl_param_get(p, "b")
zl_params <- function() {
  zlm_new_params(zlm_call("zlm_store_new"))
}

zlm_new_params <- function(ptr, metadata = NULL) {
  p <- structure(list(ptr = ptr, variant = .zlm$variant), class = "zl_params")
  if (!is.null(metadata)) {
    attr(p, "metadata") <- metadata
  }
  p
}

zlm_check_params <- function(p) {
  if (!inherits(p, "zl_params")) {
    zlm_abort("`p` must be a parameter store from zl_params() or zl_load().", "zulm_invalid_argument")
  }
  if (!identical(p$variant, .zlm$variant)) {
    zlm_abort(
      sprintf(
        "This store was made by zulm's '%s' variant; the '%s' variant is loaded now.",
        p$variant, .zlm$variant
      ),
      "zulm_stale_error"
    )
  }
  invisible(p)
}

#' @rdname zl_params
#' @export
zl_param <- function(p, name, dim, init = zl_init_normal(sd = 0.02), type = c("f32", "f16", "bf16")) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  dim <- zlm_check_dim(dim)
  if (!inherits(init, "zl_init")) {
    zlm_abort("`init` must come from zl_init_normal() or another zl_init_*() function.", "zulm_invalid_argument")
  }
  type <- zlm_match_type(type)
  values <- zlm_init_values(init, name, prod(dim))
  zlm_call("zlm_store_add", p$ptr, enc2utf8(name), dim, type, values, zlm_max_bytes())
  invisible(p)
}

zlm_check_dim <- function(dim) {
  if (!is.numeric(dim) || length(dim) < 1L || length(dim) > 4L ||
    anyNA(dim) || any(!is.finite(dim)) || any(dim != floor(dim)) || any(dim < 1)) {
    zlm_abort(
      "`dim` must hold one to four positive whole numbers.",
      "zulm_shape_error"
    )
  }
  as.double(dim)
}

zlm_match_type <- function(type) {
  types <- c("f32", "f16", "bf16")
  if (identical(type, types)) {
    return("f32")
  }
  if (!is.character(type) || length(type) != 1L || !type %in% types) {
    zlm_abort('`type` must be "f32", "f16" or "bf16".', "zulm_type_error")
  }
  type
}

#' @rdname zl_params
#' @export
zl_param_get <- function(p, name) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  info <- zlm_param_info(p, name)
  out <- zlm_call("zlm_store_get", p$ptr, enc2utf8(name))
  if (length(info$dim) > 1L) {
    dim(out) <- info$dim
  }
  out
}

#' @rdname zl_params
#' @export
zl_param_set <- function(p, name, value) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  info <- zlm_param_info(p, name)
  if (!is.numeric(value)) {
    zlm_abort("`value` must be numeric.", "zulm_type_error")
  }
  have <- dim(value)
  ok <- if (is.null(have)) {
    length(info$dim) == 1L
  } else {
    length(have) == length(info$dim) && all(have == info$dim)
  }
  if (!ok || length(value) != prod(info$dim)) {
    zlm_abort(
      sprintf(
        "`value` has dimensions %s; parameter '%s' has %s.",
        zlm_format_dim(if (is.null(have)) length(value) else have),
        name,
        zlm_format_dim(info$dim)
      ),
      "zulm_shape_error",
      shapes = list(value = if (is.null(have)) length(value) else have, param = info$dim)
    )
  }
  zlm_call("zlm_store_set", p$ptr, enc2utf8(name), value)
  invisible(p)
}

#' @rdname zl_params
#' @export
zl_param_names <- function(p) {
  zlm_check_params(p)
  zlm_call("zlm_store_info", p$ptr)$names
}

#' @rdname zl_params
#' @export
zl_param_count <- function(p) {
  zlm_check_params(p)
  dims <- zlm_call("zlm_store_info", p$ptr)$dims
  sum(vapply(dims, prod, numeric(1)))
}

#' @rdname zl_params
#' @export
zl_param_tensor <- function(p, name) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  info <- zlm_param_info(p, name)
  structure(
    list(store = p, name = name, dim = info$dim, type = info$type),
    class = c("zl_param_ref", "zl_node")
  )
}

zlm_param_info <- function(p, name) {
  info <- zlm_call("zlm_store_info", p$ptr)
  i <- match(name, info$names)
  if (is.na(i)) {
    zlm_abort(sprintf("The store has no parameter '%s'.", name), "zulm_invalid_argument")
  }
  list(dim = info$dims[[i]], type = info$types[[i]])
}

zlm_format_dim <- function(d) {
  paste0("c(", paste(format(d, scientific = FALSE, trim = TRUE), collapse = ", "), ")")
}

#' @export
print.zl_params <- function(x, ...) {
  info <- tryCatch(zlm_call("zlm_store_info", x$ptr), zulm_error = function(e) NULL)
  if (is.null(info)) {
    cat("<zl_params> (invalid: restored from a saved session or freed)\n")
    return(invisible(x))
  }
  n <- sum(vapply(info$dims, prod, numeric(1)))
  cat(sprintf(
    "<zl_params> %d tensor%s, %s value%s, %s\n",
    length(info$names), if (length(info$names) == 1L) "" else "s",
    format(n, big.mark = ","), if (n == 1) "" else "s",
    format(structure(info$bytes, class = "object_size"), units = "auto")
  ))
  if (length(info$names)) {
    shown <- utils::head(seq_along(info$names), 20L)
    lines <- sprintf(
      "  %s %s %s", format(info$names[shown]), format(info$types[shown]),
      vapply(info$dims[shown], zlm_format_dim, character(1))
    )
    cat(lines, sep = "\n")
    if (length(info$names) > 20L) {
      cat(sprintf("  ... and %d more\n", length(info$names) - 20L))
    }
  }
  invisible(x)
}

#' @export
print.zl_param_ref <- function(x, ...) {
  cat(sprintf("<zl_param_ref> '%s' %s %s\n", x$name, x$type, zlm_format_dim(x$dim)))
  invisible(x)
}
