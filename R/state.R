# State tensors and the graph cache (design §6.9).

#' State tensors
#'
#' A state tensor is a persistent f32 tensor in a parameter store that
#' graphs read like a parameter and write with `zl_state_write()`. It is
#' never trained and never saved: [zl_save()] and [zl_save_safetensors()]
#' leave it out, and [zl_param_names()] and [zl_param_count()] do not count
#' it. A key-value cache for incremental decoding is two state tensors per
#' layer, of a fixed capacity.
#'
#' `zl_state()` declares a state of zeros. Inside `build()`, take a handle
#' on it with [zl_param_tensor()]. `zl_state_write(s, x, rows)` writes the
#' second-dimension slices of `x` into the slices of `s` that `rows` names,
#' each time the graph runs. `rows` must be an integer input declared with
#' [zl_int()], counting from 0, and no row may appear twice in one run.
#' The write returns `s` after the write: read the state through it, so the
#' read follows the write. A graph that writes state cannot be trained.
#'
#' `zl_state_set()` without `offset` replaces the whole state, as
#' [zl_param_set()] does, which resets it. With `offset`, it writes the
#' values of `value` in order from that position of the state's values,
#' counted from 0 with the first index fastest, as R stores an array. A
#' write that would reach past the state's end is an error. A sliding window
#' shifts this way without rebuilding a graph.
#'
#' @param p A parameter store from [zl_params()] or [zl_load()].
#' @param name The state's name: a string of 1 to 63 bytes, unique in the
#'   store.
#' @param dim Its dimensions: one to four positive whole numbers.
#' @param value For `zl_state_set()`: without `offset`, an array of the
#'   state's dimensions; with `offset`, a numeric vector.
#' @param offset `NULL`, or the position, counted from 0, at which to start
#'   writing `value`.
#' @param s A state tensor inside `build()`, from [zl_param_tensor()].
#' @param x A tensor whose first, third and fourth dimensions are the
#'   state's.
#' @param rows An integer input of one index per second-dimension slice of
#'   `x`.
#' @return `zl_state()` and `zl_state_set()` return the store invisibly;
#'   `zl_state_get()` a double array; `zl_state_write()` the state tensor
#'   after the write.
#' @export
#' @examples
#' p <- zl_params()
#' zl_state(p, "cache", c(2, 4))
#' g <- zl_graph(
#'   function(x) {
#'     s <- zl_state_write(zl_param_tensor(p, "cache"), x$new, x$rows)
#'     zl_sum(s)
#'   },
#'   p,
#'   inputs = list(new = c(2, 1), rows = zl_int(1))
#' )
#' zl_forward(g, list(new = matrix(1, 2, 1), rows = 0L))
#' zl_forward(g, list(new = matrix(2, 2, 1), rows = 3L))
#' zl_state_get(p, "cache")
#' zl_state_set(p, "cache", c(9, 9), offset = 2)
#' zl_state_get(p, "cache")
zl_state <- function(p, name, dim) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  dim <- zlm_check_dim(dim)
  zlm_call("zlm_store_add", p$ptr, enc2utf8(name), dim, "f32", NULL, zlm_max_bytes(), 1L)
  invisible(p)
}

#' @rdname zl_state
#' @export
zl_state_get <- function(p, name) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  info <- zlm_param_info(p, name, kind = "state")
  out <- zlm_call("zlm_store_get", p$ptr, enc2utf8(name))
  if (length(info$dim) > 1L) {
    dim(out) <- info$dim
  }
  out
}

#' @rdname zl_state
#' @export
zl_state_set <- function(p, name, value, offset = NULL) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  info <- zlm_param_info(p, name, kind = "state")
  if (is.null(offset)) {
    zlm_set_whole(p, name, value, info)
    return(invisible(p))
  }
  if (!is.numeric(offset) || length(offset) != 1L || is.na(offset) || offset < 0 ||
    offset != floor(offset)) {
    zlm_abort("`offset` must be a whole number from 0.", "zulm_invalid_argument")
  }
  if (!is.numeric(value)) {
    zlm_abort("`value` must be numeric.", "zulm_type_error")
  }
  zlm_call("zlm_state_write_at", p$ptr, enc2utf8(name), as.double(offset), as.double(value))
  invisible(p)
}

#' @rdname zl_state
#' @export
zl_state_write <- function(s, x, rows) {
  if (!inherits(s, "zl_param_ref")) {
    zlm_abort("`s` must be a state tensor from zl_param_tensor().", "zulm_invalid_argument")
  }
  rank <- length(s$dim)
  s <- zlm_as_tensor(s, "s")
  x <- zlm_as_tensor(x)
  rows <- zlm_as_tensor(rows, "rows")
  r <- zlm_call("zlm_op", s$graph, "set_rows", s$ptr, rows$ptr, x$ptr, integer(), sys.call())
  zlm_new_tensor(r, s$graph, rank)
}

#' Graphs built once per input shape
#'
#' `zl_graph_cache()` wraps [zl_graph()] for loops whose input shapes vary,
#' such as a decoder whose count of new positions changes. Its graphs are
#' built by [zl_forward()] the first time it sees a set of input shapes,
#' and reused after. An input given as an R integer vector or array becomes
#' an integer input (see [zl_int()]); any other numeric input is f32.
#'
#' When the store's and the cached graphs' memory together would pass
#' `getOption("zulm.max_bytes")`, the graphs used least recently are freed
#' first; the graph just built is kept.
#'
#' @inheritParams zl_graph
#' @param x A graph cache.
#' @param ... Unused.
#' @return A graph cache, of class `zl_graph_cache`, for [zl_forward()].
#'   `length()` gives the number of graphs it holds.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(3, 2))
#' gc <- zl_graph_cache(function(x) zl_crossprod(zl_param_tensor(p, "w"), x$input), p)
#' dim(zl_forward(gc, list(input = matrix(1, 3, 5)))$output)
#' dim(zl_forward(gc, list(input = matrix(1, 3, 7)))$output)
#' length(gc)
zl_graph_cache <- function(build, p, loss = NULL, outputs = NULL) {
  if (!is.function(build)) {
    zlm_abort("`build` must be a function.", "zulm_invalid_argument")
  }
  zlm_check_params(p)
  gc <- new.env(parent = emptyenv())
  gc$build <- build
  gc$store <- p
  gc$loss <- loss
  gc$outputs <- outputs
  gc$graphs <- list()
  gc$used <- numeric()
  gc$clock <- 0
  gc$builds <- 0L
  class(gc) <- "zl_graph_cache"
  gc
}

zlm_cache_graph <- function(gc, data) {
  if (!is.list(data) || length(data) == 0L || is.null(names(data))) {
    zlm_abort("`data` must be a named list of arrays.", "zulm_invalid_argument")
  }
  data <- data[order(names(data))]
  inputs <- lapply(data, function(v) {
    d <- if (is.null(dim(v))) length(v) else dim(v)
    if (is.integer(v)) zl_int(d) else d
  })
  key <- paste(
    names(inputs),
    vapply(data, function(v) if (is.integer(v)) "i" else "f", ""),
    vapply(inputs, function(v) paste(if (inherits(v, "zl_int_spec")) v$dim else v, collapse = "x"), ""),
    sep = ":", collapse = ";"
  )
  gc$clock <- gc$clock + 1
  g <- gc$graphs[[key]]
  if (is.null(g)) {
    g <- zl_graph(gc$build, gc$store, inputs, loss = gc$loss, outputs = gc$outputs)
    gc$builds <- gc$builds + 1L
    gc$graphs[[key]] <- g
    zlm_cache_evict(gc, keep = key)
  }
  gc$used[[key]] <- gc$clock
  g
}

zlm_cache_evict <- function(gc, keep) {
  cap <- zlm_max_bytes()
  store <- zlm_call("zlm_store_info", gc$store$ptr)$bytes
  repeat {
    sizes <- vapply(gc$graphs, function(g) sum(zl_graph_info(g)$bytes), numeric(1))
    if (store + sum(sizes) <= cap || length(gc$graphs) <= 1L) break
    old <- setdiff(names(gc$graphs), keep)
    victim <- old[[which.min(gc$used[old])]]
    zlm_call("zlm_graph_free_", gc$graphs[[victim]]$ptr)
    gc$graphs[[victim]] <- NULL
    gc$used <- gc$used[names(gc$used) != victim]
  }
}

#' @export
length.zl_graph_cache <- function(x) length(x$graphs)

#' @rdname zl_graph_cache
#' @export
print.zl_graph_cache <- function(x, ...) {
  cat(sprintf(
    "<zl_graph_cache> %d graph%s held, %d built\n",
    length(x), if (length(x) == 1L) "" else "s", x$builds
  ))
  invisible(x)
}
