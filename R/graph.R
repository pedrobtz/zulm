#' Build and run graphs
#'
#' `zl_graph()` calls `build(x)` once, with `x` a named list of input
#' tensors of the declared shapes, and records the tensors `build()`
#' returns as the graph's outputs. Inside `build()`, inputs, parameters
#' (from [zl_param_tensor()]) and the results of operations such as
#' [zl_crossprod()] are tensors; `build()` is not called again, and a
#' different input shape needs another graph.
#'
#' `zl_forward()` writes `data` into the inputs, runs the graph and
#' returns the outputs as double arrays. The inputs are rewritten in place
#' on every call, so a graph runs any number of times without being
#' rebuilt.
#'
#' Inputs are single-precision (f32) unless declared with `zl_int()`,
#' which makes them 32-bit integers (i32) for [zl_get_rows()] and
#' [zl_rope()]. Integer indices count from 0, as in 'ggml'.
#'
#' @param build A function of one argument, the named list of inputs,
#'   returning a tensor or a named list of tensors.
#' @param p The parameter store whose parameters `build()` uses.
#' @param inputs A named list of input dimensions: numeric vectors of one
#'   to four positive whole numbers, or `zl_int(...)` for an integer input.
#' @param loss The name of a scalar output to train on (see the training
#'   functions); `NULL` for none.
#' @param outputs The names of the outputs to keep, of those `build()`
#'   returns; `NULL` for all.
#' @param ... Dimensions, for `zl_int()`.
#' @param g A graph.
#' @param data A named list of arrays, one per input, each matching its
#'   input's dimensions exactly (a plain vector for a one-dimensional
#'   input). `NA` is refused.
#' @return `zl_graph()` returns a graph of class `zl_graph`; `zl_int()` an
#'   input specification; `zl_forward()` a named list of double arrays;
#'   `zl_graph_info()` a list describing the graph.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(3, 2))
#' g <- zl_graph(
#'   function(x) zl_relu(zl_crossprod(zl_param_tensor(p, "w"), x$input)),
#'   p,
#'   inputs = list(input = c(3, 5))
#' )
#' out <- zl_forward(g, list(input = matrix(1, 3, 5)))
#' dim(out$output)
zl_graph <- function(build, p, inputs, loss = NULL, outputs = NULL) {
  if (!is.function(build)) {
    zlm_abort("`build` must be a function.", "zulm_invalid_argument")
  }
  zlm_check_params(p)
  spec <- zlm_check_inputs(inputs)
  if (!is.null(.zlm$building)) {
    zlm_abort("zl_graph() cannot be called inside another graph's build().", "zulm_invalid_argument")
  }
  zlm_maybe_gc()
  gptr <- zlm_call(
    "zlm_graph_new", p$ptr, spec$names, spec$dims, spec$types,
    zlm_max_bytes(), zlm_threads()
  )
  handles <- zlm_call("zlm_graph_inputs", gptr)
  x <- stats::setNames(
    lapply(seq_along(handles), function(i) zlm_new_tensor(handles[[i]], gptr, length(spec$dims[[i]]))),
    spec$names
  )
  .zlm$building <- list(ptr = gptr, store = p)
  finished <- FALSE
  on.exit({
    .zlm$building <- NULL
    if (!finished) zlm_call("zlm_graph_abandon", gptr)
  })
  res <- build(x)
  res <- zlm_check_outputs(res, loss, outputs)
  out_handles <- lapply(res, function(t) t$ptr)
  final <- zlm_call("zlm_graph_finish", gptr, out_handles)
  finished <- TRUE
  structure(
    list(
      ptr = gptr,
      store = p,
      inputs = spec,
      outputs = list(
        names = names(res),
        ranks = vapply(res, function(t) t$rank, integer(1)),
        handles = final,
        rowloss = lapply(res, function(t) t$rowloss)
      ),
      loss = loss
    ),
    class = "zl_graph"
  )
}

#' @rdname zl_graph
#' @export
zl_int <- function(...) {
  dim <- zlm_check_dim(c(...))
  structure(list(dim = dim), class = "zl_int_spec")
}

zlm_check_inputs <- function(inputs) {
  if (!is.list(inputs) || length(inputs) == 0L || is.null(names(inputs)) ||
    anyNA(names(inputs)) || any(!nzchar(names(inputs))) || anyDuplicated(names(inputs))) {
    zlm_abort("`inputs` must be a non-empty list with unique names.", "zulm_invalid_argument")
  }
  if (any(nchar(names(inputs), type = "bytes") > 63L)) {
    zlm_abort("Input names have at most 63 bytes.", "zulm_invalid_argument")
  }
  types <- character(length(inputs))
  dims <- vector("list", length(inputs))
  for (i in seq_along(inputs)) {
    v <- inputs[[i]]
    if (inherits(v, "zl_int_spec")) {
      types[[i]] <- "i32"
      dims[[i]] <- v$dim
    } else {
      types[[i]] <- "f32"
      dims[[i]] <- zlm_check_dim(v)
    }
  }
  list(names = enc2utf8(names(inputs)), dims = dims, types = types)
}

zlm_check_outputs <- function(res, loss, outputs) {
  if (inherits(res, "zl_node")) {
    res <- list(output = res)
  }
  if (!is.list(res) || length(res) == 0L || is.null(names(res)) ||
    anyNA(names(res)) || any(!nzchar(names(res))) || anyDuplicated(names(res)) ||
    !all(vapply(res, inherits, logical(1), "zl_node"))) {
    zlm_abort(
      "`build()` must return a tensor or a named list of tensors.",
      "zulm_invalid_argument"
    )
  }
  res <- lapply(res, zlm_as_tensor)
  if (!is.null(outputs)) {
    if (!is.character(outputs) || !all(outputs %in% names(res))) {
      zlm_abort("`outputs` must name outputs that `build()` returns.", "zulm_invalid_argument")
    }
    keep <- unique(c(outputs, loss))
    res <- res[keep]
  }
  if (!is.null(loss)) {
    zlm_check_string(loss, "loss")
    if (!loss %in% names(res)) {
      zlm_abort(sprintf("`build()` returns no output named '%s'.", loss), "zulm_invalid_argument")
    }
    if (prod(dim(res[[loss]])) != 1) {
      zlm_abort(sprintf("The loss '%s' must be a scalar.", loss), "zulm_shape_error")
    }
  }
  res
}

#' @rdname zl_graph
#' @export
zl_forward <- function(g, data, outputs = NULL) {
  zlm_check_graph(g)
  zlm_write_inputs(g, data)
  which <- zlm_output_index(g, outputs)
  vals <- zlm_call("zlm_graph_compute", g$ptr, which - 1L)
  names(vals) <- g$outputs$names[which]
  for (i in seq_along(vals)) {
    rl <- g$outputs$rowloss[[which[[i]]]]
    if (!is.null(rl)) {
      # the fixed-order sum of the rows, the same at any thread count
      vals[[i]] <- zlm_rowloss_sum(rl)
    }
  }
  for (i in seq_along(vals)) {
    r <- g$outputs$ranks[[which[[i]]]]
    if (r > 1L) {
      info <- zlm_call("zlm_tensor_info", g$outputs$handles[[which[[i]]]], g$ptr)
      dim(vals[[i]]) <- info[[1L]][seq_len(r)]
    }
  }
  vals
}

zlm_output_index <- function(g, outputs) {
  if (is.null(outputs)) {
    return(seq_along(g$outputs$names))
  }
  if (!is.character(outputs) || anyNA(outputs) || !all(outputs %in% g$outputs$names)) {
    zlm_abort("`outputs` must name outputs of the graph.", "zulm_invalid_argument")
  }
  match(outputs, g$outputs$names)
}

zlm_write_inputs <- function(g, data) {
  spec <- g$inputs
  if (!is.list(data) || is.null(names(data)) || !setequal(names(data), spec$names) ||
    anyDuplicated(names(data))) {
    zlm_abort(
      sprintf("`data` must be a list with one element per input: %s.", paste(spec$names, collapse = ", ")),
      "zulm_invalid_argument"
    )
  }
  for (i in seq_along(spec$names)) {
    nm <- spec$names[[i]]
    v <- data[[nm]]
    want <- spec$dims[[i]]
    if (!is.numeric(v) && !is.logical(v)) {
      zlm_abort(sprintf("Input '%s' must be numeric.", nm), "zulm_type_error")
    }
    have <- dim(v)
    ok <- if (is.null(have)) length(want) == 1L else length(have) == length(want) && all(have == want)
    if (!ok || length(v) != prod(want)) {
      zlm_abort(
        sprintf(
          "Input '%s' has dimensions %s; the graph was built for %s.",
          nm, zlm_format_dim(if (is.null(have)) length(v) else have), zlm_format_dim(want)
        ),
        "zulm_shape_error",
        shapes = list(data = if (is.null(have)) length(v) else have, input = want)
      )
    }
    zlm_call("zlm_graph_set_input", g$ptr, i - 1L, v)
  }
  invisible(g)
}

zlm_check_graph <- function(g) {
  if (!inherits(g, "zl_graph")) {
    zlm_abort("`g` must be a graph from zl_graph().", "zulm_invalid_argument")
  }
  zlm_check_params(g$store)
  invisible(g)
}

#' @rdname zl_graph
#' @export
zl_graph_info <- function(g) {
  zlm_check_graph(g)
  info <- zlm_call("zlm_graph_info", g$ptr)
  ops <- info[[1L]]
  list(
    nodes = length(ops),
    ops = sort(table(ops), decreasing = TRUE),
    inputs = stats::setNames(g$inputs$dims, g$inputs$names),
    outputs = g$outputs$names,
    loss = g$loss,
    bytes = c(inputs = info[[2L]], compute = info[[3L]])
  )
}

#' @export
print.zl_graph <- function(x, ...) {
  info <- tryCatch(zl_graph_info(x), zulm_error = function(e) NULL)
  if (is.null(info)) {
    cat("<zl_graph> (invalid)\n")
    return(invisible(x))
  }
  cat(sprintf(
    "<zl_graph> %d nodes; inputs %s; outputs %s; %s\n",
    info$nodes,
    paste(sprintf("%s %s", names(info$inputs), vapply(info$inputs, zlm_format_dim, "")), collapse = ", "),
    paste(info$outputs, collapse = ", "),
    format(structure(sum(info$bytes), class = "object_size"), units = "auto")
  ))
  invisible(x)
}

# The reported value of a loss with per-row companions: their fixed-order
# sums, added in order.
zlm_rowloss_sum <- function(rl) {
  total <- 0
  for (r in rl) total <- total + zlm_call("zlm_rowloss_get", r)
  total
}
