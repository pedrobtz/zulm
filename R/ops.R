# Tensors and operations inside build() (design §5.3, §6.1, §6.3).
#
# A tensor is an R handle on a node of the graph being built: an external
# pointer to the ggml tensor (whose protected value is the graph) and the
# tensor's R rank, the length of its dim(). ggml always has four
# dimensions; the rank says how many R shows, so a c(3, 1) matrix stays a
# matrix. Each operation's result rank is decided here, its shape and
# types are checked in C (zlm_graph.c) before ggml sees them.

zlm_new_tensor <- function(ptr, gptr, rank) {
  x <- list(ptr = ptr, graph = gptr, rank = as.integer(rank))
  class(x) <- c("zl_tensor", "zl_node")
  x
}

zlm_current_graph <- function() {
  b <- .zlm$building
  if (is.null(b)) {
    zlm_abort("Operations on tensors are used inside the build() function of zl_graph().", "zulm_invalid_argument")
  }
  b
}

# A tensor handle from a tensor or a parameter reference (zl_param_tensor()).
zlm_as_tensor <- function(x, what = "x") {
  if (inherits(x, "zl_tensor")) {
    return(x)
  }
  if (inherits(x, "zl_param_ref")) {
    b <- zlm_current_graph()
    if (!identical(x$store$ptr, b$store$ptr)) {
      zlm_abort(
        sprintf("Parameter '%s' belongs to another store than the graph's.", x$name),
        "zulm_invalid_argument"
      )
    }
    ptr <- zlm_call_quiet("zlm_graph_param", b$ptr, enc2utf8(x$name))
    return(zlm_new_tensor(ptr, b$ptr, length(x$dim)))
  }
  zlm_abort(sprintf("`%s` must be a tensor inside build().", what), "zulm_invalid_argument")
}

# Runs one operation in C and wraps the result with rank `rank`.
zlm_op <- function(op, a, b = NULL, num = numeric(), ints = integer(), rank = a$rank, call = NULL) {
  r <- zlm_call_quiet(
    "zlm_op", a$graph, op, a$ptr, if (is.null(b)) NULL else b$ptr,
    as.double(num), as.integer(ints), call
  )
  zlm_new_tensor(r, a$graph, rank)
}

#' @export
dim.zl_tensor <- function(x) {
  info <- zlm_call("zlm_tensor_info", x$ptr, x$graph)
  as.integer(info[[1L]][seq_len(x$rank)])
}

#' @export
dim.zl_param_ref <- function(x) as.integer(x$dim)

#' @export
print.zl_tensor <- function(x, ...) {
  info <- tryCatch(zlm_call("zlm_tensor_info", x$ptr, x$graph), zulm_error = function(e) NULL)
  if (is.null(info)) {
    cat("<zl_tensor> (invalid)\n")
  } else {
    cat(sprintf("<zl_tensor> %s %s (%s)\n", info[[2L]], zlm_format_dim(info[[1L]][seq_len(x$rank)]), info[[3L]]))
  }
  invisible(x)
}

zlm_check_num <- function(x, what) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x)) {
    zlm_abort(sprintf("`%s` must be a single finite number.", what), "zulm_invalid_argument")
  }
  as.double(x)
}

# ---- arithmetic ---------------------------------------------------------

#' Arithmetic, reductions and elementwise functions on tensors
#'
#' Inside `build()`, `+`, `-`, `*` and `/` combine tensors elementwise with
#' broadcasting: one operand's dimensions must each divide the other's, as
#' in 'ggml'. A number on either side is a scale or an added constant.
#' `zl_sum()` sums every element to a scalar; `zl_sum_rows()` and
#' `zl_mean_rows()` reduce the first dimension to 1; `zl_cumsum()` takes
#' cumulative sums along it. The others apply elementwise.
#'
#' `zl_cumsum()`, `zl_gelu()`, `zl_sigmoid()` and `zl_tanh()` have no
#' gradient: they serve inference and sampling graphs, and training refuses
#' them on a gradient path.
#'
#' @param x A tensor.
#' @param s A number.
#' @param approximate For `zl_gelu()`: `"none"` for the exact form with the
#'   error function (accurate to single precision), `"tanh"` for the tanh
#'   approximation, which 'ggml' evaluates through a half-precision table
#'   (accurate to about 1e-3).
#' @return A tensor.
#' @name zl-arith
#' @examples
#' p <- zl_params()
#' g <- zl_graph(
#'   function(x) list(
#'     y = zl_sum_rows(zl_sqr(x$a - 1)),
#'     z = 2 * zl_exp(x$a) + 1
#'   ),
#'   p,
#'   inputs = list(a = c(3, 2))
#' )
#' zl_forward(g, list(a = matrix(1:6, 3, 2)))
NULL

#' @export
Ops.zl_node <- function(e1, e2) {
  call <- sys.call()
  if (missing(e2)) {
    if (.Generic == "-") {
      a <- zlm_as_tensor(e1)
      return(zlm_op("neg", a, call = call))
    }
    if (.Generic == "+") {
      return(zlm_as_tensor(e1))
    }
    zlm_abort(sprintf("Unary `%s` is not defined for tensors.", .Generic), "zulm_invalid_argument")
  }
  if (!.Generic %in% c("+", "-", "*", "/")) {
    zlm_abort(sprintf("`%s` is not defined for tensors.", .Generic), "zulm_invalid_argument")
  }
  t1 <- inherits(e1, "zl_node")
  t2 <- inherits(e2, "zl_node")
  if (t1 && t2) {
    a <- zlm_as_tensor(e1)
    b <- zlm_as_tensor(e2)
    op <- switch(.Generic, "+" = "add", "-" = "sub", "*" = "mul", "/" = "div")
    out <- zlm_op(op, a, b, rank = max(a$rank, b$rank), call = call)
    if (.Generic == "+" && !is.null(a$rowloss) && !is.null(b$rowloss)) {
      # a sum of losses reports the sum of their per-row sums (design §7.1)
      out$rowloss <- c(a$rowloss, b$rowloss)
    }
    return(out)
  }
  if (t1) {
    a <- zlm_as_tensor(e1)
    s <- zlm_check_num(e2, "the number")
    return(switch(
      .Generic,
      "+" = zlm_op("scale_bias", a, num = c(1, s), call = call),
      "-" = zlm_op("scale_bias", a, num = c(1, -s), call = call),
      "*" = zlm_op("scale", a, num = s, call = call),
      "/" = zlm_op("scale", a, num = 1 / s, call = call)
    ))
  }
  a <- zlm_as_tensor(e2)
  s <- zlm_check_num(e1, "the number")
  switch(
    .Generic,
    "+" = zlm_op("scale_bias", a, num = c(1, s), call = call),
    "-" = zlm_op("scale_bias", a, num = c(-1, s), call = call),
    "*" = zlm_op("scale", a, num = s, call = call),
    "/" = zlm_op("fill_div", a, num = s, call = call)
  )
}

#' @rdname zl-arith
#' @export
zl_scale <- function(x, s) {
  zlm_op("scale", zlm_as_tensor(x), num = zlm_check_num(s, "s"), call = sys.call())
}

#' @rdname zl-arith
#' @export
zl_sum <- function(x) zlm_op("sum", zlm_as_tensor(x), rank = 1L, call = sys.call())

#' @rdname zl-arith
#' @export
zl_sum_rows <- function(x) zlm_op("sum_rows", zlm_as_tensor(x), call = sys.call())

#' @rdname zl-arith
#' @export
zl_mean_rows <- function(x) zlm_op("mean_rows", zlm_as_tensor(x), call = sys.call())

#' @rdname zl-arith
#' @export
zl_cumsum <- function(x) zlm_op("cumsum", zlm_as_tensor(x), call = sys.call())

zlm_unary <- function(op) {
  force(op)
  function(x) zlm_op(op, zlm_as_tensor(x), call = sys.call())
}

#' @rdname zl-arith
#' @export
zl_sqr <- zlm_unary("sqr")
#' @rdname zl-arith
#' @export
zl_sqrt <- zlm_unary("sqrt")
#' @rdname zl-arith
#' @export
zl_exp <- zlm_unary("exp")
#' @rdname zl-arith
#' @export
zl_log <- zlm_unary("log")
#' @rdname zl-arith
#' @export
zl_abs <- zlm_unary("abs")
#' @rdname zl-arith
#' @export
zl_sin <- zlm_unary("sin")
#' @rdname zl-arith
#' @export
zl_cos <- zlm_unary("cos")
#' @rdname zl-arith
#' @export
zl_relu <- zlm_unary("relu")
#' @rdname zl-arith
#' @export
zl_silu <- zlm_unary("silu")
#' @rdname zl-arith
#' @export
zl_softplus <- zlm_unary("softplus")
#' @rdname zl-arith
#' @export
zl_sigmoid <- zlm_unary("sigmoid")
#' @rdname zl-arith
#' @export
zl_tanh <- zlm_unary("tanh")

#' @rdname zl-arith
#' @export
zl_gelu <- function(x, approximate = c("none", "tanh")) {
  approximate <- match.arg(approximate)
  zlm_op("gelu", zlm_as_tensor(x), ints = as.integer(approximate == "tanh"), call = sys.call())
}

#' @rdname zl-arith
#' @param gate,up Tensors of the same shape.
#' @export
zl_swiglu <- function(gate, up) {
  a <- zlm_as_tensor(gate, "gate")
  b <- zlm_as_tensor(up, "up")
  zlm_op("swiglu", a, b, rank = max(a$rank, b$rank), call = sys.call())
}

# ---- products and shape -------------------------------------------------

#' Products and shapes of tensors
#'
#' `zl_crossprod(a, b)` is `crossprod(a, b)`: for `a` of dims `c(K, M)` and
#' `b` of `c(K, N)` it has dims `c(M, N)`, batched over the third and
#' fourth dimensions (where `a`'s may divide `b`'s). A linear layer stores
#' its weight as `c(n_in, n_out)` and computes `zl_crossprod(w, x)`; the
#' weight may be stored as f16 or bf16.
#'
#' `zl_reshape()` gives new dimensions with the same number of elements;
#' `zl_permute()` reorders dimensions as [aperm()] does; `zl_t()`
#' transposes the first two; `zl_cont()` makes a permuted tensor
#' contiguous (operations do it themselves when they need to);
#' `zl_view()` is a window of `prod(dim)` elements from position `offset`
#' (counting from 0) of a contiguous tensor. `zl_concat()` joins two
#' tensors along `axis`, `zl_pad()` appends zeros after each dimension,
#' and `zl_get_rows()` selects columns of a matrix (rows in 'ggml''s
#' terms) by an integer input, counting from 0. `zl_concat()` and
#' `zl_pad()` have no gradient.
#'
#' @param a,b,x Tensors.
#' @param dim New dimensions: one to four positive whole numbers.
#' @param perm A permutation of `seq_len(k)`, with `k` at least the rank of
#'   `x` and at most 4.
#' @param offset The first element of the view, counting from 0.
#' @param axis The dimension to join along, from 1 to 4.
#' @param after Numbers of zeros to append after each dimension.
#' @param idx An integer input (declared with [zl_int()]) of column indices.
#' @return A tensor.
#' @name zl-shape
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(4, 2))
#' g <- zl_graph(
#'   function(x) zl_t(zl_crossprod(zl_param_tensor(p, "w"), x$x)),
#'   p,
#'   inputs = list(x = c(4, 3))
#' )
#' dim(zl_forward(g, list(x = matrix(1, 4, 3)))$output)
NULL

#' @rdname zl-shape
#' @export
zl_crossprod <- function(a, b) {
  a <- zlm_as_tensor(a, "a")
  b <- zlm_as_tensor(b, "b")
  zlm_op("crossprod", a, b, rank = max(2L, a$rank, b$rank), call = sys.call())
}

zlm_dims4 <- function(dim) {
  d <- zlm_check_dim(dim)
  c(d, rep(1, 4L - length(d)))
}

#' @rdname zl-shape
#' @export
zl_reshape <- function(x, dim) {
  x <- zlm_as_tensor(x)
  zlm_op("reshape", x, ints = zlm_dims4(dim), rank = length(dim), call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_permute <- function(x, perm) {
  x <- zlm_as_tensor(x)
  k <- length(perm)
  if (!is.numeric(perm) || k < x$rank || k > 4L || anyNA(perm) ||
    !setequal(perm, seq_len(k)) || anyDuplicated(perm)) {
    zlm_abort("`perm` must be a permutation of seq_len(k), with k from the rank of `x` to 4.", "zulm_invalid_argument")
  }
  p4 <- c(as.integer(perm), if (k < 4L) seq.int(k + 1L, 4L))
  # ggml_permute() takes, for each old axis, its new position
  newpos <- integer(4)
  newpos[p4] <- 0:3
  zlm_op("permute", x, ints = newpos, rank = k, call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_t <- function(x) {
  x <- zlm_as_tensor(x)
  if (x$rank > 2L) {
    zlm_abort("zl_t() takes a vector or a matrix; use zl_permute() for more dimensions.", "zulm_shape_error")
  }
  zlm_op("transpose", x, rank = 2L, call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_cont <- function(x) zlm_op("cont", zlm_as_tensor(x), call = sys.call())

#' @rdname zl-shape
#' @export
zl_view <- function(x, dim, offset = 0) {
  x <- zlm_as_tensor(x)
  offset <- zlm_check_num(offset, "offset")
  if (offset < 0 || offset != floor(offset)) {
    zlm_abort("`offset` must be a non-negative whole number.", "zulm_invalid_argument")
  }
  zlm_op("view", x, num = offset, ints = zlm_dims4(dim), rank = length(dim), call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_concat <- function(a, b, axis) {
  a <- zlm_as_tensor(a, "a")
  b <- zlm_as_tensor(b, "b")
  if (!is.numeric(axis) || length(axis) != 1L || !axis %in% 1:4) {
    zlm_abort("`axis` must be 1, 2, 3 or 4.", "zulm_invalid_argument")
  }
  zlm_op("concat", a, b, ints = as.integer(axis) - 1L, rank = max(a$rank, b$rank, axis), call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_pad <- function(x, after) {
  x <- zlm_as_tensor(x)
  if (!is.numeric(after) || length(after) < 1L || length(after) > 4L || anyNA(after) ||
    any(after < 0) || any(after != floor(after))) {
    zlm_abort("`after` must hold one to four non-negative whole numbers.", "zulm_invalid_argument")
  }
  zlm_op("pad", x, ints = c(after, rep(0, 4L - length(after))), rank = max(x$rank, length(after)), call = sys.call())
}

#' @rdname zl-shape
#' @export
zl_get_rows <- function(x, idx) {
  x <- zlm_as_tensor(x)
  idx <- zlm_as_tensor(idx, "idx")
  zlm_op("get_rows", x, idx, rank = 2L, call = sys.call())
}

# ---- selection and clipping --------------------------------------------

#' Selection, clipping and triangles
#'
#' `zl_where(mask, a, b)` is `mask * a + (1 - mask) * b` for a mask of 0s
#' and 1s. `zl_clamp()` limits values to `[lo, hi]`. `zl_tri()` keeps one
#' triangle of each matrix, as [lower.tri()] and [upper.tri()] select it,
#' and sets the rest to zero. `zl_clamp()` and `zl_tri()` have no gradient.
#'
#' @param mask,a,b,x Tensors.
#' @param lo,hi Numbers.
#' @param type Which triangle: `"lower"` and `"upper"` include the
#'   diagonal, the `_strict` forms do not.
#' @return A tensor.
#' @name zl-select
#' @examples
#' p <- zl_params()
#' g <- zl_graph(
#'   function(x) zl_tri(zl_clamp(x$m, -1, 1), "lower"),
#'   p,
#'   inputs = list(m = c(3, 3))
#' )
#' zl_forward(g, list(m = matrix(-4:4, 3, 3)))$output
NULL

#' @rdname zl-select
#' @export
zl_where <- function(mask, a, b) {
  mask <- zlm_as_tensor(mask, "mask")
  a <- zlm_as_tensor(a, "a")
  b <- zlm_as_tensor(b, "b")
  b + mask * (a - b)
}

#' @rdname zl-select
#' @export
zl_clamp <- function(x, lo, hi) {
  lo <- zlm_check_num(lo, "lo")
  hi <- zlm_check_num(hi, "hi")
  if (lo > hi) {
    zlm_abort("`lo` must not exceed `hi`.", "zulm_invalid_argument")
  }
  zlm_op("clamp", zlm_as_tensor(x), num = c(lo, hi), call = sys.call())
}

#' @rdname zl-select
#' @export
zl_tri <- function(x, type = c("lower", "upper", "lower_strict", "upper_strict")) {
  type <- match.arg(type)
  x <- zlm_as_tensor(x)
  # ggml numbers the first dimension as columns: R's lower triangle (row
  # index above column index) is ggml's upper.
  code <- switch(type, lower = 0L, lower_strict = 1L, upper = 2L, upper_strict = 3L)
  zlm_op("tri", x, ints = code, rank = max(2L, x$rank), call = sys.call())
}

# ---- attention pieces and normalisation --------------------------------

#' Attention pieces and normalisation
#'
#' `zl_softmax()` normalises each column (the first dimension) after
#' multiplying by `scale` and adding `mask`, an additive f32 tensor whose
#' first two dimensions are `x`'s and whose third and fourth are 1 or
#' `x`'s, so one mask serves every head. `zl_causal_mask()` sets to `-Inf`
#' every element of each matrix whose row index exceeds its column index,
#' so that for scores of dims `c(n_kv, n_q, ...)` a query sees only keys
#' at or before it.
#'
#' `zl_rope()` applies rotary positions to `x` of dims
#' `c(head_dim, heads, n, ...)` with integer positions `pos` of length `n`
#' (an input declared with [zl_int()]). `mode = "interleaved"` rotates
#' adjacent pairs ('ggml''s default and Toto's convention); `mode =
#' "half"` rotates the first half against the second (TimesFM, Chronos-2,
#' most Hugging Face checkpoints). `n_dims` below `head_dim` rotates the
#' leading `n_dims` only.
#'
#' `zl_rms_norm()` divides each column by its root mean square;
#' `zl_layer_norm()` subtracts the column mean first.
#'
#' @param x,mask Tensors.
#' @param scale A number.
#' @param pos An integer input of positions.
#' @param n_dims The number of leading dimensions to rotate: even, at most
#'   `head_dim`.
#' @param base The frequency base.
#' @param mode `"interleaved"` or `"half"`.
#' @param eps Added to the mean square, a positive number.
#' @return A tensor.
#' @name zl-attention
#' @examples
#' p <- zl_params()
#' g <- zl_graph(
#'   function(x) zl_softmax(zl_causal_mask(x$s)),
#'   p,
#'   inputs = list(s = c(3, 3))
#' )
#' zl_forward(g, list(s = matrix(0, 3, 3)))$output
NULL

#' @rdname zl-attention
#' @export
zl_softmax <- function(x, mask = NULL, scale = 1) {
  x <- zlm_as_tensor(x)
  scale <- zlm_check_num(scale, "scale")
  m <- if (is.null(mask)) NULL else zlm_as_tensor(mask, "mask")
  zlm_op("softmax", x, m, num = scale, call = sys.call())
}

#' @rdname zl-attention
#' @export
zl_causal_mask <- function(x) {
  x <- zlm_as_tensor(x)
  zlm_op("causal_mask", x, rank = max(2L, x$rank), call = sys.call())
}

#' @rdname zl-attention
#' @export
zl_rope <- function(x, pos, n_dims, base = 10000, mode = c("interleaved", "half")) {
  mode <- match.arg(mode)
  x <- zlm_as_tensor(x)
  pos <- zlm_as_tensor(pos, "pos")
  n_dims <- zlm_check_num(n_dims, "n_dims")
  base <- zlm_check_num(base, "base")
  if (base <= 0) {
    zlm_abort("`base` must be positive.", "zulm_invalid_argument")
  }
  zlm_op(
    "rope", x, pos, num = base,
    ints = c(as.integer(n_dims), if (mode == "half") 2L else 0L),
    rank = max(3L, x$rank), call = sys.call()
  )
}

#' @rdname zl-attention
#' @export
zl_rms_norm <- function(x, eps = 1e-5) {
  eps <- zlm_check_num(eps, "eps")
  if (eps <= 0) {
    zlm_abort("`eps` must be positive.", "zulm_invalid_argument")
  }
  zlm_op("rms_norm", zlm_as_tensor(x), num = eps, call = sys.call())
}

#' @rdname zl-attention
#' @export
zl_layer_norm <- function(x, eps = 1e-5) {
  # design §6.3: rms_norm(x - mean(x)), the mean as scale(sum_rows(x), 1/n)
  x <- zlm_as_tensor(x)
  zl_rms_norm(x - zl_mean_rows(x), eps)
}

# ---- losses -------------------------------------------------------------

#' Losses
#'
#' Each returns a scalar tensor. `zl_cross_entropy(logits, prob)` is the
#' cross-entropy between the softmax of each column of `logits` and the
#' probabilities `prob` of the same shape, summed over columns and divided
#' by their number, as 'ggml' computes it. `zl_mse()` and `zl_mae()` are
#' the mean squared and mean absolute errors over every element.
#'
#' @param logits,prob,pred,target Tensors.
#' @return A scalar tensor.
#' @name zl-loss
#' @examples
#' p <- zl_params()
#' g <- zl_graph(
#'   function(x) zl_mse(x$pred, x$target),
#'   p,
#'   inputs = list(pred = 4, target = 4)
#' )
#' zl_forward(g, list(pred = c(1, 2, 3, 4), target = c(1, 2, 3, 6)))$output
NULL

#' @rdname zl-loss
#' @export
zl_cross_entropy <- function(logits, prob) {
  call <- sys.call()
  a <- zlm_as_tensor(logits, "logits")
  b <- zlm_as_tensor(prob, "prob")
  loss <- zlm_op("cross_entropy", a, b, rank = 1L, call = call)
  zlm_with_rows(loss, a, b, 0L, 1 / (prod(dim(a)) / dim(a)[[1L]]), call)
}

#' @rdname zl-loss
#' @export
zl_mse <- function(pred, target) {
  a <- zlm_as_tensor(pred, "pred")
  b <- zlm_as_tensor(target, "target")
  d <- a - b
  loss <- zl_scale(zl_sum(zl_sqr(d)), 1 / prod(dim(d)))
  zlm_with_rows(loss, a, b, 1L, 1 / prod(dim(d)), sys.call())
}

#' @rdname zl-loss
#' @export
zl_mae <- function(pred, target) {
  a <- zlm_as_tensor(pred, "pred")
  b <- zlm_as_tensor(target, "target")
  d <- a - b
  loss <- zl_scale(zl_sum(zl_abs(d)), 1 / prod(dim(d)))
  zlm_with_rows(loss, a, b, 2L, 1 / prod(dim(d)), sys.call())
}

# The reported value of a loss is a fixed-order sum of per-row values
# (design §7.1): a per-row node feeds the loss through
# 0 * sum(step(rows)), which adds exactly 0 and, because ggml never
# differentiates step()'s source, needs no gradient, yet keeps the rows
# inside ggml-opt's graphs. Only operands of the same shape have rows.
zlm_with_rows <- function(loss, a, b, mode, scale, call) {
  if (!identical(dim(a), dim(b))) {
    return(loss)
  }
  rl <- zlm_call("zlm_rowloss_op", a$graph, a$ptr, b$ptr, mode, as.double(scale), call)
  rows <- zlm_new_tensor(rl[[1L]], a$graph, 1L)
  hook <- zl_scale(zl_sum(zlm_op("step", rows, call = call)), 0)
  out <- loss + hook
  out$rowloss <- list(rl[[2L]])
  out
}

# ---- types ---------------------------------------------------------------

#' Convert a tensor's type
#'
#' `zl_cast()` converts to `"f32"`, `"f16"` or `"bf16"`. Operations take
#' f32 tensors, so a half-precision parameter is cast to f32 before any
#' operation but [zl_crossprod()] and [zl_get_rows()]. It has no gradient.
#'
#' @param x A tensor.
#' @param type The new type.
#' @return A tensor.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "h", 4, type = "f16", init = zl_init_ones())
#' g <- zl_graph(function(x) zl_cast(zl_param_tensor(p, "h"), "f32") + x$a,
#'               p, inputs = list(a = 4))
#' zl_forward(g, list(a = 1:4))$output
zl_cast <- function(x, type) {
  type <- zlm_match_type(type)
  code <- switch(type, f32 = 0L, f16 = 1L, bf16 = 30L)
  zlm_op("cast", zlm_as_tensor(x), ints = code, call = sys.call())
}
