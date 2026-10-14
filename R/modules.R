#' Modules
#'
#' Each constructor declares its parameters in the store `p`, under names
#' that start with `name`, and returns a function of tensors to use inside
#' `build()` (see [zl_graph()]). Tensors have the feature dimension first:
#' a batch of sequences is `c(dim, n, batch)`.
#'
#' * `zl_linear()`: `zl_crossprod(weight, x) + bias`, with `weight` of
#'   dims `c(n_in, n_out)`; parameters `<name>.weight` and `<name>.bias`,
#'   initialised uniformly on `(-1, 1) / sqrt(n_in)` as in PyTorch.
#' * `zl_embedding()`: rows of `<name>.weight`, of dims `c(dim, n)`, for
#'   integer indices counting from 0 (an input declared with [zl_int()]);
#'   the result has dims `c(dim, dim(idx))`. Initialised as N(0, 1).
#' * `zl_norm()`: layer or RMS normalisation of each column, then, with
#'   `affine = TRUE`, a scale `<name>.weight` (ones) and, for layer
#'   normalisation, a shift `<name>.bias` (zeros).
#' * `zl_attention()`: multi-head self-attention with linear projections
#'   `<name>.q`, `.k`, `.v` and `.o`, scaled by `1 / sqrt(dim / heads)`,
#'   causal by default.
#' * `zl_mlp()`: `down(swiglu(gate(x), up(x)))`, or `down(silu(up(x)))`
#'   with `act = "silu"`, with linears `<name>.gate`, `.up` and `.down`.
#' * `zl_block()`: a pre-norm transformer block,
#'   `h = x + attn(norm1(x))`, then `h + mlp(norm2(h))`, with layer norms.
#' * `zl_adaln()`: adaptive layer normalisation's modulation,
#'   `function(x, cond) x * (1 + scale(cond)) + shift(cond)`, with linears
#'   `<name>.scale` and `<name>.shift` from `cond_dim` to `dim`.
#' * `zl_time_embedding()`: no parameters; the sinusoidal embedding of
#'   times `t` of dims `c(1, ...)`: `cos(t f)` then `sin(t f)` for the
#'   `dim / 2` frequencies `f_k = max_period^(-k / (dim / 2))`.
#'
#' @param p A parameter store.
#' @param name The prefix of the parameters' names.
#' @param n_in,n_out Input and output widths.
#' @param bias Whether to add a bias.
#' @param n The number of rows of an embedding table.
#' @param dim The feature width.
#' @param type `"layer"` or `"rms"`.
#' @param affine Whether a normalisation has a learnt scale (and shift).
#' @param eps Added to the variance, a positive number.
#' @param heads The number of attention heads; must divide `dim`.
#' @param causal Whether position `i` attends only to positions up to `i`.
#' @param hidden The MLP's inner width.
#' @param act `"swiglu"` or `"silu"`.
#' @param cond_dim The width of the conditioning tensor.
#' @param t A tensor of times, of dims `c(1, ...)`.
#' @param max_period The longest period of the embedding.
#' @return A function of tensors, except `zl_time_embedding()`, which
#'   returns a tensor.
#' @name zl-modules
#' @examples
#' p <- zl_params()
#' emb <- zl_linear(p, "in", n_in = 3, n_out = 8)
#' blk <- zl_block(p, "b1", dim = 8, heads = 2)
#' g <- zl_graph(function(x) blk(emb(x$x)), p, inputs = list(x = c(3, 5, 2)))
#' dim(zl_forward(g, list(x = array(0.1, c(3, 5, 2))))$output)
NULL

zlm_check_width <- function(x, what) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x < 1 || x != floor(x)) {
    zlm_abort(sprintf("`%s` must be a positive whole number.", what), "zulm_invalid_argument")
  }
  as.integer(x)
}

#' @rdname zl-modules
#' @export
zl_linear <- function(p, name, n_in, n_out, bias = TRUE) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  n_in <- zlm_check_width(n_in, "n_in")
  n_out <- zlm_check_width(n_out, "n_out")
  w <- paste0(name, ".weight")
  b <- paste0(name, ".bias")
  bound <- 1 / sqrt(n_in)
  zl_param(p, w, c(n_in, n_out), init = zl_init_uniform(bound))
  if (isTRUE(bias)) {
    zl_param(p, b, n_out, init = zl_init_uniform(bound))
  }
  function(x) {
    x <- zlm_as_tensor(x)
    y <- zl_crossprod(zl_param_tensor(p, w), x)
    if (isTRUE(bias)) y <- y + zl_param_tensor(p, b)
    y
  }
}

#' @rdname zl-modules
#' @export
zl_embedding <- function(p, name, n, dim) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  n <- zlm_check_width(n, "n")
  dim <- zlm_check_width(dim, "dim")
  w <- paste0(name, ".weight")
  zl_param(p, w, c(dim, n), init = zl_init_normal(1))
  function(idx) {
    idx <- zlm_as_tensor(idx, "idx")
    d <- base::dim(idx)
    flat <- if (length(d) > 1L) zlm_op("reshape", idx, ints = c(prod(d), 1, 1, 1), rank = 1L) else idx
    rows <- zl_get_rows(zl_param_tensor(p, w), flat)
    if (length(d) > 1L) zl_reshape(rows, c(dim, d)) else rows
  }
}

#' @rdname zl-modules
#' @export
zl_norm <- function(p, name, dim, type = c("layer", "rms"), affine = TRUE, eps = 1e-5) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  dim <- zlm_check_width(dim, "dim")
  type <- match.arg(type)
  zlm_check_number(eps, "eps")
  w <- paste0(name, ".weight")
  b <- paste0(name, ".bias")
  if (isTRUE(affine)) {
    zl_param(p, w, dim, init = zl_init_ones())
    if (type == "layer") zl_param(p, b, dim, init = zl_init_zeros())
  }
  function(x) {
    x <- zlm_as_tensor(x)
    y <- if (type == "layer") zl_layer_norm(x, eps) else zl_rms_norm(x, eps)
    if (isTRUE(affine)) {
      y <- y * zl_param_tensor(p, w)
      if (type == "layer") y <- y + zl_param_tensor(p, b)
    }
    y
  }
}

#' @rdname zl-modules
#' @export
zl_attention <- function(p, name, dim, heads = 1L, causal = TRUE) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  dim <- zlm_check_width(dim, "dim")
  heads <- zlm_check_width(heads, "heads")
  if (dim %% heads != 0L) {
    zlm_abort("`heads` must divide `dim`.", "zulm_invalid_argument")
  }
  dh <- dim %/% heads
  lq <- zl_linear(p, paste0(name, ".q"), dim, dim)
  lk <- zl_linear(p, paste0(name, ".k"), dim, dim)
  lv <- zl_linear(p, paste0(name, ".v"), dim, dim)
  lo <- zl_linear(p, paste0(name, ".o"), dim, dim)
  function(x) {
    x <- zlm_as_tensor(x)
    d <- base::dim(x)
    n <- if (length(d) >= 2L) d[[2L]] else 1L
    b <- if (length(d) >= 3L) prod(d[-(1:2)]) else 1L
    split <- function(t) zl_permute(zl_reshape(t, c(dh, heads, n, b)), c(1, 3, 2, 4))
    q <- split(lq(x))
    k <- split(lk(x))
    v <- split(lv(x))
    s <- zl_crossprod(k, q)                        # c(n_k, n_q, heads, b)
    if (isTRUE(causal)) s <- zl_causal_mask(s)
    a <- zl_softmax(s, scale = 1 / sqrt(dh))
    o <- zl_crossprod(zl_permute(v, c(2, 1, 3, 4)), a)   # c(dh, n_q, heads, b)
    o <- zl_reshape(zl_permute(o, c(1, 3, 2, 4)), c(dim, n, b))
    y <- lo(o)
    zl_reshape(y, d)
  }
}

#' @rdname zl-modules
#' @export
zl_mlp <- function(p, name, dim, hidden = 4 * dim, act = c("swiglu", "silu")) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  dim <- zlm_check_width(dim, "dim")
  hidden <- zlm_check_width(hidden, "hidden")
  act <- match.arg(act)
  up <- zl_linear(p, paste0(name, ".up"), dim, hidden)
  down <- zl_linear(p, paste0(name, ".down"), hidden, dim)
  gate <- if (act == "swiglu") zl_linear(p, paste0(name, ".gate"), dim, hidden)
  function(x) {
    x <- zlm_as_tensor(x)
    h <- if (act == "swiglu") zl_swiglu(gate(x), up(x)) else zl_silu(up(x))
    down(h)
  }
}

#' @rdname zl-modules
#' @export
zl_block <- function(p, name, dim, heads, causal = TRUE, hidden = 4 * dim) {
  n1 <- zl_norm(p, paste0(name, ".norm1"), dim)
  attn <- zl_attention(p, paste0(name, ".attn"), dim, heads, causal)
  n2 <- zl_norm(p, paste0(name, ".norm2"), dim)
  mlp <- zl_mlp(p, paste0(name, ".mlp"), dim, hidden)
  function(x) {
    x <- zlm_as_tensor(x)
    h <- x + attn(n1(x))
    h + mlp(n2(h))
  }
}

#' @rdname zl-modules
#' @export
zl_adaln <- function(p, name, dim, cond_dim) {
  zlm_check_params(p)
  zlm_check_string(name, "name")
  scale <- zl_linear(p, paste0(name, ".scale"), cond_dim, dim)
  shift <- zl_linear(p, paste0(name, ".shift"), cond_dim, dim)
  function(x, cond) {
    x <- zlm_as_tensor(x)
    cond <- zlm_as_tensor(cond, "cond")
    x * (1 + scale(cond)) + shift(cond)
  }
}

#' @rdname zl-modules
#' @export
zl_time_embedding <- function(t, dim, max_period = 10000) {
  t <- zlm_as_tensor(t, "t")
  dim <- zlm_check_width(dim, "dim")
  if (dim %% 2L != 0L) {
    zlm_abort("`dim` must be even.", "zulm_invalid_argument")
  }
  zlm_check_number(max_period, "max_period")
  d <- base::dim(t)
  if (d[[1L]] != 1L) {
    zlm_abort("`t` must have a first dimension of 1.", "zulm_shape_error")
  }
  half <- dim %/% 2L
  freqs <- zlm_const(exp(-log(max_period) * seq(0, half - 1) / half), half)
  rest <- if (length(d) > 1L) d[-1L] else integer()
  args <- zlm_repeat(t, c(half, rest)) * freqs
  zl_concat(zl_cos(args), zl_sin(args), 1)
}

# A constant f32 tensor inside build().
zlm_const <- function(values, dim) {
  b <- zlm_current_graph()
  dim <- zlm_check_dim(dim)
  ptr <- zlm_call("zlm_const", b$ptr, as.double(values), c(dim, rep(1, 4L - length(dim))))
  zlm_new_tensor(ptr, b$ptr, length(dim))
}

# x repeated to dims `dim` (each a multiple of x's).
zlm_repeat <- function(x, dim) {
  x <- zlm_as_tensor(x)
  d4 <- c(dim, rep(1, 4L - length(dim)))
  zlm_op("repeat", x, ints = d4, rank = length(dim), call = sys.call())
}
