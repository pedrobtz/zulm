# Random numbers (design §7.2). Every draw is a position of a zurand stream:
# the stream key is the session key folded with a name, the step key that
# folded with a step number, and element i is position offset + i. Values
# are rounded to f32: uniforms towards zero, normals to nearest.

.zlm_rng <- new.env(parent = emptyenv())

#' The session's random-number key
#'
#' Every random number zulm draws (parameter initialisation, noise inside
#' graphs, `zl_rng_uniform()` and `zl_rng_normal()`) comes from 'zurand'
#' under a key derived from the session key and a name. `zl_seed()` sets
#' the session key from a whole number, or from a key made with
#' [zurand::rng_key()] or [zurand::rng_fold()]; with no argument it
#' returns the current key. Until it is set, the key is drawn once from
#' R's generator, so [set.seed()] before the first draw governs a session.
#'
#' @param seed `NULL` to get the key; a whole number, or a zurand
#'   `rng_key` of length one, to set it.
#' @return The session key; when setting, the previous key, invisibly.
#' @export
#' @examples
#' old <- zl_seed(42)
#' zl_rng_normal(3, "example")
#' zl_seed(old)
zl_seed <- function(seed = NULL) {
  old <- zlm_session_key()
  if (is.null(seed)) {
    return(old)
  }
  if (inherits(seed, "rng_key")) {
    if (length(seed) != 1L) {
      zlm_abort("`seed` must be a single key.", "zulm_invalid_argument")
    }
    key <- seed
  } else {
    if (!is.numeric(seed) || length(seed) != 1L || is.na(seed) || seed != floor(seed) ||
      abs(seed) > .Machine$integer.max) {
      zlm_abort("`seed` must be a whole number or a zurand rng_key.", "zulm_invalid_argument")
    }
    key <- zurand::rng_key(as.integer(seed), engine = "philox4x64")
  }
  .zlm_rng$key <- key
  invisible(old)
}

zlm_session_key <- function() {
  if (is.null(.zlm_rng$key)) {
    .zlm_rng$key <- zurand::rng_key_from_r(engine = "philox4x64")
  }
  .zlm_rng$key
}

# The key of a stream: the session key folded with the stream's name.
zlm_stream_key <- function(stream) {
  zurand::rng_fold(zlm_session_key(), stream)
}

# The key of a stream at a step, exactly as zurand's C fold_int() derives it.
zlm_step_key <- function(stream, step) {
  zurand::rng_fold(zlm_stream_key(stream), as.integer(step))
}

#' Random draws, in R
#'
#' `zl_rng_uniform()` and `zl_rng_normal()` return the single-precision
#' values that [zl_noise_uniform()] and [zl_noise_normal()] give inside a
#' graph for the same stream, step and positions: positions `offset` to
#' `offset + prod(dim) - 1` of the stream named `stream` at step `step`.
#' Uniforms lie strictly inside (0, 1); normals have mean 0 and standard
#' deviation 1.
#'
#' The values are 'zurand''s, rounded to single precision: a uniform is
#' rounded towards zero (so it never becomes 1), a normal to nearest. With
#' the session key `k`, a uniform equals `zurand::rng_uniform(
#' zurand::rng_fold(zurand::rng_fold(k, stream), step), n, offset = offset)`
#' so rounded.
#'
#' @param dim Dimensions: one to four positive whole numbers.
#' @param stream The stream's name.
#' @param step The step number (a trainer's step count; 0 for none).
#' @param offset The position of the first value.
#' @return A double vector or array of dimensions `dim`.
#' @export
#' @examples
#' zl_seed(1)
#' zl_rng_uniform(c(2, 3), "noise")
#' zl_rng_normal(4, "noise", step = 2)
zl_rng_uniform <- function(dim, stream, step = 0, offset = 0) {
  zlm_rng(dim, stream, step, offset, 0L)
}

#' @rdname zl_rng_uniform
#' @export
zl_rng_normal <- function(dim, stream, step = 0, offset = 0) {
  zlm_rng(dim, stream, step, offset, 1L)
}

zlm_check_position <- function(x, what) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x < 0 || x != floor(x) || x > 2^53) {
    zlm_abort(sprintf("`%s` must be a non-negative whole number.", what), "zulm_invalid_argument")
  }
  as.double(x)
}

zlm_rng <- function(dim, stream, step, offset, dist) {
  dim <- zlm_check_dim(dim)
  zlm_check_string(stream, "stream")
  step <- zlm_check_position(step, "step")
  offset <- zlm_check_position(offset, "offset")
  if (step > .Machine$integer.max) {
    zlm_abort("`step` must be below 2^31.", "zulm_invalid_argument")
  }
  out <- zlm_call("zlm_rng_draw", zlm_stream_key(stream), step, offset, prod(dim), dist)
  if (length(dim) > 1L) dim(out) <- dim
  out
}

#' Random draws inside a graph
#'
#' `zl_noise_normal()` and `zl_noise_uniform()` are tensors of draws from
#' the stream named `stream`, made anew at each training step: at step
#' `s` of a trainer (1 for the first) element `i` (in R's order, counting
#' from 0) is position `offset + i` of the stream at step `s`. Outside
#' training, [zl_forward()] uses the trainer's last step, or 0. The values
#' do not depend on the thread count, and equal [zl_rng_normal()] and
#' [zl_rng_uniform()] for the same stream, step and positions. Use
#' `offset` to keep the draws of a split batch the same: a second half of
#' a batch starts at the first half's size.
#'
#' They have no gradient, and none is needed: nothing flows back into a
#' draw.
#'
#' @inheritParams zl_rng_uniform
#' @return A tensor.
#' @export
#' @examples
#' p <- zl_params()
#' g <- zl_graph(function(x) x$a + zl_noise_normal(c(2, 3), "eps"), p,
#'               inputs = list(a = c(2, 3)))
#' zl_forward(g, list(a = matrix(0, 2, 3)))$output
zl_noise_normal <- function(dim, stream, offset = 0) {
  zlm_noise(dim, stream, offset, 1L, sys.call())
}

#' @rdname zl_noise_normal
#' @export
zl_noise_uniform <- function(dim, stream, offset = 0) {
  zlm_noise(dim, stream, offset, 0L, sys.call())
}

zlm_noise <- function(dim, stream, offset, dist, call) {
  b <- zlm_current_graph()
  dim <- zlm_check_dim(dim)
  zlm_check_string(stream, "stream")
  offset <- zlm_check_position(offset, "offset")
  ptr <- zlm_call(
    "zlm_noise", b$ptr, zlm_stream_key(stream), dist, offset,
    c(dim, rep(1, 4L - length(dim))), call
  )
  zlm_new_tensor(ptr, b$ptr, length(dim))
}

# zurand's C table must be version 3 or later (design §7.2).
zlm_check_zurand <- function(version) {
  if (!isTRUE(zlm_dot_call(.zlm$routines$zlm_zurand_version_ok, as.integer(version)))) {
    zlm_abort(
      sprintf(
        "zulm needs zurand's C table version 3 or later; the installed zurand provides version %d. Update zurand.",
        as.integer(version)
      ),
      "zulm_companion_error"
    )
  }
  invisible(version)
}
