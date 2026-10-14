# The session key (design §7.2). Drawn once from R's generator the first
# time zulm needs a random number, so set.seed() before that governs a
# session; Stage 5's zl_seed() sets it explicitly.

.zlm_rng <- new.env(parent = emptyenv())

zlm_session_key <- function() {
  if (is.null(.zlm_rng$key)) {
    .zlm_rng$key <- zurand::rng_key_from_r(engine = "philox4x64")
  }
  .zlm_rng$key
}

# The key of a stream (a name) at a step: fold the name, then the step,
# exactly as zurand's C fold_int() does for the step (design §7.2).
zlm_step_key <- function(stream, step) {
  zurand::rng_fold(zurand::rng_fold(zlm_session_key(), stream), as.integer(step))
}
