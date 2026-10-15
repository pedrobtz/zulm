#' Optimisers
#'
#' `zl_adamw()` is AdamW, `zl_sgd()` stochastic gradient descent, both as
#' 'ggml-opt' implements them. Weight decay is decoupled from the gradient:
#' each step multiplies the parameters by `1 - lr * weight_decay`.
#'
#' @param lr The learning rate, positive.
#' @param beta1,beta2 AdamW's moment decay rates, in `[0, 1]`.
#' @param eps AdamW's denominator offset, non-negative.
#' @param weight_decay In `[0, 1]`.
#' @return An object of class `zl_optimizer`.
#' @export
#' @examples
#' zl_adamw(lr = 3e-4, weight_decay = 0.01)
#' zl_sgd(lr = 0.1)
zl_adamw <- function(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0) {
  zlm_check_lr(lr)
  zlm_check_unit(beta1, "beta1")
  zlm_check_unit(beta2, "beta2")
  zlm_check_number(eps, "eps", lower = 0)
  zlm_check_unit(weight_decay, "weight_decay")
  structure(
    list(type = "adamw", lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, weight_decay = weight_decay),
    class = "zl_optimizer"
  )
}

#' @rdname zl_adamw
#' @export
zl_sgd <- function(lr = 1e-2, weight_decay = 0) {
  zlm_check_lr(lr)
  zlm_check_unit(weight_decay, "weight_decay")
  structure(
    list(type = "sgd", lr = lr, beta1 = 0, beta2 = 0, eps = 0, weight_decay = weight_decay),
    class = "zl_optimizer"
  )
}

#' @export
print.zl_optimizer <- function(x, ...) {
  if (x$type == "adamw") {
    cat(sprintf(
      "<zl_optimizer> AdamW, lr %s, betas %s and %s, eps %s, weight decay %s\n",
      format(x$lr), format(x$beta1), format(x$beta2), format(x$eps), format(x$weight_decay)
    ))
  } else {
    cat(sprintf("<zl_optimizer> SGD, lr %s, weight decay %s\n", format(x$lr), format(x$weight_decay)))
  }
  invisible(x)
}

zlm_check_lr <- function(lr) {
  zlm_check_number(lr, "lr")
  if (lr <= 0) {
    zlm_abort("`lr` must be positive.", "zulm_invalid_argument")
  }
  invisible(lr)
}

zlm_check_unit <- function(x, what) {
  zlm_check_number(x, what, lower = 0)
  if (x > 1) {
    zlm_abort(sprintf("`%s` must lie in [0, 1].", what), "zulm_invalid_argument")
  }
  invisible(x)
}

#' Train a graph
#'
#' `zl_trainer()` prepares a graph with a `loss` for training with an
#' optimiser: it checks that every operation between a parameter and the
#' loss has a gradient (`zulm_unsupported_backward` names the one that has
#' not, and the call that made it), and builds the gradient and optimiser
#' graphs once, through 'ggml-opt'. Every f32 parameter the loss depends on
#' is trained; half-precision parameters cannot be.
#'
#' `zl_step()` writes a batch into the graph's inputs, computes the loss and
#' the gradients, and, every `accumulate` calls, updates the parameters. It
#' returns the loss of the batch; a loss that is not finite gives a
#' `zulm_nonfinite_warning`. With `clip_norm`, each update first scales all
#' gradients together so that their joint L2 norm is at most `clip_norm`.
#'
#' `zl_grad()` returns the gradient of the loss with respect to one
#' parameter, at the parameters' current values, on the inputs last written
#' by `zl_step()` or [zl_forward()]. `zl_reset()` zeroes the optimiser's
#' state (AdamW's moments and step count).
#'
#' An optimiser's state is not saved by [zl_save()], and if a step is
#' interrupted the state is undefined: call `zl_reset()` before going on.
#'
#' @param g A graph from [zl_graph()] with a `loss`.
#' @param opt An optimiser from [zl_adamw()] or [zl_sgd()].
#' @param accumulate The number of batches whose gradients are summed
#'   before each update.
#' @param clip_norm `NULL`, or the largest joint L2 norm of the gradients
#'   at an update.
#' @param tr A trainer.
#' @param data A named list of arrays for the graph's inputs, as for
#'   [zl_forward()], or an iterator from [zl_arrow_batches()].
#' @param lr The learning rate of this step; `NULL` for the optimiser's.
#' @param name A parameter's name.
#' @return `zl_trainer()` returns a trainer; `zl_step()` the loss, a number
#'   (`NULL` when `data` is an iterator whose stream has ended);
#'   `zl_grad()` an array of the parameter's dimensions; `zl_reset()` the
#'   trainer, invisibly.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(3, 1), init = zl_init_zeros())
#' g <- zl_graph(
#'   function(x) list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), x$x), x$y)),
#'   p,
#'   inputs = list(x = c(3, 8), y = c(1, 8)),
#'   loss = "loss"
#' )
#' tr <- zl_trainer(g, zl_adamw(lr = 0.05))
#' set.seed(1)
#' x <- matrix(rnorm(24), 3, 8)
#' y <- matrix(c(1, -2, 0.5) %*% x, 1, 8)
#' for (i in 1:200) loss <- zl_step(tr, list(x = x, y = y))
#' loss
#' zl_param_get(p, "w")
zl_trainer <- function(g, opt, accumulate = 1L, clip_norm = NULL) {
  zlm_check_graph(g)
  zlm_maybe_gc()
  if (is.null(g$loss)) {
    zlm_abort("The graph has no `loss`; give zl_graph() one to train.", "zulm_invalid_argument")
  }
  if (zlm_call("zlm_graph_info", g$ptr)[[5L]] > 0L) {
    zlm_abort("A graph that writes state tensors is for inference; it cannot be trained.", "zulm_invalid_argument")
  }
  if (!inherits(opt, "zl_optimizer")) {
    zlm_abort("`opt` must come from zl_adamw() or zl_sgd().", "zulm_invalid_argument")
  }
  if (!is.numeric(accumulate) || length(accumulate) != 1L || is.na(accumulate) ||
    accumulate < 1 || accumulate != floor(accumulate) || accumulate > 1e6) {
    zlm_abort("`accumulate` must be a whole number from 1.", "zulm_invalid_argument")
  }
  clip <- 0
  if (!is.null(clip_norm)) {
    zlm_check_number(clip_norm, "clip_norm")
    if (clip_norm <= 0) {
      zlm_abort("`clip_norm` must be positive.", "zulm_invalid_argument")
    }
    if (!"p07-opt-global-norm-clipping.patch" %in% zulm_info()$ggml_patches) {
      zlm_abort(
        "`clip_norm` needs ggml patch P7 (global-norm clipping), which this build lacks.",
        "zulm_invalid_argument"
      )
    }
    clip <- clip_norm
  }
  loss <- zlm_call("zlm_graph_output", g$ptr, match(g$loss, g$outputs$names) - 1L)
  ptr <- zlm_call(
    "zlm_trainer_new", g$ptr, loss, opt$type,
    as.double(c(opt$lr, opt$beta1, opt$beta2, opt$eps, opt$weight_decay)),
    as.integer(accumulate), as.double(clip), zlm_threads()
  )
  structure(
    list(ptr = ptr, graph = g, opt = opt, accumulate = as.integer(accumulate), clip_norm = clip_norm),
    class = "zl_trainer"
  )
}

zlm_check_trainer <- function(tr) {
  if (!inherits(tr, "zl_trainer")) {
    zlm_abort("`tr` must be a trainer from zl_trainer().", "zulm_invalid_argument")
  }
  zlm_check_graph(tr$graph)
  invisible(tr)
}

#' @rdname zl_trainer
#' @export
zl_step <- function(tr, data, lr = NULL) {
  zlm_check_trainer(tr)
  if (is.null(lr)) {
    lr <- tr$opt$lr
  } else {
    zlm_check_lr(lr)
  }
  if (!zlm_write_inputs(tr$graph, data)) {
    return(NULL)
  }
  loss <- zlm_call("zlm_trainer_step", tr$ptr, as.double(lr))
  rl <- tr$graph$outputs$rowloss[[match(tr$graph$loss, tr$graph$outputs$names)]]
  if (!is.null(rl)) {
    # the fixed-order sum of per-row values (design §7.1)
    loss <- zlm_rowloss_sum(rl)
  }
  if (!is.finite(loss)) {
    zlm_warn(sprintf("The loss is %s.", format(loss)), "zulm_nonfinite_warning")
  }
  loss
}

#' @rdname zl_trainer
#' @export
zl_grad <- function(tr, name) {
  zlm_check_trainer(tr)
  zlm_check_string(name, "name")
  g <- tr$graph
  info <- zlm_param_info(g$store, name)
  loss <- zlm_call("zlm_graph_output", g$ptr, match(g$loss, g$outputs$names) - 1L)
  res <- zlm_call("zlm_grad", g$ptr, loss, enc2utf8(name), zlm_threads())
  out <- res[[2L]]
  if (length(info$dim) > 1L) {
    dim(out) <- info$dim
  }
  out
}

#' @rdname zl_trainer
#' @export
zl_reset <- function(tr) {
  zlm_check_trainer(tr)
  zlm_call("zlm_trainer_reset", tr$ptr)
  invisible(tr)
}

#' @export
print.zl_trainer <- function(x, ...) {
  evals <- tryCatch(zlm_call("zlm_trainer_evals", x$ptr), zulm_error = function(e) NA)
  cat(sprintf(
    "<zl_trainer> %s; loss '%s'; accumulate %d%s; %s batches so far\n",
    if (x$opt$type == "adamw") "AdamW" else "SGD", x$graph$loss, x$accumulate,
    if (is.null(x$clip_norm)) "" else sprintf("; clip_norm %s", format(x$clip_norm)),
    format(evals)
  ))
  invisible(x)
}
