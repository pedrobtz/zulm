#' Initialisers for parameters
#'
#' Each returns a description of how [zl_param()] fills a new parameter.
#' Random draws come from 'zurand' under a key derived from the session key
#' and the parameter's name, so they do not depend on the order in which
#' parameters are declared, and adding a parameter does not change the
#' values of the others.
#'
#' @param sd Standard deviation of a normal distribution with mean zero.
#' @param a Half-width of a uniform distribution on `(-a, a)`.
#' @return An object of class `zl_init`.
#' @export
#' @examples
#' p <- zl_params()
#' zl_param(p, "w", c(4, 3), init = zl_init_normal(sd = 0.1))
#' zl_param(p, "b", 3, init = zl_init_zeros())
#' zl_param_get(p, "b")
zl_init_normal <- function(sd = 0.02) {
  zlm_check_number(sd, "sd", lower = 0)
  structure(list(kind = "normal", sd = as.double(sd)), class = "zl_init")
}

#' @rdname zl_init_normal
#' @export
zl_init_uniform <- function(a) {
  zlm_check_number(a, "a", lower = 0)
  structure(list(kind = "uniform", a = as.double(a)), class = "zl_init")
}

#' @rdname zl_init_normal
#' @export
zl_init_zeros <- function() {
  structure(list(kind = "zeros"), class = "zl_init")
}

#' @rdname zl_init_normal
#' @export
zl_init_ones <- function() {
  structure(list(kind = "ones"), class = "zl_init")
}

#' @export
print.zl_init <- function(x, ...) {
  desc <- switch(
    x$kind,
    normal = sprintf("normal(0, sd = %s)", format(x$sd)),
    uniform = sprintf("uniform(-%s, %s)", format(x$a), format(x$a)),
    x$kind
  )
  cat("<zl_init> ", desc, "\n", sep = "")
  invisible(x)
}

# Values for a parameter of n elements named `name` (design §7.2): the
# stream key is the session key folded with the name, the step key that
# folded with step 0, and element i is position i of its stream.
zlm_init_values <- function(init, name, n) {
  switch(
    init$kind,
    zeros = NULL,
    ones = rep(1, n),
    normal = {
      if (init$sd == 0) {
        NULL
      } else {
        zurand::rng_normal(zlm_step_key(name, 0L), n, mean = 0, sd = init$sd)
      }
    },
    uniform = {
      if (init$a == 0) {
        NULL
      } else {
        zurand::rng_uniform(zlm_step_key(name, 0L), n, min = -init$a, max = init$a)
      }
    }
  )
}
