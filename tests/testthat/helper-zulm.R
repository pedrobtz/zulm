# zulm's internal .Call wrapper, for routines the tests reach directly.
zlm_call <- function(...) getFromNamespace("zlm_call", "zulm")(...)

# Rounds doubles to the nearest single-precision value, as zulm stores f32.
f32 <- function(x) {
  readBin(writeBin(as.double(x), raw(), size = 4), "double", n = length(x), size = 4)
}

# A store with a few parameters of known values.
make_store <- function() {
  p <- zl_params()
  zl_param(p, "w", c(3, 2), init = zl_init_zeros())
  zl_param_set(p, "w", matrix(c(1.5, -2, 0.25, 3, -0.125, 7), 3, 2))
  zl_param(p, "b", 2, init = zl_init_ones())
  p
}

# Runs build() once on inputs of the given dims (numeric, or zl_int()) and
# returns zl_forward()'s outputs for `data`.
run_graph <- function(build, inputs, data, p = zl_params()) {
  g <- zl_graph(build, p, inputs)
  zl_forward(g, data)
}

# Pads a dim vector with 1s to length 4.
dim4 <- function(d) c(d, rep(1L, 4L - length(d)))

# R's version of ggml's broadcast: b repeated (tiled) to dims d, where every
# dimension of b divides the matching one of d.
bcast <- function(b, d) {
  db <- dim4(if (is.null(dim(b))) length(b) else dim(b))
  d4 <- dim4(d)
  idx <- arrayInd(seq_len(prod(d4)), d4) - 1L
  bi <- sweep(idx, 2L, db, `%%`)
  lin <- bi[, 1] + db[1] * (bi[, 2] + db[2] * (bi[, 3] + db[3] * bi[, 4])) + 1
  array(as.vector(b)[lin], d)
}

# A random shape of rank 1 to 4 with small dimensions, and a random shape
# that broadcasts to it (each dimension 1, the full size, or a divisor).
rand_dims <- function(rank = sample(1:4, 1), max = 5) {
  as.integer(sample(1:max, rank, replace = TRUE))
}
bcast_dims <- function(d) {
  vapply(d, function(n) {
    divs <- which(n %% seq_len(n) == 0)
    as.integer(sample(c(1L, n, divs), 1))
  }, integer(1))
}

rand_array <- function(d, lo = -2, hi = 2) {
  x <- f32(stats::runif(prod(d), lo, hi))
  if (length(d) > 1L) array(x, d) else x
}

# Expected f32 accuracy: relative to the values' scale.
expect_f32 <- function(object, expected, tol = 1e-5) {
  expect_equal(object, expected, tolerance = tol, ignore_attr = FALSE)
}

# Gradient check (design §11): builds loss = sum(f(params, inputs) * w) with
# a fixed random weight w, and compares zl_grad() with central finite
# differences of the f32 loss, for every parameter.
grad_check <- function(f, params, inputs = list(), data = list(), h = 1e-2, tol = 2e-2) {
  p <- zl_params()
  for (nm in names(params)) {
    v <- params[[nm]]
    d <- if (is.null(dim(v))) length(v) else dim(v)
    zl_param(p, nm, d, init = zl_init_zeros())
    zl_param_set(p, nm, v)
  }
  probe <- zl_graph(function(x) f(x, p), p, c(inputs, list(.one = 1)))
  out_dim <- dim(zl_forward(probe, c(data, list(.one = 1)))$output)
  if (is.null(out_dim)) out_dim <- length(zl_forward(probe, c(data, list(.one = 1)))$output)
  w <- rand_array(out_dim, -1, 1)
  g <- zl_graph(
    function(x) list(loss = zl_sum(f(x, p) * x$.w)),
    p, c(inputs, list(.one = 1, .w = out_dim)), loss = "loss"
  )
  full <- c(data, list(.one = 1, .w = w))
  tr <- zl_trainer(g, zl_sgd(lr = 1e-3))
  loss_at <- function() zl_forward(g, full)$loss
  for (nm in names(params)) {
    loss_at()
    got <- zl_grad(tr, nm)
    v <- zl_param_get(p, nm)
    fd <- v
    for (k in seq_along(v)) {
      up <- v
      up[k] <- up[k] + h
      zl_param_set(p, nm, up)
      lp <- loss_at()
      dn <- v
      dn[k] <- dn[k] - h
      zl_param_set(p, nm, dn)
      lm <- loss_at()
      fd[k] <- (lp - lm) / (2 * h)
    }
    zl_param_set(p, nm, v)
    scale <- max(1, max(abs(fd)))
    expect_lt(max(abs(as.vector(got) - as.vector(fd))) / scale, tol, label = sprintf("gradient of %s", nm))
  }
  invisible(TRUE)
}

# Softmax of each column (the first dimension), as zl_softmax() computes it.
softmax_cols <- function(x) {
  m <- matrix(x, dim(x)[1])
  e <- exp(sweep(m, 2, apply(m, 2, max)))
  array(sweep(e, 2, colSums(e), "/"), dim(x))
}

# Rounds positive doubles to f32 towards zero, zulm's rule for uniforms.
f32_toward_zero <- function(x) {
  r <- f32(x)
  up <- r > x
  if (any(up)) {
    bits <- readBin(writeBin(r[up], raw(), size = 4), "integer", size = 4, n = sum(up))
    r[up] <- readBin(writeBin(bits - 1L, raw(), size = 4), "double", size = 4, n = sum(up))
  }
  r
}

# Sets the session's threads for the calling test, and restores them.
local_threads <- function(n, envir = parent.frame()) {
  old <- zl_threads(n)
  withr::defer(zl_threads(old), envir = envir)
  invisible(old)
}

# The bytes a store's tensors take.
zlm_store_bytes <- function(p) {
  getFromNamespace("zlm_call", "zulm")("zlm_store_info", p$ptr)$bytes
}

# The C table's fixture consumer (tests/testthat/zulmclient), built once per
# session against the installed zulm.h and loaded; its namespace.
.zulm_client <- new.env(parent = emptyenv())
zulm_client <- function() {
  if (is.null(.zulm_client$lib)) {
    inc <- system.file("include", package = "zulm")
    if (!nzchar(inc) || !file.exists(file.path(inc, "zulm.h"))) {
      stop("zulm.h not found under system.file(\"include\", package = \"zulm\")")
    }
    src <- file.path(tempfile("zulmclient-src"), "zulmclient")
    dir.create(src, recursive = TRUE)
    file.copy(list.files(test_path("zulmclient"), full.names = TRUE), src, recursive = TRUE)
    writeLines(
      c(
        sprintf('PKG_CPPFLAGS = -I"%s"', normalizePath(inc, "/")),
        "PKG_CFLAGS = $(SHLIB_OPENMP_CFLAGS)",
        "PKG_LIBS = $(SHLIB_OPENMP_CFLAGS)"
      ),
      file.path(src, "src", "Makevars")
    )
    lib <- tempfile("zulmclient-lib")
    dir.create(lib)
    out <- suppressWarnings(system2(
      file.path(R.home("bin"), "R"),
      c(
        "CMD", "INSTALL", "--no-test-load", "--no-multiarch",
        paste0("--library=", shQuote(lib)), shQuote(src)
      ),
      stdout = TRUE, stderr = TRUE
    ))
    if (!is.null(attr(out, "status")) && attr(out, "status") != 0) {
      stop("building the C table fixture failed:\n", paste(out, collapse = "\n"))
    }
    .zulm_client$lib <- lib
  }
  loadNamespace("zulmclient", lib.loc = .zulm_client$lib)
}
