# The stand-in namespace of the fuzz targets (tools/fuzz/fuzz.c): the
# package's R code, sourced into an environment registered as "zulm",
# calling the routines the harness registered on the embedding DLL.
# Registers env as namespace `name`, with what `::` and getNamespace()
# look up in a namespace's metadata.
zlm_fuzz_namespace <- function(name, env) {
  info <- new.env(parent = baseenv())
  info$spec <- c(name = name, version = "0.0.0")
  info$exports <- new.env(parent = baseenv())
  for (n in ls(env, all.names = TRUE)) assign(n, n, envir = info$exports)
  info$imports <- list(base = TRUE)
  info$path <- ""
  info$dynlibs <- character()
  info$lazydata <- new.env(parent = baseenv())
  info$S3methods <- matrix(NA_character_, 0L, 4L)
  env$.__NAMESPACE__. <- info
  env$.__S3MethodsTable__. <- new.env(parent = baseenv())
  .Internal(registerNamespace(name, env))
  invisible(env)
}

zlm_fuzz_setup <- function(variant) {
  # zurand's R functions, as stubs: the embedded R has no packages. Draws
  # are constants, and zurand's C table is absent, so in-graph noise is a
  # zulm_companion_error.
  zr <- new.env(parent = baseenv())
  zr$rng_key <- function(...) structure(list(1), class = "rng_key")
  zr$rng_key_from_r <- zr$rng_key
  zr$rng_fold <- function(key, data) key
  zr$rng_normal <- function(key, n, ...) rep(0.01, n)
  zr$rng_uniform <- function(key, n, ...) rep(0.5, n)
  zr$format.rng_key <- function(x, ...) "rng_key[fuzz]"
  zlm_fuzz_namespace("zurand", zr)
  registerS3method("format", "rng_key", zr$format.rng_key, envir = zr)

  ns <- new.env(parent = baseenv())
  for (f in sort(list.files("R", "[.]R$", full.names = TRUE))) {
    sys.source(f, ns, keep.source = FALSE)
  }
  sys.source("tools/fuzz/drive.R", ns, keep.source = FALSE)
  zlm_fuzz_namespace("zulm", ns)
  for (l in grep("^S3method\\(", readLines("NAMESPACE"), value = TRUE)) {
    a <- strsplit(sub("^S3method\\((.*)\\)$", "\\1", l), ",")[[1]]
    registerS3method(a[[1]], a[[2]], get(paste0(a[[1]], ".", a[[2]]), ns), envir = ns)
  }
  dll <- getLoadedDLLs()[["(embedding)"]]
  ns$.zlm$dll <- dll
  ns$.zlm$variant <- variant
  ns$.zlm$routines <- getDLLRegisteredRoutines(dll)$.Call
  ns$.zlm$path <- file.path(getwd(), "fuzz")
  ns$.zlm$supported <- variant
  ns$.zlm$installed <- variant
  ns$.zlm$zurand_version <- NA_integer_
  options(zulm.threads = 1L)
  for (nm in c("fuzz_graph", "fuzz_files", "fuzz_arrow")) {
    assign(nm, get(nm, ns), globalenv())
  }
  invisible(ns)
}
