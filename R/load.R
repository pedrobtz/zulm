# Variant loading (design §13.1).
#
# The registered DLL, zulm, is a probe: it reports the variants this CPU
# can run, best first. .onLoad loads the first one that is installed (or
# the one `options(zulm.variant =)` names) from the directory the probe
# was loaded from, which is libs/ when installed and src/ under
# devtools::load_all(), and looks its routines up once.

.zlm <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  zlm_torture_setup(Sys.getenv("ZULM_GCTORTURE"))
  zlm_load_variant(getOption("zulm.variant"))
  .zlm$zurand_version <- zlm_call("zlm_zurand_init")
  zlm_check_zurand(.zlm$zurand_version)
}

.onUnload <- function(libpath) {
  zlm_unload_variant()
}

zlm_variant_dir <- function() {
  # Installed: the variants sit beside the probe in libs/. Under
  # devtools::load_all(), pkgload loads a temporary copy of the probe, and
  # the variants are still in the source tree's src/.
  dir <- dirname(getLoadedDLLs()[["zulm"]][["path"]])
  if (length(zlm_installed_variants(dir)) == 0L) {
    src <- system.file("src", package = "zulm")
    if (nzchar(src) && length(zlm_installed_variants(src)) > 0L) {
      dir <- src
    }
  }
  dir
}

zlm_installed_variants <- function(dir) {
  ext <- .Platform$dynlib.ext
  files <- list.files(dir, pattern = paste0("^zulm_.*\\", ext, "$"))
  sub(paste0("^zulm_(.*)\\", ext, "$"), "\\1", files)
}

# A native build installs one variant, built for the installing CPU, and
# it is the only one usable; otherwise the variants this CPU supports.
zlm_usable_variants <- function(supported, installed) {
  if ("native" %in% installed) "native" else intersect(supported, installed)
}

zlm_choose_variant <- function(requested, supported, installed) {
  if (!is.null(requested)) {
    if (!is.character(requested) || length(requested) != 1L || is.na(requested)) {
      zlm_abort(
        "`options(zulm.variant =)` must be a single string.",
        "zulm_invalid_argument"
      )
    }
    usable <- zlm_usable_variants(supported, installed)
    if (!requested %in% usable) {
      zlm_abort_variant(requested, usable, installed)
    }
    return(requested)
  }
  usable <- zlm_usable_variants(supported, installed)
  if (length(usable) == 0L) {
    zlm_abort(
      "No installed zulm variant runs on this CPU.",
      "zulm_internal_error"
    )
  }
  usable[[1L]]
}

zlm_abort_variant <- function(requested, usable, installed) {
  zlm_abort(
    sprintf(
      "Variant '%s' cannot be loaded here; usable: %s; installed: %s.",
      requested,
      paste(usable, collapse = ", "),
      paste(installed, collapse = ", ")
    ),
    "zulm_invalid_argument"
  )
}

# Switching variants never unloads the previous one: objects it created may
# still be finalised, and their finalisers live in its shared object.
zlm_load_variant <- function(requested = NULL) {
  dir <- zlm_variant_dir()
  installed <- zlm_installed_variants(dir)
  supported <- .Call(C_zlm_cpu_variants)
  variant <- zlm_choose_variant(requested, supported, installed)
  path <- file.path(dir, paste0("zulm_", variant, .Platform$dynlib.ext))
  if (is.null(.zlm$loaded)) {
    .zlm$loaded <- list()
  }
  dll <- .zlm$loaded[[variant]]
  if (is.null(dll)) {
    # A namespace reloaded in the same session finds its variants loaded
    # already; loading one again would unload it under live objects.
    dll <- getLoadedDLLs()[[paste0("zulm_", variant)]]
    if (is.null(dll) || !identical(normalizePath(dll[["path"]]), normalizePath(path))) {
      dll <- dyn.load(path, local = TRUE, now = TRUE)
    }
    .zlm$loaded[[variant]] <- dll
  }
  .zlm$dll <- dll
  .zlm$path <- path
  .zlm$variant <- variant
  .zlm$supported <- supported
  .zlm$installed <- installed
  .zlm$routines <- getDLLRegisteredRoutines(dll)$.Call
  # the C table (design §8) is the loaded variant's
  zlm_call("zlm_api_register")
  invisible(variant)
}

# At namespace unload the variants stay loaded: a store that outlives the
# namespace is still finalised by its variant's code.
zlm_unload_variant <- function() {
  invisible(NULL)
}

# Calls a routine of the loaded variant.
zlm_call <- function(name, ...) {
  r <- .zlm$routines[[name]]
  if (is.null(r)) {
    zlm_abort(
      sprintf("No zulm variant is loaded, or it lacks '%s'.", name),
      "zulm_internal_error"
    )
  }
  on.exit(zlm_flush_log())
  if (!is.null(.zlm$torture)) {
    return(zlm_tortured_call(r, name, ...))
  }
  zlm_dot_call(r, ...)
}

# For the calls that build a graph, a few microseconds each and hundreds
# per graph: ggml's builders log nothing but the assertions that zulm turns
# into errors, so the log is left for the next zlm_call() to flush.
zlm_call_quiet <- function(name, ...) {
  if (!is.null(.zlm$torture)) {
    return(zlm_tortured_call(.zlm$routines[[name]], name, ...))
  }
  zlm_dot_call(.zlm$routines[[name]], ...)
}

# gctorture where it can find something (the gctorture CI job). A missing
# PROTECT bites only while C runs, so R's collector is forced only inside
# zulm's .Call()s, and only for the first calls of each routine with each
# combination of argument types: repeats in loops add time, not paths.
# An object C should have kept alive between calls is still caught, as the
# next tortured call collects it before C reads it.
#
#   ZULM_GCTORTURE=<step>[,<limit>]
#
# collects every `step` allocations inside the first `limit` calls of each
# (routine, argument types); a limit of 0 tortures every call. A step of 0
# tortures nothing and only records the calls, which tools/smoke-tests.R
# reads. Unset, as on CRAN, nothing is tortured or recorded.
zlm_torture_setup <- function(spec) {
  .zlm$torture <- NULL
  if (!nzchar(spec)) {
    return(invisible(NULL))
  }
  v <- suppressWarnings(as.integer(strsplit(spec, ",", fixed = TRUE)[[1L]]))
  if (length(v) < 1L || length(v) > 2L || anyNA(v) || v[[1L]] < 0L || (length(v) == 2L && v[[2L]] < 0L)) {
    zlm_abort("ZULM_GCTORTURE must be <step> or <step>,<limit> with step and limit >= 0.", "zulm_invalid_argument")
  }
  .zlm$torture <- list(
    step = v[[1L]],
    limit = if (length(v) == 2L) v[[2L]] else 0L,
    seen = new.env(parent = emptyenv())
  )
  invisible(NULL)
}

zlm_tortured_call <- function(r, name, ...) {
  t <- .zlm$torture
  if (t$limit > 0L || t$step == 0L) {
    key <- paste(c(name, vapply(list(...), typeof, "")), collapse = " ")
    n <- t$seen[[key]]
    n <- if (is.null(n)) 0L else n
    t$seen[[key]] <- n + 1L
    if (t$step == 0L || (t$limit > 0L && n >= t$limit)) {
      return(zlm_dot_call(r, ...))
    }
  }
  old <- gctorture2(t$step)
  on.exit(gctorture2(old))
  zlm_dot_call(r, ...)
}

# The routines belong to the variant shared object .onLoad chooses, so
# they exist only at run time. R's foreign-function check tries to resolve
# every literal .Call() when the package is checked and cannot resolve
# these; calling through an alias keeps it from trying.
zlm_dot_call <- .Call
