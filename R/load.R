# Variant loading (design §13.1).
#
# The registered DLL, zulm, is a probe: it reports the variants this CPU
# can run, best first. .onLoad loads the first one that is installed (or
# the one `options(zulm.variant =)` names) from the directory the probe
# was loaded from, which is libs/ when installed and src/ under
# devtools::load_all(), and looks its routines up once.

.zlm <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  zlm_load_variant(getOption("zulm.variant"))
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

zlm_load_variant <- function(requested = NULL) {
  zlm_unload_variant()
  dir <- zlm_variant_dir()
  installed <- zlm_installed_variants(dir)
  supported <- .Call(C_zlm_cpu_variants)
  variant <- zlm_choose_variant(requested, supported, installed)
  path <- file.path(dir, paste0("zulm_", variant, .Platform$dynlib.ext))
  dll <- dyn.load(path, local = TRUE, now = TRUE)
  .zlm$dll <- dll
  .zlm$path <- path
  .zlm$variant <- variant
  .zlm$supported <- supported
  .zlm$installed <- installed
  .zlm$routines <- getDLLRegisteredRoutines(dll)$.Call
  invisible(variant)
}

zlm_unload_variant <- function() {
  if (!is.null(.zlm$path)) {
    try(dyn.unload(.zlm$path), silent = TRUE)
  }
  rm(list = ls(.zlm), envir = .zlm)
}

# Calls a routine of the loaded variant.
zlm_call <- function(name, ...) {
  if (is.null(.zlm$routines[[name]])) {
    zlm_abort(
      sprintf("No zulm variant is loaded, or it lacks '%s'.", name),
      "zulm_internal_error"
    )
  }
  zlm_dot_call(.zlm$routines[[name]], ...)
}

# The routines belong to the variant shared object .onLoad chooses, so
# they exist only at run time. R's foreign-function check tries to resolve
# every literal .Call() when the package is checked and cannot resolve
# these; calling through an alias keeps it from trying.
zlm_dot_call <- .Call
