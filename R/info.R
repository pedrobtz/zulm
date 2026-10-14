#' Build and runtime information
#'
#' Reports the vendored ggml version, the variant loaded for this CPU, the
#' variants installed and usable, the native-build hooks recorded at
#' install time, and the CPU features ggml was compiled for.
#'
#' A variant is the whole compiled library built for one instruction-set
#' level: on x86-64 `"baseline"`, `"sse42"` and `"avx2"`, on arm64
#' `"neon"`, and `"native"` when the package was installed with the
#' `ZULM_CFLAGS` and `ZULM_CXXFLAGS` hooks set. The best usable variant is
#' loaded when the package is; set `options(zulm.variant =)` before loading
#' it to force another.
#'
#' @return A named list.
#' @export
#' @examples
#' info <- zulm_info()
#' info$variant
#' info$ggml_version
zulm_info <- function() {
  native <- zlm_call("zlm_native_info")
  dir <- dirname(.zlm$path)
  build <- file.path(dir, "zulm-build.dcf")
  hooks <- if (file.exists(build)) {
    as.list(read.dcf(build)[1L, ])
  } else {
    list(ZULM_CFLAGS = "", ZULM_CXXFLAGS = "")
  }
  patches <- strsplit(native$ggml_patches, ",", fixed = TRUE)[[1L]]
  c(
    list(
      ggml_version = native$ggml_version,
      ggml_commit = native$ggml_commit,
      ggml_patches = patches,
      variant = .zlm$variant,
      variants_usable = intersect(.zlm$supported, .zlm$installed),
      variants_installed = .zlm$installed,
      native_cflags = hooks$ZULM_CFLAGS,
      native_cxxflags = hooks$ZULM_CXXFLAGS,
      compiler = native$compiler
    ),
    list(cpu = native[setdiff(
      names(native),
      c("ggml_version", "ggml_commit", "ggml_patches", "variant", "compiler")
    )])
  )
}
