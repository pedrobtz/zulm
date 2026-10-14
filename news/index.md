# Changelog

## zulm 0.0.0.9000

- First development version: the package skeleton and its family
  conventions. The R interface follows in later stages.

- Bundles the CPU subset of ‘ggml’ v0.26.0, built once per
  instruction-set level (`baseline`, `sse42` and `avx2` on x86-64,
  `neon` on arm64); the best level the CPU supports is loaded with the
  package, and `options(zulm.variant =)` forces another.

- [`zulm_info()`](https://pedrobtz.github.io/zulm/reference/zulm_info.md)
  reports the bundled ‘ggml’ version, the variant loaded, and the CPU
  features it was built for.

- Installing from source with `ZULM_CFLAGS` and `ZULM_CXXFLAGS` set
  builds one variant for the local CPU (not on Windows).
