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

- Parameter stores:
  [`zl_params()`](https://pedrobtz.github.io/zulm/reference/zl_params.md),
  [`zl_param()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
  with `init` and `type` (`"f32"`, `"f16"` or `"bf16"`),
  [`zl_param_get()`](https://pedrobtz.github.io/zulm/reference/zl_params.md),
  [`zl_param_set()`](https://pedrobtz.github.io/zulm/reference/zl_params.md),
  [`zl_param_names()`](https://pedrobtz.github.io/zulm/reference/zl_params.md),
  [`zl_param_count()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
  and
  [`zl_param_tensor()`](https://pedrobtz.github.io/zulm/reference/zl_params.md);
  initialisers
  [`zl_init_normal()`](https://pedrobtz.github.io/zulm/reference/zl_init_normal.md),
  [`zl_init_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_init_normal.md),
  [`zl_init_zeros()`](https://pedrobtz.github.io/zulm/reference/zl_init_normal.md)
  and
  [`zl_init_ones()`](https://pedrobtz.github.io/zulm/reference/zl_init_normal.md),
  drawing from ‘zurand’ under a key derived from the parameter’s name.

- [`zl_save()`](https://pedrobtz.github.io/zulm/reference/zl_save.md)
  and
  [`zl_load()`](https://pedrobtz.github.io/zulm/reference/zl_save.md)
  write and read stores as GGUF files, with metadata, at f32, f16 or
  bf16.

- Every error inherits `zulm_error` and every warning `zulm_warning`;
  see
  [`?"zulm-conditions"`](https://pedrobtz.github.io/zulm/reference/zulm-conditions.md)
  for the classes. Allocations are checked against
  `getOption("zulm.max_bytes")` (4 GiB by default).
