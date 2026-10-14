# zulm 0.0.0.9000

* First development version: the package skeleton and its family
  conventions. The R interface follows in later stages.

* Bundles the CPU subset of 'ggml' v0.26.0, built once per instruction-set
  level (`baseline`, `sse42` and `avx2` on x86-64, `neon` on arm64); the
  best level the CPU supports is loaded with the package, and
  `options(zulm.variant =)` forces another.

* `zulm_info()` reports the bundled 'ggml' version, the variant loaded, and
  the CPU features it was built for.

* Installing from source with `ZULM_CFLAGS` and `ZULM_CXXFLAGS` set builds
  one variant for the local CPU (not on Windows).

* Parameter stores: `zl_params()`, `zl_param()` with `init` and `type`
  (`"f32"`, `"f16"` or `"bf16"`), `zl_param_get()`, `zl_param_set()`,
  `zl_param_names()`, `zl_param_count()` and `zl_param_tensor()`;
  initialisers `zl_init_normal()`, `zl_init_uniform()`, `zl_init_zeros()`
  and `zl_init_ones()`, drawing from 'zurand' under a key derived from the
  parameter's name.

* `zl_save()` and `zl_load()` write and read stores as GGUF files, with
  metadata, at f32, f16 or bf16.

* Every error inherits `zulm_error` and every warning `zulm_warning`; see
  `?"zulm-conditions"` for the classes. Allocations are checked against
  `getOption("zulm.max_bytes")` (4 GiB by default).
