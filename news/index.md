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

- Graphs:
  [`zl_graph()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  builds a graph once from an R function,
  [`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  declares integer inputs,
  [`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  runs it any number of times on new inputs, and
  [`zl_graph_info()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  describes it.

- Operations on tensors inside a graph: `+`, `-`, `*`, `/` with
  broadcasting,
  [`zl_scale()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_sum()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_sum_rows()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_mean_rows()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_cumsum()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  the elementwise functions
  ([`zl_sqr()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md)
  to
  [`zl_tanh()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_gelu()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md)),
  [`zl_swiglu()`](https://pedrobtz.github.io/zulm/reference/zl-arith.md),
  [`zl_crossprod()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_reshape()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_permute()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_t()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_cont()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_view()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_concat()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_pad()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_get_rows()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md),
  [`zl_where()`](https://pedrobtz.github.io/zulm/reference/zl-select.md),
  [`zl_clamp()`](https://pedrobtz.github.io/zulm/reference/zl-select.md),
  [`zl_tri()`](https://pedrobtz.github.io/zulm/reference/zl-select.md),
  [`zl_softmax()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md)
  with masks,
  [`zl_causal_mask()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md),
  [`zl_rope()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md)
  in both conventions,
  [`zl_rms_norm()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md),
  [`zl_layer_norm()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md),
  the losses
  [`zl_cross_entropy()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md),
  [`zl_mse()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md) and
  [`zl_mae()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md),
  and
  [`zl_cast()`](https://pedrobtz.github.io/zulm/reference/zl_cast.md).

- Every argument ‘ggml’ would assert on is checked first; an assertion
  that still happens while a graph is built is `zulm_engine_error`.

- ‘ggml’’s log becomes messages of class `zulm_message`, at the level of
  `options(zulm.log =)` (`"warn"` by default).
