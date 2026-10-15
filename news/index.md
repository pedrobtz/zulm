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

- Training:
  [`zl_adamw()`](https://pedrobtz.github.io/zulm/reference/zl_adamw.md)
  and
  [`zl_sgd()`](https://pedrobtz.github.io/zulm/reference/zl_adamw.md)
  describe optimisers;
  [`zl_trainer()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  prepares a graph with a `loss` for training through ‘ggml-opt’, with
  gradient accumulation and global-norm clipping (`clip_norm`);
  [`zl_step()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  trains on one batch,
  [`zl_grad()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  returns a parameter’s gradient, and
  [`zl_reset()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  clears the optimiser’s state. An operation without a gradient between
  a parameter and the loss is refused with `zulm_unsupported_backward`,
  naming it and its call.

- [`zl_graph()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  and
  [`zl_trainer()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  collect garbage when graphs and trainers no longer referenced hold
  over 128 MB, which R’s collector does not see.

- The bundled ‘ggml’ carries patches for the gradients of `MEAN`, `SUB`
  and `DIV` with broadcasting, and for gradient clipping in ‘ggml-opt’.

- A NaN reaching
  [`zl_softmax()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md)
  no longer stops R, a transposed column enters arithmetic and other
  kernels correctly, and graphs reserve memory as they grow rather than
  for the largest graph.

- Random numbers come from ‘zurand’:
  [`zl_seed()`](https://pedrobtz.github.io/zulm/reference/zl_seed.md)
  sets the session key,
  [`zl_rng_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md)
  and
  [`zl_rng_normal()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md)
  draw in R, and
  [`zl_noise_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_noise_normal.md)
  and
  [`zl_noise_normal()`](https://pedrobtz.github.io/zulm/reference/zl_noise_normal.md)
  draw inside a graph, anew at each training step. Draws do not depend
  on the thread count or on how a batch is split.

- Reported losses
  ([`zl_cross_entropy()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md),
  [`zl_mse()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md),
  [`zl_mae()`](https://pedrobtz.github.io/zulm/reference/zl-loss.md),
  from
  [`zl_step()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
  and
  [`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md))
  are fixed-order sums of per-row values, so they are the same at any
  thread count.

- Modules:
  [`zl_linear()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_embedding()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_norm()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_attention()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_mlp()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_block()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
  [`zl_adaln()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md)
  and
  [`zl_time_embedding()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md)
  declare their parameters in a store and return functions of tensors.
  They, the losses and two attention blocks in the styles of TimesFM and
  Toto are checked against PyTorch.

- [`zl_threads()`](https://pedrobtz.github.io/zulm/reference/zl_threads.md)
  sets the number of threads for the session, capped at the available
  cores and at 2 while `R CMD check` limits cores. One persistent pool
  serves every computation, and is recreated in a forked process. A
  computation can be interrupted; it then signals `zulm_interrupt`.

- safetensors:
  [`zl_safetensors_read()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md),
  [`zl_load_safetensors()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md)
  (with a name map) and
  [`zl_save_safetensors()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md).
  PyTorch’s `[out, in]` weights load as `c(in, out)` with no
  transposition.

- Half-precision weights: `zl_load(keep_types = TRUE)` and
  [`zl_load_safetensors()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md)
  keep f16 and bf16 parameters in their type, and
  `zl_param_names(types = TRUE)` reports types. The documentation states
  their cost in precision.

- State tensors:
  [`zl_state()`](https://pedrobtz.github.io/zulm/reference/zl_state.md)
  declares a persistent tensor that graphs read through
  [`zl_param_tensor()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
  and write with
  [`zl_state_write()`](https://pedrobtz.github.io/zulm/reference/zl_state.md);
  [`zl_state_get()`](https://pedrobtz.github.io/zulm/reference/zl_state.md)
  and
  [`zl_state_set()`](https://pedrobtz.github.io/zulm/reference/zl_state.md),
  with `offset`, read and write it from R. States are neither trained
  nor saved. A key-value cache for incremental decoding is built from
  them.

- [`zl_graph_cache()`](https://pedrobtz.github.io/zulm/reference/zl_graph_cache.md)
  builds a graph per new set of input shapes for
  [`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md),
  and frees the least recently used under `getOption("zulm.max_bytes")`.

- [`zl_crossprod()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md)
  accepts a transposed vector as its first argument, as
  [`zl_attention()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md)
  makes for a single position.

- [`zl_flash_attention()`](https://pedrobtz.github.io/zulm/reference/zl_flash_attention.md)
  is ‘ggml’’s fused attention, forward only: it agrees with the plain
  form within f32 rounding, gives the same bits at any thread count, and
  attends over long contexts in memory linear in their length.
