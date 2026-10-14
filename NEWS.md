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

* Graphs: `zl_graph()` builds a graph once from an R function, `zl_int()`
  declares integer inputs, `zl_forward()` runs it any number of times on
  new inputs, and `zl_graph_info()` describes it.

* Operations on tensors inside a graph: `+`, `-`, `*`, `/` with
  broadcasting, `zl_scale()`, `zl_sum()`, `zl_sum_rows()`,
  `zl_mean_rows()`, `zl_cumsum()`, the elementwise functions
  (`zl_sqr()` to `zl_tanh()`, `zl_gelu()`), `zl_swiglu()`,
  `zl_crossprod()`, `zl_reshape()`, `zl_permute()`, `zl_t()`,
  `zl_cont()`, `zl_view()`, `zl_concat()`, `zl_pad()`, `zl_get_rows()`,
  `zl_where()`, `zl_clamp()`, `zl_tri()`, `zl_softmax()` with masks,
  `zl_causal_mask()`, `zl_rope()` in both conventions, `zl_rms_norm()`,
  `zl_layer_norm()`, the losses `zl_cross_entropy()`, `zl_mse()` and
  `zl_mae()`, and `zl_cast()`.

* Every argument 'ggml' would assert on is checked first; an assertion
  that still happens while a graph is built is `zulm_engine_error`.

* 'ggml''s log becomes messages of class `zulm_message`, at the level of
  `options(zulm.log =)` (`"warn"` by default).

* Training: `zl_adamw()` and `zl_sgd()` describe optimisers;
  `zl_trainer()` prepares a graph with a `loss` for training through
  'ggml-opt', with gradient accumulation and global-norm clipping
  (`clip_norm`); `zl_step()` trains on one batch, `zl_grad()` returns a
  parameter's gradient, and `zl_reset()` clears the optimiser's state.
  An operation without a gradient between a parameter and the loss is
  refused with `zulm_unsupported_backward`, naming it and its call.

* `zl_graph()` and `zl_trainer()` collect garbage when graphs and trainers
  no longer referenced hold over 128 MB, which R's collector does not see.

* The bundled 'ggml' carries patches for the gradients of `MEAN`, `SUB`
  and `DIV` with broadcasting, and for gradient clipping in 'ggml-opt'.

* A NaN reaching `zl_softmax()` no longer stops R, a transposed column
  enters arithmetic and other kernels correctly, and graphs reserve
  memory as they grow rather than for the largest graph.

* Random numbers come from 'zurand': `zl_seed()` sets the session key,
  `zl_rng_uniform()` and `zl_rng_normal()` draw in R, and
  `zl_noise_uniform()` and `zl_noise_normal()` draw inside a graph, anew at
  each training step. Draws do not depend on the thread count or on how a
  batch is split.

* Reported losses (`zl_cross_entropy()`, `zl_mse()`, `zl_mae()`, from
  `zl_step()` and `zl_forward()`) are fixed-order sums of per-row values,
  so they are the same at any thread count.
