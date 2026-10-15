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

* Modules: `zl_linear()`, `zl_embedding()`, `zl_norm()`,
  `zl_attention()`, `zl_mlp()`, `zl_block()`, `zl_adaln()` and
  `zl_time_embedding()` declare their parameters in a store and return
  functions of tensors. They, the losses and two attention blocks in the
  styles of TimesFM and Toto are checked against PyTorch.

* `zl_threads()` sets the number of threads for the session, capped at the
  available cores and at 2 while `R CMD check` limits cores. One persistent
  pool serves every computation, and is recreated in a forked process. A
  computation can be interrupted; it then signals `zulm_interrupt`.

* safetensors: `zl_safetensors_read()`, `zl_load_safetensors()` (with a
  name map) and `zl_save_safetensors()`. PyTorch's `[out, in]` weights
  load as `c(in, out)` with no transposition.

* Half-precision weights: `zl_load(keep_types = TRUE)` and
  `zl_load_safetensors()` keep f16 and bf16 parameters in their type, and
  `zl_param_names(types = TRUE)` reports types. The documentation states
  their cost in precision.

* State tensors: `zl_state()` declares a persistent tensor that graphs
  read through `zl_param_tensor()` and write with `zl_state_write()`;
  `zl_state_get()` and `zl_state_set()`, with `offset`, read and write it
  from R. States are neither trained nor saved. A key-value cache for
  incremental decoding is built from them.

* `zl_graph_cache()` builds a graph per new set of input shapes for
  `zl_forward()`, and frees the least recently used under
  `getOption("zulm.max_bytes")`.

* `zl_crossprod()` accepts a transposed vector as its first argument, as
  `zl_attention()` makes for a single position.

* `zl_flash_attention()` is 'ggml''s fused attention, forward only: it
  agrees with the plain form within f32 rounding, gives the same bits at
  any thread count, and attends over long contexts in memory linear in
  their length.

* Arrow streams: `zl_arrow_batches()` reads record batches from
  'DuckDB', Parquet or 'arrow' through 'nanoarrow' into batches of
  exactly `rows` rows, with missing values as errors or masks and the
  last batch dropped or padded; `zl_step()` and `zl_forward()` take the
  iterator as `data`, and `zl_next()` returns a batch as R arrays.
  `zl_int()` marks integer columns in its `map`.

* The C table: `inst/include/zulm.h` declares `zulm_api`, registered as
  `zulm_api_v1`, for packages with `LinkingTo: zulm`. From C they run
  graphs built in R, write inputs, read outputs, read and write
  parameters and state tensors, draw from named streams under the session
  key, and fill inputs from an Arrow iterator. Failures are return codes
  with a message, never R errors.

* Two vignettes: `vignette("training")` trains a small event-sequence
  model and samples from it; `vignette("pretrained")` runs an attention
  block exported by PyTorch from safetensors and decodes through a
  key-value cache in state tensors.

* Fuzzing (`tools/fuzz/`) found, and this version fixes:
  - sizes, offsets and axes of 2^31 or more, or `NA`, reached 'ggml' as
    negative numbers; they are `zulm_invalid_argument`;
  - an invalid choice, such as `zl_tri(x, "x")`, gave an unclassed
    error; it is `zulm_invalid_argument`;
  - `zl_param()` allocated the initial values of a parameter larger
    than `getOption("zulm.max_bytes")` before refusing it;
  - training graphs whose gradients 'ggml' cannot carry back through a
    permutation, and cross-entropy labels that depend on a parameter,
    reached 'ggml' assertions; they are `zulm_unsupported_backward`;
  - gradient clipping stopped R with a 'ggml' assertion on a permuted
    parameter whose first dimension was 1 (patch P7);
  - a GGUF file with an invalid metadata type was undefined behaviour
    in 'ggml''s reader (patch P11).
