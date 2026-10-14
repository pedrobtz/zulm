# zulm: tensor graphs, training and inference in R on upstream ggml

- **Status:** Package design, adopted from RFC 0031 in `pedrobtz/packages`
  (merged as PR #49, revised after review in PR #51), where the package
  was called `zuggml`. Each decision in section 16 is the maintainer's;
  those marked decided there are decided. Section 13.1 (CPU dispatch on
  CRAN binaries) is a go/no-go the first stage settles. The maintainer
  has decided to build zulm as an alternative to the CRAN package
  `ggmlR` and, for the packages that build on it, to `torch`; this
  design does not reopen either decision.
- **Revisions.** 2026-10-10, after review of the RFC: CRAN's x86-64
  macOS binaries make clang a Stage 1 blocker (§13.1); upstream's
  diagnostic pragmas need stripping for `--as-cran` (§9.2); gradient
  clipping needs patch P7 (§9.3); inputs live in a static buffer (§4);
  Arrow batches, safetensors, spectra and the `zuexact` lookup left
  v0.1.0. **2026-10-13, on adoption as the package design:** the
  package is `zulm` and the R prefix `zl_` (D31, D32); a second driving
  use case joins the first, running pretrained time-series foundation
  models for `zukzeit`, which today runs them on `torch` (§1.2, D33);
  the inference work is designed (§6.9, §7.4); the forward-only operations that inference
  needs, and both rotary conventions, enter v0.1.0 (§5.3, D37, D38); a
  key-value cache is designed as state tensors (§6.9, D34); f16 and
  bf16 weights in memory are opt-in, with their precision cost stated
  (§6.9, D35); the safetensors header is parsed by the family's
  `zujson`, so yyjson is not vendored (§7.4, D36); the C table gains the
  members both consumers asked for (§8); and §7.9 draws the boundary
  with zukzeit as §7.8 does with zuflow. **2026-10-14:** random numbers
  come from `zurand`'s C table, the family generator RFC 0031 looked
  for under another name and could not find; Random123 is no longer
  vendored and Box–Muller is dropped (§7.2, D19, D20). The opt-in
  native build follows zufft's: environment hooks appended to the
  `PKG_*FLAGS` in `src/Makevars`, not on Windows, with the same README
  text and CI legs (§13.4, D40); and `-funroll-loops` leaves Makevars,
  which R's own Makevars check would reject, for a pragma in the
  force-included header (§13.1, D41). The maintainer decided the
  prefix (`zl_`, D32) and the releases: zulm goes to CRAN once, at
  0.1.0, with everything zuflow 0.1.0 and zukzeit 0.1.0 need, so the
  inference work and the Arrow iterator are in v0.1.0 and only
  `zuexact`, the FFT and sorting wait for 0.2 (§2, §14, D27);
  `zujson` is `Imports` (question 13).
- **Date:** 2026-10-14.
- **Package:** `zulm` (repository `pedrobtz/zulm`, skeleton created
  2026-10-13; no package of that name on CRAN as of 2026-10-13: crandb
  answers "not found" and the `github.com/cran` mirror has no such
  repository). Vendors the CPU part of ggml (MIT; zulm takes MIT) and
  the Arrow C data interface definitions (Apache-2.0). Imports and
  links to `zurand` for random numbers (§7.2); imports `zujson` for the
  safetensors header (§7.4); suggests `nanoarrow` and, from 0.2,
  `zuexact` and `zufft` (§7).
  Consumers: `zuflow` (RFC 0032), a flow-matching generator for event
  sequences, which trains and samples; and `zukzeit` (RFC 0002), an
  inference engine for time-series foundation models, which runs
  pretrained checkpoints.
- **Read for the 2026-10-13 revision:** `pedrobtz/zukzeit` at `main`
  (2026-09-23): `README.md`, `DESCRIPTION` (`Imports: torch,
  safetensors, hfhub`), `.agents/plan.md`, `roadmap.md`,
  `consumer-api.md`, `contract-v2.md`, `timesfm-feasibility.md`,
  `chronos2-feasibility.md`, `toto-feasibility.md`, and the ports
  `R/timesfm-module.R`, `timesfm-inference.R`, `chronos2-module.R`,
  `chronos2-inference.R`, `toto-module.R`, `toto-inference.R` (every
  `torch` operation they call, §1.2); RFC 0032 as merged (its §4, §7
  and §16 for what zuflow asks of this package); upstream ggml's
  `include/ggml.h`, `include/ggml-opt.h` and `src/ggml.c` at the
  default branch on 2026-10-13 (the `ggml_op` enumeration, the case
  list of `ggml_compute_backward`, and the contracts of
  `ggml_soft_max_ext`, `ggml_rope_ext`, `ggml_flash_attn_ext`,
  `ggml_get_rows`, `ggml_set_rows`, `ggml_cumsum`, `ggml_clamp`,
  `ggml_tri`, `ggml_pad`, `ggml_argsort` and `ggml_top_k`); `zujson`
  0.1.0 on CRAN (`DESCRIPTION`, `src/`: R API only, no C table); the
  CRAN `safetensors` 0.3.0 package (`DESCRIPTION`, `R/safetensors.R`:
  loads into `torch` or `pjrt` tensors only); James Somers's post again
  for what it states and does not state. For 2026-10-14:
  `pedrobtz/zurand` at `main` (`DESCRIPTION` 0.1.0, `NEWS.md`,
  `cran-comments.md` prepared for a first submission, `CLAUDE.md`,
  `inst/include/zurand.h` at `ZURAND_API_VERSION 3`,
  `src/zurand_engine.h`, `src/zurand_lazy.h`, the manual pages of
  `rng_key()`, `rng_fold()`, `rng_uniform()`, `rng_normal()`,
  `rng_threads()` and `rng_simd()`); crandb reports no `zurand` on
  CRAN as of 2026-10-14. `pedrobtz/zufft` at `main`: `.agents/design.md`
  §12.3 (Makevars, the native hook, the `.check_make_vars` finding),
  `src/Makevars`, `src/Makevars.win`, `README.md` ("Native build"),
  `.github/workflows/build-flags.yaml` and `native-checks.yaml`; R's
  `check.R` at the `wch/r-source` trunk for the non-portable-flag
  NOTE. Nothing was built for this revision.
- **Source:** upstream ggml (`ggml-org/ggml`) at tag `v0.26.0`
  (`d7cb574`, 2026-10-05; the default branch's head was `ffa4e8b`, the
  same day, a script change only), `LICENSE` (MIT), `CMakeLists.txt`
  (`GGML_VERSION 0.26.0`, options `GGML_NATIVE`, `GGML_OPENMP`,
  `GGML_LLAMAFILE`, `GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS`),
  `src/CMakeLists.txt` (the per-ISA variant list), `include/ggml.h`,
  `ggml-backend.h`, `ggml-cpu.h`, `ggml-opt.h`, `gguf.h`; in `src/`:
  `ggml.c` (`ggml_abort`, `ggml_print_backtrace`,
  `ggml_compute_backward` in full), `ggml.cpp`, `ggml-opt.cpp`,
  `ggml-cpu/ggml-cpu.c` (compute loop, thread pool, initialisation),
  `ggml-cpu/ops.cpp` (cross-entropy), `ggml-cpu/simd-mappings.h`,
  `ggml-cpu/vec.cpp`, `ggml-cpu/arch/x86/cpu-feats.cpp`; and
  `tests/test-opt.cpp`. Upstream `v0.9.5` is `ebc3a0f` (2025-12-31) and
  `v0.11.0` is `ac6f7b4` (2026-05-05); 38 tags run from `v0.9.5` to
  `v0.26.0`. `ggmlR` (`Zabis13/ggmlR`) at `7c22445` (2026-10-10,
  `DESCRIPTION` 0.8.7): `DESCRIPTION`, `NAMESPACE`, `NEWS.md`,
  `configure`, `src/Makevars.in`, `src/r_ggml_compat.h`,
  `src/r_ggml_io.c`, `src/ggml.h`, `inst/include/`,
  `R/ag_backward_graph.R`, `R/autograd.R`, `src/r_interface_opt.c`.
  `DESCRIPTION` files of `ggmlR`, `llamaR`, `sd2R`, `edgemodelr`,
  `localLLM` and `cayleyR` from the `github.com/cran` mirror. R's
  `src/library/tools/R/check.R`, `sotools.R` and `QC.R`
  (`.check_pragmas`) from the `wch/r-source` trunk mirror; CRAN's check
  flavours page (`web/checks/check_flavors.html`); CRAN download counts
  from `cranlogs.r-pkg.org` for the 30 days to 2026-10-09. James
  Somers, "Can you use autoregressive diffusion to generate market
  data?", Jane Street blog, 2026-09-30. **Built and
  measured** in the drafting session (GCC 13.3, one Intel Xeon at
  2.80 GHz, two cores, AVX-512F present): upstream's CPU backend in eight
  configurations (§13.1); a probe program that times f32 matrix
  products, checks thread-count determinism of seven operations, trains
  a causal-transformer encoder with a flow-matching head through
  `ggml-opt` for 1 000 steps (§1.1), reproduces two backward-pass
  assertions (§9.3) and catches an abort as a C++ exception (§13.2); a
  CPU-only shared object compiled with `-Wall -Wextra -pedantic` and
  audited with `nm -D` (§13.3). For §7: RFCs 0023, 0026 and 0027 on
  their branches (`rfc-0023-zuexact`, `rfc-0026-zufft`,
  `rfc-0027-zuspecial`) for their C tables; `ggml-cpu/ops.cpp` and
  `ggml-opt.cpp` for where sums happen; `ggml.h` and
  `ggml_compute_backward` for custom operations; `src/ggml-blas`;
  `DEShawResearch/random123` at `9545ff6`, `ibireme/yyjson` at
  `1f934d2`, `huggingface/safetensors` at `e246a25`, `google/highway` at
  `b93ad44`, `flame/blis` at `a8037cf`, `PythonOT/POT` at `604c47f`
  (licences and the files named in §7); `nanoarrow` 0.9.0 from the
  `github.com/cran` mirror; R's `src/extra/blas/`; and a Highway
  dispatch test (§7.7). Nothing was built under R.

## 1. What zulm is

`zulm` builds computation graphs of tensors in R, runs them on the CPU,
differentiates them, and trains their parameters, through upstream
ggml: Georgi Gerganov's C tensor library behind llama.cpp and
whisper.cpp. ggml gives a graph of tensor operations, its own
reverse-mode differentiation (`ggml_build_backward_expand`), optimisers
(AdamW and SGD in `ggml-opt`), a CPU backend with its own thread pool
and SIMD kernels, and the GGUF file format.

zulm takes a narrow slice of that and nothing else: tensors, explicit
graphs, ggml's autograd, its optimisers, GGUF for parameters, a small
set of model-building modules, and a versioned C table for packages
that build models on top. It is a substrate, not a framework. The
graphs that train a model are the graphs that run a pretrained one:
inference is the forward half of the same design, not a second
product.

Users: anyone who wants to train small and medium neural networks in R
on the CPU, or run pretrained transformer checkpoints of up to a few
hundred million parameters, without Python, LibTorch or a GPU
toolchain, with results that reproduce bit for bit at a fixed seed;
and family packages that build models: `zuflow`, which trains and
samples (§1.1), and `zukzeit`, which runs checkpoints (§1.2).

The one-line statement that governs every decision below:

> **Upstream ggml, unmodified and current, as a small, safe, reproducible
> R substrate for training and running models on the CPU.**

Two use cases size the design, and v0.1.0 carries both in full (D27):
zulm goes to CRAN first, and zuflow 0.1.0 and zukzeit 0.1.0 are built
on it, so neither may find a capability missing. The first use case,
training, fixes the graph, autograd and optimiser surface. The second,
inference of pretrained models, fixes the forward-only operations,
the weight formats and the decoding machinery. The two share almost
every block (causal attention, rotary positions, RMS normalisation,
SwiGLU, residual MLPs), which is why one operation set serves both.

### 1.1 The first driving use case: training an event-sequence generator

The v0.1.0 surface is sized by one model: the autoregressive generator
of market events in James Somers's post of 2026-09-30 on the Jane Street
blog, built by a research intern. As the post describes it:

- a causally masked transformer encoder reads the event history and
  produces a latent per position;
- a small categorical head predicts the next event's kind (two classes
  first, trade or best-bid-offer update; twenty in a later version);
- a diffusion head generates seven continuous targets (price and
  elapsed time among them), conditioned on the latent and the event kind
  through adaptive layer normalisation (AdaLN);
- a DDPM head with a 1 000-step cosine schedule and DDIM sampling was
  unstable (88–95 % of values beyond 8 standard deviations); rectified
  flow (linear interpolation between noise and data, predicting the
  velocity, data minus noise) worked out of the box;
- "atom smoothing" handles point masses (zero time gaps, prices at the
  bid, ask or mid);
- evaluation is one-step marginals against real data given a context of
  10 240 events (total-variation divergence per target), a discriminator,
  and autoregressive rollouts, which degrade; the author says the model
  is "not yet accurate enough to be a realistic generator".

Every operation that model needs for training exists in upstream ggml
with a backward pass, with two compositions (§6.3): matrix products,
softmax with a causal mask, RMS normalisation, SwiGLU, SiLU, sine and
cosine, broadcast add and multiply, views, cross-entropy, and AdamW.
The drafting session trained a scaled-down version through `ggml-opt`
from C: inputs of 8 features, 64 events, batches of 16; model width 64;
one single-head causal attention block with a SwiGLU MLP; a four-class
kind head; and a flow-matching head over seven targets with AdaLN
conditioning on latent, kind and time. On synthetic data whose next
kind and targets depend on the history, the mean loss fell from 2.85
(steps 1–100) to 1.25 (steps 901–1 000), and the parameters after
1 000 AdamW steps were bit-identical at one and two threads.

The same model in zulm's R API (§5) reads as follows; `zuflow` will
own the full version, with the data pipeline, atom smoothing,
multi-step sampling and evaluation.

```r
p   <- zl_params()
emb <- zl_linear(p, "in", n_in = 8, n_out = 64)
blk <- zl_block(p, "enc1", dim = 64, heads = 4, causal = TRUE)
ln  <- zl_norm(p, "enc.ln", 64)
kh  <- zl_linear(p, "kind", 64, 4)
ke  <- zl_embedding(p, "kind.emb", 4, 64)
ada <- zl_adaln(p, "flow.ada", dim = 64, cond_dim = 64)
fin <- zl_linear(p, "flow.in", 7, 64)
fo  <- zl_linear(p, "flow.out", 64, 7)

build <- function(x) {          # x: named list of input tensors
  z    <- ln(blk(emb(x$feat)))                  # latent per event
  ce   <- zl_cross_entropy(kh(z), x$kind_prob)  # next-kind head
  cond <- z + ke(x$kind) + zl_time_embedding(x$t, 64)
  v    <- fo(zl_silu(ada(zl_rms_norm(fin(x$x_t)), cond)))
  list(loss = ce + zl_mse(v, x$v_target), velocity = v)
}

g  <- zl_graph(build, p,
               inputs = list(feat = c(8, 64, 16),
                             kind = zl_int(64, 16),
                             kind_prob = c(4, 64, 16),
                             t = c(1, 64, 16), x_t = c(7, 64, 16),
                             v_target = c(7, 64, 16)),
               loss = "loss")
tr <- zl_trainer(g, zl_adamw(lr = 1e-3))
for (i in seq_len(1000)) zl_step(tr, next_batch())  # noise: Philox
```

**Acceptance (Stage 6, 2026-10-14).** `tests/testthat/test-acceptance.R`,
run by the `acceptance` workflow, trains this model (with the flow
head's noise drawn in the graph, `zl_noise_normal(c(7, 64, 16), "x0")`,
and `x_t` and the velocity target computed there) on a synthetic marked
process: four event kinds following a Markov chain that keeps its kind
with probability 0.8–0.85, and seven targets that follow an AR(1),
`y_t = 3 mu[, k_t] + 0.5 y_{t-1} + 0.1 e`, around a mean per kind; the
features at each position are the previous event's one-hot kind and
first four targets. Measured: mean loss 6.78 over steps 1–100 and 1.22
over steps 901–1 000 (18 %), with reported losses and all parameters
bit-identical at one and two threads, and eight Euler steps of the
flow head giving finite values. (A first version of the process, with
targets three times smaller and three times noisier, reached 67 %: most
of its loss was the irreducible noise of flow matching.)

Sampling integrates the learnt velocity field with a few Euler steps,
each a call of `zl_forward()` on a forward-only graph of the head with
the encoder's latent held fixed (§6.4).

### 1.2 The second driving use case: running time-series foundation models

`zukzeit` (RFC 0002) runs open time-series foundation models natively
in R and returns predictive quantiles. It ports each architecture to R
`torch` by hand and certifies the port against pinned reference
fixtures from the official PyTorch implementation; three checkpoints
have passed both its gates:

| Checkpoint | Parameters, file | Shape of the network | What is unusual |
| --- | --- | --- | --- |
| TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`) | 231 M, 925 MB f32, 232 tensors | 20 pre-norm blocks, width 1 280, 16 heads × 80, patch 32, context 16 384 (512 patches), point head 128, quantile head 1 024 × 10 | fused QKV; rotary positions in the split-half convention, then RMS normalisation of queries and keys per head; a per-dimension query scale `softplus(p) · 1.4427 / sqrt(80)`; attention with `scale = 1`; a mask that also hides each series' padded prefix; autoregressive decode of 128-value blocks over a key-value cache; cumulative reversible normalisation in `double` outside the network |
| Chronos-2 (`amazon/chronos-2`) | 119 M, 478 MB f32, 170 tensors | encoder-only T5: 12 blocks, width 768, feed-forward 3 072 with ReLU, 12 heads × 64, patch 16, context 8 192 (512 tokens), 21 quantiles | two attention sublayers per block, over time and over the series of a group; the group mask is additive and per batch row; attention `scale = 1`; T5 layer norm (no mean subtraction); split-half rotary positions; a time-encoding feature channel; `asinh` scaling |
| Toto 2.0 4M (`Datadog/Toto-2.0-4m`) | 4.1 M, 16 MB f32, 48 tensors | 4 blocks, width 256, feed-forward 688 gated (SwiGLU), 4 heads × 64, patch 32, context 4 096 (128 patches), 9 quantiles | unit scaling (every linear divided by `sqrt(fan_in)`, SiLU times 1.7668, residuals `(tau h + skip) / sqrt(1 + tau²)` with buffers); interleaved rotary positions on half of each head with an xPos decay of opposite sign on queries and keys; attention `scale = 1 / 64`; the fourth layer attends across variates, not time; a Welford scaler in float64 composed with `asinh`; one forward pass, no cache |

Running these on zulm instead of `torch` is the inference case. It is
worth doing for the same reasons as the training case: zukzeit's users
today download LibTorch to run a 16 MB checkpoint, and its README
records that its fixtures drift by a couple of float32 ulps between
machines under the same pinned environment, where zulm promises bit
identity per ISA variant (§6.5). The port stays zukzeit's (§7.9); what
zulm owes it is the operations, read from the three `torch` ports on
2026-10-13 and matched against upstream ggml's operation list:

| The ports call (R `torch`) | Upstream ggml | zulm |
| --- | --- | --- |
| `nnf_linear` with `[out, in]` weights, bias or not | `mul_mat`, `add` | `zl_linear()`; weights load without transposition (§7.4) |
| RMS norm, with a weight (TimesFM, Chronos-2) or without (Toto) | `rms_norm`, `mul` | `zl_rms_norm()` |
| `nnf_silu`, `nnf_relu`, `nnf_softplus`, SwiGLU as `a * silu(b)` on halves | unary ops, `glu` | `zl_silu()`, `zl_relu()`, `zl_softplus()`, `zl_swiglu()` |
| fused QKV then `torch_chunk` or `torch_split`; `narrow` | `view` | `zl_view()` |
| heads by `reshape`, `permute`, `transpose`, `flatten` | `reshape`, `permute`, `cont` | §5.3 |
| rotary, split-half: `cat((-x2, x1))` with `cat((freqs, freqs))` (TimesFM, Chronos-2) | `rope_ext`, mode `GGML_ROPE_TYPE_NEOX` | `zl_rope(mode = "half")` |
| rotary, interleaved adjacent pairs on the leading 32 of 64 dims (Toto) | `rope_ext`, mode `GGML_ROPE_TYPE_NORMAL`, `n_dims = 32` | `zl_rope(mode = "interleaved", n_dims =)` |
| xPos decay: a per-position, per-channel factor with exponent +1 on queries, −1 on keys (Toto) | none; `mul` by a factor tensor | the factor is an input computed in R |
| `torch_matmul(q, kᵀ) * scale` with `scale` 1 (TimesFM, Chronos-2) or `1 / d` (Toto) | `mul_mat`, `soft_max_ext(mask, scale)` | `zl_softmax(mask =, scale =)` |
| `torch_tril`, `masked_fill(-Inf)`, TimesFM's `query ≥ key & key ≥ num_masked[b]`, Chronos-2's `einsum("qb,bt->qbt")` group × time mask | `soft_max_ext`'s mask: `[n_kv, n_q, h, b]`, f32 or f16, contiguous, with `h` and `b` dividing the scores' dims (asserted in `ggml_soft_max_impl`) | an additive mask input built in R; `zl_causal_mask()` for the plain case |
| `torch_scaled_dot_product_attention` | the plain form, or `flash_attn_ext` forward-only | `zl_attention()` composes the plain form; `zl_flash_attention()` for long contexts (§6.9) |
| attention across variates or a group (Toto layer 4, Chronos-2 every block) | `permute` + `cont`, then the same attention | §5.3 |
| unit-scaled residuals `(tau h + skip) / sqrt(1 + tau²)` with `tau` buffers | `scale`, `add` | `Ops` and `zl_scale()` |
| `torch_cat` of values and masks on the feature axis; of patches | `concat` (no backward) | `zl_concat()`, forward-only |
| `torch_cat` of a key-value cache with new keys per decode step | `set_rows` into a persistent tensor, plus a mask | state tensors (§6.9) |
| `index_select`, `flip`, `expand` | `get_rows`, `repeat` | `zl_get_rows()`; `flip` is `get_rows` with reversed indices |
| `torch_where`, `clamp`, `torch_minimum` and `torch_maximum` (quantile-crossing repair) | `mul` and `add` with 0-1 masks; `clamp`; `relu` composition | `zl_where()`, `zl_clamp()`, forward-only; crossing repair on doubles in R |
| `torch_cumsum` (Toto's causal scaler), Welford statistics in float64, `asinh`, `sinh` | `cumsum` (no backward); no float64; no `asinh` | preprocessing in R on doubles (§6.9); `zl_cumsum()` forward-only for in-graph uses |
| `torch_sort`, `argsort` | `argsort`, `top_k` (no backward) | after 0.1.0, if a consumer asks; the ports sort on doubles in R |
| `torch_arange`, `torch_outer`, `torch_pow`, `torch_exp` for position tables | tables computed in R | an input, or `zl_param()` without gradient |
| `isfinite`, `nan_to_num`, `item`, dtype casts | none needed | R |
| `nn_parameter` on a meta device, then a state dict from `safetensors::safe_load_file()` | GGUF; a safetensors reader | `zl_param()` and `zl_param_set()` for arrays; `zl_load_safetensors()` for files (§7.4) |

Every row maps to an existing upstream operation or to R code around
the graph; nothing needs a ggml patch, and nothing in the transformer
needs more than four dimensions. The parts that stay in R on doubles
are exactly the parts that zukzeit's reference implementations run in
float64 or outside the network, so leaving them there is what keeps the
parity fixtures meaningful (§6.9). Memory is not a constraint: TimesFM's
parameters are 925 MB in f32, its score matrices at full context 16 MB
per sequence, and its key-value cache 105 MB per sequence (§6.9), all
under the 4 GiB default of §6.6.

TimesFM's attention block in zulm's API, to show the fit (the full port,
with its decode loop, is zukzeit's):

```r
p   <- zl_params()                 # filled by zl_load_safetensors()
qkv <- zl_linear(p, "qkv", 1280, 3 * 1280, bias = FALSE)
out <- zl_linear(p, "out", 1280, 1280, bias = FALSE)
qn  <- zl_norm(p, "q.ln", 80, type = "rms", eps = 1e-6)
kn  <- zl_norm(p, "k.ln", 80, type = "rms", eps = 1e-6)
zl_param(p, "per_dim_scale", 80)

attend <- function(x, pos, mask) {  # x: c(1280, P, B); pos: i32 P;
  h <- qkv(x)                       # mask: c(P, P, 1, B), 0 or -Inf
  q <- zl_reshape(zl_view(h, c(1280, P, B), offset = 0),        c(80, 16, P, B))
  k <- zl_reshape(zl_view(h, c(1280, P, B), offset = 1280),     c(80, 16, P, B))
  v <- zl_reshape(zl_view(h, c(1280, P, B), offset = 2 * 1280), c(80, 16, P, B))
  q <- qn(zl_rope(q, pos, n_dims = 80, mode = "half"))
  k <- kn(zl_rope(k, pos, n_dims = 80, mode = "half"))
  q <- q * (1.442695 / sqrt(80) * zl_softplus(zl_param_tensor(p, "per_dim_scale")))
  q <- zl_permute(q, c(1, 3, 2, 4)); k <- zl_permute(k, c(1, 3, 2, 4))
  s <- zl_softmax(zl_crossprod(k, q), mask = mask, scale = 1)    # c(P, P, 16, B)
  a <- zl_crossprod(zl_cont(zl_t(zl_permute(v, c(1, 3, 2, 4)))), s)
  out(zl_reshape(zl_cont(zl_permute(a, c(1, 3, 2, 4))), c(1280, P, B)))
}
```

The same block with Toto's conventions changes four lines: the rotary
mode and `n_dims`, a multiplication by the xPos factor, `scale = 1 / 64`,
and `zl_scale(…, 1 / sqrt(fan_in))` after each linear map. Chronos-2's
group attention is the same block with the mask carrying the groups
and no rotary positions.

## 2. Scope

**In v0.1.0:**

- Tensors of f32 (compute and parameters) and i32 (indices), up to four
  dimensions, mapped to R arrays without reordering; conversion of R
  doubles to f32, f16 and bf16 for storage.
- Explicit graphs: an R function builds a graph once per input shape
  from tensor operations; the graph is then run many times with new
  data.
- The operations of §5.3, each with ggml's own backward pass where
  upstream has one, and the forward-only ones marked as such; among
  them the operations pretrained transformers need (§1.2):
  concatenation, padding, clamping, selection by a mask, cumulative
  sums, triangular masks, rotary positions in both conventions with
  partial rotation, and softmax with an additive mask broadcast over
  heads and batch.
- Filling a store by name from R arrays (`zl_param_set()`), for weights
  from any source, and safetensors import and export with the header
  parsed by `zujson` (§7.4), so Hub checkpoints load without a
  conversion step.
- f16 and bf16 parameters kept in memory, opt-in, with the precision
  cost of §6.9 stated; `zl_cast()` inside graphs.
- State tensors: persistent, writable tensors in a store that graphs
  read and write, which give a key-value cache for incremental decoding
  and a sliding context window without rebuilding graphs (§6.9), with
  the C table members both consumers asked for (§8; RFC 0032's
  `input_write_f32_at` is one of them).
- Forward-only flash attention (`ggml_flash_attn_ext`) for long
  contexts in sampling and inference graphs (§6.9).
- A graph cache keyed by input shapes in the R layer, so a decode loop
  whose shapes vary builds each shape once (§6.9).
- Training batches from Arrow streams and the table's `batch_fill`
  (§7.3), which RFC 0032 lists among zuflow's first-release features.
- Training with ggml's autograd and `ggml-opt`'s AdamW and SGD, gradient
  accumulation, a learning rate per step set from R, gradient clipping
  by global norm inside `ggml-opt`'s step through patch P7 (§9.3).
- Modules: linear, embedding, layer and RMS norm, multi-head attention
  with a causal mask, MLP with SwiGLU or SiLU, a transformer block,
  AdaLN modulation, sinusoidal time embedding.
- Losses: cross-entropy on logits, mean squared error, mean absolute
  error.
- GGUF save and load of a parameter store, with metadata.
- The CPU backend only, with ggml's own thread pool; interruptible
  between graph nodes.
- A registered C table (§8) for model packages.
- Counter-based random numbers from `zurand` drawn inside graphs
  (§7.2), and losses and metrics reported from per-row values summed in
  fixed order on the main thread (§7.1), so that reported numbers do
  not depend on the thread count.

The list is longer than the one the review of 2026-10-10 left, which
had cut v0.1.0 to the §1.1 model and the determinism contract. The
maintainer's decision of 2026-10-14 (D27) is that zulm reaches CRAN
once with everything zuflow 0.1.0 and zukzeit 0.1.0 need, rather than
in three releases that each consumer would have to wait for. The
cost is in §14 and the roadmap: fifteen stages instead of nine.

**In 0.2** (approved in RFC 0031 and unchanged; needed by neither
consumer's first release, so they wait):

- Exactly summed losses and metrics through `zuexact`'s table (§7.1),
  which adds invariance under regrouping to the fixed-order fallback.
- Forward-only FFT through `zufft` and differentiable DFT spectra
  (§7.5).
- `argsort` and `top_k`, forward-only, if a consumer asks; the ports
  of §1.2 sort on doubles in R.

**Later, when asked:**

- Saving and restoring optimiser state (AdamW moments) so that training
  resumes exactly; `zl_save()` persists parameters only (§6.6).
- GPU backends as opt-in `configure` switches: Vulkan first, then Metal
  (§17).
- Quantised weights (`Q8_0`, `Q4_0`) for inference, and mixed-precision
  training (f16 parameters with f32 master weights).
- Flash attention in training graphs, once upstream has its backward
  pass.
- Custom CPU operations registered from C through the table.
- FFT with a gradient, once patch P6 is upstream (§7.5); likelihoods
  needing special functions through `zuspecial` (§7.6); Highway kernels
  for zulm-owned code (§7.7).

**Never:**

- A Keras-style model API, tidymodels or mlr3 engines: the package stays
  a substrate; frameworks belong in packages built on it.
- ONNX import, single-cell analysis helpers, image generation (`sd2R`),
  and language-model inference with tokenisers, chat templates and
  quantised GGUF models (`llamaR`, `edgemodelr` and `localLLM` do
  that). Running a pretrained transformer's forward pass is in scope;
  the model layer around it is not.
- An R-level autograd tape. ggml's backward pass is the only one.
- Local changes to ggml that upstream would not accept.
- Samplers, optimal-transport coupling, atom smoothing and evaluation
  metrics: zuflow's (§7.8). Checkpoint catalogues, Hub downloads,
  architecture ports, forecast objects and framework adapters:
  zukzeit's (§7.9).

## 3. Position in the `zu*` family

| | zulm |
| --- | --- |
| Role | provider (C table); R API |
| R prefix | `zl_` |
| Info function | `zulm_info()` |
| Root condition class | `zulm_error` |
| Public C prefix | `zulm_` for the table; internal `zlm_` |
| From C | `R_GetCCallable("zulm", "zulm_api_v1")` |
| Hides symbols | yes (R3; a linker version script, §13.3) |
| Vendored code | ggml CPU subset at `v0.26.0`, plus patches; nothing else |
| Family imports | `zurand` (`Imports`, `LinkingTo`) for random numbers (§7.2) |
| `Depends: R` | 4.1 (R2) |
| Language | en-GB (R2) |

**Relationships, as decided rather than as hoped:**

- **`zuflow`** (RFC 0032) is the first consumer. It owns the
  event-sequence model of §1.1, its data representation, atom smoothing,
  samplers and evaluation, by the boundary rule of §7.8. It is written
  mostly in R against §5 and uses the C table (§8) for sampler loops and
  batch filling. zulm's modules stay generic.
- **`zukzeit`** (RFC 0002) is the second consumer: an inference engine
  for time-series foundation models that today ports each architecture
  to R `torch` and certifies it against pinned reference fixtures. On
  zulm it keeps everything but the tensor runtime: catalogue, Hub
  download, preprocessing, decode loops, quantile repair, forecast
  objects and adapters (§7.9). The maintainer has decided that zukzeit
  uses zulm only: its 0.1.0 drops `torch`, so every certified checkpoint
  must run on zulm 0.1.0 (D33). This design proves the path on
  zukzeit's smallest checkpoint before 0.1.0 ships (§15); the TimesFM
  2.5 and Chronos-2 ports are zukzeit's work on zulm's released API.
- **`zujson`** (0.1.0 on CRAN) parses the safetensors header (§7.4):
  an `Imports` entry with an R API, not a C table.
- **`zuexact`** (RFC 0023), **`zufft`** (RFC 0026) and **`zuspecial`**
  (RFC 0027) are optional companions reached through their C tables
  (§7) from 0.2 onwards; zulm works without each. Neither driving use
  case needs them: time embeddings are sine and cosine inside the
  graph, and the foundation models' preprocessing runs in R.
- **`zurand`** (0.1.0, first CRAN submission prepared) is the one hard
  family dependency: every random number zulm draws comes from its C
  table (§7.2). zulm cannot reach CRAN before zurand does; zurand is
  ahead in the pipeline, and its table is versioned and append-only,
  so the dependency is on `ZURAND_API_VERSION >= 3`.
- No other family package is required to adopt zulm.

**What exists elsewhere.**

- **`ggmlR`** (MIT, Yuri Baramykov; `DESCRIPTION` 0.8.7 at `7c22445`;
  the `github.com/cran` mirror shows 0.8.4 published 2026-08-21). A broad
  framework on a vendored, restructured ggml: `ggml.c` is split into
  `ggml-core.c`, `ggml-graph.c`, `ggml-ops-builders.c` and others, the
  quantisation code is split, and `GGML_MAX_DIMS` is 5 where upstream's
  is 4, so its `struct ggml_tensor` differs from upstream's. It adds
  operations upstream does not have (backward kernels for `NORM`, `GELU`,
  flash attention, Mamba's SSM and RWKV, `CONCAT`; ONNX quantised kernels;
  `REL_POS_BIAS`, `SCATTER_ELEMENTS`). Its `NEWS.md` records a migration
  to ggml 0.11.0 (upstream's tag of 2026-05-05) in 0.7.7, while
  `Makevars.in` still passes `GGML_VERSION "0.9.5"`. It offers a
  Keras-like sequential and functional API training through `ggml-opt`,
  and separately an autograd engine (`ag_*`) whose tape lives in R:
  gradients come either from R closures computing with `%*%` on host
  matrices or, by default since 0.8.5, from backward graphs that R code
  emits rule by rule (`R/ag_backward_graph.R`), not from ggml's
  `ggml_build_backward_expand`. It also has Vulkan with multi-GPU
  tensor and pipeline parallelism, ONNX import, GGUF, parsnip and mlr3
  engines, single-cell helpers, 796 exports, and a C callable
  `ggmlR_register_custom_op`. It exports its modified ggml headers in
  `inst/include` and builds a static `libggml.a` into `inst/lib`, which
  `llamaR` (`Depends` and `LinkingTo`) and `sd2R` (`Imports` and
  `LinkingTo`) link; `cayleyR` suggests it. Its `configure` disables SIMD
  flags unless `--with-simd` is given, and turns on OpenMP from R's
  `Makeconf`. Its `r_ggml_compat.h` is force-included into ggml's sources
  and turns `abort()`, `exit()` and `_Exit()` into `Rf_error()`.
- **`edgemodelr`** 0.4.3 vendors llama.cpp with its own ggml
  (`GGML_VERSION "0.9.7"`) and builds x86-64 with `-msse4.2` by default,
  overridable by an environment variable. **`localLLM`** 1.3.0 downloads
  a prebuilt llama.cpp library from GitHub releases at run time. Neither
  offers training.
- **`torch`** binds LibTorch (a large download at first use, which
  `zukzeit` inherits for every user who wants to run a 16 MB
  checkpoint); **`keras3`** and **`tensorflow`** go through Python.

**The incumbent is torch, not ggmlR.** CRAN downloads in the 30 days to
2026-10-09 (`cranlogs`): `torch` 45 789, `ggmlR` 606, `llamaR` 337. A
user choosing where to train a small model in R compares zulm with
torch first. The case to make to that user is the one §13.1 and §6.5
deliver: a CRAN binary that installs in under a minute with no
LibTorch download, trains at SIMD speed without flags, and reproduces
bit for bit at a fixed seed, with a C table other packages can build
on. For inference the case is the same and the user is a package: a
pretrained transformer on zulm installs with its package from a CRAN
binary, needs no runtime download, and gives the same bits on every
machine of one ISA variant, where zukzeit's fixtures today drift by
ulps between hosts (§1.2). What torch keeps is the GPU and the
ecosystem; a user who needs either should use it, and this design does
not try to follow. The case against ggmlR below is about the shared
layer, and is addressed to the maintainer, not to users.

**Why a second binding.** ggmlR and zulm want different things, and
the differences are structural rather than a matter of polish:

1. *Upstream fidelity.* ggmlR is a fork that has diverged in layout
   (five dimensions, split files) and content, and its upstream base is
   five months and 15 minor releases behind `v0.26.0`. zulm
   vendors upstream unmodified, by script, at a pinned tag, with each
   local change a numbered patch carrying an upstream pull request
   (§9). Bugs found in zulm go upstream; upstream fixes arrive at the
   next re-vendoring. The two backward-pass bugs found while drafting
   (§9.3) are examples of the traffic this enables.
2. *One autograd.* zulm has exactly one differentiation path, ggml's,
   so its gradients are upstream's gradients, tested where upstream
   tests them; ggmlR maintains its own rules in R.
3. *Failure semantics.* ggmlR's redirection turns a ggml assertion into
   `Rf_error()`, a `longjmp` from inside ggml, which may run on a pool
   thread and may cross C++ frames. zulm validates before ggml sees an
   argument, recovers from assertions during graph construction with a
   C++ exception that unwinds properly, and treats the rest as the
   fatal path it is (§13.2).
4. *A stable provider interface.* Packages that link ggmlR compile
   against its modified headers and link another copy of the library.
   zulm offers a versioned C table of opaque handles (§8), so a
   consumer never sees a ggml type and survives every re-vendoring.
5. *Speed on default builds.* With ggmlR's default flags, an x86-64
   build of ggml has no SIMD path at all; measured on upstream with that
   flag set, an f32 product runs at 1.6 GFLOP/s against 14–33 GFLOP/s for
   AVX2 builds (§13.1). zulm's first stage settles a run-time choice
   among builds so that CRAN's binaries are fast without user flags.
6. *Reproducibility as a contract* (§6.5), and a surface of about 80
   functions rather than 800.

What ggmlR has that zulm will not have in v0.1.0: GPUs, quantised
inference, many more operations and backward kernels, a high-level
training API, ONNX, and users. The two can be loaded in one session:
both hide their ggml symbols (ggmlR's state was not checked) and share
nothing but the GGUF file format.

## 4. Architecture

```text
R: zl_params(), modules, build(x) -> tensors, zl_graph(), zl_step()
   |
   v
R layer: shapes and types checked; build() runs once per graph, with
         zl_tensor handles; a backward-support pre-check walks the graph
   |
   v
C glue (src/zlm_*.c): external pointers, R vectors <-> f32/i32/f16/bf16,
         validation of every argument ggml would assert on
   |
   v
C++ shim (src/zlm_shim.cpp): extern "C"; calls ggml under try/catch;
         the ggml abort callback throws during graph construction
         (through ggml's C frames, on their default unwind tables);
         logs buffered
   |
   v
ggml (vendored, unmodified + patches): ggml-base (graph, autograd,
         alloc, backend, opt, gguf) and ggml-cpu (kernels, thread pool),
         one build per ISA variant (§13.1)
   |
   v
R results: doubles (losses, arrays), external pointers (stores, graphs)
```

- Three contexts per graph: parameters live in a store's static
  contexts, one backend buffer per allocation batch (§6.6, revised in
  Stage 2); inputs live in a static context of the
  graph's own with its own buffer, because `ggml-opt` asserts that
  inputs of static graphs are allocated statically
  (`ggml_opt_alloc`: "the inputs must be allocated statically"), and a
  statically placed input is what lets `zl_forward()` rewrite it in
  place; activations, gradients and optimiser state live in the
  graph's compute context, allocated by `ggml-opt` through a backend
  scheduler with the CPU backend alone. Every tensor `build()` returns
  by name is flagged with `ggml_set_output()` so that the scheduler
  does not reuse its memory before R reads it.
- R's API is never called from the shim, from a pool thread, or from
  inside a ggml call, so no `longjmp` crosses ggml or C++ frames.
- The same shim functions serve the R glue and the C table (§8).

## 5. Public R API (complete v0.1.0 surface)

### 5.1 Session, parameters, graphs, training

```r
zulm_info()            # ggml tag and commit, patches, ISA variant
                         # loaded, threads, zurand key and table
                         # version, the native hook's flags if any
                         # (§13.4)
zl_threads(n = NULL)     # get, or set and return the old value
zl_seed(seed = NULL)     # session key: an integer, or a zurand
                         # rng_key; get, or set and return the old

# parameter store (one backend buffer)
p <- zl_params()
zl_param(p, name, dim, init = zl_init_normal(sd = 0.02))
zl_init_normal(sd); zl_init_uniform(a); zl_init_zeros(); zl_init_ones()
zl_param_get(p, name)            # -> double array
zl_param_set(p, name, value)
zl_param_names(p)
zl_param_count(p)
zl_param_tensor(p, name)         # handle for use inside build();
                                 # what the modules use internally

# graphs
g <- zl_graph(build, p, inputs, loss = NULL, outputs = NULL)
zl_int(...)                      # marks an i32 input's dims in `inputs`
zl_forward(g, data, outputs = NULL)   # -> named list of double arrays
                                 # inputs are rewritten in place on
                                 # every call; the graph is built once
zl_graph_info(g)                 # nodes, ops, memory, backward ops

# optimisation
zl_adamw(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
         weight_decay = 0)
zl_sgd(lr = 1e-2, weight_decay = 0)
tr <- zl_trainer(g, opt, accumulate = 1L, clip_norm = NULL)
                                 # clip_norm: global-norm clipping in
                                 # ggml-opt's step, patch P7 (§9.3)
zl_step(tr, data, lr = NULL)     # one step; returns the loss (double)
zl_grad(tr, name)                # last gradient of a parameter
zl_reset(tr)                     # zero optimiser state

# persistence
zl_save(p, path, type = c("f32", "f16", "bf16"), metadata = list())
zl_load(path)                    # -> parameter store, metadata attribute

# random numbers (§7.2)
zl_rng_uniform(dim, stream, step = 0, offset = 0)  # zurand, in R;
zl_rng_normal(dim, stream, step = 0, offset = 0)   # f32 values
```

Weights, state tensors, inference graphs and Arrow batches (§6.9, §7.3,
§7.4):

```r
zl_safetensors_read(path, names = NULL)          # -> list of arrays
zl_load_safetensors(p, path, map = NULL)         # fill a store
zl_save_safetensors(p, path, type = "f32", metadata = list())
zl_param(p, name, dim, init, type = c("f32", "f16", "bf16"))
                                 # type: in memory, not only on disk
zl_cast(x, type)                 # inside build(); forward only
zl_state(p, name, dim)           # persistent, writable, not trained
zl_state_get(p, name)            # -> double array
zl_state_set(p, name, value, offset = 0)
zl_state_write(s, x, rows)       # inside build(): rows of state s <- x
zl_flash_attention(q, k, v, mask = NULL, scale = 1)   # forward only
gc <- zl_graph_cache(build, p, loss = NULL, outputs = NULL)
zl_forward(gc, data)             # builds a graph per new input shape
it <- zl_arrow_batches(stream, map, rows, na = c("error", "mask"),
                       last = c("drop", "pad"))
zl_next(it)                      # named list of arrays, or NULL
                                 # zl_step() and zl_forward() also
                                 # take an iterator as `data`
```

Planned for 0.2 (§7.5), so that consumers can write against the name
now:

```r
zl_spectrum(x, method = c("dft", "fft"))   # power spectrum by column
```

### 5.2 Modules

Each constructor declares its parameters in a store under a name prefix
and returns a function of tensors, used inside `build()`.

```r
zl_linear(p, name, n_in, n_out, bias = TRUE)
zl_embedding(p, name, n, dim)            # i32 indices -> vectors
zl_norm(p, name, dim, type = c("layer", "rms"), affine = TRUE,
        eps = 1e-5)
zl_attention(p, name, dim, heads = 1L, causal = TRUE)
zl_mlp(p, name, dim, hidden = 4 * dim, act = c("swiglu", "silu"))
zl_block(p, name, dim, heads, causal = TRUE, hidden = 4 * dim)
                                         # pre-norm attention + MLP
zl_adaln(p, name, dim, cond_dim)         # function(x, cond):
                                         # x * (1 + scale) + shift
zl_time_embedding(t, dim, max_period = 10000)   # no parameters
```

**As built in Stage 6 (2026-10-14).** Parameter names: a linear is
`<name>.weight` (`c(n_in, n_out)`) and `<name>.bias`, initialised
uniformly on `(-1, 1) / sqrt(n_in)` as PyTorch does; an embedding is
`<name>.weight` (`c(dim, n)`), N(0, 1), taking 0-based i32 indices of
any shape (an i32 input may now be reshaped, and the index check
follows the view to its input); a norm is `<name>.weight` (ones) and,
for layer norms, `<name>.bias` (zeros); attention is the linears
`<name>.q`, `.k`, `.v` and `.o` with biases; the MLP is `.gate`, `.up`
and `.down`; the block is `.norm1`, `.attn`, `.norm2` and `.mlp`, with
layer norms; AdaLN is two linears, `<name>.scale` and `<name>.shift`
(not one linear split in halves, which would need a strided view).
`zl_time_embedding()` concatenates `cos(t f)` and `sin(t f)`; its
frequencies are a constant tensor, which graphs now hold in a buffer of
their own, and `t` is repeated to the frequencies' width with
`ggml_repeat_4d`. A sum of two losses that report per-row values
(`ce + mse` in §1.1) reports the sum of both row sums, so the §1.1
model's reported loss is thread-invariant too.

### 5.3 Operations on tensors

Inside `build()`, inputs and parameters are `zl_tensor` handles. The
`Ops` group generic maps `+`, `-`, `*`, `/` and unary minus to ggml's
broadcasting operations, and a number on either side to a scale or an
added constant.

```r
# arithmetic and reductions
zl_scale(x, s); zl_sum(x); zl_sum_rows(x); zl_mean_rows(x)
zl_sqr(x); zl_sqrt(x); zl_exp(x); zl_log(x); zl_abs(x)
zl_sin(x); zl_cos(x)
zl_cumsum(x)             # along the first dim; forward only
# activations
zl_relu(x); zl_silu(x); zl_softplus(x); zl_gelu(x)   # gelu: forward only
zl_sigmoid(x); zl_tanh(x)                            # forward only
zl_swiglu(gate, up)
# products and shape
zl_crossprod(a, b)       # t(a) %*% b on the first two dims, batched
zl_reshape(x, dim); zl_permute(x, perm); zl_t(x); zl_cont(x)
zl_view(x, dim, offset = 0)
zl_concat(a, b, axis)    # forward only
zl_pad(x, after)         # zeros after each dim; forward only
zl_get_rows(x, idx)      # rows of a matrix by i32 index
# selection and clipping (forward only)
zl_where(mask, a, b)     # mask: 0/1 f32 tensor; mask * a + (1 - mask) * b
zl_clamp(x, lo, hi)
zl_tri(x, type = c("lower", "upper", "lower_strict", "upper_strict"))
# attention pieces and normalisation
zl_softmax(x, mask = NULL, scale = 1)   # mask: additive f32 of dims
                                        # c(n_kv, n_q, h, b), h and b
                                        # each 1 or the full size
zl_causal_mask(x)        # -Inf above the diagonal of each matrix
zl_rope(x, pos, n_dims, base = 10000,
        mode = c("interleaved", "half"))  # rotary positions; has a
                                          # backward (ggml_rope_ext);
                                          # n_dims < head dim rotates
                                          # the leading n_dims only
zl_rms_norm(x, eps = 1e-5); zl_layer_norm(x, eps = 1e-5)
# losses
zl_cross_entropy(logits, prob); zl_mse(pred, target)
zl_mae(pred, target)
# random draws inside the graph (forward-only, no gradient)
zl_noise_normal(dim, stream, offset = 0)   # zurand, at the step count
zl_noise_uniform(dim, stream, offset = 0)
```

`zl_crossprod()` is ggml's `ggml_mul_mat()` under the name of the R
function it equals: for `a` of dims `c(K, M)` and `b` of `c(K, N)` it
returns `crossprod(a, b)` of dims `c(M, N)`, batched over the third and
fourth dimensions with ggml's broadcasting of `a`. A linear layer stores
its weight as `c(n_in, n_out)` and computes `crossprod(W, x)`.

Forward-only operations are ordinary nodes that the backward pre-check
(§6.2) refuses on a gradient path with `zulm_unsupported_backward`.
They exist for the inference graphs of §1.2 and the sampling graphs of
§6.4, and each costs one upstream builder (`ggml_concat`, `ggml_pad`,
`ggml_clamp`, `ggml_cumsum`, `ggml_tri`, the `SIGMOID` and `TANH`
unaries) or a composition of operations that have gradients
(`zl_where()`, which therefore has one). `zl_rope(mode = "half")` is
ggml's `GGML_ROPE_TYPE_NEOX`, cosines on the first half of the rotated
dimensions and sines on the second, the convention of TimesFM,
Chronos-2 and most Hugging Face checkpoints; `"interleaved"` is
`GGML_ROPE_TYPE_NORMAL`, adjacent pairs, the convention of Toto and of
ggml itself, and the default because it is upstream's. `n_dims` below
the head dimension rotates the leading `n_dims` and leaves the rest,
as upstream documents (`n_dims = 4` on a head of 8 gives `[cscs0000]`).
**As built in Stage 3 (2026-10-14).** Indices given to `zl_get_rows()`
count from 0, as the i32 codes of §7.3 do, and are checked against
their bound on every `zl_forward()`, so the index must be a graph input.
`zl_tri(type = "lower")` keeps what `lower.tri(diag = TRUE)` selects
(ggml's `UPPER_DIAG`, since ggml calls the first dimension columns).
`zl_gelu()` gained `approximate = c("none", "tanh")`: the exact form
(`ggml_gelu_erf`) by default, as PyTorch's default, because ggml's tanh
form reads an f16 table and is accurate to about 1e-3 only. A number
divided by a tensor is `ggml_fill` then `ggml_div`. Only f32 tensors
enter operations: an f16 or bf16 parameter may be the weight of
`zl_crossprod()`, the table of `zl_get_rows()` or the input of
`zl_cast()`, and nothing else, because ggml's CPU kernels assert on
other type combinations at compute time, where nothing can be caught.
Operands that are not contiguous (after `zl_permute()` or `zl_t()`) are
made contiguous before any kernel that is not a view. Forward-only
graphs are allocated with ggml's graph allocator and run on the CPU
backend. `zl_forward()` evaluates a copy of the graph (as `ggml-opt`
evaluates copies of its graphs): `ggml-opt`'s copies inherit the
original nodes' data pointers, so the original must stay unallocated
for training, while the allocator may reuse memory between the copy's
activations. Stage 4 trains on the original through `ggml-opt`.

Elementwise maximum and minimum, powers and `flip` are compositions in
R of the above (`b + relu(a - b)`, `exp(b * log(a))`, `get_rows` with
reversed indices) and get no function of their own until a consumer
asks; `zl_softmax()`'s mask is passed to `ggml_soft_max_ext`, whose
kernel broadcasts it over the heads and batch dimensions it does not
fill, so one `c(n_kv, n_q, 1, B)` mask serves all heads.

## 6. Semantics

### 6.1 Arrays, dimensions and types

- An R array of `dim = c(d1, d2, d3, d4)` is ggml's tensor with
  `ne = {d1, d2, d3, d4}`: both are column-major with the first index
  fastest, so data moves without reordering. Vectors are one-dimensional;
  more than four dimensions is `zulm_shape_error`.
- Inputs and parameters are f32 (parameters may be f16 or bf16 in
  memory, opt-in, §6.9). R doubles are rounded to nearest; values
  beyond the f32 range become infinite with a warning of class
  `zulm_range_warning`. Index inputs are i32 from R integers; `NA` is
  refused.
- Results are returned as doubles, which hold every f32 exactly.
- `zl_save(type = "f16" | "bf16")` converts with ggml's row converters
  (`ggml_fp32_to_fp16_row`, `ggml_fp32_to_bf16_row`); `zl_load()` widens
  to f32.
- Long vectors: element counts are `R_xlen_t` in R and `int64_t` in
  ggml, with no `int` between them; the memory ceiling (§6.6) is the
  practical limit.

### 6.2 Graphs

- `zl_graph()` calls `build(x)` once, with `x` a named list of input
  handles of the declared shapes, and records the returned tensors by
  name, each flagged as an output (§4). `build()` may return a tensor
  or a named list; `loss` names a scalar output. The R function is not
  called again: a different input shape needs another graph.
- `zl_forward()` and `zl_step()` copy `data` (a named list of arrays
  matching the declared shapes exactly) into the graph's inputs, run it,
  and copy the requested outputs back. Inputs may be rewritten between
  computations any number of times without rebuilding the graph, from R
  or through the C table; every sampler loop in RFC 0032 and every
  decode loop in zukzeit relies on this, and it is a tested contract,
  not an implementation detail.
- Before any ggml backward construction, zulm walks the forward graph
  and refuses, with `zulm_unsupported_backward` naming the operation
  and the R call that created it, any node that needs a gradient and
  whose operation lacks a backward pass in the vendored ggml (including
  patches). ggml would otherwise abort in `ggml_compute_backward`.

### 6.3 Autograd and compositions

Gradients are ggml's. Where upstream's backward pass is missing or
asserts, zulm composes the operation from ones that work, until a
patch (§9.3) lands upstream:

- **Layer normalisation** is `rms_norm(x - mean(x))`, exactly equal to
  LayerNorm's normalisation, with the mean computed as
  `scale(sum_rows(x), 1 / n)`. Upstream has no backward for
  `GGML_OP_NORM`, and `ggml_mean`'s backward asserts when its input has
  another consumer (§9.3), which is always the case here.
- **Broadcast subtraction** with a gradient on the broadcast operand is
  emitted as `a + neg(b)`, since `ADD`'s backward reduces with
  `ggml_repeat_back`. Broadcast division with a gradient on the divisor
  is refused with `zulm_unsupported_backward` until patch P2 is
  vendored.
- **GELU** is forward-only until upstream has its backward; the modules
  default to SwiGLU and SiLU, which have one.
- **Attention** is the plain form: `crossprod(k, q)`, scale, causal
  mask (`ggml_diag_mask_inf`), softmax, `crossprod(t(v), a)`. Upstream's
  softmax backward accepts a mask that needs no gradient, so
  `zl_softmax(mask =)` works in training. Heads are a reshape to
  `c(dh, heads, T, B)` and a permutation, within four dimensions.
  Rotary positions in either convention (§5.3) apply to `q` and `k`
  before the products and have ggml's backward (`ROPE_BACK`).

### 6.4 Training and sampling

- `zl_trainer()` builds the gradient and optimiser graphs once through
  `ggml-opt` with static graphs, loss type "sum" over the declared
  scalar loss, and `opt_period = accumulate`.
- `zl_step(lr =)` sets the learning rate for that step through `ggml-opt`'s
  per-evaluation parameter callback; the callback reads a C struct the
  glue filled before the call and never calls R.
- `clip_norm` cannot be applied from outside `ggml-opt`: `ggml_opt_build`
  duplicates the gradient graph, appends one `ggml_opt_step_adamw` (or
  `_sgd`) node per parameter reading the gradient accumulator directly,
  and computes the whole as one graph, with no hook between the
  backward pass and the step and no clipping anywhere in
  `ggml-opt.cpp`. Patch P7 (§9.3) adds the clip threshold to the
  per-step optimizer parameters and the global-norm nodes to the step
  graph; zulm passes the threshold through the same callback as the
  learning rate. zulm does not build an optimizer graph of its own.
- Batching and shuffling are the caller's: the user passes each batch,
  as R arrays or from an Arrow stream (§7.3); noise and timesteps come
  from Philox inside the graph (§7.2). zulm does not use
  `ggml_opt_dataset`, whose shuffling uses `std::shuffle` with a
  default-seeded `std::mt19937`; `std::shuffle`'s output is not
  specified by the C++ standard, so it differs between libstdc++ and
  libc++.
- **As built in Stage 4 (2026-10-14).** `zl_trainer()` runs the
  backward pre-check, flags every f32 parameter the loss depends on with
  `ggml_set_param()`, and calls `ggml_opt_init()` with static graphs on
  the graph's original nodes (`ctx_compute` is the graph's context,
  `inputs` its first input, `outputs` the loss, loss type "sum",
  `opt_period = accumulate`), on a CPU backend scheduler of its own.
  Half-precision parameters on a gradient path are `zulm_type_error`.
  `ggml_opt_init()` builds its forward graph at
  `GGML_DEFAULT_GRAPH_SIZE`, so a trained graph has at most 2048 nodes
  between its parameters and its loss; an inference graph has no such
  limit. `zl_grad()` does not read `ggml-opt`'s gradients, which with
  `accumulate = 1` exist only inside its private graph copies: it
  builds a backward graph of its own on a copy of the forward graph with
  `ggml_build_backward_expand()`, allocates it with the graph allocator
  and returns the gradient at the current parameters on the inputs last
  written. A number divided by a tensor became `scale_bias(x, 0, c) / x`,
  because `ggml_fill` has no backward (an infinite element of `x` now
  gives NaN, not 0). `zl_crossprod()` with a weight broadcast over `b`'s
  fourth dimension has no gradient for the weight (ggml's `MUL_MAT`
  backward reduces over the third only) and is refused when the weight
  needs one. Training is bit-identical at one, two and four threads
  with llamafile kernels and `clip_norm` (a test), which settles
  question 7's training-step condition.
- Sampling (rectified-flow Euler steps, categorical draws) runs in R
  over `zl_forward()` calls, or in zuflow's C loops over the table;
  random numbers come from Philox, keyed by `zl_seed()`.

### 6.5 Determinism

Measured in the drafting session with upstream at `v0.26.0`:

- `mul_mat`, `soft_max`, `sum`, `sum_rows`, `mean` and `rms_norm` gave
  bit-identical outputs at one, two, three and four threads on a
  1 000 × 333 input.
- `cross_entropy_loss` did not: its forward pass splits rows across
  threads and adds per-thread partial sums, so the loss value depends on
  the thread count in the last bits. Its backward is elementwise.
- 1 000 AdamW steps of the §1.1 model ended with bit-identical
  parameters at one and two threads.
- Builds for different instruction sets give different bits (baseline,
  AVX2 and AVX-512 builds ended training with different parameter
  hashes); enabling the llamafile matrix kernels changes bits too.

zulm's contract follows: **for a given seed, inputs, ISA variant and
thread count, results are bit-identical across runs**; parameters are
also identical across thread counts for graphs whose gradients avoid
thread-split reductions, which includes every module in §5.2.
Reported losses and metrics are summed by zulm from per-row values
(§7.1), so they too are identical at any thread count, and from 0.2,
with `zuexact`, also under any grouping into micro-batches. Every random draw
is a counter-based value from `zurand`, a function of the session key,
stream, step and index (§7.2), so draws do not depend on threads or
batch size either, and, unlike everything the kernels compute, they are
also identical across platforms and ISA variants. `zulm_info()` reports
the ISA variant, because a result is reproducible only within one.

### 6.6 Memory and object lifetimes

- Stores, graphs and trainers are external pointers with finalisers. A
  graph keeps its store alive by reference; a trainer keeps its graph.
- **Stage 2 (2026-10-14): a store is a list of allocation batches.**
  Each `zl_param()` call and each `zl_load()` allocates one batch: a
  ggml context for the batch's tensor metadata and one CPU backend
  buffer for its data. A single buffer for the whole store, as §4 first
  described it, would have to move every tensor whenever a parameter is
  added (or over-allocate), and a graph already built on the store would
  then point at freed memory. Batches never move, so tensor addresses
  are stable for the store's life, and loading a file is one batch. The
  scheduler and `ggml-opt` accept parameters spread over several CPU
  buffers.
- Memory is allocated when a store is created and when a graph is
  built; nothing grows during `zl_step()`. Each allocation is checked
  against `getOption("zulm.max_bytes")` (default 4 GiB) first;
  exceeding it is `zulm_memory_error`.
- **Stage 4 (2026-10-14): memory R cannot see.** A graph and a trainer
  hold memory outside R's heap, so R's garbage collector, which counts
  only its own, let unreferenced ones pile up: the gradient tests peaked
  at 1 GB, and valgrind's runner ran out of memory. Each trainer's
  scheduler was also sized for 65 536 nodes, though `ggml-opt`
  duplicates every training graph at `GGML_DEFAULT_GRAPH_SIZE`; it is
  now sized at twice that, which cut the peak to 160 MB. Under valgrind
  with origin tracking every allocated byte costs memory, touched or
  not, and the tests still passed 15 GB: graphs, the backward pass and
  trainers reserved contexts for the largest graph (16 384 nodes, about
  12 MB each). A graph's build context now starts at 256 tensors and
  doubles when full; the finished graph, its forward copy and the
  backward pass's contexts are sized from the graph's own node count;
  and a trainer gives `ggml-opt` a context of its own sized for
  `GGML_DEFAULT_GRAPH_SIZE`, and the scheduler's reserve for split
  inputs is set by `GGML_SCHED_MAX_SPLIT_INPUTS` (below). zulm also
  counts what live graphs and trainers hold, and `zl_graph()` and
  `zl_trainer()` run a full collection first when that count has
  doubled since the last one and passed 128 MB.
- Objects are not serialisable: `saveRDS()` of a store keeps a dead
  pointer, and use after reload is `zulm_stale_error`. A store made by
  one ISA variant is `zulm_stale_error` under another (after a forced
  `options(zulm.variant =)` reload); variant shared objects are never
  unloaded, so such a store is still freed correctly. A store survives
  `fork()`: its memory is the child's copy and it holds no thread, so
  only objects that own threads (Stage 7) are made stale by a fork. `zl_save()` is
  the persistence path, and it persists parameters only: a trainer's
  AdamW moments and step count are not saved, `ggml-opt` exposes no
  public accessor for them, and training resumed from a saved store
  starts with fresh optimiser state. The documentation says so; exact
  resumption is listed under Later (§2).

### 6.7 Threads and interruption

- `zl_threads()` starts at 1. A request is capped at the available cores
  and, when `_R_CHECK_LIMIT_CORES_` is set, at 2.
- zulm owns one persistent ggml thread pool per session
  (`ggml_threadpool_new`, attached with
  `ggml_backend_cpu_set_threadpool`), recreated when the size changes.
  Without one, ggml creates and joins a disposable pool on every graph
  computation. Polling (`poll`) starts at 0, which makes idle workers
  sleep rather than spin; Stage 7 measures its cost against ggml's
  default of 50.
- Oversubscription is severe with ggml's hybrid spinning: four threads
  on the two-core drafting machine ran the §1.1 step more than seven
  times slower than two. The cap above is therefore a correctness
  matter for shared machines, not only politeness.
- Fork safety: the pool records the creating process id; a forked child
  abandons the inherited pool without joining and creates its own on
  first use, as zufft does (RFC 0026 §6.5).
- Interrupts: zulm installs the CPU backend's abort callback, which
  ggml calls between graph nodes on worker 0, the calling thread (read in
  `ggml_graph_compute_thread`). It checks for an interrupt with
  `R_ToplevelExec()` and returns true to stop; zulm then signals
  `zulm_interrupt` after the compute returns. A trainer interrupted
  mid-step has undefined optimiser state for that step and says so.

### 6.8 Logging

ggml's log goes through `ggml_log_set()` into a buffer; zulm emits
buffered lines as R messages of class `zulm_message` after each call
returns, at a level set by `options(zulm.log = "warn")`. Lines from
pool threads are buffered under a mutex and never reach R directly.
(Stage 3: the buffer holds level-tagged records, 64 KB at most, with a
note when it overflowed; ggml's writes to stdout and stderr, which
`zlm_compat.h` catches, are level "info"; levels are "debug", "info",
"warn", "error" and "none".)

### 6.9 Inference graphs

Running a pretrained model is `zl_forward()` on a graph with no loss,
with weights filled by name. The items below are what incremental
decoding and large checkpoints add to that; each is forward-only and
leaves the training path untouched.

- **State tensors.** `zl_state(p, name, dim)` declares a persistent
  tensor in a store's buffer that graphs read like a parameter (never
  trained, never saved) and write with `zl_state_write(s, x, rows)`,
  which emits `ggml_set_rows` into it; the node is flagged as an output
  so the scheduler computes it. A key-value cache is two state tensors
  per layer of fixed capacity (the model's maximum context), one graph
  per count of new positions, and an additive mask input that excludes
  unfilled slots. TimesFM's decode loop, which today concatenates
  growing caches in `torch`, becomes one encode graph for the context
  and one decode graph per 128-value block that writes its keys and
  values at rows `n_filled + 1 … n_filled + n_new` and attends over the
  capacity under the mask. Capacity is the caller's and memory is
  allocated once: TimesFM 2.5 at full context is 20 layers × 2 × 512
  × 1 280 × 4 bytes, 105 MB per sequence in f32. A sliding window is
  the same mechanism from the other side, `zl_state_set(offset =)` in
  R and `state_write_f32_at` in C (§8), which is what RFC 0032 asked
  for as `input_write_f32_at`; that member exists too, for graphs that
  keep the window as an input. `zl_state_set()` without an offset
  resets a state; the determinism contract (§6.5) covers states, since
  they are plain tensors.
- **Fixed capacity, not a graph per fill.** Attending over the full
  capacity under a mask costs capacity × n_new per head per layer
  whatever the fill. At 512 slots that is small beside the projections,
  so zulm builds no graph per fill level and keeps the number of
  graphs equal to the number of distinct `n_new`.
- **Flash attention.** `zl_flash_attention()` is `ggml_flash_attn_ext`,
  forward-only (`FLASH_ATTN_BACK` is not in `ggml_compute_backward`),
  with the mask converted to f16 and padded as upstream's kernel
  requires and `q`, `k`, `v` in upstream's layouts (`[d, n, heads,
  batch]`, `v` not transposed). It holds no score matrix, so the 10 240
  -event contexts of §1.1's evaluation (§17) attend in memory linear in
  the context. Stage 10 checks its agreement with the plain
  form within f32 tolerance, and whether its kernel is thread-invariant
  (§17), before any consumer graph uses it; the plain form stays the
  reference and the training path.
- **Weight types.** `zl_param(type = "f16" | "bf16")` keeps a parameter
  in that type in memory, and `zl_load()` and `zl_load_safetensors()`
  fill it without widening; `ggml_mul_mat` takes such weights directly.
  The cost is precision, not only storage: ggml's CPU kernels compute
  the dot product in the weight's type and convert the f32 activations
  to it first (`vec_dot_type` of `F16` is `F16`, of `BF16` is `BF16`),
  so products differ from an f32 run by about 1e-3 relative for f16 and
  more for bf16, far beyond the ulp-level budget that zukzeit's parity
  fixtures allow. f32 is therefore the default and the only type the
  parity acceptance (§15) uses; f16 halves a 925 MB checkpoint for
  users who accept the difference, which the documentation quantifies,
  and `zulm_info()` and `zl_param_names()` report the types a store
  holds. `zl_cast()` converts activations inside a graph (`ggml_cast`),
  forward-only. Quantised types are under Later.
- **Graph cache.** `zl_graph_cache()` wraps `zl_graph()` with a cache
  keyed by the input shapes, so a loop whose shapes vary (horizons,
  batch sizes, `n_new`) pays graph construction once per shape; entries
  are evicted least recently used when the store's and graphs' memory
  would pass `getOption("zulm.max_bytes")`. It is an R environment;
  the C table does not see it (§17).
- **Preprocessing stays in R.** Running statistics in `double`
  (TimesFM's reversible normalisation, Toto's Welford scaler in
  float64), `asinh` and `sinh`, quantile-crossing repair and finiteness
  checks are done on R doubles outside the graph, as §1.2 lists. The
  reference implementations compute them in float64 or outside the
  network, so moving them into f32 graphs would change the numbers the
  fixtures pin; and they are a few vector operations per patch, which
  R does in microseconds.

## 7. Companion libraries

zulm composes with family packages through their versioned C tables
rather than vendoring their libraries: each is a `Suggests` entry, looked
up with `R_GetCCallable()` on first use, checked by `struct_size`, and
replaced by a fallback when absent, so zulm installs and works without
any of them. `zulm_info()` reports which companions are active, and
the tests run both paths. The one exception is `zurand`, a hard
dependency (`Imports` and `LinkingTo`) rather than a companion, because
the determinism contract cannot have a fallback (§7.2); `zujson` is
the other, for the safetensors header (§7.4). Only small, stable
things are vendored: ggml itself and the Arrow C data interface
definitions. No JSON library is vendored: the header goes through
`zujson`,
the family's CRAN package, which has an R API and no C table, so it is
reached from R rather than through `R_GetCCallable()` (§7.4). The
designs below were approved before review and stay as written; only
their release moved (§2).

| Need | Source | How | Release |
| --- | --- | --- | --- |
| Thread-invariant loss and metric sums | per-row values, fixed-order `double` | zulm | 0.1.0 |
| Exact sums, invariant under regrouping | `zuexact` (RFC 0023) | table | 0.2 |
| Counter-based random numbers | `zurand` (table version 3) | table, `Imports` | 0.1.0 |
| Batches from Arrow streams | Arrow C data interface | vendored | 0.1.0 |
| safetensors header | `zujson` 0.1.0 (CRAN) | R API, `Imports` | 0.1.0 |
| FFT | `zufft` (RFC 0026) | table | 0.2, forward-only |
| Special functions | `zuspecial` (RFC 0027) | table | Later |
| Faster f32 products | BLIS through `ggml-blas` | — | rejected |
| zulm's own SIMD kernels | Highway | — | Later |
| Samplers, transport, metrics | `zuflow` (RFC 0032) | — | not here |

### 7.1 Exact reductions with zuexact

The drafting session found one thread-dependent result (§6.5): the value
of `ggml_cross_entropy_loss`, whose CPU kernel (`ops.cpp`) gives each
thread a block of rows and adds the per-thread partial sums. The other
sums on a training step, read in the source, are thread-invariant but
not exact: `ggml-opt` builds the loss of type "sum" with `ggml_sum`,
whose kernel runs on thread 0 alone with the SIMD width of the build;
`ggml_opt_result_loss` adds per-batch losses on the host in `double`,
in batch order; the cross-entropy and AdamW backward passes are
elementwise.

Design. Gradients keep ggml's path: `zl_cross_entropy()` still builds
`ggml_cross_entropy_loss`, whose backward does not read its forward
value. Every reported number (losses, metrics, gradient norms reported
to the user) is computed instead from per-row values: a forward-only
custom operation writes one value per row, each row on one thread, and
the shim adds the rows on the main thread after the computation. The
custom node has no gradient (it is not an ancestor of the loss), so
`ggml_compute_backward` returns early for it and no ggml patch is
needed; Stage 5 confirms this in `ggml-opt`'s graph duplication.

The adding is `zuexact_api_v1` when `zuexact` is installed (`acc_size`,
`init`, `addv`, `round` from RFC 0023 §7; f32 values widen to `double`
exactly): the result is the correctly rounded sum, the same at any
thread count and under any grouping, so a loss reported with
`accumulate = 4` equals the one reported with `accumulate = 1`. The
fallback is pairwise summation in `double` in fixed row order: still
the same at any thread count, but not under regrouping. Hence
`Suggests`, not a hard dependency: the fallback already meets §6.5's
thread contract, and zuexact adds exactness and grouping invariance.
v0.1.0 ships the per-row operation and the fallback, which the
determinism contract needs; the `zuexact` lookup is 0.2, because
neither consumer's first release needs it (RFC 0032 §3 says zuflow's
own metrics do not).

In-graph sums that feed back into training (the global norm used by
`clip_norm`, `ggml_sum` of the loss) stay ggml's: invariant to threads,
not to ISA variants. Exact sums do not make results identical across
variants, because the matrix kernels differ too.

### 7.2 Counter-based random numbers

ggml has no random number generation in its compute path: the only
generator in the vendored sources is `ggml-opt`'s `std::mt19937` for
dataset shuffling, which zulm does not use (§6.4), and there is no
dropout operation. R's generator cannot be called from pool threads, and
its stream depends on the order of draws.

zulm draws through `zurand`, the family's stateless generator: every
value is a pure function of an immutable 128-bit key and the draw's
position, so a draw is independent of threads, batching, call order and
backend. RFC 0031 could not find this package (it looked under an old
name) and planned to vendor Random123's Philox headers with a
Box–Muller transform of its own; zurand makes both unnecessary and
removes the one open question that plan left, whether normals agree
across platforms. What zurand provides, read from `inst/include/zurand.h`
at `ZURAND_API_VERSION 3` and its `NEWS.md`:

- **Engines.** Philox4x64-10 and Threefry4x64 directly, or
  xoshiro256++ reseeded from Philox every 512 values (zurand's default
  for bulk speed). The engine is recorded in the key.
- **Keys.** `rng_key(seed)` in R; `rng_fold(key, data)` derives a key
  from a name or a number; in C, `fold_int(key, int64)` derives from a
  whole number with no R call. A key carries a stream version, so a
  stream can be corrected after release without changing what existing
  keys produce.
- **Fills.** `fill_uniform_at(key, start, n, min, max, double *out)`
  and `fill_normal_at(key, start, n, mean, sd, method, double *out)`
  give positions `start … start + n − 1` of the key's stream; every
  member but `key_get` calls no R API, allocates nothing and may run on
  any thread. Uniforms are `(m + 1/2) 2^-52` for a 52-bit `m`, strictly
  inside (0, 1); normals are a ziggurat on NumPy's tables with zurand's
  own port of fdlibm's `exp()` and `log1p()` and no fused multiply-add,
  so they are bit-identical on Linux, macOS and Windows, x86-64 and
  arm64 (zurand's own CI checks this).

zulm's use of it:

- **Session key.** `zl_seed()` holds one `rng_key` (an integer seed, or
  a key the user made with `rng_key()` or `rng_fold()`; the default is
  drawn once from R's RNG so `set.seed()` still governs a session),
  created with the `philox4x64` engine: a pool task that starts
  mid-stream then pays one block, where xoshiro would regenerate a
  chunk from its first position. `zulm_info()` prints the key.
- **Streams.** A stream is a name: a parameter's name for its
  initialisation, an input's name for noise, zuflow's `zuflow.x0` and
  the like. Its key is `rng_fold(session, name)`, derived in R when the
  store or graph is built and kept as a `zurand_key` struct in the
  graph's C data; adding a parameter does not shift the draws of
  others. The step's key is `fold_int(stream_key, step)` in C, with
  `step` the trainer's step count (0 for initialisation); the element's
  position is its index in ggml order plus the caller's offset, so draw
  `i` of example `j` does not depend on batch size. The counter layout
  of RFC 0031 (`{index, step, stream}`) is realised by two folds and a
  position, which is zurand's own structure.
- **Values.** zulm receives doubles and rounds to f32. A uniform in
  (0, 1) rounded to nearest f32 can become exactly 1 when it lies above
  1 − 2^-25, so uniforms are rounded towards zero instead, which keeps
  them in (0, 1); normals are rounded to nearest. The rounding rule is
  zulm's and documented, so a user who draws the same key with
  `rng_uniform()` in R knows how to reproduce the f32 values.
- **Where.** A forward-only custom operation inside the graph (noise,
  timesteps; no gradient needed) calls `fill_*_at` from each pool task
  for its slice of elements; the same calls on the main thread serve
  parameter initialisation and `zl_rng_uniform()` and `zl_rng_normal()`
  in R. zurand's own OpenMP fills are not used (its thread count does
  not change values, but zulm's pool already owns the threads).

**What zulm does not need from zurand, and two things it would take.**
Nothing in version 3 is missing for v0.1.0: name-derived keys are made
in R once, step keys by `fold_int()`, fills from any thread by
position, and long vectors through `uint64_t` positions and `size_t`
lengths. Two appended members would make the integration tighter and
are proposed to zurand, not required of it (opened 2026-10-14 as
zurand#63 and zurand#64): `fold_bytes(key, const void *, size_t)`, so
that a stream name can be folded in C and the table members of §8 can
take a name rather than a pre-folded key; and `fill_uniform_f32_at()`
and `fill_normal_f32_at()`, so that the f32 rounding rule lives in one
package and R users of `zurand` get the same f32 values as zulm's
graphs without knowing the rule. Until then both are zulm's. §11 tests that zulm's values equal `rng_uniform()` and
`rng_normal()` from R, rounded by the stated rule, on every CI platform.

**As built in Stage 5 (2026-10-14).** The step a graph's draws use is
the graph's step counter: `zl_step()` sets it to the trainer's count of
steps (1 for the first) before each evaluation, and `zl_forward()` uses
the last value (0 before any training). A noise node is a
`ggml_custom_4d` operation over all pool tasks, each filling a slice of
elements by position, so a draw is the same at any thread count and
under any split of a batch with matching offsets. `zl_rng_uniform()`
and `zl_rng_normal()` in R call the same C code, so the two agree bit
for bit. At load, zulm reads zurand's table and refuses a version below
3 with `zulm_companion_error`.

The per-row losses of §7.1 are built by `zl_cross_entropy()`,
`zl_mse()` and `zl_mae()` when their operands have the same shape: a
custom node computes each row in `double` on one thread, writes it to
its output and to a buffer it owns, and joins the loss as
`0 · sum(step(rows))`, which adds exactly 0 and whose source ggml never
differentiates (`ggml_step` is in `ggml_build_backward_expand()`'s
ignored sources), so the node is computed inside `ggml-opt`'s graphs
without a gradient. `zl_step()` and `zl_forward()` report the buffer's
pairwise sum in fixed order, scaled as the loss is; a measured
cross-entropy over 512 rows was identical at one, two, four and eight
threads while `ggml_cross_entropy_loss`'s own value changed in the last
digits. A loss composed by the user (say a sum of two losses) reports
ggml's value.

**Sequencing.** zurand is not on CRAN yet (its first submission is
prepared); zulm's own submission follows it, as zuflow's follows
zulm's.

### 7.3 Batches from Arrow streams

Training data for the driving use case lives in DuckDB, Parquet and
Delta tables. The Arrow C data interface is the stable way to hand such
data across libraries: three C structs (`ArrowSchema`, `ArrowArray`,
`ArrowArrayStream`) whose definitions the Arrow specification publishes
for copying, under Apache-2.0, guarded by `ARROW_C_DATA_INTERFACE` and
`ARROW_C_STREAM_INTERFACE` (as in nanoarrow's `nanoarrow.h`). zulm
vendors those definitions and nothing else of Arrow.

- **Input.** An R external pointer of class `nanoarrow_array_stream`
  holding an `ArrowArrayStream *`; nanoarrow's `inst/include/nanoarrow/r.h`
  reads it with `Rf_inherits()` and `R_ExternalPtrAddr()`, and zulm
  does the same itself, so `nanoarrow` (0.9.0 on the CRAN mirror,
  Apache-2.0) is `Suggests`, not `LinkingTo`. Streams from DuckDB or
  `arrow` reach it through `nanoarrow::as_nanoarrow_array_stream()`;
  which DuckDB function yields a stream most directly is unverified.
- **Types.** float32 copied; float64 rounded to f32; int8, int16 and
  int32 to f32 features or i32 indices as mapped; int64 to i32 with a
  range check; boolean to 0 and 1; dictionary-encoded columns (event
  kinds) to 0-based i32 codes, with the dictionary required identical in
  every chunk or mapped through levels given up front. Other types are
  refused.
- **Nulls.** An error naming column and row by default; `na = "mask"`
  writes 0 and fills a companion mask input with 0 and 1.
- **Chunks.** Record batches of any size are accumulated into batches of
  exactly `rows` rows; the final partial batch is dropped or padded with
  a mask, by option.
- **Copies.** One: columns are interleaved into feature-major tensors
  (`c(features, rows)`), which no Arrow layout matches.
- **Threads.** `get_next` may call into the producer (DuckDB), so the
  stream is read and released only on the main thread.

Windowing rows into sequences of events belongs to zuflow (§7.8).

RFC 0032 lists Arrow input among zuflow's v0.1.0 features through this
iterator, which is why it is in zulm's v0.1.0 (D27).

### 7.4 safetensors

safetensors is the format in which PyTorch-trained weights travel.
Read from `huggingface/safetensors` at `e246a25` (2026-09-22,
Apache-2.0), the format is: 8 bytes holding `N`, an unsigned
little-endian 64-bit integer; `N` bytes of UTF-8 JSON that must begin
with `{` and may be padded with spaces, mapping each tensor name to
`dtype`, `shape` and `data_offsets` `[begin, end)` relative to the byte
buffer, plus an optional `__metadata__` map of strings to strings; then
the byte buffer. Duplicate keys are disallowed.

- **Parser.** The header is small (TimesFM 2.5's 232 tensors need a
  few tens of kilobytes) and its grammar is restricted. zulm reads the
  eight-byte length and the header bytes in C, hands the header to R,
  parses it with `zujson` (0.1.0 on CRAN, the family's JSON package,
  which vendors yyjson itself; `Imports`, since a substrate that
  zukzeit builds on cannot treat loading weights as optional), validates
  the list in R, and then reads the byte buffer in C straight into the
  store's tensors, one read per tensor in file order, with no R copy of
  the data. Writing goes the same way: `zujson` serialises the header,
  C writes the buffer. RFC 0031
  planned to vendor yyjson (760 KB of source) because it found no
  family JSON package; `zujson` is that package, published since, so
  nothing is vendored (D36). The CRAN `safetensors` package (0.3.0,
  mlverse) loads only into `torch` or `pjrt` tensors through its
  `framework` argument and cannot fill a zulm store, so zukzeit's
  current reader does not carry over.
- **Checks.** `N` within the file and below a cap, offsets in range,
  ordered, non-overlapping and covering the buffer, known dtypes, and
  element count × size equal to `end - begin`; failures are
  `zulm_file_error`, and a `zufuzz` target covers the header. A header
  that `zujson` rejects is `zulm_file_error` with `zujson`'s message.
- **Types.** F32, F16 and BF16 to f32; F64 to f32 with
  `zulm_range_warning` when needed; I32 and I64 to index arrays;
  others refused.
- **Layout.** Shapes are row-major, so `shape = c(s1, …, sk)` is
  zulm's `dim = c(sk, …, s1)` with the same bytes. PyTorch's
  `nn.Linear.weight` of shape `[out, in]` is therefore `c(in, out)`,
  exactly `zl_linear()`'s convention, with no transposition.
- **API.** `zl_safetensors_read(path, names = NULL)` returns arrays;
  `zl_load_safetensors(p, path, map = NULL)` fills a store by a name
  map; `zl_save_safetensors(p, path, type, metadata)` writes one.
- **Purpose.** Loading Hub checkpoints for zukzeit without a conversion
  step (§1.2; a `map` from checkpoint names to store names is the
  port's), export of zulm models, and import of PyTorch weights for
  users. Cross-framework validation (§11) does not wait for it:
  upstream's `gguf` Python package writes GGUF from PyTorch tensors,
  so the v0.1.0 fixtures are GGUF files that `zl_load()` already reads.
- **Types in memory.** With `zl_param(type = "f16")` declared, an F16
  tensor in the file is copied as is and an F32 one is rounded on the
  way in (§6.9); by default everything widens to f32.

Neither ggmlR (no match for "safetensors" in `R/` or `src/`; it reads
the tensor catalogue of `.pth` files in `R/pth.R`) nor the upstream ggml
repository (no match anywhere) reads safetensors.

### 7.5 FFT through zufft (0.2)

Uses: spectral losses (power spectra of returns and inter-event times,
which carry autocorrelation and volatility clustering), long
convolutions in the style of Hyena and S4, and Fourier features.

ggml's custom operations (`ggml_map_custom1` to `3`, `ggml_custom_4d`,
ops `GGML_OP_MAP_CUSTOM1`–`3` and `GGML_OP_CUSTOM`) take a function
called on every pool thread with `ith` and `nth`; none has a case in
`ggml_compute_backward`, so a custom operation on a gradient path
aborts with "unsupported ggml op for backward pass". They are
forward-only. The design follows:

1. **Forward-only FFT** in 0.2: a custom operation calling
   `zufft_api_v1`'s `r2c`, `c2r` and `c2c` (RFC 0026 §7), each task
   transforming a slice of independent columns with `nthreads = 1`;
   zufft's members are thread-safe. For metrics, preprocessing and
   sampling graphs.
2. **Differentiable spectra** in 0.2: the DFT as dense cosine and
   sine matrices applied with `zl_crossprod()`, whose backward is
   ggml's; exact up to rounding (the DFT is linear), O(n²), for lengths
   up to a few thousand. `zl_spectrum(x, method = "dft")`.
3. **Patch P6** (§9.3): an optional backward callback for
   `GGML_OP_CUSTOM`, given the incoming gradient and user data, that
   builds the gradient sub-graph. The FFT's backward is then its adjoint
   transform as another custom operation, and `method = "fft"` gains a
   gradient.

Without `zufft`, method `"fft"` is unavailable and `"dft"` remains.

### 7.6 Special functions through zuspecial (Later)

Likelihoods for event counts and heavy tails (Student-t and negative
binomial negative log-likelihoods) need `lgamma` and, for gradients
with respect to shape parameters, `digamma`. RFC 0027's table has
`gammaln`; whether `digamma` is a member (it is among zuspecial's R
functions) is unverified. In a graph these need custom operations with
backward, so they wait for P6 and a user who asks.

### 7.7 Faster CPU products: considered and rejected for v0.1.0

- **BLIS** (BSD-3-Clause; `flame/blis` at `a8037cf`, `LICENSE` read)
  through ggml's upstream BLAS backend (`src/ggml-blas`, which calls
  `cblas_sgemm`). Its `x86_64` family build compiles each
  sub-configuration's kernels with flags such as `-mavx2 -mfma
  -march=haswell` (`config/haswell/make_defs.mk`), which meets §13.1's
  NOTE; it brings its own `configure` and `make`; and `ggml-blas` is a
  separate backend onto which the scheduler splits operations, with
  copies. R's BLAS cannot stand in: R's reference BLAS
  (`src/extra/blas/blas.f`, `cmplxblas.f`) has double and double
  complex routines only, no `SGEMM`, and ggml-blas needs the CBLAS
  interface.
- **Highway** (Apache-2.0 or BSD-3-Clause; `google/highway` at
  `b93ad44`, 1.4 in development). Tested in the drafting session: an f32
  dot product written with `foreach_target.h` and
  `HWY_DYNAMIC_DISPATCH`, compiled with plain `-O2` and no `-m` flag,
  dispatched at run time to AVX-512 (`AVX3`) under both GCC 13.3 and
  clang. Highway therefore meets CRAN's flag rule, on clang too, unlike
  §13.1's GCC pragma. It is rejected for v0.1.0 only because using it
  means writing zulm's own kernels in place of ggml's, a fork of the
  compute; it is the tool for kernels zulm owns (Philox fills, Arrow
  conversion) if those ever matter, and evidence that §13.1's
  pragma-based route is sound. Its `abort.cc` imports `abort`.

### 7.8 What belongs to zuflow (RFC 0032)

The boundary rule: **zulm holds tensors, graphs, autograd,
optimisers, input and output (GGUF, safetensors, Arrow) and determinism
primitives (Philox, exact sums); zuflow holds generative-modelling
algorithms.** To zuflow, therefore:

- ODE and SDE samplers: Euler, Heun, adaptive Dormand–Prince (SciPy's
  implementation is a candidate; its licence and form are RFC 0032's to
  check), Euler–Maruyama.
- Minibatch optimal-transport coupling (OT-CFM) by linear assignment
  (LAPJV) or Sinkhorn. POT is MIT (`PythonOT/POT` at `604c47f`,
  `LICENSE` read), but its `network_simplex_simple.h` derives from
  LEMON and carries LEMON's own notice, which RFC 0032 must read.
- Atom smoothing, sequence windowing, and evaluation: one-step
  total-variation divergence, the discriminator test, stylised facts,
  Wasserstein distances.

What zuflow needs from zulm in C is in §8: run a graph, fill inputs
from Philox and from an Arrow batch, read outputs, and read and write
parameters (for weight averages). Everything else it does in R.

### 7.9 What belongs to zukzeit (RFC 0002)

The same rule, applied to inference: **zulm holds the tensor runtime
(graphs, operations, weight types, state tensors, file formats);
zukzeit holds the model layer.** To zukzeit, therefore: the checkpoint
catalogue and Hub manifests, download and cache, each architecture's
configuration and weight map, preprocessing and its inverse (reversible
normalisation, Welford scaling, `asinh`), patching and masking, the
decode loop and when it stops, quantile-channel selection and crossing
repair, the forecast object and the fabletools and parsnip adapters,
the execution contract and conformance harness, and the parity
fixtures, which stay in zukzeit's repository and are run there.

zulm's modules stay generic: `zl_attention()` does not learn TimesFM's
per-dimension scale, Toto's unit scaling or Chronos-2's group mask. An
architecture port composes §5.3 operations inside its own `build()` and
loads its weights by name, as zukzeit's `torch` modules do today with
`nnf_*` functions; §1.2 shows the block. What zukzeit needs from zulm
in C is nothing for its first port and, if it moves a decode loop out
of R, the state-tensor members of §8.

zukzeit uses zulm only (D33): `torch` leaves its `Imports`, and with it
the CUDA and MPS devices its contract exposes through `device`. Until
zulm has a GPU backend (§17, question 5) zukzeit's `device` resolves to
the CPU alone, which is the only device its parity fixtures certified
anyway. The order of the ports is zukzeit's roadmap decision; the
sizes suggest Toto 2.0 4M (16 MB, one pass), then Chronos-2 (one pass,
478 MB), then TimesFM 2.5 (925 MB, decode over state tensors).

## 8. The C table

**Decision.** zulm is a provider through a registered C table of
opaque handles, declared in `inst/include/zulm.h` for consumers with
`LinkingTo: zulm` and `Imports: zulm`. It does not export ggml's
headers.

**Why not ggml's headers through `LinkingTo`**, as ggmlR does:

1. *Churn.* There are 38 tags from `v0.9.5` (2025-12-31) to `v0.26.0`
   (2026-10-05); between
   `v0.25.0` and `v0.26.0` (twelve days) there were 152 commits, three
   of them in public headers. Every re-vendoring would be a breaking
   change for every consumer.
2. *ABI without symbols.* R3 hides zulm's ggml symbols, so a consumer
   compiling ggml headers has nothing to link but its own copy of ggml.
   Two copies at different versions then exchange `struct ggml_tensor`
   pointers whose layout may differ; ggmlR's five-dimension tensor shows
   the hazard is real.
3. *Exported symbols are worse.* Exporting ggml's symbols from zulm
   breaks R3 and exposes them to every other library in the process,
   including other copies of ggml in other packages.
4. *Upstream already went this way for its own reasons*: its dynamic
   backends talk to the core through function pointers, not headers.
5. *Cost.* The table's indirection is one call per graph run or batch
   fill, invisible against a graph computation.

The cost of the decision: a consumer cannot add ggml operations to
graphs from C in v0.1.0. Custom operations through the table are listed
under Later.

```c
/* inst/include/zulm.h; registered with
   R_RegisterCCallable("zulm", "zulm_api_v1") */
typedef struct zulm_graph zulm_graph;     /* opaque */
typedef struct zulm_store zulm_store;     /* opaque */

typedef struct {
  size_t struct_size;
  const char *(*ggml_version)(void);          /* "0.26.0+p2" */
  const char *(*message)(void);  /* this thread's last failure */
  /* handles from R objects; main thread only */
  zulm_graph *(*graph_from_sexp)(SEXP g);
  zulm_store *(*store_from_sexp)(SEXP p);
  /* inputs and outputs by name; any thread, no R calls */
  int (*input_shape)(const zulm_graph *g, const char *name,
                     int64_t ne[4], int *type);
  int (*input_write_f32)(zulm_graph *g, const char *name,
                         const float *src, int64_t n);
  int (*input_write_i32)(zulm_graph *g, const char *name,
                         const int32_t *src, int64_t n);
  int (*output_read_f32)(const zulm_graph *g, const char *name,
                         float *dst, int64_t n);
  int (*param_read_f32)(const zulm_store *p, const char *name,
                        float *dst, int64_t n);
  int (*param_write_f32)(zulm_store *p, const char *name,
                         const float *src, int64_t n);
  /* run the forward graph on its current inputs; main thread only,
     uses zulm's pool and its interrupt check */
  int (*compute)(zulm_graph *g);
  /* draws (§7.2) under the session key: a named stream, folded with
     the step; f32 by zulm's rounding rule; any thread, no R calls.
     The stream's key is folded in R when the name is first used
     (main thread), so a name is registered before a worker draws
     from it; an unregistered name is a nonzero return */
  int (*rng_stream)(const char *name);           /* main thread */
  int (*rng_uniform_f32)(const char *name, uint32_t step,
                         uint64_t offset, float *dst, int64_t n);
  int (*rng_normal_f32)(const char *name, uint32_t step,
                        uint64_t offset, float *dst, int64_t n);
  /* offset writes and state tensors (§6.9); any thread, no R calls */
  int (*input_write_f32_at)(zulm_graph *g, const char *name,
                            int64_t offset, const float *src, int64_t n);
  int (*state_shape)(const zulm_store *p, const char *name,
                     int64_t ne[4], int *type);
  int (*state_read_f32)(const zulm_store *p, const char *name,
                        float *dst, int64_t n);
  int (*state_write_f32_at)(zulm_store *p, const char *name,
                            int64_t offset, const float *src, int64_t n);
  /* next Arrow batch into a graph's inputs (§7.3); main thread only;
     returns 1 when the stream has ended */
  int (*batch_fill)(zulm_graph *g, SEXP iterator);
  /* … appended over time … */
} zulm_api;

static inline const zulm_api *zulm_api_get(void);
```

- Return codes: 0 success; nonzero names an unknown input, a shape or
  type mismatch, an offset out of range, or a stale handle, with text
  from `message()`.
- Members never call R except `*_from_sexp`, `rng_stream`, `compute`
  and `batch_fill`, which must run on the main thread; the others may
  run on a consumer's own threads, one thread per graph or store at a
  time. A consumer that wants raw doubles or its own keys uses
  `zurand`'s table directly; zulm's members exist so that a consumer's
  draws and the graph's draws share one key and one rounding rule.
- These members are what zuflow (RFC 0032) needs for sampler loops in C
  (`compute`, `input_write_f32`, `output_read_f32`, `rng_normal_f32`),
  for sliding windows and a future key-value cache
  (`input_write_f32_at`, `state_write_f32_at`), for training
  from Arrow (`batch_fill`) and for weight averages
  (`param_read_f32`, `param_write_f32`); and what zukzeit (§7.9) would
  need for a decode loop in C (`compute`, the state members). zulm
  needs no hooks into consumers: companions are reached through their
  tables, not registered into zulm.
- The table follows `zuxml`'s versioning rules: members are appended,
  `struct_size` tells a consumer whether a member exists, and a layout
  change bumps the registered name to `zulm_api_v2`.

## 9. The vendored library, as zulm uses it

### 9.1 What and from where

- **Files.** From `src/`: `ggml.c`, `ggml-alloc.c`, `ggml-quants.c`,
  `ggml-quants.h`, `ggml-backend.cpp`, `ggml-backend-meta.cpp`,
  `ggml-threading.cpp`, `ggml-threading.h`, `ggml-opt.cpp`, `gguf.cpp`,
  `ggml-impl.h`, `ggml-common.h`, `ggml-backend-impl.h`,
  `ggml-feats.h`; `ggml-cpu/` without `kleidiai`, `spacemit` and the
  `arch/` directories of POWER, s390, RISC-V, LoongArch and WebAssembly;
  the public headers `ggml.h`, `ggml-alloc.h`, `ggml-backend.h`,
  `ggml-cpu.h`, `ggml-opt.h`, `gguf.h`, `ggml-cpp.h`; `LICENSE`. About
  4.2 MB of source, 578 KB compressed. A generated `ggml-version.h`
  replaces upstream's CMake template.
- **Left out on purpose.** `ggml.cpp`, whose only content is a static
  initialiser that replaces the process-wide `std::terminate` handler
  when the library loads (unless `GGML_NO_BACKTRACE` is set); a package
  must not do that to R. `ggml-backend-reg.cpp` and `ggml-backend-dl.cpp`
  (the backend registry and `dlopen` loader): zulm creates the CPU
  backend directly, and the probe program linked without them. All GPU
  backends.
- **Script.** `tools/vendor-ggml` clones the pinned tag, copies the file
  list, applies `tools/patches/*.patch` in order, writes
  `ggml-version.h`, and records tag, commit, patch list and each file's
  SHA-256 in `src/ggml/VERSION`; `zulm_info()` reports them.
- **Cadence.** Re-vendor on upstream releases that fix the CPU backend,
  autograd or `ggml-opt`, at most monthly, after the full test suite and
  the §12 benchmarks; never in the four weeks before a CRAN submission.

### 9.2 Build-level adaptation without patches

- `zlm_compat.h`, included by the wrapper translation unit of every
  ggml source (§13.4), redirects `printf`, `vprintf`, `puts`,
  `putchar`, `fprintf`, `vfprintf`, `sprintf`, `fputs`, `fputc`,
  `fflush`, `fwrite`, `stdout` and `stderr` to zulm's log buffer when
  they write to the standard streams (gguf's file writes pass through),
  as ggmlR's header does for the same CRAN rule. The macros are
  function-like, so `__attribute__((format(printf, …)))` is untouched.
  **Revised in Stage 1:** it also turns `abort()` and the backtrace's
  `_Exit()` into `zlm_abort()`, a trap (§13.2), so no symbol on R's
  compiled-code list is linked. ggml never qualifies these names with
  `std::`, which a test checks.
- **Diagnostic pragmas are stripped by the vendor script.** `R CMD
  check --as-cran` sets `_R_CHECK_PRAGMAS_`, and `.check_pragmas()`
  (`QC.R`) scans every C and C++ file under `src/` and `inst/include`
  for `#pragma GCC diagnostic ignored` and `#pragma clang diagnostic
  ignored`: a NOTE for any hit, a WARNING when the suppressed warning
  is on R's non-portable list or among `-Wuninitialized`,
  `-Wfloat-equal`, `-Warray-bounds` and `-Wformat`. It is a source
  scan, so it does not matter whether the file is compiled. At
  `v0.26.0` the CPU subset carries such pragmas in
  `ggml-cpu/repack.cpp`, `ggml-cpu/arch/x86/repack.cpp` and
  `ggml-cpu/arch/arm/repack.cpp` (`-Woverlength-strings`),
  `ggml-cpu/llamafile/sgemm.cpp` (`-Wpedantic`,
  `-Wignored-attributes`) and `ggml-cpu/amx/mmq.cpp` (`-Wpedantic`,
  `-Wunused-local-typedefs`). Upstream will not remove them, so this
  is not a §9.3 patch: `tools/vendor-ggml` deletes those lines (and
  the matching `push` and `pop`) after copying, records the deletions
  in `src/ggml/VERSION`, and a test greps the vendored tree for the
  pattern so that a re-vendoring cannot reintroduce one. The warnings
  the pragmas silenced then appear in the install log, which is
  acceptable (§13.3).
- `GGML_SCHED_MAX_SPLIT_INPUTS=4` (Stage 4): the scheduler reserves
  graph size × this × 2 tensors for split inputs, 41 MB a trainer at
  upstream's 30, and the one CPU backend copies none; it grows its lists
  on demand if ever needed.
- `GGML_USE_CPU_REPACK` and `GGML_SCHED_MAX_COPIES=4` as upstream's
  CMake defines them, and the feature-test macros it sets per platform
  (`_XOPEN_SOURCE=600`, `_GNU_SOURCE` on Linux, `_DARWIN_C_SOURCE` on
  macOS), from the force-included header; no OpenMP.
  `GGML_USE_LLAMAFILE` is decided by Stage 1's measurements (§13.1,
  question 7).

### 9.3 Patches

Each patch is a file in `tools/patches/` with a header naming its
upstream pull request; none is vendored without one. Seven are known
now.

| # | Patch | Why | Workaround meanwhile |
| --- | --- | --- | --- |
| P1 | `MEAN` backward accumulates | asserts (below) | `sum_rows` + scale |
| P2 | `SUB`, `DIV` broadcast backward | asserts (below) | `a + neg(b)` |
| P3 | `NORM` backward (LayerNorm) | missing | §6.3 composition |
| P4 | `GELU` backward | missing | SwiGLU, SiLU |
| P5 | backtrace without `fork` | §13.3 | abort callback set |
| P6 | backward hook for `CUSTOM` | missing | DFT by product |
| P7 | global-norm clipping in `ggml-opt` | missing (§6.4) | `clip_norm` refused |
| P8 | no `static_assert` macro under C23 | warns (below) | none: vendored |
| P9 | GGUF tensor type checked before it is a `ggml_type` | UB (below) | none: vendored |
| P10 | type-table functions called through their own types | UB (below) | none: vendored |

**Status, Stage 4 (2026-10-14).** P1, P2 and P7 are vendored
(`tools/patches/`) beside P8, P9 and P10, each pinned by a regression
test. P7 also makes
every gradient the optimiser step reads contiguous: the step kernels read
the gradient with the weight's strides, and a parameter used through a
permutation has a strided gradient, which upstream's `ggml-opt` would
apply to the wrong elements. Still none has an upstream pull request.

**Status, Stage 1 (2026-10-14).** P1, P2 and P7 are drafted against
`v0.26.0` in `tools/patches/drafts/`, which `tools/vendor-ggml` does not
apply; Stage 4 moves each up a directory once tests pin its behaviour.
P8 is vendored now, because the check fails without it. None of the
four has an upstream pull request yet: opening them is the
maintainer's call, and each patch header and this table get the link
when it exists. Until then the rule above, that no patch is vendored
without one, is suspended for P8 alone.

- **P8.** ggml's C headers (`ggml-common.h`, `ggml-impl.h`) define
  `static_assert` as a macro when it is not one. Under C23 it is a
  keyword, so the macro hides it, and clang 22 and 23 compiling C as
  `-std=gnu23` (CRAN's `r-devel-linux-x86_64-debian-clang`) warn
  "keyword is hidden by macro definition [-Wkeyword-macro]", which R's
  check counts as a significant warning. P8 skips the macro under
  C23. Found by the r-actions `clang23` and `ubuntu-clang` legs.
- **P9, P10 (found by the native checks, 2026-10-14).** UBSan, as
  CRAN's sanitizer flavours run it, reported two upstream bugs. P9:
  `gguf_init_from_reader()` loads a malformed file's tensor type into
  the `ggml_type` enum before checking its range, an invalid enum load
  in C++. P10: the CPU type table stores `ggml_vec_dot_f32` and its f16
  and bf16 siblings, and the f32-to-f16, -bf16 and -i32 converters, cast
  to function types whose parameters differ, and ggml's own type table
  does the same for the f16 and bf16 row converters; clang's
  `-fsanitize=function` reports the call. P10 stores wrappers of the
  exact types for those entries; the quantised types' entries, which
  zulm never calls, are left as upstream has them. Both are vendored,
  like P8, ahead of their upstream pull requests.
- **P1.** `ggml_mean` over rows returns one value per row, but its
  backward adds the incoming gradient with `ggml_add1_or_set`, which
  asserts `ggml_is_scalar(b)` once a second consumer has already
  contributed a gradient. Reproduced: a LayerNorm written with
  `ggml_mean` aborted inside `ggml_opt_init` at `ggml.c:2158`.
- **P2.** `SUB` and `DIV` backward pass the full-shape gradient to a
  broadcast second operand without `ggml_repeat_back` (which `ADD` and
  `MUL` use), and `ggml_compute_backward` then asserts
  `ggml_are_same_shape(src1, grads[isrc1])`. Reproduced for both at
  `ggml.c:7241` with a `4 × 3` minus `1 × 3`.
- **P3, P4.** ggmlR has CPU kernels for both (`GGML_OP_NORM_BACK`,
  `GGML_OP_GELU_BACK`) and for flash attention's backward; §17 asks
  whether to propose upstreaming them jointly.
- **P5.** `ggml_print_backtrace()` forks and runs `gdb` or `lldb` on the
  process, then `_Exit`s the child; the probe program's abort did so in
  the drafting session. With zulm's abort callback installed it is
  never called, but its symbols remain in the shared object. P5 adds a
  compile-time switch upstream (`GGML_NO_BACKTRACE` as a macro).
- **P6.** `GGML_OP_CUSTOM` and the `MAP_CUSTOM` operations have no case
  in `ggml_compute_backward`. P6 adds an optional callback, stored with
  the operation's user data, that builds the gradient sub-graph; §7.5
  uses it for the FFT's adjoint and §7.6 for special functions.
- **P7.** `ggml-opt` has no gradient clipping and no hook where a
  consumer could add it (§6.4). P7 adds a `clip` member to
  `ggml_opt_optimizer_params` (0 for none), carried in the per-step
  parameter tensor that `get_opt_pars` already fills, and makes
  `ggml_opt_build` emit, before the step nodes, the sum of squared
  gradient accumulators over all parameters, its square root, and a
  scale of `min(1, clip / norm)` applied to each gradient the step
  reads. Those nodes are `SQR`, `SUM` and `SCALE`, whose kernels are
  thread-invariant (§6.5), so clipping keeps the determinism contract.
  Until P7 is vendored (Stage 4), `zl_trainer(clip_norm =)` with a
  non-`NULL` value is `zulm_invalid_argument` naming the patch, not
  a silent no-op. Clipping is a feature upstream's own training
  examples lack and a plausible contribution; if upstream declines, the
  patch stays, as §18 says.

The backward-supported set after P1–P2, read from
`ggml_compute_backward` at `v0.26.0` and re-read at upstream's default
branch on 2026-10-13, unchanged: `DUP`, `ADD`, `ADD1`, `ACC`,
`SUB`, `MUL`, `DIV`, `SQR`, `SQRT`, `LOG`, `SIN`, `COS`, `SUM`,
`SUM_ROWS`, `MEAN`, `REPEAT`, `REPEAT_BACK`, `RMS_NORM`, `MUL_MAT`,
`SCALE`, `SET`, `CPY`, `CONT`, `RESHAPE`, `VIEW`, `PERMUTE`,
`TRANSPOSE`, `GET_ROWS`, `DIAG_MASK_INF`, `DIAG_MASK_ZERO`, `SOFT_MAX`
(mask without gradient), `ROPE`, `IM2COL`, `POOL_2D`,
`CROSS_ENTROPY_LOSS`, unary `ABS`, `SGN`, `NEG`, `STEP`, `RELU`, `SILU`,
`EXP`, `EXPM1`, `SOFTPLUS`, and split `SWIGLU`. Not supported: `NORM`,
`CONCAT`, `GELU`, `TANH`, `SIGMOID`, `CLAMP`, `CUMSUM`, `PAD`, `TRI`,
`ARGSORT`, `TOP_K`, `SET_ROWS`, `FLASH_ATTN_EXT`, the custom
operations and every other operation; §5.3 marks each of these
forward-only and the pre-check of §6.2 refuses them on a gradient path.

## 10. Errors

Every condition inherits `zulm_error`: `zulm_invalid_argument`,
`zulm_shape_error` (with `op`, the shapes, and the R call),
`zulm_type_error`, `zulm_unsupported_backward` (with `op`),
`zulm_memory_error` (with requested and allowed bytes),
`zulm_engine_error` (a ggml assertion caught during graph
construction, with ggml's message; the graph involved is discarded),
`zulm_interrupt` (also inherits `interrupt`), `zulm_file_error`
(GGUF and safetensors), `zulm_arrow_error` (type, null, dictionary or
stream failures, with column and row), `zulm_companion_error` (a
companion's table missing a member zulm needs), `zulm_stale_error`
(a dead pointer after reload or fork),
`zulm_internal_error`. Warnings inherit `zulm_warning`:
`zulm_range_warning`, `zulm_nonfinite_warning` (a non-finite loss
from `zl_step()`).

## 11. Testing

- **Gradient checks.** For every operation and module with a backward
  pass, central finite differences in R (double precision, on f32
  results) against ggml's gradients, at random shapes including
  broadcasts, with tolerances scaled by f32 epsilon and the shape.
- **R oracles.** Each forward operation against a base-R implementation
  (`crossprod()`, `apply()`-based softmax and norms) within f32
  tolerance, including every forward-only operation of §5.3; modules
  against hand-written R versions; `zl_rope()` against the split-half
  and interleaved formulas written in R, at full and partial `n_dims`,
  and `zl_softmax()` with masks of every broadcast shape the kernel
  accepts.
- **PyTorch fixtures.** `tools/make-torch-reference.py`, run offline,
  stores weights, inputs, outputs and gradients for the transformer
  block, AdaLN and the losses as GGUF files written with upstream's
  `gguf` Python package, under `tests/testthat/fixtures/` (budget
  1 MB); zulm loads the weights with `zl_load()` and compares within
  tolerance. PyTorch's row-major `[out, in]` weight is zulm's
  `c(in, out)` with the same bytes, so the script writes tensors
  unchanged. For §1.2 the script also writes, at small width, one
  TimesFM-style attention block (split-half rotary positions, query
  and key normalisation, the per-dimension scale, `scale = 1`, a
  prefix mask) and one Toto-style block (interleaved partial rotation,
  xPos factors, `scale = 1 / d`, unit-scaled linears), so that both
  conventions are pinned to PyTorch, not only to R. The same script
  also writes a safetensors copy of each fixture (§7.4).
- **Determinism.** Same seed, same results twice; parameters and
  reported losses bit-identical at one, two and four threads after 200
  steps of the §1.1 model, with and without `clip_norm`; forward
  outputs of the §1.2 blocks bit-identical at one, two and four
  threads.
- **Inference.** A decode loop over state tensors gives the same
  outputs as one full-context forward pass within f32 tolerance, for
  several `n_new` and fills; `zl_flash_attention()` against the plain
  form; f16 and bf16 weights against f32 within the tolerance §6.9
  documents, and a test that fails if the difference shrinks below it,
  so the documentation stays true; `zl_graph_cache()` builds once per
  shape and evicts under the memory cap; zukzeit's Toto 2.0 4M parity
  fixtures pass in a zukzeit branch whose runtime is zulm, run in
  zukzeit's CI against zulm's `main` (the fixtures are zukzeit's and
  stay there), with the zulm side recorded in `cran-comments.md` as a
  reverse-dependency check.
- **Companions, both paths.** The suite runs with and without
  `nanoarrow` installed and, from 0.2, `zuexact` and `zufft` (CI legs
  with each removed); fallbacks give the documented results, and a
  companion whose table is too short gives `zulm_companion_error`. With
  `zuexact`, reported losses are identical with `accumulate = 1` and
  `4`. `zujson` and `zurand` are `Imports` and have no absent path.
- **Random numbers.** In-graph draws equal `zurand::rng_uniform()` and
  `rng_normal()` on the folded keys, rounded by §7.2's rule, on every
  CI platform; draws identical across thread counts and when a batch is
  split in two with the matching offsets; uniforms never 0 or 1 in f32;
  a zurand whose table version is below 3 is refused at load with
  `zulm_companion_error`. Distributional checks are zurand's, not
  repeated here.
- **Arrow.** Streams from nanoarrow, DuckDB and Parquet fixtures,
  with record batches smaller than, equal to and larger than `rows`;
  every supported type; nulls under both policies; dictionaries that
  change between chunks; the release callbacks called exactly once.
- **safetensors.** Round trips at f32, f16 and bf16; files written by
  the Python `safetensors` package; a real Hub header with no data
  (Toto 2.0 4M's 48 tensors, a few kilobytes) parsed and validated; a
  `zufuzz` target for headers.
- **Spectra (0.2).** `zl_spectrum(method = "dft")` against `zufft` and
  against `stats::fft()`; its gradient by finite differences.
- **Regression tests for P1, P2 and P7.** With the patches vendored,
  the assertion cases of §9.3 train and a step with `clip_norm` scales
  every gradient by the same factor, checked against the norm computed
  in R; a test fails if a re-vendoring drops a patch upstream has not
  merged.
- **Vendored tree.** No file under `src/ggml/` contains a diagnostic
  pragma (§9.2), and `tools/vendor-ggml` reproduces the tree byte for
  byte.
- **Hostile input.** Every R function with wrong shapes, types, `NA`,
  zero-length dimensions, more than four dimensions, freed objects and
  objects after `saveRDS()`; none may reach a ggml assertion. A
  `zufuzz` target drives graph construction with random operation
  sequences and shapes.
- **GGUF.** Round trips at f32, f16 and bf16; files written by zulm
  read by upstream's `gguf` Python package in CI.
- **Threads and fork.** Interrupt during a long graph; a forked child
  that trains with two threads, on Unix.
- **ISA variants.** CI runs the suite with each variant forced
  (`options(zulm.variant =)` before loading), on hardware that has it.
- **Build flags (§13.4).** The `build-flags` workflow fails on the
  non-portable-flag NOTE and on any compiled-code NOTE; the
  `portability` workflow builds with `ZULM_CFLAGS` and `ZULM_CXXFLAGS`
  set on Linux and macOS, checks the flags reached every compile and
  that `zulm_info()` reports the `native` variant, and runs the suite.
- CRAN suite under 15 s on one thread; the acceptance training runs on
  CI only.

## 12. Performance targets

Measured by `tools/run-benchmarks`, recorded in the design. Measured in
the drafting session (one thread unless stated; upstream ggml in C, not
through R):

| Build of upstream ggml | 512³ | 128·512·4096 | §1.1 step |
| --- | --- | --- | --- |
| x86-64 baseline, `-O2` | 1.6 | 1.8 | 118 ms |
| SSE4.2, `-O2` | 8.2 | 5.6 | 68 ms |
| AVX2 FMA F16C, `-O2` | 14.0 | 10.8 | 40 ms |
| same, `-O2 -funroll-loops` | 31.0 | 16.1 | 34 ms |
| same by pragma, no `-m` flag | 14.8 | 10.6 | 39 ms |
| AVX2, `-O2`, llamafile kernels | 27.6 | 21.0 | 30 ms |
| AVX2, `-O3` | 32.0 | 19.4 | — |
| `-march=native` (AVX-512), `-O3` | 33.1 | 20.3 | 27 ms |

Matrix columns are f32 GFLOP/s for `K·M·N`. The baseline build has no
SIMD path because ggml's x86 mappings start at SSE3. At two threads the
§1.1 step took 61.7 ms against 109.5 ms at one (an `-O3` baseline
build), 1.77×.

**Measured in Stage 1 (2026-10-14)**, through R (`tools/bench-variants.R`
in the `measure` workflow), best of three runs of twenty 512³ f32
products, GFLOP/s at one and two threads; R's default flags (`-O2`):

| Runner, compiler | `baseline` | `sse42` | `avx2` | `-mavx2` build | `avx2` + llamafile |
| --- | --- | --- | --- | --- | --- |
| Linux, Xeon (Intel), GCC | 4.3 / 8.6 | 27.3 / 52.7 | 46.6 / 89.4 | 43.7 / 82.9 | 78.7 / 150.2 |
| Linux, EPYC 7763, clang | 2.2 / 4.4 | 20.5 / 40.1 | 31.1 / 60.8 | 31.2 / 61.0 | 87.1 / 162.9 |
| macOS x86-64 (Intel), Apple clang | 18.6 / 30.9 | 17.2 / 32.1 | 31.7 / 61.3 | 31.4 / 60.3 | 102.1 / 162.5 |
| Windows, EPYC 7763, Rtools GCC | 2.2 / 4.3 | 17.0 / 31.3 | 31.3 / 58.6 | — | see §13.1 |
| macOS arm64 (M1, virtual), Apple clang | `neon`: 18.4 / 35.8 | | | | `neon` + llamafile: 76.7 / 133.6 |

GCC with `-mavx2 -funroll-loops` gave 47.1 / 88.4, the `avx2`
variant's speed: the unrolling pragma buys what the flag buys. Each
variant's result was bit-identical at one, two and four threads, and
the `avx2` variant's bits were the same under GCC and clang on Linux,
on Intel macOS and on Windows (hash `0532cf3b` in every row). The x86-64
macOS `baseline` matches `sse42` because Apple's default target already
has SSSE3 and SSE4.1. Install with `MAKEFLAGS=-j4` took 43–96 s; the
three x86-64 variants take 5.8 MB of `libs/` without debug information
(the `clang23` leg) and 37 MB with GCC's `-g`; R 4.6 reports installed
size as INFO, not a NOTE.

Targets: on an AVX2 machine, from a build made with CRAN's default
flags, `zl_crossprod()` of 512³ at least 25 GFLOP/s and the §1.1 step at
most 35 ms at one thread; two threads at least 1.6× one; R overhead in
`zl_step()` under 5 % of the step for that model; on arm64 (NEON is
always present, unmeasured here) within 2× of the AVX2 figures per core.
A missed target is reported, not hidden by relaxing it.

Inference targets, measured through R with `zl_forward()`, f32
weights, on the same AVX2 machine, one thread unless stated:

- One TimesFM 2.5 transformer block over 512 patches plus the
  continuous head on the last patch (37.7 M parameters; the shape
  zukzeit timed on LibTorch 2.8 at 0.133 s on an x86-64 Mac): within
  2× of R `torch` on the same host at the same thread count, and
  within 1.3× at four threads. The figure is a floor to beat, not a
  claim: LibTorch's products run on MKL or OpenBLAS, and the drafting
  session's best f32 product was 33 GFLOP/s.
- Toto 2.0 4M on a 4 096-value context, one forecast: under 50 ms.
- One decode step through state tensors (§6.9): within 1.2× the cost of
  the same step's projections and MLP alone, so the cache adds nothing
  measurable.
- R overhead in a decode loop under 10 % of the graph time; graph
  construction from `zl_graph_cache()` under 5 ms per new shape for a
  twenty-block model.

## 13. Build, portability and CRAN

### 13.1 CPU dispatch on CRAN binaries: the go/no-go

ggml picks its SIMD code at compile time from the compiler's ISA macros.
Its only run-time choice is `GGML_CPU_ALL_VARIANTS`, which builds one
CPU backend library per ISA level (x86: `x64`, `sse42`, `sandybridge`,
`haswell`, `skylakex`, `icelake`, `zen4`, `alderlake` and others) and
loads the best-scoring one with `dlopen`; CMake refuses it without
`GGML_BACKEND_DL`, which requires shared libraries and a shared
`ggml-base`. CRAN builds Windows binaries on x86-64 and installs Linux
packages from source with `-O2` and no ISA flags, so an untreated build
is the 1.6 GFLOP/s row of §12.

Options:

1. **Baseline only.** Rejected: a 9–20× loss on the product that
   dominates training.
2. **Opt-in flags at install** (`--with-simd` in ggmlR, an environment
   variable in edgemodelr). Kept as an extra for source installs, in
   zufft's form: `ZULM_CFLAGS` and `ZULM_CXXFLAGS` environment hooks
   appended to the `PKG_*FLAGS` in `src/Makevars` (§13.4), but it does
   nothing for binary users.
3. **ggml's own variants.** Rejected: the shared `ggml-base` would
   export ggml's symbols (against R3), the variant libraries would live
   outside `libs/` and need `dlopen` paths on three platforms, and the
   variants would resolve ggml symbols against whatever copy of ggml is
   loaded first.
4. **Whole-package variants (recommended).** zulm's compiled code
   (glue and ggml together) is built K times from the same sources with
   different ISA settings, each a self-contained shared object with
   hidden symbols; `src/install.libs.R` installs them into `libs/`. A
   small probe library, the package's registered DLL, tests the CPU with
   `__builtin_cpu_supports()` at `.onLoad`, and the R layer loads the
   best variant with `library.dynam()` and calls its routines through
   native-symbol objects it looks up once. Within one variant every
   function may use its ISA, so the hazard of mixing ISA-specific
   objects in one library (a shared inline function compiled for AVX2
   chosen by the linker for baseline code) cannot arise. Proposed
   variants: on x86-64, `baseline`, `sse42` and `avx2` (with FMA, F16C,
   BMI2), and `avx512` if Stage 1 shows a gain over `avx2` for f32
   training; on arm64, one (NEON is in the base ABI).

How a variant gets its ISA. Per-file `-mavx2` style flags work, but R's
check (`check.R`, under `_R_CHECK_COMPILATION_FLAGS_`, set by
`--as-cran`) reports every `-m` flag in the install log except a short
list as "Compilation used the following non-portable flag(s)", a NOTE.
The drafting session tested the alternative: a force-included header
with `#pragma GCC target("sse4.2,avx,avx2,fma,f16c,bmi2")`. Findings:

- In C, GCC defines `__AVX2__` and the other ISA macros under the
  pragma; in C++ (g++ 13.3) it does not, so the header defines them
  itself when `__cplusplus` is set and they are absent.
- Five C++ files (`ggml-backend.cpp`, `ggml-backend-meta.cpp`,
  `ggml-opt.cpp`, `gguf.cpp`, `ggml-cpu/ggml-cpu.cpp`) fail under the
  pragma: libstdc++'s `always_inline` allocator functions cannot be
  inlined across the target mismatch. None holds a kernel; they are
  built without it.
- The result matched a `-mavx2` build bit for bit (identical parameter
  hash after 100 training steps) and in speed (14.8 against
  14.0 GFLOP/s).
- Clang ignores `#pragma GCC target`. **CRAN builds x86-64 macOS
  binaries**: its check flavours include `r-release-macos-x86_64` and
  `r-oldrel-macos-x86_64` beside the arm64 ones (an earlier draft of
  this section wrongly said macOS was arm64 only). Those binaries are
  built with Apple clang, so without a clang route the x86-64 macOS
  binary is the 1.6 GFLOP/s row of §12 for every Mac user on Intel.
  Clang is therefore part of the go/no-go, not a footnote. The
  candidate is `#pragma clang attribute push
  (__attribute__((target("sse4.2,avx,avx2,fma,f16c,bmi2"))),
  apply_to = function)` in the same force-included header. Two things
  are known not to come for free and must be tested: the attribute
  does not define `__AVX2__` and the other ISA macros, and ggml picks
  its kernels by those macros, so the header must define them itself
  on clang as it already does for g++; and the `always_inline`
  intrinsic wrappers that broke five C++ files under GCC may behave
  differently under clang, in either direction. Linux users on clang
  get the same route.

Two cheap gains that need no ISA flag: loop unrolling lifted the AVX2
product from 14.0 to 31.0 GFLOP/s at `-O2` without changing the trained
parameters' bits, and ggml's llamafile kernels (`GGML_LLAMAFILE`, off by
default in standalone ggml) gave the fastest training step at `-O2`.
The unrolling was measured with `-funroll-loops`, which cannot be
written into `Makevars`: R's "compilation flags in Makevars" check
(`tools:::.check_make_vars`, a WARNING) rejects every `-f*`, `-m*`,
`-O*` and `-W*` token in a `PKG_*FLAGS` variable, as zufft found in its
Stage 1 (RFC 0026 §12.3), even though the install-log scan would let
`-funroll-loops` through. It is requested in the source instead:
`#pragma GCC optimize("unroll-loops")` in the force-included header,
beside the target pragma, for GCC; clang has no file-level unrolling
pragma but unrolls at `-O2` by default, so Stage 1 measures whether a
clang build already has the gain. Whether llamafile's kernels keep
results independent of the thread count is unverified; Stage 1
measures it before either is adopted.

Stage 1 must show, before anything else is written: the variant build
on the Linux, Windows, arm64 macOS and **x86-64 macOS** CI legs with
`R CMD check --as-cran` free of the non-portable-flag NOTE and of the
pragma NOTE (§9.2); `.onLoad` choosing the right variant on each
runner; the `avx2` variant within 15 % of the `-mavx2` build under
GCC and under clang; on Windows, that the `avx2` variant built by
Rtools' MinGW-w64 GCC loads and runs (zufft's AVX code from that
compiler crashed on load, exit 139, also with
`-Wa,-muse-unaligned-vector-move`, GCC bug 54412 on unaligned AVX
stack spills; llama.cpp ships MinGW AVX2 builds, so this may be
specific to ducc0's aligned vector types, but it is checked, and the
Windows fallback is `sse42` alone); and total install time and size.
Measured for one
variant in the drafting session: 23–34 s of wall time on two cores and 1.35 MB
unstripped (1.21 MB stripped), so three x86-64 variants cost roughly a
minute and 4 MB. If the variant scheme cannot pass CRAN's checks,
zulm ships the `sse42` level as its baseline on x86-64 (5× the scalar
build; every x86-64 CPU since 2008 has SSE4.2, but it is still a flagged
`-m` flag unless the pragma route works) and returns to the maintainer
with that trade-off.

**Stage 1 result (2026-10-14): go.** The variant scheme passes
`R CMD check --as-cran` with no non-portable-flag, pragma or
compiled-code NOTE on every r-actions leg, x86-64 macOS included, and
`build-flags` checks it on Linux, both macOS architectures and Windows.
What it took, and what was measured:

- **No `-include`, no `-m` flag.** Each ggml source is compiled through
  a generated wrapper translation unit under `src/zlm_v/<variant>/`
  that defines the variant's macro, includes `zlm_variant_begin.h` (the
  force-included header), the ggml source, and `zlm_variant_end.h`;
  `tools/gen-build` writes them. So the variant is chosen without any
  flag at all, and R's own suffix rules compile it.
- **What is built per variant.** The kernels (`ggml-cpu/` except
  `ggml-cpu.cpp`, plus the architecture's `arch/` files) are built once
  per variant; ggml's base (`ggml.c`, `ggml-alloc.c`, `ggml-quants.c`,
  `ggml-backend*.cpp`, `ggml-opt.cpp`, `gguf.cpp`,
  `ggml-threading.cpp`), `ggml-cpu/ggml-cpu.cpp` and zulm's glue are
  built once and linked into every variant. The five C++ files that
  failed under GCC's target pragma are among the shared ones, so no file
  is built under a pragma it cannot take. Mixing is safe: a variant is
  loaded only on a CPU that runs all of it.
- **clang works.** `#pragma clang attribute push` with the target
  attribute, and the ISA macros defined by the header, compile every
  kernel on Apple clang and on clang 22 and 23; the x86-64 macOS
  `avx2` variant runs at the `-mavx2` build's speed with the same bits.
- **Speed.** The `avx2` variant is within 15 % of the `-mavx2` build
  under GCC (6 % faster, from the unrolling pragma) and under clang
  (equal); see §12.
- **Windows.** The `avx2` variant built by Rtools' MinGW-w64 GCC loads
  and runs (31.3 GFLOP/s, the same bits as Linux); zufft's crash is
  specific to ducc0. ggml's Win32 threads ran the suite and the
  benchmarks at one to four threads without a hang. The native hook
  stays off on Windows (§13.4) until someone needs it.
- **Probe.** `zulm.so` (the registered DLL) holds only
  `zlm_cpu_variants()`, which asks `__builtin_cpu_supports()` for AVX2,
  FMA, BMI2 and F16C (then SSE4.2 and POPCNT); `.onLoad` loads the best
  installed variant with `dyn.load()`, or the one
  `options(zulm.variant =)` names, and calls its routines through the
  native-symbol objects it registered. Every runner loaded the variant
  its CPU supports best.
- **Question 7, decided from these measurements.** llamafile's kernels
  go in (`GGML_USE_LLAMAFILE`): 2.5–4× the plain kernels on every
  platform and bit-identical across thread counts. The unrolling pragma
  stays, for GCC. No `avx512` variant: no CI runner has AVX-512, and the
  drafting session measured 3 % over `-O3` AVX2. Stage 4 re-checks
  thread invariance on training steps, the measure question 7 named.

### 13.2 Assertions, aborts and the main thread

`GGML_ASSERT` and `GGML_ABORT` call `ggml_abort()`, which formats the
message, calls the callback set by `ggml_set_abort_callback()` if there
is one (otherwise prints and runs the backtrace of §9.3 P5), and then
always calls `abort()`. A callback that returns therefore cannot save R.

zulm's design, in order of reliance:

1. **Prevention.** The C glue checks every argument a ggml builder
   asserts on (shapes, broadcast compatibility, types, contiguity,
   dimension counts, sizes) and refuses operations whose backward is
   missing (§6.2) before ggml sees them. The hostile-input suite (§11)
   is the measure of this layer.
2. **Recovery during graph construction.** While the shim builds
   tensors, graphs, backward passes or `ggml-opt` contexts (always on
   the R main thread, holding no ggml lock), it sets a thread-local
   flag; the abort callback, seeing the flag, throws a C++ exception
   that the shim catches and turns into `zulm_engine_error`. It never
   returns, so `abort()` is not reached, and the half-built context is
   discarded. Verified in the drafting session on Linux with GCC 13.3:
   the P2 assertion was raised and caught three times in a row, and a
   computation afterwards was correct. **Stage 1 (2026-10-14):** no
   `-fexceptions` is needed, and none is set. The throw crosses
   `ggml_abort`'s C frames on the unwind tables every supported
   compiler emits by default (DWARF CFI on x86-64 and arm64 Linux,
   compact unwind on macOS, SEH tables on Windows x64), and ggml's C
   has no cleanup that would need `-fexceptions` to run. The test that
   raises a `ggml_add` shape assertion as `zulm_engine_error` three
   times and then computes correctly passes on every r-actions leg,
   Windows included, so prevention is not the only layer there.
3. **The fatal path.** An assertion during a computation cannot be
   recovered on any thread: on a pool thread an exception cannot reach
   the main thread and `longjmp` to another thread is undefined; on the
   main thread (worker 0) the other workers would be left waiting at a
   barrier. Without the flag the callback writes the message to file
   descriptor 2 for the crash log and returns, and ggml's `abort()`,
   which `zlm_compat.h` turns into a trap, stops the process.
   Prevention is what keeps this path unreachable, and §15 measures it.

ggmlR's approach, a macro turning `abort()` into `Rf_error()`, was
considered and rejected: it `longjmp`s across C++ frames and, from a
pool thread, out of a thread that R does not own.

Because `abort` stays in the shared object, R's compiled-code check
(`sotools.R` lists `abort`, `_Exit`, `printf`, `puts`, `putchar`,
`stderr` and `stdout` among others) would report it. The CPU-only
object built in the drafting session imported `abort`, `_Exit`,
`__printf_chk`, `puts`, `putchar`, `stderr`, `stdout`, `fork`,
`execlp`, `waitpid`, `prctl`, `fflush`, `fputs`, `fwrite`, `fputc`,
`getenv`, `pthread_setaffinity_np` and `__cxa_throw`. §9.2 removes the
stdio names; P5 removes the backtrace's `fork`, `execlp`, `_Exit` and
`prctl`; `abort` remains, reached only on the fatal path, and the
submission comments say so plainly rather than hiding the symbol.

**Revised in Stage 1 (2026-10-14).** CRAN's incoming pretest is
automated and reads no submission comment, and a family package's first
upload was archived for justifying such a symbol instead of removing
it. So `zlm_compat.h` maps `abort()` and `_Exit()` to `zlm_abort()`,
which executes a trap instruction (`__builtin_trap()`): the fatal path
still ends the process at once, by `SIGILL` or `SIGTRAP` instead of
`SIGABRT`, after the message reaches file descriptor 2. No symbol on
R's list is linked, so P5 is not needed for the check (R does not list
`fork`, `execlp` or `prctl`); it stays a candidate for upstream. The
`build-flags` workflow fails if any listed symbol returns.

### 13.3 Everywhere

- C17 for the glue and ggml's C, C++17 for the shim and ggml's C++
  (`CXX_STD = CXX17`); no OpenMP (ggml's own pool is used; upstream's
  OpenMP path also calls `setenv("KMP_BLOCKTIME")`); no system
  libraries. **Stage 1:** no GNU make: the wrappers are ordinary
  sources for R's suffix rules, `configure` (POSIX `sh`) appends the
  variant set for the compiler's target to `src/Makevars`, and each
  variant is linked by an explicit rule, so `SystemRequirements` names
  only C++17. C is compiled in R's default standard (C17 or C23,
  §9.3 P8).
- Warnings: the CPU-only sources gave 231 warnings under
  `-Wall -Wextra -pedantic`, all `-Wunused-function`, 220 of them from
  `static` helpers in `ggml-impl.h` repeated per file (Stage 1's
  arm64 build: 232, all `-Wunused-function`). Upstream builds
  with `-Wno-unused-function`, which R's check would report as a
  warning-suppressing flag; zulm lets them stand (they are not on
  CRAN's list of significant warnings, unverified against the current
  incoming checks). The warnings that upstream's stripped pragmas
  (§9.2) used to silence join them: `-Woverlength-strings` and
  `-Wpedantic` from string literals and vector extensions in
  `repack.cpp` and `sgemm.cpp`, none on the significant list either.
- Hidden symbols (R3): `-fvisibility=hidden` everywhere, plus one
  linker mechanism per platform, since a version script is GNU ld
  only: `-Wl,--version-script` on Linux, `-Wl,-exported_symbols_list`
  on macOS, and a `.def` file (or relying on visibility alone, which
  MinGW honours for `__declspec`-less symbols) on Windows. Each exports
  only `R_init_zulm` (and `R_init_` of each variant). In the
  drafting build the Linux version script left one dynamic symbol;
  Stage 1 audits all three with `nm -D`, `nm -gU` and `objdump -p`.
  **Stage 1:** visibility alone leaves libstdc++'s template instances
  (`std::map`, `std::mt19937`, a ggml template) exported on Linux,
  because libstdc++ declares `std` with default visibility; `configure`
  therefore links each variant with a version script whenever a test
  link shows the linker takes one. macOS needs nothing beyond
  visibility, and Windows links each DLL with a `.def` file naming its
  `R_init_`. `tools/check-exports`, run by `build-flags`, fails if any
  shared object exports anything else.
- `getenv()` reads only debugging switches (`GGML_SCHED_DEBUG`,
  `GGML_META_DEBUG`, `GGML_NO_BACKTRACE`) and `GGML_CPU_DISABLE_FUSION`,
  which changes kernel fusion and so possibly bits; `zulm_info()`
  reports whether it is set. `pthread_setaffinity_np` is reached only
  with a CPU mask, which zulm never sets.
- Windows: ggml's CPU backend has its own Win32 threading layer; whether
  it is reliable under Rtools' MinGW GCC, and whether the §13.2
  exception crosses C frames under SEH, are unverified and part of
  Stage 1. macOS: unverified beyond reading the source, on both
  architectures (§13.1).

### 13.4 Makevars and the native build hook

The rules, as zufft verified them against R 4.6.1 (RFC 0026 §12.3) and
as read in R's `check.R`:

- **The install log is scanned** under `--as-cran` for non-portable
  flags: any `-m*` other than `-m32`, `-m64`, `-msse2`, `-mfpmath=sse`,
  `-mstackrealign` and `-mtune=*`; `-ffast-math` and the other
  unsafe-math `-f` flags; and `-W*` flags off a short allow-list. A NOTE,
  escalated to a WARNING by `-Wno-*`. This is the check §13.1 is built
  around.
- **`Makevars` itself is checked separately** ("compilation flags in
  Makevars"), a WARNING for every `-O*`, `-W*`, `-w`, `-ansi`,
  `-pedantic`, `-f*`, `-m*`, `-std*` or `-g` token in a `PKG_*FLAGS`
  variable. Macros from R's `Makeconf` such as `$(C_VISIBILITY)` pass
  because the check does not expand them, and so does a macro that is
  empty unless the user sets it. Hence no `-f` flag in zulm's
  `Makevars` at all: `-fexceptions` (§13.2), `-fvisibility=hidden`
  (§13.3) and the unrolling of §13.1 go through `Makeconf` macros,
  pragmas in the force-included header, or per-file rules whose flags
  the check does not see but the log scan does.
- **R's flags come last**, so an optimisation level in `PKG_*FLAGS` is
  overridden by the site's `-O2`; zulm sets none.

The file, as Stage 1 wrote it (`src/Makevars.in`; `configure` appends
the fragment `src/make/<set>.mk` that `tools/gen-build` generated for
the target, `x86_64`, `arm64`, `generic` or a `native_*` set):

```make
# src/Makevars.in (design §13.4)
CXX_STD = CXX17
PKG_CPPFLAGS = -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu \
  -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_SCHED_MAX_COPIES=4 \
  -DGGML_SCHED_MAX_SPLIT_INPUTS=4 -pthread
PKG_CFLAGS = $(C_VISIBILITY) $(ZULM_CFLAGS)
PKG_CXXFLAGS = $(CXX_VISIBILITY) $(ZULM_CXXFLAGS)
PKG_LIBS = -pthread
OBJECTS = zulm.o                 # the registered DLL is the probe alone

# src/make/x86_64.mk (generated), in outline:
ZLM_COMMON = zlm_glue.o ... zlm_v/common/ggml.o ...
ZLM_VARIANTS = zulm_baseline.so zulm_sse42.so zulm_avx2.so
all: $(SHLIB) $(ZLM_VARIANTS)
ZLM_OBJ_avx2 = zlm_v/avx2/zlm_init.o zlm_v/avx2/ggml-cpu_ops.o ...
zulm_avx2.so: $(ZLM_COMMON) $(ZLM_OBJ_avx2)
	$(SHLIB_LINK) -o $@ $(ZLM_COMMON) $(ZLM_OBJ_avx2) $(ZLM_LDFLAGS_avx2) $(ALL_LIBS)
```

Target names carry `.so` (or `.dll` in `Makevars.win`) literally: make
expands them when it reads `Makevars`, before `Makeconf` defines
`SHLIB_EXT`. `ZLM_LDFLAGS_<variant>` is the version script of §13.3,
set by `configure` where the linker takes one. `src/install.libs.R`
copies every shared object into `libs/` and records the hooks.

**The native hook, as zufft does it (D40).** A user who wants a build
for their own CPU sets the two variables and installs from source:

```r
Sys.setenv(ZULM_CFLAGS = "-march=native", ZULM_CXXFLAGS = "-march=native")
install.packages("zulm", type = "source")
```

- Two hooks, not one, because ggml's kernels are both C
  (`ggml-cpu.c`) and C++ (`ops.cpp`, `vec.cpp`, `sgemm.cpp`); a flag
  set on one language alone would build a mixed object. The README
  shows both lines and `zulm_info()` reports the flags, which
  `src/install.libs.R` records at install time.
- The hooks reach only zulm, unlike `~/.R/Makevars`, and need no
  `configure`. With them unset both of R's flag checks pass, because the
  macros expand to nothing.
- **With a hook set, one variant is built**, named `native`, and the
  force-included header skips its target pragma when the compiler
  already defines the ISA macros (`__AVX2__` and so on), so the flag,
  not the pragma, decides the kernels. `.onLoad` loads `native` without
  probing. `configure` makes the choice, so no GNU make is needed
  (§13.3). A native build is its
  own ISA variant under §6.5: results reproduce within it, and not
  against CRAN's binaries.
- **Not on Windows.** `Makevars.win` carries no hook, as zufft's does
  not, because AVX code from Rtools' MinGW-w64 GCC crashed on load in
  zufft's CI (§13.1). If Stage 1 shows ggml's AVX2 code loads under
  Rtools, the hook is added to `Makevars.win` then, with the reason
  recorded here.
- **Checked in CI**, with zufft's two workflows carried over through
  `pedrobtz/r-actions`: `build-flags` runs `R CMD check --as-cran` on
  Linux and macOS and fails on the non-portable-flag NOTE or any
  compiled-code NOTE, so no flag can drift into `Makevars` without a
  design change; `portability` builds with the hooks set on Linux and
  macOS, checks the flags reach every C and C++ compile, checks
  `zulm_info()` reports `native`, and runs the suite and the §12
  benchmark, whose native row joins the design's table. The r-actions
  check already fails on the Makevars WARNING on every leg.
- **What it buys.** On x86-64 with AVX-512, the native build is the
  `-march=native` row of §12 (33 GFLOP/s against the `avx2` variant's
  27–31); on arm64 and on an AVX2 machine it buys little or nothing over
  the shipped variants, and the README says so, as zufft's does ("a
  native build is only worth it on Linux x86-64").

## 14. Roadmap

The stages are kept in `roadmap.md`, which holds each stage's work and
exit criteria and is authoritative; this table is the summary. The
numbering follows the roadmap: Stage 0 is package identity alone, and
the vendoring go/no-go is Stage 1.

| Stage | Title | Size |
| --- | --- | --- |
| 0 | Package identity | S |
| 1 | Vendored ggml and the CPU-dispatch go/no-go: vendor script (pragma strip); P1, P2, P7 upstream; §13.1 on GCC and clang; §13.4 Makevars; symbols on three platforms | L |
| 2 | Tensors, types, store, GGUF | M |
| 3 | Graphs, operations (including the forward-only set and both rotary conventions, §5.3), forward, validation, §13.2 recovery | L |
| 4 | Backward pre-check, trainer, optimisers, P7 clipping, gradient checks | M |
| 5 | zurand in graphs (§7.2); per-row loss sums with the fixed-order fallback | S |
| 6 | Modules; the §1.1 model end to end; sampling; GGUF fixtures from PyTorch, including the two §1.2 blocks | M |
| 7 | Thread pool, polling, interrupts, fork, determinism tests | M |
| 8 | safetensors through `zujson`; f16 and bf16 weights in memory, `zl_cast()`, the precision test | M |
| 9 | State tensors, `set_rows`, the graph cache; the decode-loop test; `input_write_f32_at` | M |
| 10 | Flash attention forward-only: mask format, agreement with the plain form, thread-count measurement | S |
| 11 | Arrow iterator and `batch_fill`, with nanoarrow, DuckDB and Parquet fixtures | M |
| 12 | The C table, `zulm.h`, a fixture consumer; zukzeit's Toto 2.0 4M on zulm in a zukzeit branch, parity green | M |
| 13 | Fuzzing, benchmarks (training and inference), documentation | M |
| 14 | v0.1.0 readiness | S |

Stages 1 to 7 are the training substrate, 8 to 11 the inference and
data pieces the two consumers need, and 12 the proof that both can
build on the result: a fixture consumer for zuflow's pattern of use,
and zukzeit's smallest certified checkpoint running on zulm from a
branch of zukzeit, which is where the fixtures live. TimesFM 2.5 and
Chronos-2 are ported in zukzeit after zulm's release, on zukzeit's
roadmap; zulm's own gate for them is the TimesFM-style decode test of
Stage 9 and the §1.2 block fixtures of Stage 6.

**0.2**, in order: `zuexact` lookup (S), spectra with P6 upstream (S),
`argsort` and `top_k` if asked (S). RFC 0032's dependency on "zuggml's
Stages 5 and 9" reads zulm's Stages 6 and 12.

## 15. Acceptance criteria for v0.1.0

1. `R CMD check --as-cran` clean on every CI leg, including x86-64
   macOS, with at most two cores, apart from a documented `abort`
   symbol NOTE if §13.2's residual cannot be removed; no pragma NOTE
   (§9.2); §13.1's outcome recorded for GCC and clang.
2. The driving model trains on CPU from `zl_seed(1)`: a causal
   transformer encoder (width 64, four heads, one block), a four-class
   kind head and a flow-matching head over seven targets with AdaLN, on
   the synthetic marked process of the drafting probe, 1 000 AdamW
   steps; the mean loss of steps 901–1 000 is below 60 % of that of
   steps 1–100 (the C probe reached 44 %); parameters and reported
   losses are bit-identical at one and two threads; noise comes from
   Philox inside the graph; Euler sampling of the head returns finite
   values.
3. Every operation with a backward pass passes its gradient check; the
   PyTorch fixtures, loaded from GGUF, reproduce PyTorch's outputs and
   gradients within tolerance; a step with `clip_norm` scales
   gradients by the factor computed in R.
4. No test in the hostile-input suite and no `zufuzz` run reaches a
   ggml assertion.
5. `tools/vendor-ggml` reproduces `src/ggml/` byte for byte from the tag
   and the patches; each patch names its upstream pull request.
6. On an AVX2 CI runner, the `avx2` variant loads and meets §12.
7. A fixture package fills inputs and reads outputs through the C table.
8. `tools/vendor-ggml` leaves no diagnostic pragma in `src/ggml/`, and
   every variant's shared object exports only its `R_init_` symbol on
   Linux, macOS and Windows.
9. Every forward-only operation of §5.3 matches its R oracle; the two
   §1.2 blocks, loaded from GGUF, reproduce their PyTorch fixtures
   within tolerance, so both rotary conventions, partial rotation,
   broadcast masks and the unit-scaled forms are pinned before any
   checkpoint is ported.
10. zukzeit's committed parity fixtures for Toto 2.0 4M pass with the
    port's tensor runtime replaced by zulm, within zukzeit's documented
    tolerance, on CPU, on every CI leg, from a branch of zukzeit that
    its maintainer can merge after zulm's release.
11. A TimesFM-style decode of 256 values through state tensors equals
    the full-context forward pass within f32 tolerance; weights
    exported by PyTorch as safetensors load into the matching modules
    through `zujson`; f16 weights reproduce f32 results within the
    tolerance §6.9 documents; the §12 inference targets are met or
    their misses recorded.
12. The §1.1 model trains from a DuckDB table through an Arrow stream
    with results identical to training from the same rows as R arrays.

For 0.2: with `zuexact`, reported losses are identical under
regrouping; `zl_spectrum()` agrees with `zufft` and `stats::fft()`.

## 16. Decisions

| # | Question | Decision |
| --- | --- | --- |
| D1 | Relation to ggmlR | an alternative; no shared code or objects |
| D2 | Library | upstream ggml, unmodified, pinned, re-vendored |
| D3 | Local changes | numbered patches with upstream PRs only |
| D4 | Backends | CPU only in v0.1.0; GPUs later, opt-in |
| D5 | Autograd | ggml's backward pass; no R tape |
| D6 | Graph API | explicit: build once per shape, run many times |
| D7 | Training | `ggml-opt` static graphs; batches from the caller |
| D8 | Provider | versioned C table of opaque handles; no headers |
| D9 | Dispatch | whole-package ISA variants, chosen at load |
| D10 | Aborts | validate; catch while building; fatal in compute |
| D11 | Threads | own persistent pool; default 1; no OpenMP |
| D12 | Determinism | bit-identical per variant and thread count |
| D13 | Types | f32 compute; f16, bf16 for storage and, opt-in, in memory (§6.9) |
| D14 | Excluded files | `ggml.cpp`, backend registry, GPU backends |
| D15 | Info function | `zulm_info()`, the package name (R1) |
| D16 | Companions | `zurand` and `zujson` are `Imports`; `nanoarrow` is `Suggests` in v0.1.0; `zuexact` and `zufft` are `Suggests` from 0.2, reached through their C tables with a fallback when absent |
| D17 | Vendoring | only small stable code: ggml and the Arrow C data interface definitions in v0.1.0; no JSON library, no RNG |
| D18 | Reported sums | per-row values, fixed-order `double` in v0.1.0; `zuexact` optional in 0.2 |
| D19 | Random numbers | `zurand` keys: session key, a fold per stream name, a fold per step, a position per element (§7.2) |
| D20 | RNG source | `zurand`'s C table, version 3 or later, as `Imports` and `LinkingTo`; no fallback, no vendored Random123 |
| D21 | Data input | Arrow C stream; one copy; main thread only; v0.1.0 |
| D22 | Weight exchange | GGUF, with PyTorch fixtures via `gguf-py`; safetensors through `zujson`; both v0.1.0 |
| D23 | FFT | forward via `zufft`; DFT products for gradients; 0.2 |
| D24 | BLIS, Highway | rejected for v0.1.0 (§7.7) |
| D25 | zuflow boundary | primitives here; algorithms in RFC 0032 |
| D26 | Gradient clipping | patch P7 inside `ggml-opt`; no zulm-owned optimiser graph |
| D27 | Releases | decided by the maintainer 2026-10-14: v0.1.0 carries everything zuflow 0.1.0 and zukzeit 0.1.0 need (training, inference, safetensors, state tensors, Arrow), proved on Toto 2.0 4M; 0.2: `zuexact`, the FFT and sorting, which neither needs |
| D28 | clang | part of the §13.1 go/no-go, because CRAN ships x86-64 macOS binaries |
| D29 | Upstream pragmas | stripped by the vendor script, not patched (§9.2) |
| D30 | Inputs | in a static buffer of the graph's own; named outputs flagged (§4); rewritable between runs (§6.2) |
| D31 | Name | `zulm`, by the repository's creation; RFC 0031's `zuggml` is history |
| D32 | R prefix | `zl_`, decided by the maintainer 2026-10-14; the family's "name minus zu" rule gives `lm_`, which collides with `stats::lm()` and the `lm_*` functions on CRAN |
| D33 | Second consumer | zukzeit's inference (§1.2, §7.9); decided by the maintainer 2026-10-14 that zukzeit uses zulm only, dropping `torch`; zulm proves the path on Toto 2.0 4M before 0.1.0 |
| D34 | Key-value cache | fixed-capacity state tensors written with `set_rows` under a mask; one graph per `n_new` (§6.9) |
| D35 | Weight types in memory | f32 by default and for parity; f16, bf16 opt-in with the precision cost quantified (§6.9) |
| D36 | safetensors header | parsed by `zujson` from R; yyjson not vendored (§7.4) |
| D37 | Forward-only operations | in v0.1.0 when each is one upstream builder or a composition; refused on gradient paths (§5.3) |
| D38 | Rotary conventions | both, through `ggml_rope_ext`'s modes, with partial rotation; interleaved is the default (§5.3) |
| D39 | Preprocessing of pretrained models | in R on doubles, outside the graph (§6.9) |
| D40 | Native build | zufft's hook: `ZULM_CFLAGS` and `ZULM_CXXFLAGS` from the environment in `src/Makevars`, empty by default, not in `Makevars.win`; one `native` variant; README and CI as zufft (§13.4) |
| D41 | Flags in Makevars | none that R's Makevars check rejects; unrolling, exceptions and targets by pragma or `Makeconf` macro (§13.1, §13.4) |

## 17. Open questions

1. **The R prefix.** **Decided 2026-10-14: `zl_` (D32).** The family
   rule gives `lm_`, rejected for `stats::lm()`; `gg_` is taken by other
   packages (naniar's `gg_miss_*`, for one) and names the library rather
   than the package; `ulm_` is free but reads as nothing. `zl_` pairs
   with zuflow's `zf_` and zukzeit's `zuk_`.
2. **zukzeit's runtime.** **Decided 2026-10-14: zukzeit uses zulm
   only (D33).** `torch` leaves zukzeit; CUDA and MPS go with it until
   zulm has a GPU backend (question 5). The consequence for zulm is
   that v0.1.0 must suffice for all three certified checkpoints, which
   is what D27 and the §1.2 inventory provide; the consequence for
   zukzeit is that its 0.1.0 waits for its three ports to pass parity
   on zulm, Toto first.
3. **C table or ggml headers through `LinkingTo`?** Recommended: the
   table (§8). Revisit only if a consumer must add ggml operations from
   C, and then as table members that build nodes, not as headers.
4. **ggmlR's backward kernels.** Propose to ggmlR's maintainer to
   upstream his `NORM_BACK`, `GELU_BACK` and flash-attention backward
   kernels to ggml, with zulm's P1–P2 alongside? Recommended: yes,
   before writing P3 and P4 independently. Upstream gains training
   coverage, ggmlR's fork shrinks toward upstream, and zulm vendors the
   result. It also makes the two packages collaborators on the shared
   layer rather than competitors on it.
5. **GPUs: when, and which first?** Recommended: not before zuflow shows
   where the CPU limits it; then Vulkan as an opt-in `configure` switch
   (the path ggmlR has proven on CRAN), Metal second. The C table and
   the R API do not change for it.
6. **The zuflow split.** Keep everything specific to event sequences
   (feature representation, atom smoothing, samplers, total-variation
   and discriminator evaluation, rollouts) in zuflow, and give zulm
   only generic modules? Recommended: yes; zulm's modules stop at the
   transformer block and AdaLN.
7. **Variants and kernels.** Is `avx512` worth a fourth x86-64 variant,
   and should the AVX2 variant use llamafile kernels or the unrolling
   pragma of §13.1, or both? Recommended: decide from Stage 1's
   measurements of training steps and thread-count determinism, not
   from matrix products alone. **Decided 2026-10-14 (Stage 1), from
   products, as no training step exists before Stage 4:** both, and no
   `avx512` variant (§13.1). Stage 4 re-checks thread invariance on
   training steps and revisits llamafile if it fails.
8. **Long contexts.** The post evaluates on 10 240-event contexts; plain
   attention then holds a 10 240² score matrix (419 MB in f32) per head
   and sequence. Upstream's forward-only flash attention is in v0.1.0
   (§6.9, Stage 10) for sampling and inference graphs. Neither
   consumer's first release strictly needs it (the §1.1 model trains
   at 64 events, the foundation models attend over at most 512
   patches), so if Stage 10's measurement finds it thread-dependent or
   the mask format costly, it may ship disabled by default with the
   plain form as the reference, rather than hold the release.
9. **Two members for zurand.** **Asked 2026-10-14:**
   [zurand#63](https://github.com/pedrobtz/zurand/issues/63)
   (`fold_bytes()`) and
   [zurand#64](https://github.com/pedrobtz/zurand/issues/64) (f32
   fills). zulm's Stage 5 does not wait for them: until they exist,
   names are folded in R and the f32 rounding rule is zulm's (§7.2).
   When they land, Stage 5 or a later patch switches to them with a
   test that the bits are unchanged.
10. **`zuexact` as a hard dependency?** Recommended: no; the fallback
    already meets the thread contract, and exactness under regrouping is
    an extra worth a `Suggests`.
11. **Which zurand engine for the session key?** `philox4x64` (one
    block per four values, cheap random access) or zurand's default
    `xoshiro256pp` (faster bulk fills, a chunk regenerated when a pool
    task starts mid-chunk)? Recommended: `philox4x64`, chosen by
    `zl_seed()`; a user who passes an `xoshiro256pp` key gets it
    honoured, since the engine lives in the key. The former question
    here, whether normals agree across platforms, is answered by
    zurand's fdlibm port and CI.
12. **P6 upstream.** Propose a backward callback for custom operations?
    Recommended: yes, early; it is the one change that lets companions
    (`zufft`, `zuspecial`) take part in training without patches to
    ggml's operation list.
13. **`zujson`: `Suggests` or `Imports`?** **Decided with D27:
    `Imports`.** Loading weights is not an optional feature of a
    substrate that zukzeit 0.1.0 builds on, and zujson is one small
    CRAN package with no compiled dependency of its own. RFC 0031's
    alternative, vendoring yyjson, is withdrawn (D36).
14. **Arrow without nanoarrow.** Accept only
    `nanoarrow_array_stream` pointers, or also raw `ArrowArrayStream`
    addresses from other packages? Recommended: the nanoarrow class
    only; it is the common currency and carries ownership correctly.
15. **The shape of P7.** A `clip` member in the optimizer parameters,
    or a general hook in `ggml_opt_build` between the backward pass and
    the step, which zulm would use for clipping and others could use
    for anything? Recommended: the member; it is the smaller upstream
    change, it keeps the whole step inside one graph, and no second
    use for a hook is known.
16. **Flash attention and determinism.** Whether
    `ggml_flash_attn_ext`'s CPU kernel gives the same bits at every
    thread count is unmeasured; the plain form does (§6.5).
    Recommended: measure in Stage 10; if it does not, graphs that use
    it carry a documented exception to §6.5 above one thread, and
    zukzeit's parity runs stay on the plain form.
17. **The graph cache: R or C?** §6.9 keeps it an R environment.
    zuflow's samplers vary no shapes from C, and zukzeit's decode loops
    are in R, so no consumer needs it from the table. Recommended: R
    only until a consumer asks; a table member that builds graphs would
    have to carry the `build()` closure, which is an R object.
18. **Weight types and the parity promise.** Should zulm refuse
    f16 and bf16 weights in a store whose consumer declares a parity
    budget, or only document the cost? Recommended: document and
    report (`zulm_info()`); the consumer that runs fixtures knows which
    types it loaded. If zukzeit wants a guard, it is one line in its
    loader.

## 18. What this design does not decide

- zuflow's model, data pipeline, samplers and evaluation: its own RFC.
- The order and timing of zukzeit's three ports to zulm; this design
  proves the path on one checkpoint (§15) and zukzeit's roadmap records
  the rest. That zukzeit uses zulm only is decided (D33).
- Whether ggmlR's users should move; the packages serve different
  purposes and both remain.
- GPU backends, mixed-precision training and quantised inference beyond
  their placement under Later.
- Whether upstream accepts P1–P5 and P7; if it rejects one, the patch
  stays with its rejection recorded, or the workaround becomes
  permanent.
- Whether zuflow's rollouts adopt state tensors (§6.9) in its first
  release or later; zulm 0.1.0 offers them either way.

Review prepared with assistance from generative AI.
