# zulm: roadmap

Status: draft 2026-10-14. Implements [design.md](design.md).
Target: v0.1.0, the first CRAN release.
Tracking: [v0.1.0 (#1)](https://github.com/pedrobtz/zulm/issues/1), with one
sub-issue per stage; [CRAN-submission (#17)](https://github.com/pedrobtz/zulm/issues/17).

v0.1.0 carries everything zuflow 0.1.0 and zukzeit 0.1.0 need (design D27):
the training substrate, the forward-only operations and weight formats of
pretrained inference, state tensors, and Arrow batches. Only the `zuexact`
lookup, the FFT and sorting wait for 0.2.

Where the package is on 2026-10-14: a `usethis` skeleton with placeholder
`DESCRIPTION` fields, one empty C file, one placeholder test, an
`R-CMD-check` workflow that already calls `pedrobtz/r-actions`, coverage
and pkgdown workflows, and the design. Nothing of the design is built.

## Stages

| Stage | Title | Issue | Pull request | Done |
| --- | --- | --- | --- | --- |
| 0 | [Package identity](#stage-0--package-identity) | [#2](https://github.com/pedrobtz/zulm/issues/2) | [#19](https://github.com/pedrobtz/zulm/pull/19) | 2026-10-14 |
| 1 | [Vendored ggml and the CPU-dispatch go/no-go](#stage-1--vendored-ggml-and-the-cpu-dispatch-gono-go) | [#3](https://github.com/pedrobtz/zulm/issues/3) | [#21](https://github.com/pedrobtz/zulm/pull/21) | 2026-10-14 |
| 2 | [Tensors, types, store and GGUF](#stage-2--tensors-types-store-and-gguf) | [#4](https://github.com/pedrobtz/zulm/issues/4) | [#22](https://github.com/pedrobtz/zulm/pull/22) | 2026-10-14 |
| 3 | [Graphs, operations, forward and validation](#stage-3--graphs-operations-forward-and-validation) | [#5](https://github.com/pedrobtz/zulm/issues/5) | [#24](https://github.com/pedrobtz/zulm/pull/24) | 2026-10-14 |
| 4 | [Autograd, trainer and optimisers](#stage-4--autograd-trainer-and-optimisers) | [#6](https://github.com/pedrobtz/zulm/issues/6) | [#25](https://github.com/pedrobtz/zulm/pull/25) | 2026-10-14 |
| 5 | [zurand in graphs and per-row loss sums](#stage-5--zurand-in-graphs-and-per-row-loss-sums) | [#7](https://github.com/pedrobtz/zulm/issues/7) | [#26](https://github.com/pedrobtz/zulm/pull/26) | 2026-10-14 |
| 6 | [Modules, the driving model and PyTorch fixtures](#stage-6--modules-the-driving-model-and-pytorch-fixtures) | [#8](https://github.com/pedrobtz/zulm/issues/8) | [#27](https://github.com/pedrobtz/zulm/pull/27) | 2026-10-14 |
| 7 | [Thread pool, interrupts, fork and determinism](#stage-7--thread-pool-interrupts-fork-and-determinism) | [#9](https://github.com/pedrobtz/zulm/issues/9) | [#28](https://github.com/pedrobtz/zulm/pull/28) | 2026-10-14 |
| 8 | [safetensors and weight types](#stage-8--safetensors-and-weight-types) | [#10](https://github.com/pedrobtz/zulm/issues/10) | [#29](https://github.com/pedrobtz/zulm/pull/29) | 2026-10-14 |
| 9 | [State tensors and the graph cache](#stage-9--state-tensors-and-the-graph-cache) | [#11](https://github.com/pedrobtz/zulm/issues/11) | [#30](https://github.com/pedrobtz/zulm/pull/30) | 2026-10-14 |
| 10 | [Flash attention](#stage-10--flash-attention) | [#12](https://github.com/pedrobtz/zulm/issues/12) | [#31](https://github.com/pedrobtz/zulm/pull/31) | 2026-10-14 |
| 11 | [Arrow batches](#stage-11--arrow-batches) | [#13](https://github.com/pedrobtz/zulm/issues/13) | | |
| 12 | [The C table and the zukzeit proof](#stage-12--the-c-table-and-the-zukzeit-proof) | [#14](https://github.com/pedrobtz/zulm/issues/14) | | |
| 13 | [Fuzzing, benchmarks and documentation](#stage-13--fuzzing-benchmarks-and-documentation) | [#15](https://github.com/pedrobtz/zulm/issues/15) | | |
| 14 | [v0.1.0 readiness](#stage-14--v010-readiness) | [#16](https://github.com/pedrobtz/zulm/issues/16) | | |

Stages 1 to 7 are the training substrate, in dependency order with the
riskiest first: Stage 1 is a go/no-go that can change the design before
any R surface is written. Stages 8 to 11 are the inference and data
pieces the consumers need; 12 proves both consumers can build on the
result; 13 and 14 close the release.

## Stage 0 — Package identity

**Issue.** [#2](https://github.com/pedrobtz/zulm/issues/2), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** The skeleton describes the real package and carries the
family's conventions, so every later pull request is checked by the
same CI.

**Work.**
- `DESCRIPTION`: `Title` ("Tensor Graphs, Training and Inference on
  'ggml'"), a `Description` paragraph after design §1, `Authors@R` with
  Pedro Baltazar as `aut`, `cre`, `cph` (the vendored copyright holders
  join in Stage 1), `Depends: R (>= 4.1)`, `Imports: zurand, zujson`,
  `LinkingTo: zurand`, `Suggests: testthat (>= 3.0.0), withr, nanoarrow`,
  `SystemRequirements: C++17`, `Language: en-GB`, `URL` and `BugReports`.
  `Remotes: pedrobtz/zurand` until zurand is on CRAN, removed in Stage 14.
- `CLAUDE.md` from `templates/package-CLAUDE.md` in `pedrobtz/packages`,
  linking `design.md` and this roadmap and stating that the design
  outranks it.
- `README.md`: the one-line statement of design §1 and the two use
  cases; the placeholder example removed.
- The `stage` label, the `v0.1.0` parent issue and the stage sub-issues
  (created by the roadmap skill).
- `.Rbuildignore` already excludes `.agents`; the `R-CMD-check`,
  `coverage` and `pkgdown` workflows already call r-actions and stay.

**Exit criteria.**
- `R CMD check` passes on every r-actions leg with the skeleton.
- No placeholder text remains in `DESCRIPTION` or `README.md`.
- `CLAUDE.md` exists, follows the template, and links the design and the
  roadmap.

**Design.** §1, §3 (the position table), §13.3 (what `DESCRIPTION`
declares).

## Stage 1 — Vendored ggml and the CPU-dispatch go/no-go

**Issue.** [#3](https://github.com/pedrobtz/zulm/issues/3), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Upstream ggml's CPU subset builds inside the package on every
CI leg, as several ISA variants chosen at load, with hidden symbols and
no CRAN flag or pragma NOTE; or the design records that it cannot and
takes the documented fallback. Nothing else is built until this is
settled.

**Work.**
- `tools/vendor-ggml`: clone the pinned tag (`v0.26.0`), copy the file
  list of design §9.1, strip the diagnostic pragmas of §9.2, apply
  `tools/patches/*.patch`, write `ggml-version.h` and
  `src/ggml/VERSION` with tag, commit, patch list and per-file SHA-256.
- Patches P1, P2 and P7 drafted under `tools/patches/` with upstream
  pull requests opened and their links recorded in design §9.3; P5 if
  the backtrace symbols cannot otherwise be kept out of the object.
- `src/Makevars` and `src/Makevars.win` per design §13.4: visibility
  macros from `Makeconf`, the `ZULM_CFLAGS` and `ZULM_CXXFLAGS` hooks on
  Unix only, no token that R's Makevars check rejects; the variant
  rules of §13.1 (`baseline`, `sse42`, `avx2` on x86-64, one on arm64,
  `native` when a hook is set); `SystemRequirements: GNU make` declared
  if the rules need it.
- The force-included header: `#pragma GCC target` for GCC,
  `#pragma clang attribute push` for clang, the ISA macros defined when
  the compiler does not, skipped when the command line already defines
  them; `#pragma GCC optimize("unroll-loops")`; `-fexceptions` and
  `-fvisibility=hidden` reached without a `PKG_*FLAGS` token.
- `src/install.libs.R` installing every variant and recording the
  hooks; the probe DLL and `.onLoad` choosing a variant with
  `__builtin_cpu_supports()`; `zulm_info()` reporting tag, commit,
  patches, variant and flags.
- `zlm_compat.h` redirecting stdio to the log buffer (§9.2); the C++
  shim skeleton with the abort callback of §13.2, and the test that a
  thrown assertion is caught on Linux and macOS, with the Windows (SEH)
  outcome recorded.
- Symbol audit on Linux, macOS and Windows (`nm -D`, `nm -gU`,
  `objdump -p`): only `R_init_*` exported.
- Measurements on the AVX2 runner for the `avx2` variant against a
  `-mavx2` build under GCC and clang, the unrolling pragma against
  `-funroll-loops`, llamafile kernels on and off (speed and thread
  invariance), Windows `avx2` under Rtools (loads, or the `sse42`
  fallback), ggml's Win32 threading under Rtools, and install time and
  size; all recorded in design §12 and §13.1.
- Workflows: `build-flags` (fails on the non-portable-flag NOTE and any
  compiled-code NOTE) and `portability` (builds with the hooks set),
  both through r-actions as zufft's are.

**Exit criteria.**
- `R CMD check --as-cran` is clean on every r-actions leg, including
  x86-64 macOS, with no non-portable-flag NOTE and no pragma NOTE; the
  `abort` symbol NOTE, if it remains, is documented in design §13.2.
- `.onLoad` loads the right variant on each runner, and
  `options(zulm.variant =)` forces another.
- The `avx2` variant is within 15 % of the `-mavx2` build under GCC and
  under clang, or the shortfall and the fallback decision are written
  into design §13.1.
- Every variant's shared object exports only its `R_init_` symbol on
  Linux, macOS and Windows.
- `tools/vendor-ggml` reproduces `src/ggml/` byte for byte, and a test
  finds no diagnostic pragma under it.
- P1, P2 and P7 name their upstream pull requests.
- `build-flags` and `portability` are green.

**Design.** §9, §12, §13.1 to §13.4; decisions D2, D3, D9, D10, D28,
D29, D40, D41; open question 7.

**Deviations (2026-10-14).** P1, P2 and P7 are drafted in
`tools/patches/drafts/` but their upstream pull requests are not opened:
that is the maintainer's action, so the criterion "P1, P2 and P7 name
their upstream pull requests" moves to Stage 4, which vendors them.
Patch P8 (C23 `static_assert`) was needed for the clang legs and is
vendored ahead of its upstream pull request (design §9.3). `abort` is
not linked at all (design §13.2), so no NOTE needs documenting.
`build-flags` and `portability` are local workflows, as zufft's are;
r-actions has no reusable version of them.

## Stage 2 — Tensors, types, store and GGUF

**Issue.** [#4](https://github.com/pedrobtz/zulm/issues/4), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** A parameter store holds named f32, f16 and bf16 tensors in one
backend buffer, moves data to and from R arrays without reordering, and
persists as GGUF.

**Work.**
- `zl_params()`, `zl_param()` with `init` and `type`, the `zl_init_*()`
  constructors (normal and uniform draws through zurand's R API under
  the parameter's name, the counter layout of §7.2 fixed here and
  reused in Stage 5), `zl_param_get()`, `zl_param_set()`,
  `zl_param_names()`, `zl_param_count()`, `zl_param_tensor()`.
- R doubles to f32 with `zulm_range_warning`, f16 and bf16 through
  ggml's row converters, i32 index arrays refusing `NA`; dims up to
  four; `R_xlen_t` to `int64_t` with no `int` between.
- `zl_save()` and `zl_load()` on GGUF with metadata; `type` on save.
- External pointers with finalisers, the memory cap
  `getOption("zulm.max_bytes")`, `zulm_memory_error`,
  `zulm_stale_error` after `saveRDS()` or fork.
- The condition hierarchy of design §10 (every class defined, even
  those first signalled later).

**Exit criteria.**
- Round trips array to store to array at f32, f16 and bf16 are exact
  for f32 and within the type's rounding otherwise.
- GGUF files written by zulm are read by upstream's `gguf` Python
  package in CI, and files it writes are read by `zl_load()`.
- Hostile input (wrong types, `NA`, more than four dims, zero-length
  dims, freed objects, reloaded objects) raises the documented classes
  and never reaches a ggml assertion.
- Exceeding the memory cap is `zulm_memory_error` with the two byte
  counts.

**Design.** §5.1 (store and persistence), §6.1, §6.6, §10; D13, D22.

**Deviations (2026-10-14).** A store is a list of allocation batches,
each with its own buffer, rather than one buffer (design §4, §6.6
revised): tensors then never move, so a graph built on a store stays
valid when parameters are added. Stores survive `fork()`; only objects
that own threads (Stage 7) become stale in a child. `zl_load()` widens
every tensor to f32 here; Stage 8 adds keeping f16 and bf16.

## Stage 3 — Graphs, operations, forward and validation

**Issue.** [#5](https://github.com/pedrobtz/zulm/issues/5), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** An R function builds a graph once per input shape from the
full operation set of design §5.3, including the forward-only
operations and both rotary conventions, and `zl_forward()` runs it any
number of times; every argument ggml would assert on is validated first,
and an assertion during construction is a catchable R error.

**Work.**
- `zl_graph()`, `zl_int()`, `zl_forward()`, `zl_graph_info()`; the
  three contexts of §4 (store, static inputs, compute); named outputs
  flagged; inputs rewritable between runs.
- Every operation of §5.3: arithmetic and reductions, `zl_cumsum()`,
  activations including forward-only `zl_gelu()`, `zl_sigmoid()`,
  `zl_tanh()`, `zl_swiglu()`, products and shape, `zl_concat()`,
  `zl_pad()`, `zl_get_rows()`, `zl_where()`, `zl_clamp()`, `zl_tri()`,
  `zl_softmax()` with broadcast masks, `zl_causal_mask()`, `zl_rope()`
  in both modes with partial `n_dims`, the norms, the losses,
  `zl_cast()`; the `Ops` group generic.
- The validation layer in the C glue and the shim's `try`/`catch` with
  the abort-callback exception of §13.2; `zulm_engine_error` discards
  the half-built graph.
- Logging through `ggml_log_set()` and `options(zulm.log =)` (§6.8).

**Exit criteria.**
- Every forward operation matches its base-R oracle within f32
  tolerance at random shapes including broadcasts; `zl_rope()` matches
  the split-half and interleaved formulas at full and partial `n_dims`;
  `zl_softmax()` with masks of every broadcast shape the kernel accepts.
- A graph's inputs rewritten a thousand times between runs give the
  same outputs as fresh graphs.
- A deliberately invalid construction raises `zulm_engine_error` on
  Linux and macOS, and the process continues correctly afterwards; the
  Windows outcome is recorded in design §13.2.
- The hostile-input suite for graph construction reaches no assertion.

**Design.** §4, §5.3, §6.1, §6.2, §6.8, §13.2; D6, D10, D30, D37, D38.

**Deviations (2026-10-14).** `zl_gelu()` gained `approximate = c("none",
"tanh")`, defaulting to the exact form, because ggml's tanh form reads
an f16 table (design §5.3). Indices count from 0 and must be graph
inputs, so their range is checked before every run. A deliberately
invalid construction is reached through an internal `unchecked:`
operation prefix that skips validation, since the public functions
validate everything.

## Stage 4 — Autograd, trainer and optimisers

**Issue.** [#6](https://github.com/pedrobtz/zulm/issues/6), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Graphs with a declared loss train through `ggml-opt` with
AdamW or SGD, gradient accumulation, a per-step learning rate and
global-norm clipping, and every operation with a backward pass has a
gradient check.

**Work.**
- The backward pre-check walking the forward graph and raising
  `zulm_unsupported_backward` for forward-only operations on a gradient
  path, naming the operation and the R call.
- The compositions of §6.3: layer norm as RMS norm of a centred input,
  broadcast subtraction as add of a negation, division by a broadcast
  operand refused until P2 is vendored.
- `zl_adamw()`, `zl_sgd()`, `zl_trainer()` with `accumulate` and
  `clip_norm`, `zl_step()` with `lr`, `zl_grad()`, `zl_reset()`;
  `zulm_nonfinite_warning`.
- P1, P2 and P7 vendored once their behaviour is pinned by tests
  (whether or not upstream has merged them); regression tests for each.

**Exit criteria.**
- Central finite differences in R agree with ggml's gradients for every
  operation and composition with a backward pass, at random shapes
  including broadcasts.
- A step with `clip_norm` scales every gradient by the factor computed
  in R; without P7 vendored, `clip_norm` is `zulm_invalid_argument`
  naming the patch.
- The P1 and P2 assertion cases of design §9.3 train.
- Forward-only operations on a gradient path are refused before ggml
  sees them.

**Design.** §6.2 to §6.4, §9.3; D5, D7, D26.

**Deviations (2026-10-14).** P7 does more than clip: it also makes every
gradient the optimiser step reads contiguous, since a parameter used
through a permutation has a strided gradient that upstream's step would
apply to the wrong elements. `zl_grad()` builds its own backward graph
on a copy of the forward graph, because `ggml-opt` keeps gradients only
inside its private copies. A trained graph holds at most 2048 nodes
between its parameters and its loss (`ggml_opt_init()` builds at
`GGML_DEFAULT_GRAPH_SIZE`). A number divided by a tensor is now
`scale_bias(x, 0, c) / x`, as `ggml_fill` has no backward. A weight
broadcast over the fourth dimension of `zl_crossprod()`'s second operand
has no gradient in ggml and is refused on a gradient path. Valgrind's
runner ran out of memory on the gradient tests: the trainer's scheduler
is now sized for `ggml-opt`'s graphs, and `zl_graph()` and `zl_trainer()`
collect garbage when unreferenced objects hold over 128 MB outside R's
heap (design §6.6); contexts are sized by use, since valgrind pays for
reserved memory. Three crashes that fuzzing found later, in Stage 13 but
in Stage 3's code, are fixed here: ggml's debug `assert()` aborting R on
a NaN in softmax (ggml's sources are now compiled with `NDEBUG`, as
upstream's release builds are), a transposed column reaching kernels
that assert dense rows, and a negative `eps` reaching `rms_norm` from
C. Stage 1 was to
add the vendored copyright holders to `Authors@R`; this stage does, with
`inst/COPYRIGHTS`.

## Stage 5 — zurand in graphs and per-row loss sums

**Issue.** [#7](https://github.com/pedrobtz/zulm/issues/7), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Every random number zulm draws comes from zurand's C table
under the session key, in and out of graphs, and every reported loss is
a fixed-order sum of per-row values, so that neither depends on the
thread count.

**Work.**
- `zl_seed()` holding an `rng_key` with the `philox4x64` engine; stream
  keys folded in R from names, step keys by `fold_int()` in C; the f32
  rounding rule (uniforms towards zero, normals to nearest).
- `zl_noise_normal()` and `zl_noise_uniform()` as a forward-only custom
  operation split across pool tasks by element; `zl_rng_uniform()` and
  `zl_rng_normal()` in R; `zulm_info()` reporting the key and zurand's
  table version; a load-time check for `ZURAND_API_VERSION >= 3`.
- The per-row loss operation and the pairwise fixed-order `double` sum
  on the main thread; `zl_step()` and `zl_forward()` report through it.

**Exit criteria.**
- In-graph draws equal `zurand::rng_uniform()` and `rng_normal()` on the
  folded keys, rounded by the stated rule, on every CI platform.
- Draws are identical across thread counts and when a batch is split in
  two with matching offsets; uniforms are never 0 or 1 in f32.
- Reported losses are bit-identical at one, two and four threads on a
  cross-entropy graph, where `ggml_cross_entropy_loss`'s own value is
  not.
- A zurand below table version 3 is refused at load with
  `zulm_companion_error`.

**Design.** §7.1, §7.2; D18, D19, D20; open question 9 (not blocking).

## Stage 6 — Modules, the driving model and PyTorch fixtures

**Issue.** [#8](https://github.com/pedrobtz/zulm/issues/8), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** The modules of §5.2 exist, the §1.1 model trains end to end on
synthetic data to the acceptance target, and PyTorch fixtures pin the
transformer block, AdaLN, the losses and the two §1.2 inference blocks.

**Work.**
- `zl_linear()`, `zl_embedding()`, `zl_norm()`, `zl_attention()`,
  `zl_mlp()`, `zl_block()`, `zl_adaln()`, `zl_time_embedding()`.
- `tools/make-torch-reference.py` writing GGUF fixtures (weights,
  inputs, outputs, gradients) under 1 MB for the block, AdaLN, the
  losses, a TimesFM-style attention block (split-half rotary, query and
  key normalisation, per-dimension scale, `scale = 1`, prefix mask) and
  a Toto-style block (interleaved partial rotation, xPos factors,
  `scale = 1 / d`, unit-scaled linears).
- The §1.1 model as a CI-only acceptance test: width 64, four heads, one
  block, four-class kind head, flow head over seven targets, 1 000
  AdamW steps on the synthetic marked process; Euler sampling of the
  head on a forward-only graph.

**Exit criteria.**
- Design §15 criterion 2: mean loss of steps 901–1 000 below 60 % of
  steps 1–100; parameters and reported losses bit-identical at one and
  two threads; Euler sampling returns finite values.
- Every fixture reproduces PyTorch's outputs and gradients within
  tolerance, loaded from GGUF with `zl_load()`.
- Each module matches a hand-written R version.

**Design.** §1.1, §1.2, §5.2, §11 (fixtures), §15 criteria 2, 3 and 9.

**Deviations (2026-10-14).** The synthetic marked process first gave the
model too little signal: the last 100 steps' mean loss was 67 % of the
first 100's. Its kind means were tripled and its noise cut to 0.1, which
gives 18 %; the criterion itself is unchanged. Constants and `repeat`
were added to graphs, which the modules need (each constant has a
buffer of its own). The acceptance test runs in an `acceptance`
workflow, which sets `ZULM_ACCEPTANCE=true`.

## Stage 7 — Thread pool, interrupts, fork and determinism

**Issue.** [#9](https://github.com/pedrobtz/zulm/issues/9), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** zulm owns one persistent thread pool, is interruptible between
graph nodes, survives a fork, and meets its determinism contract under
test.

**Work.**
- `zl_threads()` with the core cap and the `_R_CHECK_LIMIT_CORES_` cap;
  the persistent `ggml_threadpool`, recreated on resize; the polling
  measurement (0 against ggml's 50) recorded in design §6.7.
- The abort callback as interrupt check with `R_ToplevelExec()`;
  `zulm_interrupt`; the documented undefined optimiser state after an
  interrupted step.
- Fork safety: the pool abandoned and recreated in a child.
- The determinism suite: same seed twice; parameters and losses at one,
  two and four threads after 200 steps of the §1.1 model with and
  without `clip_norm`; forward outputs of the §1.2 blocks at one, two
  and four threads; each ISA variant forced on hardware that has it.

**Exit criteria.**
- An interrupt during a long graph returns to R with `zulm_interrupt`
  and the session is usable afterwards.
- A forked child trains with two threads on every POSIX leg.
- Every determinism test passes; the polling decision and the
  oversubscription measurement are in design §6.7.

**Design.** §6.5, §6.7; D11, D12.

**Deviations (2026-10-14).** zulm keeps a registry of its CPU backends and
detaches the pool from every one before freeing it, since
`ggml_backend_cpu_set_threadpool()` pauses the previous pool and freeing
it first crashed the next computation. The fork test uses `parallel`,
added to `Suggests`. The oversubscription measurement is the drafting
session's, already in §6.7.

## Stage 8 — safetensors and weight types

**Issue.** [#10](https://github.com/pedrobtz/zulm/issues/10), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Hub checkpoints load into a store without conversion, and f16
and bf16 weights can stay in memory with their precision cost measured
and documented.

**Work.**
- `zl_safetensors_read()`, `zl_load_safetensors()` with a name map,
  `zl_save_safetensors()`: length and header read in C, header parsed
  and validated through `zujson` in R, buffer read in C straight into
  the store; the checks of §7.4; `zulm_file_error`.
- `zl_param(type = "f16" | "bf16")` filled without widening by
  `zl_load()` and `zl_load_safetensors()`; `ggml_mul_mat` on such
  weights; `zulm_info()` and `zl_param_names()` reporting types.
- The precision test: f16 and bf16 against f32 within the tolerance
  §6.9 states, and failing if the difference falls below it, so the
  documentation stays true.
- `tools/make-torch-reference.py` writing a safetensors copy of each
  fixture; a fixture header from a real Hub checkpoint (Toto 2.0 4M, 48
  tensors, no data); a fuzz target for headers under the r-actions
  harness.

**Exit criteria.**
- Round trips at f32, f16 and bf16; files written by the Python
  `safetensors` package load with the documented layout rule
  (`[out, in]` becomes `c(in, out)` with no transposition).
- The Toto header parses and validates; malformed headers are
  `zulm_file_error`; the fuzz target runs clean.
- Design §15 criterion 11's f16 clause holds.

**Design.** §6.9 (weight types), §7.4; D13, D22, D35, D36.

**Deviations (2026-10-14).** The libFuzzer target for headers is Stage
13's, with the other fuzz targets under one harness; this stage has a
mutation test that corrupts a valid header byte by byte and expects a
`zulm_file_error` or a clean load every time. Two arguments were added:
`zl_load(keep_types = TRUE)` keeps a GGUF file's f16 and bf16 tensors in
their type (by default it still widens, as §6.1 says), and
`zl_param_names(p, types = TRUE)` reports types; `zulm_info()` reports
the build, not a store. Measured precision bands: f16 within 2e-4 to
5e-3 of the output's scale, bf16 within 5e-4 to 1e-2.

## Stage 9 — State tensors and the graph cache

**Issue.** [#11](https://github.com/pedrobtz/zulm/issues/11), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Graphs can read and write persistent state tensors, which give
a key-value cache for incremental decoding and a sliding window without
rebuilding graphs, and a decode loop whose shapes vary builds each shape
once.

**Work.**
- `zl_state()`, `zl_state_get()`, `zl_state_set()` with `offset`,
  `zl_state_write()` emitting `ggml_set_rows` with the node flagged as
  an output; state excluded from `zl_save()` and from training.
- The C side of `input_write_f32_at()` and `state_write_f32_at()`
  (registered in the table in Stage 12).
- `zl_graph_cache()` keyed by input shapes, least-recently-used eviction
  under `getOption("zulm.max_bytes")`.
- The TimesFM-style decode test: an encode graph for the context and a
  decode graph per block writing keys and values into fixed-capacity
  state under a mask, against one full-context forward pass.

**Exit criteria.**
- A decode of 256 values through state tensors equals the full-context
  forward pass within f32 tolerance, for several `n_new` and fill
  levels (design §15 criterion 11).
- Offset writes from R and C land where documented; out-of-range
  offsets are errors.
- `zl_graph_cache()` builds once per shape and evicts under the cap;
  graph construction for a twenty-block model is under 5 ms per shape.

**Design.** §6.9 (state tensors, fixed capacity, graph cache); D34;
open question 17.

**Deviations (2026-10-14).** Graph construction for a twenty-block model
(980 nodes) takes about 21 ms per shape on an Apple M2, not under 5 ms:
the time is R's, about 20 µs per operation across the closures that check
and wrap each one, against 2.5 µs in C. Building a single-tensor lookup
for `zl_param_tensor()` and a quiet call path for graph building brought
it down from 30 ms; going further would mean building modules in C. The
cache pays it once per shape, so a decode loop over ten block sizes
spends about 0.2 s building. The acceptance workflow keeps it under
50 ms. States are read inside `build()` through `zl_param_tensor()`,
which takes a state's name as well as a parameter's, so no
`zl_state_tensor()` is added; a graph that writes a state cannot be
trained (`zl_trainer()` refuses it). `zl_graph_cache()` treats an input
given as an R integer vector as an i32 input. `input_write_f32_at()`
and the state members exist in C and are tested through internal entry
points until Stage 12 registers them.

## Stage 10 — Flash attention

**Issue.** [#12](https://github.com/pedrobtz/zulm/issues/12), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** `zl_flash_attention()` runs upstream's forward-only flash
attention with the mask format it requires, agrees with the plain form,
and its thread behaviour is measured and recorded.

**Work.**
- `zl_flash_attention()` over `ggml_flash_attn_ext`: the layouts of
  §6.9, the mask converted to f16 and padded as the kernel requires;
  refused on a gradient path.
- Agreement tests against the plain form within f32 tolerance at
  several shapes, including a 10 240-position context on CI.
- The thread-count measurement; the result written into design §6.5 and
  §6.9, and the decision whether it is enabled by default or shipped
  disabled with the plain form as the reference (open question 16).

**Exit criteria.**
- Agreement with the plain form within f32 tolerance at every tested
  shape.
- Thread-count behaviour measured at one, two and four threads and
  recorded in the design, with the default decided.
- A 10 240-position forward pass runs within the memory cap.

**Design.** §6.9 (flash attention); open questions 8 and 16.

**Deviations (2026-10-14).** The kernel in v0.26.0 does not require the
mask to be padded, so none is; it does split a single query's keys
across threads, so `zl_flash_attention()` computes a single query as two
identical ones and is thread-invariant everywhere, which answers open
question 16 with "enabled by default". The 10 240-position test runs in
the acceptance workflow, under a 64 MB cap, rather than in every check
leg, since sanitizers and valgrind would take minutes over it.

## Stage 11 — Arrow batches

**Issue.** [#13](https://github.com/pedrobtz/zulm/issues/13), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Training batches come straight from Arrow streams (DuckDB,
Parquet, `arrow`) through `nanoarrow`, with one copy and the type, null
and chunk rules of design §7.3.

**Work.**
- The Arrow C data interface definitions vendored (Apache-2.0 notice in
  `inst/COPYRIGHTS`); `nanoarrow` in `Suggests`.
- `zl_arrow_batches()` and `zl_next()`: the type map, `na = "error"` and
  `"mask"`, chunk accumulation to exactly `rows`, `last = "drop"` and
  `"pad"`, dictionary codes; `zulm_arrow_error` with column and row;
  stream read and released on the main thread only.
- `batch_fill` in C (registered in Stage 12); `zl_step()` and
  `zl_forward()` accepting an iterator as `data`.
- Fixtures: nanoarrow streams, a DuckDB table and a Parquet file, with
  record batches smaller than, equal to and larger than `rows`.

**Exit criteria.**
- The §1.1 model trained from a DuckDB table through an Arrow stream
  gives results identical to training from the same rows as R arrays
  (design §15 criterion 12).
- Every supported type, both null policies, dictionaries that change
  between chunks, and the release callbacks called exactly once are
  tested.
- The suite passes with `nanoarrow` absent, skipping the Arrow tests
  and giving `zulm_companion_error` where documented.

**Design.** §7.3; D17, D21; open question 14.

**Deviations (2026-10-14).** The DuckDB and Parquet fixtures run as
`tools/acceptance/arrow-duckdb.R` in the acceptance workflow, not as
package tests: `duckdb` in `Suggests` would be built from source,
instrumented, in the sanitizer images. The Parquet file is written by
DuckDB at run time rather than committed. The package tests use
nanoarrow streams and a counting stream in C, which also runs without
nanoarrow. `map` marks integer inputs with `zl_int()` around column
names, which the design left open. A dictionary that changes between
record batches is an error; levels given up front are not offered.
Dictionary codes may be unsigned, as DuckDB's are. The R CMD check
caller now runs r-actions' `nosuggests` leg, and the acceptance workflow
a job without nanoarrow.

## Stage 12 — The C table and the zukzeit proof

**Issue.** [#14](https://github.com/pedrobtz/zulm/issues/14), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Other packages drive zulm from C through a versioned table of
opaque handles, and both consumers are shown to build on it: a fixture
consumer for zuflow's pattern, and zukzeit's Toto 2.0 4M running on
zulm with its parity fixtures green.

**Work.**
- `inst/include/zulm.h` with every member of design §8, registered as
  `zulm_api_v1`; `zulm_api_get()`; return codes and `message()`;
  `rng_stream()` and the name-based draw members; the state and offset
  members; `batch_fill`.
- A fixture consumer package under `tests/testthat/` that builds
  against the header: a sampler loop over `compute`,
  `input_write_f32`, `output_read_f32` and `rng_normal_f32`; weight
  averaging through `param_read_f32` and `param_write_f32`; a state
  write; `batch_fill`; `struct_size` and version checks; a standalone
  compile of the header as C99 and C++17.
- A branch of `pedrobtz/zukzeit` in which the Toto 2.0 4M port's tensor
  runtime is zulm, run in zukzeit's CI against zulm's `main`; the
  branch link recorded in design §15 and in `cran-comments.md` as a
  reverse-dependency check.

**Exit criteria.**
- The fixture consumer passes under r-actions on every leg (skipped on
  CRAN as the family's C-table tests are).
- zukzeit's committed parity fixtures for Toto 2.0 4M pass with zulm as
  the runtime, within zukzeit's documented tolerance, on CPU, on every
  CI leg (design §15 criterion 10).
- A consumer compiled against a shorter table version is refused with a
  clear message.

**Design.** §7.8, §7.9, §8; D8, D25, D33; the §15 criteria 7 and 10.

## Stage 13 — Fuzzing, benchmarks and documentation

**Issue.** [#15](https://github.com/pedrobtz/zulm/issues/15), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** The package is fuzzed, measured against its targets, and
documented for users and for the two consumers.

**Work.**
- Fuzz targets under the r-actions harness: graph construction with
  random operation sequences and shapes, the safetensors header (from
  Stage 8), the Arrow type map.
- `tools/run-benchmarks` measuring the §12 training targets (512³
  product, the §1.1 step at one and two threads, R overhead) and
  inference targets (the TimesFM block, Toto 2.0 4M on 4 096 values, a
  decode step, the native hook's row), recorded in design §12.
- pkgdown reference for every exported function; vignettes: training
  the §1.1 model from synthetic data, and running a pretrained block
  from safetensors with state tensors; `README.md` with the native
  build section as zufft's; `NEWS.md` under the development heading.
- `cran-comments.md` drafted: the r-actions legs as test environments,
  the `abort` symbol if it remains, the reverse-dependency check on the
  zukzeit branch.

**Exit criteria.**
- Every fuzz target runs clean under ASan and UBSan for the configured
  time.
- Every §12 target is met, or the miss is recorded in the design with
  the measurement.
- Every exported function has a reference page; both vignettes build on
  CI in under ten minutes; coverage is reported.

**Design.** §11, §12, §15 criteria 4 and 6.

## Stage 14 — v0.1.0 readiness

**Issue.** [#16](https://github.com/pedrobtz/zulm/issues/16), sub-issue of
[v0.1.0](https://github.com/pedrobtz/zulm/issues/1).

**Goal.** Only the submission itself is left.

**Work.**
- `Version: 0.1.0` and the matching `NEWS.md` heading.
- `Remotes:` removed: zurand and zujson are on CRAN (zurand's own
  submission precedes this stage).
- `cran-comments.md` final, listing the r-actions CI legs as test
  environments.
- The `pkg-cran-review` skill run, every finding resolved.
- `urlchecker::url_check()` and `spelling::spell_check_package()` clean.
- A full-profile r-actions run green on the release commit.

**Exit criteria.**
- All of the above hold on the release commit.
- Every design §15 criterion for v0.1.0 is met and referenced from a
  test or a CI job.

**Design.** §15.

## After 0.1.0

- **0.2:** the `zuexact` lookup for exactly summed losses (design §7.1);
  forward-only FFT through `zufft` and differentiable DFT spectra with
  patch P6 upstream (§7.5); `argsort` and `top_k` if a consumer asks.
  Neither consumer's first release needs them.
- Adopting zurand's `fold_bytes()` and f32 fills if zurand#63 and
  zurand#64 land, with a test that the bits are unchanged (§7.2).
- Saving and restoring optimiser state for exact resumption (§6.6).
- GPU backends as opt-in `configure` switches, Vulkan then Metal (§17
  question 5); until then zukzeit's `device` is the CPU.
- Quantised weights, mixed-precision training, flash attention in
  training graphs, custom operations through the table, special
  functions through `zuspecial` (§2, Later).

## Open questions

- **Stage 1** is the go/no-go of design §13.1: if the variant scheme or
  the clang route fails CRAN's checks, the fallback (`sse42` as the
  x86-64 baseline) changes §13.1 before Stage 2 begins. Open question 7
  (a fourth `avx512` variant; llamafile kernels or the unrolling pragma)
  is decided from Stage 1's measurements.
- **Stage 1** also decides whether `SystemRequirements: GNU make` is
  needed, and whether the native hook can be offered on Windows.
- **Stage 10** decides open question 16, whether flash attention is
  thread-invariant, and so whether it is on by default.
- **Stage 12** depends on zukzeit's maintainer accepting a branch that
  swaps Toto's runtime; the order of the other two ports is zukzeit's
  roadmap decision (design §18).
- **Stage 14** cannot close before zurand and zujson are on CRAN;
  zujson is, zurand's submission is prepared.
- Open question 9 (zurand#63, zurand#64) blocks nothing: Stage 5 ships
  with zulm's own folding and rounding.
