# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## What this is

zulm builds computation graphs of tensors in R, runs them on the CPU,
differentiates them and trains their parameters, through upstream ggml
(the C tensor library behind llama.cpp), vendored in its CPU subset. It
is a substrate, not a framework: tensors, explicit graphs, ggml’s
autograd and optimisers, GGUF and safetensors for weights, a few
modules, and a versioned C table for packages that build models on top.
Its two consumers are `zuflow`, which trains and samples an
event-sequence generator, and `zukzeit`, which runs pretrained
time-series foundation models. Hard dependencies are the family’s
`zurand` (random numbers, `Imports` and `LinkingTo`) and `zujson`
(safetensors headers); nothing else. No GPU, no OpenMP, no system
libraries.

[.agents/design.md](https://pedrobtz.github.io/zulm/.agents/design.md)
is the specification and outranks this file.
[.agents/roadmap.md](https://pedrobtz.github.io/zulm/.agents/roadmap.md)
is the stage plan with exit criteria. `CLAUDE.md` orients; the design
decides.

## Current state

Verified 2026-10-14: Stage 0 of the roadmap (package identity). The
package is a skeleton with its real `DESCRIPTION`, an empty native
library that only registers itself, and CI. ggml is not vendored yet and
no R function exists. Stage 1 is a go/no-go on CPU dispatch for CRAN
binaries (design §13.1) and may change the design before any R surface
is written.

## Stage tracking

Progress toward the next version is tracked as GitHub sub-issues, so the
parent issue shows a progress bar such as “6 of 7”.

- **One parent issue per target version**, titled with the bare version:
  [v0.1.0 (#1)](https://github.com/pedrobtz/zulm/issues/1).
- **One sub-issue per roadmap stage**, titled as the roadmap titles it,
  for example `Stage 3 — Graphs, operations, forward and validation`,
  and linking to that section.
- **Every tracking issue carries the `stage` label**, so it can be told
  apart from bug and feature issues and left out of open-issue counts.
- **Close a stage by merging its pull request.** Put `Closes #<n>` in
  the body of the stage’s pull request. Never close a stage whose exit
  criteria are not met; record a deviation in the roadmap first.
- **The roadmap stays authoritative.** The issues record which stages
  are done, not what a stage requires. Adding, removing or renaming a
  stage means editing the roadmap and the sub-issues in the same change.
- Close the parent issue when CRAN accepts the version and it is tagged.
  Submission is tracked in [CRAN-submission
  (#17)](https://github.com/pedrobtz/zulm/issues/17).

## Versioning

The package develops at `0.0.0.9000` and targets `0.1.0` for its first
CRAN release. Set `Version: 0.1.0` and the matching `NEWS.md` heading
when the package enters `Pre-flight`. Bump to `0.1.0.9000` only after
CRAN accepts it.

Until the first release the R API may change freely. The C table in
`inst/include/zulm.h` (Stage 12) is different: `zuflow` and `zukzeit`
build against it, so it follows the append-only rules of design §8 from
the day it exists. zulm needs `zurand` on CRAN before it can be
submitted; until then `Remotes: pedrobtz/zurand` stays in `DESCRIPTION`.

## Commands

Run from the package root.

``` sh
Rscript -e 'devtools::load_all()'                # compile src/ and load
Rscript -e 'devtools::document()'                # roxygen -> NAMESPACE, man/
Rscript -e 'devtools::test()'                    # full testthat suite
Rscript -e 'devtools::test(filter = "<name>")'   # one test file
Rscript -e 'devtools::test(shuffle = TRUE)'      # order independence
Rscript -e 'devtools::check()'                   # R CMD check
air format .                                     # format R sources
```

`zurand` is not on CRAN yet: install it from GitHub
(`pak::pak("pedrobtz/zurand")`) before building.

## Architecture

Design §4 is the reference. In outline, as the stages build it:

- `R/` is the user surface, prefixed `zl_`; `zulm_info()` reports the
  build. Everything a user calls validates its arguments in R or in the
  C glue before ggml sees them.
- `src/` holds the C glue (prefix `zlm_`), a small C++ shim that turns
  ggml assertions into R errors (design §13.2), and `src/ggml/`, the
  vendored upstream tree, built once per ISA variant (design §13.1).
- `inst/include/zulm.h` is the public C table, `zulm_api_v1`.

## Invariants that are easy to break

- **No ggml assertion may reach `abort()`.** Arguments ggml asserts on
  are validated first; what remains is caught by the abort callback and
  the shim (design §13.2). A test that reaches an assertion is a bug.
- **Nothing on a ggml pool thread calls R.** Only the main thread
  touches R objects; errors cross as status codes.
- **Results must not depend on the thread count.** Reported losses are
  fixed-order sums of per-row values and random draws come from `zurand`
  under keys, never from a thread-local generator (design §6.5).

## Testing conventions

- **Self-sufficient.** Every test builds its own inputs inside the
  `test_that()` block. No file-scope objects.
- **Self-contained.** Global state goes through `withr::local_*()`,
  randomised input through
  [`withr::local_seed()`](https://withr.r-lib.org/reference/with_seed.html).
  Tests write only under
  [`tempdir()`](https://rdrr.io/r/base/tempfile.html).
- **Assert on condition classes, never message text.** Use
  `expect_error(..., class = "zulm_invalid_argument")` and the other
  classes of design §10.
- **Order independence.** `devtools::test(shuffle = TRUE)` is part of
  the definition of done.
- **Oracles before hand-written cases.** Base-R implementations of each
  operation, central finite differences for gradients, and PyTorch
  fixtures written by `tools/make-torch-reference.py` (design §11).
- **Keep the suite inside the CRAN time budget.** Long acceptance tests
  (the §1.1 model) run on CI only.

## Definition of done

`devtools::document()` and `devtools::check()` clean, meaning 0 errors,
0 warnings and 0 notes. The one allowed note is “New submission” before
the first release. `devtools::test(shuffle = TRUE)` green.
`gctorture(TRUE)` clean when C changed. CI green on every leg. A
user-facing change also needs a test, roxygen documentation and a
`NEWS.md` entry.

## Releasing to CRAN

- **CI is the pre-submission check.** The `pedrobtz/r-actions` R CMD
  check runs `--as-cran` on the CRAN-like runners and containers, and it
  replaces win-builder, the macOS builder and R-hub. `cran-comments.md`
  lists the CI legs as its test environments.
- **Entering `Pre-flight`.** The roadmap’s feature stages are done, the
  version is set as in Versioning, `NEWS.md` and `cran-comments.md` are
  written, and CI is green. Only submission work remains.
- **The pretest is automated and does not read `cran-comments.md`.** Fix
  a NOTE instead of justifying it.
- **After acceptance,** tag `v0.1.0`, publish the GitHub release, bump
  to `0.1.0.9000`, and close the version’s parent issue.

## Editing rules

- roxygen comments are the source. Never edit `man/` or `NAMESPACE` by
  hand.
- Keep prose simple and short. One idea per sentence. en-GB spelling.
- Wrap roxygen text at 80 characters and run `air format .` on R
  sources.
- Use `lower_snake_case`; R functions take the `zl_` prefix, internal
  native symbols `zlm_`, the public C table `zulm_`.
- No hard runtime dependency beyond `zurand` and `zujson` without a
  recorded design decision.
- Export and document public functions. Do not create roxygen topics for
  internal ones.

## Continuous integration

Workflows come from `pedrobtz/r-actions`, pinned to `@v1`:
`r-cmd-check.yml` (quick on pull requests, full on `main`, full on a
pull request with the `full-ci` label) and `coverage.yml`, plus the
pkgdown workflow. Stage 1 adds `build-flags` and `portability`, and the
native checks once ggml is vendored.

Small edits get a small run. The shared workflows look at the files a
pull request changes, compared with its merge base:

- **Markdown only**, such as `README.md`, `NEWS.md` or
  `cran-comments.md`, skips every check. Only coverage and the pkgdown
  site run.
- **`DESCRIPTION` or `README.Rmd` as well**, and nothing else, adds one
  R CMD check on Ubuntu R-release.
- **Any other file** runs everything.

## Vendored native code

ggml’s CPU subset (MIT), pinned at tag `v0.26.0`, arrives in Stage 1
under `src/ggml/` through `tools/vendor-ggml` (design §9).

- Never edit a file under `src/ggml/`. Changes go in `tools/patches/`,
  each with its reason and its upstream pull request (design §9.3).
- **CRAN’s compiled-code check applies to vendored code too.** No
  `stdout` or `stderr`, no `printf`, `abort` or `exit` may be linked;
  upstream’s stdio is redirected to the log buffer and its diagnostic
  pragmas are stripped (design §9.2).
- **No `-f`, `-m` or `-O` flag in `Makevars`.** ISA targets are chosen
  by pragmas in a force-included header (design §13.1, §13.4).

## Commits and pull requests

Short, imperative, sentence-case commit subjects. Keep each commit
focused. A pull request explains the user-visible outcome and the
rationale, links related issues, lists the checks that were run and the
tests that were skipped, and flags platform-sensitive or vendored
changes. Performance claims need evidence.
