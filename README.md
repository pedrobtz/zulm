# zulm

<!-- badges: start -->
[![R-CMD-check](https://github.com/pedrobtz/zulm/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/pedrobtz/zulm/actions/workflows/R-CMD-check.yaml)
[![coverage](https://raw.githubusercontent.com/pedrobtz/zulm/gh-pages/badges/coverage.svg)](https://github.com/pedrobtz/zulm/actions/workflows/coverage.yaml)
<!-- badges: end -->

zulm builds computation graphs of tensors in R, runs them on the CPU,
differentiates them and trains their parameters, through upstream
[ggml](https://github.com/ggml-org/ggml), the C tensor library behind
llama.cpp and whisper.cpp.

> Upstream ggml, unmodified and current, as a small, safe, reproducible
> R substrate for training and running models on the CPU.

Two use cases size the package:

- **Training.** Small and medium neural networks trained in R on the
  CPU with ggml's autograd and its AdamW and SGD optimisers, with
  results that reproduce bit for bit at a fixed seed. The first
  consumer is [zuflow](https://github.com/pedrobtz/zuflow), which
  trains and samples an event-sequence generator.
- **Inference.** Pretrained transformer checkpoints of up to a few
  hundred million parameters, loaded from GGUF or safetensors and run
  without Python, LibTorch or a GPU toolchain. The first consumer is
  [zukzeit](https://github.com/pedrobtz/zukzeit), which runs
  time-series foundation models.

zulm is under development toward its first CRAN release. Nothing of
the R interface is built yet; the
[roadmap](https://github.com/pedrobtz/zulm/issues/1) tracks progress.

## Installation

You can install the development version of zulm from
[GitHub](https://github.com/pedrobtz/zulm) with:

``` r
# install.packages("pak")
pak::pak("pedrobtz/zulm")
```
