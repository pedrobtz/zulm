# zulm

zulm builds computation graphs of tensors in R, runs them on the CPU,
differentiates them and trains their parameters, through upstream
[ggml](https://github.com/ggml-org/ggml), the C tensor library behind
llama.cpp and whisper.cpp.

> Upstream ggml, unmodified and current, as a small, safe, reproducible
> R substrate for training and running models on the CPU.

Two use cases size the package:

- **Training.** Small and medium neural networks trained in R on the CPU
  with ggml’s autograd and its AdamW and SGD optimisers, with results
  that reproduce bit for bit at a fixed seed. The first consumer is
  zuflow, which trains and samples an event-sequence generator.
- **Inference.** Pretrained transformer checkpoints of up to a few
  hundred million parameters, loaded from GGUF or safetensors and run
  without Python, LibTorch or a GPU toolchain. The first consumer is
  [zukzeit](https://github.com/pedrobtz/zukzeit), which runs time-series
  foundation models.

zulm is under development toward its first CRAN release; the
[roadmap](https://github.com/pedrobtz/zulm/issues/1) tracks progress. So
far it bundles ggml’s CPU subset and loads the build that suits the CPU;
[`zulm_info()`](https://pedrobtz.github.io/zulm/reference/zulm_info.md)
reports it.

## A build for your own CPU

The package ships several builds of its compiled code, one per
instruction-set level, and picks the best one the CPU supports. On Linux
and macOS, a source install can instead build one variant for the
installing machine:

``` r

Sys.setenv(ZULM_CFLAGS = "-march=native", ZULM_CXXFLAGS = "-march=native")
install.packages("zulm", type = "source")
```

It is only worth it on x86-64 CPUs with AVX-512; on arm64 and on AVX2
machines the shipped builds are as fast. Results reproduce within a
build, not between a native build and the shipped ones.

## Installation

You can install the development version of zulm from
[GitHub](https://github.com/pedrobtz/zulm) with:

``` r

# install.packages("pak")
pak::pak("pedrobtz/zulm")
```
