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
[roadmap](https://github.com/pedrobtz/zulm/issues/1) tracks progress.

## Usage

A graph is built once by an R function of its inputs, then runs any
number of times. Parameters live in a store; modules declare theirs and
return functions to apply inside a graph:

``` r

library(zulm)
zl_seed(1)
p <- zl_params()
lin <- zl_linear(p, "lin", n_in = 3, n_out = 1)
g <- zl_graph(
  function(x) list(loss = zl_mse(lin(x$x), x$y)),
  p,
  inputs = list(x = c(3, 32), y = c(1, 32)),
  loss = "loss"
)
tr <- zl_trainer(g, zl_adamw(lr = 0.05))
x <- matrix(rnorm(96), 3, 32)
y <- matrix(c(1, -2, 0.5) %*% x + 1, 1, 32)
for (i in 1:200) loss <- zl_step(tr, list(x = x, y = y))
zl_param_get(p, "lin.weight")
```

Pretrained weights load from GGUF with
[`zl_load()`](https://pedrobtz.github.io/zulm/reference/zl_save.md) or
from safetensors with
[`zl_load_safetensors()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md);
a PyTorch `[out, in]` weight is zulm’s `c(in, out)` with the same bytes.
State tensors
([`zl_state()`](https://pedrobtz.github.io/zulm/reference/zl_state.md))
hold a key-value cache across the runs of a decoding graph. Training
batches can come straight from Arrow streams, such as a DuckDB query or
a Parquet file, through
[`zl_arrow_batches()`](https://pedrobtz.github.io/zulm/reference/zl_arrow_batches.md).

The vignettes go further:
[`vignette("training", package = "zulm")`](https://pedrobtz.github.io/zulm/articles/training.md)
trains a small event-sequence model and samples from it, and
[`vignette("pretrained", package = "zulm")`](https://pedrobtz.github.io/zulm/articles/pretrained.md)
runs an attention block exported by PyTorch and decodes through a cache.

Every result reproduces bit for bit at a fixed seed, whatever the thread
count: noise inside graphs comes from ‘zurand’ under named keys, and
reported losses are fixed-order sums.

## For package authors

Packages that build models on zulm can drive its graphs from C through a
versioned table of function pointers, declared in `zulm.h`:

``` c
#include <zulm.h>   /* LinkingTo: zulm; Imports: zulm */

const zulm_api *zl = zulm_api_get();
zulm_graph *g = zl->graph_from_sexp(graph);
zl->input_write_f32(g, "x", buf, n);
if (zl->compute(g) != ZULM_OK) Rf_error("%s", zl->message());
zl->output_read_f32(g, "output", out, m);
```

zulm does not export ggml’s headers, which change between upstream
releases; the table only grows, and its `struct_size` tells a consumer
which members exist.

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
machines the shipped builds are as fast. On an AMD EPYC 7763 (AVX2), a
native build ran a small model’s training step in 35.0 ms against the
shipped build’s 36.7 ms, and a 512 × 512 product at 62 against 72
GFLOP/s. Results reproduce within a build, not between a native build
and the shipped ones.

## Installation

You can install the development version of zulm from
[GitHub](https://github.com/pedrobtz/zulm) with:

``` r

# install.packages("pak")
pak::pak("pedrobtz/zulm")
```
