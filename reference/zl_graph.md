# Build and run graphs

`zl_graph()` calls `build(x)` once, with `x` a named list of input
tensors of the declared shapes, and records the tensors `build()`
returns as the graph's outputs. Inside `build()`, inputs, parameters
(from
[`zl_param_tensor()`](https://pedrobtz.github.io/zulm/reference/zl_params.md))
and the results of operations such as
[`zl_crossprod()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md)
are tensors; `build()` is not called again, and a different input shape
needs another graph.

## Usage

``` r
zl_graph(build, p, inputs, loss = NULL, outputs = NULL)

zl_int(...)

zl_forward(g, data, outputs = NULL)

zl_graph_info(g)
```

## Arguments

- build:

  A function of one argument, the named list of inputs, returning a
  tensor or a named list of tensors.

- p:

  The parameter store whose parameters `build()` uses.

- inputs:

  A named list of input dimensions: numeric vectors of one to four
  positive whole numbers, or `zl_int(...)` for an integer input.

- loss:

  The name of a scalar output to train on (see the training functions);
  `NULL` for none.

- outputs:

  The names of the outputs to keep, of those `build()` returns; `NULL`
  for all.

- ...:

  Dimensions, for `zl_int()`.

- g:

  A graph.

- data:

  A named list of arrays, one per input, each matching its input's
  dimensions exactly (a plain vector for a one-dimensional input). `NA`
  is refused.

## Value

`zl_graph()` returns a graph of class `zl_graph`; `zl_int()` an input
specification; `zl_forward()` a named list of double arrays;
`zl_graph_info()` a list describing the graph.

## Details

`zl_forward()` writes `data` into the inputs, runs the graph and returns
the outputs as double arrays. The inputs are rewritten in place on every
call, so a graph runs any number of times without being rebuilt.

Inputs are single-precision (f32) unless declared with `zl_int()`, which
makes them 32-bit integers (i32) for
[`zl_get_rows()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md)
and
[`zl_rope()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md).
Integer indices count from 0, as in 'ggml'.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(3, 2))
g <- zl_graph(
  function(x) zl_relu(zl_crossprod(zl_param_tensor(p, "w"), x$input)),
  p,
  inputs = list(input = c(3, 5))
)
out <- zl_forward(g, list(input = matrix(1, 3, 5)))
dim(out$output)
#> [1] 2 5
```
