# Graphs built once per input shape

`zl_graph_cache()` wraps
[`zl_graph()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
for loops whose input shapes vary, such as a decoder whose count of new
positions changes. Its graphs are built by
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
the first time it sees a set of input shapes, and reused after. An input
given as an R integer vector or array becomes an integer input (see
[`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md));
any other numeric input is f32.

## Usage

``` r
zl_graph_cache(build, p, loss = NULL, outputs = NULL)

# S3 method for class 'zl_graph_cache'
print(x, ...)
```

## Arguments

- build:

  A function of one argument, the named list of inputs, returning a
  tensor or a named list of tensors.

- p:

  The parameter store whose parameters `build()` uses.

- loss:

  The name of a scalar output to train on (see the training functions);
  `NULL` for none.

- outputs:

  The names of the outputs to keep, of those `build()` returns; `NULL`
  for all.

- x:

  A graph cache.

- ...:

  Unused.

## Value

A graph cache, of class `zl_graph_cache`, for
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md).
[`length()`](https://rdrr.io/r/base/length.html) gives the number of
graphs it holds.

## Details

When the store's and the cached graphs' memory together would pass
`getOption("zulm.max_bytes")`, the graphs used least recently are freed
first; the graph just built is kept.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(3, 2))
gc <- zl_graph_cache(function(x) zl_crossprod(zl_param_tensor(p, "w"), x$input), p)
dim(zl_forward(gc, list(input = matrix(1, 3, 5)))$output)
#> [1] 2 5
dim(zl_forward(gc, list(input = matrix(1, 3, 7)))$output)
#> [1] 2 7
length(gc)
#> [1] 2
```
