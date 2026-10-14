# Attention pieces and normalisation

`zl_softmax()` normalises each column (the first dimension) after
multiplying by `scale` and adding `mask`, an additive f32 tensor whose
first two dimensions are `x`'s and whose third and fourth are 1 or
`x`'s, so one mask serves every head. `zl_causal_mask()` sets to `-Inf`
every element of each matrix whose row index exceeds its column index,
so that for scores of dims `c(n_kv, n_q, ...)` a query sees only keys at
or before it.

## Usage

``` r
zl_softmax(x, mask = NULL, scale = 1)

zl_causal_mask(x)

zl_rope(x, pos, n_dims, base = 10000, mode = c("interleaved", "half"))

zl_rms_norm(x, eps = 1e-05)

zl_layer_norm(x, eps = 1e-05)
```

## Arguments

- x, mask:

  Tensors.

- scale:

  A number.

- pos:

  An integer input of positions.

- n_dims:

  The number of leading dimensions to rotate: even, at most `head_dim`.

- base:

  The frequency base.

- mode:

  `"interleaved"` or `"half"`.

- eps:

  Added to the mean square, a positive number.

## Value

A tensor.

## Details

`zl_rope()` applies rotary positions to `x` of dims
`c(head_dim, heads, n, ...)` with integer positions `pos` of length `n`
(an input declared with
[`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)).
`mode = "interleaved"` rotates adjacent pairs ('ggml”s default and
Toto's convention); `mode = "half"` rotates the first half against the
second (TimesFM, Chronos-2, most Hugging Face checkpoints). `n_dims`
below `head_dim` rotates the leading `n_dims` only.

`zl_rms_norm()` divides each column by its root mean square;
`zl_layer_norm()` subtracts the column mean first.

## Examples

``` r
p <- zl_params()
g <- zl_graph(
  function(x) zl_softmax(zl_causal_mask(x$s)),
  p,
  inputs = list(s = c(3, 3))
)
zl_forward(g, list(s = matrix(0, 3, 3)))$output
#>      [,1] [,2]      [,3]
#> [1,]    1  0.5 0.3333333
#> [2,]    0  0.5 0.3333333
#> [3,]    0  0.0 0.3333333
```
