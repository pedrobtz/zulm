# Flash attention

`zl_flash_attention()` computes `softmax(scale * t(k) %*% q + mask)`
weighted sums of `v` for every query, head and batch, in one fused
operation that holds no matrix of scores, so long contexts attend in
memory linear in their length. It is 'ggml”s `ggml_flash_attn_ext`.

## Usage

``` r
zl_flash_attention(q, k, v, mask = NULL, scale = 1)
```

## Arguments

- q, k, v:

  Tensors.

- mask:

  `NULL`, or a tensor added to the scores.

- scale:

  The factor applied to the scores before the mask, usually
  `1 / sqrt(d)`.

## Value

A tensor.

## Details

The layouts are 'ggml”s: `q` is `c(d, n, heads, batch)`, `k` is
`c(d, n_kv, heads_kv, batch)` and `v` is
`c(d_v, n_kv, heads_kv, batch)`, not transposed; `heads_kv` must divide
`heads`. The result is `c(d_v, heads, n, batch)`, heads before
positions, so that `zl_reshape(o, c(d_v * heads, n, batch))` joins the
heads. Trailing dimensions of 1 may be left out of the inputs; the
result has at least three.

The mask is added to the scaled scores. It is `c(n_kv, n)`, optionally
with third and fourth dimensions that divide `heads` and `batch`. The
kernel reads it in half precision: 0 and `-Inf` are exact, other values
are rounded to about three significant digits.

The operation has no gradient, so a graph that trains through it is
refused;
[`zl_softmax()`](https://pedrobtz.github.io/zulm/reference/zl-attention.md)
and
[`zl_crossprod()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md)
give the same attention with gradients. Results agree with that plain
form to within f32 rounding, and do not depend on
[`zl_threads()`](https://pedrobtz.github.io/zulm/reference/zl_threads.md):
a single query, which 'ggml' would split across threads by key, is
computed as two identical queries so that each stays on one thread.

## Examples

``` r
p <- zl_params()
g <- zl_graph(
  function(x) zl_flash_attention(x$q, x$k, x$v, scale = 1 / 2),
  p,
  inputs = list(q = c(4, 3, 2), k = c(4, 5, 2), v = c(4, 5, 2))
)
dim(zl_forward(g, list(
  q = array(1, c(4, 3, 2)), k = array(1, c(4, 5, 2)), v = array(1, c(4, 5, 2))
))$output)
#> [1] 4 2 3
```
