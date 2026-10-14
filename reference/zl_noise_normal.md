# Random draws inside a graph

`zl_noise_normal()` and `zl_noise_uniform()` are tensors of draws from
the stream named `stream`, made anew at each training step: at step `s`
of a trainer (1 for the first) element `i` (in R's order, counting from
0) is position `offset + i` of the stream at step `s`. Outside training,
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
uses the trainer's last step, or 0. The values do not depend on the
thread count, and equal
[`zl_rng_normal()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md)
and
[`zl_rng_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md)
for the same stream, step and positions. Use `offset` to keep the draws
of a split batch the same: a second half of a batch starts at the first
half's size.

## Usage

``` r
zl_noise_normal(dim, stream, offset = 0)

zl_noise_uniform(dim, stream, offset = 0)
```

## Arguments

- dim:

  Dimensions: one to four positive whole numbers.

- stream:

  The stream's name.

- offset:

  The position of the first value.

## Value

A tensor.

## Details

They have no gradient, and none is needed: nothing flows back into a
draw.

## Examples

``` r
p <- zl_params()
g <- zl_graph(function(x) x$a + zl_noise_normal(c(2, 3), "eps"), p,
              inputs = list(a = c(2, 3)))
zl_forward(g, list(a = matrix(0, 2, 3)))$output
#>            [,1]        [,2]       [,3]
#> [1,] -1.0074857 -0.15439574 -0.6295075
#> [2,] -0.2536615  0.09861942  0.7901454
```
