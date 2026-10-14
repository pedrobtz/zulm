# Selection, clipping and triangles

`zl_where(mask, a, b)` is `mask * a + (1 - mask) * b` for a mask of 0s
and 1s. `zl_clamp()` limits values to `[lo, hi]`. `zl_tri()` keeps one
triangle of each matrix, as
[`lower.tri()`](https://rdrr.io/r/base/lower.tri.html) and
[`upper.tri()`](https://rdrr.io/r/base/lower.tri.html) select it, and
sets the rest to zero. `zl_clamp()` and `zl_tri()` have no gradient.

## Usage

``` r
zl_where(mask, a, b)

zl_clamp(x, lo, hi)

zl_tri(x, type = c("lower", "upper", "lower_strict", "upper_strict"))
```

## Arguments

- mask, a, b, x:

  Tensors.

- lo, hi:

  Numbers.

- type:

  Which triangle: `"lower"` and `"upper"` include the diagonal, the
  `_strict` forms do not.

## Value

A tensor.

## Examples

``` r
p <- zl_params()
g <- zl_graph(
  function(x) zl_tri(zl_clamp(x$m, -1, 1), "lower"),
  p,
  inputs = list(m = c(3, 3))
)
zl_forward(g, list(m = matrix(-4:4, 3, 3)))$output
#>      [,1] [,2] [,3]
#> [1,]   -1    0    0
#> [2,]   -1    0    0
#> [3,]   -1    1    1
```
