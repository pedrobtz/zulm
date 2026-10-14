# Arithmetic, reductions and elementwise functions on tensors

Inside `build()`, `+`, `-`, `*` and `/` combine tensors elementwise with
broadcasting: one operand's dimensions must each divide the other's, as
in 'ggml'. A number on either side is a scale or an added constant.
`zl_sum()` sums every element to a scalar; `zl_sum_rows()` and
`zl_mean_rows()` reduce the first dimension to 1; `zl_cumsum()` takes
cumulative sums along it. The others apply elementwise.

## Usage

``` r
zl_scale(x, s)

zl_sum(x)

zl_sum_rows(x)

zl_mean_rows(x)

zl_cumsum(x)

zl_sqr(x)

zl_sqrt(x)

zl_exp(x)

zl_log(x)

zl_abs(x)

zl_sin(x)

zl_cos(x)

zl_relu(x)

zl_silu(x)

zl_softplus(x)

zl_sigmoid(x)

zl_tanh(x)

zl_gelu(x, approximate = c("none", "tanh"))

zl_swiglu(gate, up)
```

## Arguments

- x:

  A tensor.

- s:

  A number.

- approximate:

  For `zl_gelu()`: `"none"` for the exact form with the error function
  (accurate to single precision), `"tanh"` for the tanh approximation,
  which 'ggml' evaluates through a half-precision table (accurate to
  about 1e-3).

- gate, up:

  Tensors of the same shape.

## Value

A tensor.

## Details

`zl_cumsum()`, `zl_gelu()`, `zl_sigmoid()` and `zl_tanh()` have no
gradient: they serve inference and sampling graphs, and training refuses
them on a gradient path.

## Examples

``` r
p <- zl_params()
g <- zl_graph(
  function(x) list(
    y = zl_sum_rows(zl_sqr(x$a - 1)),
    z = 2 * zl_exp(x$a) + 1
  ),
  p,
  inputs = list(a = c(3, 2))
)
zl_forward(g, list(a = matrix(1:6, 3, 2)))
#> $y
#>      [,1] [,2]
#> [1,]    5   50
#> 
#> $z
#>           [,1]     [,2]
#> [1,]  6.436563 110.1963
#> [2,] 15.778112 297.8263
#> [3,] 41.171074 807.8576
#> 
```
