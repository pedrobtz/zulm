# Convert a tensor's type

`zl_cast()` converts to `"f32"`, `"f16"` or `"bf16"`. Operations take
f32 tensors, so a half-precision parameter is cast to f32 before any
operation but
[`zl_crossprod()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md)
and
[`zl_get_rows()`](https://pedrobtz.github.io/zulm/reference/zl-shape.md).
It has no gradient.

## Usage

``` r
zl_cast(x, type)
```

## Arguments

- x:

  A tensor.

- type:

  The new type.

## Value

A tensor.

## Examples

``` r
p <- zl_params()
zl_param(p, "h", 4, type = "f16", init = zl_init_ones())
g <- zl_graph(function(x) zl_cast(zl_param_tensor(p, "h"), "f32") + x$a,
              p, inputs = list(a = 4))
zl_forward(g, list(a = 1:4))$output
#> [1] 2 3 4 5
```
