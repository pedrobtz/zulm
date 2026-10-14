# Random draws, in R

`zl_rng_uniform()` and `zl_rng_normal()` return the single-precision
values that
[`zl_noise_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_noise_normal.md)
and
[`zl_noise_normal()`](https://pedrobtz.github.io/zulm/reference/zl_noise_normal.md)
give inside a graph for the same stream, step and positions: positions
`offset` to `offset + prod(dim) - 1` of the stream named `stream` at
step `step`. Uniforms lie strictly inside (0, 1); normals have mean 0
and standard deviation 1.

## Usage

``` r
zl_rng_uniform(dim, stream, step = 0, offset = 0)

zl_rng_normal(dim, stream, step = 0, offset = 0)
```

## Arguments

- dim:

  Dimensions: one to four positive whole numbers.

- stream:

  The stream's name.

- step:

  The step number (a trainer's step count; 0 for none).

- offset:

  The position of the first value.

## Value

A double vector or array of dimensions `dim`.

## Details

The values are 'zurand”s, rounded to single precision: a uniform is
rounded towards zero (so it never becomes 1), a normal to nearest. With
the session key `k`, a uniform equals
`zurand::rng_uniform( zurand::rng_fold(zurand::rng_fold(k, stream), step), n, offset = offset)`
so rounded.

## Examples

``` r
zl_seed(1)
zl_rng_uniform(c(2, 3), "noise")
#>           [,1]      [,2]       [,3]
#> [1,] 0.4639272 0.4965113 0.04710322
#> [2,] 0.4593310 0.3300929 0.53220999
zl_rng_normal(4, "noise", step = 2)
#> [1]  1.8157096 -1.6734138 -1.5309385  0.7882175
```
