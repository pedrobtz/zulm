# Initialisers for parameters

Each returns a description of how
[`zl_param()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
fills a new parameter. Random draws come from 'zurand' under a key
derived from the session key and the parameter's name, so they do not
depend on the order in which parameters are declared, and adding a
parameter does not change the values of the others.

## Usage

``` r
zl_init_normal(sd = 0.02)

zl_init_uniform(a)

zl_init_zeros()

zl_init_ones()
```

## Arguments

- sd:

  Standard deviation of a normal distribution with mean zero.

- a:

  Half-width of a uniform distribution on `(-a, a)`.

## Value

An object of class `zl_init`.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(4, 3), init = zl_init_normal(sd = 0.1))
zl_param(p, "b", 3, init = zl_init_zeros())
zl_param_get(p, "b")
#> [1] 0 0 0
```
