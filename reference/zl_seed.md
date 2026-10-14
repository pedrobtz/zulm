# The session's random-number key

Every random number zulm draws (parameter initialisation, noise inside
graphs,
[`zl_rng_uniform()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md)
and
[`zl_rng_normal()`](https://pedrobtz.github.io/zulm/reference/zl_rng_uniform.md))
comes from 'zurand' under a key derived from the session key and a name.
`zl_seed()` sets the session key from a whole number, or from a key made
with [`zurand::rng_key()`](https://rdrr.io/pkg/zurand/man/rng_key.html)
or [`zurand::rng_fold()`](https://rdrr.io/pkg/zurand/man/rng_fold.html);
with no argument it returns the current key. Until it is set, the key is
drawn once from R's generator, so
[`set.seed()`](https://rdrr.io/r/base/Random.html) before the first draw
governs a session.

## Usage

``` r
zl_seed(seed = NULL)
```

## Arguments

- seed:

  `NULL` to get the key; a whole number, or a zurand `rng_key` of length
  one, to set it.

## Value

The session key; when setting, the previous key, invisibly.

## Examples

``` r
old <- zl_seed(42)
zl_rng_normal(3, "example")
#> [1] 0.3497453 0.3490455 0.2920494
zl_seed(old)
```
