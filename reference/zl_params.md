# Parameter stores

A parameter store holds named tensors of up to four dimensions. Its
memory belongs to the 'ggml' library, outside R's heap, and is freed
when the store is garbage collected. Stores cannot be saved with
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html): a restored store is
invalid (`zulm_stale_error`); use
[`zl_save()`](https://pedrobtz.github.io/zulm/reference/zl_save.md) and
[`zl_load()`](https://pedrobtz.github.io/zulm/reference/zl_save.md).

## Usage

``` r
zl_params()

zl_param(
  p,
  name,
  dim,
  init = zl_init_normal(sd = 0.02),
  type = c("f32", "f16", "bf16")
)

zl_param_get(p, name)

zl_param_set(p, name, value)

zl_param_names(p)

zl_param_count(p)

zl_param_tensor(p, name)
```

## Arguments

- p:

  A parameter store from `zl_params()` or
  [`zl_load()`](https://pedrobtz.github.io/zulm/reference/zl_save.md).

- name:

  The parameter's name: a string of 1 to 63 bytes, unique in the store.

- dim:

  Its dimensions: one to four positive whole numbers.

- init:

  How to fill it: see
  [`zl_init_normal()`](https://pedrobtz.github.io/zulm/reference/zl_init_normal.md).

- type:

  The storage type: `"f32"`, `"f16"` or `"bf16"`.

- value:

  An array of `dim(value)` equal to the parameter's dimensions, or a
  vector of the right length for a one-dimensional parameter. `NA` is
  refused.

## Value

`zl_params()` returns an empty store. `zl_param()` and `zl_param_set()`
return the store invisibly. `zl_param_get()` returns a double vector or
array; `zl_param_names()` the names in the order added;
`zl_param_count()` the total number of values; `zl_param_tensor()` a
handle for use when building a graph.

## Details

`zl_param()` adds a parameter. Its values are stored as `type`: `"f32"`
(single precision, the default), or `"f16"` and `"bf16"` (half
precision, for weights that are only read). R doubles are rounded to the
nearest representable value; values beyond the type's range become
infinite with a `zulm_range_warning`.

An R array of `dim = c(d1, d2, d3, d4)` is stored with the first index
fastest, as R stores it, so values move without reordering.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(3, 2))
zl_param(p, "b", 2, init = zl_init_zeros())
zl_param_names(p)
#> [1] "w" "b"
zl_param_count(p)
#> [1] 8
zl_param_set(p, "b", c(0.5, -0.5))
zl_param_get(p, "b")
#> [1]  0.5 -0.5
```
