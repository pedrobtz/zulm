# State tensors

A state tensor is a persistent f32 tensor in a parameter store that
graphs read like a parameter and write with `zl_state_write()`. It is
never trained and never saved:
[`zl_save()`](https://pedrobtz.github.io/zulm/reference/zl_save.md) and
[`zl_save_safetensors()`](https://pedrobtz.github.io/zulm/reference/zl_safetensors_read.md)
leave it out, and
[`zl_param_names()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
and
[`zl_param_count()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
do not count it. A key-value cache for incremental decoding is two state
tensors per layer, of a fixed capacity.

## Usage

``` r
zl_state(p, name, dim)

zl_state_get(p, name)

zl_state_set(p, name, value, offset = NULL)

zl_state_write(s, x, rows)
```

## Arguments

- p:

  A parameter store from
  [`zl_params()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
  or
  [`zl_load()`](https://pedrobtz.github.io/zulm/reference/zl_save.md).

- name:

  The state's name: a string of 1 to 63 bytes, unique in the store.

- dim:

  Its dimensions: one to four positive whole numbers.

- value:

  For `zl_state_set()`: without `offset`, an array of the state's
  dimensions; with `offset`, a numeric vector.

- offset:

  `NULL`, or the position, counted from 0, at which to start writing
  `value`.

- s:

  A state tensor inside `build()`, from
  [`zl_param_tensor()`](https://pedrobtz.github.io/zulm/reference/zl_params.md).

- x:

  A tensor whose first, third and fourth dimensions are the state's.

- rows:

  An integer input of one index per second-dimension slice of `x`.

## Value

`zl_state()` and `zl_state_set()` return the store invisibly;
`zl_state_get()` a double array; `zl_state_write()` the state tensor
after the write.

## Details

`zl_state()` declares a state of zeros. Inside `build()`, take a handle
on it with
[`zl_param_tensor()`](https://pedrobtz.github.io/zulm/reference/zl_params.md).
`zl_state_write(s, x, rows)` writes the second-dimension slices of `x`
into the slices of `s` that `rows` names, each time the graph runs.
`rows` must be an integer input declared with
[`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md),
counting from 0, and no row may appear twice in one run. The write
returns `s` after the write: read the state through it, so the read
follows the write. A graph that writes state cannot be trained.

`zl_state_set()` without `offset` replaces the whole state, as
[`zl_param_set()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
does, which resets it. With `offset`, it writes the values of `value` in
order from that position of the state's values, counted from 0 with the
first index fastest, as R stores an array. A write that would reach past
the state's end is an error. A sliding window shifts this way without
rebuilding a graph.

## Examples

``` r
p <- zl_params()
zl_state(p, "cache", c(2, 4))
g <- zl_graph(
  function(x) {
    s <- zl_state_write(zl_param_tensor(p, "cache"), x$new, x$rows)
    zl_sum(s)
  },
  p,
  inputs = list(new = c(2, 1), rows = zl_int(1))
)
zl_forward(g, list(new = matrix(1, 2, 1), rows = 0L))
#> $output
#> [1] 2
#> 
zl_forward(g, list(new = matrix(2, 2, 1), rows = 3L))
#> $output
#> [1] 6
#> 
zl_state_get(p, "cache")
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    0    0    2
#> [2,]    1    0    0    2
zl_state_set(p, "cache", c(9, 9), offset = 2)
zl_state_get(p, "cache")
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    9    0    2
#> [2,]    1    9    0    2
```
