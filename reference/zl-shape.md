# Products and shapes of tensors

`zl_crossprod(a, b)` is `crossprod(a, b)`: for `a` of dims `c(K, M)` and
`b` of `c(K, N)` it has dims `c(M, N)`, batched over the third and
fourth dimensions (where `a`'s may divide `b`'s). A linear layer stores
its weight as `c(n_in, n_out)` and computes `zl_crossprod(w, x)`; the
weight may be stored as f16 or bf16.

## Usage

``` r
zl_crossprod(a, b)

zl_reshape(x, dim)

zl_permute(x, perm)

zl_t(x)

zl_cont(x)

zl_view(x, dim, offset = 0)

zl_concat(a, b, axis)

zl_pad(x, after)

zl_get_rows(x, idx)
```

## Arguments

- a, b, x:

  Tensors.

- dim:

  New dimensions: one to four positive whole numbers.

- perm:

  A permutation of `seq_len(k)`, with `k` at least the rank of `x` and
  at most 4.

- offset:

  The first element of the view, counting from 0.

- axis:

  The dimension to join along, from 1 to 4.

- after:

  Numbers of zeros to append after each dimension.

- idx:

  An integer input (declared with
  [`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md))
  of column indices.

## Value

A tensor.

## Details

`zl_reshape()` gives new dimensions with the same number of elements;
`zl_permute()` reorders dimensions as
[`aperm()`](https://rdrr.io/r/base/aperm.html) does; `zl_t()` transposes
the first two; `zl_cont()` makes a permuted tensor contiguous
(operations do it themselves when they need to); `zl_view()` is a window
of `prod(dim)` elements from position `offset` (counting from 0) of a
contiguous tensor. `zl_concat()` joins two tensors along `axis`,
`zl_pad()` appends zeros after each dimension, and `zl_get_rows()`
selects columns of a matrix (rows in 'ggml”s terms) by an integer input,
counting from 0. `zl_concat()` and `zl_pad()` have no gradient.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(4, 2))
g <- zl_graph(
  function(x) zl_t(zl_crossprod(zl_param_tensor(p, "w"), x$x)),
  p,
  inputs = list(x = c(4, 3))
)
dim(zl_forward(g, list(x = matrix(1, 4, 3)))$output)
#> [1] 3 2
```
