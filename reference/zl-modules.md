# Modules

Each constructor declares its parameters in the store `p`, under names
that start with `name`, and returns a function of tensors to use inside
`build()` (see
[`zl_graph()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)).
Tensors have the feature dimension first: a batch of sequences is
`c(dim, n, batch)`.

## Usage

``` r
zl_linear(p, name, n_in, n_out, bias = TRUE)

zl_embedding(p, name, n, dim)

zl_norm(p, name, dim, type = c("layer", "rms"), affine = TRUE, eps = 1e-05)

zl_attention(p, name, dim, heads = 1L, causal = TRUE)

zl_mlp(p, name, dim, hidden = 4 * dim, act = c("swiglu", "silu"))

zl_block(p, name, dim, heads, causal = TRUE, hidden = 4 * dim)

zl_adaln(p, name, dim, cond_dim)

zl_time_embedding(t, dim, max_period = 10000)
```

## Arguments

- p:

  A parameter store.

- name:

  The prefix of the parameters' names.

- n_in, n_out:

  Input and output widths.

- bias:

  Whether to add a bias.

- n:

  The number of rows of an embedding table.

- dim:

  The feature width.

- type:

  `"layer"` or `"rms"`.

- affine:

  Whether a normalisation has a learnt scale (and shift).

- eps:

  Added to the variance, a positive number.

- heads:

  The number of attention heads; must divide `dim`.

- causal:

  Whether position `i` attends only to positions up to `i`.

- hidden:

  The MLP's inner width.

- act:

  `"swiglu"` or `"silu"`.

- cond_dim:

  The width of the conditioning tensor.

- t:

  A tensor of times, of dims `c(1, ...)`.

- max_period:

  The longest period of the embedding.

## Value

A function of tensors, except `zl_time_embedding()`, which returns a
tensor.

## Details

- `zl_linear()`: `zl_crossprod(weight, x) + bias`, with `weight` of dims
  `c(n_in, n_out)`; parameters `<name>.weight` and `<name>.bias`,
  initialised uniformly on `(-1, 1) / sqrt(n_in)` as in PyTorch.

- `zl_embedding()`: rows of `<name>.weight`, of dims `c(dim, n)`, for
  integer indices counting from 0 (an input declared with
  [`zl_int()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md));
  the result has dims `c(dim, dim(idx))`. Initialised as N(0, 1).

- `zl_norm()`: layer or RMS normalisation of each column, then, with
  `affine = TRUE`, a scale `<name>.weight` (ones) and, for layer
  normalisation, a shift `<name>.bias` (zeros).

- `zl_attention()`: multi-head self-attention with linear projections
  `<name>.q`, `.k`, `.v` and `.o`, scaled by `1 / sqrt(dim / heads)`,
  causal by default.

- `zl_mlp()`: `down(swiglu(gate(x), up(x)))`, or `down(silu(up(x)))`
  with `act = "silu"`, with linears `<name>.gate`, `.up` and `.down`.

- `zl_block()`: a pre-norm transformer block, `h = x + attn(norm1(x))`,
  then `h + mlp(norm2(h))`, with layer norms.

- `zl_adaln()`: adaptive layer normalisation's modulation,
  `function(x, cond) x * (1 + scale(cond)) + shift(cond)`, with linears
  `<name>.scale` and `<name>.shift` from `cond_dim` to `dim`.

- `zl_time_embedding()`: no parameters; the sinusoidal embedding of
  times `t` of dims `c(1, ...)`: `cos(t f)` then `sin(t f)` for the
  `dim / 2` frequencies `f_k = max_period^(-k / (dim / 2))`.

## Examples

``` r
p <- zl_params()
emb <- zl_linear(p, "in", n_in = 3, n_out = 8)
blk <- zl_block(p, "b1", dim = 8, heads = 2)
g <- zl_graph(function(x) blk(emb(x$x)), p, inputs = list(x = c(3, 5, 2)))
dim(zl_forward(g, list(x = array(0.1, c(3, 5, 2))))$output)
#> [1] 8 5 2
```
