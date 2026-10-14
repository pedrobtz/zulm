# Train a graph

`zl_trainer()` prepares a graph with a `loss` for training with an
optimiser: it checks that every operation between a parameter and the
loss has a gradient (`zulm_unsupported_backward` names the one that has
not, and the call that made it), and builds the gradient and optimiser
graphs once, through 'ggml-opt'. Every f32 parameter the loss depends on
is trained; half-precision parameters cannot be.

## Usage

``` r
zl_trainer(g, opt, accumulate = 1L, clip_norm = NULL)

zl_step(tr, data, lr = NULL)

zl_grad(tr, name)

zl_reset(tr)
```

## Arguments

- g:

  A graph from
  [`zl_graph()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
  with a `loss`.

- opt:

  An optimiser from
  [`zl_adamw()`](https://pedrobtz.github.io/zulm/reference/zl_adamw.md)
  or
  [`zl_sgd()`](https://pedrobtz.github.io/zulm/reference/zl_adamw.md).

- accumulate:

  The number of batches whose gradients are summed before each update.

- clip_norm:

  `NULL`, or the largest joint L2 norm of the gradients at an update.

- tr:

  A trainer.

- data:

  A named list of arrays for the graph's inputs, as for
  [`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md).

- lr:

  The learning rate of this step; `NULL` for the optimiser's.

- name:

  A parameter's name.

## Value

`zl_trainer()` returns a trainer; `zl_step()` the loss, a number;
`zl_grad()` an array of the parameter's dimensions; `zl_reset()` the
trainer, invisibly.

## Details

`zl_step()` writes a batch into the graph's inputs, computes the loss
and the gradients, and, every `accumulate` calls, updates the
parameters. It returns the loss of the batch; a loss that is not finite
gives a `zulm_nonfinite_warning`. With `clip_norm`, each update first
scales all gradients together so that their joint L2 norm is at most
`clip_norm`.

`zl_grad()` returns the gradient of the loss with respect to one
parameter, at the parameters' current values, on the inputs last written
by `zl_step()` or
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md).
`zl_reset()` zeroes the optimiser's state (AdamW's moments and step
count).

An optimiser's state is not saved by
[`zl_save()`](https://pedrobtz.github.io/zulm/reference/zl_save.md), and
if a step is interrupted the state is undefined: call `zl_reset()`
before going on.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(3, 1), init = zl_init_zeros())
g <- zl_graph(
  function(x) list(loss = zl_mse(zl_crossprod(zl_param_tensor(p, "w"), x$x), x$y)),
  p,
  inputs = list(x = c(3, 8), y = c(1, 8)),
  loss = "loss"
)
tr <- zl_trainer(g, zl_adamw(lr = 0.05))
set.seed(1)
x <- matrix(rnorm(24), 3, 8)
y <- matrix(c(1, -2, 0.5) %*% x, 1, 8)
for (i in 1:200) loss <- zl_step(tr, list(x = x, y = y))
loss
#> [1] 3.528795e-09
zl_param_get(p, "w")
#>            [,1]
#> [1,]  0.9999790
#> [2,] -1.9999368
#> [3,]  0.5000194
```
