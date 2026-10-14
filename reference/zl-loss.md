# Losses

Each returns a scalar tensor. `zl_cross_entropy(logits, prob)` is the
cross-entropy between the softmax of each column of `logits` and the
probabilities `prob` of the same shape, summed over columns and divided
by their number, as 'ggml' computes it. `zl_mse()` and `zl_mae()` are
the mean squared and mean absolute errors over every element.

## Usage

``` r
zl_cross_entropy(logits, prob)

zl_mse(pred, target)

zl_mae(pred, target)
```

## Arguments

- logits, prob, pred, target:

  Tensors.

## Value

A scalar tensor.

## Examples

``` r
p <- zl_params()
g <- zl_graph(
  function(x) zl_mse(x$pred, x$target),
  p,
  inputs = list(pred = 4, target = 4)
)
zl_forward(g, list(pred = c(1, 2, 3, 4), target = c(1, 2, 3, 6)))$output
#> [1] 1
```
