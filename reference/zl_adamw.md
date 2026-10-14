# Optimisers

`zl_adamw()` is AdamW, `zl_sgd()` stochastic gradient descent, both as
'ggml-opt' implements them. Weight decay is decoupled from the gradient:
each step multiplies the parameters by `1 - lr * weight_decay`.

## Usage

``` r
zl_adamw(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-08, weight_decay = 0)

zl_sgd(lr = 0.01, weight_decay = 0)
```

## Arguments

- lr:

  The learning rate, positive.

- beta1, beta2:

  AdamW's moment decay rates, in `[0, 1]`.

- eps:

  AdamW's denominator offset, non-negative.

- weight_decay:

  In `[0, 1]`.

## Value

An object of class `zl_optimizer`.

## Examples

``` r
zl_adamw(lr = 3e-4, weight_decay = 0.01)
#> <zl_optimizer> AdamW, lr 3e-04, betas 0.9 and 0.999, eps 1e-08, weight decay 0.01
zl_sgd(lr = 0.1)
#> <zl_optimizer> SGD, lr 0.1, weight decay 0
```
