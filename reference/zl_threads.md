# Threads

`zl_threads()` sets how many threads 'ggml' computes with, for every
graph, trainer and gradient of the session, and returns the previous
number; with no argument it returns the current one. It starts at 1. A
request is capped at the number of available cores and, while
`R CMD check` limits cores (`_R_CHECK_LIMIT_CORES_`), at 2: too many
threads on too few cores make 'ggml' many times slower.

## Usage

``` r
zl_threads(n = NULL)
```

## Arguments

- n:

  `NULL`, or the number of threads: a positive whole number.

## Value

The number of threads; when setting, the previous number, invisibly.

## Details

One pool of threads serves the session; it is recreated when the number
changes, and in a process forked from this one. Results are identical at
any number of threads (see the package documentation on determinism). A
computation can be interrupted between two of its operations; it then
signals a condition of class `zulm_interrupt`, which also inherits
`interrupt`.

## Examples

``` r
old <- zl_threads(2)
zl_threads()
#> [1] 2
zl_threads(old)
```
