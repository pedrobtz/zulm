# Save and load parameters as GGUF

`zl_save()` writes every parameter of a store, and optional metadata, to
a GGUF file, the format of the 'ggml' library, which other tools
(llama.cpp, the 'gguf' Python package) read. `zl_load()` reads one into
a new store, with every tensor in single precision unless `keep_types`
is `TRUE`.

## Usage

``` r
zl_save(p, path, type = c("f32", "f16", "bf16"), metadata = list())

zl_load(path, keep_types = FALSE)
```

## Arguments

- p:

  A parameter store.

- path:

  The file to write or read.

- type:

  The type tensors are written as: `"f32"`, `"f16"` or `"bf16"`.
  Half-precision types halve the file and round each value.

- metadata:

  A named list of key-value pairs to store with the tensors: each value
  a character, integer, double or logical vector without `NA`. Keys
  starting with `zulm.` are reserved.

- keep_types:

  For `zl_load()`: `FALSE` widens every tensor to f32; `TRUE` keeps f16
  and bf16 tensors in their type in memory (see
  [`zl_params()`](https://pedrobtz.github.io/zulm/reference/zl_params.md)
  for the cost in precision).

## Value

`zl_save()` returns `path` invisibly. `zl_load()` returns a parameter
store whose `"metadata"` attribute is the file's metadata as a named
list.

## Details

Only parameters are saved: the state of an optimiser is not, so training
resumed from a loaded store starts with fresh optimiser state.

Tensor dimensions follow R's order: a GGUF tensor of `ne = {d1, d2}` is
an R matrix of `dim = c(d1, d2)`, which NumPy reads as an array of shape
`(d2, d1)` holding the same bytes.

## Examples

``` r
p <- zl_params()
zl_param(p, "w", c(3, 2))
path <- tempfile(fileext = ".gguf")
zl_save(p, path, metadata = list(model = "example", width = 3L))
q <- zl_load(path)
all.equal(zl_param_get(q, "w"), zl_param_get(p, "w"))
#> [1] TRUE
attr(q, "metadata")$model
#> [1] "example"
unlink(path)
```
