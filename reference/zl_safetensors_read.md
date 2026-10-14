# safetensors files

safetensors is the format PyTorch-trained weights travel in.
`zl_safetensors_read()` returns a file's tensors as R arrays;
`zl_load_safetensors()` fills a store's parameters from one, reading
each tensor straight into the store; `zl_save_safetensors()` writes a
store.

## Usage

``` r
zl_safetensors_read(path, names = NULL)

zl_load_safetensors(p, path, map = NULL)

zl_save_safetensors(p, path, type = c("f32", "f16", "bf16"), metadata = list())
```

## Arguments

- path:

  The file.

- names:

  Tensors to read; `NULL` for all.

- p:

  A parameter store.

- map:

  `NULL`, or a named character vector from the file's tensor names (the
  names) to the store's parameter names (the values). Without one,
  tensors fill parameters of the same name.

- type:

  The type to write: `"f32"`, `"f16"` or `"bf16"`.

- metadata:

  A named list of strings, written as the header's `__metadata__`.

## Value

`zl_safetensors_read()`: a named list of arrays, with the file's
metadata as the `"metadata"` attribute. `zl_load_safetensors()`: the
names of the parameters it filled, invisibly. `zl_save_safetensors()`:
`path`, invisibly.

## Details

Shapes are row-major in the file, so a tensor of
`shape = c(s1, ..., sk)` is an R array of `dim = c(sk, ..., s1)` holding
the same bytes: PyTorch's `nn.Linear` weight of shape `[out, in]` is
`c(in, out)`, the layout of
[`zl_linear()`](https://pedrobtz.github.io/zulm/reference/zl-modules.md),
with no transposition.

Types: F32, F16 and BF16 values are read as single precision; F64 is
rounded to single precision (with a `zulm_range_warning` if a value
overflows); I32 and I64 are read as integers by `zl_safetensors_read()`.
A parameter declared `type = "f16"` or `"bf16"` keeps a matching tensor
as it is and rounds others to its type. Other types are refused.

The header is checked before any data is read: its length, every
tensor's offsets (in range, in order, without overlap, covering the
data) and every size against its type and shape. A failure is
`zulm_file_error`.

## Examples

``` r
p <- zl_params()
zl_param(p, "lin.weight", c(3, 2))
path <- tempfile(fileext = ".safetensors")
zl_save_safetensors(p, path, metadata = list(source = "example"))
str(zl_safetensors_read(path))
#> List of 1
#>  $ lin.weight: num [1:3, 1:2] -0.000154 0.037743 0.033246 0.020397 -0.035935 ...
#>  - attr(*, "metadata")=List of 1
#>   ..$ source: chr "example"
q <- zl_params()
zl_param(q, "w", c(3, 2), init = zl_init_zeros())
zl_load_safetensors(q, path, map = c(lin.weight = "w"))
identical(zl_param_get(q, "w"), zl_param_get(p, "lin.weight"))
#> [1] TRUE
unlink(path)
```
