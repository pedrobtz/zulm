# Build and runtime information

Reports the vendored ggml version, the variant loaded for this CPU, the
variants installed and usable, the native-build hooks recorded at
install time, and the CPU features ggml was compiled for.

## Usage

``` r
zulm_info()
```

## Value

A named list.

## Details

A variant is the whole compiled library built for one instruction-set
level: on x86-64 `"baseline"`, `"sse42"` and `"avx2"`, on arm64
`"neon"`, and `"native"` when the package was installed with the
`ZULM_CFLAGS` and `ZULM_CXXFLAGS` hooks set. The best usable variant is
loaded when the package is; set `options(zulm.variant =)` before loading
it to force another.

## Examples

``` r
info <- zulm_info()
info$variant
#> [1] "avx2"
info$ggml_version
#> [1] "0.26.0"
```
