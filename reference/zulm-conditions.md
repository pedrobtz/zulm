# Conditions signalled by zulm

Every error zulm signals inherits `zulm_error`, and every warning
`zulm_warning`, so a handler can catch them all or one class at a time
with [`tryCatch()`](https://rdrr.io/r/base/conditions.html) or
[`withCallingHandlers()`](https://rdrr.io/r/base/conditions.html).

## Details

Errors:

- `zulm_invalid_argument`: an argument of the wrong kind or value.

- `zulm_shape_error`: dimensions that do not fit; carries `op`, the
  shapes and the R call where an operation is involved.

- `zulm_type_error`: a tensor or value type that is not accepted.

- `zulm_unsupported_backward`: an operation without a gradient on a
  gradient path; carries `op`.

- `zulm_memory_error`: an allocation beyond
  `getOption("zulm.max_bytes")` (4 GiB by default); carries `requested`
  and `allowed`, in bytes.

- `zulm_engine_error`: an assertion of the ggml library caught while a
  graph was being built; carries `ggml_message`. The graph is discarded.

- `zulm_interrupt`: a computation stopped by a user interrupt; also
  inherits `interrupt`.

- `zulm_file_error`: a GGUF or safetensors file that cannot be read or
  written.

- `zulm_arrow_error`: an Arrow stream with an unsupported type, a null,
  a dictionary or a stream failure; carries `column` and `row`.

- `zulm_companion_error`: a companion package whose interface lacks
  something zulm needs.

- `zulm_stale_error`: an object that is no longer valid, because it was
  restored from a saved R session or created by a variant no longer in
  use.

- `zulm_internal_error`: a bug in zulm.

Warnings:

- `zulm_range_warning`: values beyond the range of the storage type,
  which became infinite.

- `zulm_nonfinite_warning`: a training step whose loss is not finite.

## Examples

``` r
p <- zl_params()
tryCatch(
  zl_param_get(p, "missing"),
  zulm_invalid_argument = function(e) conditionMessage(e)
)
#> [1] "The store has no parameter 'missing'."
```
