# Batches from Arrow streams

`zl_arrow_batches()` reads record batches from an Arrow stream, such as
a 'DuckDB' query, a Parquet file or an 'arrow' table, and cuts them into
batches of exactly `rows` rows for
[`zl_step()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
and
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md),
which take the iterator as `data`. `zl_next()` returns the next batch as
R arrays, for inspection. Each batch is one copy of the data: columns
are interleaved into feature-major arrays, `c(features, rows)`.

## Usage

``` r
zl_arrow_batches(
  stream,
  map,
  rows,
  na = c("error", "mask"),
  last = c("drop", "pad")
)

zl_next(it)

# S3 method for class 'zl_arrow_batches'
print(x, ...)
```

## Arguments

- stream:

  An Arrow stream: a `nanoarrow_array_stream`, or an object that
  'nanoarrow' converts to one (a data frame, an 'arrow' table or
  reader).

- map:

  A named list from input names to column names, as above.

- rows:

  The rows per batch.

- na:

  What a missing value does: `"error"` or `"mask"`.

- last:

  What the final, partial batch does: `"drop"` or `"pad"`.

- it:

  An iterator from `zl_arrow_batches()`.

- x:

  An iterator, for [`print()`](https://rdrr.io/r/base/print.html).

- ...:

  Unused.

## Value

`zl_arrow_batches()` returns an iterator of class `zl_arrow_batches`;
`zl_next()` a named list of arrays, or `NULL` at the end of the stream.

## Details

`map` names the graph's inputs and gives each its columns: a character
vector for an f32 input of `c(length(columns), rows)`, or `zl_int(...)`
around the column names for an i32 input (a vector of `rows` for one
column). float32 columns are copied, float64 rounded to f32, int8 to
int64 and boolean (0 and 1) go to either kind of input, and
dictionary-encoded columns of strings go to i32 inputs as their codes,
counting from 0. int64 values beyond the 32-bit range are errors in an
i32 input. The dictionary must be the same in every record batch. Other
types are refused.

Missing values are errors naming the column and row (`na = "error"`).
With `na = "mask"` they are written as 0, and every input `x` gets a
companion input `x_mask` of its shape holding 1 for a value and 0 for a
missing one. The rows left at the end of the stream are dropped
(`last = "drop"`), or padded with zeros to a full batch
(`last = "pad"`), in which case every batch carries a `row_mask` input
of `rows` values, 1 for a real row and 0 for padding. A graph that takes
the iterator declares these inputs too.

The stream is taken over by the iterator: it is read on the main thread
only, as producers such as 'DuckDB' require, and released when it ends
or when the iterator is garbage collected.
[`zl_step()`](https://pedrobtz.github.io/zulm/reference/zl_trainer.md)
and
[`zl_forward()`](https://pedrobtz.github.io/zulm/reference/zl_graph.md)
return `NULL` once it has ended.

A `nanoarrow_array_stream` is used directly; anything else is converted
with
[`nanoarrow::as_nanoarrow_array_stream()`](https://arrow.apache.org/nanoarrow/latest/r/reference/as_nanoarrow_array_stream.html),
which needs the 'nanoarrow' package (a `zulm_companion_error` without
it).

## Examples

``` r
if (requireNamespace("nanoarrow", quietly = TRUE)) {
  df <- data.frame(a = c(1, 2, 3, 4, 5), b = c(0.5, 1, 1.5, 2, 2.5),
                   kind = factor(c("x", "y", "x", "x", "y")))
  it <- zl_arrow_batches(df, list(x = c("a", "b"), kind = zl_int("kind")), rows = 2)
  zl_next(it)
  zl_next(it)
  zl_next(it)
}
#> NULL
```
