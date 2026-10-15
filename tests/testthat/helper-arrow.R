# Helpers of test-arrow.R.

# A stream of `batches` record batches of `size` rows of one float32 column
# "x" counting from 0, whose releases are counted in `counts`.
counting_stream <- function(batches, size, counts) {
  zlm_call("zlm_arrow_test_stream", as.integer(batches), as.integer(size), counts)
}

# Every batch of an iterator, as a list of lists.
drain <- function(it) {
  out <- list()
  while (!is.null(b <- zl_next(it))) out[[length(out) + 1L]] <- b
  out
}

# nanoarrow converts R vectors to int8, uint8 and int16 only through the 'arrow'
# package, which zulm does not suggest: those columns are built from their
# bytes, and every other column by nanoarrow.
small_int_array <- function(v, type, width) {
  a <- nanoarrow::nanoarrow_array_init(type)
  bytes <- writeBin(as.integer(v), raw(), size = width, endian = "little")
  nanoarrow::nanoarrow_array_modify(a, list(
    length = length(v), null_count = 0,
    buffers = list(NULL, nanoarrow::as_nanoarrow_buffer(bytes))
  ))
}

batch_array <- function(df, schema) {
  children <- lapply(names(schema$children), function(nm) {
    type <- schema$children[[nm]]
    width <- c(c = 1L, C = 1L, s = 2L)[type$format]
    if (!is.na(width)) {
      small_int_array(df[[nm]], type, width)
    } else {
      nanoarrow::as_nanoarrow_array(df[[nm]], schema = type)
    }
  })
  names(children) <- names(schema$children)
  nanoarrow::nanoarrow_array_modify(nanoarrow::nanoarrow_array_init(schema), list(
    length = nrow(df), null_count = 0, children = children
  ))
}

na_stream <- function(batches, schema) {
  arrays <- lapply(batches, batch_array, schema = schema)
  nanoarrow::basic_array_stream(arrays, schema = schema)
}
