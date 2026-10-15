client_info <- function() .Call(C_client_info)
client_require <- function(extra) .Call(C_client_require, as.integer(extra))
client_sampler <- function(
  g,
  stream,
  input,
  output,
  size,
  steps,
  index = NULL
) {
  .Call(
    C_client_sampler,
    g,
    stream,
    input,
    output,
    as.integer(size),
    as.integer(steps),
    index
  )
}
client_average <- function(a, b, name, size) {
  .Call(C_client_average, a, b, name, as.integer(size))
}
client_state <- function(p, name, offset, values) {
  .Call(C_client_state, p, name, as.double(offset), as.double(values))
}
client_batches <- function(g, it, output, size) {
  .Call(C_client_batches, g, it, output, as.integer(size))
}
client_codes <- function(g, p) .Call(C_client_codes, g, p)
client_draw_threads <- function(stream, step, n, threads, register = TRUE) {
  .Call(
    C_client_draw_threads,
    stream,
    as.integer(step),
    as.integer(n),
    as.integer(threads),
    register
  )
}
