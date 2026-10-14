# State tensors, offset writes and the graph cache (design §6.9).

test_that("a state starts at zero, is read and replaced from R", {
  p <- zl_params()
  zl_state(p, "s", c(2, 3))
  expect_identical(zl_state_get(p, "s"), matrix(0, 2, 3))
  zl_state_set(p, "s", matrix(1:6 / 4, 2, 3))
  expect_identical(zl_state_get(p, "s"), matrix(1:6 / 4, 2, 3))
  expect_error(zl_state_set(p, "s", 1:6), class = "zulm_shape_error")
  expect_error(zl_state(p, "s", 2), class = "zulm_invalid_argument")
})

test_that("offset writes land where documented and refuse to overrun", {
  p <- zl_params()
  zl_state(p, "s", c(3, 4))
  zl_state_set(p, "s", c(1, 2, 3), offset = 4)
  expect_identical(as.vector(zl_state_get(p, "s")), c(0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0))
  zl_state_set(p, "s", c(5, 6), offset = 10)
  expect_identical(as.vector(zl_state_get(p, "s"))[11:12], c(5, 6))
  zl_state_set(p, "s", numeric(), offset = 12)
  expect_error(zl_state_set(p, "s", c(1, 2), offset = 11), "outside state", class = "zulm_invalid_argument")
  expect_error(zl_state_set(p, "s", 1, offset = 12), class = "zulm_invalid_argument")
  expect_error(zl_state_set(p, "s", 1, offset = -1), class = "zulm_invalid_argument")
  expect_error(zl_state_set(p, "s", 1, offset = 1.5), class = "zulm_invalid_argument")
  expect_error(zl_state_set(p, "s", NA_real_, offset = 0), class = "zulm_invalid_argument")
  expect_identical(as.vector(zl_state_get(p, "s"))[11:12], c(5, 6))
})

test_that("states and parameters keep to their own functions", {
  p <- make_store()
  zl_state(p, "s", 4)
  expect_identical(zl_param_names(p), c("w", "b"))
  expect_identical(zl_param_count(p), 8)
  expect_error(zl_param_get(p, "s"), "state tensor", class = "zulm_invalid_argument")
  expect_error(zl_param_set(p, "s", 1:4), "state tensor", class = "zulm_invalid_argument")
  expect_error(zl_state_get(p, "w"), "no state", class = "zulm_invalid_argument")
  expect_error(zl_state_set(p, "w", 1, offset = 0), "no state", class = "zulm_invalid_argument")
  expect_output(print(p), "(state)", fixed = TRUE)
})

test_that("states are not saved", {
  p <- make_store()
  zl_state(p, "s", 4)
  zl_state_set(p, "s", 1:4)
  f <- withr::local_tempfile(fileext = ".gguf")
  zl_save(p, f)
  q <- zl_load(f)
  expect_identical(zl_param_names(q), c("w", "b"))
  expect_identical(zl_param_get(q, "w"), zl_param_get(p, "w"))
  f2 <- withr::local_tempfile(fileext = ".safetensors")
  zl_save_safetensors(p, f2)
  expect_setequal(names(zl_safetensors_read(f2)), c("w", "b"))
})

test_that("a graph writes rows of a state on every run and reads them back", {
  p <- zl_params()
  zl_state(p, "s", c(2, 4))
  g <- zl_graph(
    function(x) {
      s <- zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows)
      list(all = s + 0, total = zl_sum(s))
    },
    p,
    inputs = list(new = c(2, 2), rows = zl_int(2))
  )
  expect_identical(zl_graph_info(g)$state_writes, 1L)
  out <- zl_forward(g, list(new = matrix(1:4, 2, 2), rows = c(3L, 0L)))
  expect_identical(out$all, matrix(c(3, 4, 0, 0, 0, 0, 1, 2), 2, 4))
  expect_identical(out$total, 10)
  out <- zl_forward(g, list(new = matrix(5:8, 2, 2), rows = c(1L, 2L)))
  expect_identical(out$all, matrix(c(3, 4, 5, 6, 7, 8, 1, 2), 2, 4))
  expect_identical(zl_state_get(p, "s"), out$all)
  # an output that does not read the state still runs the write
  zl_state_set(p, "s", matrix(0, 2, 4))
  g2 <- zl_graph(
    function(x) {
      zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows)
      zl_sum(x$new)
    },
    p,
    inputs = list(new = c(2, 1), rows = zl_int(1))
  )
  zl_forward(g2, list(new = matrix(c(7, 9), 2, 1), rows = 2L))
  expect_identical(zl_state_get(p, "s")[, 3], c(7, 9))
})

test_that("state writes across heads use one row index per slice", {
  p <- zl_params()
  zl_state(p, "s", c(2, 3, 2))
  g <- zl_graph(
    function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows) + 0,
    p,
    inputs = list(new = c(2, 1, 2), rows = zl_int(1))
  )
  out <- zl_forward(g, list(new = array(1:4, c(2, 1, 2)), rows = 1L))$output
  want <- array(0, c(2, 3, 2))
  want[, 2, 1] <- 1:2
  want[, 2, 2] <- 3:4
  expect_identical(out, want)
})

test_that("bad rows and bad writes are refused before ggml sees them", {
  p <- make_store()
  zl_state(p, "s", c(2, 4))
  g <- zl_graph(
    function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows) + 0,
    p,
    inputs = list(new = c(2, 2), rows = zl_int(2))
  )
  before <- zl_state_get(p, "s")
  expect_error(zl_forward(g, list(new = matrix(1, 2, 2), rows = c(1L, 1L))), "twice", class = "zulm_invalid_argument")
  expect_error(zl_forward(g, list(new = matrix(1, 2, 2), rows = c(0L, 4L))), "below 4", class = "zulm_invalid_argument")
  expect_error(zl_forward(g, list(new = matrix(1, 2, 2), rows = c(-1L, 0L))), class = "zulm_invalid_argument")
  expect_identical(zl_state_get(p, "s"), before)
  build_err <- function(build, inputs) {
    expect_error(zl_graph(build, p, inputs), class = "zulm_error")
  }
  # into a parameter
  expect_error(zl_graph(function(x) zl_state_write(zl_param_tensor(p, "w"), x$new, x$rows),
                        p, list(new = c(3, 1), rows = zl_int(1))), "state tensor", class = "zulm_invalid_argument")
  # not a reference at all
  expect_error(zl_graph(function(x) zl_state_write(x$new, x$new, x$rows),
                        p, list(new = c(2, 1), rows = zl_int(1))), class = "zulm_invalid_argument")
  # wrong row width, wrong row count, f32 rows, rows not an input
  build_err(function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows),
            list(new = c(3, 1), rows = zl_int(1)))
  build_err(function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows),
            list(new = c(2, 2), rows = zl_int(3)))
  build_err(function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows),
            list(new = c(2, 1), rows = 1))
  build_err(function(x) zl_state_write(zl_param_tensor(p, "s"), x$new, x$new),
            list(new = c(2, 2)))
})

test_that("a graph that writes state cannot be trained, and states are never trained", {
  p <- make_store()
  zl_state(p, "s", c(2, 1))
  g <- zl_graph(
    function(x) {
      s <- zl_state_write(zl_param_tensor(p, "s"), x$new, x$rows)
      list(loss = zl_sum(s))
    },
    p,
    inputs = list(new = c(2, 1), rows = zl_int(1)),
    loss = "loss"
  )
  expect_error(zl_trainer(g, zl_sgd(0.1)), "inference", class = "zulm_invalid_argument")
  # a state read by a trained graph stays as it was
  zl_state_set(p, "s", matrix(c(0.5, -1), 2, 1))
  g2 <- zl_graph(
    function(x) list(loss = zl_sum(zl_crossprod(zl_param_tensor(p, "w"), x$x) * zl_param_tensor(p, "s"))),
    p,
    inputs = list(x = c(3, 1)),
    loss = "loss"
  )
  tr <- zl_trainer(g2, zl_sgd(0.1))
  w0 <- zl_param_get(p, "w")
  zl_step(tr, list(x = matrix(1, 3, 1)))
  expect_false(identical(zl_param_get(p, "w"), w0))
  expect_identical(zl_state_get(p, "s"), matrix(f32(c(0.5, -1)), 2, 1))
})

test_that("the C offset write into an input lands where documented", {
  p <- zl_params()
  g <- zl_graph(function(x) x$a + 0, p, inputs = list(a = c(2, 3), i = zl_int(1)))
  zl_forward(g, list(a = matrix(0, 2, 3), i = 0L))
  zlm_call <- getFromNamespace("zlm_call", "zulm")
  zlm_call("zlm_input_write_at", g$ptr, "a", 3, c(7, 8, 9))
  expect_error(zlm_call("zlm_input_write_at", g$ptr, "a", 4, c(1, 2, 3)), "returned 3", class = "zulm_invalid_argument")
  expect_error(zlm_call("zlm_input_write_at", g$ptr, "a", -1, 1), "returned 3", class = "zulm_invalid_argument")
  expect_error(zlm_call("zlm_input_write_at", g$ptr, "i", 0, 1), "returned 2", class = "zulm_invalid_argument")
  expect_error(zlm_call("zlm_input_write_at", g$ptr, "nope", 0, 1), "returned 1", class = "zulm_invalid_argument")
  out <- zlm_call("zlm_graph_compute", g$ptr, 0L)[[1L]]
  expect_identical(out, c(0, 0, 0, 7, 8, 9))
})

test_that("the graph cache builds once per shape and evicts under the cap", {
  p <- make_store()
  gc <- zl_graph_cache(function(x) zl_crossprod(zl_param_tensor(p, "w"), x$input), p)
  expect_s3_class(gc, "zl_graph_cache")
  for (n in c(5, 7, 5, 7, 5, 1)) {
    out <- zl_forward(gc, list(input = matrix(1, 3, n)))$output
    expect_identical(dim(out), c(2L, as.integer(n)))
  }
  expect_identical(gc$builds, 3L)
  expect_length(gc, 3L)
  expect_output(print(gc), "3 graphs held, 3 built")
  # integer data makes an integer input
  ge <- zl_graph_cache(function(x) zl_get_rows(zl_param_tensor(p, "w"), x$i), p)
  expect_identical(zl_forward(ge, list(i = c(1L, 0L)))$output, zl_param_get(p, "w")[, c(2, 1)])
  # under a cap that holds one graph, the least recently used goes first
  one <- sum(zl_graph_info(gc$graphs[[1]])$bytes)
  store <- zlm_store_bytes(p)
  withr::local_options(zulm.max_bytes = store + 1.5 * one)
  small <- zl_graph_cache(function(x) zl_crossprod(zl_param_tensor(p, "w"), x$input), p)
  zl_forward(small, list(input = matrix(1, 3, 5)))
  zl_forward(small, list(input = matrix(1, 3, 6)))
  expect_length(small, 1L)
  zl_forward(small, list(input = matrix(1, 3, 5)))
  expect_identical(small$builds, 3L)
  expect_error(zl_graph_cache("f", p), class = "zulm_invalid_argument")
  expect_error(zl_forward(gc, list(matrix(1, 3, 5))), class = "zulm_invalid_argument")
})
