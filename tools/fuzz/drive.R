# The fuzz targets' R side (tools/fuzz/fuzz.c): each input's bytes become
# calls to zulm's R functions. Every zulm condition but the two that mean
# a bug is an expected refusal; a zulm_engine_error (a ggml assertion
# caught in a builder), a zulm_internal_error or an error without a zulm
# class is a finding, and stops the target.

# A reader of the input: each call takes the next byte, modulo k.
fz_reader <- function(raw) {
  i <- 0L
  n <- length(raw)
  function(k = 256L) {
    i <<- i + 1L
    if (i > n) 0L else as.integer(raw[[i]]) %% as.integer(k)
  }
}

fz_is_finding <- function(e) {
  !inherits(e, "zulm_error") || inherits(e, c("zulm_engine_error", "zulm_internal_error"))
}

# Reports a finding with the calls that led to it, then aborts.
fz_finding <- function(e) {
  calls <- vapply(sys.calls(), function(cl) paste(deparse(cl, nlines = 1L), collapse = ""), "")
  cat("fuzz finding: ", paste(class(e), collapse = "/"), ": ", conditionMessage(e), "\n",
    paste0("  ", rev(calls), collapse = "\n"), "\n",
    sep = "", file = stderr()
  )
  zlm_dot_call(.zlm$routines[["zlm_fuzz_abort"]])
}

# Runs f(): a finding aborts where it is signalled, with its calls; any
# other zulm error is an expected refusal.
fz_guard <- function(f) {
  tryCatch(
    withCallingHandlers(
      f(),
      error = function(e) if (fz_is_finding(e)) fz_finding(e),
      warning = function(w) invokeRestart("muffleWarning")
    ),
    zulm_error = function(e) {
      if (nzchar(Sys.getenv("ZLM_FUZZ_TRACE"))) cat("refused: ", conditionMessage(e), "\n", sep = "", file = stderr())
    }
  )
  invisible(TRUE)
}

fz_pick <- function(b, x) x[[b(length(x)) + 1L]]

# Mostly valid values, sometimes the edge cases R must refuse.
fz_num <- function(b) fz_pick(b, list(0, 1, -1, 0.5, 2, 1e-5, 3, 1e30, -Inf, NaN, NA, 10000))
fz_int <- function(b) fz_pick(b, list(0, 1, 2, 3, 4, -1, NA, 8, 1.5, 2^31))
fz_extent <- function(b) {
  if (b(32L) == 0L) fz_pick(b, list(0, -1, NA, 1e10, 2.5)) else fz_pick(b, list(1, 2, 3, 4, 5, 8))
}
fz_dim <- function(b) {
  r <- 1L + b(if (b(16L) == 0L) 6L else 4L)
  vapply(seq_len(r), function(i) fz_extent(b), numeric(1))
}

# One operation on tensors of the pool; NULL when it is refused.
fz_op <- function(b, pool) {
  ia <- b(length(pool)) + 1L
  ic <- b(length(pool)) + 1L
  id <- b(length(pool)) + 1L
  a <- pool[[ia]]
  c <- pool[[ic]]
  d <- pool[[id]]
  ops <- list(
    function() a + c, function() a - c, function() a * c, function() a / c,
    function() a + fz_num(b), function() fz_num(b) * a, function() zl_scale(a, fz_num(b)),
    function() zl_crossprod(a, c), function() zl_t(a), function() zl_cont(a),
    function() zl_reshape(a, fz_dim(b)),
    function() zl_permute(a, fz_pick(b, list(c(2, 1), c(2, 1, 3), c(3, 1, 2, 4), c(1, 2, 3, 4), c(4, 3, 2, 1)))),
    function() zl_permute(a, vapply(1:4, function(i) fz_int(b), numeric(1))),
    function() zl_view(a, fz_dim(b), offset = fz_int(b)),
    function() zl_concat(a, c, axis = fz_int(b)),
    function() zl_pad(a, after = vapply(seq_len(1L + b(4L)), function(i) fz_int(b), numeric(1))),
    function() zl_get_rows(a, c),
    function() zl_clamp(a, fz_num(b), fz_num(b)),
    function() zl_tri(a, fz_pick(b, list("lower", "upper", "lower_strict", "upper_strict", "x"))),
    function() zl_softmax(a, mask = if (b(2L)) c, scale = fz_num(b)),
    function() zl_causal_mask(a),
    function() zl_rope(a, c, n_dims = fz_int(b), base = fz_num(b), mode = fz_pick(b, list("interleaved", "half"))),
    function() zl_rms_norm(a, eps = fz_num(b)), function() zl_layer_norm(a, eps = fz_num(b)),
    function() zl_cross_entropy(a, c), function() zl_mse(a, c), function() zl_mae(a, c),
    function() zl_cast(a, fz_pick(b, list("f32", "f16", "bf16", "i32"))),
    function() zl_sum(a), function() zl_sum_rows(a), function() zl_mean_rows(a),
    function() zl_cumsum(a), function() zl_gelu(a, approximate = fz_pick(b, list("none", "tanh"))),
    function() zl_sigmoid(a), function() zl_tanh(a), function() zl_exp(a), function() zl_log(a),
    function() zl_sqrt(a), function() zl_sqr(a), function() zl_abs(a), function() zl_relu(a),
    function() zl_silu(a), function() zl_softplus(a), function() zl_sin(a), function() zl_cos(a),
    function() zl_swiglu(a, c), function() zl_where(a, c, d),
    function() zl_flash_attention(a, c, d, mask = NULL, scale = fz_num(b)),
    function() zl_state_write(a, c, d),
    function() zl_time_embedding(a, dim = fz_int(b), max_period = fz_num(b))
  )
  op <- fz_pick(b, ops)
  r <- tryCatch(op(), zulm_error = function(e) if (fz_is_finding(e)) fz_finding(e))
  if (nzchar(Sys.getenv("ZLM_FUZZ_TRACE"))) {
    shape <- function(x) if (is.null(x)) "-" else paste(dim(x), collapse = "x")
    cat("[", length(pool) + 1L, "] ", deparse(body(op)), ": ", ia, " ", shape(a), ", ", ic, " ", shape(c), ", ",
      id, " ", shape(d), " -> ", shape(r), "\n",
      sep = "", file = stderr()
    )
  }
  r
}

fz_values <- function(b, dim, int) {
  n <- prod(dim)
  v <- if (int) vapply(seq_len(n), function(i) b(10L) - 1, numeric(1)) else vapply(seq_len(n), function(i) (b() - 128) / 16, numeric(1))
  if (length(dim) > 1L) array(v, dim) else v
}

fz_graph <- function(b) {
  p <- zl_params()
  for (k in seq_len(b(3L))) {
    zl_param(p, paste0("p", k), fz_dim(b), init = if (b(2L)) zl_init_ones() else zl_init_zeros())
  }
  if (b(4L) == 0L) zl_state(p, "s", fz_dim(b))
  inputs <- list()
  for (k in seq_len(1L + b(3L))) {
    d <- fz_dim(b)
    inputs[[paste0("x", k)]] <- if (b(4L) == 0L) zl_int(d) else d
  }
  train <- b(3L) == 0L
  if (nzchar(Sys.getenv("ZLM_FUZZ_TRACE"))) {
    cat("params: ", paste(zl_param_names(p), collapse = " "), "; inputs: ",
      paste(names(inputs), vapply(inputs, function(s) paste(if (is.list(s)) c("i32", s$dim) else s, collapse = "x"), ""), collapse = ", "),
      "; train: ", train, "\n", sep = "", file = stderr()
    )
  }
  build <- function(x) {
    refs <- lapply(c(zl_param_names(p), if (b(2L)) "s"), function(nm) {
      tryCatch(zl_param_tensor(p, nm), zulm_error = function(e) NULL)
    })
    pool <- c(unname(x), Filter(Negate(is.null), refs))
    for (s in seq_len(1L + b(10L))) {
      r <- fz_op(b, pool)
      if (!is.null(r)) pool[[length(pool) + 1L]] <- r
    }
    out <- pool[[length(pool)]]
    if (train) list(y = out, loss = zl_sum(out)) else list(y = out)
  }
  g <- zl_graph(build, p, inputs, loss = if (train) "loss")
  data <- lapply(names(inputs), function(nm) {
    s <- inputs[[nm]]
    if (inherits(s, "zl_int_spec")) fz_values(b, s$dim, TRUE) else fz_values(b, s, FALSE)
  })
  names(data) <- names(inputs)
  if (train) {
    tr <- zl_trainer(g, fz_pick(b, list(zl_sgd(lr = 0.1), zl_adamw(lr = 0.01))),
      clip_norm = if (b(2L)) 1
    )
    for (s in seq_len(1L + b(2L))) zl_step(tr, data)
    zl_grad(tr, "p1")
  }
  zl_forward(g, data)
}

fuzz_graph <- function(raw) fz_guard(function() fz_graph(fz_reader(raw)))

# Target 2: arbitrary bytes as a GGUF file and as a safetensors header.
fuzz_files <- function(raw) {
  path <- tempfile()
  on.exit(unlink(path))
  writeBin(raw, path)
  fz_guard(function() {
    p <- zl_load(path, keep_types = length(raw) %% 2L == 0L)
    for (nm in zl_param_names(p)) zl_param_get(p, nm)
  })
  fz_guard(function() zlm_call("zlm_st_header", path))
}

# Target 3: the type map, on a stream built from the bytes in C.
fuzz_arrow <- function(raw) {
  b <- fz_reader(rev(raw))
  fz_guard(function() {
    s <- zlm_call("zlm_fuzz_stream", raw)
    cols <- c("a", "b", "c")[seq_along(attr(s, "formats"))]
    map <- list(x = if (b(4L) == 0L) zl_int(cols[[1]]) else cols[seq_len(1L + b(length(cols)))])
    it <- zl_arrow_batches(s, map,
      rows = 1L + b(5L),
      na = fz_pick(b, list("error", "mask")), last = fz_pick(b, list("drop", "pad"))
    )
    while (!is.null(zl_next(it))) NULL
  })
}
