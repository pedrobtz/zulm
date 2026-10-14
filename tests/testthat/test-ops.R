# Every forward operation against a base-R oracle within f32 tolerance, at
# random shapes including broadcasts (design §11; Stage 3 exit criterion).

as_r <- function(x, d) if (length(d) > 1L) array(x, d) else as.vector(x)

test_that("binary arithmetic broadcasts either way, as ggml repeats", {
  withr::local_seed(10)
  ops <- list(`+` = `+`, `-` = `-`, `*` = `*`, `/` = `/`)
  for (rep in 1:12) {
    da <- rand_dims()
    db <- bcast_dims(da)
    if (rep %% 2 == 0) {
      tmp <- da
      da <- db
      db <- tmp
    }
    a <- rand_array(da, 0.5, 2)
    b <- rand_array(db, 0.5, 2)
    big <- if (prod(da) >= prod(db)) da else db
    out <- run_graph(
      function(x) list(add = x$a + x$b, sub = x$a - x$b, mul = x$a * x$b, div = x$a / x$b),
      list(a = da, b = db), list(a = a, b = b)
    )
    A <- bcast(a, big)
    B <- bcast(b, big)
    for (op in names(ops)) {
      key <- switch(op, `+` = "add", `-` = "sub", `*` = "mul", `/` = "div")
      expect_f32(as.vector(out[[key]]), as.vector(ops[[op]](A, B)))
      expect_identical(length(out[[key]]), as.integer(prod(big)))
    }
  }
})

test_that("numbers on either side are scales and constants", {
  a <- f32(c(0.5, 1, 2, 4))
  out <- run_graph(
    function(x) list(
      p1 = x$a + 3, p2 = 3 + x$a, m1 = x$a - 3, m2 = 3 - x$a,
      t1 = x$a * 3, t2 = 3 * x$a, d1 = x$a / 4, d2 = 4 / x$a,
      neg = -x$a, pos = +x$a, s = zl_scale(x$a, -2)
    ),
    list(a = 4), list(a = a)
  )
  expect_f32(out$p1, a + 3)
  expect_f32(out$p2, 3 + a)
  expect_f32(out$m1, a - 3)
  expect_f32(out$m2, 3 - a)
  expect_f32(out$t1, a * 3)
  expect_f32(out$t2, 3 * a)
  expect_f32(out$d1, a / 4)
  expect_f32(out$d2, 4 / a)
  expect_identical(out$neg, -a)
  expect_identical(out$pos, a)
  expect_f32(out$s, -2 * a)
})

test_that("elementwise functions match R", {
  withr::local_seed(11)
  d <- c(4, 3, 2)
  x <- rand_array(d, -3, 3)
  pos <- rand_array(d, 0.1, 3)
  out <- run_graph(
    function(x) list(
      sqr = zl_sqr(x$x), sqrt = zl_sqrt(x$p), exp = zl_exp(x$x), log = zl_log(x$p),
      abs = zl_abs(x$x), sin = zl_sin(x$x), cos = zl_cos(x$x), relu = zl_relu(x$x),
      silu = zl_silu(x$x), softplus = zl_softplus(x$x), sigmoid = zl_sigmoid(x$x),
      tanh = zl_tanh(x$x), gelu = zl_gelu(x$x), gelu_tanh = zl_gelu(x$x, "tanh")
    ),
    list(x = d, p = d), list(x = x, p = pos)
  )
  expect_f32(out$sqr, x^2)
  expect_f32(out$sqrt, sqrt(pos))
  expect_f32(out$exp, exp(x))
  expect_f32(out$log, log(pos))
  expect_f32(out$abs, abs(x))
  expect_f32(out$sin, sin(x))
  expect_f32(out$cos, cos(x))
  expect_f32(out$relu, pmax(x, 0))
  expect_f32(out$silu, x / (1 + exp(-x)))
  expect_f32(out$softplus, log1p(exp(x)))
  expect_f32(out$sigmoid, 1 / (1 + exp(-x)))
  expect_f32(out$tanh, tanh(x))
  expect_f32(out$gelu, x * stats::pnorm(x), tol = 1e-4)
  # ggml evaluates the tanh form through an f16 table: about 1e-3
  gt <- 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  expect_lt(max(abs(out$gelu_tanh - gt)), 2e-3)
  expect_identical(dim(out$gelu), as.integer(d))
})

test_that("reductions and cumulative sums match R", {
  withr::local_seed(12)
  for (d in list(5, c(4, 3), c(3, 2, 2), c(2, 3, 2, 2))) {
    x <- rand_array(d)
    out <- run_graph(
      function(x) list(sum = zl_sum(x$x), rows = zl_sum_rows(x$x), mean = zl_mean_rows(x$x), cum = zl_cumsum(x$x)),
      list(x = d), list(x = x)
    )
    m <- matrix(x, d[1])
    expect_f32(out$sum, sum(x))
    rd <- d
    rd[1] <- 1
    expect_f32(out$rows, as_r(colSums(m), rd))
    expect_f32(out$mean, as_r(colMeans(m), rd))
    expect_f32(out$cum, as_r(apply(m, 2, cumsum), d))
  }
})

test_that("swiglu is silu(gate) * up", {
  withr::local_seed(13)
  g <- rand_array(c(6, 4))
  u <- rand_array(c(6, 4))
  out <- run_graph(function(x) zl_swiglu(x$g, x$u), list(g = c(6, 4), u = c(6, 4)), list(g = g, u = u))
  expect_f32(out$output, g / (1 + exp(-g)) * u)
})

test_that("crossprod is crossprod, batched with broadcasting", {
  withr::local_seed(14)
  a <- rand_array(c(5, 3))
  b <- rand_array(c(5, 4))
  out <- run_graph(function(x) zl_crossprod(x$a, x$b), list(a = c(5, 3), b = c(5, 4)), list(a = a, b = b))
  expect_f32(out$output, crossprod(a, b))
  # batched: a has 1 or 2 batches, b has 2 x 3
  for (ab in list(c(1, 1), c(2, 1), c(2, 3))) {
    da <- c(4, 3, ab)
    db <- c(4, 5, 2, 3)
    a <- rand_array(da)
    b <- rand_array(db)
    out <- run_graph(function(x) zl_crossprod(x$a, x$b), list(a = da, b = db), list(a = a, b = b))
    want <- array(0, c(3, 5, 2, 3))
    for (i in 1:2) for (j in 1:3) {
      want[, , i, j] <- crossprod(
        array(a, dim4(da))[, , (i - 1) %% ab[1] + 1, (j - 1) %% ab[2] + 1], b[, , i, j]
      )
    }
    expect_f32(out$output, want)
  }
})

test_that("crossprod takes f16 and bf16 weights", {
  withr::local_seed(15)
  x <- rand_array(c(8, 3))
  for (type in c("f16", "bf16")) {
    p <- zl_params()
    zl_param(p, "w", c(8, 5), type = type, init = zl_init_uniform(1))
    out <- run_graph(function(x) zl_crossprod(zl_param_tensor(p, "w"), x$x), list(x = c(8, 3)), list(x = x), p = p)
    w <- zl_param_get(p, "w")
    expect_f32(out$output, crossprod(w, x), tol = if (type == "f16") 2e-3 else 1e-2)
  }
})

test_that("reshape, permute, transpose, cont and view match R", {
  withr::local_seed(16)
  x <- rand_array(c(2, 3, 4))
  out <- run_graph(
    function(x) list(
      re = zl_reshape(x$x, c(6, 4)),
      p1 = zl_permute(x$x, c(3, 1, 2)),
      p2 = zl_permute(x$x, c(2, 3, 1)),
      p3 = zl_permute(x$x, c(1, 3, 2, 4)),
      t = zl_t(zl_reshape(x$x, c(6, 4))),
      c = zl_cont(zl_permute(x$x, c(2, 1, 3))),
      v = zl_view(x$x, c(3, 2), offset = 5),
      pr = zl_reshape(zl_permute(x$x, c(3, 2, 1)), 24)
    ),
    list(x = c(2, 3, 4)), list(x = x)
  )
  expect_identical(out$re, array(x, c(6, 4)))
  expect_identical(out$p1, aperm(x, c(3, 1, 2)))
  expect_identical(out$p2, aperm(x, c(2, 3, 1)))
  expect_identical(out$p3, array(aperm(array(x, c(2, 3, 4, 1)), c(1, 3, 2, 4)), c(2, 4, 3, 1)))
  expect_identical(out$t, t(array(x, c(6, 4))))
  expect_identical(out$c, aperm(x, c(2, 1, 3)))
  expect_identical(out$v, array(as.vector(x)[6:11], c(3, 2)))
  expect_identical(out$pr, as.vector(aperm(x, c(3, 2, 1))))
  for (perm in list(c(4, 3, 2, 1), c(2, 4, 1, 3), c(3, 1, 4, 2))) {
    y <- rand_array(c(2, 3, 4, 5))
    o <- run_graph(function(x) zl_permute(x$y, perm), list(y = c(2, 3, 4, 5)), list(y = y))
    expect_identical(o$output, aperm(y, perm))
  }
})

test_that("concat, pad and get_rows match R", {
  withr::local_seed(17)
  a <- rand_array(c(3, 2, 2))
  for (axis in 1:3) {
    db <- c(3, 2, 2)
    db[axis] <- 4
    b <- rand_array(db)
    out <- run_graph(function(x) zl_concat(x$a, x$b, axis), list(a = c(3, 2, 2), b = db), list(a = a, b = b))
    want <- switch(axis,
      {
        w <- array(0, c(7, 2, 2)); w[1:3, , ] <- a; w[4:7, , ] <- b; w
      },
      {
        w <- array(0, c(3, 6, 2)); w[, 1:2, ] <- a; w[, 3:6, ] <- b; w
      },
      {
        w <- array(0, c(3, 2, 6)); w[, , 1:2] <- a; w[, , 3:6] <- b; w
      }
    )
    expect_identical(out$output, want)
  }
  x <- rand_array(c(3, 2))
  out <- run_graph(function(x) zl_pad(x$x, c(2, 1)), list(x = c(3, 2)), list(x = x))
  want <- matrix(0, 5, 3)
  want[1:3, 1:2] <- x
  expect_identical(out$output, want)
  tab <- rand_array(c(4, 6))
  idx <- c(5L, 0L, 2L, 2L)
  out <- run_graph(function(x) zl_get_rows(x$tab, x$idx), list(tab = c(4, 6), idx = zl_int(4)), list(tab = tab, idx = idx))
  expect_identical(out$output, tab[, idx + 1L])
})

test_that("where, clamp and tri match R", {
  withr::local_seed(18)
  m <- matrix(rbinom(12, 1, 0.5), 3, 4)
  a <- rand_array(c(3, 4))
  b <- rand_array(c(3, 4))
  x <- rand_array(c(4, 4, 2), -3, 3)
  out <- run_graph(
    function(x) list(
      w = zl_where(x$m, x$a, x$b), c = zl_clamp(x$x, -1, 1.5),
      lo = zl_tri(x$x, "lower"), up = zl_tri(x$x, "upper"),
      los = zl_tri(x$x, "lower_strict"), ups = zl_tri(x$x, "upper_strict")
    ),
    list(m = c(3, 4), a = c(3, 4), b = c(3, 4), x = c(4, 4, 2)),
    list(m = m, a = a, b = b, x = x)
  )
  expect_f32(out$w, ifelse(m == 1, a, b))
  expect_identical(out$c, pmin(pmax(x, -1), 1.5))
  tri <- function(x, keep) {
    for (k in 1:2) x[, , k][!keep] <- 0
    x
  }
  expect_identical(out$lo, tri(x, lower.tri(x[, , 1], diag = TRUE)))
  expect_identical(out$up, tri(x, upper.tri(x[, , 1], diag = TRUE)))
  expect_identical(out$los, tri(x, lower.tri(x[, , 1])))
  expect_identical(out$ups, tri(x, upper.tri(x[, , 1])))
})

test_that("softmax matches R with masks of every broadcast shape", {
  withr::local_seed(19)
  d <- c(5, 4, 3, 2)
  x <- rand_array(d)
  for (md in list(c(5, 4, 1, 1), c(5, 4, 3, 1), c(5, 4, 1, 2), c(5, 4, 3, 2))) {
    mask <- array(ifelse(runif(prod(md)) < 0.2, -Inf, rnorm(prod(md))), md)
    mask[1, , , ] <- 0 # no column entirely masked
    out <- run_graph(
      function(x) zl_softmax(x$x, x$m, scale = 0.5),
      list(x = d, m = md), list(x = x, m = mask)
    )
    expect_f32(out$output, softmax_cols(0.5 * x + bcast(mask, d)))
  }
  out <- run_graph(function(x) zl_softmax(x$x), list(x = d), list(x = x))
  expect_f32(out$output, softmax_cols(x))
})

test_that("causal_mask hides every key after its query", {
  x <- matrix(1, 4, 4)
  out <- run_graph(function(x) zl_causal_mask(x$x), list(x = c(4, 4)), list(x = x))
  want <- x
  want[row(want) > col(want)] <- -Inf
  expect_identical(out$output, want)
})

rope_r <- function(x, pos, n_dims, base, mode) {
  d <- dim(x)
  out <- x
  half <- n_dims / 2
  for (t in seq_len(d[3])) {
    p <- pos[t]
    for (i in 0:(half - 1)) {
      theta <- p * base^(-2 * i / n_dims)
      if (mode == "interleaved") {
        j0 <- 2 * i + 1
        j1 <- 2 * i + 2
      } else {
        j0 <- i + 1
        j1 <- i + 1 + half
      }
      a <- x[j0, , t, ]
      b <- x[j1, , t, ]
      out[j0, , t, ] <- a * cos(theta) - b * sin(theta)
      out[j1, , t, ] <- a * sin(theta) + b * cos(theta)
    }
  }
  out
}

test_that("rope matches both conventions at full and partial n_dims", {
  withr::local_seed(20)
  d <- c(8, 3, 5, 2)
  x <- rand_array(d)
  pos <- c(0L, 3L, 1L, 7L, 100L)
  for (mode in c("interleaved", "half")) {
    for (n_dims in c(8, 4, 2)) {
      out <- run_graph(
        function(x) zl_rope(x$x, x$pos, n_dims = n_dims, base = 500, mode = mode),
        list(x = d, pos = zl_int(5)), list(x = x, pos = pos)
      )
      expect_f32(out$output, rope_r(x, pos, n_dims, 500, mode), tol = 1e-4)
    }
  }
})

test_that("rms_norm and layer_norm normalise each column", {
  withr::local_seed(21)
  x <- rand_array(c(6, 4))
  out <- run_graph(
    function(x) list(rms = zl_rms_norm(x$x, 1e-5), ln = zl_layer_norm(x$x, 1e-5)),
    list(x = c(6, 4)), list(x = x)
  )
  rms <- sweep(x, 2, sqrt(colMeans(x^2) + 1e-5), "/")
  cen <- sweep(x, 2, colMeans(x))
  ln <- sweep(cen, 2, sqrt(colMeans(cen^2) + 1e-5), "/")
  expect_f32(out$rms, rms, tol = 1e-4)
  expect_f32(out$ln, ln, tol = 1e-4)
})

test_that("losses match R", {
  withr::local_seed(22)
  logits <- rand_array(c(5, 3))
  prob <- softmax_cols(rand_array(c(5, 3)))
  pred <- rand_array(c(4, 2))
  target <- rand_array(c(4, 2))
  out <- run_graph(
    function(x) list(ce = zl_cross_entropy(x$l, x$p), mse = zl_mse(x$a, x$b), mae = zl_mae(x$a, x$b)),
    list(l = c(5, 3), p = c(5, 3), a = c(4, 2), b = c(4, 2)),
    list(l = logits, p = array(f32(prob), dim(prob)), a = pred, b = target)
  )
  logsm <- log(softmax_cols(logits))
  expect_f32(out$ce, -sum(f32(prob) * logsm) / ncol(logits), tol = 1e-4)
  expect_f32(out$mse, mean((pred - target)^2))
  expect_f32(out$mae, mean(abs(pred - target)))
})

test_that("cast converts between f32, f16 and bf16", {
  x <- f32(c(1 / 3, 1000.123, -2.5e-3, 70000))
  out <- run_graph(
    function(x) list(h = zl_cast(zl_cast(x$x, "f16"), "f32"), b = zl_cast(zl_cast(x$x, "bf16"), "f32")),
    list(x = 4), list(x = x)
  )
  p <- zl_params()
  zl_param(p, "h", 4, type = "f16", init = zl_init_zeros())
  zl_param(p, "b", 4, type = "bf16", init = zl_init_zeros())
  expect_warning(zl_param_set(p, "h", x), class = "zulm_range_warning")
  zl_param_set(p, "b", x)
  expect_identical(out$h, zl_param_get(p, "h"))
  expect_identical(out$b, zl_param_get(p, "b"))
})

test_that("parameters, inputs and views can be outputs", {
  p <- zl_params()
  zl_param(p, "w", c(2, 3), init = zl_init_ones())
  out <- run_graph(
    function(x) list(w = zl_param_tensor(p, "w"), a = x$a, t = zl_t(x$a)),
    list(a = c(2, 3)), list(a = matrix(1:6, 2)), p = p
  )
  expect_identical(out$w, matrix(1, 2, 3))
  expect_identical(out$a, matrix(as.double(1:6), 2))
  expect_identical(out$t, t(matrix(as.double(1:6), 2)))
})
