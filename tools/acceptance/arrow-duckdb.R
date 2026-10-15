# Design §15 criterion 12 (Stage 11): the §1.1 model trained from a DuckDB
# table through an Arrow stream gives results identical to training from
# the same rows as R arrays; and from a Parquet file read by DuckDB.
# Run by the acceptance workflow on the installed package:
#
#   Rscript tools/acceptance/arrow-duckdb.R
#
# It needs DBI, duckdb and nanoarrow, which the package does not declare:
# they are this script's, not the package's tests'.
library(testthat)
library(zulm)
source("tests/testthat/helper-zulm.R")
source("tests/testthat/helper-flow.R")

steps <- 12L
n <- 64L
b <- 16L
rows <- n * b

# One row per event, sequence by sequence: the layout c(features, n, b)
# flattened to c(features, n * b), which a graph reshapes back.
events <- function(step) {
  d <- synth_batch(step, n, b)
  flat <- function(a) matrix(a, dim(a)[[1]], rows)
  out <- data.frame(
    flat(d$feat) |> t() |> as.data.frame() |> stats::setNames(paste0("f", 1:8)),
    kind = factor(as.vector(d$kind), levels = 0:3),
    flat(d$kind_prob) |> t() |> as.data.frame() |> stats::setNames(paste0("p", 1:4)),
    t = as.vector(d$t),
    flat(d$y) |> t() |> as.data.frame() |> stats::setNames(paste0("y", 1:7))
  )
  out
}
df <- do.call(rbind, lapply(seq_len(steps), events))
df$rid <- seq_len(nrow(df)) - 1L

as_arrays <- function(step) {
  d <- df[(step - 1L) * rows + seq_len(rows), ]
  m <- function(cols) t(as.matrix(d[cols]))
  list(
    feat = unname(m(paste0("f", 1:8))),
    kind = as.integer(d$kind) - 1L,
    kind_prob = unname(m(paste0("p", 1:4))),
    t = matrix(d$t, 1),
    y = unname(m(paste0("y", 1:7)))
  )
}

map <- list(
  feat = paste0("f", 1:8), kind = zl_int("kind"), kind_prob = paste0("p", 1:4),
  t = "t", y = paste0("y", 1:7)
)

train <- function(next_batch) {
  old <- zl_seed(2026)
  on.exit(zl_seed(old), add = TRUE)
  fm <- flow_model()
  m <- fm$m
  build <- function(x) {
    r <- function(v, f) zl_reshape(v, c(f, n, b))
    feat <- r(x$feat, 8)
    kind <- zl_reshape(x$kind, c(n, b))
    tt <- r(x$t, 1)
    y <- r(x$y, 7)
    z <- m$ln(m$blk(m$emb(feat)))
    ce <- zl_cross_entropy(m$kh(z), r(x$kind_prob, 4))
    eps <- zl_noise_normal(c(7, n, b), "x0")
    x_t <- (1 - tt) * eps + tt * y
    v <- velocity(m, z, kind, x_t, tt)
    list(loss = ce + zl_mse(v, y - eps))
  }
  g <- zl_graph(build, fm$p,
    inputs = list(feat = c(8, rows), kind = zl_int(rows), kind_prob = c(4, rows), t = c(1, rows), y = c(7, rows)),
    loss = "loss"
  )
  tr <- zl_trainer(g, zl_adamw(lr = 1e-3))
  losses <- c()
  repeat {
    d <- next_batch()
    if (is.null(d)) break
    l <- zl_step(tr, d)
    if (is.null(l)) break
    losses <- c(losses, l)
  }
  list(losses = losses,
       params = lapply(stats::setNames(nm = zl_param_names(fm$p)), function(nm) zl_param_get(fm$p, nm)))
}

# from R arrays
i <- 0L
arrays <- train(function() {
  i <<- i + 1L
  if (i > steps) NULL else as_arrays(i)
})

con <- DBI::dbConnect(duckdb::duckdb())
on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
DBI::dbWriteTable(con, "events", df)

test_that("training from a DuckDB table equals training from R arrays", {
  stream <- DBI::dbGetQueryArrow(con, "SELECT * FROM events ORDER BY rid")
  it <- zl_arrow_batches(stream, map, rows = rows)
  duck <- train(function() it)
  expect_length(duck$losses, steps)
  expect_identical(duck$losses, arrays$losses)
  expect_identical(duck$params, arrays$params)
})

test_that("training from a Parquet file equals training from R arrays", {
  path <- tempfile(fileext = ".parquet")
  DBI::dbExecute(con, sprintf("COPY (SELECT * FROM events ORDER BY rid) TO '%s' (FORMAT parquet, ROW_GROUP_SIZE 1000)", path))
  # Parquet keeps the kinds as strings; an ENUM makes them a dictionary again
  DBI::dbExecute(con, "CREATE TYPE kinds AS ENUM ('0', '1', '2', '3')")
  stream <- DBI::dbGetQueryArrow(con, sprintf(
    "SELECT * REPLACE (kind::kinds AS kind) FROM read_parquet('%s') ORDER BY rid", path
  ))
  it <- zl_arrow_batches(stream, map, rows = rows)
  pq <- train(function() it)
  expect_identical(pq$losses, arrays$losses)
  expect_identical(pq$params, arrays$params)
})

cat(sprintf("%d steps from DuckDB and Parquet match the arrays; loss %.4f -> %.4f\n",
            steps, arrays$losses[[1]], arrays$losses[[steps]]))
