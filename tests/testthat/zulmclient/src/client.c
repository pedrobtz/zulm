/* A downstream package driving zulm through its C table, as zuflow's
 * samplers and zukzeit's decode loop would. */
#include <string.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <zulm.h>

#ifdef __has_include
#  if __has_include(<omp.h>)
#    define CLIENT_HAVE_OMP_H 1
#  endif
#else
#  define CLIENT_HAVE_OMP_H 1
#endif
#if defined(_OPENMP) && defined(CLIENT_HAVE_OMP_H)
#  define CLIENT_OPENMP 1
#endif

static const char *str(SEXP x) {
    return Rf_translateCharUTF8(STRING_ELT(x, 0));
}

static void check(const zulm_api *zl, int rc, const char *what) {
    if (rc != ZULM_OK) Rf_error("%s returned %d: %s", what, rc, zl->message());
}

static zulm_graph *graph(const zulm_api *zl, SEXP g) {
    zulm_graph *h = zl->graph_from_sexp(g);
    if (h == NULL) Rf_error("graph_from_sexp: %s", zl->message());
    return h;
}

static zulm_store *store(const zulm_api *zl, SEXP p) {
    zulm_store *h = zl->store_from_sexp(p);
    if (h == NULL) Rf_error("store_from_sexp: %s", zl->message());
    return h;
}

static int64_t count(const int64_t ne[4]) {
    return ne[0] * ne[1] * ne[2] * ne[3];
}

/* The table's size, the size this fixture was compiled against, and the
   ggml version. */
SEXP client_info(void) {
    const zulm_api *zl = zulm_api_get();
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
    SET_VECTOR_ELT(out, 0, Rf_ScalarReal((double) zl->struct_size));
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal((double) sizeof(zulm_api)));
    SET_VECTOR_ELT(out, 2, Rf_mkString(zl->ggml_version()));
    UNPROTECT(1);
    return out;
}

/* A consumer built against a table `extra` bytes longer than zulm's. */
SEXP client_require(SEXP extra) {
    zulm_api_require(sizeof(zulm_api) + (size_t) Rf_asInteger(extra));
    return R_NilValue;
}

/* A sampler loop: at each step, normal draws of the stream fill the
   input, the graph runs, and the output, of `size` values, is read; the
   outputs are the columns of the result. An optional i32 input `index` is written once. */
SEXP client_sampler(SEXP g, SEXP stream, SEXP input, SEXP output, SEXP size, SEXP steps,
                    SEXP index) {
    const zulm_api *zl = zulm_api_get();
    zulm_graph *h = graph(zl, g);
    int64_t ne[4];
    int type = -1;
    check(zl, zl->input_shape(h, str(input), ne, &type), "input_shape");
    if (type != ZULM_TYPE_F32) Rf_error("input is not f32");
    int64_t n = count(ne);
    if (!Rf_isNull(index)) {
        int64_t ni[4];
        check(zl, zl->input_shape(h, "index", ni, &type), "input_shape");
        if (type != ZULM_TYPE_I32 || count(ni) != Rf_xlength(index)) Rf_error("bad index input");
        check(zl, zl->input_write_i32(h, "index", INTEGER(index), count(ni)), "input_write_i32");
    }
    check(zl, zl->rng_stream(str(stream)), "rng_stream");
    float *buf = (float *) R_alloc((size_t) n, sizeof(float));
    int k = Rf_asInteger(steps);
    int64_t m = (int64_t) Rf_asInteger(size);
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int) m, k));
    float *obuf = (float *) R_alloc((size_t) m, sizeof(float));
    for (int s = 1; s <= k; s++) {
        check(zl, zl->rng_normal_f32(str(stream), (uint32_t) s, 0, buf, n), "rng_normal_f32");
        check(zl, zl->input_write_f32(h, str(input), buf, n), "input_write_f32");
        check(zl, zl->compute(h), "compute");
        check(zl, zl->output_read_f32(h, str(output), obuf, m), "output_read_f32");
        for (int64_t i = 0; i < m; i++) REAL(out)[(s - 1) * m + i] = obuf[i];
    }
    UNPROTECT(1);
    return out;
}

/* A weight average: parameter `name` of store a becomes the mean of a's
   and b's; it holds `size` values. */
SEXP client_average(SEXP a, SEXP b, SEXP name, SEXP size) {
    const zulm_api *zl = zulm_api_get();
    zulm_store *pa = store(zl, a), *pb = store(zl, b);
    const char *nm = str(name);
    int64_t n = (int64_t) Rf_asInteger(size);
    float *x = (float *) R_alloc((size_t) n, sizeof(float));
    float *y = (float *) R_alloc((size_t) n, sizeof(float));
    check(zl, zl->param_read_f32(pa, nm, x, n), "param_read_f32");
    check(zl, zl->param_read_f32(pb, nm, y, n), "param_read_f32");
    for (int64_t i = 0; i < n; i++) x[i] = 0.5f * (x[i] + y[i]);
    check(zl, zl->param_write_f32(pa, nm, x, n), "param_write_f32");
    return R_NilValue;
}

/* Writes values into state `name` at an offset, then reads the whole
   state back. */
SEXP client_state(SEXP p, SEXP name, SEXP offset, SEXP values) {
    const zulm_api *zl = zulm_api_get();
    zulm_store *h = store(zl, p);
    const char *nm = str(name);
    int64_t ne[4];
    int type = -1;
    check(zl, zl->state_shape(h, nm, ne, &type), "state_shape");
    if (type != ZULM_TYPE_F32) Rf_error("state is not f32");
    R_xlen_t k = Rf_xlength(values);
    float *src = (float *) R_alloc((size_t) (k > 0 ? k : 1), sizeof(float));
    for (R_xlen_t i = 0; i < k; i++) src[i] = (float) REAL(values)[i];
    check(zl, zl->state_write_f32_at(h, nm, (int64_t) Rf_asReal(offset), src, (int64_t) k),
          "state_write_f32_at");
    int64_t n = count(ne);
    float *dst = (float *) R_alloc((size_t) n, sizeof(float));
    check(zl, zl->state_read_f32(h, nm, dst, n), "state_read_f32");
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) n));
    for (int64_t i = 0; i < n; i++) REAL(out)[i] = dst[i];
    UNPROTECT(1);
    return out;
}

/* Every batch of an Arrow iterator through the graph; one column per
   batch of the output, which holds `size` values. */
SEXP client_batches(SEXP g, SEXP it, SEXP output, SEXP size) {
    const zulm_api *zl = zulm_api_get();
    zulm_graph *h = graph(zl, g);
    SEXP cols = PROTECT(Rf_allocVector(VECSXP, 0));
    int nprot = 1;
    for (;;) {
        int rc = zl->batch_fill(h, it);
        if (rc == ZULM_END) break;
        check(zl, rc, "batch_fill");
        check(zl, zl->compute(h), "compute");
        int64_t m = (int64_t) Rf_asInteger(size);
        float *buf = (float *) R_alloc((size_t) m, sizeof(float));
        check(zl, zl->output_read_f32(h, str(output), buf, m), "output_read_f32");
        SEXP col = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) m));
        for (int64_t i = 0; i < m; i++) REAL(col)[i] = buf[i];
        R_xlen_t len = Rf_xlength(cols);
        SEXP grown = PROTECT(Rf_allocVector(VECSXP, len + 1));
        for (R_xlen_t i = 0; i < len; i++) SET_VECTOR_ELT(grown, i, VECTOR_ELT(cols, i));
        SET_VECTOR_ELT(grown, len, col);
        UNPROTECT(3);
        cols = PROTECT(grown);
    }
    UNPROTECT(nprot);
    return cols;
}

/* The codes of calls that must fail, in a fixed order (see the test). */
SEXP client_codes(SEXP g, SEXP p) {
    const zulm_api *zl = zulm_api_get();
    zulm_graph *h = graph(zl, g);
    zulm_store *s = store(zl, p);
    int64_t ne[4];
    int type;
    float buf[64] = {0};
    int32_t ibuf[4] = {0};
    int codes[] = {
        zl->input_shape(h, "nope", ne, &type),
        zl->input_write_f32(h, "x", buf, 1),
        zl->input_write_i32(h, "x", ibuf, 4),
        zl->input_write_f32_at(h, "x", 5, buf, 4),
        zl->input_write_f32(h, "x", NULL, 4),
        zl->output_read_f32(h, "nope", buf, 1),
        zl->param_read_f32(s, "nope", buf, 1),
        zl->param_read_f32(s, "w", buf, 1),
        zl->param_write_f32(s, "h", buf, 4),
        zl->state_shape(s, "w", ne, &type),
        zl->state_write_f32_at(s, "h", 3, buf, 2),
        zl->rng_normal_f32("never registered", 1, 0, buf, 4),
        zl->rng_uniform_f32("never registered", 1, 0, buf, -1),
        zl->compute(NULL),
        zl->graph_from_sexp(R_NilValue) == NULL ? ZULM_EARGUMENT : ZULM_OK,
        zl->store_from_sexp(g) == NULL ? ZULM_EARGUMENT : ZULM_OK
    };
    int n = (int) (sizeof codes / sizeof codes[0]);
    SEXP out = PROTECT(Rf_allocVector(INTSXP, n));
    for (int i = 0; i < n; i++) INTEGER(out)[i] = codes[i];
    UNPROTECT(1);
    return out;
}

/* Uniform draws of a registered stream, `threads` blocks of n drawn on
   worker threads when OpenMP is there; block t starts at position t * n.
   With `reg` false the stream is not registered first; then the code of
   the first failed draw is returned. */
SEXP client_draw_threads(SEXP stream, SEXP step, SEXP n_, SEXP threads, SEXP reg) {
    const zulm_api *zl = zulm_api_get();
    const char *nm = str(stream);
    if (Rf_asLogical(reg)) check(zl, zl->rng_stream(nm), "rng_stream");
    int n = Rf_asInteger(n_), t = Rf_asInteger(threads);
    uint32_t st = (uint32_t) Rf_asInteger(step);
    float *buf = (float *) R_alloc((size_t) n * (size_t) t, sizeof(float));
    int failed = 0;
#ifdef CLIENT_OPENMP
#pragma omp parallel for reduction(|:failed) num_threads(t) schedule(static, 1)
#endif
    for (int j = 0; j < t; j++) {
        failed |= zl->rng_uniform_f32(nm, st, (uint64_t) j * (uint64_t) n, buf + (size_t) j * (size_t) n, n);
    }
    if (failed) return Rf_ScalarInteger(failed);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) n * t));
    for (R_xlen_t i = 0; i < (R_xlen_t) n * t; i++) REAL(out)[i] = buf[i];
    UNPROTECT(1);
    return out;
}

static const R_CallMethodDef methods[] = {
    {"C_client_info", (DL_FUNC) &client_info, 0},
    {"C_client_require", (DL_FUNC) &client_require, 1},
    {"C_client_sampler", (DL_FUNC) &client_sampler, 7},
    {"C_client_average", (DL_FUNC) &client_average, 4},
    {"C_client_state", (DL_FUNC) &client_state, 4},
    {"C_client_batches", (DL_FUNC) &client_batches, 4},
    {"C_client_codes", (DL_FUNC) &client_codes, 2},
    {"C_client_draw_threads", (DL_FUNC) &client_draw_threads, 5},
    {NULL, NULL, 0}
};

void R_init_zulmclient(DllInfo *dll) {
    R_registerRoutines(dll, NULL, methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
