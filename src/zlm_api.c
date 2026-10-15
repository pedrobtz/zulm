/*
 * zlm_api.c: the C table of design §8, inst/include/zulm.h.
 *
 * Each variant has its own copy of the table. R/load.R calls
 * zlm_api_register() in the variant it loads, which registers that copy as
 * zulm's "zulm_api_v1", so a consumer that fetches the table afterwards
 * reaches the loaded variant. Handles are checked against the variant: a
 * store made under another is ZULM_ESTALE, as zlm_check_params() makes it
 * in R.
 *
 * No member raises an R error. Failures are return codes, with the text
 * in a buffer of the calling thread that message() returns.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-version.h"
#include "zlm_glue.h"
#include "zlm_graph.h"
#include "zlm_pool.h"
#include "zlm_rng.h"
#include "zlm_store.h"
#include "zlm_arrow.h"

#include "../inst/include/zulm.h"

#if defined(_MSC_VER)
#define ZLM_TLS __declspec(thread)
#else
#define ZLM_TLS _Thread_local
#endif

static ZLM_TLS char zlm_api_msg[1024];

static const char *api_message(void) {
  return zlm_api_msg;
}

static int fail(int code, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(zlm_api_msg, sizeof zlm_api_msg, fmt, ap);
  va_end(ap);
  return code;
}

static int ok(void) {
  zlm_api_msg[0] = '\0';
  return ZULM_OK;
}

static int zulm_type(enum ggml_type t) {
  switch (t) {
  case GGML_TYPE_F32: return ZULM_TYPE_F32;
  case GGML_TYPE_F16: return ZULM_TYPE_F16;
  case GGML_TYPE_BF16: return ZULM_TYPE_BF16;
  case GGML_TYPE_I32: return ZULM_TYPE_I32;
  default: return -1;
  }
}

static const char *api_ggml_version(void) {
  static char v[256] = "";
  if (v[0] == '\0') {
    /* "0.26.0+p1,p2,...": the patch numbers of ZULM_GGML_PATCHES, whose
       files are named pNN-<what>.patch */
    size_t n = (size_t) snprintf(v, sizeof v, "%s", ggml_version());
    const char *start = ZULM_GGML_PATCHES, *p = start;
    int first = 1;
    while (*p != '\0' && n + 8 < sizeof v) {
      if (*p == 'p' && (p == start || p[-1] == ' ' || p[-1] == ',') &&
          p[1] >= '0' && p[1] <= '9') {
        int num = 0;
        for (p++; *p >= '0' && *p <= '9'; p++) num = 10 * num + (*p - '0');
        n += (size_t) snprintf(v + n, sizeof v - n, "%sp%d", first ? "+" : ",", num);
        first = 0;
      } else {
        p++;
      }
    }
  }
  return v;
}

/* ---- handles ------------------------------------------------------- */

static SEXP list_get(SEXP x, const char *name) {
  if (TYPEOF(x) != VECSXP) return R_NilValue;
  SEXP nm = Rf_getAttrib(x, R_NamesSymbol);
  for (R_xlen_t i = 0; i < Rf_xlength(nm); i++) {
    if (strcmp(CHAR(STRING_ELT(nm, i)), name) == 0) return VECTOR_ELT(x, i);
  }
  return R_NilValue;
}

/* The store of a zl_params object made by this variant, or NULL. */
static zlm_store *store_of(SEXP p) {
  if (!Rf_inherits(p, "zl_params")) {
    fail(ZULM_EARGUMENT, "Expected a parameter store from zl_params().");
    return NULL;
  }
  SEXP variant = list_get(p, "variant");
  if (TYPEOF(variant) != STRSXP || Rf_xlength(variant) != 1 ||
      strcmp(CHAR(STRING_ELT(variant, 0)), zlm_variant_name) != 0) {
    fail(ZULM_ESTALE, "This store was made by another zulm variant than the '%s' variant here.",
         zlm_variant_name);
    return NULL;
  }
  zlm_store *s = zlm_store_peek(list_get(p, "ptr"));
  if (s == NULL) fail(ZULM_ESTALE, "This parameter store is no longer valid.");
  return s;
}

static zulm_store *api_store_from_sexp(SEXP p) {
  zlm_store *s = store_of(p);
  if (s != NULL) ok();
  return (zulm_store *) s;
}

static zulm_graph *api_graph_from_sexp(SEXP g) {
  if (!Rf_inherits(g, "zl_graph")) {
    fail(ZULM_EARGUMENT, "Expected a graph from zl_graph().");
    return NULL;
  }
  if (store_of(list_get(g, "store")) == NULL) return NULL;
  zlm_graph *gr = zlm_graph_peek(list_get(g, "ptr"));
  if (gr == NULL || gr->state != ZLM_GRAPH_READY) {
    fail(ZULM_ESTALE, "This graph is no longer valid, or failed to build.");
    return NULL;
  }
  ok();
  return (zulm_graph *) gr;
}

#define GRAPH(g) ((zlm_graph *) (g))
#define STORE(p) ((zlm_store *) (p))

static struct ggml_tensor *find_input(const zlm_graph *g, const char *name) {
  for (int i = 0; i < g->n_in; i++) {
    if (strcmp(g->in[i]->name, name) == 0) return g->in[i];
  }
  return NULL;
}

static int check_graph(const zulm_graph *g, const char *name) {
  if (g == NULL || name == NULL) return fail(ZULM_EARGUMENT, "A NULL graph or name.");
  if (GRAPH(g)->state != ZLM_GRAPH_READY) return fail(ZULM_ESTALE, "This graph is not ready.");
  return ZULM_OK;
}

static int check_count(const struct ggml_tensor *t, const void *buf, int64_t n, const char *what) {
  if (n < 0 || (n > 0 && buf == NULL)) return fail(ZULM_EARGUMENT, "A NULL buffer or a negative count.");
  if (n != ggml_nelements(t)) {
    return fail(ZULM_EMISMATCH, "%s '%s' holds %lld values, not %lld.", what, t->name,
                (long long) ggml_nelements(t), (long long) n);
  }
  return ZULM_OK;
}

/* ---- inputs and outputs -------------------------------------------- */

static int api_input_shape(const zulm_graph *g, const char *name, int64_t ne[4], int *type) {
  int rc = check_graph(g, name);
  if (rc != ZULM_OK) return rc;
  if (ne == NULL) return fail(ZULM_EARGUMENT, "A NULL shape.");
  const struct ggml_tensor *t = find_input(GRAPH(g), name);
  if (t == NULL) return fail(ZULM_EUNKNOWN, "The graph has no input '%s'.", name);
  for (int k = 0; k < 4; k++) ne[k] = t->ne[k];
  if (type != NULL) *type = zulm_type(t->type);
  return ok();
}

static int input_write(zulm_graph *g, const char *name, const void *src, int64_t n, enum ggml_type type) {
  int rc = check_graph(g, name);
  if (rc != ZULM_OK) return rc;
  struct ggml_tensor *t = find_input(GRAPH(g), name);
  if (t == NULL) return fail(ZULM_EUNKNOWN, "The graph has no input '%s'.", name);
  if (t->type != type) {
    return fail(ZULM_EMISMATCH, "Input '%s' is %s.", name, t->type == GGML_TYPE_I32 ? "i32" : "f32");
  }
  rc = check_count(t, src, n, "Input");
  if (rc != ZULM_OK) return rc;
  if (n > 0) memcpy(t->data, src, (size_t) n * ggml_type_size(type));
  return ok();
}

static int api_input_write_f32(zulm_graph *g, const char *name, const float *src, int64_t n) {
  return input_write(g, name, src, n, GGML_TYPE_F32);
}

static int api_input_write_i32(zulm_graph *g, const char *name, const int32_t *src, int64_t n) {
  return input_write(g, name, src, n, GGML_TYPE_I32);
}

static int api_input_write_f32_at(zulm_graph *g, const char *name, int64_t offset, const float *src,
                                  int64_t n) {
  int rc = check_graph(g, name);
  if (rc != ZULM_OK) return rc;
  rc = zlm_api_input_write_f32_at(GRAPH(g), name, offset, src, n);
  switch (rc) {
  case 0: return ok();
  case 1: return fail(ZULM_EUNKNOWN, "The graph has no input '%s'.", name);
  case 2: return fail(ZULM_EMISMATCH, "Input '%s' is not f32.", name);
  default:
    return fail(ZULM_ERANGE, "Writing %lld values at offset %lld reaches outside input '%s'.",
                (long long) n, (long long) offset, name);
  }
}

static int api_output_read_f32(const zulm_graph *g, const char *name, float *dst, int64_t n) {
  int rc = check_graph(g, name);
  if (rc != ZULM_OK) return rc;
  const zlm_graph *gr = GRAPH(g);
  for (int i = 0; i < gr->n_out && i < gr->n_out_names; i++) {
    if (strcmp(gr->out_names[i], name) == 0) {
      const struct ggml_tensor *t = gr->out_fwd[i];
      rc = check_count(t, dst, n, "Output");
      if (rc != ZULM_OK) return rc;
      if (n > 0) memcpy(dst, t->data, (size_t) n * sizeof(float));
      return ok();
    }
  }
  return fail(ZULM_EUNKNOWN, "The graph has no output '%s'.", name);
}

/* ---- parameters and state ------------------------------------------ */

static struct ggml_tensor *find_param(const zlm_store *s, const char *name) {
  for (int i = 0; i < s->n; i++) {
    if (s->kinds[i] == ZLM_KIND_PARAM && strcmp(s->tensors[i]->name, name) == 0) return s->tensors[i];
  }
  return NULL;
}

static int param_f32(const zulm_store *p, const char *name, const void *buf, int64_t n,
                     struct ggml_tensor **out) {
  if (p == NULL || name == NULL) return fail(ZULM_EARGUMENT, "A NULL store or name.");
  struct ggml_tensor *t = find_param(STORE(p), name);
  if (t == NULL) return fail(ZULM_EUNKNOWN, "The store has no parameter '%s'.", name);
  if (t->type != GGML_TYPE_F32) return fail(ZULM_EMISMATCH, "Parameter '%s' is not f32.", name);
  int rc = check_count(t, buf, n, "Parameter");
  if (rc != ZULM_OK) return rc;
  *out = t;
  return ZULM_OK;
}

static int api_param_read_f32(const zulm_store *p, const char *name, float *dst, int64_t n) {
  struct ggml_tensor *t = NULL;
  int rc = param_f32(p, name, dst, n, &t);
  if (rc != ZULM_OK) return rc;
  if (n > 0) memcpy(dst, t->data, (size_t) n * sizeof(float));
  return ok();
}

static int api_param_write_f32(zulm_store *p, const char *name, const float *src, int64_t n) {
  struct ggml_tensor *t = NULL;
  int rc = param_f32(p, name, src, n, &t);
  if (rc != ZULM_OK) return rc;
  if (n > 0) memcpy(t->data, src, (size_t) n * sizeof(float));
  return ok();
}

static int state_code(int rc, const char *name, const char *what) {
  switch (rc) {
  case 0: return ok();
  case 1: return fail(ZULM_EUNKNOWN, "The store has no state '%s'.", name);
  case 2: return fail(ZULM_EMISMATCH, "State '%s' is not f32, or %s.", name, what);
  default: return fail(ZULM_ERANGE, "The write reaches outside state '%s'.", name);
  }
}

static int api_state_shape(const zulm_store *p, const char *name, int64_t ne[4], int *type) {
  if (p == NULL || name == NULL || ne == NULL) return fail(ZULM_EARGUMENT, "A NULL store, name or shape.");
  int t = 0;
  int rc = zlm_api_state_shape(STORE(p), name, ne, &t);
  if (rc == 0 && type != NULL) *type = zulm_type((enum ggml_type) t);
  return state_code(rc, name, "");
}

static int api_state_read_f32(const zulm_store *p, const char *name, float *dst, int64_t n) {
  if (p == NULL || name == NULL) return fail(ZULM_EARGUMENT, "A NULL store or name.");
  return state_code(zlm_api_state_read_f32(STORE(p), name, dst, n), name,
                    "the count is not its number of values");
}

static int api_state_write_f32_at(zulm_store *p, const char *name, int64_t offset, const float *src,
                                  int64_t n) {
  if (p == NULL || name == NULL) return fail(ZULM_EARGUMENT, "A NULL store or name.");
  return state_code(zlm_api_state_write_f32_at(STORE(p), name, offset, src, n), name, "");
}

/* ---- running ------------------------------------------------------- */

static int api_compute(zulm_graph *g) {
  if (g == NULL) return fail(ZULM_EARGUMENT, "A NULL graph.");
  zlm_graph *gr = GRAPH(g);
  if (gr->state != ZLM_GRAPH_READY) return fail(ZULM_ESTALE, "This graph is not ready.");
  int rc = zlm_graph_index_status(gr, zlm_api_msg, sizeof zlm_api_msg);
  if (rc != 0) return rc;
  zlm_prepare_backend(gr->backend);
  enum ggml_status st = ggml_backend_graph_compute(gr->backend, gr->gf_fwd);
  if (zlm_interrupted()) return fail(ZULM_EINTERRUPT, "The computation was interrupted.");
  if (st != GGML_STATUS_SUCCESS) return fail(ZULM_EFAILED, "ggml failed to compute the graph (status %d).", (int) st);
  return ok();
}

struct fill_job {
  zlm_graph *g;
  SEXP it;
  int rc;
};

static SEXP fill_body(void *data) {
  struct fill_job *j = (struct fill_job *) data;
  j->rc = zlm_api_batch_fill(j->g, j->it);
  return R_NilValue;
}

static SEXP fill_handler(SEXP cond, void *data) {
  struct fill_job *j = (struct fill_job *) data;
  SEXP msg = list_get(cond, "message");
  fail(ZULM_EFAILED, "%s", TYPEOF(msg) == STRSXP && Rf_xlength(msg) > 0 ? CHAR(STRING_ELT(msg, 0)) : "Reading the batch failed.");
  j->rc = ZULM_EFAILED;
  return R_NilValue;
}

static int api_batch_fill(zulm_graph *g, SEXP iterator) {
  if (g == NULL) return fail(ZULM_EARGUMENT, "A NULL graph.");
  if (GRAPH(g)->state != ZLM_GRAPH_READY) return fail(ZULM_ESTALE, "This graph is not ready.");
  SEXP it = Rf_inherits(iterator, "zl_arrow_batches") ? list_get(iterator, "ptr") : iterator;
  struct fill_job j = { GRAPH(g), it, ZULM_EFAILED };
  /* an Arrow error is an R condition with its column and row; here it
     becomes a code and its message */
  R_tryCatchError(fill_body, &j, fill_handler, &j);
  if (j.rc == ZULM_EFAILED) return ZULM_EFAILED;
  if (j.rc == 1) {
    fail(ZULM_END, "The stream has ended.");
    return ZULM_END;
  }
  return ok();
}

/* ---- draws --------------------------------------------------------- */

static int api_rng_stream(const char *name) {
  int rc = zlm_rng_stream_register(name, zlm_api_msg, sizeof zlm_api_msg);
  return rc == 0 ? ok() : rc;
}

static int api_rng_uniform_f32(const char *name, uint32_t step, uint64_t offset, float *dst, int64_t n) {
  int rc = zlm_rng_stream_draw(name, step, offset, ZLM_DIST_UNIFORM, dst, n, zlm_api_msg, sizeof zlm_api_msg);
  return rc == 0 ? ok() : rc;
}

static int api_rng_normal_f32(const char *name, uint32_t step, uint64_t offset, float *dst, int64_t n) {
  int rc = zlm_rng_stream_draw(name, step, offset, ZLM_DIST_NORMAL, dst, n, zlm_api_msg, sizeof zlm_api_msg);
  return rc == 0 ? ok() : rc;
}

/* ---- the table ----------------------------------------------------- */

static const zulm_api zlm_api_table = {
  sizeof(zulm_api),
  api_ggml_version,
  api_message,
  api_graph_from_sexp,
  api_store_from_sexp,
  api_input_shape,
  api_input_write_f32,
  api_input_write_i32,
  api_output_read_f32,
  api_param_read_f32,
  api_param_write_f32,
  api_compute,
  api_rng_stream,
  api_rng_uniform_f32,
  api_rng_normal_f32,
  api_input_write_f32_at,
  api_state_shape,
  api_state_read_f32,
  api_state_write_f32_at,
  api_batch_fill
};

static const zulm_api *zlm_api_get_table(void) {
  return &zlm_api_table;
}

/* Called by R/load.R in the variant it loads. */
SEXP zlm_api_register(void) {
  R_RegisterCCallable("zulm", "zulm_api_v1", (DL_FUNC) zlm_api_get_table);
  return R_NilValue;
}
