/*
 * zlm_rng.c: random numbers from zurand, and per-row loss sums (design
 * §7.1, §7.2).
 *
 * Every draw is a position of a zurand stream: the stream key is the
 * session key folded with a name (in R), the step key that folded with the
 * step number (fold_int, here), and element i is position offset + i.
 * zurand returns doubles; zulm rounds them to f32 by one rule: uniforms
 * towards zero, so they stay inside (0, 1), normals to nearest.
 *
 * Per-row losses: a forward-only custom operation writes one value per row
 * (each row on one thread, in double) into the graph's output and into a
 * buffer of the operation's own, which the main thread adds in a fixed
 * pairwise order after the computation. The operation feeds the loss only
 * through ggml_step(), whose source ggml never differentiates, so it is
 * computed inside ggml-opt's graphs without a gradient of its own.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-threading.h"
#include "zlm_glue.h"
#include "zlm_graph.h"
#include "zlm_r.h"
#include "zlm_rng.h"
#include "zurand.h"

static const zurand_api *zlm_zr = NULL;

/* Looks up zurand's C table; returns its version, or -1 when the table is
   missing. The R side raises zulm_companion_error below version 3. */
SEXP zlm_zurand_init(void) {
  typedef const zurand_api *(*getter)(void);
  getter get = (getter) R_GetCCallable("zurand", "zurand_api");
  if (get == NULL) return Rf_ScalarInteger(-1);
  const zurand_api *api = get();
  if (api == NULL) return Rf_ScalarInteger(-1);
  if (api->version >= ZURAND_API_VERSION && api->size >= sizeof(zurand_api)) {
    zlm_zr = api;
  }
  return Rf_ScalarInteger(api->version);
}

static const zurand_api *zr(void) {
  if (zlm_zr == NULL) {
    zlm_raise("zulm_companion_error", R_NilValue,
              "zurand's C table (version %d or later) is not available.", ZURAND_API_VERSION);
  }
  return zlm_zr;
}

zurand_key zlm_key_from_r(SEXP key) {
  zurand_key k;
  if (zr()->key_get(key, 0, &k) != ZURAND_OK) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "Expected a zurand rng_key.");
  }
  return k;
}

static float round_uniform(double x) {
  /* towards zero: a uniform just below 1 must not round up to 1 */
  float f = (float) x;
  if (fabs((double) f) > fabs(x)) f = nextafterf(f, 0.0f);
  return f;
}

/* Positions [start, start + n) of the stream `k` at step `step`, rounded to
   f32 into out. Calls no R API: safe on any thread. Returns a zurand code. */
static int draw_f32(const zurand_key *k, int64_t step, int dist, uint64_t start, size_t n,
                    float *out) {
  zurand_key sk;
  int rc = zlm_zr->fold_int(*k, step, &sk);
  if (rc != ZURAND_OK) return rc;
  double buf[512];
  size_t done = 0;
  while (done < n) {
    size_t m = n - done < 512 ? n - done : 512;
    if (dist == ZLM_DIST_UNIFORM) {
      rc = zlm_zr->fill_uniform_at(sk, start + done, m, 0.0, 1.0, buf);
      if (rc != ZURAND_OK) return rc;
      for (size_t i = 0; i < m; i++) out[done + i] = round_uniform(buf[i]);
    } else {
      rc = zlm_zr->fill_normal_at(sk, start + done, m, 0.0, 1.0, ZURAND_NORMAL_ZIGGURAT, buf);
      if (rc != ZURAND_OK) return rc;
      for (size_t i = 0; i < m; i++) out[done + i] = (float) buf[i];
    }
    done += m;
  }
  return ZURAND_OK;
}

/* zl_rng_uniform() and zl_rng_normal(): n f32 values as doubles. */
SEXP zlm_rng_draw(SEXP stream_key, SEXP step, SEXP offset, SEXP n, SEXP dist) {
  zurand_key k = zlm_key_from_r(stream_key);
  double nn = Rf_asReal(n);
  R_xlen_t len = (R_xlen_t) nn;
  float *tmp = (float *) R_alloc((size_t) (len > 0 ? len : 1), sizeof(float));
  int rc = draw_f32(&k, (int64_t) Rf_asReal(step), Rf_asInteger(dist), (uint64_t) Rf_asReal(offset),
                    (size_t) len, tmp);
  if (rc != ZURAND_OK) {
    zlm_raise("zulm_internal_error", R_NilValue, "zurand refused the draw (code %d).", rc);
  }
  SEXP out = PROTECT(Rf_allocVector(REALSXP, len));
  for (R_xlen_t i = 0; i < len; i++) REAL(out)[i] = tmp[i];
  UNPROTECT(1);
  return out;
}

/* Internal, for the version test: the check .onLoad applies. */
SEXP zlm_zurand_version_ok(SEXP version) {
  return Rf_ScalarLogical(Rf_asInteger(version) >= ZURAND_API_VERSION);
}

/* ---- named streams of the C table (design §8) ------------------------- */

/* The keys of the streams a consumer registered, by name. Registration is
   on the main thread; draws read the list from any thread, so both hold
   ggml's critical section while they touch it. */
typedef struct {
  char *name;
  zurand_key key;
} zlm_stream;

static zlm_stream *zlm_streams = NULL;
static int zlm_n_streams = 0, zlm_cap_streams = 0;

static int find_stream(const char *name) {
  for (int i = 0; i < zlm_n_streams; i++) {
    if (strcmp(zlm_streams[i].name, name) == 0) return i;
  }
  return -1;
}

/* Folds the session key with `name` in R (zlm_stream_key()) and records
   the stream's key. Main thread. 0, or a ZULM_E* code with msg set. */
int zlm_rng_stream_register(const char *name, char *msg, size_t cap) {
  if (name == NULL) {
    snprintf(msg, cap, "The stream's name is NULL.");
    return 7;
  }
  if (zlm_zr == NULL) {
    snprintf(msg, cap, "zurand's C table (version %d or later) is not available.", ZURAND_API_VERSION);
    return 5;
  }
  SEXP ns = PROTECT(R_FindNamespace(PROTECT(Rf_mkString("zulm"))));
  SEXP arg = PROTECT(Rf_mkString(name));
  SEXP call = PROTECT(Rf_lang2(Rf_install("zlm_stream_key"), arg));
  int err = 0;
  SEXP key = PROTECT(R_tryEvalSilent(call, ns, &err));
  zurand_key k;
  int bad = err || zlm_zr->key_get(key, 0, &k) != ZURAND_OK;
  UNPROTECT(5);
  if (bad) {
    snprintf(msg, cap, "The key of stream '%s' could not be made.", name);
    return 5;
  }
  size_t len = strlen(name) + 1;
  int rc = 0;
  ggml_critical_section_start();
  int i = find_stream(name);
  if (i >= 0) {
    zlm_streams[i].key = k;
  } else {
    if (zlm_n_streams == zlm_cap_streams) {
      int cap2 = zlm_cap_streams > 0 ? 2 * zlm_cap_streams : 16;
      zlm_stream *s = (zlm_stream *) realloc(zlm_streams, (size_t) cap2 * sizeof *s);
      if (s != NULL) {
        zlm_streams = s;
        zlm_cap_streams = cap2;
      }
    }
    char *copy = zlm_n_streams < zlm_cap_streams ? (char *) malloc(len) : NULL;
    if (copy == NULL) {
      rc = 5;
    } else {
      memcpy(copy, name, len);
      zlm_streams[zlm_n_streams].name = copy;
      zlm_streams[zlm_n_streams].key = k;
      zlm_n_streams++;
    }
  }
  ggml_critical_section_end();
  if (rc != 0) snprintf(msg, cap, "Out of memory registering stream '%s'.", name);
  return rc;
}

/* Positions [offset, offset + n) of a registered stream at `step`, as f32.
   Any thread, no R calls. 0, or a ZULM_E* code with msg set. */
int zlm_rng_stream_draw(const char *name, uint32_t step, uint64_t offset, int dist, float *dst,
                        int64_t n, char *msg, size_t cap) {
  if (name == NULL || n < 0 || (n > 0 && dst == NULL)) {
    snprintf(msg, cap, "A NULL name or buffer, or a negative count.");
    return 7;
  }
  zurand_key k;
  ggml_critical_section_start();
  int i = find_stream(name);
  if (i >= 0) k = zlm_streams[i].key;
  ggml_critical_section_end();
  if (i < 0) {
    snprintf(msg, cap, "Stream '%s' is not registered: call rng_stream() first.", name);
    return 1;
  }
  if (n == 0) return 0;
  if (draw_f32(&k, (int64_t) step, dist, offset, (size_t) n, dst) != ZURAND_OK) {
    snprintf(msg, cap, "zurand refused the draw from stream '%s'.", name);
    return 5;
  }
  return 0;
}

/* zl_seed() changed the session key: every registered stream's key is
   stale, so the list is emptied. */
SEXP zlm_rng_streams_clear(void) {
  ggml_critical_section_start();
  for (int i = 0; i < zlm_n_streams; i++) free(zlm_streams[i].name);
  free(zlm_streams);
  zlm_streams = NULL;
  zlm_n_streams = zlm_cap_streams = 0;
  ggml_critical_section_end();
  return R_NilValue;
}

/* ---- in-graph noise --------------------------------------------------- */

static void noise_op(struct ggml_tensor *dst, int ith, int nth, void *userdata) {
  zlm_noise_t *nz = (zlm_noise_t *) userdata;
  int64_t n = ggml_nelements(dst);
  int64_t per = (n + nth - 1) / nth;
  int64_t i0 = per * ith;
  int64_t i1 = i0 + per < n ? i0 + per : n;
  if (i0 >= i1) return;
  /* positions follow ggml's element order: the first index fastest */
  draw_f32(&nz->key, *nz->step, nz->dist, nz->offset + (uint64_t) i0, (size_t) (i1 - i0),
           (float *) dst->data + i0);
}

struct ggml_tensor *zlm_noise_node(struct ggml_context *ctx, zlm_noise_t *nz, const int64_t ne[4]) {
  /* One task, which ggml runs on thread 0, the thread that called the
     computation: zurand's fills enter an OpenMP region even when they run
     serially, and libgomp keeps state for each thread that does so, which
     ggml's pool threads leaked when they exited (valgrind). The draws are
     positional, so the values do not change. */
  return ggml_custom_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3], NULL, 0, noise_op, 1, nz);
}

/* ---- per-row losses --------------------------------------------------- */

static void rowloss_op(struct ggml_tensor *dst, int ith, int nth, void *userdata) {
  zlm_rowloss *rl = (zlm_rowloss *) userdata;
  const struct ggml_tensor *a = dst->src[0];
  const struct ggml_tensor *b = dst->src[1];
  int64_t ncol = a->ne[0];
  int64_t nrow = ggml_nrows(a);
  int64_t per = (nrow + nth - 1) / nth;
  int64_t r0 = per * ith;
  int64_t r1 = r0 + per < nrow ? r0 + per : nrow;
  float *out = (float *) dst->data;
  for (int64_t r = r0; r < r1; r++) {
    /* rows of contiguous tensors (the operands are made contiguous) */
    const float *x = (const float *) ((const char *) a->data + r * a->nb[1]);
    const float *y = (const float *) ((const char *) b->data + r * b->nb[1]);
    double v = 0;
    if (rl->mode == ZLM_ROWLOSS_CE) {
      double mx = -INFINITY;
      for (int64_t c = 0; c < ncol; c++) mx = x[c] > mx ? x[c] : mx;
      double se = 0;
      for (int64_t c = 0; c < ncol; c++) se += exp((double) x[c] - mx);
      double lse = mx + log(se);
      for (int64_t c = 0; c < ncol; c++) v -= (double) y[c] * ((double) x[c] - lse);
    } else if (rl->mode == ZLM_ROWLOSS_SQ) {
      for (int64_t c = 0; c < ncol; c++) {
        double d = (double) x[c] - (double) y[c];
        v += d * d;
      }
    } else {
      for (int64_t c = 0; c < ncol; c++) v += fabs((double) x[c] - (double) y[c]);
    }
    out[r] = (float) v;
    if (rl->rows != NULL && r < rl->n_rows) rl->rows[r] = v;
  }
}

struct ggml_tensor *zlm_rowloss_node(struct ggml_context *ctx, zlm_rowloss *rl,
                                     struct ggml_tensor *a, struct ggml_tensor *b) {
  struct ggml_tensor *args[2] = { a, b };
  return ggml_custom_4d(ctx, GGML_TYPE_F32, ggml_nrows(a), 1, 1, 1, args, 2, rowloss_op,
                        GGML_N_TASKS_MAX, rl);
}

/* Pairwise sum in a fixed order: the same at any thread count. */
static double pairwise(const double *x, int64_t n) {
  if (n <= 8) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += x[i];
    return s;
  }
  int64_t h = n / 2;
  return pairwise(x, h) + pairwise(x + h, n - h);
}

double zlm_rowloss_value(const zlm_rowloss *rl) {
  return rl->scale * pairwise(rl->rows, rl->n_rows);
}
