/*
 * zlm_store.c: the parameter store and its persistence as GGUF
 * (design §5.1, §6.1, §6.6).
 *
 * A store is a list of named ggml tensors of type f32, f16 or bf16. Each
 * zl_param() call, and each zl_load(), allocates one batch: a ggml context
 * holding the batch's tensor metadata and one CPU backend buffer holding
 * its data. Batches never move, so a tensor's address is stable for the
 * store's lifetime and a graph built on a store stays valid when more
 * parameters are added later.
 *
 * Arguments are validated in R first; the checks here are the ones that
 * must hold before ggml sees a value (design §13.2, layer 1).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "zlm_glue.h"
#include "zlm_r.h"
#include "zlm_store.h"

SEXP zlm_store_tag = NULL;

static SEXP store_tag(void) {
  if (zlm_store_tag == NULL) {
    zlm_store_tag = Rf_install("zlm_store");
  }
  return zlm_store_tag;
}

zlm_store *zlm_store_from(SEXP x) {
  return (zlm_store *) zlm_ptr(x, store_tag(), "parameter store");
}

static void store_free(zlm_store *s) {
  if (s == NULL) {
    return;
  }
  for (int i = 0; i < s->n_batches; i++) {
    if (s->batches[i].buf) {
      ggml_backend_buffer_free(s->batches[i].buf);
    }
    if (s->batches[i].ctx) {
      ggml_free(s->batches[i].ctx);
    }
  }
  free(s->batches);
  free(s->tensors);
  free(s->ranks);
  free(s);
}

static void store_finalize(SEXP x) {
  zlm_store *s = (zlm_store *) R_ExternalPtrAddr(x);
  store_free(s);
  R_ClearExternalPtr(x);
}

static SEXP store_wrap(zlm_store *s) {
  SEXP ptr = PROTECT(R_MakeExternalPtr(s, store_tag(), R_NilValue));
  R_RegisterCFinalizerEx(ptr, store_finalize, TRUE);
  UNPROTECT(1);
  return ptr;
}

static zlm_store *store_alloc(void) {
  zlm_store *s = (zlm_store *) calloc(1, sizeof *s);
  if (s == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a parameter store.");
  }
  return s;
}

/* Grows the store's arrays to hold `extra` more tensors and one more
   batch. On failure nothing is changed. */
static int store_reserve(zlm_store *s, int extra) {
  if (s->n + extra > s->cap) {
    int cap = s->cap ? s->cap : 16;
    while (cap < s->n + extra) {
      cap *= 2;
    }
    struct ggml_tensor **t = (struct ggml_tensor **) realloc(s->tensors, (size_t) cap * sizeof *t);
    if (t == NULL) {
      return 0;
    }
    s->tensors = t;
    int *r = (int *) realloc(s->ranks, (size_t) cap * sizeof *r);
    if (r == NULL) {
      return 0;
    }
    s->ranks = r;
    s->cap = cap;
  }
  if (s->n_batches + 1 > s->cap_batches) {
    int cap = s->cap_batches ? 2 * s->cap_batches : 8;
    zlm_batch *b = (zlm_batch *) realloc(s->batches, (size_t) cap * sizeof *b);
    if (b == NULL) {
      return 0;
    }
    s->batches = b;
    s->cap_batches = cap;
  }
  return 1;
}

struct ggml_tensor *zlm_store_find(const zlm_store *s, const char *name) {
  for (int i = 0; i < s->n; i++) {
    if (strcmp(s->tensors[i]->name, name) == 0) {
      return s->tensors[i];
    }
  }
  return NULL;
}

enum ggml_type zlm_type_from_string(const char *type) {
  if (strcmp(type, "f32") == 0) return GGML_TYPE_F32;
  if (strcmp(type, "f16") == 0) return GGML_TYPE_F16;
  if (strcmp(type, "bf16") == 0) return GGML_TYPE_BF16;
  zlm_raise("zulm_type_error", R_NilValue,
            "`type` must be \"f32\", \"f16\" or \"bf16\", not \"%s\".", type);
}

const char *zlm_type_name(enum ggml_type type) {
  switch (type) {
    case GGML_TYPE_F32: return "f32";
    case GGML_TYPE_F16: return "f16";
    case GGML_TYPE_BF16: return "bf16";
    case GGML_TYPE_I32: return "i32";
    default: return ggml_type_name(type);
  }
}

static const char *store_name(SEXP name) {
  if (TYPEOF(name) != STRSXP || XLENGTH(name) != 1 || STRING_ELT(name, 0) == NA_STRING) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "`name` must be a single string.");
  }
  const char *nm = Rf_translateCharUTF8(STRING_ELT(name, 0));
  size_t len = strlen(nm);
  if (len == 0 || len >= GGML_MAX_NAME) {
    zlm_raise("zulm_invalid_argument", R_NilValue,
              "A parameter name must have 1 to %d bytes; \"%s\" has %d.",
              GGML_MAX_NAME - 1, nm, (int) len);
  }
  return nm;
}

/* dims: 1 to 4 positive whole numbers; fills ne (padded with 1) and
   returns the rank. */
static int store_dims(SEXP dims, int64_t ne[GGML_MAX_DIMS]) {
  R_xlen_t k = Rf_xlength(dims);
  if ((TYPEOF(dims) != INTSXP && TYPEOF(dims) != REALSXP) || k < 1 || k > GGML_MAX_DIMS) {
    zlm_raise("zulm_shape_error", R_NilValue,
              "`dim` must hold one to four dimensions.");
  }
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    ne[i] = 1;
  }
  double total = 1;
  for (R_xlen_t i = 0; i < k; i++) {
    double d = TYPEOF(dims) == INTSXP ? (INTEGER(dims)[i] == NA_INTEGER ? NAN : INTEGER(dims)[i])
                                      : REAL(dims)[i];
    if (!(d >= 1) || d != floor(d) || d > 9.007199254740992e15) {
      zlm_raise("zulm_shape_error", R_NilValue,
                "Every dimension must be a positive whole number.");
    }
    ne[i] = (int64_t) d;
    total *= d;
  }
  if (total > 4.611686018427388e18) {
    zlm_raise("zulm_shape_error", R_NilValue, "The tensor has too many elements.");
  }
  return (int) k;
}

/* Converts n R values (double or integer) to f32, rounding to nearest.
   Returns the number of finite values that became infinite. */
static R_xlen_t to_f32(SEXP x, float *out, R_xlen_t n) {
  R_xlen_t over = 0;
  if (TYPEOF(x) == REALSXP) {
    const double *v = REAL(x);
    for (R_xlen_t i = 0; i < n; i++) {
      float f = (float) v[i];
      if (isinf(f) && isfinite(v[i])) {
        over++;
      }
      out[i] = f;
    }
  } else {
    const int *v = INTEGER(x);
    for (R_xlen_t i = 0; i < n; i++) {
      out[i] = (float) v[i];
    }
  }
  return over;
}

/* Writes n R values into a tensor's data of type f32, f16 or bf16. The
   caller has checked for NA. Returns the number that overflowed. */
static R_xlen_t write_values(struct ggml_tensor *t, SEXP x) {
  R_xlen_t n = (R_xlen_t) ggml_nelements(t);
  if (t->type == GGML_TYPE_F32) {
    return to_f32(x, (float *) t->data, n);
  }
  float *tmp = (float *) malloc((size_t) n * sizeof(float));
  if (tmp == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory converting values.");
  }
  R_xlen_t over = to_f32(x, tmp, n);
  if (t->type == GGML_TYPE_F16) {
    ggml_fp32_to_fp16_row(tmp, (ggml_fp16_t *) t->data, n);
    const ggml_fp16_t *h = (const ggml_fp16_t *) t->data;
    for (R_xlen_t i = 0; i < n; i++) {
      if (isfinite(tmp[i]) && isinf(ggml_fp16_to_fp32(h[i]))) {
        over++;
      }
    }
  } else {
    ggml_fp32_to_bf16_row(tmp, (ggml_bf16_t *) t->data, n);
    const ggml_bf16_t *h = (const ggml_bf16_t *) t->data;
    for (R_xlen_t i = 0; i < n; i++) {
      if (isfinite(tmp[i]) && isinf(ggml_bf16_to_fp32(h[i]))) {
        over++;
      }
    }
  }
  free(tmp);
  return over;
}

/* Reads a tensor of type f32, f16 or bf16 into n doubles. */
void zlm_read_values(const struct ggml_tensor *t, double *out) {
  int64_t n = ggml_nelements(t);
  switch (t->type) {
    case GGML_TYPE_F32: {
      const float *v = (const float *) t->data;
      for (int64_t i = 0; i < n; i++) out[i] = v[i];
    } break;
    case GGML_TYPE_F16: {
      const ggml_fp16_t *v = (const ggml_fp16_t *) t->data;
      for (int64_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(v[i]);
    } break;
    case GGML_TYPE_BF16: {
      const ggml_bf16_t *v = (const ggml_bf16_t *) t->data;
      for (int64_t i = 0; i < n; i++) out[i] = ggml_bf16_to_fp32(v[i]);
    } break;
    default:
      memset(out, 0, (size_t) n * sizeof(double));
  }
}

static void check_values(SEXP x, R_xlen_t n, const char *what) {
  if (TYPEOF(x) != REALSXP && TYPEOF(x) != INTSXP) {
    zlm_raise("zulm_type_error", R_NilValue, "%s must be numeric.", what);
  }
  if (Rf_xlength(x) != n) {
    zlm_raise("zulm_shape_error", R_NilValue, "%s has %.0f values; %.0f are needed.",
              what, (double) Rf_xlength(x), (double) n);
  }
  if (TYPEOF(x) == REALSXP) {
    const double *v = REAL(x);
    for (R_xlen_t i = 0; i < n; i++) {
      if (ISNA(v[i])) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "%s contains NA.", what);
      }
    }
  } else {
    const int *v = INTEGER(x);
    for (R_xlen_t i = 0; i < n; i++) {
      if (v[i] == NA_INTEGER) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "%s contains NA.", what);
      }
    }
  }
}

/* Allocates one batch of `count` tensors with the given names, shapes and
   types, appends them to the store and returns the index of the first.
   Nothing is changed when the memory cap or an allocation fails. */
static int store_add_batch(zlm_store *s, int count, const char **names,
                           int64_t (*ne)[GGML_MAX_DIMS], const int *ranks,
                           const enum ggml_type *types, double max_bytes,
                           const char *what) {
  double bytes = 0;
  for (int i = 0; i < count; i++) {
    double n = (double) ne[i][0] * (double) ne[i][1] * (double) ne[i][2] * (double) ne[i][3];
    bytes += n * (double) ggml_type_size(types[i]) + 64;
  }
  if (bytes > max_bytes) {
    zlm_raise_memory(bytes, max_bytes, what);
  }
  if (!store_reserve(s, count)) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory growing the parameter store.");
  }
  struct ggml_init_params params = {
    (size_t) count * ggml_tensor_overhead() + 1024, NULL, true
  };
  struct ggml_context *ctx = ggml_init(params);
  if (ctx == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating tensor metadata.");
  }
  for (int i = 0; i < count; i++) {
    struct ggml_tensor *t = ggml_new_tensor(ctx, types[i], GGML_MAX_DIMS, ne[i]);
    ggml_set_name(t, names[i]);
  }
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
  if (buf == NULL) {
    ggml_free(ctx);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory allocating %.0f bytes for %s.",
              bytes, what);
  }
  ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  s->batches[s->n_batches].ctx = ctx;
  s->batches[s->n_batches].buf = buf;
  s->n_batches++;
  int first = s->n;
  int i = 0;
  for (struct ggml_tensor *t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
    s->tensors[s->n] = t;
    s->ranks[s->n] = ranks[i++];
    s->n++;
  }
  s->bytes += (double) ggml_backend_buffer_get_size(buf);
  return first;
}

/* R integers (or whole doubles in int32 range) to int32, for index inputs
   (design §6.1): NA is refused, so ggml never sees an index R would not. */
void zlm_to_i32(SEXP x, int32_t *out, R_xlen_t n, const char *what) {
  if (TYPEOF(x) == INTSXP) {
    const int *v = INTEGER(x);
    for (R_xlen_t i = 0; i < n; i++) {
      if (v[i] == NA_INTEGER) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "%s contains NA.", what);
      }
      out[i] = (int32_t) v[i];
    }
  } else if (TYPEOF(x) == REALSXP) {
    const double *v = REAL(x);
    for (R_xlen_t i = 0; i < n; i++) {
      if (ISNAN(v[i])) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "%s contains NA.", what);
      }
      if (v[i] != floor(v[i]) || v[i] < -2147483647.0 || v[i] > 2147483647.0) {
        zlm_raise("zulm_type_error", R_NilValue,
                  "%s must hold whole numbers in the 32-bit integer range.", what);
      }
      out[i] = (int32_t) v[i];
    }
  } else {
    zlm_raise("zulm_type_error", R_NilValue, "%s must be an integer vector.", what);
  }
}

/* Internal: the converter above, round-tripped, for its tests. */
SEXP zlm_i32_roundtrip(SEXP x) {
  R_xlen_t n = Rf_xlength(x);
  int32_t *buf = (int32_t *) R_alloc((size_t) (n > 0 ? n : 1), sizeof(int32_t));
  zlm_to_i32(x, buf, n, "The index");
  SEXP out = PROTECT(Rf_allocVector(INTSXP, n));
  for (R_xlen_t i = 0; i < n; i++) INTEGER(out)[i] = buf[i];
  UNPROTECT(1);
  return out;
}

SEXP zlm_store_new(void) {
  zlm_store *s = store_alloc();
  return store_wrap(s);
}

SEXP zlm_store_free_(SEXP ptr) {
  zlm_store *s = zlm_store_from(ptr);
  store_free(s);
  R_ClearExternalPtr(ptr);
  return R_NilValue;
}

/* zl_param(): adds one tensor, filled from `values` (or zeros if NULL). */
SEXP zlm_store_add(SEXP ptr, SEXP name, SEXP dims, SEXP type, SEXP values, SEXP max_bytes) {
  zlm_store *s = zlm_store_from(ptr);
  const char *nm = store_name(name);
  if (zlm_store_find(s, nm) != NULL) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The store already has a parameter \"%s\".", nm);
  }
  int64_t ne[1][GGML_MAX_DIMS];
  int rank = store_dims(dims, ne[0]);
  enum ggml_type tp = zlm_type_from_string(CHAR(STRING_ELT(type, 0)));
  R_xlen_t n = (R_xlen_t) (ne[0][0] * ne[0][1] * ne[0][2] * ne[0][3]);
  if (!Rf_isNull(values)) {
    check_values(values, n, "The initial value");
  }
  int idx = store_add_batch(s, 1, &nm, ne, &rank, &tp, Rf_asReal(max_bytes), "The parameter");
  struct ggml_tensor *t = s->tensors[idx];
  if (Rf_isNull(values)) {
    memset(t->data, 0, ggml_nbytes(t));
  } else {
    R_xlen_t over = write_values(t, values);
    if (over > 0) {
      zlm_warn_range(over, nm);
    }
  }
  return R_NilValue;
}

static struct ggml_tensor *store_get_tensor(zlm_store *s, SEXP name) {
  const char *nm = store_name(name);
  struct ggml_tensor *t = zlm_store_find(s, nm);
  if (t == NULL) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The store has no parameter \"%s\".", nm);
  }
  return t;
}

SEXP zlm_store_get(SEXP ptr, SEXP name) {
  zlm_store *s = zlm_store_from(ptr);
  struct ggml_tensor *t = store_get_tensor(s, name);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) ggml_nelements(t)));
  zlm_read_values(t, REAL(out));
  UNPROTECT(1);
  return out;
}

SEXP zlm_store_set(SEXP ptr, SEXP name, SEXP values) {
  zlm_store *s = zlm_store_from(ptr);
  struct ggml_tensor *t = store_get_tensor(s, name);
  check_values(values, (R_xlen_t) ggml_nelements(t), "`value`");
  R_xlen_t over = write_values(t, values);
  if (over > 0) {
    zlm_warn_range(over, t->name);
  }
  return R_NilValue;
}

/* list(names, dims, types, bytes) in insertion order. */
SEXP zlm_store_info(SEXP ptr) {
  zlm_store *s = zlm_store_from(ptr);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, s->n));
  SEXP dims = PROTECT(Rf_allocVector(VECSXP, s->n));
  SEXP types = PROTECT(Rf_allocVector(STRSXP, s->n));
  for (int i = 0; i < s->n; i++) {
    struct ggml_tensor *t = s->tensors[i];
    SET_STRING_ELT(names, i, Rf_mkCharCE(t->name, CE_UTF8));
    SEXP d = Rf_allocVector(REALSXP, s->ranks[i]);
    SET_VECTOR_ELT(dims, i, d);
    for (int k = 0; k < s->ranks[i]; k++) {
      REAL(d)[k] = (double) t->ne[k];
    }
    SET_STRING_ELT(types, i, Rf_mkChar(zlm_type_name(t->type)));
  }
  SET_VECTOR_ELT(out, 0, names);
  SET_VECTOR_ELT(out, 1, dims);
  SET_VECTOR_ELT(out, 2, types);
  SET_VECTOR_ELT(out, 3, Rf_ScalarReal(s->bytes));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 4));
  SET_STRING_ELT(nm, 0, Rf_mkChar("names"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("dims"));
  SET_STRING_ELT(nm, 2, Rf_mkChar("types"));
  SET_STRING_ELT(nm, 3, Rf_mkChar("bytes"));
  Rf_setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(5);
  return out;
}

/* ---- GGUF ------------------------------------------------------------ */

#define ZLM_RANKS_KEY "zulm.tensor_ranks"

static void file_error(const char *fmt, const char *path, const char *detail) {
  zlm_raise("zulm_file_error", R_NilValue, fmt, path, detail);
}

static const char *log_detail(char *buf, size_t cap) {
  int dropped = 0;
  zlm_log_take(buf, cap, &dropped);
  size_t n = strlen(buf);
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' ')) {
    buf[--n] = '\0';
  }
  return n > 0 ? buf : "no detail";
}

struct save_job {
  struct gguf_context *g;
  const char *path;
  int ok;
};

static void save_body(void *data) {
  struct save_job *j = (struct save_job *) data;
  j->ok = gguf_write_to_file(j->g, j->path, false);
}

/* Sets one metadata value from R: a character, integer, double or
   logical vector without NA (checked in R). */
static void set_meta(struct gguf_context *g, const char *key, SEXP v) {
  R_xlen_t n = Rf_xlength(v);
  switch (TYPEOF(v)) {
    case STRSXP:
      if (n == 1) {
        gguf_set_val_str(g, key, Rf_translateCharUTF8(STRING_ELT(v, 0)));
      } else {
        const char **strs = (const char **) R_alloc((size_t) n, sizeof(char *));
        for (R_xlen_t i = 0; i < n; i++) strs[i] = Rf_translateCharUTF8(STRING_ELT(v, i));
        gguf_set_arr_str(g, key, strs, (size_t) n);
      }
      break;
    case INTSXP:
      if (n == 1) gguf_set_val_i32(g, key, INTEGER(v)[0]);
      else gguf_set_arr_data(g, key, GGUF_TYPE_INT32, INTEGER(v), (size_t) n);
      break;
    case REALSXP:
      if (n == 1) gguf_set_val_f64(g, key, REAL(v)[0]);
      else gguf_set_arr_data(g, key, GGUF_TYPE_FLOAT64, REAL(v), (size_t) n);
      break;
    case LGLSXP:
      if (n == 1) {
        gguf_set_val_bool(g, key, LOGICAL(v)[0] != 0);
      } else {
        int8_t *b = (int8_t *) R_alloc((size_t) n, 1);
        for (R_xlen_t i = 0; i < n; i++) b[i] = (int8_t) (LOGICAL(v)[i] != 0);
        gguf_set_arr_data(g, key, GGUF_TYPE_BOOL, b, (size_t) n);
      }
      break;
    default:
      break;
  }
}

/* Converts a tensor's data to `to` into a fresh malloc'd buffer. */
static void *convert_data(const struct ggml_tensor *t, enum ggml_type to) {
  int64_t n = ggml_nelements(t);
  float *f32 = (float *) malloc((size_t) n * sizeof(float));
  if (f32 == NULL) return NULL;
  switch (t->type) {
    case GGML_TYPE_F32: memcpy(f32, t->data, (size_t) n * sizeof(float)); break;
    case GGML_TYPE_F16: ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f32, n); break;
    case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, f32, n); break;
    default: break;
  }
  if (to == GGML_TYPE_F32) return f32;
  void *out = malloc((size_t) n * ggml_type_size(to));
  if (out != NULL) {
    if (to == GGML_TYPE_F16) ggml_fp32_to_fp16_row(f32, (ggml_fp16_t *) out, n);
    else ggml_fp32_to_bf16_row(f32, (ggml_bf16_t *) out, n);
  }
  free(f32);
  return out;
}

SEXP zlm_store_save(SEXP ptr, SEXP path, SEXP type, SEXP meta, SEXP max_bytes) {
  zlm_store *s = zlm_store_from(ptr);
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  enum ggml_type to = zlm_type_from_string(CHAR(STRING_ELT(type, 0)));
  double need = 0;
  for (int i = 0; i < s->n; i++) {
    if (s->tensors[i]->type != to) {
      need += (double) ggml_nelements(s->tensors[i]) * (4.0 + (double) ggml_type_size(to));
    }
  }
  if (need > Rf_asReal(max_bytes)) {
    zlm_raise_memory(need, Rf_asReal(max_bytes), "Converting the store for saving");
  }
  struct gguf_context *g = gguf_init_empty();
  SEXP keys = Rf_getAttrib(meta, R_NamesSymbol);
  for (R_xlen_t i = 0; i < Rf_xlength(meta); i++) {
    set_meta(g, Rf_translateCharUTF8(STRING_ELT(keys, i)), VECTOR_ELT(meta, i));
  }
  if (s->n > 0) {
    gguf_set_arr_data(g, ZLM_RANKS_KEY, GGUF_TYPE_INT32, s->ranks, (size_t) s->n);
  }
  void **converted = (void **) calloc((size_t) (s->n > 0 ? s->n : 1), sizeof(void *));
  int failed = converted == NULL;
  for (int i = 0; i < s->n && !failed; i++) {
    struct ggml_tensor *t = s->tensors[i];
    gguf_add_tensor(g, t);
    if (t->type != to) {
      converted[i] = convert_data(t, to);
      if (converted[i] == NULL) {
        failed = 1;
        break;
      }
      gguf_set_tensor_type(g, t->name, to);
      gguf_set_tensor_data(g, t->name, converted[i]);
    }
  }
  char msg[2048] = "";
  int status = ZLM_STATUS_OK;
  struct save_job job = { g, fname, 0 };
  if (!failed) {
    status = zlm_guarded(save_body, &job, msg, sizeof msg);
  }
  for (int i = 0; i < s->n && converted != NULL; i++) {
    free(converted[i]);
  }
  free(converted);
  gguf_free(g);
  if (failed) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory converting the store for saving.");
  }
  char detail[1024];
  if (status != ZLM_STATUS_OK) {
    file_error("Could not write '%s': %s", fname, msg);
  }
  if (!job.ok) {
    file_error("Could not write '%s': %s", fname, log_detail(detail, sizeof detail));
  }
  return R_NilValue;
}

/* Metadata value to R. Arrays of strings, numbers and booleans become
   vectors; nested arrays are skipped. */
static SEXP meta_value(const struct gguf_context *g, int64_t k) {
  enum gguf_type tp = gguf_get_kv_type(g, k);
  switch (tp) {
    case GGUF_TYPE_UINT8: return Rf_ScalarInteger(gguf_get_val_u8(g, k));
    case GGUF_TYPE_INT8: return Rf_ScalarInteger(gguf_get_val_i8(g, k));
    case GGUF_TYPE_UINT16: return Rf_ScalarInteger(gguf_get_val_u16(g, k));
    case GGUF_TYPE_INT16: return Rf_ScalarInteger(gguf_get_val_i16(g, k));
    case GGUF_TYPE_UINT32: return Rf_ScalarReal((double) gguf_get_val_u32(g, k));
    case GGUF_TYPE_INT32: return Rf_ScalarInteger(gguf_get_val_i32(g, k));
    case GGUF_TYPE_FLOAT32: return Rf_ScalarReal(gguf_get_val_f32(g, k));
    case GGUF_TYPE_BOOL: return Rf_ScalarLogical(gguf_get_val_bool(g, k));
    case GGUF_TYPE_STRING: return Rf_ScalarString(Rf_mkCharCE(gguf_get_val_str(g, k), CE_UTF8));
    case GGUF_TYPE_UINT64: return Rf_ScalarReal((double) gguf_get_val_u64(g, k));
    case GGUF_TYPE_INT64: return Rf_ScalarReal((double) gguf_get_val_i64(g, k));
    case GGUF_TYPE_FLOAT64: return Rf_ScalarReal(gguf_get_val_f64(g, k));
    case GGUF_TYPE_ARRAY: {
      enum gguf_type at = gguf_get_arr_type(g, k);
      size_t n = gguf_get_arr_n(g, k);
      if (at == GGUF_TYPE_STRING) {
        SEXP out = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) n));
        for (size_t i = 0; i < n; i++) {
          SET_STRING_ELT(out, (R_xlen_t) i, Rf_mkCharCE(gguf_get_arr_str(g, k, i), CE_UTF8));
        }
        UNPROTECT(1);
        return out;
      }
      if (at == GGUF_TYPE_ARRAY) {
        return R_NilValue;
      }
      const void *d = gguf_get_arr_data(g, k);
      if (at == GGUF_TYPE_BOOL) {
        SEXP out = Rf_allocVector(LGLSXP, (R_xlen_t) n);
        for (size_t i = 0; i < n; i++) LOGICAL(out)[i] = ((const int8_t *) d)[i] != 0;
        return out;
      }
      int as_int = at == GGUF_TYPE_UINT8 || at == GGUF_TYPE_INT8 || at == GGUF_TYPE_UINT16 ||
                   at == GGUF_TYPE_INT16 || at == GGUF_TYPE_INT32;
      SEXP out = Rf_allocVector(as_int ? INTSXP : REALSXP, (R_xlen_t) n);
      for (size_t i = 0; i < n; i++) {
        double v = 0;
        switch (at) {
          case GGUF_TYPE_UINT8: v = ((const uint8_t *) d)[i]; break;
          case GGUF_TYPE_INT8: v = ((const int8_t *) d)[i]; break;
          case GGUF_TYPE_UINT16: v = ((const uint16_t *) d)[i]; break;
          case GGUF_TYPE_INT16: v = ((const int16_t *) d)[i]; break;
          case GGUF_TYPE_UINT32: v = ((const uint32_t *) d)[i]; break;
          case GGUF_TYPE_INT32: v = ((const int32_t *) d)[i]; break;
          case GGUF_TYPE_FLOAT32: v = ((const float *) d)[i]; break;
          case GGUF_TYPE_UINT64: v = (double) ((const uint64_t *) d)[i]; break;
          case GGUF_TYPE_INT64: v = (double) ((const int64_t *) d)[i]; break;
          case GGUF_TYPE_FLOAT64: v = ((const double *) d)[i]; break;
          default: break;
        }
        if (as_int) INTEGER(out)[i] = (int) v;
        else REAL(out)[i] = v;
      }
      return out;
    }
    default:
      return R_NilValue;
  }
}

struct load_job {
  const char *path;
  struct gguf_context *g;
};

static void load_body(void *data) {
  struct load_job *j = (struct load_job *) data;
  struct gguf_init_params params = { true, NULL };
  j->g = gguf_init_from_file(j->path, params);
}

static void gguf_finalize(SEXP x) {
  struct gguf_context *g = (struct gguf_context *) R_ExternalPtrAddr(x);
  if (g != NULL) {
    gguf_free(g);
    R_ClearExternalPtr(x);
  }
}

#if defined(_WIN32)
#define zlm_fseek _fseeki64
#else
#define zlm_fseek fseeko
#endif

/* zl_load(): a new store with every tensor of a GGUF file widened to f32
   (design §6.1), and the metadata as a named list. */
SEXP zlm_store_load(SEXP path, SEXP max_bytes) {
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  char msg[2048] = "";
  char detail[1024];
  struct load_job job = { fname, NULL };
  int status = zlm_guarded(load_body, &job, msg, sizeof msg);
  if (status != ZLM_STATUS_OK) {
    if (job.g) gguf_free(job.g);
    file_error("'%s' is not a valid GGUF file: %s", fname, msg);
  }
  if (job.g == NULL) {
    file_error("'%s' is not a valid GGUF file: %s", fname, log_detail(detail, sizeof detail));
  }
  struct gguf_context *g = job.g;
  /* From here every R error frees g through this pointer's finalizer. */
  SEXP gptr = PROTECT(R_MakeExternalPtr(g, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(gptr, gguf_finalize, TRUE);
  int64_t n = gguf_get_n_tensors(g);
  if (n > 1000000) {
    file_error("'%s' %s", fname, "lists too many tensors.");
  }
  /* ranks written by zl_save(), when present and consistent */
  const int32_t *ranks_meta = NULL;
  int64_t kr = gguf_find_key(g, ZLM_RANKS_KEY);
  if (kr >= 0 && gguf_get_kv_type(g, kr) == GGUF_TYPE_ARRAY &&
      gguf_get_arr_type(g, kr) == GGUF_TYPE_INT32 && (int64_t) gguf_get_arr_n(g, kr) == n) {
    ranks_meta = (const int32_t *) gguf_get_arr_data(g, kr);
  }
  const char **names = (const char **) R_alloc((size_t) (n > 0 ? n : 1), sizeof(char *));
  int64_t (*ne)[GGML_MAX_DIMS] = (int64_t (*)[GGML_MAX_DIMS]) R_alloc((size_t) (n > 0 ? n : 1), sizeof *ne);
  int *ranks = (int *) R_alloc((size_t) (n > 0 ? n : 1), sizeof(int));
  enum ggml_type *types = (enum ggml_type *) R_alloc((size_t) (n > 0 ? n : 1), sizeof(enum ggml_type));
  for (int64_t i = 0; i < n; i++) {
    enum ggml_type tp = gguf_get_tensor_type(g, i);
    if (tp != GGML_TYPE_F32 && tp != GGML_TYPE_F16 && tp != GGML_TYPE_BF16) {
      const char *tn = ggml_type_name(tp);
      zlm_raise("zulm_file_error", R_NilValue,
                "'%s' holds a tensor of type %s; zulm reads f32, f16 and bf16.", fname, tn);
    }
    names[i] = gguf_get_tensor_name(g, i);
    const int64_t *tne = gguf_get_tensor_ne(g, i);
    int rank = 1;
    for (int k = 0; k < GGML_MAX_DIMS; k++) {
      ne[i][k] = tne[k];
      if (tne[k] < 1) {
        file_error("'%s' %s", fname, "holds a tensor with an empty dimension.");
      }
      if (tne[k] > 1) rank = k + 1;
    }
    if (ranks_meta != NULL && ranks_meta[i] >= rank && ranks_meta[i] <= GGML_MAX_DIMS) {
      rank = ranks_meta[i];
    }
    ranks[i] = rank;
    types[i] = GGML_TYPE_F32;
  }
  zlm_store *s = (zlm_store *) calloc(1, sizeof *s);
  if (s == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a parameter store.");
  }
  SEXP sptr = PROTECT(store_wrap(s));
  if (n > 0) {
    store_add_batch(s, (int) n, names, ne, ranks, types, Rf_asReal(max_bytes), "The file's tensors");
  }
  FILE *f = n > 0 ? ggml_fopen(fname, "rb") : NULL;
  if (n > 0 && f == NULL) {
    file_error("Could not open '%s': %s", fname, "fopen failed");
  }
  size_t data_off = gguf_get_data_offset(g);
  int bad = 0;
  void *tmp = NULL;
  size_t tmp_cap = 0;
  for (int64_t i = 0; i < n && !bad; i++) {
    struct ggml_tensor *t = s->tensors[i];
    enum ggml_type tp = gguf_get_tensor_type(g, i);
    size_t size = gguf_get_tensor_size(g, i);
    if (zlm_fseek(f, (long long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) != 0) {
      bad = 1;
      break;
    }
    if (tp == GGML_TYPE_F32) {
      bad = fread(t->data, 1, size, f) != size;
    } else {
      if (size > tmp_cap) {
        free(tmp);
        tmp = malloc(size);
        tmp_cap = tmp ? size : 0;
      }
      if (tmp == NULL) {
        bad = 2;
        break;
      }
      bad = fread(tmp, 1, size, f) != size;
      if (!bad) {
        if (tp == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) tmp, (float *) t->data, ggml_nelements(t));
        else ggml_bf16_to_fp32_row((const ggml_bf16_t *) tmp, (float *) t->data, ggml_nelements(t));
      }
    }
  }
  free(tmp);
  if (f) fclose(f);
  if (bad) {
    if (bad == 2) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory reading '%s'.", fname);
    file_error("'%s' %s", fname, "is truncated: a tensor's data could not be read.");
  }
  int64_t n_kv = gguf_get_n_kv(g);
  SEXP meta = PROTECT(Rf_allocVector(VECSXP, n_kv));
  SEXP keys = PROTECT(Rf_allocVector(STRSXP, n_kv));
  int64_t m = 0;
  for (int64_t k = 0; k < n_kv; k++) {
    const char *key = gguf_get_key(g, k);
    if (strcmp(key, ZLM_RANKS_KEY) == 0) continue;
    SEXP v = meta_value(g, k);
    if (Rf_isNull(v)) continue;
    SET_VECTOR_ELT(meta, m, v);
    SET_STRING_ELT(keys, m, Rf_mkCharCE(key, CE_UTF8));
    m++;
  }
  gguf_free(g);
  R_ClearExternalPtr(gptr);
  SEXP meta2 = PROTECT(Rf_lengthgets(meta, m));
  SEXP keys2 = PROTECT(Rf_lengthgets(keys, m));
  Rf_setAttrib(meta2, R_NamesSymbol, keys2);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, sptr);
  SET_VECTOR_ELT(out, 1, meta2);
  UNPROTECT(7);
  return out;
}
