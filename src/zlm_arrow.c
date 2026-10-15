/* zlm_arrow.c: training batches from Arrow C streams (design §7.3).
 *
 * zl_arrow_batches() takes over a nanoarrow_array_stream (an external
 * pointer to an ArrowArrayStream) and reads it on the main thread only,
 * since get_next may call into its producer. Record batches of any size
 * are accumulated into batches of exactly `rows` rows, interleaved into
 * feature-major tensors c(features, rows): the one copy the design allows.
 * The stream, its schema and every array are released exactly once: when
 * consumed, when the stream ends, or when the iterator is collected.
 */
#include "zlm_compat.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "zlm_arrow.h"
#include "zlm_arrow_abi.h"
#include "zlm_graph.h"
#include "zlm_r.h"

enum {
  K_F32, K_F64, K_I8, K_I16, K_I32, K_I64, K_BOOL, K_DICT,
  /* unsigned integers: dictionary codes only */
  K_U8, K_U16, K_U32, K_U64
};

typedef struct {
  int child;        /* index among the schema's children */
  int kind;         /* K_* of the column (of its indices, for K_DICT) */
  int index_kind;   /* K_I8 … K_I64, for K_DICT */
  int large;        /* dictionary of large strings ("U") */
  char *name;
  /* the dictionary of the first chunk, which later chunks must repeat */
  int have_dict;
  int64_t dict_n;
  int64_t *dict_off; /* dict_n + 1 offsets into dict_data */
  char *dict_data;
} zlm_acol;

typedef struct {
  char name[64];
  int is_int;
  int nf;
  int *cols; /* nf indices into the iterator's columns */
} zlm_ainput;

typedef struct zlm_arrow_it {
  struct ArrowArrayStream stream; /* moved from the producer's pointer */
  struct ArrowSchema schema;
  struct ArrowArray chunk;
  int have_chunk;
  int64_t pos, len;  /* the next row of the chunk, and its length */
  int64_t row;       /* rows read from the stream so far */
  int done;          /* the stream has ended and is released */
  int64_t rows;
  int na_mask, pad;
  int n_cols, n_in;
  zlm_acol *cols;
  zlm_ainput *in;
} zlm_arrow_it;

/* ---- lifetime ------------------------------------------------------ */

static SEXP it_tag(void) {
  static SEXP tag = NULL;
  if (tag == NULL) tag = Rf_install("zulm_arrow_batches");
  return tag;
}

static void it_release_stream(zlm_arrow_it *it) {
  if (it->have_chunk) {
    if (it->chunk.release != NULL) it->chunk.release(&it->chunk);
    it->have_chunk = 0;
  }
  if (it->schema.release != NULL) it->schema.release(&it->schema);
  if (it->stream.release != NULL) it->stream.release(&it->stream);
  it->done = 1;
}

static void it_free(zlm_arrow_it *it) {
  if (it == NULL) return;
  it_release_stream(it);
  for (int i = 0; i < it->n_cols; i++) {
    free(it->cols[i].name);
    free(it->cols[i].dict_off);
    free(it->cols[i].dict_data);
  }
  for (int i = 0; i < it->n_in; i++) free(it->in[i].cols);
  free(it->cols);
  free(it->in);
  free(it);
}

static void it_finalize(SEXP x) {
  it_free((zlm_arrow_it *) R_ExternalPtrAddr(x));
  R_ClearExternalPtr(x);
}

static zlm_arrow_it *it_from(SEXP x) {
  if (TYPEOF(x) != EXTPTRSXP || R_ExternalPtrTag(x) != it_tag()) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "Expected an iterator from zl_arrow_batches().");
  }
  zlm_arrow_it *it = (zlm_arrow_it *) R_ExternalPtrAddr(x);
  if (it == NULL) {
    zlm_raise("zulm_stale_error", R_NilValue, "This iterator is no longer valid.");
  }
  return it;
}

/* ---- errors -------------------------------------------------------- */

static void arrow_error(const char *column, int64_t row, const char *fmt, const char *a, const char *b) {
  SEXP extra = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(nm, 0, Rf_mkChar("column"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("row"));
  Rf_setAttrib(extra, R_NamesSymbol, nm);
  SET_VECTOR_ELT(extra, 0, column == NULL ? Rf_ScalarString(NA_STRING) : Rf_mkString(column));
  SET_VECTOR_ELT(extra, 1, Rf_ScalarReal(row < 0 ? NA_REAL : (double) row));
  zlm_raise("zulm_arrow_error", extra, fmt, a == NULL ? "" : a, b == NULL ? "" : b);
}

static void stream_error(struct ArrowArrayStream *st, int code, const char *what) {
  const char *msg = NULL;
  if (st->get_last_error != NULL) msg = st->get_last_error(st);
  char buf[512];
  snprintf(buf, sizeof buf, "%s (error %d)", msg != NULL ? msg : "no description", code);
  arrow_error(NULL, -1, "The Arrow stream failed while %s: %s.", what, buf);
}

/* ---- types --------------------------------------------------------- */

static int kind_of(const char *fmt) {
  if (fmt == NULL || fmt[0] == '\0' || fmt[1] != '\0') return -1;
  switch (fmt[0]) {
    case 'f': return K_F32;
    case 'g': return K_F64;
    case 'c': return K_I8;
    case 's': return K_I16;
    case 'i': return K_I32;
    case 'l': return K_I64;
    case 'b': return K_BOOL;
    default: return -1;
  }
}

/* The integer type of a dictionary's codes, signed or unsigned. */
static int index_kind_of(const char *fmt) {
  int k = kind_of(fmt);
  if (k >= K_I8 && k <= K_I64) return k;
  if (fmt == NULL || fmt[0] == '\0' || fmt[1] != '\0') return -1;
  switch (fmt[0]) {
    case 'C': return K_U8;
    case 'S': return K_U16;
    case 'I': return K_U32;
    case 'L': return K_U64;
    default: return -1;
  }
}

static int bit(const void *buf, int64_t i) {
  return (((const uint8_t *) buf)[i >> 3] >> (i & 7)) & 1;
}

static int64_t read_int(int kind, const void *values, int64_t i) {
  switch (kind) {
    case K_I8: return ((const int8_t *) values)[i];
    case K_I16: return ((const int16_t *) values)[i];
    case K_I32: return ((const int32_t *) values)[i];
    case K_U8: return ((const uint8_t *) values)[i];
    case K_U16: return ((const uint16_t *) values)[i];
    case K_U32: return ((const uint32_t *) values)[i];
    case K_U64: {
      uint64_t u = ((const uint64_t *) values)[i];
      return u > (uint64_t) INT64_MAX ? -1 : (int64_t) u;
    }
    default: return ((const int64_t *) values)[i];
  }
}

/* ---- opening -------------------------------------------------------- */

static int find_child(const struct ArrowSchema *s, const char *name) {
  for (int64_t c = 0; c < s->n_children; c++) {
    const struct ArrowSchema *ch = s->children[c];
    if (ch != NULL && ch->name != NULL && strcmp(ch->name, name) == 0) return (int) c;
  }
  return -1;
}

/* zl_arrow_batches(): takes over `xstream` (a nanoarrow_array_stream),
   with inputs `names`, `is_int` and their columns `cols` (a list of
   character vectors), batches of `rows` rows. */
SEXP zlm_arrow_open(SEXP xstream, SEXP names, SEXP is_int, SEXP cols, SEXP rows, SEXP na_mask, SEXP pad) {
  if (TYPEOF(xstream) != EXTPTRSXP || !Rf_inherits(xstream, "nanoarrow_array_stream")) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "`stream` must be a nanoarrow_array_stream.");
  }
  struct ArrowArrayStream *src = (struct ArrowArrayStream *) R_ExternalPtrAddr(xstream);
  if (src == NULL || src->release == NULL) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The Arrow stream was already consumed or released.");
  }
  zlm_arrow_it *it = (zlm_arrow_it *) calloc(1, sizeof *it);
  if (it == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
  /* the iterator owns the schema from here, and the stream only once the
     map is checked: a refused map leaves the stream with its producer */
  it->done = 1;
  /* the producer's pointer stays reachable from the iterator: what it
     keeps alive (R data a producer's callbacks still read) must outlive
     the stream the iterator takes over (found by gctorture) */
  SEXP ptr = PROTECT(R_MakeExternalPtr(it, it_tag(), xstream));
  R_RegisterCFinalizerEx(ptr, it_finalize, TRUE);
  it->rows = (int64_t) Rf_asReal(rows);
  it->na_mask = Rf_asLogical(na_mask) == TRUE;
  it->pad = Rf_asLogical(pad) == TRUE;

  int rc = src->get_schema(src, &it->schema);
  if (rc != 0) stream_error(src, rc, "reading its schema");
  const struct ArrowSchema *s = &it->schema;
  if (s->format == NULL || strcmp(s->format, "+s") != 0) {
    arrow_error(NULL, -1, "The Arrow stream must hold record batches (a struct type), not '%s'.%s",
                s->format == NULL ? "?" : s->format, "");
  }

  it->n_in = Rf_length(names);
  it->in = (zlm_ainput *) calloc((size_t) it->n_in, sizeof *it->in);
  int total = 0;
  for (int i = 0; i < it->n_in; i++) total += Rf_length(VECTOR_ELT(cols, i));
  it->cols = (zlm_acol *) calloc((size_t) (total > 0 ? total : 1), sizeof *it->cols);
  if (it->in == NULL || it->cols == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");

  for (int i = 0; i < it->n_in; i++) {
    zlm_ainput *in = &it->in[i];
    snprintf(in->name, sizeof in->name, "%s", Rf_translateCharUTF8(STRING_ELT(names, i)));
    in->is_int = LOGICAL(is_int)[i] == TRUE;
    SEXP cv = VECTOR_ELT(cols, i);
    in->nf = Rf_length(cv);
    in->cols = (int *) calloc((size_t) in->nf, sizeof(int));
    if (in->cols == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
    for (int f = 0; f < in->nf; f++) {
      const char *cn = Rf_translateCharUTF8(STRING_ELT(cv, f));
      int child = find_child(s, cn);
      if (child < 0) {
        arrow_error(cn, -1, "The Arrow stream has no column '%s'.%s", cn, "");
      }
      const struct ArrowSchema *cs = s->children[child];
      zlm_acol *c = &it->cols[it->n_cols];
      c->child = child;
      c->name = (char *) malloc(strlen(cn) + 1);
      if (c->name == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
      strcpy(c->name, cn);
      it->n_cols++;
      if (cs->dictionary != NULL) {
        c->kind = K_DICT;
        c->index_kind = index_kind_of(cs->format);
        const char *df = cs->dictionary->format;
        c->large = df != NULL && strcmp(df, "U") == 0;
        if (c->index_kind < 0 || df == NULL ||
            (strcmp(df, "u") != 0 && strcmp(df, "U") != 0)) {
          arrow_error(cn, -1, "Column '%s' is a dictionary whose type zulm cannot read%s; it needs string values and integer codes.", cn, "");
        }
        if (!in->is_int) {
          arrow_error(cn, -1, "Column '%s' is a dictionary, whose codes go to an integer input; declare '%s' with zl_int().", cn, in->name);
        }
      } else {
        c->kind = kind_of(cs->format);
        if (c->kind < 0) {
          arrow_error(cn, -1, "Column '%s' has Arrow type '%s', which zulm does not read.", cn, cs->format == NULL ? "?" : cs->format);
        }
        if (in->is_int && (c->kind == K_F32 || c->kind == K_F64)) {
          arrow_error(cn, -1, "Column '%s' holds floating-point values and cannot fill integer input '%s'.", cn, in->name);
        }
      }
      in->cols[f] = it->n_cols - 1;
    }
  }
  /* move the stream: the producer's struct is marked released, so its own
     finalizer does nothing, and this iterator releases it once */
  it->stream = *src;
  src->release = NULL;
  it->done = 0;
  UNPROTECT(1);
  return ptr;
}

/* ---- chunks ------------------------------------------------------- */

static int dict_equal(const zlm_acol *c, const struct ArrowArray *d) {
  if (d->length != c->dict_n || d->n_buffers < 3) return 0;
  for (int64_t k = 0; k < c->dict_n; k++) {
    int64_t a0, a1;
    if (c->large) {
      const int64_t *o = (const int64_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    } else {
      const int32_t *o = (const int32_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    }
    int64_t n = a1 - a0;
    if (n != c->dict_off[k + 1] - c->dict_off[k]) return 0;
    if (n > 0 && memcmp((const char *) d->buffers[2] + a0, c->dict_data + c->dict_off[k], (size_t) n) != 0) return 0;
  }
  return 1;
}

static void dict_keep(zlm_acol *c, const struct ArrowArray *d) {
  int64_t n = d->length;
  c->dict_off = (int64_t *) calloc((size_t) n + 1, sizeof(int64_t));
  int64_t bytes = 0;
  for (int64_t k = 0; k < n; k++) {
    int64_t a0, a1;
    if (c->large) {
      const int64_t *o = (const int64_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    } else {
      const int32_t *o = (const int32_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    }
    bytes += a1 - a0;
  }
  c->dict_data = (char *) malloc((size_t) (bytes > 0 ? bytes : 1));
  if (c->dict_off == NULL || c->dict_data == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
  int64_t at = 0;
  for (int64_t k = 0; k < n; k++) {
    int64_t a0, a1;
    if (c->large) {
      const int64_t *o = (const int64_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    } else {
      const int32_t *o = (const int32_t *) d->buffers[1];
      a0 = o[d->offset + k];
      a1 = o[d->offset + k + 1];
    }
    c->dict_off[k] = at;
    if (a1 > a0) memcpy(c->dict_data + at, (const char *) d->buffers[2] + a0, (size_t) (a1 - a0));
    at += a1 - a0;
  }
  c->dict_off[n] = at;
  c->dict_n = n;
  c->have_dict = 1;
}

/* Reads the next non-empty chunk; 0 when the stream has ended. */
static int next_chunk(zlm_arrow_it *it) {
  for (;;) {
    if (it->have_chunk) {
      it->chunk.release(&it->chunk);
      it->have_chunk = 0;
    }
    if (it->done) return 0;
    struct ArrowArray a;
    memset(&a, 0, sizeof a);
    int rc = it->stream.get_next(&it->stream, &a);
    if (rc != 0) stream_error(&it->stream, rc, "reading a record batch");
    if (a.release == NULL) {
      it_release_stream(it);
      return 0;
    }
    it->chunk = a;
    it->have_chunk = 1;
    if (a.n_children != it->schema.n_children || a.length < 0) {
      arrow_error(NULL, it->row + 1, "A record batch from row %s does not match the stream's schema.%s", "", "");
    }
    for (int i = 0; i < it->n_cols; i++) {
      zlm_acol *c = &it->cols[i];
      const struct ArrowArray *ch = a.children[c->child];
      if (ch == NULL || ch->n_buffers < 2 || ch->length < a.offset + a.length ||
          (a.length > 0 && ch->buffers[1] == NULL)) {
        arrow_error(c->name, it->row + 1, "Column '%s' of a record batch is malformed.%s", c->name, "");
      }
      if (c->kind == K_DICT) {
        if (ch->dictionary == NULL || ch->dictionary->n_buffers < 3) {
          arrow_error(c->name, it->row + 1, "Column '%s' of a record batch has no dictionary.%s", c->name, "");
        }
        if (!c->have_dict) {
          dict_keep(c, ch->dictionary);
        } else if (!dict_equal(c, ch->dictionary)) {
          char at[32];
          snprintf(at, sizeof at, "%lld", (long long) it->row + 1);
          arrow_error(c->name, it->row + 1,
                      "The dictionary of column '%s' changes at row %s; zulm needs the same levels in every record batch.",
                      c->name, at);
        }
      }
    }
    it->pos = 0;
    it->len = a.length;
    if (it->len > 0) return 1;
  }
}

/* ---- filling ------------------------------------------------------ */

/* Fills up to `rows` rows into dst (float or int32 per input), the masks
   and the row mask (either may be NULL); returns the rows read. */
static int64_t fill_rows(zlm_arrow_it *it, void **dst, float **mdst, float *rowmask) {
  int64_t r = 0;
  for (; r < it->rows; r++) {
    while (!it->have_chunk || it->pos >= it->len) {
      if (!next_chunk(it)) return r;
    }
    const struct ArrowArray *top = &it->chunk;
    int64_t ti = top->offset + it->pos;
    int row_null = top->null_count != 0 && top->n_buffers > 0 && top->buffers[0] != NULL && !bit(top->buffers[0], ti);
    for (int i = 0; i < it->n_in; i++) {
      const zlm_ainput *in = &it->in[i];
      for (int f = 0; f < in->nf; f++) {
        const zlm_acol *c = &it->cols[in->cols[f]];
        const struct ArrowArray *ch = top->children[c->child];
        int64_t k = ch->offset + ti;
        int null = row_null || (ch->null_count != 0 && ch->buffers[0] != NULL && !bit(ch->buffers[0], k));
        int64_t at = f + (int64_t) in->nf * r;
        if (null) {
          if (!it->na_mask) {
            char rowtxt[32];
            snprintf(rowtxt, sizeof rowtxt, "%lld", (long long) it->row + 1);
            arrow_error(c->name, it->row + 1, "Column '%s' is missing at row %s; use na = \"mask\" to train with a mask.", c->name, rowtxt);
          }
          if (in->is_int) ((int32_t *) dst[i])[at] = 0;
          else ((float *) dst[i])[at] = 0.0f;
          if (mdst != NULL) mdst[i][at] = 0.0f;
          continue;
        }
        if (mdst != NULL) mdst[i][at] = 1.0f;
        if (in->is_int) {
          int64_t v;
          if (c->kind == K_BOOL) {
            v = bit(ch->buffers[1], k);
          } else if (c->kind == K_DICT) {
            v = read_int(c->index_kind, ch->buffers[1], k);
            if (v < 0 || v >= c->dict_n) {
              char rowtxt[32];
              snprintf(rowtxt, sizeof rowtxt, "%lld", (long long) it->row + 1);
              arrow_error(c->name, it->row + 1, "Column '%s' has a dictionary code outside its levels at row %s.", c->name, rowtxt);
            }
          } else {
            v = read_int(c->kind, ch->buffers[1], k);
            if (v < INT32_MIN + 1 || v > INT32_MAX) {
              char rowtxt[32];
              snprintf(rowtxt, sizeof rowtxt, "%lld", (long long) it->row + 1);
              arrow_error(c->name, it->row + 1, "Column '%s' holds a value beyond the 32-bit integer range at row %s.", c->name, rowtxt);
            }
          }
          ((int32_t *) dst[i])[at] = (int32_t) v;
        } else {
          float v;
          switch (c->kind) {
            case K_F32: v = ((const float *) ch->buffers[1])[k]; break;
            case K_F64: v = (float) ((const double *) ch->buffers[1])[k]; break;
            case K_BOOL: v = (float) bit(ch->buffers[1], k); break;
            default: v = (float) read_int(c->kind, ch->buffers[1], k); break;
          }
          ((float *) dst[i])[at] = v;
        }
      }
    }
    if (rowmask != NULL) rowmask[r] = 1.0f;
    it->pos++;
    it->row++;
  }
  return r;
}

/* One batch: 1 when filled (padded if allowed), 0 at the end. */
static int fill_batch(zlm_arrow_it *it, void **dst, float **mdst, float *rowmask) {
  if (it->done && !it->have_chunk) return 0;
  int64_t r = fill_rows(it, dst, mdst, rowmask);
  if (r == it->rows) return 1;
  if (!it->pad || r == 0) return 0;
  for (int i = 0; i < it->n_in; i++) {
    const zlm_ainput *in = &it->in[i];
    int64_t from = (int64_t) in->nf * r, to = (int64_t) in->nf * it->rows;
    size_t sz = in->is_int ? sizeof(int32_t) : sizeof(float);
    memset((char *) dst[i] + from * (int64_t) sz, 0, (size_t) (to - from) * sz);
    if (mdst != NULL) memset(mdst[i] + from, 0, (size_t) (to - from) * sizeof(float));
  }
  if (rowmask != NULL) memset(rowmask + r, 0, (size_t) (it->rows - r) * sizeof(float));
  return 1;
}

/* zl_next(): a named list of arrays, or NULL at the end. */
SEXP zlm_arrow_next(SEXP x) {
  zlm_arrow_it *it = it_from(x);
  void **dst = (void **) R_alloc((size_t) it->n_in + 1, sizeof(void *));
  float **mdst = it->na_mask ? (float **) R_alloc((size_t) it->n_in + 1, sizeof(float *)) : NULL;
  for (int i = 0; i < it->n_in; i++) {
    size_t n = (size_t) it->in[i].nf * (size_t) it->rows;
    dst[i] = R_alloc(n, it->in[i].is_int ? sizeof(int32_t) : sizeof(float));
    if (mdst != NULL) mdst[i] = (float *) R_alloc(n, sizeof(float));
  }
  float *rowmask = it->pad ? (float *) R_alloc((size_t) it->rows, sizeof(float)) : NULL;
  if (!fill_batch(it, dst, mdst, rowmask)) return R_NilValue;
  int n_out = it->n_in * (it->na_mask ? 2 : 1) + (it->pad ? 1 : 0);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n_out));
  SEXP nms = PROTECT(Rf_allocVector(STRSXP, n_out));
  int o = 0;
  for (int pass = 0; pass < (it->na_mask ? 2 : 1); pass++) {
    for (int i = 0; i < it->n_in; i++, o++) {
      const zlm_ainput *in = &it->in[i];
      R_xlen_t n = (R_xlen_t) in->nf * (R_xlen_t) it->rows;
      int as_int = pass == 0 && in->is_int;
      SEXP v = Rf_allocVector(as_int ? INTSXP : REALSXP, n);
      SET_VECTOR_ELT(out, o, v);
      if (as_int) {
        memcpy(INTEGER(v), dst[i], (size_t) n * sizeof(int32_t));
      } else {
        const float *s = pass == 0 ? (const float *) dst[i] : mdst[i];
        for (R_xlen_t k = 0; k < n; k++) REAL(v)[k] = (double) s[k];
      }
      if (in->nf > 1 || !in->is_int) {
        SEXP d = PROTECT(Rf_allocVector(INTSXP, 2));
        INTEGER(d)[0] = in->nf;
        INTEGER(d)[1] = (int) it->rows;
        Rf_setAttrib(v, R_DimSymbol, d);
        UNPROTECT(1);
      }
      char buf[80];
      snprintf(buf, sizeof buf, pass == 0 ? "%s" : "%s_mask", in->name);
      SET_STRING_ELT(nms, o, Rf_mkCharCE(buf, CE_UTF8));
    }
  }
  if (it->pad) {
    SEXP v = Rf_allocVector(REALSXP, (R_xlen_t) it->rows);
    SET_VECTOR_ELT(out, o, v);
    for (int64_t k = 0; k < it->rows; k++) REAL(v)[k] = (double) rowmask[k];
    SET_STRING_ELT(nms, o, Rf_mkChar("row_mask"));
  }
  Rf_setAttrib(out, R_NamesSymbol, nms);
  UNPROTECT(2);
  return out;
}

/* ---- into a graph -------------------------------------------------- */

static struct ggml_tensor *graph_input(zlm_graph *g, const char *name) {
  for (int i = 0; i < g->n_in; i++) {
    if (strcmp(g->in[i]->name, name) == 0) return g->in[i];
  }
  return NULL;
}

static struct ggml_tensor *need_input(zlm_graph *g, const char *name, int is_int, int64_t n, int64_t nf) {
  struct ggml_tensor *t = graph_input(g, name);
  if (t == NULL) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The graph has no input '%s', which the iterator fills.", name);
  }
  if ((t->type == GGML_TYPE_I32) != is_int) {
    zlm_raise("zulm_type_error", R_NilValue, "Input '%s' is %s; the iterator fills it with %s values.", name,
              t->type == GGML_TYPE_I32 ? "i32" : "f32", is_int ? "integer" : "f32");
  }
  if (ggml_nelements(t) != n || (t->ne[0] != nf && !(nf == 1 && t->ne[0] == n))) {
    zlm_raise("zulm_shape_error", R_NilValue,
              "Input '%s' does not have the iterator's shape, %lld features by %lld rows.", name,
              (long long) nf, (long long) (n / nf));
  }
  return t;
}

/* The C table's batch_fill (design §8, registered in Stage 12): the next
   batch straight into the graph's inputs; 0 when filled, 1 at the end.
   Main thread only; errors are R errors. */
int zlm_api_batch_fill(zlm_graph *g, SEXP iterator) {
  zlm_arrow_it *it = it_from(iterator);
  int expected = it->n_in * (it->na_mask ? 2 : 1) + (it->pad ? 1 : 0);
  if (g->n_in != expected) {
    zlm_raise("zulm_invalid_argument", R_NilValue,
              "The graph has %d inputs; the iterator fills %d.", g->n_in, expected);
  }
  void **dst = (void **) R_alloc((size_t) it->n_in + 1, sizeof(void *));
  float **mdst = it->na_mask ? (float **) R_alloc((size_t) it->n_in + 1, sizeof(float *)) : NULL;
  for (int i = 0; i < it->n_in; i++) {
    const zlm_ainput *in = &it->in[i];
    int64_t n = (int64_t) in->nf * it->rows;
    dst[i] = need_input(g, in->name, in->is_int, n, in->nf)->data;
    if (mdst != NULL) {
      char buf[80];
      snprintf(buf, sizeof buf, "%s_mask", in->name);
      mdst[i] = (float *) need_input(g, buf, 0, n, in->nf)->data;
    }
  }
  float *rowmask = it->pad ? (float *) need_input(g, "row_mask", 0, it->rows, 1)->data : NULL;
  return fill_batch(it, dst, mdst, rowmask) ? 0 : 1;
}

SEXP zlm_batch_fill(SEXP gptr, SEXP x) {
  zlm_graph *g = zlm_graph_from(gptr);
  if (g->state != ZLM_GRAPH_READY) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "This graph is not ready.");
  }
  return Rf_ScalarLogical(zlm_api_batch_fill(g, x) == 0);
}

/* Frees an iterator now, releasing what it still holds of the stream;
   for the fuzz target, which cannot wait for the garbage collector. */
SEXP zlm_arrow_close(SEXP x) {
  if (TYPEOF(x) == EXTPTRSXP && R_ExternalPtrTag(x) == it_tag()) it_finalize(x);
  return R_NilValue;
}

/* list(rows read, ended) for print() and tests. */
SEXP zlm_arrow_state(SEXP x) {
  zlm_arrow_it *it = it_from(x);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, Rf_ScalarReal((double) it->row));
  SET_VECTOR_ELT(out, 1, Rf_ScalarLogical(it->done));
  UNPROTECT(1);
  return out;
}

/* ---- a counting stream, for the tests ------------------------------ */

/* Batches of one float32 column "x" (0, 1, 2, …) whose stream and array
   release callbacks count their calls into an R integer vector, so a test
   can see each called exactly once. */
typedef struct {
  int *counts; /* [0]: stream releases, [1]: array releases */
  int batches_left, batch_rows;
  float next;
} zlm_test_stream;

typedef struct {
  int *counts;
  const void *buffers[2];
  float *values;
} zlm_test_array;

static void test_array_release(struct ArrowArray *a) {
  zlm_test_array *p = (zlm_test_array *) a->private_data;
  p->counts[1]++;
  free(p->values);
  free(p);
  a->release = NULL;
}

/* A released schema is one whose release is NULL; this one owns nothing. */
static void test_schema_release(struct ArrowSchema *s) {
  s->release = NULL;
}

static int test_get_schema(struct ArrowArrayStream *s, struct ArrowSchema *out) {
  (void) s;
  /* struct<x: float32>, from static storage */
  static struct ArrowSchema child = {"f", "x", NULL, ARROW_FLAG_NULLABLE, 0, NULL, NULL, NULL, NULL};
  static struct ArrowSchema *children[1] = {&child};
  memset(out, 0, sizeof *out);
  out->format = "+s";
  out->name = "";
  out->n_children = 1;
  out->children = children;
  out->release = test_schema_release;
  return 0;
}

typedef struct {
  struct ArrowArray top, child;
  struct ArrowArray *children[1];
  const void *top_buffers[1];
} zlm_test_batch;

static void test_top_release(struct ArrowArray *a) {
  zlm_test_batch *b = (zlm_test_batch *) a->private_data;
  if (b->child.release != NULL) b->child.release(&b->child);
  free(b);
  a->release = NULL;
}

static int test_get_next(struct ArrowArrayStream *s, struct ArrowArray *out) {
  zlm_test_stream *p = (zlm_test_stream *) s->private_data;
  memset(out, 0, sizeof *out);
  if (p->batches_left == 0) return 0; /* released: the end */
  zlm_test_batch *b = (zlm_test_batch *) calloc(1, sizeof *b);
  zlm_test_array *ca = (zlm_test_array *) calloc(1, sizeof *ca);
  float *v = (float *) calloc((size_t) (p->batch_rows > 0 ? p->batch_rows : 1), sizeof(float));
  if (b == NULL || ca == NULL || v == NULL) {
    free(b);
    free(ca);
    free(v);
    return 12; /* ENOMEM */
  }
  for (int i = 0; i < p->batch_rows; i++) v[i] = p->next++;
  ca->counts = p->counts;
  ca->values = v;
  ca->buffers[0] = NULL;
  ca->buffers[1] = v;
  b->child.length = p->batch_rows;
  b->child.n_buffers = 2;
  b->child.buffers = ca->buffers;
  b->child.private_data = ca;
  b->child.release = test_array_release;
  b->children[0] = &b->child;
  b->top_buffers[0] = NULL;
  out->length = p->batch_rows;
  out->n_buffers = 1;
  out->buffers = b->top_buffers;
  out->n_children = 1;
  out->children = b->children;
  out->private_data = b;
  out->release = test_top_release;
  p->batches_left--;
  return 0;
}

static const char *test_last_error(struct ArrowArrayStream *s) {
  (void) s;
  return NULL;
}

static void test_stream_release(struct ArrowArrayStream *s) {
  zlm_test_stream *p = (zlm_test_stream *) s->private_data;
  p->counts[0]++;
  free(p);
  s->release = NULL;
}

static void test_stream_finalize(SEXP x) {
  struct ArrowArrayStream *s = (struct ArrowArrayStream *) R_ExternalPtrAddr(x);
  if (s == NULL) return;
  if (s->release != NULL) s->release(s);
  free(s);
  R_ClearExternalPtr(x);
}

SEXP zlm_arrow_test_stream(SEXP batches, SEXP batch_rows, SEXP counts) {
  struct ArrowArrayStream *s = (struct ArrowArrayStream *) calloc(1, sizeof *s);
  zlm_test_stream *p = (zlm_test_stream *) calloc(1, sizeof *p);
  if (s == NULL || p == NULL) {
    free(s);
    free(p);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
  }
  p->counts = INTEGER(counts);
  p->batches_left = Rf_asInteger(batches);
  p->batch_rows = Rf_asInteger(batch_rows);
  s->get_schema = test_get_schema;
  s->get_next = test_get_next;
  s->get_last_error = test_last_error;
  s->release = test_stream_release;
  s->private_data = p;
  /* the counts vector stays alive with the stream */
  SEXP ptr = PROTECT(R_MakeExternalPtr(s, R_NilValue, counts));
  R_RegisterCFinalizerEx(ptr, test_stream_finalize, TRUE);
  SEXP cls = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(cls, 0, Rf_mkChar("nanoarrow_array_stream"));
  Rf_setAttrib(ptr, R_ClassSymbol, cls);
  UNPROTECT(2);
  return ptr;
}
