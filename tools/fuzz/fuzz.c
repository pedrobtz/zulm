/*
 * fuzz.c: zulm's libFuzzer targets (design §11, §15 criterion 4).
 *
 * The target embeds R and drives zulm through its own R functions, so R's
 * validation is part of what is fuzzed: the glue and one variant's ggml
 * are compiled into this executable, instrumented, and the package's
 * R files are sourced into a stand-in "zulm" namespace whose routines
 * are this executable's. tools/fuzz/drive.R turns each input into calls.
 *
 * Built once per target, with -DZLM_FUZZ_TARGET=1 (graph construction,
 * random operation sequences and shapes, forward and training), 2 (GGUF
 * and the safetensors header reader on arbitrary bytes) or 3 (the Arrow
 * type map, on streams this file builds from the input). A finding is a
 * crash, a sanitizer report, a ggml assertion (caught in a builder, it is
 * a zulm_engine_error; in a kernel it aborts) or an R error without a
 * zulm condition class.
 *
 * Run from the package root; tools/run-fuzz builds and runs it locally.
 */
#define CSTACK_DEFNS 1 /* R_CStackLimit, from Rinterface.h */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/zlm_init.h" /* the variant's routine table, R_init_zulm_<v> */
#include "../../src/zlm_arrow_abi.h"

#include <R_ext/Parse.h>
#include <Rembedded.h>
#include <Rinterface.h>

#ifndef ZLM_FUZZ_TARGET
#define ZLM_FUZZ_TARGET 1
#endif

/* ---- Arrow streams from bytes (target 3) ----------------------------- */

/* A stream of record batches with one to three columns, each of a type
   picked from the input, nullable or not, sometimes dictionary-encoded.
   Every array is well formed (buffers as long as length + offset need):
   the values, nulls, lengths, offsets, dictionaries (which may change
   between batches) and their codes are the input's. */

typedef struct {
  const uint8_t *p;
  size_t n, i;
} fz_in;

static uint8_t u8(fz_in *in) {
  return in->i < in->n ? in->p[in->i++] : 0;
}

static const char *const fz_formats[] = {
  "f", "g", "e", "i", "l", "s", "c", "C", "S", "I", "L", "b", "n", "u", "tdD", "+s"
};
#define FZ_N_FORMATS ((int) (sizeof fz_formats / sizeof fz_formats[0]))

static int64_t fz_width(const char *fmt) {
  switch (fmt[0]) {
  case 'g': case 'l': case 'L': return 8;
  case 'f': case 'i': case 'I': return 4;
  case 'e': case 's': case 'S': return 2;
  case 'c': case 'C': return 1;
  case 't': return 4;
  default: return 0;
  }
}

typedef struct {
  int n_cols;
  int fmt[3];
  int dict[3]; /* a dictionary of strings, its codes in fmt */
  int nullable[3];
  int batches_left;
  fz_in in;
  uint8_t *copy;
} fz_stream;

static void release_schema(struct ArrowSchema *s) {
  for (int64_t i = 0; i < s->n_children; i++) {
    if (s->children[i]->release) s->children[i]->release(s->children[i]);
    free(s->children[i]);
  }
  free(s->children);
  if (s->dictionary) {
    if (s->dictionary->release) s->dictionary->release(s->dictionary);
    free(s->dictionary);
  }
  free((void *) s->name);
  s->release = NULL;
}

static void schema_leaf(struct ArrowSchema *s, const char *fmt, const char *name, int nullable) {
  memset(s, 0, sizeof *s);
  s->format = fmt;
  s->name = name ? strdup(name) : NULL;
  s->flags = nullable ? 2 : 0;
  s->release = release_schema;
}

static int fz_get_schema(struct ArrowArrayStream *st, struct ArrowSchema *out) {
  fz_stream *z = (fz_stream *) st->private_data;
  schema_leaf(out, "+s", NULL, 0);
  out->n_children = z->n_cols;
  out->children = (struct ArrowSchema **) calloc((size_t) z->n_cols, sizeof *out->children);
  static const char *names[] = { "a", "b", "c" };
  for (int k = 0; k < z->n_cols; k++) {
    out->children[k] = (struct ArrowSchema *) malloc(sizeof(struct ArrowSchema));
    const char *fmt = fz_formats[z->fmt[k]];
    schema_leaf(out->children[k], fmt, names[k], z->nullable[k]);
    if (z->dict[k]) {
      out->children[k]->dictionary = (struct ArrowSchema *) malloc(sizeof(struct ArrowSchema));
      schema_leaf(out->children[k]->dictionary, "u", NULL, 0);
    }
  }
  return 0;
}

static void release_array(struct ArrowArray *a) {
  for (int64_t i = 0; i < a->n_children; i++) {
    if (a->children[i]->release) a->children[i]->release(a->children[i]);
    free(a->children[i]);
  }
  free(a->children);
  if (a->dictionary) {
    if (a->dictionary->release) a->dictionary->release(a->dictionary);
    free(a->dictionary);
  }
  for (int64_t i = 0; i < a->n_buffers; i++) free((void *) a->buffers[i]);
  free(a->buffers);
  a->release = NULL;
}

/* An array of `fmt` with `len` elements after `off`, its bytes from the
   input. */
static void array_leaf(struct ArrowArray *a, fz_in *in, const char *fmt, int64_t len, int64_t off,
                       int nullable) {
  memset(a, 0, sizeof *a);
  a->length = len;
  a->offset = off;
  a->release = release_array;
  int64_t total = len + off;
  if (fmt[0] == 'n') {
    a->n_buffers = 0;
    a->null_count = len;
    return;
  }
  if (fmt[0] == '+') { /* a nested struct: refused by the type map */
    a->n_buffers = 1;
    a->buffers = (const void **) calloc(1, sizeof(void *));
    return;
  }
  int is_utf8 = fmt[0] == 'u';
  a->n_buffers = is_utf8 ? 3 : 2;
  a->buffers = (const void **) calloc((size_t) a->n_buffers, sizeof(void *));
  if (nullable && (u8(in) & 1)) {
    size_t nb = (size_t) (total + 7) / 8 + 1;
    uint8_t *v = (uint8_t *) malloc(nb);
    for (size_t i = 0; i < nb; i++) v[i] = u8(in);
    a->buffers[0] = v;
    int64_t nulls = 0;
    for (int64_t i = off; i < total; i++) nulls += !((v[i / 8] >> (i % 8)) & 1);
    /* sometimes unknown, as producers may say */
    a->null_count = u8(in) % 4 == 0 ? -1 : nulls;
  }
  if (fmt[0] == 'b') {
    size_t nb = (size_t) (total + 7) / 8 + 1;
    uint8_t *v = (uint8_t *) malloc(nb);
    for (size_t i = 0; i < nb; i++) v[i] = u8(in);
    a->buffers[1] = v;
    return;
  }
  if (is_utf8) {
    int32_t *o = (int32_t *) malloc((size_t) (total + 1) * sizeof(int32_t));
    o[0] = 0;
    for (int64_t i = 0; i < total; i++) o[i + 1] = o[i] + u8(in) % 3;
    char *d = (char *) malloc((size_t) o[total] + 1);
    for (int32_t i = 0; i < o[total]; i++) d[i] = (char) ('a' + u8(in) % 26);
    a->buffers[1] = o;
    a->buffers[2] = d;
    return;
  }
  int64_t w = fz_width(fmt);
  size_t nb = (size_t) (total * w) + 1;
  uint8_t *d = (uint8_t *) malloc(nb);
  for (size_t i = 0; i < nb; i++) d[i] = u8(in);
  a->buffers[1] = d;
}

static int fz_get_next(struct ArrowArrayStream *st, struct ArrowArray *out) {
  fz_stream *z = (fz_stream *) st->private_data;
  memset(out, 0, sizeof *out);
  if (z->batches_left-- <= 0) return 0; /* release NULL: the end */
  fz_in *in = &z->in;
  int64_t len = u8(in) % 9;
  int64_t off = u8(in) % 3;
  out->length = len;
  out->n_buffers = 1;
  out->buffers = (const void **) calloc(1, sizeof(void *));
  out->n_children = z->n_cols;
  out->children = (struct ArrowArray **) calloc((size_t) z->n_cols, sizeof *out->children);
  out->release = release_array;
  for (int k = 0; k < z->n_cols; k++) {
    out->children[k] = (struct ArrowArray *) malloc(sizeof(struct ArrowArray));
    const char *fmt = fz_formats[z->fmt[k]];
    array_leaf(out->children[k], in, fmt, len, off, z->nullable[k]);
    if (z->dict[k]) {
      int64_t dl = u8(in) % 5;
      out->children[k]->dictionary = (struct ArrowArray *) malloc(sizeof(struct ArrowArray));
      array_leaf(out->children[k]->dictionary, in, "u", dl, 0, 0);
    }
  }
  return 0;
}

static const char *fz_last_error(struct ArrowArrayStream *st) {
  (void) st;
  return "fuzz stream";
}

static void fz_release(struct ArrowArrayStream *st) {
  fz_stream *z = (fz_stream *) st->private_data;
  if (z) {
    free(z->copy);
    free(z);
  }
  st->release = NULL;
}

static void fz_finalize(SEXP x) {
  struct ArrowArrayStream *st = (struct ArrowArrayStream *) R_ExternalPtrAddr(x);
  if (st == NULL) return;
  if (st->release) st->release(st);
  free(st);
  R_ClearExternalPtr(x);
}

/* zlm_fuzz_stream(raw): a nanoarrow_array_stream built from the bytes. */
static SEXP zlm_fuzz_stream(SEXP raw) {
  fz_stream *z = (fz_stream *) calloc(1, sizeof *z);
  struct ArrowArrayStream *st = (struct ArrowArrayStream *) calloc(1, sizeof *st);
  size_t n = (size_t) Rf_xlength(raw);
  z->copy = (uint8_t *) malloc(n + 1);
  memcpy(z->copy, RAW(raw), n);
  z->in.p = z->copy;
  z->in.n = n;
  fz_in *in = &z->in;
  z->n_cols = 1 + u8(in) % 3;
  for (int k = 0; k < z->n_cols; k++) {
    uint8_t b = u8(in);
    z->fmt[k] = b % FZ_N_FORMATS;
    z->nullable[k] = (b >> 5) & 1;
    z->dict[k] = ((b >> 6) & 3) == 3 && fz_width(fz_formats[z->fmt[k]]) > 0 &&
                 strchr("csilCSIL", fz_formats[z->fmt[k]][0]) != NULL;
  }
  z->batches_left = u8(in) % 6;
  st->get_schema = fz_get_schema;
  st->get_next = fz_get_next;
  st->get_last_error = fz_last_error;
  st->release = fz_release;
  st->private_data = z;
  SEXP ptr = PROTECT(R_MakeExternalPtr(st, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ptr, fz_finalize, TRUE);
  Rf_setAttrib(ptr, R_ClassSymbol, Rf_mkString("nanoarrow_array_stream"));
  /* what the type map is asked for: the column formats, for drive.R */
  SEXP fmts = PROTECT(Rf_allocVector(STRSXP, z->n_cols));
  for (int k = 0; k < z->n_cols; k++) SET_STRING_ELT(fmts, k, Rf_mkChar(fz_formats[z->fmt[k]]));
  Rf_setAttrib(ptr, Rf_install("formats"), fmts);
  UNPROTECT(2);
  return ptr;
}

/* ---- embedding ------------------------------------------------------- */

static SEXP zlm_fuzz_abort(void) {
  abort();
}

static int fz_ready = 0;

static void fz_eval_text(const char *text) {
  ParseStatus st;
  SEXP src = PROTECT(Rf_mkString(text));
  SEXP ex = PROTECT(R_ParseVector(src, -1, &st, R_NilValue));
  if (st != PARSE_OK) {
    fprintf(stderr, "fuzz: parse error in setup\n");
    abort();
  }
  for (R_xlen_t i = 0; i < Rf_xlength(ex); i++) {
    int err = 0;
    R_tryEval(VECTOR_ELT(ex, i), R_GlobalEnv, &err);
    if (err) {
      fprintf(stderr, "fuzz: setup failed\n");
      abort();
    }
  }
  UNPROTECT(2);
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void) argc;
  (void) argv;
  char *rargv[] = { "R", "--vanilla", "--silent", "--no-echo" };
  if (!getenv("R_HOME")) {
    fprintf(stderr, "fuzz: R_HOME is not set\n");
    abort();
  }
  R_SignalHandlers = 0;
  Rf_initEmbeddedR(4, rargv);
  R_CStackLimit = (uintptr_t) -1;

  /* the variant's routines and two of the harness's own, registered on
     the embedding DLL, where drive.R finds them by name */
  int n = 0;
  while (zlm_call_methods[n].name != NULL) n++;
  R_CallMethodDef *all = (R_CallMethodDef *) calloc((size_t) n + 3, sizeof *all);
  memcpy(all, zlm_call_methods, (size_t) n * sizeof *all);
  all[n].name = "zlm_fuzz_stream";
  all[n].fun = (DL_FUNC) &zlm_fuzz_stream;
  all[n].numArgs = 1;
  all[n + 1].name = "zlm_fuzz_abort";
  all[n + 1].fun = (DL_FUNC) &zlm_fuzz_abort;
  all[n + 1].numArgs = 0;
  DllInfo *dll = R_getEmbeddingDllInfo();
  R_registerRoutines(dll, NULL, all, NULL, NULL);
  zlm_variant_name = ZLM_STR(ZLM_VARIANT);
  ggml_set_abort_callback(zlm_abort_callback);
  ggml_log_set(zlm_ggml_log, NULL);

  fz_eval_text("source('tools/fuzz/setup.R'); zlm_fuzz_setup('" ZLM_STR(ZLM_VARIANT) "')");
  fz_ready = 1;
  return 0;
}

struct fz_job {
  SEXP call;
  SEXP result;
};

static void fz_run(void *data) {
  struct fz_job *j = (struct fz_job *) data;
  j->result = Rf_eval(j->call, R_GlobalEnv);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (!fz_ready || size > 4096) return 0;
  static const char *fn[] = { "", "fuzz_graph", "fuzz_files", "fuzz_arrow" };
  SEXP raw = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) size));
  if (size > 0) memcpy(RAW(raw), data, size);
  struct fz_job j = { PROTECT(Rf_lang2(Rf_install(fn[ZLM_FUZZ_TARGET]), raw)), R_NilValue };
  /* drive.R catches zulm's conditions and calls zlm_fuzz_abort() on a
     finding; anything that escapes it is a harness error, also fatal */
  if (!R_ToplevelExec(fz_run, &j)) {
    fprintf(stderr, "fuzz: an R error escaped drive.R\n");
    abort();
  }
  UNPROTECT(2);
  return 0;
}
