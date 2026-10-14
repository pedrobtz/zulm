/*
 * zlm_glue.c: the .Call entry points of a variant shared object.
 *
 * Stage 1 surface: build information, the log drain, the engine-error
 * self-test of design §13.2, and the product benchmark of §12 used for the
 * §13.1 measurements. Later stages add the R API proper.
 */
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-version.h"
#include "zlm_glue.h"

#if defined(_WIN32)
#include <windows.h>
#endif

const char *zlm_variant_name = "unknown";

static SEXP zlm_named_list(int n, const char **names) {
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, n));
  for (int i = 0; i < n; i++) {
    SET_STRING_ELT(nm, i, Rf_mkChar(names[i]));
  }
  Rf_setAttrib(out, R_NamesSymbol, nm);
  UNPROTECT(2);
  return out;
}

SEXP zlm_native_info(void) {
  static const char *names[] = {
    "ggml_version", "ggml_commit", "ggml_patches", "variant", "compiler",
    "sse3", "ssse3", "avx", "avx2", "f16c", "fma", "bmi2", "avx512",
    "neon", "arm_fma", "fp16_va", "dotprod", "matmul_int8", "sve",
    "llamafile", "cpu_repack", "size_t_bytes"
  };
  int n = (int) (sizeof names / sizeof names[0]);
  SEXP out = PROTECT(zlm_named_list(n, names));
  int i = 0;
  SET_VECTOR_ELT(out, i++, Rf_mkString(ggml_version()));
  SET_VECTOR_ELT(out, i++, Rf_mkString(ggml_commit()));
  SET_VECTOR_ELT(out, i++, Rf_mkString(ZULM_GGML_PATCHES));
  SET_VECTOR_ELT(out, i++, Rf_mkString(zlm_variant_name));
#if defined(__clang__)
  SET_VECTOR_ELT(out, i++, Rf_mkString("clang " __clang_version__));
#elif defined(__GNUC__)
  SET_VECTOR_ELT(out, i++, Rf_mkString("gcc " __VERSION__));
#else
  SET_VECTOR_ELT(out, i++, Rf_mkString("unknown"));
#endif
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_sse3()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_ssse3()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_avx()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_avx2()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_f16c()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_fma()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_bmi2()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_avx512()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_neon()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_arm_fma()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_fp16_va()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_dotprod()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_matmul_int8()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_sve()));
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(ggml_cpu_has_llamafile()));
#if defined(GGML_USE_CPU_REPACK)
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(1));
#else
  SET_VECTOR_ELT(out, i++, Rf_ScalarLogical(0));
#endif
  SET_VECTOR_ELT(out, i++, Rf_ScalarInteger((int) sizeof(size_t)));
  UNPROTECT(1);
  return out;
}

/* list(levels, texts) of the buffered log records, oldest first, with a
   "dropped" attribute when the buffer overflowed. */
SEXP zlm_log_drain(void) {
  static char buf[65536];
  int dropped = 0;
  size_t n = zlm_log_take(buf, sizeof buf, &dropped);
  int count = 0;
  for (size_t i = 0; i < n; i++) count += buf[i] == '\x1e';
  SEXP levels = PROTECT(Rf_allocVector(INTSXP, count));
  SEXP texts = PROTECT(Rf_allocVector(STRSXP, count));
  int k = -1;
  size_t start = 0;
  for (size_t i = 0; i <= n; i++) {
    if (i == n || buf[i] == '\x1e') {
      if (k >= 0) {
        SET_STRING_ELT(texts, k, Rf_mkCharLenCE(buf + start, (int) (i - start), CE_UTF8));
      }
      if (i < n) {
        k++;
        INTEGER(levels)[k] = i + 1 < n ? buf[i + 1] - '0' : 3;
        start = i + 2 <= n ? i + 2 : n;
        i++;
      }
    }
  }
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, levels);
  SET_VECTOR_ELT(out, 1, texts);
  Rf_setAttrib(out, Rf_install("dropped"), Rf_ScalarLogical(dropped));
  UNPROTECT(3);
  return out;
}

/* Internal: one record through ggml's own logger, for the tests. */
void ggml_log_internal(enum ggml_log_level level, const char *format, ...);

SEXP zlm_log_emit(SEXP level, SEXP text) {
  ggml_log_internal((enum ggml_log_level) Rf_asInteger(level), "%s", CHAR(STRING_ELT(text, 0)));
  return R_NilValue;
}

SEXP zlm_log_pending_(void) {
  return Rf_ScalarLogical(zlm_log_pending());
}

/* Design §13.2: a construction ggml asserts on, run under the guard.
   ggml_add() asserts ggml_can_repeat(b, a); a 4 x 3 plus a 5 x 2 fails. */
struct zlm_selftest {
  struct ggml_context *ctx;
};

static void zlm_selftest_body(void *data) {
  struct zlm_selftest *st = (struct zlm_selftest *) data;
  struct ggml_tensor *a = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, 4, 3);
  struct ggml_tensor *b = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, 5, 2);
  ggml_add(st->ctx, a, b);
}

SEXP zlm_engine_selftest(SEXP valid) {
  struct ggml_init_params params = { 1 << 20, NULL, true };
  struct zlm_selftest st = { ggml_init(params) };
  if (st.ctx == NULL) {
    Rf_error("ggml_init failed");
  }
  char msg[2048] = "";
  int status;
  if (Rf_asLogical(valid) == 1) {
    /* A valid construction under the guard, to show the guard is cheap. */
    struct ggml_tensor *a = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, 4, 3);
    ggml_add(st.ctx, a, a);
    status = ZLM_STATUS_OK;
  } else {
    status = zlm_guarded(zlm_selftest_body, &st, msg, sizeof msg);
  }
  ggml_free(st.ctx);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(status));
  SET_VECTOR_ELT(out, 1, Rf_mkString(msg));
  UNPROTECT(1);
  return out;
}

static double zlm_now(void) {
#if defined(_WIN32)
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double) c.QuadPart / (double) f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
#endif
}

/* Design §12: an n x n by n x n f32 product, reps times, on threads
   threads. Returns seconds per product, GFLOP/s, and an FNV-1a hash of
   the result's bits, so thread invariance can be checked. */
SEXP zlm_bench_mul_mat(SEXP n_, SEXP reps_, SEXP threads_) {
  int64_t n = (int64_t) Rf_asInteger(n_);
  int reps = Rf_asInteger(reps_);
  int threads = Rf_asInteger(threads_);
  if (n < 1 || n > 8192 || reps < 1 || threads < 1 || threads > 256) {
    Rf_error("invalid benchmark arguments");
  }
  size_t mem = (size_t) (3 * n * n) * sizeof(float) + ggml_graph_overhead() +
               (size_t) 64 * 1024 * 1024;
  struct ggml_init_params params = { mem, NULL, false };
  struct ggml_context *ctx = ggml_init(params);
  if (ctx == NULL) {
    Rf_error("ggml_init failed");
  }
  struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
  struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
  float *pa = (float *) a->data;
  float *pb = (float *) b->data;
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  for (int64_t i = 0; i < n * n; i++) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    pa[i] = (float) ((double) (s >> 40) / (double) (1 << 24) - 0.5);
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    pb[i] = (float) ((double) (s >> 40) / (double) (1 << 24) - 0.5);
  }
  struct ggml_tensor *c = ggml_mul_mat(ctx, a, b);
  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, c);
  /* one warm-up run */
  ggml_graph_compute_with_ctx(ctx, gf, threads);
  double t0 = zlm_now();
  for (int r = 0; r < reps; r++) {
    ggml_graph_compute_with_ctx(ctx, gf, threads);
  }
  double secs = (zlm_now() - t0) / reps;
  uint32_t h = 2166136261u;
  const unsigned char *bytes = (const unsigned char *) c->data;
  for (int64_t i = 0; i < n * n * (int64_t) sizeof(float); i++) {
    h ^= bytes[i];
    h *= 16777619u;
  }
  ggml_free(ctx);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, 3));
  REAL(out)[0] = secs;
  REAL(out)[1] = 2.0 * (double) n * (double) n * (double) n / secs / 1e9;
  REAL(out)[2] = (double) h;
  UNPROTECT(1);
  return out;
}
