/*
 * zulm.c: the package's registered DLL, the probe of design §13.1.
 *
 * It holds no ggml code. .onLoad asks it which variants this CPU can run,
 * best first, and loads one of the variant shared objects beside it.
 * __builtin_cpu_supports() checks the CPU and, for AVX, that the operating
 * system saves the vector registers.
 */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

static SEXP zlm_cpu_variants(void) {
  const char *v[4];
  int n = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
      __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("f16c")) {
    v[n++] = "avx2";
  }
  if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("popcnt")) {
    v[n++] = "sse42";
  }
#endif
  v[n++] = "baseline";
#elif defined(__aarch64__) || defined(_M_ARM64)
  v[n++] = "neon";
#else
  v[n++] = "generic";
#endif
  SEXP out = PROTECT(Rf_allocVector(STRSXP, n));
  for (int i = 0; i < n; i++) {
    SET_STRING_ELT(out, i, Rf_mkChar(v[i]));
  }
  UNPROTECT(1);
  return out;
}

static const R_CallMethodDef call_methods[] = {
  {"zlm_cpu_variants", (DL_FUNC) &zlm_cpu_variants, 0},
  {NULL, NULL, 0}
};

void attribute_visible R_init_zulm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
