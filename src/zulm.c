/*
 * zulm.c: the package's registered DLL, the probe of design §13.1.
 *
 * It holds no ggml code. .onLoad asks it which variants this CPU can run,
 * best first, and loads one of the variant shared objects beside it. CPUID
 * checks the CPU and XGETBV that the operating system saves the vector
 * registers AVX needs.
 */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>

/* XCR0: which register states the operating system saves. */
static unsigned long long zlm_xgetbv(void) {
  unsigned int eax, edx;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return ((unsigned long long) edx << 32) | eax;
}

/* The CPU features of each x86-64 variant, read with CPUID. Not
   __builtin_cpu_supports(): older compilers (clang 14) do not know every
   feature name, such as "f16c". */
static void zlm_x86_features(int *sse42, int *avx2) {
  unsigned int a, b, c, d;
  *sse42 = *avx2 = 0;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return;
  int sse3 = (c >> 0) & 1, ssse3 = (c >> 9) & 1, sse41 = (c >> 19) & 1;
  int sse42_ = (c >> 20) & 1, popcnt = (c >> 23) & 1, fma = (c >> 12) & 1;
  int osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, f16c = (c >> 29) & 1;
  *sse42 = sse3 && ssse3 && sse41 && sse42_ && popcnt;
  /* AVX needs the OS to save the XMM and YMM registers (XCR0 bits 1, 2). */
  int os_avx = osxsave && (zlm_xgetbv() & 6) == 6;
  if (!(*sse42 && avx && fma && f16c && os_avx)) return;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return;
  int avx2_ = (b >> 5) & 1, bmi1 = (b >> 3) & 1, bmi2 = (b >> 8) & 1;
  *avx2 = avx2_ && bmi1 && bmi2;
}
#endif

static SEXP zlm_cpu_variants(void) {
  const char *v[4];
  int n = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  int sse42 = 0, avx2 = 0;
  zlm_x86_features(&sse42, &avx2);
  if (avx2) {
    v[n++] = "avx2";
  }
  if (sse42) {
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
