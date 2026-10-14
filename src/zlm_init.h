/*
 * zlm_init.h: the R_init_ function of one variant shared object
 * (design §13.1). Each zlm_v/<variant>/zlm_init.c defines ZLM_VARIANT as the
 * variant's name and includes this file, so the function is
 * R_init_zulm_<variant>, the only symbol the object exports.
 */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

#include "ggml.h"
#include "zlm_glue.h"

SEXP zlm_native_info(void);
SEXP zlm_log_drain(void);
SEXP zlm_engine_selftest(SEXP valid);
SEXP zlm_bench_mul_mat(SEXP n, SEXP reps, SEXP threads);

#define ZLM_STR2(x) #x
#define ZLM_STR(x) ZLM_STR2(x)
#define ZLM_CAT2(a, b) a##b
#define ZLM_CAT(a, b) ZLM_CAT2(a, b)

static const R_CallMethodDef zlm_call_methods[] = {
  {"zlm_native_info", (DL_FUNC) &zlm_native_info, 0},
  {"zlm_log_drain", (DL_FUNC) &zlm_log_drain, 0},
  {"zlm_engine_selftest", (DL_FUNC) &zlm_engine_selftest, 1},
  {"zlm_bench_mul_mat", (DL_FUNC) &zlm_bench_mul_mat, 3},
  {NULL, NULL, 0}
};

void attribute_visible ZLM_CAT(R_init_zulm_, ZLM_VARIANT)(DllInfo *dll) {
  R_registerRoutines(dll, NULL, zlm_call_methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
  zlm_variant_name = ZLM_STR(ZLM_VARIANT);
  ggml_set_abort_callback(zlm_abort_callback);
}
