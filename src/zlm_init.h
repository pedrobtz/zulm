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
#include "zlm_store.h"
#include "zlm_graph.h"
#include "zlm_train.h"
#include "zlm_rng.h"
#include "zlm_pool.h"

SEXP zlm_native_info(void);
SEXP zlm_log_drain(void);
SEXP zlm_log_pending_(void);
SEXP zlm_log_emit(SEXP level, SEXP text);
void zlm_ggml_log(enum ggml_log_level level, const char *text, void *user);
SEXP zlm_engine_selftest(SEXP valid);
SEXP zlm_bench_mul_mat(SEXP n, SEXP reps, SEXP threads);

#define ZLM_STR2(x) #x
#define ZLM_STR(x) ZLM_STR2(x)
#define ZLM_CAT2(a, b) a##b
#define ZLM_CAT(a, b) ZLM_CAT2(a, b)

static const R_CallMethodDef zlm_call_methods[] = {
  {"zlm_native_info", (DL_FUNC) &zlm_native_info, 0},
  {"zlm_log_drain", (DL_FUNC) &zlm_log_drain, 0},
  {"zlm_log_pending_", (DL_FUNC) &zlm_log_pending_, 0},
  {"zlm_log_emit", (DL_FUNC) &zlm_log_emit, 2},
  {"zlm_engine_selftest", (DL_FUNC) &zlm_engine_selftest, 1},
  {"zlm_bench_mul_mat", (DL_FUNC) &zlm_bench_mul_mat, 3},
  {"zlm_store_new", (DL_FUNC) &zlm_store_new, 0},
  {"zlm_store_free_", (DL_FUNC) &zlm_store_free_, 1},
  {"zlm_store_add", (DL_FUNC) &zlm_store_add, 6},
  {"zlm_store_get", (DL_FUNC) &zlm_store_get, 2},
  {"zlm_store_set", (DL_FUNC) &zlm_store_set, 3},
  {"zlm_store_info", (DL_FUNC) &zlm_store_info, 1},
  {"zlm_store_save", (DL_FUNC) &zlm_store_save, 5},
  {"zlm_store_load", (DL_FUNC) &zlm_store_load, 2},
  {"zlm_i32_roundtrip", (DL_FUNC) &zlm_i32_roundtrip, 1},
  {"zlm_graph_new", (DL_FUNC) &zlm_graph_new, 6},
  {"zlm_graph_free_", (DL_FUNC) &zlm_graph_free_, 1},
  {"zlm_live_bytes", (DL_FUNC) &zlm_live_bytes, 0},
  {"zlm_graph_inputs", (DL_FUNC) &zlm_graph_inputs, 1},
  {"zlm_graph_param", (DL_FUNC) &zlm_graph_param, 2},
  {"zlm_tensor_info", (DL_FUNC) &zlm_tensor_info, 2},
  {"zlm_op", (DL_FUNC) &zlm_op, 7},
  {"zlm_graph_finish", (DL_FUNC) &zlm_graph_finish, 2},
  {"zlm_graph_abandon", (DL_FUNC) &zlm_graph_abandon, 1},
  {"zlm_graph_set_input", (DL_FUNC) &zlm_graph_set_input, 3},
  {"zlm_graph_compute", (DL_FUNC) &zlm_graph_compute, 2},
  {"zlm_graph_info", (DL_FUNC) &zlm_graph_info, 1},
  {"zlm_graph_output", (DL_FUNC) &zlm_graph_output, 2},
  {"zlm_grad", (DL_FUNC) &zlm_grad, 4},
  {"zlm_trainer_new", (DL_FUNC) &zlm_trainer_new, 7},
  {"zlm_trainer_step", (DL_FUNC) &zlm_trainer_step, 2},
  {"zlm_trainer_reset", (DL_FUNC) &zlm_trainer_reset, 1},
  {"zlm_trainer_evals", (DL_FUNC) &zlm_trainer_evals, 1},
  {"zlm_trainer_free_", (DL_FUNC) &zlm_trainer_free_, 1},
  {"zlm_zurand_init", (DL_FUNC) &zlm_zurand_init, 0},
  {"zlm_zurand_version_ok", (DL_FUNC) &zlm_zurand_version_ok, 1},
  {"zlm_rng_draw", (DL_FUNC) &zlm_rng_draw, 5},
  {"zlm_noise", (DL_FUNC) &zlm_noise, 6},
  {"zlm_rowloss_op", (DL_FUNC) &zlm_rowloss_op, 6},
  {"zlm_rowloss_get", (DL_FUNC) &zlm_rowloss_get, 1},
  {"zlm_const", (DL_FUNC) &zlm_const, 3},
  {"zlm_threads_", (DL_FUNC) &zlm_threads_, 1},
  {"zlm_poll_", (DL_FUNC) &zlm_poll_, 1},
  {"zlm_cores", (DL_FUNC) &zlm_cores, 0},
  {NULL, NULL, 0}
};

void attribute_visible ZLM_CAT(R_init_zulm_, ZLM_VARIANT)(DllInfo *dll) {
  R_registerRoutines(dll, NULL, zlm_call_methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
  zlm_variant_name = ZLM_STR(ZLM_VARIANT);
  ggml_set_abort_callback(zlm_abort_callback);
  ggml_log_set(zlm_ggml_log, NULL);
}
