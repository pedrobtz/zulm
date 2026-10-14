/* zlm_graph.h: graphs and the forward pass (design §4, §6.2). */
#ifndef ZLM_GRAPH_H
#define ZLM_GRAPH_H

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "zlm_r.h"
#include "zlm_store.h"

enum { ZLM_GRAPH_BUILDING = 0, ZLM_GRAPH_READY = 1, ZLM_GRAPH_FAILED = 2 };

typedef struct {
  struct ggml_tensor *idx; /* an i32 input used as an index */
  int64_t bound;           /* every value must lie in [0, bound) */
} zlm_index_check;

typedef struct zlm_graph {
  zlm_store *store;             /* borrowed; the R object keeps it alive */
  struct ggml_context *ctx_in;  /* inputs, with a buffer of their own */
  ggml_backend_buffer_t buf_in;
  struct ggml_context *ctx;     /* nodes built by build(); never allocated,
                                   so ggml-opt can train on them (§4) */
  struct ggml_cgraph *gf;
  struct ggml_context *ctx_fwd; /* the copy zl_forward() evaluates */
  struct ggml_cgraph *gf_fwd;
  struct ggml_tensor **out_fwd;
  ggml_gallocr_t galloc;
  ggml_backend_t backend;
  struct ggml_tensor **in;
  int *in_rank;
  int n_in;
  struct ggml_tensor **out;
  int n_out;
  zlm_index_check *checks;
  int n_checks, cap_checks;
  int state;
  int threads;
  double max_bytes, bytes_inputs, bytes_compute;
} zlm_graph;

zlm_graph *zlm_graph_from(SEXP x);

#ifdef __cplusplus
extern "C" {
#endif
struct ggml_cgraph *zlm_graph_dup(struct ggml_context *ctx, struct ggml_cgraph *src, int n,
                                  struct ggml_tensor **map_in, struct ggml_tensor **map_out);
#ifdef __cplusplus
}
#endif

SEXP zlm_graph_new(SEXP sptr, SEXP names, SEXP dims, SEXP types, SEXP max_bytes, SEXP threads);
SEXP zlm_graph_free_(SEXP x);
SEXP zlm_graph_inputs(SEXP gptr);
SEXP zlm_graph_param(SEXP gptr, SEXP name);
SEXP zlm_tensor_info(SEXP h, SEXP gptr);
SEXP zlm_op(SEXP gptr, SEXP op, SEXP ha, SEXP hb, SEXP num, SEXP ints, SEXP call);
SEXP zlm_graph_finish(SEXP gptr, SEXP outputs);
SEXP zlm_graph_abandon(SEXP gptr);
SEXP zlm_graph_set_input(SEXP gptr, SEXP index, SEXP values);
SEXP zlm_graph_compute(SEXP gptr, SEXP which);
SEXP zlm_graph_info(SEXP gptr);

#endif
