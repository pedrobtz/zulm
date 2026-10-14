/*
 * zlm_train.c: gradients and training (design §6.2 to §6.4).
 *
 * The backward pre-check walks a graph's original nodes from the loss and
 * refuses, naming the operation and the R call, any node on a gradient
 * path whose operation has no backward pass in the vendored ggml: ggml
 * would abort in ggml_compute_backward().
 *
 * zl_grad() builds a backward graph of its own on a copy of the forward
 * graph (ggml_build_backward_expand), allocates it with ggml's graph
 * allocator and reads the parameters' gradients. zl_step() trains through
 * ggml-opt with static graphs on the original nodes, which zl_forward()
 * never allocates (zlm_dup.cpp).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "zlm_glue.h"
#include "zlm_graph.h"
#include "zlm_pool.h"
#include "zlm_r.h"
#include "zlm_store.h"
#include "zlm_train.h"

/* ---- the backward pre-check ------------------------------------------ */

static int unary_has_backward(enum ggml_unary_op u) {
  switch (u) {
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SGN:
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_STEP:
    case GGML_UNARY_OP_RELU:
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_EXP:
    case GGML_UNARY_OP_EXPM1:
    case GGML_UNARY_OP_SOFTPLUS:
      return 1;
    default:
      return 0;
  }
}

/* Whether ggml computes `t`'s backward pass, given which sources need a
   gradient. Read from ggml_compute_backward() at v0.26.0 with patches P1,
   P2 and P7 (design §9.3). */
static int has_backward(const struct ggml_tensor *t, const int *src_needs) {
  switch (t->op) {
    case GGML_OP_DUP:
    case GGML_OP_ADD:
    case GGML_OP_ADD1:
    case GGML_OP_ACC:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_LOG:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_RMS_NORM:
    case GGML_OP_SCALE:
    case GGML_OP_SET:
    case GGML_OP_CONT:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_DIAG_MASK_ZERO:
    case GGML_OP_ROPE:
    case GGML_OP_CROSS_ENTROPY_LOSS:
      return 1;
    case GGML_OP_CPY:
      /* a copy between types is zl_cast(), forward only */
      return t->type == t->src[0]->type;
    case GGML_OP_MUL_MAT:
      /* a weight broadcast over b's batches reduces its gradient over the
         third dimension only: ggml asserts b's fourth is 1 */
      if (src_needs[0] && (t->src[0]->ne[2] != t->src[1]->ne[2] || t->src[0]->ne[3] != t->src[1]->ne[3])) {
        return t->src[1]->ne[3] == 1;
      }
      return 1;
    case GGML_OP_GET_ROWS:
      return !src_needs[1];
    case GGML_OP_SOFT_MAX:
      /* the mask may not need a gradient */
      return !src_needs[1];
    case GGML_OP_UNARY:
      return unary_has_backward(ggml_get_unary_op(t));
    case GGML_OP_GLU:
      return ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU && t->src[1] != NULL;
    default:
      return 0;
  }
}

/* Sources whose gradient ggml ignores, as ggml_build_backward_expand()
   decides it: piecewise-constant unaries, copy targets, indices and
   positions. */
static int ignores_src(const struct ggml_tensor *t, int j) {
  switch (t->op) {
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_BACK:
      return j == 0;
    case GGML_OP_UNARY: {
      enum ggml_unary_op u = ggml_get_unary_op(t);
      return j == 0 && (u == GGML_UNARY_OP_SGN || u == GGML_UNARY_OP_STEP);
    }
    case GGML_OP_CPY:
    case GGML_OP_GET_ROWS:
    case GGML_OP_GET_ROWS_BACK:
    case GGML_OP_ROPE:
      return j == 1;
    default:
      return 0;
  }
}

static int index_of(struct ggml_tensor **v, int n, const struct ggml_tensor *t) {
  for (int i = 0; i < n; i++) {
    if (v[i] == t) return i;
  }
  return -1;
}

/* Refuses, before ggml sees it, a node whose backward ggml would compute
   and cannot: one that depends on a parameter (as ggml propagates that,
   ignoring the sources above) and receives a gradient from the loss.
   Flags the f32 parameters the loss depends on with ggml_set_param()
   when set_params is true, and returns their number. */
int zlm_backward_check(SEXP gptr, struct ggml_tensor *loss, int set_params) {
  zlm_graph *g = zlm_graph_from(gptr);
  size_t gsize = ZLM_GRAPH_SIZE(g);
  struct ggml_init_params p = { ggml_graph_overhead_custom(gsize, false), NULL, true };
  struct ggml_context *ctx = ggml_init(p);
  struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, gsize, false);
  ggml_build_forward_expand(gf, loss);
  int n = ggml_graph_n_nodes(gf);
  struct ggml_tensor **nodes = (struct ggml_tensor **) malloc((size_t) (n > 0 ? n : 1) * sizeof *nodes);
  unsigned char *needs = (unsigned char *) calloc((size_t) (n > 0 ? n : 1), 1);
  unsigned char *gets = (unsigned char *) calloc((size_t) (n > 0 ? n : 1), 1);
  struct ggml_tensor **params = (struct ggml_tensor **) malloc((size_t) (g->store->n + 1) * sizeof *params);
  int n_params = 0;
  for (int k = 0; k < n; k++) nodes[k] = ggml_graph_node(gf, k);
  /* parameters the loss uses, directly or through a view */
  for (int i = 0; i < g->store->n; i++) {
    struct ggml_tensor *w = g->store->tensors[i];
    int used = 0;
    for (int k = 0; k < n && !used; k++) {
      for (int j = 0; j < GGML_MAX_SRC && !used; j++) {
        used = nodes[k]->src[j] == w;
      }
    }
    if (!used) continue;
    if (w->type != GGML_TYPE_F32) {
      char name[GGML_MAX_NAME];
      snprintf(name, sizeof name, "%s", w->name);
      const char *type = zlm_type_name(w->type);
      free(nodes); free(needs); free(gets); free(params);
      ggml_free(ctx);
      zlm_raise("zulm_type_error", R_NilValue,
                "Parameter '%s' is %s; training needs f32 parameters.", name, type);
    }
    params[n_params++] = w;
  }
  /* forward: which nodes depend on a parameter */
  for (int k = 0; k < n; k++) {
    struct ggml_tensor *nd = nodes[k];
    for (int j = 0; j < GGML_MAX_SRC && !needs[k]; j++) {
      struct ggml_tensor *src = nd->src[j];
      if (src == NULL || ignores_src(nd, j)) continue;
      if (index_of(params, n_params, src) >= 0) { needs[k] = 1; break; }
      int si = index_of(nodes, k, src);
      if (si >= 0 && needs[si]) needs[k] = 1;
    }
  }
  /* backward: which nodes receive a gradient from the loss */
  if (n > 0) gets[n - 1] = 1;
  for (int k = n - 1; k >= 0; k--) {
    if (!gets[k] || !needs[k]) continue;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
      struct ggml_tensor *src = nodes[k]->src[j];
      if (src == NULL || ignores_src(nodes[k], j)) continue;
      int si = index_of(nodes, k, src);
      if (si >= 0) gets[si] = 1;
    }
  }
  for (int k = 0; k < n; k++) {
    if (!needs[k] || !gets[k]) continue;
    struct ggml_tensor *nd = nodes[k];
    int src_needs[GGML_MAX_SRC] = { 0 };
    for (int j = 0; j < GGML_MAX_SRC; j++) {
      struct ggml_tensor *src = nd->src[j];
      if (src == NULL || ignores_src(nd, j)) continue;
      int si = index_of(nodes, k, src);
      src_needs[j] = index_of(params, n_params, src) >= 0 || (si >= 0 && needs[si]);
    }
    if (!has_backward(nd, src_needs)) {
      const char *desc = ggml_op_desc(nd);
      SEXP call = PROTECT(zlm_graph_call_of(gptr, nd));
      free(nodes); free(needs); free(gets); free(params);
      ggml_free(ctx);
      SEXP extra = PROTECT(Rf_allocVector(VECSXP, 2));
      SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
      SET_VECTOR_ELT(extra, 0, Rf_mkString(desc));
      SET_VECTOR_ELT(extra, 1, call);
      SET_STRING_ELT(nm, 0, Rf_mkChar("op"));
      SET_STRING_ELT(nm, 1, Rf_mkChar("call"));
      Rf_setAttrib(extra, R_NamesSymbol, nm);
      /* the condition's call is the R call, which R prints with the message */
      zlm_raise("zulm_unsupported_backward", extra,
                "%s has no gradient, and it lies between a parameter and the loss.", desc);
    }
  }
  if (set_params) {
    for (int i = 0; i < n_params; i++) ggml_set_param(params[i]);
  }
  free(nodes); free(needs); free(gets); free(params);
  ggml_free(ctx);
  return n_params;
}

/* ---- zl_grad(): gradients on a copy --------------------------------- */

struct grad_job {
  zlm_graph *g;
  struct ggml_tensor *loss;
  struct ggml_context *ctx;
  size_t size_f, size_b; /* the forward copy's and the backward graph's sizes */
  struct ggml_cgraph *gb;
  struct ggml_tensor *loss_copy;
  int n;
  struct ggml_tensor **params, **param_copies, **grads;
};

static void grad_body(void *data) {
  struct grad_job *j = (struct grad_job *) data;
  struct ggml_cgraph *gf = ggml_new_graph_custom(j->ctx, j->size_f, false);
  ggml_build_forward_expand(gf, j->loss);
  int n = j->n + 1;
  struct ggml_tensor *in[ZLM_MAX_GRAD_PARAMS + 1];
  struct ggml_tensor *out[ZLM_MAX_GRAD_PARAMS + 1];
  in[0] = j->loss;
  for (int i = 0; i < j->n; i++) in[i + 1] = j->params[i];
  j->gb = zlm_graph_dup(j->ctx, gf, n, in, out, (int) j->size_b, 1);
  j->loss_copy = out[0];
  ggml_set_loss(j->loss_copy);
  ggml_build_backward_expand(j->ctx, j->gb, NULL);
  for (int i = 0; i < j->n; i++) {
    j->param_copies[i] = out[i + 1];
    struct ggml_tensor *gr = ggml_graph_get_grad(j->gb, out[i + 1]);
    if (gr != NULL) {
      /* a gradient can be a view (of a permuted parameter); it is read as
         contiguous memory */
      if (!ggml_is_contiguous(gr)) {
        gr = ggml_cont(j->ctx, gr);
        ggml_build_forward_expand(j->gb, gr);
      }
      ggml_set_output(gr);
    }
    j->grads[i] = gr;
  }
}

/* zlm_grad(gptr, loss handle, names): the gradients of the loss with
   respect to the named parameters at their current values, on the inputs
   last written. */
SEXP zlm_grad(SEXP gptr, SEXP loss_h, SEXP names, SEXP threads) {
  zlm_graph *g = zlm_graph_from(gptr);
  struct ggml_tensor *loss = (struct ggml_tensor *) R_ExternalPtrAddr(loss_h);
  int n = (int) Rf_xlength(names);
  if (n > ZLM_MAX_GRAD_PARAMS) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "At most %d parameters at a time.", ZLM_MAX_GRAD_PARAMS);
  }
  struct ggml_tensor *params[ZLM_MAX_GRAD_PARAMS];
  for (int i = 0; i < n; i++) {
    const char *nm = Rf_translateCharUTF8(STRING_ELT(names, i));
    params[i] = zlm_store_find(g->store, nm);
    if (params[i] == NULL) {
      zlm_raise("zulm_invalid_argument", R_NilValue, "The graph's store has no parameter \"%s\".", nm);
    }
  }
  zlm_backward_check(gptr, loss, 1);
  /* index inputs are checked as zl_forward() checks them */
  zlm_graph_check_indices(g);
  /* the copy holds the forward graph's tensors, and the backward pass adds
     at most about three per node */
  size_t size_f = ZLM_GRAPH_SIZE(g), size_b = 4 * size_f;
  struct ggml_init_params p = {
    size_b * ggml_tensor_overhead() + ggml_graph_overhead_custom(size_f, false) +
      ggml_graph_overhead_custom(size_b, true),
    NULL, true
  };
  struct grad_job j;
  memset(&j, 0, sizeof j);
  j.size_f = size_f;
  j.size_b = size_b;
  j.g = g;
  j.loss = loss;
  j.n = n;
  j.params = params;
  struct ggml_tensor *copies[ZLM_MAX_GRAD_PARAMS], *grads[ZLM_MAX_GRAD_PARAMS];
  j.param_copies = copies;
  j.grads = grads;
  j.ctx = ggml_init(p);
  if (j.ctx == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory building gradients.");
  char msg[2048] = "";
  int st = zlm_guarded(grad_body, &j, msg, sizeof msg);
  if (st != ZLM_STATUS_OK) {
    ggml_free(j.ctx);
    zlm_raise(st == ZLM_STATUS_ENGINE ? "zulm_engine_error" : "zulm_internal_error", R_NilValue,
              "ggml could not build the gradients: %s", msg);
  }
  ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (galloc == NULL || !ggml_gallocr_reserve(galloc, j.gb)) {
    if (galloc) ggml_gallocr_free(galloc);
    ggml_free(j.ctx);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory reserving the gradient graph.");
  }
  double bytes = (double) ggml_gallocr_get_buffer_size(galloc, 0);
  if (bytes > g->max_bytes) {
    ggml_gallocr_free(galloc);
    ggml_free(j.ctx);
    zlm_raise_memory(bytes, g->max_bytes, "The gradient graph");
  }
  if (!ggml_gallocr_alloc_graph(galloc, j.gb)) {
    ggml_gallocr_free(galloc);
    ggml_free(j.ctx);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory allocating the gradient graph.");
  }
  ggml_graph_reset(j.gb);
  (void) threads;
  ggml_backend_t backend = ggml_backend_cpu_init();
  zlm_backend_register(backend);
  zlm_prepare_backend(backend);
  enum ggml_status status = ggml_backend_graph_compute(backend, j.gb);
  zlm_backend_unregister(backend);
  ggml_backend_free(backend);
  if (zlm_interrupted()) {
    ggml_gallocr_free(galloc);
    ggml_free(j.ctx);
    zlm_raise("zulm_interrupt", R_NilValue, "The computation was interrupted.");
  }
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n + 1));
  if (status == GGML_STATUS_SUCCESS) {
    SEXP lv = Rf_allocVector(REALSXP, 1);
    SET_VECTOR_ELT(out, 0, lv);
    zlm_read_values(j.loss_copy, REAL(lv));
    for (int i = 0; i < n; i++) {
      SEXP v = Rf_allocVector(REALSXP, (R_xlen_t) ggml_nelements(params[i]));
      SET_VECTOR_ELT(out, i + 1, v);
      if (grads[i] != NULL) {
        zlm_read_values(grads[i], REAL(v));
      } else {
        memset(REAL(v), 0, (size_t) ggml_nelements(params[i]) * sizeof(double));
      }
    }
  }
  ggml_gallocr_free(galloc);
  ggml_free(j.ctx);
  if (status != GGML_STATUS_SUCCESS) {
    zlm_raise("zulm_internal_error", R_NilValue, "ggml failed to compute the gradients (status %d).", (int) status);
  }
  UNPROTECT(1);
  return out;
}

/* ---- the trainer: ggml-opt -------------------------------------------- */

static SEXP trainer_tag(void) {
  static SEXP tag = NULL;
  if (tag == NULL) tag = Rf_install("zlm_trainer");
  return tag;
}

static void trainer_free(zlm_trainer *t) {
  if (t == NULL) return;
  zlm_live_add(-t->bytes_live);
  if (t->opt) ggml_opt_free(t->opt);
  if (t->ctx_opt) ggml_free(t->ctx_opt);
  if (t->sched) ggml_backend_sched_free(t->sched);
  if (t->backend) {
    zlm_backend_unregister(t->backend);
    ggml_backend_free(t->backend);
  }
  free(t);
}

static void trainer_finalize(SEXP x) {
  trainer_free((zlm_trainer *) R_ExternalPtrAddr(x));
  R_ClearExternalPtr(x);
}

static zlm_trainer *trainer_from(SEXP x) {
  return (zlm_trainer *) zlm_ptr(x, trainer_tag(), "trainer");
}

static struct ggml_opt_optimizer_params trainer_pars(void *ud) {
  return ((zlm_trainer *) ud)->pars;
}

struct opt_job {
  zlm_trainer *t;
  struct ggml_opt_params params;
};

static void opt_init_body(void *data) {
  struct opt_job *j = (struct opt_job *) data;
  j->t->opt = ggml_opt_init(j->params);
}

/* zlm_trainer_new(gptr, loss handle, optimizer, pars, accumulate, clip,
   threads): pars = c(lr, beta1, beta2, eps, weight_decay). */
SEXP zlm_trainer_new(SEXP gptr, SEXP loss_h, SEXP optimizer, SEXP pars, SEXP accumulate,
                     SEXP clip, SEXP threads) {
  zlm_graph *g = zlm_graph_from(gptr);
  struct ggml_tensor *loss = (struct ggml_tensor *) R_ExternalPtrAddr(loss_h);
  if (g->n_in < 1) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "A graph to train needs at least one input.");
  }
  int n_params = zlm_backward_check(gptr, loss, 1);
  if (n_params == 0) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The loss depends on no parameter of the store.");
  }
  /* parameters, two AdamW moments and an accumulator, against the cap */
  double need = 4.0 * g->store->bytes;
  if (need > g->max_bytes) {
    zlm_raise_memory(need, g->max_bytes, "Training (parameters, gradients and optimiser state)");
  }
  zlm_trainer *t = (zlm_trainer *) calloc(1, sizeof *t);
  if (t == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a trainer.");
  SEXP tptr = PROTECT(R_MakeExternalPtr(t, trainer_tag(), gptr));
  R_RegisterCFinalizerEx(tptr, trainer_finalize, TRUE);
  t->g = g;
  t->loss = loss;
  const double *pp = REAL(pars);
  int adamw = strcmp(CHAR(STRING_ELT(optimizer, 0)), "adamw") == 0;
  t->pars = ggml_opt_get_default_optimizer_params(NULL);
  t->pars.adamw.alpha = (float) pp[0];
  t->pars.adamw.beta1 = (float) pp[1];
  t->pars.adamw.beta2 = (float) pp[2];
  t->pars.adamw.eps = (float) pp[3];
  t->pars.adamw.wd = (float) pp[4];
  t->pars.sgd.alpha = (float) pp[0];
  t->pars.sgd.wd = (float) pp[4];
  t->pars.clip = (float) Rf_asReal(clip);
  t->base_lr = (float) pp[0];
  t->backend = ggml_backend_cpu_init();
  if (t->backend == NULL) zlm_raise("zulm_internal_error", R_NilValue, "ggml's CPU backend could not start.");
  zlm_backend_register(t->backend);
  (void) threads;
  zlm_prepare_backend(t->backend);
  /* ggml-opt duplicates every training graph at the forward graph's size,
     GGML_DEFAULT_GRAPH_SIZE, so no graph the scheduler sees is larger; the
     scheduler's tables and its reserve for splits grow with this size */
  t->sched = ggml_backend_sched_new(&t->backend, NULL, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
  if (t->sched == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a scheduler.");
  struct opt_job j;
  j.t = t;
  j.params = ggml_opt_default_params(t->sched, GGML_OPT_LOSS_TYPE_SUM);
  /* a context of the trainer's own for ggml-opt's gradients and steps:
     its graphs are GGML_DEFAULT_GRAPH_SIZE nodes, three of them, and the
     backward pass and the optimiser add a few tensors per node */
  struct ggml_init_params po = {
    (size_t) 4 * GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
      3 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
    NULL, true
  };
  t->ctx_opt = ggml_init(po);
  if (t->ctx_opt == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a trainer.");
  j.params.ctx_compute = t->ctx_opt;
  j.params.inputs = g->in[0];
  j.params.outputs = loss;
  j.params.build_type = GGML_OPT_BUILD_TYPE_OPT;
  j.params.opt_period = Rf_asInteger(accumulate);
  j.params.get_opt_pars = trainer_pars;
  j.params.get_opt_pars_ud = t;
  j.params.optimizer = adamw ? GGML_OPT_OPTIMIZER_TYPE_ADAMW : GGML_OPT_OPTIMIZER_TYPE_SGD;
  char msg[2048] = "";
  int st = zlm_guarded(opt_init_body, &j, msg, sizeof msg);
  if (st != ZLM_STATUS_OK || t->opt == NULL) {
    zlm_raise(st == ZLM_STATUS_ENGINE ? "zulm_engine_error" : "zulm_internal_error", R_NilValue,
              "ggml-opt could not build the training graphs: %s", msg[0] ? msg : "unknown error");
  }
  /* what the trainer reserves: ggml-opt's context, the scheduler's buffer,
     which ggml reports, and the scheduler's own reserve for splits, which
     it does not (graph size x GGML_SCHED_MAX_SPLIT_INPUTS x 2 tensors, as
     ggml_backend_sched_new() computes it; 5 MB with the 4 Makevars sets),
     and ggml-opt's other tables, about 1 MB */
  t->bytes_live = 1.0 * 1024 * 1024 + (double) ggml_get_mem_size(t->ctx_opt) +
                  (double) ggml_backend_sched_get_buffer_size(t->sched, t->backend) +
                  (double) GGML_DEFAULT_GRAPH_SIZE * GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(struct ggml_tensor) +
                  (double) ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false);
  zlm_live_add(t->bytes_live);
  UNPROTECT(1);
  return tptr;
}

/* One evaluation with a backward pass: a gradient step every `accumulate`
   calls. Returns the loss of this call. */
SEXP zlm_trainer_step(SEXP tptr, SEXP lr) {
  zlm_trainer *t = trainer_from(tptr);
  zlm_graph_check_indices(t->g);
  float a = (float) Rf_asReal(lr);
  t->pars.adamw.alpha = a;
  t->pars.sgd.alpha = a;
  /* draws in the graph use this step's number, 1 for the first */
  t->g->step = (int64_t) t->evals + 1;
  zlm_prepare_backend(t->backend);
  ggml_opt_alloc(t->opt, true);
  ggml_opt_eval(t->opt, NULL);
  t->evals += 1;
  if (zlm_interrupted()) {
    zlm_raise("zulm_interrupt", R_NilValue,
              "The step was interrupted; the optimiser's state is now undefined: call zl_reset().");
  }
  struct ggml_tensor *l = ggml_opt_loss(t->opt);
  float v = 0;
  ggml_backend_tensor_get(l, &v, 0, sizeof v);
  return Rf_ScalarReal((double) v);
}

SEXP zlm_trainer_reset(SEXP tptr) {
  zlm_trainer *t = trainer_from(tptr);
  ggml_opt_reset(t->opt, true);
  t->evals = 0;
  return R_NilValue;
}

SEXP zlm_trainer_evals(SEXP tptr) {
  return Rf_ScalarReal(trainer_from(tptr)->evals);
}

SEXP zlm_trainer_free_(SEXP tptr) {
  trainer_free(trainer_from(tptr));
  R_ClearExternalPtr(tptr);
  return R_NilValue;
}
