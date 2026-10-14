/* zlm_train.h: gradients and training (design §6.2 to §6.4). */
#ifndef ZLM_TRAIN_H
#define ZLM_TRAIN_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "zlm_graph.h"
#include "zlm_r.h"

#define ZLM_GRAPH_NODES_TRAIN 16384
#define ZLM_MAX_GRAD_PARAMS 1024

typedef struct zlm_trainer {
  zlm_graph *g;                 /* borrowed; the R object keeps it alive */
  struct ggml_tensor *loss;
  ggml_backend_t backend;
  ggml_backend_sched_t sched;
  ggml_opt_context_t opt;
  struct ggml_opt_optimizer_params pars; /* read by ggml-opt each step */
  float base_lr;
  double evals;
  double bytes_live; /* counted in zlm_live_add() until freed */
} zlm_trainer;

int zlm_backward_check(SEXP gptr, struct ggml_tensor *loss, int set_params);
SEXP zlm_grad(SEXP gptr, SEXP loss_h, SEXP names, SEXP threads);
SEXP zlm_trainer_new(SEXP gptr, SEXP loss_h, SEXP optimizer, SEXP pars, SEXP accumulate,
                     SEXP clip, SEXP threads);
SEXP zlm_trainer_step(SEXP tptr, SEXP lr);
SEXP zlm_trainer_reset(SEXP tptr);
SEXP zlm_trainer_evals(SEXP tptr);
SEXP zlm_trainer_free_(SEXP tptr);

#endif
