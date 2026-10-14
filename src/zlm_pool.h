/* zlm_pool.h: the session's thread pool and interrupts (design §6.7). */
#ifndef ZLM_POOL_H
#define ZLM_POOL_H

#include "ggml-backend.h"
#include "zlm_r.h"

/* Before every computation: the session's thread count and pool, and the
   interrupt check. */
void zlm_prepare_backend(ggml_backend_t backend);
/* After it: whether the user interrupted it. */
int zlm_interrupted(void);
int zlm_threads_now(void);
/* Every CPU backend that computes is registered from creation to free. */
void zlm_backend_register(ggml_backend_t backend);
void zlm_backend_unregister(ggml_backend_t backend);

SEXP zlm_threads_(SEXP n);
SEXP zlm_poll_(SEXP poll);
SEXP zlm_cores(void);

#endif
