/*
 * zlm_pool.c: the session's thread pool and interrupts (design §6.7).
 *
 * One persistent ggml thread pool serves every computation; it is
 * recreated when zl_threads() changes its size, and abandoned without
 * being joined in a forked child, which creates its own on first use.
 * Idle workers sleep (poll 0) rather than spin. Every computation installs
 * the CPU backend's abort callback, which ggml calls between nodes on
 * worker 0, the calling thread: it checks for a user interrupt inside
 * R_ToplevelExec(), so no longjmp crosses ggml's frames, and stops the
 * computation; the caller then signals zulm_interrupt.
 */
#define R_NO_REMAP
#include <R.h>
#include <R_ext/Utils.h>
#include <Rinternals.h>

#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#define zlm_getpid _getpid
#else
#include <unistd.h>
#define zlm_getpid getpid
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "zlm_pool.h"

#include <stdlib.h>

static struct ggml_threadpool *zlm_tp = NULL;
static int zlm_tp_n = 0;
static long zlm_tp_pid = 0;
static int zlm_tp_poll = 0;
static int zlm_n_threads = 1;
static volatile int zlm_was_interrupted = 0;

/* Every live CPU backend: ggml_backend_cpu_set_threadpool() pauses a
   backend's previous pool, so a pool may be freed only after every backend
   that holds it has been detached from it. */
static ggml_backend_t *zlm_backends = NULL;
static int zlm_n_backends = 0, zlm_cap_backends = 0;

void zlm_backend_register(ggml_backend_t backend) {
  if (zlm_n_backends == zlm_cap_backends) {
    int cap = zlm_cap_backends ? 2 * zlm_cap_backends : 16;
    ggml_backend_t *b = (ggml_backend_t *) realloc(zlm_backends, (size_t) cap * sizeof *b);
    if (b == NULL) return;
    zlm_backends = b;
    zlm_cap_backends = cap;
  }
  zlm_backends[zlm_n_backends++] = backend;
}

void zlm_backend_unregister(ggml_backend_t backend) {
  for (int i = 0; i < zlm_n_backends; i++) {
    if (zlm_backends[i] == backend) {
      zlm_backends[i] = zlm_backends[--zlm_n_backends];
      return;
    }
  }
}

static void pool_free(void) {
  if (zlm_tp == NULL) return;
  for (int i = 0; i < zlm_n_backends; i++) {
    ggml_backend_cpu_set_threadpool(zlm_backends[i], NULL);
  }
  ggml_threadpool_free(zlm_tp);
  zlm_tp = NULL;
  zlm_tp_n = 0;
}

static void check_interrupt(void *data) {
  (void) data;
  R_CheckUserInterrupt();
}

/* ggml's abort callback: true stops the computation. */
static bool interrupt_cb(void *data) {
  (void) data;
  if (zlm_was_interrupted) return true;
  if (!R_ToplevelExec(check_interrupt, NULL)) {
    zlm_was_interrupted = 1;
    return true;
  }
  return false;
}

void zlm_prepare_backend(ggml_backend_t backend) {
  int n = zlm_n_threads > 0 ? zlm_n_threads : 1;
  long pid = (long) zlm_getpid();
  if (zlm_tp != NULL && zlm_tp_pid != pid) {
    /* a forked child: the parent's workers do not exist here */
    zlm_tp = NULL;
    zlm_tp_n = 0;
  }
  if (n > 1) {
    if (zlm_tp == NULL || zlm_tp_n != n) {
      pool_free();
      struct ggml_threadpool_params params = ggml_threadpool_params_default(n);
      params.poll = (uint32_t) zlm_tp_poll;
      zlm_tp = ggml_threadpool_new(&params);
      zlm_tp_n = zlm_tp ? n : 0;
      zlm_tp_pid = pid;
    }
    ggml_backend_cpu_set_threadpool(backend, zlm_tp);
  } else {
    ggml_backend_cpu_set_threadpool(backend, NULL);
  }
  ggml_backend_cpu_set_n_threads(backend, n);
  zlm_was_interrupted = 0;
  ggml_backend_cpu_set_abort_callback(backend, interrupt_cb, NULL);
}

int zlm_interrupted(void) {
  return zlm_was_interrupted;
}

int zlm_threads_now(void) {
  return zlm_n_threads;
}

SEXP zlm_threads_(SEXP n) {
  int old = zlm_n_threads;
  if (!Rf_isNull(n)) zlm_n_threads = Rf_asInteger(n);
  return Rf_ScalarInteger(old);
}

/* Internal: ggml's polling level for new pools (design §6.7 measures it). */
SEXP zlm_poll_(SEXP poll) {
  int old = zlm_tp_poll;
  if (!Rf_isNull(poll)) {
    zlm_tp_poll = Rf_asInteger(poll);
    pool_free();
  }
  return Rf_ScalarInteger(old);
}

SEXP zlm_cores(void) {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return Rf_ScalarInteger((int) si.dwNumberOfProcessors);
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return Rf_ScalarInteger(n > 0 ? (int) n : 1);
#endif
}
