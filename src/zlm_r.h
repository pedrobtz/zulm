/* zlm_r.h: R-side helpers of zulm's C glue: classed conditions and
   external-pointer checks. Only main-thread code may call these. */
#ifndef ZLM_R_H
#define ZLM_R_H

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdint.h>

/* Signals zulm:::zlm_abort(message, class, ...) and does not return.
   `extra` is NULL or a named list of further condition fields. Free every
   C resource before calling it. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn, format(printf, 3, 4)))
#endif
void zlm_raise(const char *cls, SEXP extra, const char *fmt, ...);

/* zulm_memory_error with the two byte counts (design §6.6). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void zlm_raise_memory(double requested, double allowed, const char *what);

/* Signals a zulm_range_warning (design §6.1). */
void zlm_warn_range(R_xlen_t n, const char *what);

/* The external pointer of a zulm object, or a zulm_stale_error when it is
   NULL (after saveRDS() or after it was freed). `tag` names the kind. */
void *zlm_ptr(SEXP x, SEXP tag, const char *kind);

#endif
