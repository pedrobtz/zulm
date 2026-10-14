/* zlm_r.c: classed R conditions raised from the C glue (design §10). */
#include "zlm_r.h"

#include <stdarg.h>
#include <stdio.h>

static SEXP zlm_ns(void) {
  static SEXP ns = NULL;
  if (ns == NULL) {
    ns = R_FindNamespace(Rf_mkString("zulm"));
    R_PreserveObject(ns);
  }
  return ns;
}

static SEXP zlm_condition_call(const char *fun, const char *cls, SEXP extra, const char *msg) {
  R_xlen_t n_extra = Rf_isNull(extra) ? 0 : Rf_xlength(extra);
  SEXP call = PROTECT(Rf_allocVector(LANGSXP, 3 + n_extra));
  SETCAR(call, Rf_install(fun));
  SEXP p = CDR(call);
  SETCAR(p, Rf_mkString(msg));
  p = CDR(p);
  SETCAR(p, Rf_mkString(cls));
  if (n_extra > 0) {
    SEXP nm = Rf_getAttrib(extra, R_NamesSymbol);
    for (R_xlen_t i = 0; i < n_extra; i++) {
      p = CDR(p);
      SETCAR(p, VECTOR_ELT(extra, i));
      SET_TAG(p, Rf_installChar(STRING_ELT(nm, i)));
    }
  }
  UNPROTECT(1);
  return call;
}

void zlm_raise(const char *cls, SEXP extra, const char *fmt, ...) {
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  SEXP call = PROTECT(zlm_condition_call("zlm_abort", cls, extra, msg));
  Rf_eval(call, zlm_ns());
  UNPROTECT(1);
  Rf_error("%s", msg); /* not reached: zlm_abort() signals an error */
}

void zlm_raise_memory(double requested, double allowed, const char *what) {
  SEXP extra = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_VECTOR_ELT(extra, 0, Rf_ScalarReal(requested));
  SET_VECTOR_ELT(extra, 1, Rf_ScalarReal(allowed));
  SET_STRING_ELT(nm, 0, Rf_mkChar("requested"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("allowed"));
  Rf_setAttrib(extra, R_NamesSymbol, nm);
  zlm_raise("zulm_memory_error", extra,
            "%s needs %.0f bytes, more than getOption(\"zulm.max_bytes\") allows (%.0f).",
            what, requested, allowed);
}

void zlm_warn_range(R_xlen_t n, const char *what) {
  char msg[256];
  snprintf(msg, sizeof msg,
           "%.0f value(s) of %s lie beyond the range of the storage type and became infinite.",
           (double) n, what);
  SEXP call = PROTECT(zlm_condition_call("zlm_warn", "zulm_range_warning", R_NilValue, msg));
  Rf_eval(call, zlm_ns());
  UNPROTECT(1);
}

void *zlm_ptr(SEXP x, SEXP tag, const char *kind) {
  if (TYPEOF(x) != EXTPTRSXP || R_ExternalPtrTag(x) != tag) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "Expected a %s.", kind);
  }
  void *p = R_ExternalPtrAddr(x);
  if (p == NULL) {
    zlm_raise("zulm_stale_error", R_NilValue,
              "This %s is no longer valid: it was freed, or restored from a saved R session. "
              "Use zl_save() and zl_load() to keep parameters.", kind);
  }
  return p;
}
