/* zlm_st.h: safetensors in C (design §7.4). */
#ifndef ZLM_ST_H
#define ZLM_ST_H

#include "zlm_r.h"

SEXP zlm_st_header(SEXP path);
SEXP zlm_st_read(SEXP path, SEXP data_start, SEXP begins, SEXP ends, SEXP dtypes);
SEXP zlm_st_fill(SEXP sptr, SEXP path, SEXP data_start, SEXP params, SEXP begins, SEXP ends,
                 SEXP dtypes);
SEXP zlm_st_write(SEXP sptr, SEXP path, SEXP header, SEXP type);

#endif
