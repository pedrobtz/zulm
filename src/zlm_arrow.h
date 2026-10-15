/* zlm_arrow.h: training batches from Arrow C streams (design §7.3). */
#ifndef ZLM_ARROW_H
#define ZLM_ARROW_H

#include <Rinternals.h>

struct zlm_graph;

int zlm_api_batch_fill(struct zlm_graph *g, SEXP iterator);

SEXP zlm_arrow_open(SEXP xstream, SEXP names, SEXP is_int, SEXP cols, SEXP rows, SEXP na_mask, SEXP pad);
SEXP zlm_arrow_next(SEXP x);
SEXP zlm_batch_fill(SEXP gptr, SEXP x);
SEXP zlm_arrow_state(SEXP x);
SEXP zlm_arrow_close(SEXP x);
SEXP zlm_arrow_test_stream(SEXP batches, SEXP batch_rows, SEXP counts);

#endif
