/* zlm_store.h: the parameter store (design §5.1, §6.6). */
#ifndef ZLM_STORE_H
#define ZLM_STORE_H

#include "ggml.h"
#include "ggml-backend.h"
#include "zlm_r.h"
#include <stdint.h>

typedef struct {
  struct ggml_context *ctx;   /* tensor metadata of one allocation batch */
  ggml_backend_buffer_t buf;  /* the batch's data */
} zlm_batch;

typedef struct zlm_store {
  zlm_batch *batches;
  int n_batches, cap_batches;
  struct ggml_tensor **tensors; /* insertion order */
  int *ranks;                   /* the R rank (length of dim) per tensor */
  int n, cap;
  double bytes;
} zlm_store;

zlm_store *zlm_store_from(SEXP x);
struct ggml_tensor *zlm_store_find(const zlm_store *s, const char *name);
enum ggml_type zlm_type_from_string(const char *type);
const char *zlm_type_name(enum ggml_type type);
void zlm_read_values(const struct ggml_tensor *t, double *out);
void zlm_to_i32(SEXP x, int32_t *out, R_xlen_t n, const char *what);
SEXP zlm_i32_roundtrip(SEXP x);

SEXP zlm_store_new(void);
SEXP zlm_store_free_(SEXP ptr);
SEXP zlm_store_add(SEXP ptr, SEXP name, SEXP dims, SEXP type, SEXP values, SEXP max_bytes);
SEXP zlm_store_get(SEXP ptr, SEXP name);
SEXP zlm_store_set(SEXP ptr, SEXP name, SEXP values);
SEXP zlm_store_info(SEXP ptr);
SEXP zlm_store_save(SEXP ptr, SEXP path, SEXP type, SEXP meta, SEXP max_bytes);
SEXP zlm_store_load(SEXP path, SEXP max_bytes);

#endif
