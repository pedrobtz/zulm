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

enum { ZLM_KIND_PARAM = 0, ZLM_KIND_STATE = 1 };

typedef struct zlm_store {
  zlm_batch *batches;
  int n_batches, cap_batches;
  struct ggml_tensor **tensors; /* insertion order */
  int *ranks;                   /* the R rank (length of dim) per tensor */
  unsigned char *kinds;         /* ZLM_KIND_PARAM or ZLM_KIND_STATE */
  int n, cap;
  double bytes;
} zlm_store;

zlm_store *zlm_store_from(SEXP x);
/* The store of a pointer, or NULL; raises nothing (for the C table). */
zlm_store *zlm_store_peek(SEXP x);
struct ggml_tensor *zlm_store_find(const zlm_store *s, const char *name);
enum ggml_type zlm_type_from_string(const char *type);
const char *zlm_type_name(enum ggml_type type);
void zlm_read_values(const struct ggml_tensor *t, double *out);
void zlm_to_i32(SEXP x, int32_t *out, R_xlen_t n, const char *what);
SEXP zlm_i32_roundtrip(SEXP x);

/* The C table's state members (design §8, registered in Stage 12): no R
   calls, any thread; 0 on success, nonzero for an unknown state, a type
   other than f32 or a range outside the tensor. */
int zlm_api_state_shape(const zlm_store *s, const char *name, int64_t ne[4], int *type);
int zlm_api_state_read_f32(const zlm_store *s, const char *name, float *dst, int64_t n);
int zlm_api_state_write_f32_at(zlm_store *s, const char *name, int64_t offset, const float *src, int64_t n);

SEXP zlm_store_new(void);
SEXP zlm_store_free_(SEXP ptr);
SEXP zlm_store_add(SEXP ptr, SEXP name, SEXP dims, SEXP type, SEXP values, SEXP max_bytes, SEXP kind);
SEXP zlm_state_write_at(SEXP ptr, SEXP name, SEXP offset, SEXP values);
SEXP zlm_store_get(SEXP ptr, SEXP name);
SEXP zlm_store_set(SEXP ptr, SEXP name, SEXP values);
SEXP zlm_store_info(SEXP ptr);
SEXP zlm_store_tensor_info(SEXP ptr, SEXP name);
SEXP zlm_store_save(SEXP ptr, SEXP path, SEXP type, SEXP meta, SEXP max_bytes);
SEXP zlm_store_load(SEXP path, SEXP max_bytes, SEXP keep_types);

#endif
