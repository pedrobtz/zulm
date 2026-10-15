/* zlm_rng.h: zurand draws and per-row losses (design §7.1, §7.2). */
#ifndef ZLM_RNG_H
#define ZLM_RNG_H

#include <stdint.h>

#include "ggml.h"
#include "zlm_r.h"
#include "zurand.h"

enum { ZLM_DIST_UNIFORM = 0, ZLM_DIST_NORMAL = 1 };
enum { ZLM_ROWLOSS_CE = 0, ZLM_ROWLOSS_SQ = 1, ZLM_ROWLOSS_ABS = 2 };

/* One noise node: the stream's key, the distribution, the position of
   its first element, and the graph's step counter (read at run time). */
typedef struct {
  zurand_key key;
  int dist;
  uint64_t offset;
  const int64_t *step;
} zlm_noise_t;

/* One per-row loss node: what it computes, the factor the reported loss
   applies to the row sum, and the rows of the last run, in double. */
typedef struct {
  int mode;
  double scale;
  int64_t n_rows;
  double *rows;
} zlm_rowloss;

zurand_key zlm_key_from_r(SEXP key);
struct ggml_tensor *zlm_noise_node(struct ggml_context *ctx, zlm_noise_t *nz, const int64_t ne[4]);
struct ggml_tensor *zlm_rowloss_node(struct ggml_context *ctx, zlm_rowloss *rl,
                                     struct ggml_tensor *a, struct ggml_tensor *b);
double zlm_rowloss_value(const zlm_rowloss *rl);

SEXP zlm_zurand_init(void);
SEXP zlm_zurand_version_ok(SEXP version);
/* The C table's named streams (design §8). */
int zlm_rng_stream_register(const char *name, char *msg, size_t cap);
int zlm_rng_stream_draw(const char *name, uint32_t step, uint64_t offset, int dist, float *dst,
                        int64_t n, char *msg, size_t cap);
SEXP zlm_rng_streams_clear(void);
SEXP zlm_rng_draw(SEXP stream_key, SEXP step, SEXP offset, SEXP n, SEXP dist);

#endif
