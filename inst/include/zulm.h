/* zulm C table, for packages that declare `LinkingTo: zulm` and
 * `Imports: zulm` (design §8).
 *
 * zulm does not export ggml's headers. A consumer reaches graphs and
 * parameter stores built in R through opaque handles and a table of
 * function pointers:
 *
 *   #include <zulm.h>
 *
 *   SEXP my_sampler(SEXP graph, SEXP steps) {
 *       const zulm_api *zl = zulm_api_get();           // main thread
 *       zulm_graph *g = zl->graph_from_sexp(graph);    // main thread
 *       if (g == NULL) Rf_error("%s", zl->message());
 *       if (zl->rng_stream("eps") != ZULM_OK) ...       // main thread
 *       for (int s = 1; s <= n; s++) {
 *           zl->rng_normal_f32("eps", s, 0, buf, len);  // any thread
 *           zl->input_write_f32(g, "x", buf, len);      // any thread
 *           if (zl->compute(g) != ZULM_OK) ...          // main thread
 *           zl->output_read_f32(g, "v", out, len);      // any thread
 *       }
 *   }
 *
 * Threads. graph_from_sexp(), store_from_sexp(), rng_stream(), compute()
 * and batch_fill() call R and run on the main thread only. The other
 * members call no R API and may run on a consumer's own threads, one
 * thread per graph or store at a time.
 *
 * Errors. Every member that returns int returns ZULM_OK (0) on success
 * and one of the ZULM_E* codes otherwise; message() then describes the
 * failure on the calling thread. No member raises an R error. A handle
 * stays valid while the R object it came from is alive: keep that object
 * protected.
 *
 * Versioning. Members are only ever appended. struct_size is the size of
 * the installed zulm's table, so a consumer can tell whether a member
 * exists. zulm_api_get() refuses, with an R error, a table shorter than
 * the header the consumer was compiled against. A change to the layout
 * would register a new name, zulm_api_v2.
 */
#ifndef ZULM_H
#define ZULM_H

#include <stddef.h>
#include <stdint.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define ZULM_OK 0
#define ZULM_EUNKNOWN 1     /* no input, output, parameter, state or stream of that name */
#define ZULM_EMISMATCH 2    /* the type, or the number of values, does not match */
#define ZULM_ERANGE 3       /* an offset or index outside the tensor */
#define ZULM_ESTALE 4       /* a freed, failed or foreign handle */
#define ZULM_EFAILED 5      /* the computation, or reading the stream, failed */
#define ZULM_EINTERRUPT 6   /* the user interrupted the computation */
#define ZULM_EARGUMENT 7    /* a NULL pointer or a negative count */

/* batch_fill() returns ZULM_END, not an error, when the stream has ended. */
#define ZULM_END 1

/* Element types, as input_shape() and state_shape() report them. */
#define ZULM_TYPE_F32 0
#define ZULM_TYPE_F16 1
#define ZULM_TYPE_BF16 2
#define ZULM_TYPE_I32 3

typedef struct zulm_graph zulm_graph; /* opaque */
typedef struct zulm_store zulm_store; /* opaque */

typedef struct zulm_api {
    size_t struct_size;
    /* the vendored ggml and zulm's patches, as "0.26.0+p1,p2" */
    const char *(*ggml_version)(void);
    /* the calling thread's last failure */
    const char *(*message)(void);

    /* handles from R objects: a zl_graph and a zl_params; NULL, with
       message(), for anything else. Main thread only. */
    zulm_graph *(*graph_from_sexp)(SEXP g);
    zulm_store *(*store_from_sexp)(SEXP p);

    /* inputs and outputs by name, in R's element order; n must equal the
       tensor's element count. Any thread, no R calls. */
    int (*input_shape)(const zulm_graph *g, const char *name,
                       int64_t ne[4], int *type);
    int (*input_write_f32)(zulm_graph *g, const char *name,
                           const float *src, int64_t n);
    int (*input_write_i32)(zulm_graph *g, const char *name,
                           const int32_t *src, int64_t n);
    int (*output_read_f32)(const zulm_graph *g, const char *name,
                           float *dst, int64_t n);
    /* f32 parameters of a store, whole. Any thread, no R calls. */
    int (*param_read_f32)(const zulm_store *p, const char *name,
                          float *dst, int64_t n);
    int (*param_write_f32)(zulm_store *p, const char *name,
                           const float *src, int64_t n);

    /* runs the graph on its current inputs, on zulm's thread pool, and
       checks for a user interrupt. Main thread only. */
    int (*compute)(zulm_graph *g);

    /* draws under the session key (design §7.2): position offset + i of
       the named stream at a step, as zl_rng_uniform() and zl_rng_normal()
       give them. rng_stream() folds the session key with the name, on the
       main thread; a stream must be registered before it is drawn from,
       and again after zl_seed() changes the key. The draws run on any
       thread, with no R calls. */
    int (*rng_stream)(const char *name);
    int (*rng_uniform_f32)(const char *name, uint32_t step,
                           uint64_t offset, float *dst, int64_t n);
    int (*rng_normal_f32)(const char *name, uint32_t step,
                          uint64_t offset, float *dst, int64_t n);

    /* offset writes and state tensors (design §6.9). Any thread, no R
       calls. */
    int (*input_write_f32_at)(zulm_graph *g, const char *name,
                              int64_t offset, const float *src, int64_t n);
    int (*state_shape)(const zulm_store *p, const char *name,
                       int64_t ne[4], int *type);
    int (*state_read_f32)(const zulm_store *p, const char *name,
                          float *dst, int64_t n);
    int (*state_write_f32_at)(zulm_store *p, const char *name,
                              int64_t offset, const float *src, int64_t n);

    /* the next batch of a zl_arrow_batches() iterator into the graph's
       inputs (design §7.3); ZULM_END when the stream has ended. Main
       thread only. */
    int (*batch_fill)(zulm_graph *g, SEXP iterator);

    /* appended members go here */
} zulm_api;

/* The table, refused with an R error unless it holds at least `size`
   bytes. Call zulm_api_get() instead. */
static inline const zulm_api *zulm_api_require(size_t size) {
    typedef const zulm_api *(*zulm_getter)(void);
    zulm_getter get = (zulm_getter) R_GetCCallable("zulm", "zulm_api_v1");
    const zulm_api *api = get();
    if (api == NULL || api->struct_size < size)
        Rf_error("the installed zulm provides a C table of %d bytes, but this "
                 "package was built against zulm.h with %d; update zulm",
                 api == NULL ? 0 : (int) api->struct_size, (int) size);
    return api;
}

/* Fetch the table. Call on the main thread; the pointer stays valid for
   the session and may then be used from any thread. */
static inline const zulm_api *zulm_api_get(void) {
    return zulm_api_require(sizeof(zulm_api));
}

#ifdef __cplusplus
}
#endif

#endif /* ZULM_H */
