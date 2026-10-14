/* zlm_glue.h: internal declarations shared by zulm's C and C++ glue. */
#ifndef ZLM_GLUE_H
#define ZLM_GLUE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* zlm_log.c */
void zlm_log_append(const char *s, size_t n);
void zlm_log_record(int level, const char *s, size_t n);
int zlm_log_pending(void);
size_t zlm_log_take(char *out, size_t cap, int *dropped);
void zlm_fatal_message(const char *msg);

/* zlm_shim.cpp: design §13.2. zlm_guarded() runs fn(data) with the abort
   callback armed: a ggml assertion inside fn unwinds back here and is
   reported as ZLM_STATUS_ENGINE with ggml's message in msg. fn must not
   touch R (no PROTECT, no allocation, no error). */
enum {
  ZLM_STATUS_OK = 0,
  ZLM_STATUS_ENGINE = 1,
  ZLM_STATUS_MEMORY = 2,
  ZLM_STATUS_INTERNAL = 3
};
int zlm_guarded(void (*fn)(void *), void *data, char *msg, size_t msgsize);
void zlm_abort_callback(const char *message);

/* zlm_glue.c: the C memory held by live graphs and trainers, which R's
   garbage collector cannot see (design §6.6); main thread only. */
void zlm_live_add(double bytes);
double zlm_live_get(void);

/* The variant this shared object was built as (set by its R_init_). */
extern const char *zlm_variant_name;

#ifdef __cplusplus
}
#endif

#endif
