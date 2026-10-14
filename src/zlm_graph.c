/*
 * zlm_graph.c: graphs, operations and the forward pass (design §4, §5.3,
 * §6.1, §6.2, §13.2).
 *
 * A graph owns three kinds of tensor: the parameters of its store (the
 * store's own, borrowed), its inputs (a static context with a buffer of
 * its own, so zl_forward() rewrites them in place) and the nodes build()
 * creates (a compute context allocated by ggml's graph allocator when the
 * graph is finished). Outputs are flagged so the allocator never reuses
 * their memory.
 *
 * Every operation is checked here against what the ggml builder and the
 * CPU kernel assert, before ggml sees it (design §13.2, layer 1), and the
 * builder runs under zlm_guarded() (layer 2). Two rules keep the kernels'
 * own assertions out of reach: only f32 tensors enter operations (f16 and
 * bf16 parameters only as the weight of zl_crossprod(), the table of
 * zl_get_rows() and the input of zl_cast()), and an operand that is not
 * contiguous is made contiguous first. Indices are checked against their
 * bounds every time zl_forward() writes them.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "zlm_glue.h"
#include "zlm_graph.h"
#include "zlm_r.h"
#include "zlm_store.h"

/* build contexts: the first holds this many tensors, each next one twice
   the last, up to ZLM_CTX_MAX; an operation needs room for ZLM_CTX_SLACK */
#define ZLM_CTX_FIRST 256
#define ZLM_CTX_MAX 65536
#define ZLM_CTX_SLACK 32

static SEXP graph_tag(void) {
  static SEXP tag = NULL;
  if (tag == NULL) tag = Rf_install("zlm_graph");
  return tag;
}

static SEXP tensor_tag(void) {
  static SEXP tag = NULL;
  if (tag == NULL) tag = Rf_install("zlm_tensor");
  return tag;
}

static void graph_free(zlm_graph *g) {
  if (g == NULL) return;
  zlm_live_add(-g->bytes_live);
  if (g->galloc) ggml_gallocr_free(g->galloc);
  if (g->backend) ggml_backend_free(g->backend);
  if (g->buf_in) ggml_backend_buffer_free(g->buf_in);
  if (g->ctx_in) ggml_free(g->ctx_in);
  if (g->ctx) ggml_free(g->ctx);
  for (int i = 0; i < g->n_ctx_full; i++) ggml_free(g->ctx_full[i]);
  free(g->ctx_full);
  if (g->ctx_graph) ggml_free(g->ctx_graph);
  if (g->ctx_fwd) ggml_free(g->ctx_fwd);
  free(g->out_fwd);
  free(g->in);
  free(g->in_rank);
  free(g->out);
  free(g->checks);
  free(g->node_t);
  free(g->node_c);
  free(g);
}

static void graph_finalize(SEXP x) {
  graph_free((zlm_graph *) R_ExternalPtrAddr(x));
  R_ClearExternalPtr(x);
}

zlm_graph *zlm_graph_from(SEXP x) {
  return (zlm_graph *) zlm_ptr(x, graph_tag(), "graph");
}

SEXP zlm_graph_free_(SEXP x) {
  graph_free(zlm_graph_from(x));
  R_ClearExternalPtr(x);
  return R_NilValue;
}

/* A tensor handle: an external pointer to the ggml tensor whose protected
   value is the graph's external pointer, which keeps the graph alive. */
static SEXP tensor_wrap(SEXP gptr, struct ggml_tensor *t) {
  return R_MakeExternalPtr(t, tensor_tag(), gptr);
}

/* The tensor of a handle, checked to belong to graph `gptr`. */
static struct ggml_tensor *tensor_from(SEXP h, SEXP gptr) {
  if (TYPEOF(h) != EXTPTRSXP || R_ExternalPtrTag(h) != tensor_tag()) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "Expected a tensor handle.");
  }
  if (R_ExternalPtrProtected(h) != gptr) {
    zlm_raise("zulm_invalid_argument", R_NilValue,
              "A tensor from another graph cannot be used here.");
  }
  void *t = R_ExternalPtrAddr(h);
  if (t == NULL || R_ExternalPtrAddr(gptr) == NULL) {
    zlm_raise("zulm_stale_error", R_NilValue, "This tensor's graph is no longer valid.");
  }
  return (struct ggml_tensor *) t;
}

SEXP zlm_graph_store_ptr(SEXP gptr) {
  return VECTOR_ELT(R_ExternalPtrProtected(gptr), 0);
}

int zlm_graph_is_param(const zlm_graph *g, const struct ggml_tensor *t) {
  for (int i = 0; i < g->store->n; i++) {
    if (g->store->tensors[i] == t) return 1;
  }
  return 0;
}

/* Records `call` as the origin of every node created since the last call
   (an operation may create several: a contiguous copy, a negation). */
static void record_call(zlm_graph *g, SEXP gptr, SEXP call) {
  SEXP prot = R_ExternalPtrProtected(gptr);
  SEXP calls = VECTOR_ELT(prot, 1);
  if (g->n_calls == Rf_length(calls)) {
    calls = Rf_lengthgets(calls, 2 * Rf_length(calls));
    SET_VECTOR_ELT(prot, 1, calls);
  }
  SET_VECTOR_ELT(calls, g->n_calls, call);
  int c = g->n_calls++;
  /* the operation's tensors are in the current context; if it took over
     since the last call, they start at its beginning */
  struct ggml_tensor *t = g->last_seen != NULL && g->last_seen_ctx == g->ctx
                            ? ggml_get_next_tensor(g->ctx, g->last_seen)
                            : ggml_get_first_tensor(g->ctx);
  g->last_seen_ctx = g->ctx;
  for (; t != NULL; t = ggml_get_next_tensor(g->ctx, t)) {
    if (g->n_node == g->cap_node) {
      int cap = g->cap_node ? 2 * g->cap_node : 64;
      struct ggml_tensor **nt = (struct ggml_tensor **) realloc(g->node_t, (size_t) cap * sizeof *nt);
      if (nt == NULL) return;
      g->node_t = nt;
      int *nc = (int *) realloc(g->node_c, (size_t) cap * sizeof *nc);
      if (nc == NULL) return;
      g->node_c = nc;
      g->cap_node = cap;
    }
    g->node_t[g->n_node] = t;
    g->node_c[g->n_node] = c;
    g->n_node++;
    g->last_seen = t;
  }
}

/* The R call that created node `t`, or NULL. */
SEXP zlm_graph_call_of(SEXP gptr, struct ggml_tensor *t) {
  zlm_graph *g = zlm_graph_from(gptr);
  SEXP calls = VECTOR_ELT(R_ExternalPtrProtected(gptr), 1);
  for (int k = 0; k < g->n_node; k++) {
    if (g->node_t[k] == t) return VECTOR_ELT(calls, g->node_c[k]);
  }
  return R_NilValue;
}

/* ---- creation ------------------------------------------------------- */

/* zl_graph(): a graph on store `sptr` with inputs of the given names,
   dimensions (list of numeric vectors) and types ("f32" or "i32"). */
SEXP zlm_graph_new(SEXP sptr, SEXP names, SEXP dims, SEXP types, SEXP max_bytes, SEXP threads) {
  zlm_store *s = zlm_store_from(sptr);
  int n = (int) Rf_xlength(names);
  double cap = Rf_asReal(max_bytes);
  double bytes = 0;
  int64_t (*ne)[GGML_MAX_DIMS] = (int64_t (*)[GGML_MAX_DIMS]) R_alloc((size_t) (n > 0 ? n : 1), sizeof *ne);
  int *rank = (int *) R_alloc((size_t) (n > 0 ? n : 1), sizeof(int));
  enum ggml_type *tp = (enum ggml_type *) R_alloc((size_t) (n > 0 ? n : 1), sizeof(enum ggml_type));
  for (int i = 0; i < n; i++) {
    SEXP d = VECTOR_ELT(dims, i);
    rank[i] = (int) Rf_xlength(d);
    double total = 1;
    for (int k = 0; k < GGML_MAX_DIMS; k++) {
      ne[i][k] = k < rank[i] ? (int64_t) REAL(d)[k] : 1;
      total *= (double) ne[i][k];
    }
    tp[i] = strcmp(CHAR(STRING_ELT(types, i)), "i32") == 0 ? GGML_TYPE_I32 : GGML_TYPE_F32;
    bytes += total * 4 + 64;
  }
  if (bytes > cap) {
    zlm_raise_memory(bytes, cap, "The graph's inputs");
  }
  zlm_graph *g = (zlm_graph *) calloc(1, sizeof *g);
  if (g == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a graph.");
  }
  /* protected: list(store, calls), the store kept alive and the R call
     of every operation, for errors */
  SEXP prot = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(prot, 0, sptr);
  SET_VECTOR_ELT(prot, 1, Rf_allocVector(VECSXP, 16));
  SEXP gptr = PROTECT(R_MakeExternalPtr(g, graph_tag(), prot));
  R_RegisterCFinalizerEx(gptr, graph_finalize, TRUE);
  g->store = s;
  g->threads = Rf_asInteger(threads);
  g->max_bytes = cap;
  g->n_in = n;
  g->in = (struct ggml_tensor **) calloc((size_t) (n > 0 ? n : 1), sizeof *g->in);
  g->in_rank = (int *) calloc((size_t) (n > 0 ? n : 1), sizeof(int));
  if (g->in == NULL || g->in_rank == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a graph.");
  }
  struct ggml_init_params pin = { (size_t) (n + 1) * ggml_tensor_overhead(), NULL, true };
  g->ctx_in = ggml_init(pin);
  struct ggml_init_params pc = { (size_t) ZLM_CTX_FIRST * ggml_tensor_overhead(), NULL, true };
  g->ctx = ggml_init(pc);
  if (g->ctx_in == NULL || g->ctx == NULL) {
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory creating a graph.");
  }
  for (int i = 0; i < n; i++) {
    struct ggml_tensor *t = ggml_new_tensor(g->ctx_in, tp[i], GGML_MAX_DIMS, ne[i]);
    ggml_set_name(t, Rf_translateCharUTF8(STRING_ELT(names, i)));
    ggml_set_input(t);
    g->in[i] = t;
    g->in_rank[i] = rank[i];
  }
  if (n > 0) {
    g->buf_in = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx_in, ggml_backend_cpu_buffer_type());
    if (g->buf_in == NULL) {
      zlm_raise("zulm_memory_error", R_NilValue, "Out of memory allocating the graph's inputs.");
    }
    ggml_backend_buffer_clear(g->buf_in, 0);
  }
  g->bytes_inputs = n > 0 ? (double) ggml_backend_buffer_get_size(g->buf_in) : 0;
  UNPROTECT(2);
  return gptr;
}

/* Handles of the inputs, in declaration order. */
SEXP zlm_graph_inputs(SEXP gptr) {
  zlm_graph *g = zlm_graph_from(gptr);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, g->n_in));
  for (int i = 0; i < g->n_in; i++) {
    SET_VECTOR_ELT(out, i, tensor_wrap(gptr, g->in[i]));
  }
  UNPROTECT(1);
  return out;
}

/* A handle on a parameter of the graph's store. */
SEXP zlm_graph_param(SEXP gptr, SEXP name) {
  zlm_graph *g = zlm_graph_from(gptr);
  if (g->state != ZLM_GRAPH_BUILDING) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "Parameters are used inside build() only.");
  }
  const char *nm = Rf_translateCharUTF8(STRING_ELT(name, 0));
  struct ggml_tensor *t = zlm_store_find(g->store, nm);
  if (t == NULL) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "The graph's store has no parameter \"%s\".", nm);
  }
  return tensor_wrap(gptr, t);
}

/* ne and type of a handle's tensor: list(ne = double[4], type = chr). */
SEXP zlm_tensor_info(SEXP h, SEXP gptr) {
  struct ggml_tensor *t = tensor_from(h, gptr);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SEXP ne = Rf_allocVector(REALSXP, GGML_MAX_DIMS);
  SET_VECTOR_ELT(out, 0, ne);
  for (int k = 0; k < GGML_MAX_DIMS; k++) REAL(ne)[k] = (double) t->ne[k];
  SET_VECTOR_ELT(out, 1, Rf_mkString(zlm_type_name(t->type)));
  SET_VECTOR_ELT(out, 2, Rf_mkString(ggml_op_desc(t)));
  UNPROTECT(1);
  return out;
}

/* ---- operations ----------------------------------------------------- */

typedef struct {
  zlm_graph *g;
  const char *op;
  struct ggml_tensor *a, *b, *c;
  double p[8];
  int ip[8];
  struct ggml_tensor *out;
} zlm_opjob;

static int is_half(const struct ggml_tensor *t) {
  return t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
}

/* Contiguous operands for every kernel that is not a pure view. */
static struct ggml_tensor *cont(struct ggml_context *ctx, struct ggml_tensor *t) {
  return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

static int can_repeat(const struct ggml_tensor *b, const struct ggml_tensor *a) {
  for (int k = 0; k < GGML_MAX_DIMS; k++) {
    if (a->ne[k] % b->ne[k] != 0) return 0;
  }
  return 1;
}

/* Makes sure the build context has room for an operation's tensors (no
   operation makes more than a few); otherwise a new one, twice as large,
   takes over. Contexts are sized by use rather than for the largest graph,
   since their memory is reserved whether used or not. Runs inside
   zlm_guarded(), so it may not touch R. */
static void ensure_room(zlm_graph *g) {
  size_t over = ggml_tensor_overhead();
  if (ggml_get_mem_size(g->ctx) - ggml_used_mem(g->ctx) >= (size_t) ZLM_CTX_SLACK * over) return;
  if (g->n_ctx_full == g->cap_ctx_full) {
    int cap = g->cap_ctx_full ? 2 * g->cap_ctx_full : 8;
    struct ggml_context **c = (struct ggml_context **) realloc(g->ctx_full, (size_t) cap * sizeof *c);
    if (c == NULL) GGML_ABORT("out of memory growing a graph");
    g->ctx_full = c;
    g->cap_ctx_full = cap;
  }
  size_t tensors = 2 * (ggml_get_mem_size(g->ctx) / over);
  if (tensors > ZLM_CTX_MAX) tensors = ZLM_CTX_MAX;
  struct ggml_init_params p = { tensors * over, NULL, true };
  struct ggml_context *next = ggml_init(p);
  if (next == NULL) GGML_ABORT("out of memory growing a graph");
  g->ctx_full[g->n_ctx_full++] = g->ctx;
  g->ctx = next;
}

/* Tensors made in the build contexts so far, at most. */
static size_t build_tensors(const zlm_graph *g) {
  size_t over = ggml_tensor_overhead(), n = ggml_used_mem(g->ctx) / over;
  for (int i = 0; i < g->n_ctx_full; i++) n += ggml_used_mem(g->ctx_full[i]) / over;
  return n;
}

static void op_body(void *data) {
  zlm_opjob *j = (zlm_opjob *) data;
  ensure_room(j->g);
  struct ggml_context *ctx = j->g->ctx;
  const char *op = j->op;
  struct ggml_tensor *a = j->a, *b = j->b;
  struct ggml_tensor *r = NULL;
  if (strcmp(op, "add") == 0 || strcmp(op, "mul") == 0) {
    struct ggml_tensor *x = cont(ctx, a), *y = cont(ctx, b);
    if (!can_repeat(y, x)) { struct ggml_tensor *t = x; x = y; y = t; }
    r = strcmp(op, "add") == 0 ? ggml_add(ctx, x, y) : ggml_mul(ctx, x, y);
  } else if (strcmp(op, "sub") == 0) {
    struct ggml_tensor *x = cont(ctx, a), *y = cont(ctx, b);
    if (ggml_are_same_shape(x, y)) {
      r = ggml_sub(ctx, x, y);
    } else if (can_repeat(y, x)) {
      /* design §6.3: a + neg(b), whose backward reduces the broadcast */
      r = ggml_add(ctx, x, ggml_neg(ctx, y));
    } else {
      r = ggml_add(ctx, ggml_neg(ctx, y), x);
    }
  } else if (strcmp(op, "div") == 0) {
    struct ggml_tensor *x = cont(ctx, a), *y = cont(ctx, b);
    if (can_repeat(y, x)) {
      r = ggml_div(ctx, x, y);
    } else {
      r = ggml_div(ctx, ggml_repeat(ctx, x, y), y);
    }
  } else if (strcmp(op, "neg") == 0) {
    r = ggml_neg(ctx, cont(ctx, a));
  } else if (strcmp(op, "scale") == 0) {
    r = ggml_scale(ctx, cont(ctx, a), (float) j->p[0]);
  } else if (strcmp(op, "scale_bias") == 0) {
    r = ggml_scale_bias(ctx, cont(ctx, a), (float) j->p[0], (float) j->p[1]);
  } else if (strcmp(op, "fill_div") == 0) {
    /* number / tensor: c as 0 x + c, so the graph keeps a gradient (ggml's
       FILL has none); an infinite x gives NaN rather than 0 */
    struct ggml_tensor *x = cont(ctx, a);
    r = ggml_div(ctx, ggml_scale_bias(ctx, x, 0.0f, (float) j->p[0]), x);
  } else if (strcmp(op, "mean") == 0) {
    /* only through "unchecked:", for the patch P1 regression test */
    r = ggml_mean(ctx, a);
  } else if (strcmp(op, "sub_raw") == 0) {
    /* only through "unchecked:", for the patch P2 regression test */
    r = ggml_sub(ctx, a, b);
  } else if (strcmp(op, "sum") == 0) {
    r = ggml_sum(ctx, cont(ctx, a));
  } else if (strcmp(op, "sum_rows") == 0) {
    r = ggml_sum_rows(ctx, cont(ctx, a));
  } else if (strcmp(op, "mean_rows") == 0) {
    /* design §6.3: scale(sum_rows(x), 1 / n), not ggml_mean (patch P1) */
    struct ggml_tensor *x = cont(ctx, a);
    r = ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f / (float) x->ne[0]);
  } else if (strcmp(op, "sqr") == 0) {
    r = ggml_sqr(ctx, cont(ctx, a));
  } else if (strcmp(op, "sqrt") == 0) {
    r = ggml_sqrt(ctx, cont(ctx, a));
  } else if (strcmp(op, "exp") == 0) {
    r = ggml_exp(ctx, cont(ctx, a));
  } else if (strcmp(op, "log") == 0) {
    r = ggml_log(ctx, cont(ctx, a));
  } else if (strcmp(op, "abs") == 0) {
    r = ggml_abs(ctx, cont(ctx, a));
  } else if (strcmp(op, "sin") == 0) {
    r = ggml_sin(ctx, cont(ctx, a));
  } else if (strcmp(op, "cos") == 0) {
    r = ggml_cos(ctx, cont(ctx, a));
  } else if (strcmp(op, "cumsum") == 0) {
    r = ggml_cumsum(ctx, cont(ctx, a));
  } else if (strcmp(op, "relu") == 0) {
    r = ggml_relu(ctx, cont(ctx, a));
  } else if (strcmp(op, "silu") == 0) {
    r = ggml_silu(ctx, cont(ctx, a));
  } else if (strcmp(op, "softplus") == 0) {
    r = ggml_softplus(ctx, cont(ctx, a));
  } else if (strcmp(op, "gelu") == 0) {
    r = j->ip[0] ? ggml_gelu(ctx, cont(ctx, a)) : ggml_gelu_erf(ctx, cont(ctx, a));
  } else if (strcmp(op, "sigmoid") == 0) {
    r = ggml_sigmoid(ctx, cont(ctx, a));
  } else if (strcmp(op, "tanh") == 0) {
    r = ggml_tanh(ctx, cont(ctx, a));
  } else if (strcmp(op, "swiglu") == 0) {
    r = ggml_swiglu_split(ctx, cont(ctx, a), cont(ctx, b));
  } else if (strcmp(op, "crossprod") == 0) {
    r = ggml_mul_mat(ctx, cont(ctx, a), cont(ctx, b));
  } else if (strcmp(op, "reshape") == 0) {
    r = ggml_reshape_4d(ctx, cont(ctx, a), j->ip[0], j->ip[1], j->ip[2], j->ip[3]);
  } else if (strcmp(op, "permute") == 0) {
    r = ggml_permute(ctx, a, j->ip[0], j->ip[1], j->ip[2], j->ip[3]);
  } else if (strcmp(op, "transpose") == 0) {
    r = ggml_transpose(ctx, a);
  } else if (strcmp(op, "cont") == 0) {
    r = ggml_cont(ctx, a);
  } else if (strcmp(op, "view") == 0) {
    struct ggml_tensor *x = cont(ctx, a);
    size_t es = ggml_element_size(x);
    size_t nb1 = (size_t) j->ip[0] * es;
    size_t nb2 = nb1 * (size_t) j->ip[1];
    size_t nb3 = nb2 * (size_t) j->ip[2];
    r = ggml_view_4d(ctx, x, j->ip[0], j->ip[1], j->ip[2], j->ip[3], nb1, nb2, nb3,
                     (size_t) j->p[0] * es);
  } else if (strcmp(op, "concat") == 0) {
    r = ggml_concat(ctx, cont(ctx, a), cont(ctx, b), j->ip[0]);
  } else if (strcmp(op, "pad") == 0) {
    r = ggml_pad(ctx, cont(ctx, a), j->ip[0], j->ip[1], j->ip[2], j->ip[3]);
  } else if (strcmp(op, "get_rows") == 0) {
    r = ggml_get_rows(ctx, cont(ctx, a), b);
  } else if (strcmp(op, "clamp") == 0) {
    r = ggml_clamp(ctx, cont(ctx, a), (float) j->p[0], (float) j->p[1]);
  } else if (strcmp(op, "tri") == 0) {
    r = ggml_tri(ctx, cont(ctx, a), (enum ggml_tri_type) j->ip[0]);
  } else if (strcmp(op, "softmax") == 0) {
    r = ggml_soft_max_ext(ctx, cont(ctx, a), b ? cont(ctx, b) : NULL, (float) j->p[0], 0.0f);
  } else if (strcmp(op, "causal_mask") == 0) {
    r = ggml_diag_mask_inf(ctx, cont(ctx, a), 0);
  } else if (strcmp(op, "rope") == 0) {
    r = ggml_rope_ext(ctx, cont(ctx, a), b, NULL, j->ip[0], j->ip[1], 0,
                      (float) j->p[0], 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  } else if (strcmp(op, "rms_norm") == 0) {
    r = ggml_rms_norm(ctx, cont(ctx, a), (float) j->p[0]);
  } else if (strcmp(op, "cross_entropy") == 0) {
    r = ggml_cross_entropy_loss(ctx, cont(ctx, a), cont(ctx, b));
  } else if (strcmp(op, "cast") == 0) {
    r = ggml_cast(ctx, a, (enum ggml_type) j->ip[0]);
  }
  j->out = r;
}

static void shape_error(SEXP call, const char *op, const struct ggml_tensor *a,
                        const struct ggml_tensor *b, const char *why) {
  char sa[96] = "", sb[96] = "";
  if (a) snprintf(sa, sizeof sa, "c(%lld, %lld, %lld, %lld)", (long long) a->ne[0],
                  (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3]);
  if (b) snprintf(sb, sizeof sb, " and c(%lld, %lld, %lld, %lld)", (long long) b->ne[0],
                  (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
  SEXP extra = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_VECTOR_ELT(extra, 0, Rf_mkString(op));
  SET_VECTOR_ELT(extra, 1, call);
  SET_STRING_ELT(nm, 0, Rf_mkChar("op"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("call"));
  Rf_setAttrib(extra, R_NamesSymbol, nm);
  zlm_raise("zulm_shape_error", extra, "%s: %s (operand shapes %s%s).", op, why, sa, sb);
}

static void type_error(const char *op, const struct ggml_tensor *t) {
  zlm_raise("zulm_type_error", R_NilValue,
            "%s takes f32 tensors, not %s; convert with zl_cast(x, \"f32\").", op,
            zlm_type_name(t->type));
}

static void need_f32(const char *op, const struct ggml_tensor *t) {
  if (t != NULL && t->type != GGML_TYPE_F32) type_error(op, t);
}

static int64_t n_el(const struct ggml_tensor *t) { return ggml_nelements(t); }

/* Validation of every operation: what the ggml builder or CPU kernel
   would assert, refused here first. */
static void validate(zlm_opjob *j, SEXP call) {
  const char *op = j->op;
  struct ggml_tensor *a = j->a, *b = j->b;
  if (strcmp(op, "crossprod") == 0) {
    need_f32(op, b);
    if (a->type != GGML_TYPE_F32 && !is_half(a)) type_error(op, a);
    if (a->ne[0] != b->ne[0]) shape_error(call, op, a, b, "the first dimensions differ");
    if (b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0)
      shape_error(call, op, a, b, "the batch dimensions of b are not multiples of a's");
    return;
  }
  if (strcmp(op, "get_rows") == 0) {
    if (a->type != GGML_TYPE_F32 && !is_half(a)) type_error(op, a);
    if (b->type != GGML_TYPE_I32) {
      zlm_raise("zulm_type_error", R_NilValue, "zl_get_rows() takes an i32 index input declared with zl_int().");
    }
    if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1)
      shape_error(call, op, a, b, "x must be a matrix and idx a vector");
    return;
  }
  if (strcmp(op, "cast") == 0) {
    if (a->type != GGML_TYPE_F32 && !is_half(a)) type_error(op, a);
    return;
  }
  if (strcmp(op, "rope") == 0) {
    need_f32(op, a);
    if (b->type != GGML_TYPE_I32) {
      zlm_raise("zulm_type_error", R_NilValue, "zl_rope() takes i32 positions declared with zl_int().");
    }
    if (b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1 || b->ne[0] != a->ne[2])
      shape_error(call, op, a, b, "pos must be a vector with one position per third-dimension slice of x");
    if (j->ip[0] < 2 || j->ip[0] > a->ne[0] || j->ip[0] % 2 != 0)
      shape_error(call, op, a, NULL, "n_dims must be even and at most the first dimension");
    return;
  }
  need_f32(op, a);
  need_f32(op, b);
  if (strcmp(op, "add") == 0 || strcmp(op, "mul") == 0 || strcmp(op, "sub") == 0 ||
      strcmp(op, "div") == 0) {
    if (!can_repeat(b, a) && !can_repeat(a, b))
      shape_error(call, op, a, b, "neither shape broadcasts to the other");
  } else if (strcmp(op, "swiglu") == 0 || strcmp(op, "cross_entropy") == 0) {
    if (!ggml_are_same_shape(a, b)) shape_error(call, op, a, b, "the shapes differ");
  } else if (strcmp(op, "reshape") == 0) {
    int64_t n = (int64_t) j->ip[0] * j->ip[1] * j->ip[2] * j->ip[3];
    if (n != n_el(a)) shape_error(call, op, a, NULL, "the new dimensions hold a different number of elements");
  } else if (strcmp(op, "view") == 0) {
    int64_t n = (int64_t) j->ip[0] * j->ip[1] * j->ip[2] * j->ip[3];
    if (j->p[0] < 0 || (double) n + j->p[0] > (double) n_el(a))
      shape_error(call, op, a, NULL, "the view reaches past the end of x");
  } else if (strcmp(op, "concat") == 0) {
    for (int k = 0; k < GGML_MAX_DIMS; k++) {
      if (k != j->ip[0] && a->ne[k] != b->ne[k])
        shape_error(call, op, a, b, "the shapes differ outside the concatenation axis");
    }
  } else if (strcmp(op, "tri") == 0) {
    if (a->ne[0] != a->ne[1]) shape_error(call, op, a, NULL, "x must hold square matrices");
  } else if (strcmp(op, "softmax") == 0 && b != NULL) {
    if (b->ne[0] != a->ne[0] || b->ne[1] < a->ne[1] || a->ne[2] % b->ne[2] != 0 ||
        a->ne[3] % b->ne[3] != 0)
      shape_error(call, op, a, b,
                  "the mask must have x's first two dimensions and third and fourth of 1 or x's");
  }
}

/* zlm_op(g, op, a, b, num, int, call): one operation; returns a handle. */
SEXP zlm_op(SEXP gptr, SEXP op, SEXP ha, SEXP hb, SEXP num, SEXP ints, SEXP call) {
  zlm_graph *g = zlm_graph_from(gptr);
  if (g->state != ZLM_GRAPH_BUILDING) {
    zlm_raise("zulm_invalid_argument", R_NilValue,
              "Operations on tensors are used inside build() only.");
  }
  zlm_opjob j;
  memset(&j, 0, sizeof j);
  j.g = g;
  j.op = CHAR(STRING_ELT(op, 0));
  /* "unchecked:<op>" skips layer 1 so the tests can reach layer 2 (design
     §13.2); only the internal zlm_op() of the tests uses it. */
  int unchecked = strncmp(j.op, "unchecked:", 10) == 0;
  if (unchecked) j.op += 10;
  j.a = tensor_from(ha, gptr);
  j.b = Rf_isNull(hb) ? NULL : tensor_from(hb, gptr);
  for (int i = 0; i < (int) Rf_xlength(num) && i < 8; i++) j.p[i] = REAL(num)[i];
  for (int i = 0; i < (int) Rf_xlength(ints) && i < 8; i++) j.ip[i] = INTEGER(ints)[i];
  if (!unchecked) validate(&j, call);
  if (strcmp(j.op, "get_rows") == 0) {
    /* the index must be an input, so zl_forward() can check its range */
    int found = 0;
    for (int i = 0; i < g->n_in; i++) found |= g->in[i] == j.b;
    if (!found) {
      zlm_raise("zulm_invalid_argument", R_NilValue, "zl_get_rows() takes an index that is a graph input.");
    }
  }
  char msg[2048] = "";
  int status = zlm_guarded(op_body, &j, msg, sizeof msg);
  if (status == ZLM_STATUS_ENGINE) {
    g->state = ZLM_GRAPH_FAILED;
    SEXP extra = PROTECT(Rf_allocVector(VECSXP, 1));
    Rf_setAttrib(extra, R_NamesSymbol, Rf_mkString("ggml_message"));
    SET_VECTOR_ELT(extra, 0, Rf_mkString(msg));
    zlm_raise("zulm_engine_error", extra, "ggml rejected %s: %s. The graph is discarded.", j.op, msg);
  }
  if (status != ZLM_STATUS_OK || j.out == NULL) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise(status == ZLM_STATUS_MEMORY ? "zulm_memory_error" : "zulm_internal_error", R_NilValue,
              "Building %s failed: %s", j.op, msg[0] ? msg : "unknown operation");
  }
  record_call(g, gptr, call);
  if (strcmp(j.op, "get_rows") == 0) {
    if (g->n_checks == g->cap_checks) {
      int cap = g->cap_checks ? 2 * g->cap_checks : 8;
      zlm_index_check *c = (zlm_index_check *) realloc(g->checks, (size_t) cap * sizeof *c);
      if (c == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
      g->checks = c;
      g->cap_checks = cap;
    }
    g->checks[g->n_checks].idx = j.b;
    g->checks[g->n_checks].bound = j.a->ne[1];
    g->n_checks++;
  }
  return tensor_wrap(gptr, j.out);
}

/* ---- finishing and running ------------------------------------------ */

struct finish_job {
  zlm_graph *g;
  int n;
  struct ggml_tensor **outs;
};

static void finish_body(void *data) {
  struct finish_job *f = (struct finish_job *) data;
  zlm_graph *g = f->g;
  /* every node and leaf of gf is a tensor of the build contexts, an input
     or a parameter; the graph is sized for all of them */
  size_t size = build_tensors(g) + (size_t) g->n_in + (size_t) g->store->n + 64;
  struct ggml_init_params pg = { ggml_graph_overhead_custom(size, false), NULL, true };
  g->ctx_graph = ggml_init(pg);
  if (g->ctx_graph == NULL) GGML_ABORT("out of memory finishing a graph");
  g->gf = ggml_new_graph_custom(g->ctx_graph, size, false);
  for (int i = 0; i < f->n; i++) {
    ggml_set_output(f->outs[i]);
    ggml_build_forward_expand(g->gf, f->outs[i]);
  }
  /* the copy zl_forward() evaluates */
  g->gf_fwd = zlm_graph_dup(g->ctx_fwd, g->gf, f->n, f->outs, g->out_fwd, 0, 0);
}

/* Ends build(): flags and records the outputs, then allocates the
   intermediate tensors within the memory cap. */
SEXP zlm_graph_finish(SEXP gptr, SEXP outputs) {
  zlm_graph *g = zlm_graph_from(gptr);
  if (g->state != ZLM_GRAPH_BUILDING) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "This graph is not being built.");
  }
  int n = (int) Rf_xlength(outputs);
  g->out = (struct ggml_tensor **) calloc((size_t) (n > 0 ? n : 1), sizeof *g->out);
  if (g->out == NULL) zlm_raise("zulm_memory_error", R_NilValue, "Out of memory.");
  for (int i = 0; i < n; i++) {
    struct ggml_tensor *t = tensor_from(VECTOR_ELT(outputs, i), gptr);
    if (t->type != GGML_TYPE_F32) {
      zlm_raise("zulm_type_error", R_NilValue, "Outputs must be f32 tensors; convert with zl_cast(x, \"f32\").");
    }
    /* A parameter or input returned as is, and a view, is copied: the
       graph needs a node of its own to flag as an output, and outputs
       are read as contiguous memory. */
    int copy = t->op == GGML_OP_NONE || !ggml_is_contiguous(t) || t->view_src != NULL;
    g->out[i] = copy ? NULL : t;
  }
  struct finish_job f = { g, n, g->out };
  char msg[2048] = "";
  g->out_fwd = (struct ggml_tensor **) calloc((size_t) (n > 0 ? n : 1), sizeof *g->out_fwd);
  /* the forward copy holds a tensor per node and leaf of gf, which has at
     most this many */
  size_t fsize = build_tensors(g) + (size_t) n + (size_t) g->n_in + (size_t) g->store->n + 64;
  struct ggml_init_params pf = {
    fsize * ggml_tensor_overhead() + ggml_graph_overhead_custom(fsize, false), NULL, true
  };
  g->ctx_fwd = ggml_init(pf);
  if (g->out_fwd == NULL || g->ctx_fwd == NULL) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory finishing the graph.");
  }
  for (int i = 0; i < n; i++) {
    if (g->out[i] == NULL) {
      struct ggml_tensor *t = tensor_from(VECTOR_ELT(outputs, i), gptr);
      zlm_opjob j;
      memset(&j, 0, sizeof j);
      j.g = g; j.op = "cont"; j.a = t;
      int st = zlm_guarded(op_body, &j, msg, sizeof msg);
      if (st != ZLM_STATUS_OK) {
        g->state = ZLM_GRAPH_FAILED;
        zlm_raise("zulm_engine_error", R_NilValue, "ggml rejected an output: %s", msg);
      }
      g->out[i] = j.out;
    }
  }
  int status = zlm_guarded(finish_body, &f, msg, sizeof msg);
  if (status != ZLM_STATUS_OK) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise("zulm_engine_error", R_NilValue, "ggml rejected the graph: %s", msg);
  }
  g->n_out = n;
  g->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (g->galloc == NULL || !ggml_gallocr_reserve(g->galloc, g->gf_fwd)) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory reserving the graph's intermediate tensors.");
  }
  g->bytes_compute = (double) ggml_gallocr_get_buffer_size(g->galloc, 0);
  if (g->bytes_compute + g->bytes_inputs > g->max_bytes) {
    double need = g->bytes_compute + g->bytes_inputs;
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise_memory(need, g->max_bytes, "The graph");
  }
  if (!ggml_gallocr_alloc_graph(g->galloc, g->gf_fwd)) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory allocating the graph.");
  }
  g->backend = ggml_backend_cpu_init();
  if (g->backend == NULL) {
    g->state = ZLM_GRAPH_FAILED;
    zlm_raise("zulm_internal_error", R_NilValue, "ggml's CPU backend could not start.");
  }
  ggml_backend_cpu_set_n_threads(g->backend, g->threads > 0 ? g->threads : 1);
  g->bytes_live = g->bytes_compute + g->bytes_inputs + (double) ggml_get_mem_size(g->ctx) +
                  (double) ggml_get_mem_size(g->ctx_fwd) + (double) ggml_get_mem_size(g->ctx_graph);
  for (int i = 0; i < g->n_ctx_full; i++) g->bytes_live += (double) ggml_get_mem_size(g->ctx_full[i]);
  zlm_live_add(g->bytes_live);
  g->state = ZLM_GRAPH_READY;
  SEXP handles = PROTECT(Rf_allocVector(VECSXP, n));
  for (int i = 0; i < n; i++) SET_VECTOR_ELT(handles, i, tensor_wrap(gptr, g->out[i]));
  UNPROTECT(1);
  return handles;
}

/* Marks a graph whose build() failed in R as unusable. */
SEXP zlm_graph_abandon(SEXP gptr) {
  zlm_graph *g = zlm_graph_from(gptr);
  if (g->state == ZLM_GRAPH_BUILDING) g->state = ZLM_GRAPH_FAILED;
  return R_NilValue;
}

static void need_ready(zlm_graph *g) {
  if (g->state == ZLM_GRAPH_FAILED) {
    zlm_raise("zulm_stale_error", R_NilValue, "This graph failed to build and cannot be used.");
  }
  if (g->state != ZLM_GRAPH_READY) {
    zlm_raise("zulm_invalid_argument", R_NilValue, "This graph is still being built.");
  }
}

/* Writes input i from R values (checked in R for shape and type). */
SEXP zlm_graph_set_input(SEXP gptr, SEXP index, SEXP values) {
  zlm_graph *g = zlm_graph_from(gptr);
  need_ready(g);
  int i = Rf_asInteger(index);
  struct ggml_tensor *t = g->in[i];
  R_xlen_t n = (R_xlen_t) ggml_nelements(t);
  if (Rf_xlength(values) != n) {
    zlm_raise("zulm_shape_error", R_NilValue, "Input '%s' needs %.0f values.", t->name, (double) n);
  }
  if (t->type == GGML_TYPE_I32) {
    zlm_to_i32(values, (int32_t *) t->data, n, t->name);
    return R_NilValue;
  }
  float *d = (float *) t->data;
  R_xlen_t over = 0;
  if (TYPEOF(values) == REALSXP) {
    const double *v = REAL(values);
    for (R_xlen_t k = 0; k < n; k++) {
      if (ISNA(v[k])) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "Input '%s' contains NA.", t->name);
      }
      d[k] = (float) v[k];
      if (isinf(d[k]) && isfinite(v[k])) over++;
    }
  } else if (TYPEOF(values) == INTSXP || TYPEOF(values) == LGLSXP) {
    const int *v = TYPEOF(values) == INTSXP ? INTEGER(values) : LOGICAL(values);
    for (R_xlen_t k = 0; k < n; k++) {
      if (v[k] == NA_INTEGER) {
        zlm_raise("zulm_invalid_argument", R_NilValue, "Input '%s' contains NA.", t->name);
      }
      d[k] = (float) v[k];
    }
  } else {
    zlm_raise("zulm_type_error", R_NilValue, "Input '%s' must be numeric.", t->name);
  }
  if (over > 0) zlm_warn_range(over, t->name);
  return R_NilValue;
}

/* Every index input must lie within its table before a kernel reads it. */
void zlm_graph_check_indices(zlm_graph *g) {
  for (int c = 0; c < g->n_checks; c++) {
    const zlm_index_check *k = &g->checks[c];
    const int32_t *v = (const int32_t *) k->idx->data;
    int64_t n = ggml_nelements(k->idx);
    for (int64_t i = 0; i < n; i++) {
      if (v[i] < 0 || v[i] >= k->bound) {
        zlm_raise("zulm_invalid_argument", R_NilValue,
                  "Index input '%s' holds %d at position %lld; indices count from 0 and must be below %lld.",
                  k->idx->name, (int) v[i], (long long) i + 1, (long long) k->bound);
      }
    }
  }
}

/* The original node of output i, for training on it. */
SEXP zlm_graph_output(SEXP gptr, SEXP index) {
  zlm_graph *g = zlm_graph_from(gptr);
  return R_MakeExternalPtr(g->out[Rf_asInteger(index)], R_NilValue, gptr);
}

/* Runs the graph once; returns the requested outputs as doubles. */
SEXP zlm_graph_compute(SEXP gptr, SEXP which) {
  zlm_graph *g = zlm_graph_from(gptr);
  need_ready(g);
  zlm_graph_check_indices(g);
  enum ggml_status st = ggml_backend_graph_compute(g->backend, g->gf_fwd);
  if (st != GGML_STATUS_SUCCESS) {
    zlm_raise("zulm_internal_error", R_NilValue, "ggml failed to compute the graph (status %d).", (int) st);
  }
  int n = (int) Rf_xlength(which);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  for (int i = 0; i < n; i++) {
    struct ggml_tensor *t = g->out_fwd[INTEGER(which)[i]];
    SEXP v = Rf_allocVector(REALSXP, (R_xlen_t) ggml_nelements(t));
    SET_VECTOR_ELT(out, i, v);
    zlm_read_values(t, REAL(v));
  }
  UNPROTECT(1);
  return out;
}

SEXP zlm_graph_info(SEXP gptr) {
  zlm_graph *g = zlm_graph_from(gptr);
  int n = g->gf ? ggml_graph_n_nodes(g->gf) : 0;
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SEXP ops = PROTECT(Rf_allocVector(STRSXP, n));
  for (int i = 0; i < n; i++) {
    SET_STRING_ELT(ops, i, Rf_mkChar(ggml_op_desc(ggml_graph_node(g->gf, i))));
  }
  SET_VECTOR_ELT(out, 0, ops);
  SET_VECTOR_ELT(out, 1, Rf_ScalarReal(g->bytes_inputs));
  SET_VECTOR_ELT(out, 2, Rf_ScalarReal(g->bytes_compute));
  SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(g->state));
  UNPROTECT(2);
  return out;
}
