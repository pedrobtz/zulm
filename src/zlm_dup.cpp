// zlm_dup.cpp: a copy of a forward graph for zl_forward() (design §4).
//
// ggml-opt, with static graphs, evaluates a copy of the graph it is given
// and lets the copy's tensors inherit the original's data pointers. So the
// original nodes must stay unallocated for training, and zl_forward()
// evaluates a copy of its own, allocated by ggml's graph allocator, which
// may reuse memory between activations. This follows ggml-opt's
// map_tensor() and dup_graph() (ggml-opt.cpp), without gradients.
#include "zlm_compat.h"

#include <cstring>
#include <unordered_map>

#include "ggml.h"
#include "ggml-impl.h"

namespace {

ggml_tensor *map_tensor(std::unordered_map<ggml_tensor *, ggml_tensor *> &m, ggml_context *ctx,
                        ggml_tensor *t) {
  if (t == nullptr) {
    return nullptr;
  }
  auto it = m.find(t);
  if (it != m.end()) {
    return it->second;
  }
  ggml_tensor *c = ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne);
  m[t] = c;
  c->op = t->op;
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    c->nb[i] = t->nb[i];
  }
  c->flags = t->flags;
  std::memcpy(c->op_params, t->op_params, sizeof(t->op_params));
  std::strcpy(c->name, t->name);
  c->data = t->data;
  c->buffer = t->buffer;
  c->extra = t->extra;
  c->view_offs = t->view_offs;
  c->view_src = map_tensor(m, ctx, t->view_src);
  for (int i = 0; i < GGML_MAX_SRC; i++) {
    c->src[i] = map_tensor(m, ctx, t->src[i]);
  }
  return c;
}

}  // namespace

// Copies `src` into `ctx`; map_out[i] receives the copy of map_in[i].
extern "C" ggml_cgraph *zlm_graph_dup(ggml_context *ctx, ggml_cgraph *src, int n,
                                      ggml_tensor **map_in, ggml_tensor **map_out) {
  std::unordered_map<ggml_tensor *, ggml_tensor *> m;
  ggml_cgraph *dst = ggml_new_graph_custom(ctx, src->size, false);
  for (int i = 0; i < src->n_leafs; i++) {
    ggml_build_forward_expand(dst, map_tensor(m, ctx, src->leafs[i]));
  }
  for (int i = 0; i < src->n_nodes; i++) {
    ggml_build_forward_expand(dst, map_tensor(m, ctx, src->nodes[i]));
  }
  for (int i = 0; i < n; i++) {
    map_out[i] = map_tensor(m, ctx, map_in[i]);
  }
  return dst;
}
