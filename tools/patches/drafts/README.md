# Patch drafts

Drafts of the ggml patches of design §9.3, written against `v0.26.0`.
`tools/vendor-ggml` applies only `tools/patches/*.patch`, so nothing here
reaches `src/ggml/` until Stage 4 moves a draft up a directory, once its
behaviour is pinned by tests and its upstream pull request is open.

| Draft | Design | Upstream pull request |
| --- | --- | --- |
| `0001-mean-backward-accumulates.patch` | P1 | not opened yet |
| `0002-sub-div-broadcast-backward.patch` | P2 | not opened yet |
| `0003-opt-global-norm-clipping.patch` | P7 | not opened yet |

Each applies to the vendored tree with `patch -p1 -d src/ggml`.
