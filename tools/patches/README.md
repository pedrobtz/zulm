# Patches to ggml

`tools/vendor-ggml` applies every `*.patch` here, in name order, to the
pinned upstream tag; `src/ggml/VERSION` and `zulm_info()$ggml_patches`
list them. Each file's header names its upstream pull request (design
§9.3). A test fails if a re-vendoring drops one.

| File | Design | What | Upstream pull request |
| --- | --- | --- | --- |
| `p01-mean-backward-accumulates.patch` | P1 | `MEAN` backward repeats the scaled gradient | not opened yet |
| `p02-sub-div-broadcast-backward.patch` | P2 | `SUB` and `DIV` backward reduce a broadcast operand's gradient | not opened yet |
| `p07-opt-global-norm-clipping.patch` | P7 | global-norm gradient clipping in `ggml-opt` | not opened yet |
| `p08-c23-static-assert.patch` | P8 | no `static_assert` macro under C23 | not opened yet |
| `p09-gguf-type-range.patch` | P9 | a GGUF tensor type is checked before it becomes a `ggml_type` | not opened yet |
| `p10-function-pointer-types.patch` | P10 | type-table functions called through their own types | not opened yet |

Each applies to a clean tree with `patch -p1 -d src/ggml`.
