# Patches to ggml

`tools/vendor-ggml` applies every `*.patch` here, in name order, to the
pinned upstream tag; `src/ggml/VERSION` and `zulm_info()$ggml_patches`
list them. Each file's header names its upstream pull request (design
§9.3). A test fails if a re-vendoring drops one.

| File | Design | What | Upstream pull request |
| --- | --- | --- | --- |
| `p01-mean-backward-accumulates.patch` | P1 | `MEAN` backward repeats the scaled gradient | issue [ggml#1663](https://github.com/ggml-org/ggml/issues/1663) |
| `p02-sub-div-broadcast-backward.patch` | P2 | `SUB` and `DIV` backward reduce a broadcast operand's gradient | issue [ggml#1663](https://github.com/ggml-org/ggml/issues/1663) |
| `p07-opt-global-norm-clipping.patch` | P7 | global-norm gradient clipping in `ggml-opt` | not opened yet |
| `p08-c23-static-assert.patch` | P8 | no `static_assert` macro under C23 | issue [ggml#1662](https://github.com/ggml-org/ggml/issues/1662) |
| `p09-gguf-type-range.patch` | P9 | a GGUF tensor type is checked before it becomes a `ggml_type` | issue [ggml#1660](https://github.com/ggml-org/ggml/issues/1660) |
| `p10-function-pointer-types.patch` | P10 | type-table functions called through their own types | issue [ggml#1661](https://github.com/ggml-org/ggml/issues/1661) |
| `p11-gguf-key-type-range.patch` | P11 | a GGUF key's type is checked before it becomes a `gguf_type` | issue [ggml#1660](https://github.com/ggml-org/ggml/issues/1660) |

Each applies to a clean tree with `patch -p1 -d src/ggml`.
