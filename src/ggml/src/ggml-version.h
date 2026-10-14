#pragma once

/* Written by tools/vendor-ggml; replaces upstream's ggml-version.h.in. */
#define GGML_VERSION "0.26.0"
#define GGML_COMMIT  "d7cb574130e6f01ad25b3289685489200febcd74"
/* The tools/patches/ files applied, comma-separated (design §9.3). */
#define ZULM_GGML_PATCHES "p01-mean-backward-accumulates.patch,p02-sub-div-broadcast-backward.patch,p07-opt-global-norm-clipping.patch,p08-c23-static-assert.patch,p09-gguf-type-range.patch,p10-function-pointer-types.patch"
