/*
 * zlm_variant_begin.h: the force-included header of design §13.1.
 *
 * Every translation unit under zlm_v/ includes this file, then one ggml
 * source, then zlm_variant_end.h. The macro its wrapper defines selects
 * the ISA the unit is compiled for, by pragma rather than by a -m flag,
 * so R's check sees no non-portable flag:
 *
 *   ZLM_TARGET_SSE42  x86-64 with SSE4.2 and POPCNT
 *   ZLM_TARGET_AVX2   x86-64 with AVX2, FMA, F16C, BMI2
 *   (none)            the compiler's default target
 *
 * GCC takes '#pragma GCC target', clang '#pragma clang attribute push'
 * of a target attribute. Neither defines the ISA macros ggml selects its
 * kernels by in every case (GCC does in C, not in C++; clang never does),
 * so they are defined here. When the command line already defines them
 * (the native build hook of §13.4), no pragma is used: the flags decide.
 * '#pragma GCC optimize("unroll-loops")' asks GCC for the unrolling
 * -funroll-loops gives; clang unrolls at -O2 already.
 */

/* ggml's assert() checks are debugging aids that upstream's release builds
   compile out (CMake defines NDEBUG). R does not define NDEBUG, and an
   assert() calls the C library's abort path directly, past zlm_compat.h:
   a NaN reaching softmax's debug check killed the R session. Defined
   here, before any header, the checks go as they do upstream; GGML_ASSERT,
   which zulm routes to its abort callback, is unaffected. */
#ifndef NDEBUG
#define NDEBUG 1
#endif
#ifndef ZLM_VARIANT_BEGIN_H
#define ZLM_VARIANT_BEGIN_H

/* The feature-test macros upstream's CMake defines (ggml/src/CMakeLists.txt),
   before any system header. */
#if !defined(_XOPEN_SOURCE) && !defined(_AIX)
#if defined(__OpenBSD__)
#define _XOPEN_SOURCE 700
#else
#define _XOPEN_SOURCE 600
#endif
#endif
#if (defined(__linux__) || defined(__ANDROID__)) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#if defined(__FreeBSD__) && !defined(__BSD_VISIBLE)
#define __BSD_VISIBLE 1
#endif
#if defined(__NetBSD__) && !defined(_NETBSD_SOURCE)
#define _NETBSD_SOURCE
#endif
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("unroll-loops")
#endif

#if defined(ZLM_TARGET_AVX2) && !defined(__AVX2__)
#define ZLM_TARGET_STRING "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi,bmi2"
#define ZLM_TARGET_PRAGMA 1
#elif defined(ZLM_TARGET_SSE42) && !defined(__SSE4_2__)
#define ZLM_TARGET_STRING "sse3,ssse3,sse4.1,sse4.2,popcnt"
#define ZLM_TARGET_PRAGMA 1
#endif

#ifdef ZLM_TARGET_PRAGMA
#if defined(__clang__)
#if defined(ZLM_TARGET_AVX2)
#pragma clang attribute push(__attribute__((target("sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi,bmi2"))), apply_to = function)
#else
#pragma clang attribute push(__attribute__((target("sse3,ssse3,sse4.1,sse4.2,popcnt"))), apply_to = function)
#endif
#elif defined(__GNUC__)
#if defined(ZLM_TARGET_AVX2)
#pragma GCC target("sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,f16c,bmi,bmi2")
#else
#pragma GCC target("sse3,ssse3,sse4.1,sse4.2,popcnt")
#endif
#endif
#ifndef __SSE3__
#define __SSE3__ 1
#endif
#ifndef __SSSE3__
#define __SSSE3__ 1
#endif
#ifndef __SSE4_1__
#define __SSE4_1__ 1
#endif
#ifndef __SSE4_2__
#define __SSE4_2__ 1
#endif
#ifndef __POPCNT__
#define __POPCNT__ 1
#endif
#if defined(ZLM_TARGET_AVX2)
#ifndef __AVX__
#define __AVX__ 1
#endif
#ifndef __AVX2__
#define __AVX2__ 1
#endif
#ifndef __FMA__
#define __FMA__ 1
#endif
#ifndef __F16C__
#define __F16C__ 1
#endif
#ifndef __BMI__
#define __BMI__ 1
#endif
#ifndef __BMI2__
#define __BMI2__ 1
#endif
#endif
#endif /* ZLM_TARGET_PRAGMA */

#include "zlm_compat.h"

#endif
