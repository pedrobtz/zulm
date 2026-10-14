/*
 * zlm_compat.h: build-level adaptation of ggml without patches (design §9.2).
 *
 * Included by every wrapper translation unit before the ggml source it
 * wraps. R's compiled-code check reports stdout, stderr, printf, puts,
 * putchar, sprintf, abort and _Exit in a shared object; this header
 * routes ggml's uses of them through zulm:
 *
 * - writes to stdout or stderr go to zulm's log buffer (design §6.8);
 *   the same functions on any other stream behave as the C library's;
 * - abort() becomes zlm_abort(), which traps: the fatal path of design
 *   §13.2 stays fatal, without linking abort;
 * - _Exit() in ggml's backtrace child (never reached: zulm always sets
 *   the abort callback) becomes zlm_abort() as well.
 *
 * The macros cover names ggml calls unqualified; ggml never writes std::fprintf and the like (checked by
 * tests/testthat/test-vendor.R).
 */
#ifndef ZLM_COMPAT_H
#define ZLM_COMPAT_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
extern "C" {
#endif

extern char zlm_stdout_tag;
extern char zlm_stderr_tag;

int zlm_printf(const char *fmt, ...);
int zlm_vprintf(const char *fmt, va_list ap);
int zlm_fprintf(FILE *stream, const char *fmt, ...);
int zlm_vfprintf(FILE *stream, const char *fmt, va_list ap);
int zlm_sprintf(char *buf, const char *fmt, ...);
int zlm_fputs(const char *s, FILE *stream);
int zlm_puts(const char *s);
int zlm_fputc(int c, FILE *stream);
int zlm_putchar(int c);
int zlm_fflush(FILE *stream);
size_t zlm_fwrite(const void *ptr, size_t size, size_t n, FILE *stream);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void zlm_abort(void);

#ifdef __cplusplus
}
#endif

#undef stdout
#undef stderr
#define stdout ((FILE *) &zlm_stdout_tag)
#define stderr ((FILE *) &zlm_stderr_tag)

#undef printf
#undef vprintf
#undef fprintf
#undef vfprintf
#undef sprintf
#undef fputs
#undef puts
#undef fputc
#undef putc
#undef putchar
#undef fflush
#undef fwrite
/* Function-like, so that __attribute__((format(printf, ...))) and the like
   are left alone. */
#define printf(...) zlm_printf(__VA_ARGS__)
#define vprintf(...) zlm_vprintf(__VA_ARGS__)
#define fprintf(...) zlm_fprintf(__VA_ARGS__)
#define vfprintf(...) zlm_vfprintf(__VA_ARGS__)
#define sprintf(...) zlm_sprintf(__VA_ARGS__)
#define fputs(...) zlm_fputs(__VA_ARGS__)
#define puts(...) zlm_puts(__VA_ARGS__)
#define fputc(...) zlm_fputc(__VA_ARGS__)
#define putc(...) zlm_fputc(__VA_ARGS__)
#define putchar(...) zlm_putchar(__VA_ARGS__)
#define fflush(...) zlm_fflush(__VA_ARGS__)
#define fwrite(...) zlm_fwrite(__VA_ARGS__)
#define abort() zlm_abort()
#define _Exit(status) zlm_abort()

#endif
