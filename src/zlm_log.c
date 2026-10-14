/*
 * zlm_log.c: the targets of zlm_compat.h (design §9.2, §6.8, §13.2).
 *
 * Text ggml writes to stdout or stderr is appended to a bounded log
 * buffer that R drains on the main thread; writes to any other stream go
 * to the C library unchanged (gguf.cpp writes files through fwrite). This
 * file must not include zlm_compat.h.
 */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "zlm_glue.h"

char zlm_stdout_tag;
char zlm_stderr_tag;

#define ZLM_LOG_CAP 65536

static char zlm_log_buf[ZLM_LOG_CAP];
static size_t zlm_log_len;
static int zlm_log_dropped;

/* ggml may write from pool threads; the buffer is guarded by ggml's own
   critical section (ggml-threading.h), a process-wide mutex. */
void ggml_critical_section_start(void);
void ggml_critical_section_end(void);

static int zlm_is_tag(FILE *stream) {
  return stream == (FILE *) &zlm_stdout_tag || stream == (FILE *) &zlm_stderr_tag;
}

void zlm_log_append(const char *s, size_t n) {
  ggml_critical_section_start();
  size_t room = ZLM_LOG_CAP - 1 - zlm_log_len;
  if (n > room) {
    n = room;
    zlm_log_dropped = 1;
  }
  memcpy(zlm_log_buf + zlm_log_len, s, n);
  zlm_log_len += n;
  zlm_log_buf[zlm_log_len] = '\0';
  ggml_critical_section_end();
}

size_t zlm_log_take(char *out, size_t cap, int *dropped) {
  ggml_critical_section_start();
  size_t n = zlm_log_len < cap - 1 ? zlm_log_len : cap - 1;
  memcpy(out, zlm_log_buf, n);
  out[n] = '\0';
  *dropped = zlm_log_dropped;
  zlm_log_len = 0;
  zlm_log_dropped = 0;
  ggml_critical_section_end();
  return n;
}

int zlm_vfprintf(FILE *stream, const char *fmt, va_list ap) {
  if (!zlm_is_tag(stream)) {
    return vfprintf(stream, fmt, ap);
  }
  char tmp[4096];
  int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
  if (n > 0) {
    zlm_log_append(tmp, (size_t) n < sizeof tmp ? (size_t) n : sizeof tmp - 1);
  }
  return n;
}

int zlm_fprintf(FILE *stream, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = zlm_vfprintf(stream, fmt, ap);
  va_end(ap);
  return n;
}

int zlm_vprintf(const char *fmt, va_list ap) {
  return zlm_vfprintf((FILE *) &zlm_stdout_tag, fmt, ap);
}

int zlm_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = zlm_vprintf(fmt, ap);
  va_end(ap);
  return n;
}

/* sprintf without linking sprintf: callers guarantee the buffer is large
   enough, exactly as for sprintf. */
int zlm_sprintf(char *buf, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, (size_t) 1 << 30, fmt, ap);
  va_end(ap);
  return n;
}

int zlm_fputs(const char *s, FILE *stream) {
  if (!zlm_is_tag(stream)) {
    return fputs(s, stream);
  }
  zlm_log_append(s, strlen(s));
  return 1;
}

int zlm_puts(const char *s) {
  zlm_log_append(s, strlen(s));
  zlm_log_append("\n", 1);
  return 1;
}

int zlm_fputc(int c, FILE *stream) {
  if (!zlm_is_tag(stream)) {
    return fputc(c, stream);
  }
  char ch = (char) c;
  zlm_log_append(&ch, 1);
  return (unsigned char) c;
}

int zlm_putchar(int c) {
  return zlm_fputc(c, (FILE *) &zlm_stdout_tag);
}

int zlm_fflush(FILE *stream) {
  if (stream == NULL || zlm_is_tag(stream)) {
    return 0;
  }
  return fflush(stream);
}

size_t zlm_fwrite(const void *ptr, size_t size, size_t n, FILE *stream) {
  if (!zlm_is_tag(stream)) {
    return fwrite(ptr, size, n, stream);
  }
  zlm_log_append((const char *) ptr, size * n);
  return n;
}

/* The fatal path of design §13.2: an assertion outside graph construction.
   The message goes to file descriptor 2 for the crash log; then the process
   stops as abort() would, without linking abort. */
void zlm_fatal_message(const char *msg) {
  static const char prefix[] = "zulm: fatal ggml assertion: ";
  size_t n = strlen(msg);
#if defined(_WIN32)
  _write(2, prefix, (unsigned) (sizeof prefix - 1));
  _write(2, msg, (unsigned) n);
  _write(2, "\n", 1);
#else
  ssize_t r = write(2, prefix, sizeof prefix - 1);
  r = write(2, msg, n);
  r = write(2, "\n", 1);
  (void) r;
#endif
}

void zlm_abort(void) {
  __builtin_trap();
}
