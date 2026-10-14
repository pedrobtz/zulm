// zlm_shim.cpp: assertions during graph construction as R errors
// (design §13.2, layer 2).
//
// ggml_abort() calls the callback set by ggml_set_abort_callback() and then
// aborts. While zlm_guarded() runs, the callback throws instead of returning,
// so abort is never reached: the exception unwinds through ggml's C frames
// (they carry unwind tables by default on every supported platform) back to
// the catch below. Outside zlm_guarded() it records the message for the
// crash log and returns, and ggml stops the process: the fatal path.
#include <cstring>
#include <new>

#include "zlm_glue.h"

namespace {

thread_local int guard_depth = 0;

struct engine_abort {
  char message[2048];
};

void copy_message(char *dst, size_t cap, const char *src) {
  if (cap == 0) {
    return;
  }
  std::strncpy(dst, src, cap - 1);
  dst[cap - 1] = '\0';
}

}  // namespace

extern "C" void zlm_abort_callback(const char *message) {
  if (guard_depth > 0) {
    engine_abort e;
    copy_message(e.message, sizeof e.message, message);
    throw e;
  }
  zlm_fatal_message(message);
}

extern "C" int zlm_guarded(void (*fn)(void *), void *data, char *msg, size_t msgsize) {
  int status = ZLM_STATUS_OK;
  guard_depth++;
  try {
    fn(data);
  } catch (const engine_abort &e) {
    copy_message(msg, msgsize, e.message);
    status = ZLM_STATUS_ENGINE;
  } catch (const std::bad_alloc &) {
    copy_message(msg, msgsize, "out of memory");
    status = ZLM_STATUS_MEMORY;
  } catch (...) {
    copy_message(msg, msgsize, "unexpected C++ exception");
    status = ZLM_STATUS_INTERNAL;
  }
  guard_depth--;
  return status;
}
