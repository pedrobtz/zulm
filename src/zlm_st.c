/*
 * zlm_st.c: safetensors in C (design §7.4).
 *
 * C reads the eight-byte header length and the header, hands the header to
 * R (parsed by zujson and validated there), and then reads each tensor's
 * bytes straight into an R array or a store tensor, converting the type;
 * writing goes the other way, the header serialised in R.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "zlm_r.h"
#include "zlm_st.h"
#include "zlm_store.h"

#if defined(_WIN32)
#define zlm_fseek _fseeki64
#define zlm_ftell _ftelli64
#else
#define zlm_fseek fseeko
#define zlm_ftell ftello
#endif

#define ZLM_ST_HEADER_CAP (100.0 * 1024 * 1024)

enum { ST_F32, ST_F16, ST_BF16, ST_F64, ST_I32, ST_I64 };

static int st_code(const char *dtype) {
  if (strcmp(dtype, "F32") == 0) return ST_F32;
  if (strcmp(dtype, "F16") == 0) return ST_F16;
  if (strcmp(dtype, "BF16") == 0) return ST_BF16;
  if (strcmp(dtype, "F64") == 0) return ST_F64;
  if (strcmp(dtype, "I32") == 0) return ST_I32;
  if (strcmp(dtype, "I64") == 0) return ST_I64;
  return -1;
}

static size_t st_size(int code) {
  switch (code) {
    case ST_F16: case ST_BF16: return 2;
    case ST_F32: case ST_I32: return 4;
    default: return 8;
  }
}

static FILE *st_open(const char *path, const char *mode) {
  FILE *f = ggml_fopen(path, mode);
  if (f == NULL) {
    zlm_raise("zulm_file_error", R_NilValue, "Could not open '%s'.", path);
  }
  return f;
}

/* list(header = <string>, data_start = <double>, file_size = <double>) */
SEXP zlm_st_header(SEXP path) {
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  FILE *f = st_open(fname, "rb");
  unsigned char len[8];
  if (fread(len, 1, 8, f) != 8) {
    fclose(f);
    zlm_raise("zulm_file_error", R_NilValue, "'%s' is too short to be a safetensors file.", fname);
  }
  uint64_t n = 0;
  for (int i = 7; i >= 0; i--) n = (n << 8) | len[i];
  zlm_fseek(f, 0, SEEK_END);
  double size = (double) zlm_ftell(f);
  if ((double) n < 2 || (double) n > ZLM_ST_HEADER_CAP || 8.0 + (double) n > size) {
    fclose(f);
    zlm_raise("zulm_file_error", R_NilValue,
              "'%s' declares a header of %.0f bytes, which the file (%.0f bytes) cannot hold.",
              fname, (double) n, size);
  }
  char *buf = (char *) malloc((size_t) n);
  if (buf == NULL) {
    fclose(f);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory reading a safetensors header.");
  }
  zlm_fseek(f, 8, SEEK_SET);
  size_t got = fread(buf, 1, (size_t) n, f);
  fclose(f);
  if (got != (size_t) n) {
    free(buf);
    zlm_raise("zulm_file_error", R_NilValue, "'%s' is truncated in its header.", fname);
  }
  for (uint64_t i = 0; i < n; i++) {
    if (buf[i] == '\0') {
      free(buf);
      zlm_raise("zulm_file_error", R_NilValue, "'%s' has a NUL byte in its header.", fname);
    }
  }
  SEXP hdr = PROTECT(Rf_mkCharLenCE(buf, (int) n, CE_UTF8));
  free(buf);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SET_VECTOR_ELT(out, 0, Rf_ScalarString(hdr));
  SET_VECTOR_ELT(out, 1, Rf_ScalarReal(8.0 + (double) n));
  SET_VECTOR_ELT(out, 2, Rf_ScalarReal(size));
  UNPROTECT(2);
  return out;
}

static float half_to_f32(uint16_t h) {
  ggml_fp16_t x;
  memcpy(&x, &h, 2);
  return ggml_fp16_to_fp32(x);
}

static float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t) h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

/* Reads one tensor's raw bytes. */
static void *st_bytes(FILE *f, const char *fname, double start, double nbytes) {
  void *buf = malloc(nbytes > 0 ? (size_t) nbytes : 1);
  if (buf == NULL) {
    fclose(f);
    zlm_raise("zulm_memory_error", R_NilValue, "Out of memory reading '%s'.", fname);
  }
  if (zlm_fseek(f, (long long) start, SEEK_SET) != 0 ||
      (nbytes > 0 && fread(buf, 1, (size_t) nbytes, f) != (size_t) nbytes)) {
    free(buf);
    fclose(f);
    zlm_raise("zulm_file_error", R_NilValue, "'%s' is truncated: a tensor's data could not be read.", fname);
  }
  return buf;
}

/* zl_safetensors_read(): tensors as R vectors, f32 values as doubles,
   integers as integers. */
SEXP zlm_st_read(SEXP path, SEXP data_start, SEXP begins, SEXP ends, SEXP dtypes) {
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  R_xlen_t k = Rf_xlength(begins);
  double base = Rf_asReal(data_start);
  FILE *f = st_open(fname, "rb");
  SEXP out = PROTECT(Rf_allocVector(VECSXP, k));
  R_xlen_t over = 0;
  for (R_xlen_t t = 0; t < k; t++) {
    int code = st_code(CHAR(STRING_ELT(dtypes, t)));
    double nb = REAL(ends)[t] - REAL(begins)[t];
    size_t sz = st_size(code);
    R_xlen_t n = (R_xlen_t) (nb / (double) sz);
    void *buf = st_bytes(f, fname, base + REAL(begins)[t], nb);
    SEXP v;
    if (code == ST_I32 || code == ST_I64) {
      v = Rf_allocVector(INTSXP, n);
      SET_VECTOR_ELT(out, t, v);
      for (R_xlen_t i = 0; i < n; i++) {
        int64_t x = code == ST_I32 ? (int64_t) ((int32_t *) buf)[i] : ((int64_t *) buf)[i];
        if (x > 2147483647LL || x < -2147483647LL) {
          free(buf);
          fclose(f);
          zlm_raise("zulm_type_error", R_NilValue, "'%s' holds an integer beyond R's integer range.", fname);
        }
        INTEGER(v)[i] = (int) x;
      }
    } else {
      v = Rf_allocVector(REALSXP, n);
      SET_VECTOR_ELT(out, t, v);
      for (R_xlen_t i = 0; i < n; i++) {
        float x;
        switch (code) {
          case ST_F32: x = ((float *) buf)[i]; break;
          case ST_F16: x = half_to_f32(((uint16_t *) buf)[i]); break;
          case ST_BF16: x = bf16_to_f32(((uint16_t *) buf)[i]); break;
          default: {
            double d = ((double *) buf)[i];
            x = (float) d;
            if (isinf(x) && isfinite(d)) over++;
          }
        }
        REAL(v)[i] = x;
      }
    }
    free(buf);
  }
  fclose(f);
  if (over > 0) zlm_warn_range(over, "the file");
  UNPROTECT(1);
  return out;
}

/* zl_load_safetensors(): tensors into store parameters, converting to each
   parameter's type (f16 and bf16 kept as they are when they match). */
SEXP zlm_st_fill(SEXP sptr, SEXP path, SEXP data_start, SEXP params, SEXP begins, SEXP ends,
                 SEXP dtypes) {
  zlm_store *s = zlm_store_from(sptr);
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  R_xlen_t k = Rf_xlength(begins);
  double base = Rf_asReal(data_start);
  FILE *f = st_open(fname, "rb");
  R_xlen_t over = 0;
  for (R_xlen_t t = 0; t < k; t++) {
    struct ggml_tensor *dst = zlm_store_find(s, Rf_translateCharUTF8(STRING_ELT(params, t)));
    int code = st_code(CHAR(STRING_ELT(dtypes, t)));
    if (dst == NULL || code == ST_I32 || code == ST_I64 || code < 0) {
      fclose(f);
      zlm_raise("zulm_internal_error", R_NilValue, "A checked tensor could not be loaded.");
    }
    double nb = REAL(ends)[t] - REAL(begins)[t];
    int64_t n = ggml_nelements(dst);
    void *buf = st_bytes(f, fname, base + REAL(begins)[t], nb);
    if ((code == ST_F16 && dst->type == GGML_TYPE_F16) || (code == ST_BF16 && dst->type == GGML_TYPE_BF16) ||
        (code == ST_F32 && dst->type == GGML_TYPE_F32)) {
      memcpy(dst->data, buf, ggml_nbytes(dst));
    } else {
      float *tmp = (float *) malloc((size_t) n * sizeof(float));
      if (tmp == NULL) {
        free(buf);
        fclose(f);
        zlm_raise("zulm_memory_error", R_NilValue, "Out of memory converting '%s'.", fname);
      }
      for (int64_t i = 0; i < n; i++) {
        switch (code) {
          case ST_F32: tmp[i] = ((float *) buf)[i]; break;
          case ST_F16: tmp[i] = half_to_f32(((uint16_t *) buf)[i]); break;
          case ST_BF16: tmp[i] = bf16_to_f32(((uint16_t *) buf)[i]); break;
          default: {
            double d = ((double *) buf)[i];
            tmp[i] = (float) d;
            if (isinf(tmp[i]) && isfinite(d)) over++;
          }
        }
      }
      if (dst->type == GGML_TYPE_F32) {
        memcpy(dst->data, tmp, (size_t) n * sizeof(float));
      } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp, (ggml_fp16_t *) dst->data, n);
      } else {
        ggml_fp32_to_bf16_row(tmp, (ggml_bf16_t *) dst->data, n);
      }
      free(tmp);
    }
    free(buf);
  }
  fclose(f);
  if (over > 0) zlm_warn_range(over, "the file");
  return R_NilValue;
}

/* zl_save_safetensors(): the header (already padded) and every parameter
   in store order as `type`. */
SEXP zlm_st_write(SEXP sptr, SEXP path, SEXP header, SEXP type) {
  zlm_store *s = zlm_store_from(sptr);
  const char *fname = Rf_translateCharUTF8(STRING_ELT(path, 0));
  enum ggml_type to = zlm_type_from_string(CHAR(STRING_ELT(type, 0)));
  const char *hdr = CHAR(STRING_ELT(header, 0));
  uint64_t n = (uint64_t) strlen(hdr);
  FILE *f = ggml_fopen(fname, "wb");
  if (f == NULL) {
    zlm_raise("zulm_file_error", R_NilValue, "Could not write '%s'.", fname);
  }
  unsigned char len[8];
  for (int i = 0; i < 8; i++) len[i] = (unsigned char) ((n >> (8 * i)) & 0xff);
  int ok = fwrite(len, 1, 8, f) == 8 && fwrite(hdr, 1, (size_t) n, f) == (size_t) n;
  for (int t = 0; t < s->n && ok; t++) {
    struct ggml_tensor *src = s->tensors[t];
    if (s->kinds[t] != ZLM_KIND_PARAM) continue;
    int64_t m = ggml_nelements(src);
    if (src->type == to) {
      ok = fwrite(src->data, 1, ggml_nbytes(src), f) == ggml_nbytes(src);
      continue;
    }
    double *dv = (double *) malloc((size_t) m * sizeof(double));
    float *fv = (float *) malloc((size_t) m * sizeof(float));
    void *ov = malloc((size_t) m * ggml_type_size(to));
    if (dv == NULL || fv == NULL || ov == NULL) {
      free(dv); free(fv); free(ov);
      fclose(f);
      zlm_raise("zulm_memory_error", R_NilValue, "Out of memory writing '%s'.", fname);
    }
    zlm_read_values(src, dv);
    for (int64_t i = 0; i < m; i++) fv[i] = (float) dv[i];
    if (to == GGML_TYPE_F32) memcpy(ov, fv, (size_t) m * sizeof(float));
    else if (to == GGML_TYPE_F16) ggml_fp32_to_fp16_row(fv, (ggml_fp16_t *) ov, m);
    else ggml_fp32_to_bf16_row(fv, (ggml_bf16_t *) ov, m);
    ok = fwrite(ov, 1, (size_t) m * ggml_type_size(to), f) == (size_t) m * ggml_type_size(to);
    free(dv); free(fv); free(ov);
  }
  if (fclose(f) != 0) ok = 0;
  if (!ok) {
    zlm_raise("zulm_file_error", R_NilValue, "Could not write '%s'.", fname);
  }
  return R_NilValue;
}
