/* C ABI test, compiled as strict C11 (no CUDA headers): error paths, the
 * manifest buffer protocol, a forced kernel-launch failure the process must
 * survive, and aegis_gemm_f16 against a CPU reference with the bench
 * tolerance (max_abs_err / max|ref| <= 4e-3). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aegis/aegis.h"

static int failures = 0;

#define EXPECT(cond, ...)                                    \
  do {                                                       \
    if (!(cond)) {                                           \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
      fprintf(stderr, __VA_ARGS__);                          \
      fputc('\n', stderr);                                   \
      ++failures;                                            \
    }                                                        \
  } while (0)

/* IEEE binary16 <-> float, round-to-nearest-even; enough for test data. */
static uint16_t f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) {  /* subnormal or zero */
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return (uint16_t)(sign | h);
  }
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return (uint16_t)(sign | h);
}

static float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu;
  float f;
  if (exp == 0) {
    f = ldexpf((float)mant, -24);
  } else if (exp == 31) {
    f = mant ? NAN : INFINITY;
  } else {
    f = ldexpf((float)(mant | 0x400u), (int)exp - 25);
  }
  return sign ? -f : f;
}

static void test_errors(aegis_context* ctx) {
  EXPECT(aegis_create(NULL, 0) == AEGIS_ERR_INVALID_ARGUMENT, "create(NULL)");
  aegis_context* bad = NULL;
  EXPECT(aegis_create(&bad, 9999) == AEGIS_ERR_NO_DEVICE && bad == NULL, "create(9999)");
  EXPECT(aegis_destroy(NULL) == AEGIS_OK, "destroy(NULL)");
  EXPECT(aegis_gemm_f16(ctx, NULL, NULL, NULL, 1, 1, 1) == AEGIS_ERR_INVALID_ARGUMENT,
         "gemm with NULL operands");
  void* p = NULL;
  EXPECT(aegis_device_alloc(ctx, 64, &p) == AEGIS_OK && p, "alloc");
  EXPECT(aegis_gemm_f16(ctx, p, p, p, 0, 4, 4) == AEGIS_OK, "gemm M=0 is a no-op (ABI 1.2)");
  EXPECT(aegis_gemm_f16(ctx, NULL, NULL, NULL, 4, 0, 4) == AEGIS_OK, "gemm N=0 with null operands is a no-op");
  EXPECT(aegis_gemm_f16(ctx, p, p, p, -1, 4, 4) == AEGIS_ERR_INVALID_ARGUMENT, "gemm M<0");
  EXPECT(aegis_gemm_f16_ex(ctx, 7, AEGIS_ROW_MAJOR, 4, 4, 4, 1.f, p, 4, p, 4, 0.f, p, 4) ==
             AEGIS_ERR_INVALID_ARGUMENT, "bad layout");
  EXPECT(aegis_gemm_f16_ex(ctx, AEGIS_ROW_MAJOR, AEGIS_ROW_MAJOR, 4, 4, 8, 1.f, p, 4, p, 4, 0.f, p, 4) ==
             AEGIS_ERR_INVALID_ARGUMENT, "lda < K");
  EXPECT(aegis_gemm_f16(NULL, p, p, p, 4, 4, 4) == AEGIS_ERR_INVALID_ARGUMENT, "gemm ctx=NULL");
  EXPECT(aegis_device_free(ctx, p) == AEGIS_OK, "free");
  EXPECT(strcmp(aegis_status_string(AEGIS_ERR_CUDA), "unknown status") != 0, "status string");
  EXPECT(aegis_abi_version() == AEGIS_ABI_VERSION_MAJOR * 1000 + AEGIS_ABI_VERSION_MINOR,
         "abi version");
  EXPECT(aegis_gemm_f16(ctx, NULL, NULL, NULL, 1, 1, 1) == AEGIS_ERR_INVALID_ARGUMENT &&
             strlen(aegis_last_error_string()) > 0,
         "invalid argument sets aegis_last_error_string");
}

/* A shape whose tuned kernel cannot be launched: M = 10^7 rows with a tuned
 * N, K (896 x 896) selects a tuned kernel (nearest-M winner) whose grid needs
 * > 65535 blocks in y, so the launch is rejected (CUDA 13: "invalid
 * argument") before touching memory. The process must survive with
 * AEGIS_ERR_LAUNCH_FAILED, and the context must stay usable. */
static void test_launch_failure(aegis_context* ctx) {
  const int M = 10000000, N = 896, K = 896;
  char choice[64] = "";
  EXPECT(aegis_gemm_choice(ctx, M, N, K, choice, sizeof choice, NULL) == AEGIS_OK, "choice");
  if (strcmp(choice, "cublas") == 0) {
    /* No tuned kernel for 896x896 on this arch (no winners header yet): cuBLAS
     * would really run a 10^7-row GEMM on these tiny buffers. Not testable. */
    printf("skip launch-failure test: %dx%dx%d would use cuBLAS on this arch\n", M, N, K);
    return;
  }
  void* p = NULL;
  EXPECT(aegis_device_alloc(ctx, 256, &p) == AEGIS_OK, "alloc");
  const int s = aegis_gemm_f16(ctx, p, p, p, M, N, K);
  const char* detail = aegis_last_error_string();
  printf("%-4s forced launch failure via %s: status %d (%s), detail \"%s\"\n",
         s == AEGIS_ERR_LAUNCH_FAILED ? "ok" : "FAIL", choice, s, aegis_status_string(s), detail);
  EXPECT(s == AEGIS_ERR_LAUNCH_FAILED, "expected AEGIS_ERR_LAUNCH_FAILED, got %d", s);
  EXPECT(strstr(detail, choice) != NULL, "detail names the kernel: %s", detail);
  /* CUDA 13 reports the y-grid overflow as "invalid argument"; require some
   * CUDA error text after the kernel name rather than one exact wording. */
  EXPECT(strstr(detail, ": ") != NULL && strlen(strstr(detail, ": ")) > 2,
         "detail has the CUDA error text: %s", detail);
  EXPECT(aegis_synchronize(ctx) == AEGIS_OK, "context still usable after the failed launch");
  EXPECT(aegis_device_free(ctx, p) == AEGIS_OK, "free");
}

static void test_manifest(void) {
  size_t needed = 0;
  EXPECT(aegis_device_manifest(0, NULL, 0, &needed) == AEGIS_OK && needed > 1, "size query");
  char small[8];
  EXPECT(aegis_device_manifest(0, small, sizeof small, &needed) == AEGIS_ERR_BUFFER_TOO_SMALL,
         "small buffer");
  char* buf = malloc(needed);
  size_t again = 0;
  EXPECT(buf && aegis_device_manifest(0, buf, needed, &again) == AEGIS_OK, "fill");
  EXPECT(again == needed && strlen(buf) + 1 == needed, "size stable across calls");
  EXPECT(strstr(buf, "\"measured_d2d_bw_gbps\"") && strstr(buf, "\"sm_arch\""), "manifest keys");
  EXPECT(aegis_device_manifest(-1, buf, needed, NULL) == AEGIS_ERR_NO_DEVICE, "bad device");
  free(buf);
}

static void test_gemm(aegis_context* ctx, int M, int N, int K) {
  const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
  uint16_t* hA = malloc(nA * 2);
  uint16_t* hB = malloc(nB * 2);
  uint16_t* hC = malloc(nC * 2);
  srand(42);
  for (size_t i = 0; i < nA; ++i) hA[i] = f2h((float)rand() / RAND_MAX * 2.f - 1.f);
  for (size_t i = 0; i < nB; ++i) hB[i] = f2h((float)rand() / RAND_MAX * 2.f - 1.f);

  void *dA = NULL, *dB = NULL, *dC = NULL;
  int s = aegis_device_alloc(ctx, nA * 2, &dA);
  if (s == AEGIS_OK) s = aegis_device_alloc(ctx, nB * 2, &dB);
  if (s == AEGIS_OK) s = aegis_device_alloc(ctx, nC * 2, &dC);
  if (s == AEGIS_OK) s = aegis_copy_to_device(ctx, dA, hA, nA * 2);
  if (s == AEGIS_OK) s = aegis_copy_to_device(ctx, dB, hB, nB * 2);
  if (s == AEGIS_OK) s = aegis_gemm_f16(ctx, dA, dB, dC, M, N, K);
  if (s == AEGIS_OK) s = aegis_copy_to_host(ctx, hC, dC, nC * 2);
  EXPECT(s == AEGIS_OK, "gemm %dx%dx%d: %s (%s)", M, N, K, aegis_status_string(s),
         aegis_last_error(ctx));

  double max_err = 0, max_ref = 0;
  for (int i = 0; s == AEGIS_OK && i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += (double)h2f(hA[(size_t)i * K + k]) * h2f(hB[(size_t)k * N + j]);
      const double got = h2f(hC[(size_t)i * N + j]);
      max_err = fmax(max_err, isfinite(got) ? fabs(got - ref) : INFINITY);
      max_ref = fmax(max_ref, fabs(ref));
    }
  const double rel = max_err / fmax(max_ref, 1e-12);
  char choice[64] = "";
  aegis_gemm_choice(ctx, M, N, K, choice, sizeof choice, NULL);
  printf("%-4s gemm %5dx%5dx%5d  %-30s rel_err=%.3g\n", rel <= 4e-3 ? "ok" : "FAIL", M, N, K,
         choice, rel);
  EXPECT(rel <= 4e-3, "gemm %dx%dx%d rel_err %.3g", M, N, K, rel);

  aegis_device_free(ctx, dA);
  aegis_device_free(ctx, dB);
  aegis_device_free(ctx, dC);
  free(hA);
  free(hB);
  free(hC);
}


/* aegis_gemm_f16_strided_batched / _ex against a double-precision CPU
 * reference: one layout pair, padded leading dimensions, alpha/beta with C
 * prefilled, `batch` batches. Tolerance: the Production-bar bound
 * c*sqrt(K)*eps_fp16*max|A|*max|B|*|alpha| + eps_fp16*|beta|*max|C0|, c = 4. */
static void test_gemm_general(aegis_context* ctx, int la, int lb, int M, int N, int K, float alpha,
                              float beta, int batch) {
  const int64_t lda = (la == AEGIS_ROW_MAJOR ? K : M) + 3, ldb = (lb == AEGIS_ROW_MAJOR ? N : K) + 5;
  const int64_t ldc = N + 2;
  const int64_t sa = lda * (la == AEGIS_ROW_MAJOR ? M : K) + 1, sb = ldb * (lb == AEGIS_ROW_MAJOR ? K : N) + 7;
  const int64_t sc = ldc * M;
  const size_t nA = (size_t)(sa * batch), nB = (size_t)(sb * batch), nC = (size_t)(sc * batch);
  uint16_t* hA = malloc(nA * 2);
  uint16_t* hB = malloc(nB * 2);
  uint16_t* hC = malloc(nC * 2);
  uint16_t* hC0 = malloc(nC * 2);
  srand(7);
  for (size_t i = 0; i < nA; ++i) hA[i] = f2h((float)rand() / RAND_MAX * 2.f - 1.f);
  for (size_t i = 0; i < nB; ++i) hB[i] = f2h((float)rand() / RAND_MAX * 2.f - 1.f);
  for (size_t i = 0; i < nC; ++i) hC0[i] = f2h((float)rand() / RAND_MAX * 2.f - 1.f);
  void *dA = NULL, *dB = NULL, *dC = NULL;
  int s = aegis_device_alloc(ctx, nA * 2, &dA);
  if (s == AEGIS_OK) s = aegis_device_alloc(ctx, nB * 2, &dB);
  if (s == AEGIS_OK) s = aegis_device_alloc(ctx, nC * 2, &dC);
  if (s == AEGIS_OK) s = aegis_copy_to_device(ctx, dA, hA, nA * 2);
  if (s == AEGIS_OK) s = aegis_copy_to_device(ctx, dB, hB, nB * 2);
  if (s == AEGIS_OK) s = aegis_copy_to_device(ctx, dC, hC0, nC * 2);
  if (s == AEGIS_OK)
    s = batch == 1 ? aegis_gemm_f16_ex(ctx, la, lb, M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc)
                   : aegis_gemm_f16_strided_batched(ctx, la, lb, M, N, K, alpha, dA, lda, sa, dB, ldb, sb,
                                                    beta, dC, ldc, sc, batch);
  if (s == AEGIS_OK) s = aegis_copy_to_host(ctx, hC, dC, nC * 2);
  EXPECT(s == AEGIS_OK, "gemm_general: %s (%s)", aegis_status_string(s), aegis_last_error_string());

  const double eps = 9.765625e-4; /* 2^-10 */
  double worst = 0;
  for (int b = 0; s == AEGIS_OK && b < batch; ++b)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        double ref = 0;
        for (int k = 0; k < K; ++k) {
          const double a = h2f(hA[b * sa + (la == AEGIS_ROW_MAJOR ? i * lda + k : i + k * lda)]);
          const double x = h2f(hB[b * sb + (lb == AEGIS_ROW_MAJOR ? k * ldb + j : k + j * ldb)]);
          ref += a * x;
        }
        const size_t c = (size_t)(b * sc + i * ldc + j);
        ref = alpha * ref + beta * h2f(hC0[c]);
        const double tol = 4.0 * sqrt((double)K) * eps * fabs(alpha) + eps * fabs(beta) + 1e-7;
        const double err = fabs(h2f(hC[c]) - ref);
        if (err / tol > worst) worst = err / tol;
      }
  /* C padding (columns N..ldc-1) must be untouched. */
  int pad_ok = 1;
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < M; ++i)
      for (int64_t j = N; j < ldc; ++j) pad_ok &= hC[b * sc + i * ldc + j] == hC0[b * sc + i * ldc + j];
  printf("%-4s gemm_general %s%s %3dx%3dx%3d batch %d alpha %.2f beta %.2f  err/tol=%.3f\n",
         worst <= 1.0 && pad_ok ? "ok" : "FAIL", la == AEGIS_ROW_MAJOR ? "R" : "C",
         lb == AEGIS_ROW_MAJOR ? "R" : "C", M, N, K, batch, alpha, beta, worst);
  EXPECT(worst <= 1.0, "gemm_general err/tol %.3f", worst);
  EXPECT(pad_ok, "gemm_general wrote C padding");
  aegis_device_free(ctx, dA);
  aegis_device_free(ctx, dB);
  aegis_device_free(ctx, dC);
  free(hA);
  free(hB);
  free(hC);
  free(hC0);
}
int main(void) {
  aegis_context* ctx = NULL;
  const int s = aegis_create(&ctx, 0);
  if (s != AEGIS_OK) {
    fprintf(stderr, "aegis_create: %s\n", aegis_status_string(s));
    return 1;
  }
  test_errors(ctx);
  test_manifest();
  test_launch_failure(ctx);
  test_gemm(ctx, 127, 129, 65);  /* unaligned; also proves the process survived */
  test_gemm(ctx, 16, 896, 896);  /* tuned decode shape (if winners are compiled in) */
  test_gemm(ctx, 256, 256, 256);
  for (int la = 0; la < 2; ++la)
    for (int lb = 0; lb < 2; ++lb) test_gemm_general(ctx, la, lb, 37, 29, 45, 0.5f, -1.f, 1);
  test_gemm_general(ctx, AEGIS_COL_MAJOR, AEGIS_ROW_MAJOR, 16, 896, 896, -1.5f, 0.f, 1); /* tuned shape */
  test_gemm_general(ctx, AEGIS_ROW_MAJOR, AEGIS_COL_MAJOR, 33, 20, 70, 1.f, 1.f, 3);
  EXPECT(aegis_destroy(ctx) == AEGIS_OK, "destroy");
  if (failures) fprintf(stderr, "%d failure(s)\n", failures);
  else printf("capi_test: all checks passed\n");
  return failures ? 1 : 0;
}
