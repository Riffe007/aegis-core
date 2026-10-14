// GEMM reference and tolerance (see reference.h).
//
// The constant c (kToleranceC = 2.0). Error sources for FP16-in, FP32-
// accumulate, FP16-out GEMM: the FP32 accumulation of K products (relative
// error ~ sqrt(K) * 2^-24 for random-sign data, negligible here) and the final
// rounding to FP16 (at most 0.5 ulp = eps_fp16 / 2 * |C|), plus, with beta,
// the FP16 rounding of C0. For zero-mean inputs |C| grows like sqrt(K) *
// max|A| * max|B|, which is what the bound scales with.
// Measured with aegis_fuzz --calibrate (2026-10-15, seeds 7 and 11, 13,000
// cases, 9,070 of them against the FP64 reference): the largest
// err / (sqrt(K) * eps * max|A| * max|B| * |alpha| + eps * |beta| * max|C0|) was
//   naive 0.66, regblock 0.71, mma 0.78, mma_pipe 0.77, aegis::matmul 0.98
//   (cuBLAS on untuned shapes), cuBLASLt 1.12.
// c = 2 leaves about 2x over our worst observed error and 1.8x over
// cuBLASLt's. Against a cuBLASLt reference the allowance is 2 * tol, which
// covers both results' errors (0.98 + 1.12 < 4). Sensitivity: a missing
// k-term changes an element by |a * b|; at K = 4096 the bound is
// 0.125 * max|A| * max|B|, which a uniform product exceeds with probability
// ~0.6 per element, so a dropped k-slice is caught on almost every element.
#include "reference.h"

#include "gpu_budget.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "aegis/common.cuh"

#define AEGIS_LT_CHECK(expr)                                                         \
  do {                                                                               \
    cublasStatus_t s_ = (expr);                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                               \
      std::fprintf(stderr, "cuBLASLt error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                                \
  } while (0)

namespace aegis::bench {

using gemm::GemmParams;
using gemm::Layout;

namespace {
size_t span(long long rows_minus1_times_ld_plus_cols, long long stride, int batch) {
  return size_t((batch - 1) * stride + rows_minus1_times_ld_plus_cols);
}
}  // namespace

size_t span_a(const GemmParams& p) {
  if (gemm::is_empty(p)) return 0;
  const long long one = p.layout_a == Layout::RowMajor ? (long long)(p.M - 1) * p.lda + p.K
                                                       : (long long)(p.K - 1) * p.lda + p.M;
  return span(one, p.stride_a, p.batch);
}
size_t span_b(const GemmParams& p) {
  if (gemm::is_empty(p)) return 0;
  const long long one = p.layout_b == Layout::RowMajor ? (long long)(p.K - 1) * p.ldb + p.N
                                                       : (long long)(p.N - 1) * p.ldb + p.K;
  return span(one, p.stride_b, p.batch);
}
size_t span_c(const GemmParams& p) {
  if (gemm::is_empty(p)) return 0;
  return span((long long)(p.M - 1) * p.ldc + p.N, p.stride_c, p.batch);
}

void cpu_reference(const HostGemm& g, std::vector<double>& out) {
  const GemmParams& p = g.p;
  out.assign(size_t(p.batch) * p.M * p.N, 0.0);
  const bool ra = p.layout_a == Layout::RowMajor, rb = p.layout_b == Layout::RowMajor;
  std::vector<double> arow(size_t(p.K)), bcol(size_t(p.K));
  for (int b = 0; b < p.batch; ++b)
    for (int i = 0; i < p.M; ++i) {
      for (int k = 0; k < p.K; ++k)
        arow[k] = __half2float(g.A[size_t(b * p.stride_a + (ra ? i * p.lda + k : i + k * p.lda))]);
      for (int j = 0; j < p.N; ++j) {
        double acc = 0;
        for (int k = 0; p.alpha != 0.f && k < p.K; ++k) {  // alpha == 0: A*B is not needed
          const double x = __half2float(g.B[size_t(b * p.stride_b + (rb ? k * p.ldb + j : k + j * p.ldb))]);
          acc += arow[k] * x;
        }
        double v = p.alpha == 0.f ? 0.0 : double(p.alpha) * acc;  // BLAS: alpha == 0 ignores A*B
        if (p.beta != 0.f) v += double(p.beta) * __half2float(g.C0[size_t(b * p.stride_c + i * p.ldc + j)]);
        out[(size_t(b) * p.M + i) * p.N + j] = v;
      }
    }
  (void)bcol;
}

LtReference::LtReference() {
  AEGIS_LT_CHECK(cublasLtCreate(&lt_));
  ws_bytes_ = size_t(32) << 20;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&ws_, ws_bytes_));
}

LtReference::~LtReference() {
  aegis::budget::gpu_free(ws_);
  cublasLtDestroy(lt_);
}

// Row-major C (M x N, ldc) is column-major C^T (N x M): C^T = op(B) * op(A).
// Lt's "A" is our B: op N if B is row-major (stored as column-major N x K),
// op T if column-major (stored K x N). Lt's "B" is our A: op N if row-major
// (stored K x M), op T if column-major (stored M x K).
void LtReference::run(const GemmParams& p, cudaStream_t stream) {
  if (gemm::is_empty(p)) return;
  cublasLtMatmulDesc_t op;
  AEGIS_LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = p.layout_b == Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t tb = p.layout_a == Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  AEGIS_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
  AEGIS_LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  cublasLtMatrixLayout_t la, lb, lc;
  const bool rb = p.layout_b == Layout::RowMajor, ra = p.layout_a == Layout::RowMajor;
  AEGIS_LT_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, rb ? p.N : p.K, rb ? p.K : p.N, p.ldb));
  AEGIS_LT_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, ra ? p.K : p.M, ra ? p.M : p.K, p.lda));
  AEGIS_LT_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, p.N, p.M, p.ldc));
  if (p.batch > 1) {
    const int32_t n = p.batch;
    const int64_t sa = p.stride_b, sb = p.stride_a, sc = p.stride_c;
    for (auto [l, s] : {std::pair{la, sa}, std::pair{lb, sb}, std::pair{lc, sc}}) {
      AEGIS_LT_CHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &n, sizeof n));
      AEGIS_LT_CHECK(
          cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s, sizeof s));
    }
  }
  cublasLtMatmulPreference_t pref;
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes_,
                                                      sizeof ws_bytes_));
  // The heuristic assumes 256-byte-aligned operands unless told otherwise.
  auto align = [](const void* q) {
    uint32_t a = 256;
    while (a > 1 && (reinterpret_cast<uintptr_t>(q) % a) != 0) a /= 2;
    return a;
  };
  const uint32_t al_a = align(p.B), al_b = align(p.A), al_c = align(p.C);
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al_a, sizeof al_a));
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al_b, sizeof al_b));
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al_c, sizeof al_c));
  AEGIS_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al_c, sizeof al_c));
  cublasLtMatmulHeuristicResult_t res{};
  int found = 0;
  AEGIS_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_, op, la, lb, lc, lc, pref, 1, &res, &found));
  if (found == 0) {
    std::fprintf(stderr, "LtReference: no cuBLASLt algorithm for %dx%dx%d\n", p.M, p.N, p.K);
    std::exit(EXIT_FAILURE);
  }
  AEGIS_LT_CHECK(cublasLtMatmul(lt_, op, &p.alpha, p.B, la, p.A, lb, &p.beta, p.C, lc, p.C, lc, &res.algo, ws_,
                                res.workspaceSize, stream));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatmulDescDestroy(op);
}

namespace {
double max_abs_finite(const std::vector<__half>& v) {
  double m = 0;
  for (const __half& h : v) {
    const double x = std::fabs(double(__half2float(h)));
    if (std::isfinite(x) && x > m) m = x;
  }
  return m;
}
}  // namespace

Tolerance tolerance(const HostGemm& g, double c) {
  Tolerance t;
  t.max_a = max_abs_finite(g.A);
  t.max_b = max_abs_finite(g.B);
  t.max_c0 = g.p.beta != 0.f ? max_abs_finite(g.C0) : 0.0;
  t.tol = c * std::sqrt(double(g.p.K)) * kEpsFp16 * t.max_a * t.max_b * std::fabs(double(g.p.alpha)) +
          kEpsFp16 * std::fabs(double(g.p.beta)) * t.max_c0;
  return t;
}

Comparison compare(const HostGemm& g, const std::vector<__half>& out, const std::vector<double>& ref,
                   double allowed) {
  const GemmParams& p = g.p;
  constexpr double kOverflow = 65520.0;  // round-to-nearest FP16 overflow threshold
  Comparison c;
  auto fail_at = [&](int b, int i, int j, double got, double want) {
    if (!c.first_failure.empty()) return;
    char buf[160];
    std::snprintf(buf, sizeof buf, "batch %d row %d col %d: got %.6g, ref %.6g (allowed %.3g)", b, i, j, got,
                  want, allowed);
    c.first_failure = buf;
  };
  for (int b = 0; b < p.batch; ++b)
    for (int i = 0; i < p.M; ++i) {
      for (int j = 0; j < p.N; ++j) {
        const size_t oi = size_t(b * p.stride_c + i * p.ldc + j);
        const double o = __half2float(out[oi]);
        const double r = ref[(size_t(b) * p.M + i) * p.N + j];
        ++c.checked;
        if (std::isnan(r)) {
          if (!std::isnan(o)) { ++c.nan_mismatch; fail_at(b, i, j, o, r); }
          continue;
        }
        if (std::isinf(r) || std::fabs(r) >= kOverflow) {
          const bool same_inf = std::isinf(o) && (o > 0) == (r > 0);
          const bool near_max = std::isfinite(o) && std::fabs(std::fabs(r) - 65504.0) <= allowed;
          if (!same_inf && !(std::isfinite(r) && near_max)) { ++c.inf_mismatch; fail_at(b, i, j, o, r); }
          continue;
        }
        if (!std::isfinite(o)) {
          // A finite reference just under the overflow threshold may round to Inf.
          if (!(std::isinf(o) && std::fabs(r) + allowed >= kOverflow)) { ++c.inf_mismatch; fail_at(b, i, j, o, r); }
          continue;
        }
        const double err = std::fabs(o - r);
        if (err > c.max_err) c.max_err = err;
        const double ratio = allowed > 0 ? err / allowed : (err > 0 ? INFINITY : 0.0);
        if (ratio > c.max_ratio) c.max_ratio = ratio;
        if (err > allowed) { ++c.out_of_tol; fail_at(b, i, j, o, r); }
      }
      // Padding columns of C must be untouched.
      for (long long j = p.N; j < p.ldc; ++j) {
        const size_t oi = size_t(b * p.stride_c + i * p.ldc + j);
        if (oi < out.size() && __half_as_ushort(out[oi]) != __half_as_ushort(g.C0[oi])) {
          ++c.out_of_tol;
          fail_at(b, i, int(j), __half2float(out[oi]), __half2float(g.C0[oi]));
        }
      }
    }
  c.pass = c.nan_mismatch == 0 && c.inf_mismatch == 0 && c.out_of_tol == 0;
  return c;
}

}  // namespace aegis::bench
