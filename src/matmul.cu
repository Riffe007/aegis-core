// aegis::matmul dispatch. See include/aegis/matmul.h for the selection rules.
#include <cmath>
#include <cstdio>
#include <span>
#include <string>

#include <cublas_v2.h>

#include "aegis/gemm.h"
#include "aegis/matmul.h"

// One generated table per arch; a missing file just means no tuned winners.
#if __has_include("aegis/winners/sm_121.h")
#include "aegis/winners/sm_121.h"
#endif
#if __has_include("aegis/winners/sm_87.h")
#include "aegis/winners/sm_87.h"
#endif

namespace aegis {
namespace {

std::span<const detail::Winner> winners_for(int arch) {
  switch (arch) {
#ifdef AEGIS_WINNERS_SM121
    case 121: return kWinners_sm121;
#endif
#ifdef AEGIS_WINNERS_SM87
    case 87: return kWinners_sm87;
#endif
    default: return {};
  }
}

thread_local std::string last_error;

cudaError_t fail(cudaError_t e, const std::string& what) {
  last_error = what + ": " + cudaGetErrorString(e);
  return e;
}

// 0 if the device cannot be queried (then only cuBLAS is used).
int current_arch() {
  int dev = 0, major = 0, minor = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return major * 10 + minor;
}

const gemm::KernelEntry* select(int M, int N, int K) {
  const detail::Winner* best = nullptr;
  double best_dist = INFINITY;
  for (const detail::Winner& w : winners_for(current_arch())) {
    if (w.N != N || w.K != K) continue;
    if (w.M == M) { best = &w; break; }
    const double d = std::abs(std::log2(double(w.M)) - std::log2(double(M)));
    if (d < best_dist || (d == best_dist && w.M > best->M)) { best = &w; best_dist = d; }
  }
  return best ? gemm::find(best->impl) : nullptr;
}

cudaError_t cublas_fail(const char* what, cublasStatus_t s) {
  last_error = std::string(what) + " failed: cuBLAS status " + std::to_string(int(s));
  return cudaErrorUnknown;
}

// General cuBLAS call for GemmParams. Row-major C (M x N, ldc) is column-major
// C^T (N x M, ldc) = op(B) * op(A): a row-major operand read as column-major is
// already transposed (op N); a column-major one needs op T.
cudaError_t cublas_gemm_general(const gemm::GemmParams& p, cudaStream_t stream) {
  thread_local cublasHandle_t h = nullptr;
  if (!h) {
    const cublasStatus_t s = cublasCreate(&h);
    if (s != CUBLAS_STATUS_SUCCESS) {
      h = nullptr;
      return cublas_fail("cublasCreate", s);
    }
  }
  cublasStatus_t s = cublasSetStream(h, stream);
  if (s != CUBLAS_STATUS_SUCCESS) return cublas_fail("cublasSetStream", s);
  constexpr long long kIntMax = 2147483647;
  if (p.lda > kIntMax || p.ldb > kIntMax || p.ldc > kIntMax) {
    last_error = "cuBLAS fallback: leading dimension exceeds int";
    return cudaErrorInvalidValue;
  }
  const cublasOperation_t op_b = p.layout_b == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t op_a = p.layout_a == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  s = cublasGemmStridedBatchedEx(h, op_b, op_a, p.N, p.M, p.K, &p.alpha, p.B, CUDA_R_16F, int(p.ldb),
                                 p.stride_b, p.A, CUDA_R_16F, int(p.lda), p.stride_a, &p.beta, p.C,
                                 CUDA_R_16F, int(p.ldc), p.stride_c, p.batch, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT);
  return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cublas_fail("cublasGemmStridedBatchedEx", s);
}

}  // namespace

cudaError_t matmul(const gemm::GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = gemm::validate(p); e != cudaSuccess)
    return fail(e, "aegis::matmul: invalid GEMM parameters");
  if (gemm::is_empty(p)) return cudaSuccess;
  if (const gemm::KernelEntry* e = select(p.M, p.N, p.K); e && e->run) {
    const cudaError_t err = e->run(p, stream);
    return err == cudaSuccess ? err : fail(err, std::string("launch ") + e->name);
  }
  return cublas_gemm_general(p, stream);
}

cudaError_t matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  return matmul(gemm::row_major_params(A, B, C, M, N, K), stream);
}

const char* matmul_last_error() { return last_error.c_str(); }

const char* matmul_choice(int M, int N, int K) {
  const gemm::KernelEntry* e = select(M, N, K);
  return e ? e->name : "cublas";
}

}  // namespace aegis
