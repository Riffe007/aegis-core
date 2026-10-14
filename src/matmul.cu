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

// Same transposed-operand call as bench_gemm's reference. One handle per host
// thread, created on first use and intentionally never destroyed (destroying
// it at thread exit can race CUDA context teardown). A failed create is
// retried on the next call.
cudaError_t cublas_gemm(const __half* A, const __half* B, __half* C, int M, int N, int K,
                        cudaStream_t stream) {
  thread_local cublasHandle_t h = nullptr;
  if (!h) {
    const cublasStatus_t s = cublasCreate(&h);
    if (s != CUBLAS_STATUS_SUCCESS) {
      h = nullptr;
      return cublas_fail("cublasCreate", s);
    }
  }
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t s = cublasSetStream(h, stream);
  if (s != CUBLAS_STATUS_SUCCESS) return cublas_fail("cublasSetStream", s);
  s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
                   CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cublas_fail("cublasGemmEx", s);
}

}  // namespace

cudaError_t matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  if (const gemm::KernelEntry* e = select(M, N, K)) {
    const cudaError_t err = e->launch(A, B, C, M, N, K, stream);
    return err == cudaSuccess ? err : fail(err, std::string("launch ") + e->name);
  }
  return cublas_gemm(A, B, C, M, N, K, stream);
}

const char* matmul_last_error() { return last_error.c_str(); }

const char* matmul_choice(int M, int N, int K) {
  const gemm::KernelEntry* e = select(M, N, K);
  return e ? e->name : "cublas";
}

}  // namespace aegis
