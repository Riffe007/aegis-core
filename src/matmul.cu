// aegis::matmul dispatch. See include/aegis/matmul.h for the selection rules.
#include <cmath>
#include <span>

#include <cublas_v2.h>

#include "aegis/common.cuh"
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

int current_arch() {
  int dev = 0, major = 0, minor = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  AEGIS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  AEGIS_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
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

// Same transposed-operand call as bench_gemm's reference. One handle per host
// thread, created on first use and intentionally never destroyed (destroying
// it at thread exit can race CUDA context teardown).
void cublas_gemm(const __half* A, const __half* B, __half* C, int M, int N, int K,
                 cudaStream_t stream) {
  thread_local cublasHandle_t h = [] {
    cublasHandle_t x;
    if (cublasCreate(&x) != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "aegis::matmul: cublasCreate failed\n");
      std::exit(EXIT_FAILURE);
    }
    return x;
  }();
  const float alpha = 1.f, beta = 0.f;
  cublasSetStream(h, stream);
  const cublasStatus_t s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                                        CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                                        CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "aegis::matmul: cublasGemmEx failed (%d)\n", int(s));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

void matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
            cudaStream_t stream) {
  if (const gemm::KernelEntry* e = select(M, N, K)) e->launch(A, B, C, M, N, K, stream);
  else cublas_gemm(A, B, C, M, N, K, stream);
}

const char* matmul_choice(int M, int N, int K) {
  const gemm::KernelEntry* e = select(M, N, K);
  return e ? e->name : "cublas";
}

}  // namespace aegis
