// aegis::matmul end to end: for shapes that hit each selection path (exact
// winner, nearest-M fallback, cuBLAS fallback, unaligned), compare against
// cuBLAS with the bench tolerance and print which impl was chosen.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cublas_v2.h>

#include "aegis/common.cuh"
#include "aegis/matmul.h"

namespace {

constexpr double kRelTol = 4e-3;  // same as bench_gemm / CLAUDE.md rule 2

bool check(int M, int N, int K) {
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  std::vector<__half> hA(nA), hB(nB), hC(nC), hR(nC);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& x : hA) x = __float2half(d(rng));
  for (auto& x : hB) x = __float2half(d(rng));

  __half *A, *B, *C, *R;
  AEGIS_CUDA_CHECK(cudaMalloc(&A, nA * 2));
  AEGIS_CUDA_CHECK(cudaMalloc(&B, nB * 2));
  AEGIS_CUDA_CHECK(cudaMalloc(&C, nC * 2));
  AEGIS_CUDA_CHECK(cudaMalloc(&R, nC * 2));
  AEGIS_CUDA_CHECK(cudaMemcpy(A, hA.data(), nA * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, hB.data(), nB * 2, cudaMemcpyHostToDevice));

  AEGIS_CUDA_CHECK(aegis::matmul(A, B, C, M, N, K));
  cublasHandle_t h;
  cublasCreate(&h);
  const float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
               CUDA_R_16F, K, &beta, R, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  AEGIS_CUDA_CHECK(cudaMemcpy(hC.data(), C, nC * 2, cudaMemcpyDeviceToHost));
  AEGIS_CUDA_CHECK(cudaMemcpy(hR.data(), R, nC * 2, cudaMemcpyDeviceToHost));

  double err = 0, ref = 0;
  for (size_t i = 0; i < nC; ++i) {
    const double c = __half2float(hC[i]), r = __half2float(hR[i]);
    err = std::isfinite(c) ? std::max(err, std::abs(c - r)) : INFINITY;
    ref = std::max(ref, std::abs(r));
  }
  const double rel = err / std::max(ref, 1e-12);
  const bool ok = rel <= kRelTol;
  std::printf("%-5s %5dx%5dx%5d  %-30s rel_err=%.3g\n", ok ? "ok" : "FAIL", M, N, K,
              aegis::matmul_choice(M, N, K), rel);
  cublasDestroy(h);
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check(1024, 1024, 1024);  // tuned exact shape
  ok &= check(512, 896, 4864);    // tuned exact shape
  ok &= check(8, 896, 896);       // nearest M among tuned (1, 16, 512)
  ok &= check(300, 4864, 896);    // nearest M among tuned
  ok &= check(100, 200, 300);     // no tuned N,K: cuBLAS
  ok &= check(127, 129, 65);      // unaligned, cuBLAS
  return ok ? 0 : 1;
}
