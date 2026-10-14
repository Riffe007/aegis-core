// Tier 0: naive GEMM. Correctness anchor and the floor of the ladder.
// Each thread reads a full row of A and a full column of B from global memory.
// Expect it to be memory-bound and far below cuBLAS; that's the point.
#include "aegis/common.cuh"
#include "aegis/gemm.h"

namespace aegis::gemm {
namespace {

__global__ void gemm_naive_kernel(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  __half* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc += __half2float(A[static_cast<size_t>(row) * K + k]) *
           __half2float(B[static_cast<size_t>(k) * N + col]);
  }
  C[static_cast<size_t>(row) * N + col] = __float2half(acc);
}

}  // namespace

void launch_naive(const __half* A, const __half* B, __half* C, int M, int N,
                  int K, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
  gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  AEGIS_CUDA_CHECK(cudaGetLastError());
}

cudaFuncAttributes attrs_naive() {
  cudaFuncAttributes a{};
  AEGIS_CUDA_CHECK(cudaFuncGetAttributes(&a, gemm_naive_kernel));
  return a;
}

}  // namespace aegis::gemm
