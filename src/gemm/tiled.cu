// Tier 1: shared-memory tiled GEMM, one output element per thread.
// Each block stages a TILExTILE slab of A and B in shared memory per K-step,
// cutting global loads by a factor of TILE versus naive. Still CUDA-core FP32
// math and one output per thread, so it stays far from the tensor-core roof.
// Tier 2 (register blocking) is the next step up.
#include "aegis/common.cuh"
#include "aegis/gemm.h"

namespace aegis::gemm {
namespace {

constexpr int TILE = 32;

__global__ void gemm_tiled_kernel(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  __half* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // Coalesced loads: consecutive tx read consecutive addresses in both A and B.
    As[ty][tx] = (row < M && k0 + tx < K)
                     ? __half2float(A[static_cast<size_t>(row) * K + k0 + tx])
                     : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N)
                     ? __half2float(B[static_cast<size_t>(k0 + ty) * N + col])
                     : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }

  if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = __float2half(acc);
}

}  // namespace

cudaError_t launch_tiled(const __half* A, const __half* B, __half* C, int M, int N,
                         int K, cudaStream_t stream) {
  const dim3 block(TILE, TILE);
  const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();  // launch-configuration errors; never exits
}

cudaError_t attrs_tiled(cudaFuncAttributes* attrs) {
  return cudaFuncGetAttributes(attrs, gemm_tiled_kernel);
}

}  // namespace aegis::gemm
