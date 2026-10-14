// Tier 0: naive GEMM. Correctness anchor and the floor of the ladder.
// Each thread reads a full row of A and a full column of B from global memory.
// Expect it to be memory-bound and far below cuBLAS; that's the point.
// General GEMM (GemmParams): any layout, ld, alpha/beta, strided batch
// (blockIdx.z), any alignment (scalar loads only).
#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "gemm_params.cuh"

namespace aegis::gemm {
namespace {

template <Layout LA, Layout LB>
__global__ void gemm_naive_kernel(KParams p) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p.M || col >= p.N) return;
  const long long z = blockIdx.z;
  const __half* A = p.A + z * p.sa;
  const __half* B = p.B + z * p.sb;
  __half* C = p.C + z * p.sc + (long long)row * p.ldc + col;

  float acc = 0.f;
  for (int k = 0; k < p.K; ++k)
    acc += __half2float(A[elem<LA>(row, k, p.lda)]) * __half2float(B[elem<LB>(k, col, p.ldb)]);
  float out = p.alpha * acc;
  if (p.beta != 0.f) out += p.beta * __half2float(*C);  // beta == 0: C is not read
  *C = __float2half(out);
}

template <Layout LA, Layout LB>
cudaError_t launch_layout(const GemmParams& gp, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid(ceil_div(gp.N, block.x), ceil_div(gp.M, block.y), gp.batch);
  gemm_naive_kernel<LA, LB><<<grid, block, 0, stream>>>(to_kparams(gp));
  return cudaGetLastError();  // launch-configuration errors; never exits
}

}  // namespace

cudaError_t run_naive(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  if (p.layout_a == R && p.layout_b == R) return launch_layout<R, R>(p, stream);
  if (p.layout_a == R) return launch_layout<R, Cm>(p, stream);
  if (p.layout_b == R) return launch_layout<Cm, R>(p, stream);
  return launch_layout<Cm, Cm>(p, stream);
}

cudaError_t launch_naive(const __half* A, const __half* B, __half* C, int M, int N,
                         int K, cudaStream_t stream) {
  return run_naive(row_major_params(A, B, C, M, N, K), stream);
}

cudaError_t attrs_naive(cudaFuncAttributes* attrs) {
  return cudaFuncGetAttributes(attrs, gemm_naive_kernel<Layout::RowMajor, Layout::RowMajor>);
}

}  // namespace aegis::gemm
