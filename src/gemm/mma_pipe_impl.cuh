#pragma once
// Tier 4's kernel and launcher (cp.async pipeline; see mma_pipe.cu for the
// design), as templates any tier can launch on its own tile: mma_pipe.cu
// registers its configs with them, mma_tma.cu runs them when TMA cannot take a
// problem (misaligned pointers, ld or strides, devices below sm_90).
// Internal linkage (anonymous namespace): every including file instantiates
// its own copies, no kernel symbol is shared between translation units.
#include "aegis/gemm.h"
#include "mma_common.cuh"

namespace aegis::gemm {
namespace {
namespace mma_pipe_detail {

using namespace mma_detail;

template <class T, int STAGES, bool MASK>
__device__ __forceinline__ void pipe_mainloop(__half* smem, const __half* A, const __half* B,
                                              const KParams& p, int m0, int n0,
                                              float (&acc)[T::MI][T::NI][4]) {
  PlanA<T> pa;
  PlanB<T> pb;
  pa.init(A, p.lda, m0, p.M);
  pb.init(B, p.ldb, n0, p.N);
  const int KT = ceil_div(p.K, T::BK);
  const bool k_tail = p.K % T::BK != 0;
  const uint32_t s0 = smem_addr(smem);

  // Copy k-tile kt into stage s; the partial last k-tile (if any) is masked in k.
  auto load = [&](int s, int kt) {
    const uint32_t sa = s0 + uint32_t(s * T::STAGE_BYTES);
    const uint32_t sb = sa + uint32_t(T::A_ELEMS * sizeof(__half));
    const int k0 = kt * T::BK;
    if (k_tail && kt == KT - 1) {
      pa.template load_async<MASK, true>(sa, k0, p.K);
      pb.template load_async<MASK, true>(sb, k0, p.K);
    } else {
      pa.template load_async<MASK, false>(sa, k0, p.K);
      pb.template load_async<MASK, false>(sb, k0, p.K);
    }
    pa.advance();
    pb.advance();
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int pf = kt + STAGES - 1;
    if (pf < KT) load(pf % STAGES, pf);
    cp_async_commit();
    const __half* As = smem + (kt % STAGES) * T::STAGE_ELEMS;
    compute_tile<T>(As, As + T::A_ELEMS, acc);
  }
  cp_async_wait<0>();  // no copy may outlive the block (all are empty by now)
}

template <class T, int STAGES>
__global__ void __launch_bounds__(T::THREADS)
gemm_mma_pipe_kernel(KParams p, bool vec, bool pairs) {
  static_assert(STAGES >= 2, "a pipeline needs at least two buffers");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  const long long z = blockIdx.z;
  const __half* A = p.A + z * p.sa;
  const __half* B = p.B + z * p.sb;
  __half* C = p.C + z * p.sc;
  const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
  float acc[T::MI][T::NI][4] = {};

  if (!vec) {  // misaligned: synchronous, scalar, fully bounds-checked
    __half* As = smem;
    __half* Bs = smem + T::A_ELEMS;
    for (int k0 = 0; k0 < p.K; k0 += T::BK) {
      load_tile_scalar<T>(As, Bs, A, B, p, m0, n0, k0);
      __syncthreads();
      compute_tile<T>(As, Bs, acc);
      __syncthreads();
    }
    store_tile<T, true>(C, p, m0, n0, pairs, acc);
    return;
  }
  if (m0 + T::BM <= p.M && n0 + T::BN <= p.N) {  // interior CTA: unpredicated
    pipe_mainloop<T, STAGES, false>(smem, A, B, p, m0, n0, acc);
    store_tile<T, false>(C, p, m0, n0, pairs, acc);
  } else {
    pipe_mainloop<T, STAGES, true>(smem, A, B, p, m0, n0, acc);
    store_tile<T, true>(C, p, m0, n0, pairs, acc);
  }
}

template <class T, int STAGES>
cudaError_t launch_layout(const GemmParams& gp, cudaStream_t stream) {
  constexpr size_t smem = T::STAGE_BYTES * STAGES;
  // Once per instantiation; a failure is returned on every call.
  static const cudaError_t configured = cudaFuncSetAttribute(
      gemm_mma_pipe_kernel<T, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (configured != cudaSuccess) return configured;
  const dim3 grid(ceil_div(gp.N, T::BN), ceil_div(gp.M, T::BM), gp.batch);
  if (!grid_ok(grid)) return cudaErrorInvalidValue;
  gemm_mma_pipe_kernel<T, STAGES><<<grid, T::THREADS, smem, stream>>>(to_kparams(gp), vec16_ok(gp),
                                                                       c_pairs_ok(gp));
  return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  if (p.layout_a == R && p.layout_b == R) return launch_layout<Tile<BM, BN, BK, WM, WN, R, R>, STAGES>(p, stream);
  if (p.layout_a == R) return launch_layout<Tile<BM, BN, BK, WM, WN, R, Cm>, STAGES>(p, stream);
  if (p.layout_b == R) return launch_layout<Tile<BM, BN, BK, WM, WN, Cm, R>, STAGES>(p, stream);
  return launch_layout<Tile<BM, BN, BK, WM, WN, Cm, Cm>, STAGES>(p, stream);
}


}  // namespace mma_pipe_detail
}  // namespace
}  // namespace aegis::gemm
