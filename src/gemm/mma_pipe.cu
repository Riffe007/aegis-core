// Tier 4: tier 3 plus a STAGES-deep cp.async pipeline.
//
// Versus tier 3: global->smem copies are cp.async.cg (16 B, L2-only, no
// register staging), issued STAGES-1 K-tiles ahead into a ring of smem
// buffers. Tile kt+STAGES-1 is in flight while tile kt is on the tensor cores,
// which targets the exposed per-tile global load latency of tier 3 (its loop is
// LDG -> STS -> BAR -> LDSM -> HMMA in series).
//
// Per iteration: wait until tile kt has landed (wait_group STAGES-2), barrier,
// prefetch tile kt+STAGES-1 into the buffer computed in iteration kt-1 (safe:
// every warp is past the barrier, so done with it), then compute tile kt. One
// commit_group per iteration even when there is nothing left to prefetch, so
// the group count stays uniform for wait_group.
//
// Smem layout, padding and compute are tier 3's (mma_common.cuh). cp.async
// needs 16 B-aligned sources, so the pipeline runs when K and N are multiples
// of 8; other shapes take tier 3's synchronous loop with the scalar loader.
// PTX ISA citations for every instruction used: see mma_common.cuh.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "mma_common.cuh"

namespace aegis::gemm {
namespace {

using namespace mma_detail;

template <class T, int STAGES>
__global__ void __launch_bounds__(T::THREADS)
gemm_mma_pipe_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                     __half* __restrict__ C, int M, int N, int K, bool vec) {
  static_assert(STAGES >= 2, "a pipeline needs at least two buffers");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  auto As = [&](int s) { return smem + s * T::STAGE_ELEMS; };
  auto Bs = [&](int s) { return smem + s * T::STAGE_ELEMS + T::A_ELEMS; };

  const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
  float acc[T::MI][T::NI][4] = {};

  if (!vec) {  // unaligned shapes: tier 3's synchronous loop in buffer 0
    for (int k0 = 0; k0 < K; k0 += T::BK) {
      load_tile_sync<T>(As(0), Bs(0), A, B, M, N, K, m0, n0, k0, false);
      __syncthreads();
      compute_tile<T>(As(0), Bs(0), acc);
      __syncthreads();
    }
    store_tile<T>(C, M, N, m0, n0, acc);
    return;
  }

  const int KT = ceil_div(K, T::BK);
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_tile_async<T>(As(s), Bs(s), A, B, M, N, K, m0, n0, s * T::BK);
    cp_async_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int pf = kt + STAGES - 1;
    if (pf < KT) load_tile_async<T>(As(pf % STAGES), Bs(pf % STAGES), A, B, M, N, K, m0, n0, pf * T::BK);
    cp_async_commit();
    compute_tile<T>(As(kt % STAGES), Bs(kt % STAGES), acc);
  }
  cp_async_wait<0>();  // no copy may outlive the block (all are empty by now)
  store_tile<T>(C, M, N, m0, n0, acc);
}

template <class T, int STAGES>
void launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
            cudaStream_t stream) {
  constexpr size_t smem = T::STAGE_BYTES * STAGES;
  static const bool configured = [] {
    AEGIS_CUDA_CHECK(cudaFuncSetAttribute(gemm_mma_pipe_kernel<T, STAGES>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          int(smem)));
    return true;
  }();
  (void)configured;
  const bool vec = K % 8 == 0 && N % 8 == 0 && aligned16(A) && aligned16(B);
  const dim3 grid(ceil_div(N, T::BN), ceil_div(M, T::BM));
  gemm_mma_pipe_kernel<T, STAGES><<<grid, T::THREADS, smem, stream>>>(A, B, C, M, N, K, vec);
  AEGIS_CUDA_CHECK(cudaGetLastError());
}

template <class T, int STAGES>
cudaFuncAttributes attrs() {
  cudaFuncAttributes a{};
  AEGIS_CUDA_CHECK(cudaFuncGetAttributes(&a, gemm_mma_pipe_kernel<T, STAGES>));
  a.sharedSizeBytes += T::STAGE_BYTES * STAGES;  // report dynamic smem per block too
  return a;
}

constexpr size_t kN = std::size(MMA_PIPE_CONFIGS);
template <size_t I>
using TileOf = Tile<MMA_PIPE_CONFIGS[I].BM, MMA_PIPE_CONFIGS[I].BN, MMA_PIPE_CONFIGS[I].BK,
                    MMA_PIPE_CONFIGS[I].WARPS_M, MMA_PIPE_CONFIGS[I].WARPS_N>;

std::string name_of(const MmaPipeConfig& c) {
  return "mma_pipe_" + std::to_string(c.BM) + "x" + std::to_string(c.BN) + "x" +
         std::to_string(c.BK) + "_" + std::to_string(c.WARPS_M) + "x" +
         std::to_string(c.WARPS_N) + "_s" + std::to_string(c.STAGES);
}

std::string config_of(const MmaPipeConfig& c) {
  return "{\"BM\":" + std::to_string(c.BM) + ",\"BN\":" + std::to_string(c.BN) +
         ",\"BK\":" + std::to_string(c.BK) + ",\"WARPS_M\":" + std::to_string(c.WARPS_M) +
         ",\"WARPS_N\":" + std::to_string(c.WARPS_N) + ",\"STAGES\":" +
         std::to_string(c.STAGES) + ",\"smem\":\"pad8\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(MMA_PIPE_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(MMA_PIPE_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{
      names[I].c_str(), configs[I].c_str(),
      &launch<TileOf<I>, MMA_PIPE_CONFIGS[I].STAGES>,
      &attrs<TileOf<I>, MMA_PIPE_CONFIGS[I].STAGES>}...};
  return entries;
}

}  // namespace

std::span<const KernelEntry> mma_pipe_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
