// Tier 3: tensor-core GEMM with inline PTX. ldmatrix.x4 loads A fragments,
// ldmatrix.x4.trans loads B fragments (B is row-major [K,N]; the .col operand of
// mma wants it k-major per thread, which .trans provides), and
// mma.sync.m16n8k16.row.col.f32.f16.f16.f32 accumulates in FP32 registers.
//
// Versus tier 2: the math moves from FFMA on CUDA cores to HMMA on tensor
// cores, a far higher roof, so the bottleneck moves to feeding them from smem
// and global memory. Loads here are synchronous (global -> regs -> smem, then
// barrier), so load latency is exposed once per BK step; tier 4 hides it.
//
// Shared memory is padded, not XOR-swizzled: +8 halves per row makes the row
// stride an odd number of 16 B chunks, so every ldmatrix phase and every 16 B
// store hits 8 distinct bank groups (see mma_common.cuh). Padding costs ~18%
// more smem at 128x128x32 but needs no index math in the hot loop and stays
// correct for any BK/BN that is a multiple of 16; a swizzle would save that
// smem, which matters only if a STAGES config stops fitting.
// PTX ISA citations for every instruction used: see mma_common.cuh.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "mma_common.cuh"

namespace aegis::gemm {
namespace {

using namespace mma_detail;

template <class T>
__global__ void __launch_bounds__(T::THREADS)
gemm_mma_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                __half* __restrict__ C, int M, int N, int K, bool vec) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);
  __half* Bs = As + T::A_ELEMS;

  const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
  float acc[T::MI][T::NI][4] = {};

  for (int k0 = 0; k0 < K; k0 += T::BK) {
    load_tile_sync<T>(As, Bs, A, B, M, N, K, m0, n0, k0, vec);
    __syncthreads();
    compute_tile<T>(As, Bs, acc);
    __syncthreads();
  }
  store_tile<T>(C, M, N, m0, n0, acc);
}

template <class T>
void launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
            cudaStream_t stream) {
  static const bool configured = [] {
    AEGIS_CUDA_CHECK(cudaFuncSetAttribute(gemm_mma_kernel<T>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          int(T::STAGE_BYTES)));
    return true;
  }();
  (void)configured;
  const bool vec = K % 8 == 0 && N % 8 == 0 && aligned16(A) && aligned16(B);
  const dim3 grid(ceil_div(N, T::BN), ceil_div(M, T::BM));
  gemm_mma_kernel<T><<<grid, T::THREADS, T::STAGE_BYTES, stream>>>(A, B, C, M, N, K, vec);
  AEGIS_CUDA_CHECK(cudaGetLastError());
}

template <class T>
cudaFuncAttributes attrs() {
  cudaFuncAttributes a{};
  AEGIS_CUDA_CHECK(cudaFuncGetAttributes(&a, gemm_mma_kernel<T>));
  a.sharedSizeBytes += T::STAGE_BYTES;  // report dynamic smem per block too
  return a;
}

constexpr size_t kN = std::size(MMA_CONFIGS);
template <size_t I>
using TileOf = Tile<MMA_CONFIGS[I].BM, MMA_CONFIGS[I].BN, MMA_CONFIGS[I].BK,
                    MMA_CONFIGS[I].WARPS_M, MMA_CONFIGS[I].WARPS_N>;

std::string name_of(const MmaConfig& c) {
  return "mma_" + std::to_string(c.BM) + "x" + std::to_string(c.BN) + "x" +
         std::to_string(c.BK) + "_" + std::to_string(c.WARPS_M) + "x" +
         std::to_string(c.WARPS_N);
}

std::string config_of(const MmaConfig& c) {
  return "{\"BM\":" + std::to_string(c.BM) + ",\"BN\":" + std::to_string(c.BN) +
         ",\"BK\":" + std::to_string(c.BK) + ",\"WARPS_M\":" + std::to_string(c.WARPS_M) +
         ",\"WARPS_N\":" + std::to_string(c.WARPS_N) + ",\"smem\":\"pad8\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(MMA_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(MMA_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{
      names[I].c_str(), configs[I].c_str(), &launch<TileOf<I>>, &attrs<TileOf<I>>}...};
  return entries;
}

}  // namespace

std::span<const KernelEntry> mma_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
