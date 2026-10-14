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
//
// General GEMM (GemmParams). Paths: vec (16 B-compatible) with an unpredicated
// interior fast path and per-CTA masks on boundary CTAs (k mask only on the
// partial last k-tile); otherwise the scalar, element-wise bounds-checked
// loader. Operand smem layouts per mma_common.cuh.
// PTX ISA citations for every instruction used: see mma_common.cuh.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "mma_common.cuh"

namespace aegis::gemm {
namespace {

using namespace mma_detail;

template <class T, bool MASK>
__device__ __forceinline__ void sync_mainloop(__half* As, __half* Bs, const __half* A, const __half* B,
                                              const KParams& p, int m0, int n0,
                                              float (&acc)[T::MI][T::NI][4]) {
  PlanA<T> pa;
  PlanB<T> pb;
  pa.init(A, p.lda, m0, p.M);
  pb.init(B, p.ldb, n0, p.N);
  const int KT = ceil_div(p.K, T::BK);
  const bool k_tail = p.K % T::BK != 0;
  for (int kt = 0; kt < KT; ++kt) {
    const int k0 = kt * T::BK;
    if (k_tail && kt == KT - 1) {
      pa.template load_sync<MASK, true>(As, k0, p.K);
      pb.template load_sync<MASK, true>(Bs, k0, p.K);
    } else {
      pa.template load_sync<MASK, false>(As, k0, p.K);
      pb.template load_sync<MASK, false>(Bs, k0, p.K);
    }
    pa.advance();
    pb.advance();
    __syncthreads();
    compute_tile<T>(As, Bs, acc);
    __syncthreads();
  }
}

template <class T>
__global__ void __launch_bounds__(T::THREADS)
gemm_mma_kernel(KParams p, bool vec, bool pairs) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);
  __half* Bs = As + T::A_ELEMS;
  const long long z = blockIdx.z;
  const __half* A = p.A + z * p.sa;
  const __half* B = p.B + z * p.sb;
  __half* C = p.C + z * p.sc;
  const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
  float acc[T::MI][T::NI][4] = {};

  if (!vec) {  // misaligned: scalar, fully bounds-checked
    for (int k0 = 0; k0 < p.K; k0 += T::BK) {
      load_tile_scalar<T>(As, Bs, A, B, p, m0, n0, k0);
      __syncthreads();
      compute_tile<T>(As, Bs, acc);
      __syncthreads();
    }
    store_tile<T, true>(C, p, m0, n0, pairs, acc);
    return;
  }
  if (m0 + T::BM <= p.M && n0 + T::BN <= p.N) {
    sync_mainloop<T, false>(As, Bs, A, B, p, m0, n0, acc);
    store_tile<T, false>(C, p, m0, n0, pairs, acc);
  } else {
    sync_mainloop<T, true>(As, Bs, A, B, p, m0, n0, acc);
    store_tile<T, true>(C, p, m0, n0, pairs, acc);
  }
}

template <class T>
cudaError_t launch_layout(const GemmParams& gp, cudaStream_t stream) {
  // Once per instantiation; a failure is returned on every call.
  static const cudaError_t configured = cudaFuncSetAttribute(
      gemm_mma_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(T::STAGE_BYTES));
  if (configured != cudaSuccess) return configured;
  const dim3 grid(ceil_div(gp.N, T::BN), ceil_div(gp.M, T::BM), gp.batch);
  if (!grid_ok(grid)) return cudaErrorInvalidValue;
  gemm_mma_kernel<T><<<grid, T::THREADS, T::STAGE_BYTES, stream>>>(to_kparams(gp), vec16_ok(gp),
                                                                   c_pairs_ok(gp));
  return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  if (p.layout_a == R && p.layout_b == R) return launch_layout<Tile<BM, BN, BK, WM, WN, R, R>>(p, stream);
  if (p.layout_a == R) return launch_layout<Tile<BM, BN, BK, WM, WN, R, Cm>>(p, stream);
  if (p.layout_b == R) return launch_layout<Tile<BM, BN, BK, WM, WN, Cm, R>>(p, stream);
  return launch_layout<Tile<BM, BN, BK, WM, WN, Cm, Cm>>(p, stream);
}

template <int BM, int BN, int BK, int WM, int WN>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  return run<BM, BN, BK, WM, WN>(row_major_params(A, B, C, M, N, K), stream);
}

template <int BM, int BN, int BK, int WM, int WN>
cudaError_t attrs(cudaFuncAttributes* a) {
  using T = Tile<BM, BN, BK, WM, WN>;  // row-major NN
  const cudaError_t e = cudaFuncGetAttributes(a, gemm_mma_kernel<T>);
  if (e == cudaSuccess) a->sharedSizeBytes += T::STAGE_BYTES;  // report dynamic smem per block too
  return e;
}

constexpr size_t kN = std::size(MMA_CONFIGS);
#define AEGIS_MMA_ARGS(I) \
  MMA_CONFIGS[I].BM, MMA_CONFIGS[I].BN, MMA_CONFIGS[I].BK, MMA_CONFIGS[I].WARPS_M, MMA_CONFIGS[I].WARPS_N

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
      names[I].c_str(), configs[I].c_str(), &launch<AEGIS_MMA_ARGS(I)>, &attrs<AEGIS_MMA_ARGS(I)>,
      &run<AEGIS_MMA_ARGS(I)>}...};
  return entries;
}
#undef AEGIS_MMA_ARGS

}  // namespace

std::span<const KernelEntry> mma_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
