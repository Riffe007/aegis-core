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
// General GEMM (GemmParams: layouts, ld, alpha/beta, strided batch over
// blockIdx.z). Paths, chosen per launch / per CTA:
//   - vec (16 B-compatible pointers, ld and strides; vec16_ok): the pipeline.
//     Interior CTAs (tile wholly inside M x N) copy unpredicated; boundary
//     CTAs use row/column masks built once per CTA (OperandPlan). When K is
//     not a multiple of BK only the last k-tile is copied with a k mask.
//     Boundary chunks never read past the valid elements (whole chunks:
//     cp.async; empty: zero store; partial: valid halves + explicit zeros).
//   - otherwise (misaligned): tier 3's synchronous loop with the scalar,
//     element-wise bounds-checked loader.
// Smem layout per operand layout (XOR-swizzled) and compute: mma_common.cuh.
// PTX ISA citations for every instruction used: see mma_common.cuh.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "mma_common.cuh"

namespace aegis::gemm {
namespace {

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

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  return run<BM, BN, BK, WM, WN, STAGES>(row_major_params(A, B, C, M, N, K), stream);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t attrs(cudaFuncAttributes* a) {
  using T = Tile<BM, BN, BK, WM, WN>;  // row-major NN: the tuned kernel
  const cudaError_t e = cudaFuncGetAttributes(a, gemm_mma_pipe_kernel<T, STAGES>);
  if (e == cudaSuccess) a->sharedSizeBytes += T::STAGE_BYTES * STAGES;  // report dynamic smem per block too
  return e;
}

constexpr size_t kN = std::size(MMA_PIPE_CONFIGS);
#define AEGIS_PIPE_ARGS(I)                                                               \
  MMA_PIPE_CONFIGS[I].BM, MMA_PIPE_CONFIGS[I].BN, MMA_PIPE_CONFIGS[I].BK,                \
      MMA_PIPE_CONFIGS[I].WARPS_M, MMA_PIPE_CONFIGS[I].WARPS_N, MMA_PIPE_CONFIGS[I].STAGES

std::string name_of(const MmaPipeConfig& c) {
  return "mma_pipe_" + std::to_string(c.BM) + "x" + std::to_string(c.BN) + "x" +
         std::to_string(c.BK) + "_" + std::to_string(c.WARPS_M) + "x" +
         std::to_string(c.WARPS_N) + "_s" + std::to_string(c.STAGES);
}

std::string config_of(const MmaPipeConfig& c) {
  return "{\"BM\":" + std::to_string(c.BM) + ",\"BN\":" + std::to_string(c.BN) +
         ",\"BK\":" + std::to_string(c.BK) + ",\"WARPS_M\":" + std::to_string(c.WARPS_M) +
         ",\"WARPS_N\":" + std::to_string(c.WARPS_N) + ",\"STAGES\":" +
         std::to_string(c.STAGES) + ",\"smem\":\"xor\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(MMA_PIPE_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(MMA_PIPE_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{
      names[I].c_str(), configs[I].c_str(), &launch<AEGIS_PIPE_ARGS(I)>, &attrs<AEGIS_PIPE_ARGS(I)>,
      &run<AEGIS_PIPE_ARGS(I)>}...};
  return entries;
}
#undef AEGIS_PIPE_ARGS

}  // namespace

std::span<const KernelEntry> mma_pipe_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
