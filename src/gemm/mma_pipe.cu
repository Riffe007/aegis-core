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
#include "mma_pipe_impl.cuh"  // the kernel, launch_layout and run (shared with mma_tma.cu's fallback)

namespace aegis::gemm {
namespace {

using namespace mma_detail;
using namespace mma_pipe_detail;

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
