// Tier 5: tier 4's tensor-core main loop fed by TMA (cp.async.bulk.tensor)
// instead of per-thread cp.async.
//
// PTX ISA 9.4, Target ISA Notes (as encoded in CUDA 13.0's
// cccl/cuda/__ptx/instructions/generated/*.h; CLAUDE.md "TMA is available on
// sm_121"):
//   9.7.9.25.5.2 cp.async.bulk.tensor (.tile, .mbarrier::complete_tx::bytes):
//     "PTX ISA 8.0, Requires sm_90 or higher"; the .shared::cta destination
//     used here needs PTX ISA 8.6 (CUDA 13 emits 9.0). With .shared::cluster
//     the compiler adds a run-time check for a remote CTA plus a syscall path
//     (8-byte stack frame; measured 2026-10-17); .shared::cta compiles to the
//     inline UTMALDG only.
//     Only the .cta_group, .im2col::w and gather4/scatter4 forms need
//     sm_100a/sm_101a; none is used. ptxas: the 3d tile form assembles to
//     UTMALDG.3D for sm_120 and sm_121 and is rejected for sm_87.
//   9.7.13.15.9 mbarrier.init: "Requires sm_80 or higher."
//   9.7.13.15.11 mbarrier.expect_tx / .12 mbarrier.arrive with .expect_tx:
//     "Requires sm_90 or higher."
//   9.7.13.15.16 mbarrier.try_wait (.parity): "Requires sm_90 or higher."
//   9.7.13.4 fence.mbarrier_init.release.cluster: "Requires sm_90 or higher."
// ldmatrix / mma.sync: see mma_common.cuh (sm_75 / sm_80).
// sm_87 (Orin) has none of the sm_90 instructions: the kernel body compiles
// to an empty function for __CUDA_ARCH__ < 900 and the host launcher never
// takes the TMA path there (device check), so sm_87 keeps tier 4 (cp.async).
//
// Data movement: per stage one elected thread (thread 0) arms the stage's
// mbarrier with the stage's byte count (arrive.expect_tx) and issues one TMA
// box per 64-element column slice of each operand tile; the copies complete
// the barrier's transaction count. Every warp waits on that barrier
// (try_wait.parity, phase = use count & 1) and computes; as in tier 4, one
// __syncthreads per k-tile guarantees every warp is done with the stage the
// next prefetch overwrites, so no "empty" barriers are needed. All warps are
// MMA consumers (warp specialization is a later experiment).
//
// Smem layout: TMA's hardware swizzle. Each operand tile is ROWS x COLS with
// COLS the global-contiguous dimension (A row-major: As[m][k]; B row-major:
// Bs[k][n]; column-major operands transposed, as in mma_common.cuh), stored as
// boxes of BOX = min(COLS, 64) columns: 128-byte rows with
// CU_TENSOR_MAP_SWIZZLE_128B (BOX = 64), 64-byte rows with SWIZZLE_64B (BOX =
// 32). Swizzle = byte offset bits [4, 4+b) ^= bits [7, 7+b), b = 3 (128B) / 2
// (64B), within a box region aligned to 1024 B (TmaOperand::at). ldmatrix reads
// 8 rows at one 16-byte column: with 128-byte rows the 8 chunks land at c ^ (r
// & 7), with 64-byte rows at 4 * (r & 1) + (c ^ ((r >> 1) & 3)): 8 distinct
// positions of a 128-byte line, conflict-free (ncu: see LOG 2026-10-17).
//
// Boundaries: TMA zero-fills every element of a box outside the tensor
// (M, N and K tails), so the main loop has no masks and no boundary path; only
// the epilogue masks C (store_tile<T, true> on boundary CTAs).
//
// TMA's rules: global base 16-byte aligned, strides (ld, batch) multiples of
// 16 bytes. Anything else (and any device below sm_90) runs tier 4's kernel
// with the same tile (mma_pipe_impl.cuh, its own misaligned/scalar paths).
//
// Tensor maps (cuTensorMapEncodeTiled, through cudaGetDriverEntryPoint: no
// libcuda link) are built on the host and cached per host thread, keyed by
// everything they encode (pointer, sizes, ld, batch stride, box, swizzle);
// a call looks its two maps up and encodes only on a miss.
#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>

#include "aegis/gemm.h"
#include "mma_common.cuh"
#include "mma_pipe_impl.cuh"

namespace aegis::gemm {
namespace {

using namespace mma_detail;

// The sm_87 pass compiles the kernel body out, leaving the tile constants
// unreferenced there (nvcc diagnostic 177, "declared but never referenced").
#pragma nv_diag_suppress 177

// One operand's smem tile in TMA's swizzled layout (see the header comment).
template <int ROWS_, int COLS_>
struct TmaOperand {
  static constexpr int ROWS = ROWS_, COLS = COLS_;
  static constexpr int BOX = COLS < 64 ? COLS : 64;  // columns per TMA box
  static constexpr int ROW_BYTES = BOX * 2;          // 128 (SWIZZLE_128B) or 64 (SWIZZLE_64B)
  static constexpr int NBOX = COLS / BOX;
  static constexpr int BOX_BYTES = ROWS * ROW_BYTES;
  static constexpr int ELEMS = ROWS * COLS;
  static constexpr int SWZ_MASK = ROW_BYTES / 16 - 1;  // 7 (128B) or 3 (64B)
  static_assert(BOX == 64 || BOX == 32, "128B or 64B swizzle rows");
  static_assert(COLS % BOX == 0, "whole boxes");
  static_assert(BOX_BYTES % 1024 == 0, "box regions stay on the 1024 B swizzle atom");
  static_assert(ROWS <= 256, "TMA box dimension limit");

  __device__ __forceinline__ static int at(int row, int col) {
    const int b = col / BOX, c = col % BOX;
    int off = row * ROW_BYTES + c * 2;
    off ^= ((off >> 7) & SWZ_MASK) << 4;
    return (b * BOX_BYTES + off) >> 1;
  }
};

// Tier 3/4's Tile with the TMA smem operands: the same fields compute_tile and
// store_tile read.
template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_, Layout LA_, Layout LB_>
struct TmaTile {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
  static constexpr Layout LA = LA_, LB = LB_;
  static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
  static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  static constexpr int MI = WM / 16, NI = WN / 8;
  static constexpr bool A_K_CONTIG = LA == Layout::RowMajor;
  static constexpr bool B_K_CONTIG = LB == Layout::ColMajor;
  using SA = TmaOperand<A_K_CONTIG ? BM : BK, A_K_CONTIG ? BK : BM>;
  using SB = TmaOperand<B_K_CONTIG ? BN : BK, B_K_CONTIG ? BK : BN>;
  static constexpr int A_ELEMS = SA::ELEMS, B_ELEMS = SB::ELEMS;
  static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
  static constexpr size_t STAGE_BYTES = size_t(STAGE_ELEMS) * sizeof(__half);
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0);
};

// --- PTX wrappers (sm_90+) ---------------------------------------------------
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!done);
}
// One 3-D box (contiguous dim, other dim, batch) into smem, completing `bar`.
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
#endif

// Issue one operand tile's boxes. K_IS_ROW: the tile's rows are k (A
// column-major, B row-major): boxes start at (contig = mn0 + b * BOX, row = k0);
// else (contig = k0 + b * BOX, row = mn0).
template <class SOp, bool K_IS_ROW>
__device__ __forceinline__ void issue_operand(uint32_t dst, const CUtensorMap* map, int k0, int mn0, int z,
                                              uint32_t bar) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#pragma unroll
  for (int b = 0; b < SOp::NBOX; ++b) {
    const uint32_t d = dst + uint32_t(b * SOp::BOX_BYTES);
    if constexpr (K_IS_ROW) tma_load_3d(d, map, mn0 + b * SOp::BOX, k0, z, bar);
    else tma_load_3d(d, map, k0 + b * SOp::BOX, mn0, z, bar);
  }
#endif
}

template <class T, int STAGES>
__global__ void __launch_bounds__(T::THREADS)
gemm_mma_tma_kernel(const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b, KParams p,
                    bool pairs) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
  static_assert(STAGES >= 2, "a pipeline needs at least two buffers");
  // Box regions need 1024 B alignment (the 128B swizzle atom). The kernel has
  // no static smem, so the dynamic buffer starts at the aligned window base; no
  // slack: 1024 B of it cost 128x128x32 s3 its second block per SM (ncu,
  // LOG 2026-10-17). Barriers after the stages.
  extern __shared__ __align__(1024) unsigned char smem[];
  const uint32_t base = smem_addr(smem);
  if (base & 1023u) __trap();  // never seen; the launch returns an error, not wrong results
  const uint32_t bars = base + uint32_t(STAGES * T::STAGE_BYTES);
  const int z = blockIdx.z;
  const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
  const int KT = ceil_div(p.K, T::BK);
  const bool leader = threadIdx.x == 0;

  if (leader) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) mbar_init(bars + 8u * s, 1);
    fence_mbar_init();
  }
  __syncthreads();

  // The maps' addresses in kernel parameter space, taken once here.
  const CUtensorMap* const pa = &map_a;
  const CUtensorMap* const pb = &map_b;
  auto issue = [=](int s, int kt) {  // leader only
    const uint32_t bar = bars + 8u * s;
    const uint32_t sa = base + uint32_t(s * T::STAGE_BYTES);
    const uint32_t sb = sa + uint32_t(T::A_ELEMS * sizeof(__half));
    mbar_arrive_expect_tx(bar, uint32_t(T::STAGE_BYTES));
    issue_operand<typename T::SA, !T::A_K_CONTIG>(sa, pa, kt * T::BK, m0, z, bar);
    issue_operand<typename T::SB, T::B_K_CONTIG ? false : true>(sb, pb, kt * T::BK, n0, z, bar);
  };

  float acc[T::MI][T::NI][4] = {};
  if (leader) {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
      if (s < KT) issue(s, s);
  }
  for (int kt = 0; kt < KT; ++kt) {
    const int s = kt % STAGES;
    mbar_wait(bars + 8u * s, uint32_t((kt / STAGES) & 1));
    __syncthreads();  // every warp is past tile kt-1: its stage may be refilled
    const int pf = kt + STAGES - 1;
    if (leader && pf < KT) issue(pf % STAGES, pf);
    const __half* As = reinterpret_cast<const __half*>(smem + size_t(s) * T::STAGE_BYTES);
    compute_tile<T>(As, As + T::A_ELEMS, acc);
  }
  __half* C = p.C + z * p.sc;
  if (m0 + T::BM <= p.M && n0 + T::BN <= p.N) store_tile<T, false>(C, p, m0, n0, pairs, acc);
  else store_tile<T, true>(C, p, m0, n0, pairs, acc);
#endif
}

// --- Host: tensor maps ---------------------------------------------------------
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                 const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                 CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encode_fn() {
  static const EncodeTiled fn = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q{};
    // The CUDA 12.0 ABI of cuTensorMapEncodeTiled (the versioned query exists from 12.5).
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess) {
      cudaGetLastError();
      return EncodeTiled(nullptr);
    }
    return reinterpret_cast<EncodeTiled>(f);
  }();
  return fn;
}

struct MapKey {
  const void* ptr;
  uint64_t contig, other, batch, ld, stride;
  uint32_t box_contig, box_rows;
  bool operator==(const MapKey&) const = default;
};
struct MapEntry {
  MapKey key;
  CUtensorMap map;
};

// The tensor map of one operand: dims (contiguous, other, batch), strides in
// bytes (ld, batch stride), box (BOX, ROWS, 1), swizzle by row bytes, OOB zero
// fill. Cached per host thread (bounded; oldest entries recycled).
cudaError_t operand_map(const MapKey& k, CUtensorMap* out) {
  thread_local std::vector<MapEntry> cache;
  thread_local size_t next = 0;
  constexpr size_t kMaxMaps = 256;
  for (const MapEntry& e : cache)
    if (e.key == k) {
      *out = e.map;
      return cudaSuccess;
    }
  const EncodeTiled enc = encode_fn();
  if (!enc) return cudaErrorNotSupported;
  const cuuint64_t dims[3] = {k.contig, k.other, k.batch};
  const cuuint64_t strides[2] = {k.ld * 2, (k.batch > 1 ? k.stride : k.ld * k.other) * 2};
  const cuuint32_t box[3] = {k.box_contig, k.box_rows, 1};
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  const CUtensorMapSwizzle swz = k.box_contig * 2 == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
  MapEntry e{k, {}};
  const CUresult r = enc(&e.map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<void*>(k.ptr), dims, strides, box,
                         elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
  if (cache.size() < kMaxMaps) cache.push_back(e);
  else cache[next++ % kMaxMaps] = e;
  *out = e.map;
  return cudaSuccess;
}

// TMA can take this problem: device sm_90+, 16 B base pointers, ld and batch
// strides multiples of 8 elements (16 B), sizes within the encoder's range.
bool tma_ok(const GemmParams& p) {
  // Per device: the one the launch runs on (hw::physical), not the first one seen.
  const hw::DeviceLimits* d = hw::physical();
  const bool device_ok = d && d->arch >= 90 && encode_fn() != nullptr;
  constexpr long long kMaxDim = (1ll << 32) - 1;
  // A batch-shared operand (stride 0) has no TMA batch dimension: tier 4 takes it.
  const bool strides_ok = p.batch <= 1 || (p.stride_a > 0 && p.stride_b > 0);
  return device_ok && vec16_ok(p) && strides_ok && p.M <= kMaxDim && p.N <= kMaxDim && p.K <= kMaxDim;
}

constexpr cudaError_t kNoMap = cudaErrorNotSupported;

template <class T, int STAGES>
cudaError_t launch_tma(const GemmParams& gp, cudaStream_t stream) {
  constexpr size_t smem = T::STAGE_BYTES * STAGES + 8 * STAGES;
  static hw::PerDevice once;
  const cudaError_t configured = hw::configure_smem(once, gemm_mma_tma_kernel<T, STAGES>, smem);
  if (configured != cudaSuccess) return configured;
  const dim3 grid(ceil_div(gp.N, T::BN), ceil_div(gp.M, T::BM), gp.batch);
  if (!grid_ok(grid)) return cudaErrorInvalidValue;
  // A: row-major dims (K, M), col-major (M, K); B: row-major (N, K), col-major (K, N).
  const bool ra = gp.layout_a == Layout::RowMajor, rb = gp.layout_b == Layout::RowMajor;
  using SA = typename T::SA;
  using SB = typename T::SB;
  const MapKey ka{gp.A, uint64_t(ra ? gp.K : gp.M), uint64_t(ra ? gp.M : gp.K), uint64_t(gp.batch),
                  uint64_t(gp.lda), uint64_t(gp.stride_a), uint32_t(SA::BOX), uint32_t(SA::ROWS)};
  const MapKey kb{gp.B, uint64_t(rb ? gp.N : gp.K), uint64_t(rb ? gp.K : gp.N), uint64_t(gp.batch),
                  uint64_t(gp.ldb), uint64_t(gp.stride_b), uint32_t(SB::BOX), uint32_t(SB::ROWS)};
  CUtensorMap ma, mb;
  // The encoder refusing a map (a limit tma_ok does not model) is not an error:
  // run() then takes tier 4 (kNoMap).
  if (operand_map(ka, &ma) != cudaSuccess || operand_map(kb, &mb) != cudaSuccess) return kNoMap;
  gemm_mma_tma_kernel<T, STAGES><<<grid, T::THREADS, smem, stream>>>(ma, mb, to_kparams(gp), c_pairs_ok(gp));
  return cudaGetLastError();
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  // TMA cannot take it: tier 4's kernel on the same tile (2 stages: its
  // misaligned path is synchronous, and BK = 64 tiles fit its smem that way).
  if (!tma_ok(p)) return mma_pipe_detail::run<BM, BN, BK, WM, WN, 2>(p, stream);
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  cudaError_t e;
  if (p.layout_a == R && p.layout_b == R) e = launch_tma<TmaTile<BM, BN, BK, WM, WN, R, R>, STAGES>(p, stream);
  else if (p.layout_a == R) e = launch_tma<TmaTile<BM, BN, BK, WM, WN, R, Cm>, STAGES>(p, stream);
  else if (p.layout_b == R) e = launch_tma<TmaTile<BM, BN, BK, WM, WN, Cm, R>, STAGES>(p, stream);
  else e = launch_tma<TmaTile<BM, BN, BK, WM, WN, Cm, Cm>, STAGES>(p, stream);
  return e == kNoMap ? mma_pipe_detail::run<BM, BN, BK, WM, WN, 2>(p, stream) : e;
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream) {
  return run<BM, BN, BK, WM, WN, STAGES>(row_major_params(A, B, C, M, N, K), stream);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
cudaError_t attrs(cudaFuncAttributes* a) {
  using T = TmaTile<BM, BN, BK, WM, WN, Layout::RowMajor, Layout::RowMajor>;  // row-major NN: the tuned kernel
  const cudaError_t e = cudaFuncGetAttributes(a, gemm_mma_tma_kernel<T, STAGES>);
  if (e == cudaSuccess) a->sharedSizeBytes += T::STAGE_BYTES * STAGES + 8 * STAGES;
  return e;
}

// Requirements (KernelReq): sm_90 and TMA; the TMA ring (stages + one mbarrier each) or the
// tier-4 fallback on the same tile (2 stages), whichever is larger, over every layout.
template <int BM, int BN, int BK, int WM, int WN, int STAGES>
constexpr KernelReq req() {
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  constexpr size_t tma = std::max({TmaTile<BM, BN, BK, WM, WN, R, R>::STAGE_BYTES, TmaTile<BM, BN, BK, WM, WN, R, Cm>::STAGE_BYTES,
                                   TmaTile<BM, BN, BK, WM, WN, Cm, R>::STAGE_BYTES, TmaTile<BM, BN, BK, WM, WN, Cm, Cm>::STAGE_BYTES}) *
                             STAGES + 8 * STAGES;
  constexpr size_t pipe = std::max({Tile<BM, BN, BK, WM, WN, R, R>::STAGE_BYTES, Tile<BM, BN, BK, WM, WN, R, Cm>::STAGE_BYTES,
                                    Tile<BM, BN, BK, WM, WN, Cm, R>::STAGE_BYTES, Tile<BM, BN, BK, WM, WN, Cm, Cm>::STAGE_BYTES}) * 2;
  return KernelReq{90, true, std::max(tma, pipe), TmaTile<BM, BN, BK, WM, WN, R, R>::THREADS};
}

constexpr size_t kN = std::size(MMA_TMA_CONFIGS);
#define AEGIS_TMA_ARGS(I)                                                                      \
  MMA_TMA_CONFIGS[I].BM, MMA_TMA_CONFIGS[I].BN, MMA_TMA_CONFIGS[I].BK, MMA_TMA_CONFIGS[I].WARPS_M, \
      MMA_TMA_CONFIGS[I].WARPS_N, MMA_TMA_CONFIGS[I].STAGES

std::string name_of(const MmaPipeConfig& c) {
  return "mma_tma_" + std::to_string(c.BM) + "x" + std::to_string(c.BN) + "x" + std::to_string(c.BK) + "_" +
         std::to_string(c.WARPS_M) + "x" + std::to_string(c.WARPS_N) + "_s" + std::to_string(c.STAGES);
}

std::string config_of(const MmaPipeConfig& c) {
  return "{\"BM\":" + std::to_string(c.BM) + ",\"BN\":" + std::to_string(c.BN) + ",\"BK\":" + std::to_string(c.BK) +
         ",\"WARPS_M\":" + std::to_string(c.WARPS_M) + ",\"WARPS_N\":" + std::to_string(c.WARPS_N) +
         ",\"STAGES\":" + std::to_string(c.STAGES) + ",\"load\":\"tma\",\"smem\":\"tma_swizzle\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(MMA_TMA_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(MMA_TMA_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{
      names[I].c_str(), configs[I].c_str(), &launch<AEGIS_TMA_ARGS(I)>, &attrs<AEGIS_TMA_ARGS(I)>,
      &run<AEGIS_TMA_ARGS(I)>, 0, req<AEGIS_TMA_ARGS(I)>()}...};
  return entries;
}
#undef AEGIS_TMA_ARGS

}  // namespace

std::span<const KernelEntry> mma_tma_entries() { return make_entries(std::make_index_sequence<kN>{}); }

const char* mma_tma_path(const GemmParams& p) { return tma_ok(p) ? "tma" : "cp.async (mma_pipe)"; }

}  // namespace aegis::gemm
