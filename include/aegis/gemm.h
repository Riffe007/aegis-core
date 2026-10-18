#pragma once
// AEGIS Core GEMM implementations.
//
// Convention for every impl:
//   C[M,N] = A[M,K] * B[K,N], all row-major, FP16 in, FP32 accumulate, FP16 out
//   for the legacy launch(); run() takes GemmParams (layouts, ld, alpha/beta,
//   strided batch) for every production tier (0, 2, 3, 4).
//
// Each impl exposes:
//   launch_<impl>(A, B, C, M, N, K, stream)   asynchronous, no sync; returns the
//                                             launch status (never exits)
//   attrs_<impl>(&attrs)                      cudaFuncAttributes of its kernel
//   CONFIG_<impl>                             JSON string of its fixed config
// Library code returns cudaError_t; only bench/test/tool binaries may abort.
//
// Templated impls (tiers 2+) expose a constexpr config table instead. Each row
// is one instantiation; its impl name and JSON config are derived from the row.
// Every launchable instantiation, of every tier, is listed by registry().

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <span>
#include <string_view>

namespace aegis::gemm {

// ---------------------------------------------------------------------------
// General GEMM parameters (Production bar: layouts, leading dimensions,
// alpha/beta, strided batch). For batch b in [0, batch):
//   C_b = alpha * A_b * B_b + beta * C_b
//   A_b = A + b * stride_a, an M x K matrix, row- or column-major with lda;
//   B_b = B + b * stride_b, K x N, row- or column-major with ldb;
//   C_b = C + b * stride_c, M x N, row-major with ldc (ldc >= N).
// Row-major X (R x C): X[r * ld + c], ld >= C. Column-major: X[r + c * ld], ld >= R.
// FP16 in, FP32 accumulate, FP16 out. alpha == 0: A*B does not contribute
// (C = beta*C; NaN/Inf in A or B do not propagate), as in BLAS; otherwise
// NaN and Inf propagate. beta == 0: C is not read (it may hold
// NaN). M, N, K or batch == 0: no-op, success, nothing launched (C untouched).
// Negative sizes, too-small ld or null pointers with nonzero sizes:
// cudaErrorInvalidValue.
// ---------------------------------------------------------------------------
enum class Layout : int { RowMajor = 0, ColMajor = 1 };

struct GemmParams {
  const __half* A = nullptr;
  const __half* B = nullptr;
  __half* C = nullptr;
  int M = 0, N = 0, K = 0;
  Layout layout_a = Layout::RowMajor, layout_b = Layout::RowMajor;
  long long lda = 0, ldb = 0, ldc = 0;
  float alpha = 1.f, beta = 0.f;
  int batch = 1;
  long long stride_a = 0, stride_b = 0, stride_c = 0;
};

// The legacy convention: everything row-major, packed, alpha 1, beta 0, one batch.
inline GemmParams row_major_params(const __half* A, const __half* B, __half* C, int M, int N,
                                   int K) {
  GemmParams p;
  p.A = A; p.B = B; p.C = C;
  p.M = M; p.N = N; p.K = K;
  p.lda = K; p.ldb = N; p.ldc = N;
  return p;
}

// cudaSuccess if p is valid, else cudaErrorInvalidValue (see above).
cudaError_t validate(const GemmParams& p);
// True if the call is a no-op (some size is zero).
inline bool is_empty(const GemmParams& p) {
  return p.M == 0 || p.N == 0 || p.K == 0 || p.batch == 0;
}

// Tier 0: one thread per output, global memory only.
cudaError_t launch_naive(const __half* A, const __half* B, __half* C,
                         int M, int N, int K, cudaStream_t stream);
cudaError_t attrs_naive(cudaFuncAttributes* attrs);
cudaError_t run_naive(const GemmParams& p, cudaStream_t stream);
inline constexpr const char* CONFIG_naive = R"({"block":[16,16]})";

// Tier 1: 32x32 shared-memory tiles, one output per thread.
cudaError_t launch_tiled(const __half* A, const __half* B, __half* C,
                         int M, int N, int K, cudaStream_t stream);
cudaError_t attrs_tiled(cudaFuncAttributes* attrs);
inline constexpr const char* CONFIG_tiled = R"({"tile":32,"block":[32,32]})";

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
using LaunchFn = cudaError_t (*)(const __half* A, const __half* B, __half* C,
                                 int M, int N, int K, cudaStream_t stream);
using AttrsFn = cudaError_t (*)(cudaFuncAttributes* attrs);
// General entry (layouts, ld, alpha/beta, batch). Validates p; asynchronous.
using RunFn = cudaError_t (*)(const GemmParams& p, cudaStream_t stream);

struct KernelEntry {
  const char* name;    // impl name: bench_gemm --impl, aegis_tune, perf DB
  const char* config;  // JSON; states every template parameter
  LaunchFn launch;
  AttrsFn attrs;       // sharedSizeBytes includes dynamic smem per block (row-major NN kernel)
  RunFn run;           // nullptr: legacy row-major-only impl (tier 1); launch() is then the only entry
  int tune_max_m = 0;  // > 0: aegis_tune times it only on shapes with M <= this (decode families); any M runs
};

// Every launchable impl, tiers 0 and up, in tier order. Defined in registry.cu.
std::span<const KernelEntry> registry();
const KernelEntry* find(std::string_view name);  // nullptr if not registered

// ---------------------------------------------------------------------------
// Tier 2: register-blocked. Block computes BMxBN, each of the
// (BM/TM)*(BN/TN) threads a TMxTN micro-tile; A staged transposed in smem.
// Name: regblock_<BM>x<BN>x<BK>_<TM>x<TN>
// ---------------------------------------------------------------------------
struct RegblockConfig { int BM, BN, BK, TM, TN; };
inline constexpr RegblockConfig REGBLOCK_CONFIGS[] = {
    {64, 64, 8, 4, 4},
    {128, 128, 8, 8, 8},
    {128, 64, 16, 8, 4},
};
std::span<const KernelEntry> regblock_entries();

// ---------------------------------------------------------------------------
// Tier 3: tensor cores via inline PTX (ldmatrix + mma.sync m16n8k16, FP32
// accumulate), synchronous global->smem loads. WARPS_M x WARPS_N warps, each
// computing (BM/WARPS_M) x (BN/WARPS_N). Name: mma_<BM>x<BN>x<BK>_<WM>x<WN>
// ---------------------------------------------------------------------------
struct MmaConfig { int BM, BN, BK, WARPS_M, WARPS_N; };
inline constexpr MmaConfig MMA_CONFIGS[] = {
    {128, 128, 32, 2, 4},  // default: 8 warps of 64x32
    {128, 64, 32, 2, 2},
    {64, 128, 32, 2, 4},
    {64, 64, 32, 2, 2},
};
std::span<const KernelEntry> mma_entries();

// ---------------------------------------------------------------------------
// Tier 4: tier 3 + STAGES-deep cp.async (16 B, .cg) global->smem pipeline.
// Name: mma_pipe_<BM>x<BN>x<BK>_<WM>x<WN>_s<STAGES>
// ---------------------------------------------------------------------------
struct MmaPipeConfig { int BM, BN, BK, WARPS_M, WARPS_N, STAGES; };
inline constexpr MmaPipeConfig MMA_PIPE_CONFIGS[] = {
    {128, 128, 32, 2, 4, 2},
    {128, 128, 32, 2, 4, 3},
    {128, 128, 32, 2, 4, 4},
    {128, 64, 32, 2, 2, 2},
    {128, 64, 32, 2, 2, 3},
    {128, 64, 32, 2, 2, 4},
    {64, 128, 32, 2, 4, 2},
    {64, 128, 32, 2, 4, 3},
    {64, 128, 32, 2, 4, 4},
    {64, 64, 32, 2, 2, 2},
    {64, 64, 32, 2, 2, 3},
    {64, 64, 32, 2, 2, 4},
    // 64x64 warp tiles (8 warps, 1 block per SM): MI=4 x NI=8 = 32 MMAs per k16
    // step against 4 + 4 ldmatrix.x4. s4 fits since the XOR swizzle replaced the
    // padding: 24,576 B per stage, 98,304 B for 4 stages (sm_121 opt-in limit
    // 101,376 B); with the old pad8 layout it needed 108,544 / 116,736 B.
    {128, 256, 32, 2, 4, 2},
    {128, 256, 32, 2, 4, 3},
    {128, 256, 32, 2, 4, 4},
    {256, 128, 32, 4, 2, 2},
    {256, 128, 32, 4, 2, 3},
    {256, 128, 32, 4, 2, 4},
};
std::span<const KernelEntry> mma_pipe_entries();

// ---------------------------------------------------------------------------
// Tier 5: tier 4's tensor-core loop with TMA (cp.async.bulk.tensor) loads into
// a STAGES-deep ring, mbarrier completion, TMA's 128B/64B hardware swizzle and
// out-of-bounds zero fill (src/gemm/mma_tma.cu). sm_90+ only; problems TMA
// cannot take (misaligned pointers, ld or batch strides; batch-shared operands;
// devices below sm_90) run tier 4's kernel on the same tile.
// Name: mma_tma_<BM>x<BN>x<BK>_<WM>x<WN>_s<STAGES>. Smem per block:
// STAGES * (BM + BN) * BK * 2 + 1024 (alignment) + 8 * STAGES (barriers),
// within sm_121's 101,376 B.
// ---------------------------------------------------------------------------
inline constexpr MmaPipeConfig MMA_TMA_CONFIGS[] = {
    {128, 128, 32, 2, 4, 3},
    {128, 128, 32, 2, 4, 4},
    {128, 128, 32, 2, 4, 6},
    {128, 128, 64, 2, 4, 2},
    {128, 128, 64, 2, 4, 3},
    {256, 128, 32, 4, 2, 3},
    {256, 128, 32, 4, 2, 4},
    {128, 256, 32, 2, 4, 4},
    {256, 128, 64, 4, 2, 2},
};
std::span<const KernelEntry> mma_tma_entries();
// Which path an mma_tma impl takes for p: "tma" or "cp.async (mma_pipe)".
const char* mma_tma_path(const GemmParams& p);

// ---------------------------------------------------------------------------
// Decode GEMV family (M <= 16; B = weights [K, N], memory-bound on B), with
// deterministic split-K (src/gemm/splitk.cuh): S CTAs per output tile each
// take a K slice, write FP32 partials to the per-stream workspace
// (src/gemm/workspace.h), and the last of them sums the S partials in split
// order 0..S-1 (an ordered ticket, no atomic accumulation) and applies
// alpha/beta. Bitwise identical across runs for a given (impl, shape).
// Any M runs (tiles of up to 16 rows); tune_max_m = 16.
//
// CUDA-core path (src/gemm/gemv.cu): lanes span N with VEC-half (8: 16 B,
// 4: 8 B) B loads, warps split the CTA's K slice, each thread keeps
// MT x VEC FP32 accumulators (MT = the smallest of 1, 2, 4, 8, 16 >= M, picked
// per call), A's slice staged in smem as FP32.
// Name: gemv_<BN>x<THREADS>_v<VEC>_k<S|sm>.
//
// Tensor-core path (src/gemm/gemv_tc.cu): m16n8k16 with A padded to 16 rows,
// WARPS_N warps across BN, B streamed by a STAGES-deep cp.async ring.
// Name: gemv_tc_16x<BN>x<BK>_1x<WARPS_N>_s<STAGES>_k<S|sm>.
//
// S is the requested split count. 0 ("_ksm", every tuned config): sized at launch
// from the detected SM count (splitk::sm_load_splits: the S minimizing the busiest
// SM's K rows, splits at least 2 * STAGES * BK rows on the tensor-core path, 64
// on the CUDA-core path). Winners are per arch, but one arch spans devices with
// different SM counts (sm_87: Orin Nano, AGX Orin), so no table fixes S; a
// fixed S > 0 remains for experiments. The effective count is
// ceil(K / round_up(ceil(K / S), 32)), lowered so the partials fit in
// kSplitKMaxWorkspace (S = 1 needs no workspace). gemv_launch_plan reports it
// and the grid.
// ---------------------------------------------------------------------------
struct GemvConfig { int BN, THREADS, VEC, S; };  // S = 0: from the SM count
inline constexpr GemvConfig GEMV_CONFIGS[] = {
    {64, 64, 8, 0},  {64, 128, 8, 0}, {128, 128, 8, 0}, {128, 256, 8, 0},
    {256, 256, 8, 0}, {64, 128, 4, 0}, {128, 256, 4, 0},
};
std::span<const KernelEntry> gemv_entries();

struct GemvTcConfig { int BN, BK, WARPS_N, STAGES, S; };  // S = 0: from the SM count
// Smem per CTA (ring): 64x64 s4 42 KiB, s5 52 KiB, s6 63 KiB; 64x32 s6 32 KiB, s8
// 43 KiB; 32x64 s4 25 KiB; 128x32 s4 38 KiB, s6 57 KiB; 128x64 s4 73 KiB; 256x32 s4
// 69 KiB. CTAs per SM follow from the device's smem per SM (occupancy API).
inline constexpr GemvTcConfig GEMV_TC_CONFIGS[] = {
    {64, 64, 4, 4, 0},  {64, 64, 4, 5, 0},  {64, 64, 4, 6, 0},  {128, 64, 4, 4, 0},
    {64, 32, 4, 6, 0},  {64, 32, 4, 8, 0},  {32, 64, 2, 4, 0},  {32, 64, 2, 6, 0},
    {128, 32, 4, 4, 0}, {128, 32, 4, 6, 0}, {64, 64, 2, 4, 0},  {256, 32, 8, 4, 0},
};
std::span<const KernelEntry> gemv_tc_entries();

// Partials beyond this lower S (bytes; split-K workspace, per call).
inline constexpr long long kSplitKMaxWorkspace = 64ll << 20;

// The launch a gemv / gemv_tc impl makes for p on the current device (or the
// thread's simulated profile): effective splits (1: no split) and grid.
struct DecodeLaunch {
  int splits = 0;
  unsigned grid_x = 0, grid_y = 0, grid_z = 0;  // N tiles, M tiles, batch x splits
};
// False if `impl` is not a gemv / gemv_tc impl.
bool gemv_launch_plan(const char* impl, const GemmParams& p, DecodeLaunch* out);
// The effective split count (gemv_launch_plan's splits), or 0 if `impl` is not
// a gemv / gemv_tc impl.
int gemv_split_count(const char* impl, const GemmParams& p);

}  // namespace aegis::gemm
