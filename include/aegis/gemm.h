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

}  // namespace aegis::gemm
