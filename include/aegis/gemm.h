#pragma once
// AEGIS Core GEMM implementations.
//
// Convention for every impl:
//   C[M,N] = A[M,K] * B[K,N], all row-major, FP16 in, FP32 accumulate, FP16 out.
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

// Tier 0: one thread per output, global memory only.
cudaError_t launch_naive(const __half* A, const __half* B, __half* C,
                         int M, int N, int K, cudaStream_t stream);
cudaError_t attrs_naive(cudaFuncAttributes* attrs);
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

struct KernelEntry {
  const char* name;    // impl name: bench_gemm --impl, aegis_tune, perf DB
  const char* config;  // JSON; states every template parameter
  LaunchFn launch;
  AttrsFn attrs;       // sharedSizeBytes includes dynamic smem per block
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
    // step against 4 + 4 ldmatrix.x4. s4 is left out: with the pad8 layout one
    // stage is 27,136 B (128x256) / 29,184 B (256x128), so s4 needs 108,544 /
    // 116,736 B, over the 101,376 B per-block opt-in limit of sm_121.
    {128, 256, 32, 2, 4, 2},
    {128, 256, 32, 2, 4, 3},
    {256, 128, 32, 4, 2, 2},
    {256, 128, 32, 4, 2, 3},
};
std::span<const KernelEntry> mma_pipe_entries();

}  // namespace aegis::gemm
