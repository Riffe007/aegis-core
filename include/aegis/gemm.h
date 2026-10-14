#pragma once
// AEGIS Core GEMM implementations.
//
// Convention for every impl:
//   C[M,N] = A[M,K] * B[K,N], all row-major, FP16 in, FP32 accumulate, FP16 out.
//
// Each impl exposes:
//   launch_<impl>(A, B, C, M, N, K, stream)   asynchronous, no sync
//   attrs_<impl>()                            cudaFuncAttributes of its kernel
//   CONFIG_<impl>                             JSON string of its fixed config
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
void launch_naive(const __half* A, const __half* B, __half* C,
                  int M, int N, int K, cudaStream_t stream);
cudaFuncAttributes attrs_naive();
inline constexpr const char* CONFIG_naive = R"({"block":[16,16]})";

// Tier 1: 32x32 shared-memory tiles, one output per thread.
void launch_tiled(const __half* A, const __half* B, __half* C,
                  int M, int N, int K, cudaStream_t stream);
cudaFuncAttributes attrs_tiled();
inline constexpr const char* CONFIG_tiled = R"({"tile":32,"block":[32,32]})";

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
using LaunchFn = void (*)(const __half* A, const __half* B, __half* C,
                          int M, int N, int K, cudaStream_t stream);
using AttrsFn = cudaFuncAttributes (*)();

struct KernelEntry {
  const char* name;    // impl name: bench_gemm --impl, sweep.py, perf DB
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

}  // namespace aegis::gemm
