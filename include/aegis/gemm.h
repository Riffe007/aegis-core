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
// Templated impls (tiers 2+) will expose one launch per instantiation through a
// config table that the autotuner enumerates. Keep tier 0/1 simple.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

}  // namespace aegis::gemm
