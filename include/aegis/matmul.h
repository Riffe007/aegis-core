#pragma once
// aegis::matmul: one entry point that runs the tuned winner for this device.
//
//   C[M,N] = A[M,K] * B[K,N], row-major, FP16 in, FP32 accumulate, FP16 out.
//
// Selection, against the winners compiled in for the device's sm_arch
// (include/aegis/winners/sm_<arch>.h, written by aegis-perf emit-header):
//   1. exact (M, N, K) match;
//   2. else the winner with the same N and K whose M is nearest in log2
//      (ties go to the larger M);
//   3. else cuBLAS (also if the winner's impl is no longer registered).
// Asynchronous on `stream`, like the per-impl launchers. Never exits the
// process: returns cudaSuccess or the error (a launch-configuration error, or
// cudaErrorUnknown when cuBLAS fails); matmul_last_error() describes the last
// failure on this host thread.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace aegis {

namespace detail {
struct Winner {
  int M, N, K;
  const char* impl;  // aegis::gemm registry name
};
}  // namespace detail

cudaError_t matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream = nullptr);

// Text of the last matmul() failure on this host thread ("" if none yet).
const char* matmul_last_error();

// The impl matmul() would run for this shape on the current device, or
// "cublas". For logging and tests.
const char* matmul_choice(int M, int N, int K);

}  // namespace aegis
