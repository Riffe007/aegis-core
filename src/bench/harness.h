#pragma once
// AEGIS timing protocol, shared by bench_gemm (one impl, one shape, one
// process) and aegis_tune (every registered config, many shapes, one process).
// Keep it identical across impls, or comparisons are invalid:
//   - inputs: uniform [-1, 1] from mt19937(seed), A then B, FP16
//   - one untimed run into C, synchronized
//   - --check: cuBLAS (FP16 in, FP32 compute, FP16 out) into Ref; pass if
//     max_abs_err / max|ref| <= kRelTol
//   - warmup launches are not timed
//   - before every timed rep, L2 is flushed by writing a buffer 2x the L2 size
//   - each rep is bracketed by its own cudaEvent pair; median/p10/p90 reported
//
// The harness is benchmark code: it aborts the process on CUDA/cuBLAS errors.
// The library path (aegis::matmul, the C ABI) never does.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "aegis/gemm.h"

namespace aegis::bench {

inline constexpr double kRelTol = 4e-3;  // CLAUDE.md rule 2

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T. Aborts on error.
void cublas_gemm(cublasHandle_t h, const __half* A, const __half* B, __half* C, int M, int N,
                 int K);

// One device-resident GEMM problem with generated inputs.
class Problem {
 public:
  Problem(int M, int N, int K, unsigned seed);
  ~Problem();
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;

  const int M, N, K;
  __half* A = nullptr;
  __half* B = nullptr;
  __half* C = nullptr;
  __half* Ref = nullptr;
};

struct Check {
  bool checked = false;
  double max_abs_err = -1, rel_err = -1;
  bool pass = true;
};

struct Timing {
  int reps = 0;
  double median_us = 0, p10_us = 0, p90_us = 0;
};

// Linear interpolation between order statistics.
double percentile(std::vector<float> v, double p);

class Harness {
 public:
  Harness();  // current device: properties, stream, cuBLAS handle, L2 flush buffer
  ~Harness();
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

  const cudaDeviceProp& prop() const { return prop_; }
  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublas() const { return cublas_; }

  // Launch `e` (nullptr = cuBLAS) on p into p.C, asynchronously. Aborts on error.
  void launch(const gemm::KernelEntry* e, Problem& p);
  // The untimed run, synchronized.
  void run_once(const gemm::KernelEntry* e, Problem& p);
  // cuBLAS into p.Ref, compared with p.C (call after run_once).
  Check check(Problem& p);
  Timing time(const gemm::KernelEntry* e, Problem& p, int warmup, int reps);

  // One JSON row (no trailing newline). `extra` (e.g. ,"k":v) is appended
  // inside the object after the standard fields; bench_gemm passes "".
  std::string json_row(const char* impl, const gemm::KernelEntry* e, const Problem& p,
                       const Check& c, const Timing& t, const std::string& extra = "") const;

 private:
  cudaDeviceProp prop_{};
  int runtime_ = 0, driver_ = 0;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  void* flush_ = nullptr;
  size_t flush_bytes_ = 0;
};

}  // namespace aegis::bench
