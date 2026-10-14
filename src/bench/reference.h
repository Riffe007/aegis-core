#pragma once
// GEMM reference and tolerance (Production bar, Correctness): an FP64-
// accumulate CPU reference for small problems, cuBLASLt for large ones, and the
// K-scaled tolerance
//
//   tol = c * sqrt(K) * eps_fp16 * max|A| * max|B| * |alpha|  +  eps_fp16 * |beta| * max|C0|
//
// eps_fp16 = 2^-10 (FP16 machine epsilon). For alpha = 1, beta = 0 this is the
// bar's c * sqrt(K) * eps_fp16 * max|A| * max|B|; the alpha factor and the beta
// term extend it to C = alpha*A*B + beta*C (C0: C before the call). c is
// kToleranceC (documented in reference.cu with its measurement). The bound
// assumes zero-mean inputs (random signs), where |C| grows like sqrt(K); the
// fuzzer generates such data. When the reference is cuBLASLt (itself rounded
// to FP16), the comparison allows 2 * tol.
//
// NaN/Inf: a reference NaN must be NaN in the result, a reference +-Inf (or a
// finite reference beyond the FP16 range) the same Inf; a finite reference
// must be finite and within tol.
//
// Benchmark/test code: aborts on CUDA/cuBLAS errors.
#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

#include "aegis/gemm.h"

namespace aegis::bench {

inline constexpr double kEpsFp16 = 9.765625e-4;  // 2^-10
inline constexpr double kToleranceC = 2.0;       // see reference.cu for the measurement

// Host description of one GEMM: the same fields as gemm::GemmParams, with host
// arrays. A, B, C0 hold every batch (index with the strides).
struct HostGemm {
  gemm::GemmParams p;  // pointers unused; sizes, layouts, ld, strides, alpha, beta
  std::vector<__half> A, B, C0;
};

// Number of elements a batched operand spans (0 if any size is 0).
size_t span_a(const gemm::GemmParams& p);
size_t span_b(const gemm::GemmParams& p);
size_t span_c(const gemm::GemmParams& p);

// FP64-accumulate reference: out[b * M * N + i * N + j] = alpha * sum_k A*B + beta * C0
// (beta == 0: C0 not read), as double.
void cpu_reference(const HostGemm& g, std::vector<double>& out);

// cuBLASLt reference on device: Cdev (laid out like C, holding C0) becomes
// alpha*A*B + beta*C0. Adev/Bdev/Cdev are the batch-0 pointers.
class LtReference {
 public:
  LtReference();
  ~LtReference();
  LtReference(const LtReference&) = delete;
  LtReference& operator=(const LtReference&) = delete;
  void run(const gemm::GemmParams& p_dev, cudaStream_t stream);

 private:
  cublasLtHandle_t lt_ = nullptr;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 0;
};

struct Tolerance {
  double max_a = 0, max_b = 0, max_c0 = 0;  // finite maxima
  double tol = 0;
};
Tolerance tolerance(const HostGemm& g, double c = kToleranceC);

struct Comparison {
  bool pass = true;
  long long checked = 0, nan_mismatch = 0, inf_mismatch = 0, out_of_tol = 0;
  double max_err = 0, max_ratio = 0;  // finite elements: |err|, |err| / tol
  std::string first_failure;          // "batch b row i col j: got x, ref y"
};
// out: the result C (every batch, ldc layout); ref: double per element, packed
// b * M * N + i * N + j; allowed = tol (CPU reference) or 2 * tol (cuBLASLt).
// Also checks that C's padding columns (N..ldc-1) are unchanged from C0.
Comparison compare(const HostGemm& g, const std::vector<__half>& out, const std::vector<double>& ref,
                   double allowed);

}  // namespace aegis::bench
