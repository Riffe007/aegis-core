#pragma once
// Device-side GEMM parameters and the host checks shared by every production
// tier (naive, regblock, mma, mma_pipe). See GemmParams in aegis/gemm.h.
#include <cstdint>

#include "aegis/common.cuh"
#include "aegis/gemm.h"

namespace aegis::gemm {

// Kernel argument: GemmParams flattened (passed by value).
struct KParams {
  const __half* A;
  const __half* B;
  __half* C;
  int M, N, K;
  long long lda, ldb, ldc;
  long long sa, sb, sc;  // batch strides (elements)
  float alpha, beta;
};

inline KParams to_kparams(const GemmParams& p) {
  return KParams{p.A, p.B, p.C, p.M, p.N, p.K, p.lda, p.ldb, p.ldc,
                 p.stride_a, p.stride_b, p.stride_c, p.alpha, p.beta};
}

// Element offset of (r, c) in a matrix stored with layout L and leading dim ld.
template <Layout L>
__host__ __device__ __forceinline__ long long elem(long long r, long long c, long long ld) {
  return L == Layout::RowMajor ? r * ld + c : r + c * ld;
}

inline bool aligned_to(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) & (bytes - 1)) == 0;
}

// 16-byte global vectors (8 halves along the contiguous dimension) are legal
// for every chunk of every batch: base pointers 16 B aligned, leading
// dimensions and batch strides multiples of 8 elements. Otherwise the scalar
// path (any alignment) is taken.
inline bool vec16_ok(const GemmParams& p) {
  const bool batched = p.batch > 1;
  return aligned_to(p.A, 16) && aligned_to(p.B, 16) && p.lda % 8 == 0 && p.ldb % 8 == 0 &&
         (!batched || (p.stride_a % 8 == 0 && p.stride_b % 8 == 0));
}

// half2 stores of C at even columns stay 4-byte aligned.
inline bool c_pairs_ok(const GemmParams& p) {
  return aligned_to(p.C, 4) && p.ldc % 2 == 0 && (p.batch <= 1 || p.stride_c % 2 == 0);
}

}  // namespace aegis::gemm
