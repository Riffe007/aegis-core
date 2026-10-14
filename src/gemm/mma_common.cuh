#pragma once
// Shared pieces of the tensor-core tiers (3: mma, 4: mma_pipe): inline-PTX
// wrappers, the smem tile layout, global->smem loaders, the per-BK warp compute
// and the epilogue. Kernels themselves live in mma.cu / mma_pipe.cu.
//
// PTX ISA 9.4 (docs.nvidia.com/cuda/parallel-thread-execution), Target ISA Notes:
//   9.7.16.5.15 ldmatrix (.m8n8 .x4 .b16, incl. .trans): "Requires sm_75 or
//     higher." (The sm_100a/sm_120a-family restriction there applies only to
//     the .m16n16/.m8n16 shapes, which we do not use.)
//   9.7.16.5.14 mma: ".f16 floating point type mma operation with .m16n8k16
//     shape requires sm_80 or higher." (sm_120a/f restrictions apply only to
//     .e3m2/.e2m3/.e2m1 types and .kind/.block_scale, which we do not use.)
//   9.7.10.28.3.1-3 cp.async, cp.async.commit_group, cp.async.wait_group:
//     "Requires sm_80 or higher."
// None carries an arch-specific (a) or family (f) suffix, so all are valid on
// sm_87 (Orin) and sm_121 (GB10). Checked by compiling for both with ptxas.
#include <cstdint>

#include "aegis/common.cuh"

namespace aegis::gemm::mma_detail {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Four 8x8 b16 matrices; lane l supplies the row address of matrix l/8.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// D = A*B + D, A 16x16 row (4 regs of f16x2), B 16x8 col (2 regs), D 16x8 f32.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16-byte async copy global->shared. src_bytes = 0 zero-fills the destination
// without reading global memory (used for out-of-bounds vectors).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Block tile BMxBN, K-step BK, WARPS_M x WARPS_N warps each computing
// (BM/WARPS_M) x (BN/WARPS_N) as MI x NI m16n8 MMA tiles.
//
// Shared memory: row-major As[BM][LDA], Bs[BK][LDB] in f16, with each row padded
// by 8 halves (16 B). ldmatrix serves 8 row addresses of 16 B per phase; with a
// row stride of (BK/8 + 1) 16-byte chunks, an odd number, 8 consecutive rows
// land in 8 distinct chunk positions of a 128 B bank line: conflict-free. The
// same holds for the 16-byte stores from global (sync or cp.async).
template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_>
struct Tile {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
  static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
  static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  static constexpr int MI = WM / 16, NI = WN / 8;
  static constexpr int PAD = 8;
  static constexpr int LDA = BK + PAD, LDB = BN + PAD;
  static constexpr int A_ELEMS = BM * LDA, B_ELEMS = BK * LDB;
  static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
  static constexpr size_t STAGE_BYTES = size_t(STAGE_ELEMS) * sizeof(__half);

  static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
  static_assert(WM % 16 == 0, "warp M must be a multiple of the m16 MMA tile");
  static_assert(WN % 16 == 0, "warp N must be a multiple of 16 (ldmatrix.x4.trans loads 2 n8 tiles)");
  static_assert(BK % 16 == 0, "BK must be a multiple of the k16 MMA step");
  static_assert((LDA / 8) % 2 == 1 && (LDB / 8) % 2 == 1, "padding must give an odd chunk stride");
};

// Synchronous global->smem tile load. vec: K and N are multiples of 8 and A, B
// are 16-byte aligned, so each 8-half vector is wholly in or out of bounds.
template <class T>
__device__ __forceinline__ void load_tile_sync(__half* As, __half* Bs, const __half* A,
                                               const __half* B, int M, int N, int K,
                                               int m0, int n0, int k0, bool vec) {
  const int tid = threadIdx.x;
  if (vec) {
#pragma unroll
    for (int v = tid; v < T::BM * (T::BK / 8); v += T::THREADS) {
      const int r = v / (T::BK / 8), c = (v % (T::BK / 8)) * 8;
      uint4 x = make_uint4(0, 0, 0, 0);
      if (m0 + r < M && k0 + c < K)
        x = *reinterpret_cast<const uint4*>(A + size_t(m0 + r) * K + k0 + c);
      *reinterpret_cast<uint4*>(As + r * T::LDA + c) = x;
    }
#pragma unroll
    for (int v = tid; v < T::BK * (T::BN / 8); v += T::THREADS) {
      const int r = v / (T::BN / 8), c = (v % (T::BN / 8)) * 8;
      uint4 x = make_uint4(0, 0, 0, 0);
      if (k0 + r < K && n0 + c < N)
        x = *reinterpret_cast<const uint4*>(B + size_t(k0 + r) * N + n0 + c);
      *reinterpret_cast<uint4*>(Bs + r * T::LDB + c) = x;
    }
  } else {
    const __half zero = __float2half(0.f);
    for (int i = tid; i < T::BM * T::BK; i += T::THREADS) {
      const int r = i / T::BK, c = i % T::BK;
      As[r * T::LDA + c] = (m0 + r < M && k0 + c < K) ? A[size_t(m0 + r) * K + k0 + c] : zero;
    }
    for (int i = tid; i < T::BK * T::BN; i += T::THREADS) {
      const int r = i / T::BN, c = i % T::BN;
      Bs[r * T::LDB + c] = (k0 + r < K && n0 + c < N) ? B[size_t(k0 + r) * N + n0 + c] : zero;
    }
  }
}

// Asynchronous version of the vec path (cp.async.cg, 16 B per copy, zero-fill
// out of bounds). Caller commits and waits.
template <class T>
__device__ __forceinline__ void load_tile_async(__half* As, __half* Bs, const __half* A,
                                                const __half* B, int M, int N, int K,
                                                int m0, int n0, int k0) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int v = tid; v < T::BM * (T::BK / 8); v += T::THREADS) {
    const int r = v / (T::BK / 8), c = (v % (T::BK / 8)) * 8;
    const bool in = m0 + r < M && k0 + c < K;
    cp_async_16(smem_addr(As + r * T::LDA + c),
                in ? A + size_t(m0 + r) * K + k0 + c : A, in ? 16 : 0);
  }
#pragma unroll
  for (int v = tid; v < T::BK * (T::BN / 8); v += T::THREADS) {
    const int r = v / (T::BN / 8), c = (v % (T::BN / 8)) * 8;
    const bool in = k0 + r < K && n0 + c < N;
    cp_async_16(smem_addr(Bs + r * T::LDB + c),
                in ? B + size_t(k0 + r) * N + n0 + c : B, in ? 16 : 0);
  }
}

// One BK slab: per k16 step, ldmatrix A (MI x4) and B (NI/2 x4.trans) fragments,
// then MI*NI MMAs into acc.
template <class T>
__device__ __forceinline__ void compute_tile(const __half* As, const __half* Bs,
                                             float (&acc)[T::MI][T::NI][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wm = (warp / T::WARPS_N) * T::WM, wn = (warp % T::WARPS_N) * T::WN;
#pragma unroll
  for (int kk = 0; kk < T::BK; kk += 16) {
    uint32_t a[T::MI][4];
    uint32_t b[T::NI][2];
    // A m16k16: matrices (rows 0-7,k0-7) (rows 8-15,k0-7) (rows 0-7,k8-15)
    // (rows 8-15,k8-15) = a0..a3; lane l addresses row l%16, k-offset (l/16)*8.
#pragma unroll
    for (int mi = 0; mi < T::MI; ++mi)
      ldmatrix_x4(a[mi], smem_addr(As + (wm + mi * 16 + lane % 16) * T::LDA + kk + (lane / 16) * 8));
    // B k16n16 from row-major [k][n] with .trans: matrices (k0-7,n0-7)
    // (k8-15,n0-7) (k0-7,n8-15) (k8-15,n8-15) = {b0,b1} of n-tile 0, then 1.
#pragma unroll
    for (int nj = 0; nj < T::NI / 2; ++nj) {
      uint32_t r[4];
      ldmatrix_x4_trans(r, smem_addr(Bs + (kk + lane % 8 + ((lane / 8) % 2) * 8) * T::LDB +
                                     wn + nj * 16 + (lane / 16) * 8));
      b[2 * nj][0] = r[0]; b[2 * nj][1] = r[1];
      b[2 * nj + 1][0] = r[2]; b[2 * nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < T::MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < T::NI; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
  }
}

// m16n8 f32 accumulator: c0,c1 at (lane/4, 2*(lane%4) + {0,1}); c2,c3 at row+8.
template <class T>
__device__ __forceinline__ void store_tile(__half* C, int M, int N, int m0, int n0,
                                           const float (&acc)[T::MI][T::NI][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wm = m0 + (warp / T::WARPS_N) * T::WM, wn = n0 + (warp % T::WARPS_N) * T::WN;
  const bool pairs = N % 2 == 0;  // half2 stores stay 4-byte aligned
#pragma unroll
  for (int mi = 0; mi < T::MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < T::NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = wm + mi * 16 + lane / 4 + h * 8;
        const int col = wn + ni * 8 + (lane % 4) * 2;
        if (row >= M) continue;
        const float x = acc[mi][ni][2 * h], y = acc[mi][ni][2 * h + 1];
        __half* p = C + size_t(row) * N + col;
        if (pairs && col + 1 < N) {
          *reinterpret_cast<__half2*>(p) = __floats2half2_rn(x, y);
        } else {
          if (col < N) p[0] = __float2half(x);
          if (col + 1 < N) p[1] = __float2half(y);
        }
      }
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace aegis::gemm::mma_detail
