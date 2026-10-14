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
#include "gemm_params.cuh"

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


// 16-byte async copy global->shared, unpredicated (interior fast path).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// Predicated form: copies src_bytes (0..16) and zero-fills the rest of the 16
// bytes; src_bytes = 0 reads nothing (out-of-bounds chunks, K tail).
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst, const void* src, int src_bytes) {
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
// (BM/WARPS_M) x (BN/WARPS_N) as MI x NI m16n8 MMA tiles, for operand layouts
// LA (A, M x K) and LB (B, K x N).
//
// Shared memory keeps each operand in its global layout: the operand's
// contiguous dimension is the smem row, so every global 16 B chunk lands in
// one smem chunk (cp.async cannot transpose). A row-major: As[m][k]; A
// column-major: As[k][m]; B row-major: Bs[k][n]; B column-major: Bs[n][k].
// ldmatrix then picks .trans or not per layout (compute_tile). Each smem row
// is padded by 8 halves (16 B): ldmatrix serves 8 row addresses of 16 B per
// phase, and with an odd row stride in 16-byte chunks 8 consecutive rows land
// in 8 distinct chunk positions of a 128 B bank line: conflict-free. The same
// holds for the 16-byte stores from global.
template <int ROWS_, int COLS_>
struct SmemOperand {
  static constexpr int ROWS = ROWS_, COLS = COLS_;
  static constexpr int PAD = 8;
  static constexpr int LD = COLS + PAD;
  static constexpr int ELEMS = ROWS * LD;
  static_assert(COLS % 8 == 0, "16-byte chunks");
  static_assert((LD / 8) % 2 == 1, "padding must give an odd chunk stride");
};

template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_, Layout LA_ = Layout::RowMajor,
          Layout LB_ = Layout::RowMajor>
struct Tile {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
  static constexpr Layout LA = LA_, LB = LB_;
  static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
  static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  static constexpr int MI = WM / 16, NI = WN / 8;
  // k contiguous in global (and smem): A row-major, B column-major.
  static constexpr bool A_K_CONTIG = LA == Layout::RowMajor;
  static constexpr bool B_K_CONTIG = LB == Layout::ColMajor;
  using SA = SmemOperand<A_K_CONTIG ? BM : BK, A_K_CONTIG ? BK : BM>;
  using SB = SmemOperand<B_K_CONTIG ? BN : BK, B_K_CONTIG ? BK : BN>;
  static constexpr int LDA = SA::LD, LDB = SB::LD;
  static constexpr int A_ELEMS = SA::ELEMS, B_ELEMS = SB::ELEMS;
  static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
  static constexpr size_t STAGE_BYTES = size_t(STAGE_ELEMS) * sizeof(__half);

  static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
  static_assert(WM % 16 == 0, "warp M must be a multiple of the m16 MMA tile");
  static_assert(WN % 16 == 0, "warp N must be a multiple of 16 (one ldmatrix.x4 loads 2 n8 tiles)");
  static_assert(BK % 16 == 0, "BK must be a multiple of the k16 MMA step");
};

// One operand's global->smem copy plan for this thread, built once per CTA
// before the main loop (Production-bar predication design: no per-load mask
// or address arithmetic beyond a pointer add per k-tile).
//
// The smem tile is ROWS x COLS (COLS contiguous, 8-half chunks). Thread tid
// copies chunks tid + i * THREADS, i < ITERS: all in the same chunk column
// col0, rows row0 + i * ROW_STEP. K_IS_ROW: the tile's rows are the k
// dimension (A column-major, B row-major); else its columns are (A row-major,
// B column-major). The other dimension is M (A) or N (B), bounded by mn_limit.
template <int ROWS, int COLS, int LD_S, int THREADS, bool K_IS_ROW>
struct OperandPlan {
  static constexpr int CHUNKS = COLS / 8;
  static constexpr int ROW_STEP = THREADS / CHUNKS;
  static constexpr int ITERS = ROWS * CHUNKS / THREADS;
  static constexpr int K_TILE = K_IS_ROW ? ROWS : COLS;
  static_assert(THREADS % CHUNKS == 0 && (ROWS * CHUNKS) % THREADS == 0,
                "every thread copies the same number of chunks, one chunk column");
  static_assert(ITERS <= 32, "row mask is 32 bits");

  const __half* base;    // the operand's matrix (a valid address for 0-byte copies)
  const __half* g;       // this thread's chunk 0 of the current k-tile
  long long row_step;    // ROW_STEP rows, in elements
  long long k_advance;   // one k-tile, in elements
  uint32_t s_off;        // chunk 0, bytes from the operand's smem tile start
  int row0, col0;        // chunk 0, tile coordinates (elements)
  uint32_t mn_rows_ok;   // !K_IS_ROW: bit i = row of chunk i < mn_limit
  int mn_col_bytes;      // K_IS_ROW: valid bytes of the chunk column (0..16)

  __device__ __forceinline__ void init(const __half* X, long long ld, int mn0, int mn_limit) {
    const int tid = threadIdx.x;
    row0 = tid / CHUNKS;
    col0 = (tid % CHUNKS) * 8;
    base = X;
    row_step = (long long)ROW_STEP * ld;
    s_off = uint32_t(row0 * LD_S + col0) * sizeof(__half);
    if constexpr (K_IS_ROW) {  // rows = k (start 0), cols = mn
      g = X + (long long)row0 * ld + mn0 + col0;
      k_advance = (long long)K_TILE * ld;
      mn_col_bytes = max(0, min(8, mn_limit - (mn0 + col0))) * 2;
      mn_rows_ok = ~0u;
    } else {  // rows = mn, cols = k (start 0)
      g = X + (long long)(mn0 + row0) * ld + col0;
      k_advance = K_TILE;
      uint32_t ok = 0;
#pragma unroll
      for (int i = 0; i < ITERS; ++i)
        if (mn0 + row0 + i * ROW_STEP < mn_limit) ok |= 1u << i;
      mn_rows_ok = ok;
      mn_col_bytes = 16;
    }
  }

  // Valid bytes of chunk i (k0: first k of this tile). MASK_MN: boundary CTA;
  // MASK_K: the partial last k-tile.
  template <bool MASK_MN, bool MASK_K>
  __device__ __forceinline__ int chunk_bytes(int i, int k0, int K) const {
    int bytes = 16;
    if constexpr (K_IS_ROW) {
      if constexpr (MASK_MN) bytes = mn_col_bytes;
      if constexpr (MASK_K)
        if (k0 + row0 + i * ROW_STEP >= K) bytes = 0;
    } else {
      if constexpr (MASK_MN)
        if (!((mn_rows_ok >> i) & 1u)) bytes = 0;
      if constexpr (MASK_K) bytes = min(bytes, max(0, min(8, K - (k0 + col0))) * 2);
    }
    return bytes;
  }

  // Asynchronous copy of the current k-tile into the smem tile at s_tile.
  template <bool MASK_MN, bool MASK_K>
  __device__ __forceinline__ void load_async(uint32_t s_tile, int k0, int K) const {
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      const uint32_t dst = s_tile + s_off + uint32_t(i * ROW_STEP * LD_S * sizeof(__half));
      const __half* src = g + i * row_step;
      if constexpr (!MASK_MN && !MASK_K) {
        cp_async_16(dst, src);
      } else {
        const int bytes = chunk_bytes<MASK_MN, MASK_K>(i, k0, K);
        cp_async_16_zfill(dst, bytes > 0 ? src : base, bytes);
      }
    }
  }

  // Synchronous version (tier 3): 16 B vectors, element-wise for partial chunks.
  template <bool MASK_MN, bool MASK_K>
  __device__ __forceinline__ void load_sync(__half* s_tile, int k0, int K) const {
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      __half* dst = s_tile + s_off / sizeof(__half) + i * ROW_STEP * LD_S;
      const __half* src = g + i * row_step;
      const int bytes = (!MASK_MN && !MASK_K) ? 16 : chunk_bytes<MASK_MN, MASK_K>(i, k0, K);
      if (bytes == 16) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      } else {
        const __half zero = __float2half(0.f);
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[j] = j * 2 < bytes ? src[j] : zero;
      }
    }
  }

  __device__ __forceinline__ void advance() { g += k_advance; }
};

template <class T>
using PlanA = OperandPlan<T::SA::ROWS, T::SA::COLS, T::LDA, T::THREADS, !T::A_K_CONTIG>;
template <class T>
using PlanB = OperandPlan<T::SB::ROWS, T::SB::COLS, T::LDB, T::THREADS, !T::B_K_CONTIG>;

// Scalar global->smem copy of one operand tile, any alignment, every element
// bounds-checked: the fallback for misaligned pointers / leading dimensions.
// Tile rows/cols start at (r0, c0) in the operand's storage order.
template <int ROWS, int COLS, int LD_S, int THREADS>
__device__ __forceinline__ void load_operand_scalar(__half* s, const __half* X, long long ld, int r0,
                                                    int r_limit, int c0, int c_limit) {
  const __half zero = __float2half(0.f);
  for (int i = threadIdx.x; i < ROWS * COLS; i += THREADS) {
    const int r = i / COLS, c = i % COLS;
    s[r * LD_S + c] = (r0 + r < r_limit && c0 + c < c_limit) ? X[(long long)(r0 + r) * ld + c0 + c] : zero;
  }
}

// Both operands of k-tile k0, scalar.
template <class T>
__device__ __forceinline__ void load_tile_scalar(__half* As, __half* Bs, const __half* A, const __half* B,
                                                 const KParams& p, int m0, int n0, int k0) {
  if constexpr (T::A_K_CONTIG)
    load_operand_scalar<T::SA::ROWS, T::SA::COLS, T::LDA, T::THREADS>(As, A, p.lda, m0, p.M, k0, p.K);
  else
    load_operand_scalar<T::SA::ROWS, T::SA::COLS, T::LDA, T::THREADS>(As, A, p.lda, k0, p.K, m0, p.M);
  if constexpr (T::B_K_CONTIG)
    load_operand_scalar<T::SB::ROWS, T::SB::COLS, T::LDB, T::THREADS>(Bs, B, p.ldb, n0, p.N, k0, p.K);
  else
    load_operand_scalar<T::SB::ROWS, T::SB::COLS, T::LDB, T::THREADS>(Bs, B, p.ldb, k0, p.K, n0, p.N);
}

// One BK slab: per k16 step, ldmatrix A (MI x4) and B (NI/2 x4) fragments,
// then MI*NI MMAs into acc. Fragment layouts (m16n8k16 .row.col): A matrices
// (m0-7,k0-7) (m8-15,k0-7) (m0-7,k8-15) (m8-15,k8-15) = a0..a3; B per n8 tile
// (k0-7) (k8-15) = b0, b1. Stored k-contiguous, a matrix loads as is; stored
// transposed, with .trans.
template <class T>
__device__ __forceinline__ void compute_tile(const __half* As, const __half* Bs,
                                             float (&acc)[T::MI][T::NI][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wm = (warp / T::WARPS_N) * T::WM, wn = (warp % T::WARPS_N) * T::WN;
  const int j = lane / 8, l8 = lane % 8;  // ldmatrix: lane supplies row l8 of matrix j
#pragma unroll
  for (int kk = 0; kk < T::BK; kk += 16) {
    uint32_t a[T::MI][4];
    uint32_t b[T::NI][2];
#pragma unroll
    for (int mi = 0; mi < T::MI; ++mi) {
      if constexpr (T::A_K_CONTIG)  // As[m][k]: matrix j = rows m + (j%2)*8, k + (j/2)*8
        ldmatrix_x4(a[mi], smem_addr(As + (wm + mi * 16 + lane % 16) * T::LDA + kk + (lane / 16) * 8));
      else  // As[k][m]: rows k + (j/2)*8, cols m + (j%2)*8, transposed
        ldmatrix_x4_trans(a[mi], smem_addr(As + (kk + l8 + (j / 2) * 8) * T::LDA + wm + mi * 16 + (j % 2) * 8));
    }
#pragma unroll
    for (int nj = 0; nj < T::NI / 2; ++nj) {
      uint32_t r[4];
      if constexpr (T::B_K_CONTIG)  // Bs[n][k]: matrix j = rows n + (j/2)*8, k + (j%2)*8
        ldmatrix_x4(r, smem_addr(Bs + (wn + nj * 16 + (j / 2) * 8 + l8) * T::LDB + kk + (j % 2) * 8));
      else  // Bs[k][n]: rows k + (j%2)*8, cols n + (j/2)*8, transposed
        ldmatrix_x4_trans(r, smem_addr(Bs + (kk + l8 + (j % 2) * 8) * T::LDB + wn + nj * 16 + (j / 2) * 8));
      b[2 * nj][0] = r[0]; b[2 * nj][1] = r[1];
      b[2 * nj + 1][0] = r[2]; b[2 * nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < T::MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < T::NI; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
  }
}

// Epilogue: C = alpha * acc + beta * C (C row-major, ldc). beta == 0 does not
// read C. MASK: boundary CTA (row/col bounds); pairs: half2 stores are aligned.
// m16n8 f32 accumulator: c0,c1 at (lane/4, 2*(lane%4) + {0,1}); c2,c3 at row+8.
template <class T, bool MASK>
__device__ __forceinline__ void store_tile(__half* C, const KParams& p, int m0, int n0, bool pairs,
                                           const float (&acc)[T::MI][T::NI][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wm = m0 + (warp / T::WARPS_N) * T::WM, wn = n0 + (warp % T::WARPS_N) * T::WN;
  const float alpha = p.alpha, beta = p.beta;
  const bool read_c = beta != 0.f;
#pragma unroll
  for (int mi = 0; mi < T::MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < T::NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = wm + mi * 16 + lane / 4 + h * 8;
        const int col = wn + ni * 8 + (lane % 4) * 2;
        if (MASK && row >= p.M) continue;
        float x = alpha * acc[mi][ni][2 * h], y = alpha * acc[mi][ni][2 * h + 1];
        __half* q = C + (long long)row * p.ldc + col;
        if (pairs && (!MASK || col + 1 < p.N)) {
          if (read_c) {
            const float2 old = __half22float2(*reinterpret_cast<const __half2*>(q));
            x += beta * old.x;
            y += beta * old.y;
          }
          *reinterpret_cast<__half2*>(q) = __floats2half2_rn(x, y);
        } else {
          if (!MASK || col < p.N) q[0] = __float2half(read_c ? x + beta * __half2float(q[0]) : x);
          if (!MASK || col + 1 < p.N) q[1] = __float2half(read_c ? y + beta * __half2float(q[1]) : y);
        }
      }
}

}  // namespace aegis::gemm::mma_detail
