// Tier 2: register-blocked GEMM. Each block computes a BMxBN tile of C; each
// thread computes a TMxTN micro-tile held in registers.
//
// Versus tier 1 (one output per thread, two smem loads per FMA), each k-step a
// thread loads TM values of A and TN values of B from smem and issues TM*TN
// FMAs, so smem traffic per FMA drops by ~TM*TN/(TM+TN) (4x at 8x8). Targets the
// shared-memory bandwidth bound that limits tiled. Still CUDA-core FP32 math.
//
// Layout: A is staged transposed (As[k][m]) so that a thread's TM values of one
// A column are contiguous and load as float4, like B's TN values. A thread's TM
// rows are split into TM/4 groups of 4 spaced BM/(TM/4) apart (same for TN), so a
// warp's 16-byte smem reads cover contiguous 256 B spans: conflict-free.
//
// General GEMM (GemmParams: layouts, ld, alpha/beta, strided batch). Global
// loads: 16-byte vectors when pointers, ld and strides allow (vec16_ok); interior
// CTAs load whole chunks unchecked, boundary CTAs and the partial last k-tile
// check chunk bounds (partial chunks element-wise); misaligned operands take a
// scalar, element-wise bounds-checked path.
#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "gemm_params.cuh"

namespace aegis::gemm {
namespace {

// 8 halves of global memory from src (valid of them; the rest 0) as floats.
// full: one 16-byte load.
__device__ __forceinline__ void load8(float (&v)[8], const __half* src, int valid) {
  if (valid == 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(h[j]);
      v[2 * j] = f.x;
      v[2 * j + 1] = f.y;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = j < valid ? __half2float(src[j]) : 0.f;
  }
}

__device__ __forceinline__ int clamp8(int n) { return n < 0 ? 0 : (n > 8 ? 8 : n); }

// Vectorized tile loads (vec path). CHECK: bounds-check chunks (boundary CTA
// or partial last k-tile); otherwise every chunk is wholly in bounds.
template <int BM, int BN, int BK, int THREADS, Layout LA, Layout LB, bool CHECK>
__device__ __forceinline__ void load_tiles_vec(float (&As)[BK][BM], float (&Bs)[BK][BN], const __half* A,
                                               const __half* B, const KParams& p, int m0, int n0, int k0) {
  const int tid = threadIdx.x;
  if constexpr (LA == Layout::RowMajor) {  // chunk = (m, 8 k); consecutive threads, consecutive m
    for (int i = tid; i < BM * (BK / 8); i += THREADS) {
      const int r = i % BM, c = (i / BM) * 8;
      const int valid = CHECK ? (m0 + r < p.M ? clamp8(p.K - (k0 + c)) : 0) : 8;
      float v[8];
      load8(v, A + (long long)(m0 + r) * p.lda + k0 + c, valid);
#pragma unroll
      for (int j = 0; j < 8; ++j) As[c + j][r] = v[j];
    }
  } else {  // chunk = (k, 8 m)
    for (int i = tid; i < BK * (BM / 8); i += THREADS) {
      const int r = i / (BM / 8), c = (i % (BM / 8)) * 8;
      const int valid = CHECK ? (k0 + r < p.K ? clamp8(p.M - (m0 + c)) : 0) : 8;
      float v[8];
      load8(v, A + (long long)(k0 + r) * p.lda + m0 + c, valid);
      *reinterpret_cast<float4*>(&As[r][c]) = make_float4(v[0], v[1], v[2], v[3]);
      *reinterpret_cast<float4*>(&As[r][c + 4]) = make_float4(v[4], v[5], v[6], v[7]);
    }
  }
  if constexpr (LB == Layout::RowMajor) {  // chunk = (k, 8 n)
    for (int i = tid; i < BK * (BN / 8); i += THREADS) {
      const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
      const int valid = CHECK ? (k0 + r < p.K ? clamp8(p.N - (n0 + c)) : 0) : 8;
      float v[8];
      load8(v, B + (long long)(k0 + r) * p.ldb + n0 + c, valid);
      *reinterpret_cast<float4*>(&Bs[r][c]) = make_float4(v[0], v[1], v[2], v[3]);
      *reinterpret_cast<float4*>(&Bs[r][c + 4]) = make_float4(v[4], v[5], v[6], v[7]);
    }
  } else {  // chunk = (n, 8 k); consecutive threads, consecutive n
    for (int i = tid; i < BN * (BK / 8); i += THREADS) {
      const int r = i % BN, c = (i / BN) * 8;
      const int valid = CHECK ? (n0 + r < p.N ? clamp8(p.K - (k0 + c)) : 0) : 8;
      float v[8];
      load8(v, B + (long long)(n0 + r) * p.ldb + k0 + c, valid);
#pragma unroll
      for (int j = 0; j < 8; ++j) Bs[c + j][r] = v[j];
    }
  }
}

// Scalar tile loads, any alignment, every element bounds-checked.
template <int BM, int BN, int BK, int THREADS, Layout LA, Layout LB>
__device__ __forceinline__ void load_tiles_scalar(float (&As)[BK][BM], float (&Bs)[BK][BN], const __half* A,
                                                  const __half* B, const KParams& p, int m0, int n0, int k0) {
  for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
    const int m = i % BM, k = i / BM;
    As[k][m] = (m0 + m < p.M && k0 + k < p.K) ? __half2float(A[elem<LA>(m0 + m, k0 + k, p.lda)]) : 0.f;
  }
  for (int i = threadIdx.x; i < BK * BN; i += THREADS) {
    const int k = i / BN, n = i % BN;
    Bs[k][n] = (k0 + k < p.K && n0 + n < p.N) ? __half2float(B[elem<LB>(k0 + k, n0 + n, p.ldb)]) : 0.f;
  }
}

template <int BM, int BN, int BK, int TM, int TN, Layout LA, Layout LB>
// Minimum 1 block/SM: without it ptxas capped <64,64,8,4,4,Col,Row> at 64
// registers on sm_87 and spilled 16 bytes (sm87_gate). Regblock is not a tuned
// winner; this changes only its register allocation.
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 1)
gemm_regblock_kernel(KParams p, bool vec) {
  constexpr int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
  constexpr int MC = TM / 4, NC = TN / 4;            // float4 groups per thread
  constexpr int MSTRIDE = BM / MC, NSTRIDE = BN / NC;  // spacing between groups
  static_assert(TM % 4 == 0 && TN % 4 == 0, "micro-tile must be float4 groups");
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide evenly");
  static_assert(BK % 8 == 0 && BN % 8 == 0 && BM % 8 == 0, "16-byte global vectors");

  __shared__ __align__(16) float As[BK][BM];  // transposed: As[k][m]
  __shared__ __align__(16) float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const long long z = blockIdx.z;
  const __half* A = p.A + z * p.sa;
  const __half* B = p.B + z * p.sb;
  __half* C = p.C + z * p.sc;
  // Interior CTA with whole k-tiles: no bounds checks on the vec path.
  const bool interior = m0 + BM <= p.M && n0 + BN <= p.N;
  const int k_full = (p.K / BK) * BK;

  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < p.K; k0 += BK) {
    if (!vec)
      load_tiles_scalar<BM, BN, BK, THREADS, LA, LB>(As, Bs, A, B, p, m0, n0, k0);
    else if (interior && k0 < k_full)
      load_tiles_vec<BM, BN, BK, THREADS, LA, LB, false>(As, Bs, A, B, p, m0, n0, k0);
    else
      load_tiles_vec<BM, BN, BK, THREADS, LA, LB, true>(As, Bs, A, B, p, m0, n0, k0);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
#pragma unroll
      for (int g = 0; g < MC; ++g) {
        const float4 t = *reinterpret_cast<const float4*>(&As[k][g * MSTRIDE + ty * 4]);
        a[g * 4 + 0] = t.x; a[g * 4 + 1] = t.y; a[g * 4 + 2] = t.z; a[g * 4 + 3] = t.w;
      }
#pragma unroll
      for (int g = 0; g < NC; ++g) {
        const float4 t = *reinterpret_cast<const float4*>(&Bs[k][g * NSTRIDE + tx * 4]);
        b[g * 4 + 0] = t.x; b[g * 4 + 1] = t.y; b[g * 4 + 2] = t.z; b[g * 4 + 3] = t.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

  // Epilogue: C = alpha * acc + beta * C (beta == 0: C not read).
  const bool read_c = p.beta != 0.f;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + (i / 4) * MSTRIDE + ty * 4 + i % 4;
    if (!interior && row >= p.M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + (j / 4) * NSTRIDE + tx * 4 + j % 4;
      if (!interior && col >= p.N) continue;
      __half* q = C + (long long)row * p.ldc + col;
      float out = p.alpha == 0.f ? 0.f : p.alpha * acc[i][j];  // alpha == 0: A*B does not contribute
      if (read_c) out += p.beta * __half2float(*q);
      *q = __float2half(out);
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, Layout LA, Layout LB>
cudaError_t launch_layout(const GemmParams& gp, cudaStream_t stream) {
  const dim3 block((BM / TM) * (BN / TN));
  const dim3 grid(ceil_div(gp.N, BN), ceil_div(gp.M, BM), gp.batch);
  gemm_regblock_kernel<BM, BN, BK, TM, TN, LA, LB><<<grid, block, 0, stream>>>(to_kparams(gp), vec16_ok(gp));
  return cudaGetLastError();
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  if (p.layout_a == R && p.layout_b == R) return launch_layout<BM, BN, BK, TM, TN, R, R>(p, stream);
  if (p.layout_a == R) return launch_layout<BM, BN, BK, TM, TN, R, Cm>(p, stream);
  if (p.layout_b == R) return launch_layout<BM, BN, BK, TM, TN, Cm, R>(p, stream);
  return launch_layout<BM, BN, BK, TM, TN, Cm, Cm>(p, stream);
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  return run<BM, BN, BK, TM, TN>(row_major_params(A, B, C, M, N, K), stream);
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t attrs(cudaFuncAttributes* a) {
  return cudaFuncGetAttributes(a, gemm_regblock_kernel<BM, BN, BK, TM, TN, Layout::RowMajor, Layout::RowMajor>);
}

constexpr size_t kN = std::size(REGBLOCK_CONFIGS);

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{(
      "regblock_" + std::to_string(REGBLOCK_CONFIGS[I].BM) + "x" +
      std::to_string(REGBLOCK_CONFIGS[I].BN) + "x" + std::to_string(REGBLOCK_CONFIGS[I].BK) +
      "_" + std::to_string(REGBLOCK_CONFIGS[I].TM) + "x" + std::to_string(REGBLOCK_CONFIGS[I].TN))...};
  static const std::array<std::string, kN> configs{(
      "{\"BM\":" + std::to_string(REGBLOCK_CONFIGS[I].BM) +
      ",\"BN\":" + std::to_string(REGBLOCK_CONFIGS[I].BN) +
      ",\"BK\":" + std::to_string(REGBLOCK_CONFIGS[I].BK) +
      ",\"TM\":" + std::to_string(REGBLOCK_CONFIGS[I].TM) +
      ",\"TN\":" + std::to_string(REGBLOCK_CONFIGS[I].TN) +
      ",\"threads\":" +
      std::to_string((REGBLOCK_CONFIGS[I].BM / REGBLOCK_CONFIGS[I].TM) *
                     (REGBLOCK_CONFIGS[I].BN / REGBLOCK_CONFIGS[I].TN)) + "}")...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{
      names[I].c_str(), configs[I].c_str(),
      &launch<REGBLOCK_CONFIGS[I].BM, REGBLOCK_CONFIGS[I].BN, REGBLOCK_CONFIGS[I].BK,
              REGBLOCK_CONFIGS[I].TM, REGBLOCK_CONFIGS[I].TN>,
      &attrs<REGBLOCK_CONFIGS[I].BM, REGBLOCK_CONFIGS[I].BN, REGBLOCK_CONFIGS[I].BK,
             REGBLOCK_CONFIGS[I].TM, REGBLOCK_CONFIGS[I].TN>,
      &run<REGBLOCK_CONFIGS[I].BM, REGBLOCK_CONFIGS[I].BN, REGBLOCK_CONFIGS[I].BK,
           REGBLOCK_CONFIGS[I].TM, REGBLOCK_CONFIGS[I].TN>}...};
  return entries;
}

}  // namespace

std::span<const KernelEntry> regblock_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
