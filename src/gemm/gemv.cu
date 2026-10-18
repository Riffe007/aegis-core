// Decode GEMV, CUDA-core path (M <= 16, B = weights [K, N]; memory-bound on B).
//
// CTA: an MT x BN output tile (MT in {1, 2, 4, 8, 16}, the smallest >= M,
// chosen per call; larger M tiles by 16) and one split-K slice of K
// (splitk.cuh). Its THREADS threads form RG = THREADS / LPR row groups of
// LPR = BN / VEC lanes: lane l of a group owns columns n0 + l*VEC .. +VEC-1,
// group g takes k = g, g + RG, ... of the slice. Each k step: one VEC-half
// load of B (16 B for VEC 8, 8 B for VEC 4; a warp reads 512 / 256 contiguous
// bytes of a row, or LPR-lane pieces of several rows), converted to FP32, then
// MT x VEC FFMAs against A's column k read from smem (FP32, [k][MT], one
// float4 per four rows: a broadcast within a row group). U = 4 k steps are
// loaded before any is used, so 4 B loads per thread are in flight.
//
// A: the slice of A's MT rows is staged once in smem as FP32, in chunks of
// KCH rows (A_SMEM bytes); decode slices fit in one chunk. Any layout and
// alignment (element loads: A is at most 16 x K, read from L2).
// B: row-major with VEC-aligned base, ld and batch stride: vector loads; a
// partial last column group loads only its valid halves. Otherwise (column-
// major or misaligned B) element loads, same structure.
//
// End of the slice: the RG row groups' partial sums are added in group order
// 0..RG-1 through smem (passes of MP rows), giving the CTA's MT x BN tile in
// smem; splitk::finish_tile writes C (S = 1) or the partial and, in the last
// CTA of the tile, the ordered sum. Nothing is read past M, N or K: invalid
// columns load nothing and invalid k steps compute nothing (smem past the
// staged chunk is never read), so NaN outside the logical matrices cannot
// reach a product.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "gemm_params.cuh"
#include "splitk.cuh"

namespace aegis::gemm {
namespace {

constexpr int kASmem = 28 << 10;  // A chunk (FP32); + the MT x BN tile (<= 16 KiB) stays within 48 KiB static smem
constexpr int kU = 4;             // k steps in flight per thread

template <int VEC>
struct BVec;
template <>
struct BVec<8> { using T = uint4; };
template <>
struct BVec<4> { using T = uint2; };

template <int VEC>
__device__ __forceinline__ void to_float(const typename BVec<VEC>::T& v, float (&f)[VEC]) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int j = 0; j < VEC / 2; ++j) {
    const float2 x = __half22float2(h[j]);
    f[2 * j] = x.x;
    f[2 * j + 1] = x.y;
  }
}

struct GemvArgs {
  splitk::Args sk;
  int mtiles, ntiles;
  bool a_row, b_row;  // layouts (element path of B, and A staging)
};

template <int MT, int VEC, int BN, int THREADS, bool BVEC>
__global__ void __launch_bounds__(THREADS, 1) gemv_kernel(KParams p, GemvArgs g) {
  constexpr int LPR = BN / VEC, RG = THREADS / LPR;
  constexpr int KCH = kASmem / (MT * 4);
  // Row-group reduction: MP rows of A per pass, THREADS * VEC floats each.
  constexpr int MP_FIT = kASmem / (THREADS * VEC * 4);
  constexpr int MP = MP_FIT >= MT ? MT : (MP_FIT >= 8 ? 8 : MP_FIT >= 4 ? 4 : MP_FIT >= 2 ? 2 : 1);
  static_assert(THREADS % LPR == 0 && LPR <= 32 && 32 % LPR == 0, "a row group is a power-of-two part of a warp");
  static_assert(MT % MP == 0 && MP_FIT >= 1, "reduction scratch fits the A chunk");
  __shared__ __align__(16) float As[kASmem / 4];
  __shared__ __align__(16) float red[MT * BN];

  const int S = g.sk.S;
  const int z = blockIdx.z, b = z / S, s = z % S;
  const int m0 = blockIdx.y * MT, n0 = blockIdx.x * BN;
  const int lc = threadIdx.x % LPR, grp = threadIdx.x / LPR;
  const int n = n0 + lc * VEC;
  const int ncols = min(VEC, p.N - n);  // <= 0: this lane has no columns
  const int mrows = min(MT, p.M - m0);
  const __half* A = p.A + b * p.sa;
  const __half* B = p.B + b * p.sb;
  const int kb = s * g.sk.kc, ke = min(p.K, kb + g.sk.kc);
  const long long b_sk = g.b_row ? p.ldb : 1, b_sj = g.b_row ? 1 : p.ldb;  // element path strides

  float acc[MT][VEC];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[m][j] = 0.f;

  for (int c0 = kb; c0 < ke; c0 += KCH) {
    const int rows = min(KCH, ke - c0);
    if (c0 != kb) __syncthreads();  // the previous chunk is consumed
    for (int i = threadIdx.x; i < rows * MT; i += THREADS) {
      const int k = i / MT, m = i % MT;
      float v = 0.f;
      if (m < mrows) {
        const long long r = m0 + m, c = c0 + k;
        v = __half2float(A[g.a_row ? r * p.lda + c : r + c * p.lda]);
      }
      As[i] = v;
    }
    __syncthreads();
    for (int k = grp; k < rows; k += RG * kU) {
      float bf[kU][VEC];
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int kk = k + u * RG;
#pragma unroll
        for (int j = 0; j < VEC; ++j) bf[u][j] = 0.f;
        if (kk < rows && ncols > 0) {
          const long long kg = c0 + kk;
          if constexpr (BVEC) {
            const __half* src = B + kg * p.ldb + n;
            if (ncols == VEC) {
              to_float<VEC>(*reinterpret_cast<const typename BVec<VEC>::T*>(src), bf[u]);
            } else {
#pragma unroll
              for (int j = 0; j < VEC; ++j)
                if (j < ncols) bf[u][j] = __half2float(src[j]);
            }
          } else {  // element (k, n + j) at kg * sk + (n + j) * sj
            const __half* src = B + kg * b_sk + n * b_sj;
#pragma unroll
            for (int j = 0; j < VEC; ++j)
              if (j < ncols) bf[u][j] = __half2float(src[j * b_sj]);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int kk = k + u * RG;
        if (kk >= rows) break;
        const float* a = As + kk * MT;
        float av[MT];
        if constexpr (MT >= 4) {
#pragma unroll
          for (int m = 0; m < MT; m += 4) {
            const float4 q = *reinterpret_cast<const float4*>(a + m);
            av[m] = q.x; av[m + 1] = q.y; av[m + 2] = q.z; av[m + 3] = q.w;
          }
        } else {
#pragma unroll
          for (int m = 0; m < MT; ++m) av[m] = a[m];
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int j = 0; j < VEC; ++j) acc[m][j] = fmaf(av[m], bf[u][j], acc[m][j]);
      }
    }
  }

  // Row groups -> the CTA tile, in group order.
  if constexpr (RG == 1) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < VEC; ++j) red[m * BN + lc * VEC + j] = acc[m][j];
  } else {
#pragma unroll
    for (int mp = 0; mp < MT; mp += MP) {
      __syncthreads();  // As (the scratch) is free
#pragma unroll
      for (int mm = 0; mm < MP; ++mm)
#pragma unroll
        for (int j = 0; j < VEC; ++j) As[(mm * RG + grp) * BN + lc * VEC + j] = acc[mp + mm][j];
      __syncthreads();
      for (int i = threadIdx.x; i < MP * BN; i += THREADS) {
        const int mm = i / BN, col = i % BN;
        float v = As[(mm * RG) * BN + col];
        for (int r = 1; r < RG; ++r) v += As[(mm * RG + r) * BN + col];
        red[(mp + mm) * BN + col] = v;
      }
    }
  }
  const long long tile = ((long long)b * g.mtiles + blockIdx.y) * g.ntiles + blockIdx.x;
  const splitk::Out o{p.C + b * p.sc, p.ldc, m0, n0, mrows, min(BN, p.N - n0), p.alpha, p.beta};
  splitk::finish_tile<MT, BN, THREADS>(red, g.sk, tile, s, o);
}

bool b_vec_ok(const GemmParams& p, int vec) {
  const uintptr_t bytes = uintptr_t(vec) * sizeof(__half);
  return p.layout_b == Layout::RowMajor && (reinterpret_cast<uintptr_t>(p.B) & (bytes - 1)) == 0 &&
         p.ldb % vec == 0 && (p.batch <= 1 || p.stride_b % vec == 0);
}

int pick_mt(int M) { return M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16; }

template <int BN, int THREADS>
splitk::Plan split_plan(const GemmParams& p, int S, int MT) {
  const long long tiles = (long long)p.batch * ceil_div(p.M, MT) * ceil_div(p.N, BN);
  return splitk::plan(p.K, S, tiles, MT, BN);
}

template <int MT, int VEC, int BN, int THREADS>
cudaError_t launch_mt(const GemmParams& p, int S, cudaStream_t stream) {
  GemvArgs g;
  g.mtiles = ceil_div(p.M, MT);
  g.ntiles = ceil_div(p.N, BN);
  g.a_row = p.layout_a == Layout::RowMajor;
  g.b_row = p.layout_b == Layout::RowMajor;
  const splitk::Plan pl = split_plan<BN, THREADS>(p, S, MT);
  const long long tiles = (long long)p.batch * g.mtiles * g.ntiles;
  if (const cudaError_t e = splitk::acquire(stream, pl, tiles, MT, BN, &g.sk); e != cudaSuccess) return e;
  const dim3 grid(g.ntiles, g.mtiles, unsigned(p.batch) * unsigned(pl.S));
  if (!grid_ok(grid) || (long long)p.batch * pl.S > 65535) return cudaErrorInvalidValue;
  if (b_vec_ok(p, VEC))
    gemv_kernel<MT, VEC, BN, THREADS, true><<<grid, THREADS, 0, stream>>>(to_kparams(p), g);
  else
    gemv_kernel<MT, VEC, BN, THREADS, false><<<grid, THREADS, 0, stream>>>(to_kparams(p), g);
  return cudaGetLastError();
}

template <int BN, int THREADS, int VEC, int S>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  switch (pick_mt(p.M)) {
    case 1: return launch_mt<1, VEC, BN, THREADS>(p, S, stream);
    case 2: return launch_mt<2, VEC, BN, THREADS>(p, S, stream);
    case 4: return launch_mt<4, VEC, BN, THREADS>(p, S, stream);
    case 8: return launch_mt<8, VEC, BN, THREADS>(p, S, stream);
    default: return launch_mt<16, VEC, BN, THREADS>(p, S, stream);
  }
}

template <int BN, int THREADS, int VEC, int S>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream) {
  return run<BN, THREADS, VEC, S>(row_major_params(A, B, C, M, N, K), stream);
}

// The M = 1 kernel (row-major, vector B): the decode case the family is for.
template <int BN, int THREADS, int VEC, int S>
cudaError_t attrs(cudaFuncAttributes* a) {
  return cudaFuncGetAttributes(a, gemv_kernel<1, VEC, BN, THREADS, true>);
}

constexpr size_t kN = std::size(GEMV_CONFIGS);
#define AEGIS_GEMV_ARGS(I) GEMV_CONFIGS[I].BN, GEMV_CONFIGS[I].THREADS, GEMV_CONFIGS[I].VEC, GEMV_CONFIGS[I].S

std::string name_of(const GemvConfig& c) {
  return "gemv_" + std::to_string(c.BN) + "x" + std::to_string(c.THREADS) + "_v" + std::to_string(c.VEC) + "_k" +
         std::to_string(c.S);
}

std::string config_of(const GemvConfig& c) {
  return "{\"BN\":" + std::to_string(c.BN) + ",\"THREADS\":" + std::to_string(c.THREADS) +
         ",\"VEC\":" + std::to_string(c.VEC) + ",\"S\":" + std::to_string(c.S) + ",\"path\":\"cuda_core\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(GEMV_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(GEMV_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{names[I].c_str(), configs[I].c_str(),
                                                               &launch<AEGIS_GEMV_ARGS(I)>, &attrs<AEGIS_GEMV_ARGS(I)>,
                                                               &run<AEGIS_GEMV_ARGS(I)>, 16}...};
  return entries;
}
#undef AEGIS_GEMV_ARGS

}  // namespace

std::span<const KernelEntry> gemv_entries() { return make_entries(std::make_index_sequence<kN>{}); }

namespace detail {
// For gemv_split_count (registry.cu): the effective split count of config i.
int gemv_splits(size_t i, const GemmParams& p) {
  const GemvConfig& c = GEMV_CONFIGS[i];
  const int MT = pick_mt(p.M);
  const long long tiles = (long long)p.batch * ceil_div(p.M, MT) * ceil_div(p.N, c.BN);
  return splitk::plan(p.K, c.S, tiles, MT, c.BN).S;
}
}  // namespace detail

}  // namespace aegis::gemm
