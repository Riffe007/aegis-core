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
// float4 per four rows: a broadcast within a row group). B loads are
// double-buffered in registers: the next kU = 4 steps (2 for MT 16 x VEC 8)
// load while the current kU compute (up to 2 kU loads per thread in flight),
// and a chunk's first loads go out before A is staged, so B latency
// overlaps A's load and the barrier.
//
// A: the slice of A's MT rows is staged once in smem as FP32, in chunks of
// KCH rows (kASmem bytes); decode slices fit in one chunk. Dynamic smem is
// sized to the call (GemvSmem), so small MT does not cap occupancy. Any layout and
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
#include "hw/device.h"
#include "splitk.cuh"

namespace aegis::gemm {
namespace {

constexpr int kASmem = 28 << 10;  // A chunk (FP32) budget per CTA; the launch sets the dynamic smem limit (configure)
// A split is at least this many K rows (the CUDA-core path has no smem ring to fill).
constexpr int kMinSplitRows = 64;
// k steps per register buffer (two buffers): 4, or 2 when the MT x VEC accumulators
// already take 128 registers (MT 16, VEC 8), so the kernel stays under 255 without spills.
template <int MT, int VEC>
constexpr int k_steps() { return MT * VEC >= 128 ? 2 : 4; }

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
  int region;         // floats of the A chunk / scratch region (GemvSmem::region)
  bool a_row, b_row;  // layouts (element path of B, and A staging)
};

// Smem plan of one instantiation: [A chunk / row-group scratch][MT x BN tile], FP32.
template <int MT, int VEC, int BN, int THREADS>
struct GemvSmem {
  static constexpr int LPR = BN / VEC, RG = THREADS / LPR;
  static constexpr int KCH = kASmem / (MT * 4);  // A rows per chunk
  // Row-group reduction: MP rows of the tile per pass, THREADS * VEC floats each.
  static constexpr int MP_FIT = kASmem / (THREADS * VEC * 4);
  static constexpr int MP = MP_FIT >= MT ? MT : (MP_FIT >= 8 ? 8 : MP_FIT >= 4 ? 4 : MP_FIT >= 2 ? 2 : 1);
  static constexpr int SCRATCH = RG == 1 ? 0 : MP * THREADS * VEC;
  static_assert(THREADS % LPR == 0 && LPR <= 32 && 32 % LPR == 0, "a row group is a power-of-two part of a warp");
  static_assert(MT % MP == 0 && MP_FIT >= 1, "reduction scratch fits the A budget");
  // Floats of the first region for slices of kc rows (16 B multiple, so the tile stays float4 aligned).
  static int region(int kc) {
    const int a = (kc < KCH ? kc : KCH) * MT;
    const int r = a > SCRATCH ? a : SCRATCH;
    return (r + 3) / 4 * 4;
  }
  // + 16 B: the split-K ticket flag after the tile.
  static size_t bytes(int kc) { return size_t(region(kc) + MT * BN) * sizeof(float) + 16; }
};

template <int MT, int VEC, int BN, int THREADS, bool BVEC>
__global__ void __launch_bounds__(THREADS, 1) gemv_kernel(KParams p, GemvArgs g) {
  using SM = GemvSmem<MT, VEC, BN, THREADS>;
  constexpr int LPR = SM::LPR, RG = SM::RG, KCH = SM::KCH, MP = SM::MP;
  using V = typename BVec<VEC>::T;
  constexpr int kU = k_steps<MT, VEC>();
  extern __shared__ __align__(16) float smem[];
  float* As = smem;  // A chunk, then the row-group scratch
  float* red = smem + g.region;

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

  // B(k, n .. n + VEC - 1) as halves; zeros past N. Nothing outside the matrix is read.
  auto load_b = [&](long long kg) -> V {
    union { V v; __half h[VEC]; } u;
    u.v = V{};
    if (ncols <= 0) return u.v;
    if constexpr (BVEC) {
      const __half* src = B + kg * p.ldb + n;
      if (ncols == VEC) return __ldg(reinterpret_cast<const V*>(src));
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        if (j < ncols) u.h[j] = src[j];
    } else {  // element (k, n + j) at kg * sk + (n + j) * sj
      const __half* src = B + kg * b_sk + n * b_sj;
#pragma unroll
      for (int j = 0; j < VEC; ++j)
        if (j < ncols) u.h[j] = src[j * b_sj];
    }
    return u.v;
  };

  float acc[MT][VEC];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[m][j] = 0.f;

  for (int c0 = kb; c0 < ke; c0 += KCH) {
    const int rows = min(KCH, ke - c0);
    // The chunk's first kU B loads go out before A is staged: B latency overlaps
    // A's load and the barrier.
    V cur[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int kk = grp + u * RG;
      cur[u] = kk < rows ? load_b(c0 + kk) : V{};
    }
    if (c0 != kb) __syncthreads();  // the previous chunk is consumed
    if (g.a_row) {  // k fastest: coalesced along A's rows
      for (int i = threadIdx.x; i < rows * MT; i += THREADS) {
        const int m = i / rows, k = i % rows;
        As[k * MT + m] = m < mrows ? __half2float(A[(long long)(m0 + m) * p.lda + c0 + k]) : 0.f;
      }
    } else {  // m fastest: coalesced along A's columns
      for (int i = threadIdx.x; i < rows * MT; i += THREADS) {
        const int k = i / MT, m = i % MT;
        As[i] = m < mrows ? __half2float(A[(long long)(c0 + k) * p.lda + m0 + m]) : 0.f;
      }
    }
    __syncthreads();
    for (int k = grp; k < rows; k += RG * kU) {
      V nxt[kU];  // the next kU steps load while these compute
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int kk = k + RG * kU + u * RG;
        nxt[u] = kk < rows ? load_b(c0 + kk) : V{};
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) {
        const int kk = k + u * RG;
        if (kk >= rows) break;
        float bf[VEC];
        to_float<VEC>(cur[u], bf);
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
          for (int j = 0; j < VEC; ++j) acc[m][j] = fmaf(av[m], bf[j], acc[m][j]);
      }
#pragma unroll
      for (int u = 0; u < kU; ++u) cur[u] = nxt[u];
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
  splitk::finish_tile<MT, BN, THREADS>(red, g.sk, tile, s, o, reinterpret_cast<int*>(red + MT * BN));
}

bool b_vec_ok(const GemmParams& p, int vec) {
  const uintptr_t bytes = uintptr_t(vec) * sizeof(__half);
  return p.layout_b == Layout::RowMajor && (reinterpret_cast<uintptr_t>(p.B) & (bytes - 1)) == 0 &&
         p.ldb % vec == 0 && (p.batch <= 1 || p.stride_b % vec == 0);
}

int pick_mt(int M) { return M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16; }

// The split plan of one call; S == 0 ("_ksm"): from the SM count
// (splitk::sm_load_splits), splits of at least kMinSplitRows of K.
template <int MT, int VEC, int BN, int THREADS>
splitk::Plan plan_for(const GemmParams& p, int S) {
  const long long tiles = (long long)p.batch * ceil_div(p.M, MT) * ceil_div(p.N, BN);
  if (S == 0) S = splitk::sm_load_splits(splitk::sm_count(), tiles, p.K, kMinSplitRows);
  return splitk::plan(p.K, S, tiles, MT, BN);
}

template <int MT, int VEC, int BN, int THREADS>
DecodeLaunch launch_shape(const GemmParams& p, int S) {
  const splitk::Plan pl = plan_for<MT, VEC, BN, THREADS>(p, S);
  return DecodeLaunch{pl.S, unsigned(ceil_div(p.N, BN)), unsigned(ceil_div(p.M, MT)),
                      unsigned(p.batch) * unsigned(pl.S)};
}

// The dynamic smem limit of both B paths at the instantiation's largest
// slice, set once per device (the default limit is the device's
// sharedMemPerBlock; nothing assumes its value).
template <int MT, int VEC, int BN, int THREADS>
cudaError_t configure() {
  using SM = GemvSmem<MT, VEC, BN, THREADS>;
  static hw::PerDevice once_vec, once_elem;
  if (const cudaError_t e = hw::configure_smem(once_vec, gemv_kernel<MT, VEC, BN, THREADS, true>, SM::bytes(SM::KCH));
      e != cudaSuccess)
    return e;
  return hw::configure_smem(once_elem, gemv_kernel<MT, VEC, BN, THREADS, false>, SM::bytes(SM::KCH));
}

template <int MT, int VEC, int BN, int THREADS>
cudaError_t launch_mt(const GemmParams& p, int S, cudaStream_t stream) {
  if (const cudaError_t e = configure<MT, VEC, BN, THREADS>(); e != cudaSuccess) return e;
  GemvArgs g;
  g.mtiles = ceil_div(p.M, MT);
  g.ntiles = ceil_div(p.N, BN);
  g.a_row = p.layout_a == Layout::RowMajor;
  g.b_row = p.layout_b == Layout::RowMajor;
  const splitk::Plan pl = plan_for<MT, VEC, BN, THREADS>(p, S);
  const long long tiles = (long long)p.batch * g.mtiles * g.ntiles;
  if (const cudaError_t e = splitk::acquire(stream, pl, tiles, MT, BN, &g.sk); e != cudaSuccess) return e;
  const dim3 grid(g.ntiles, g.mtiles, unsigned(p.batch) * unsigned(pl.S));
  if (!grid_ok(grid) || (long long)p.batch * pl.S > 65535) return cudaErrorInvalidValue;
  using SM = GemvSmem<MT, VEC, BN, THREADS>;
  g.region = SM::region(pl.kc);
  const size_t smem = SM::bytes(pl.kc);  // <= SM::bytes(KCH), the limit configure() set
  if (b_vec_ok(p, VEC))
    gemv_kernel<MT, VEC, BN, THREADS, true><<<grid, THREADS, smem, stream>>>(to_kparams(p), g);
  else
    gemv_kernel<MT, VEC, BN, THREADS, false><<<grid, THREADS, smem, stream>>>(to_kparams(p), g);
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

// The launch of one call: effective splits and grid (gemv_launch_plan).
template <int BN, int THREADS, int VEC, int S>
DecodeLaunch splits(const GemmParams& p) {
  switch (pick_mt(p.M)) {
    case 1: return launch_shape<1, VEC, BN, THREADS>(p, S);
    case 2: return launch_shape<2, VEC, BN, THREADS>(p, S);
    case 4: return launch_shape<4, VEC, BN, THREADS>(p, S);
    case 8: return launch_shape<8, VEC, BN, THREADS>(p, S);
    default: return launch_shape<16, VEC, BN, THREADS>(p, S);
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
         (c.S == 0 ? std::string("sm") : std::to_string(c.S));
}

std::string config_of(const GemvConfig& c) {
  return "{\"BN\":" + std::to_string(c.BN) + ",\"THREADS\":" + std::to_string(c.THREADS) +
         ",\"VEC\":" + std::to_string(c.VEC) + ",\"S\":" + std::to_string(c.S) + (c.S == 0 ? ",\"split\":\"sm_load\"" : "") + ",\"path\":\"cuda_core\"}";
}

using SplitsFn = DecodeLaunch (*)(const GemmParams&);

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(GEMV_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(GEMV_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{names[I].c_str(), configs[I].c_str(),
                                                               &launch<AEGIS_GEMV_ARGS(I)>, &attrs<AEGIS_GEMV_ARGS(I)>,
                                                               &run<AEGIS_GEMV_ARGS(I)>, 16}...};
  return entries;
}

template <size_t... I>
SplitsFn splits_fn(size_t i, std::index_sequence<I...>) {
  static constexpr SplitsFn fns[] = {&splits<AEGIS_GEMV_ARGS(I)>...};
  return fns[i];
}
#undef AEGIS_GEMV_ARGS

}  // namespace

std::span<const KernelEntry> gemv_entries() { return make_entries(std::make_index_sequence<kN>{}); }

namespace detail {
// For gemv_split_count (registry.cu): the effective split count of config i.
DecodeLaunch gemv_splits(size_t i, const GemmParams& p) { return splits_fn(i, std::make_index_sequence<kN>{})(p); }
}  // namespace detail

}  // namespace aegis::gemm
