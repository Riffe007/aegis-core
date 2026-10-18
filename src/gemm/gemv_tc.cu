// Decode GEMV, tensor-core path (M = 9..16 is the fit; any M runs, tiled by 16).
//
// CTA: a 16 x BN output tile over one split-K slice (splitk.cuh). WARPS_N
// warps side by side, each 16 x WN (NI = WN / 8 m16n8k16 tiles, one A fragment
// shared by all of them): M padded to 16 is a perfect m16 fit, and the tile
// is chosen for bandwidth (B bytes in flight), not compute.
// B, the streamed operand: tier 4's OperandPlan (16 B cp.async chunks,
// per-CTA N mask, k mask on the slice's partial last tile) into a STAGES-deep
// ring, XOR-swizzled as in mma_common.cuh, ldmatrix(.trans) per layout.
// A: 16 x BK per stage, padded rows (pitch BK + 8 halves row-major, 24
// column-major: 8 consecutive ldmatrix rows hit 8 distinct 16 B bank groups),
// 16 B cp.async when a chunk is whole, else only its valid halves plus zeros.
// Misaligned pointers, ld or strides (vec16_ok false): the same loop with
// synchronous element loads (tier 3 style), no cp.async.
// Epilogue: the accumulators go to an FP32 16 x BN tile in smem (aliasing the
// drained ring), then splitk::finish_tile (C directly for S = 1, else the
// partial and the ordered sum in the tile's last CTA).
// PTX ISA citations for ldmatrix, mma and cp.async: see mma_common.cuh.
#include <array>
#include <string>
#include <utility>

#include "aegis/gemm.h"
#include "mma_common.cuh"
#include "splitk.cuh"

namespace aegis::gemm {
namespace {

using namespace mma_detail;

template <int BN_, int BK_, int WARPS_N_, Layout LA_, Layout LB_>
struct DecTile {
  static constexpr int BN = BN_, BK = BK_, WARPS_N = WARPS_N_;  // BM = 16
  static constexpr Layout LA = LA_, LB = LB_;
  static constexpr int THREADS = 32 * WARPS_N, WN = BN / WARPS_N, NI = WN / 8;
  static constexpr bool A_K_CONTIG = LA == Layout::RowMajor;
  static constexpr bool B_K_CONTIG = LB == Layout::ColMajor;
  static constexpr int A_ROWS = A_K_CONTIG ? 16 : BK, A_PITCH = (A_K_CONTIG ? BK : 16) + 8;
  static constexpr int A_ELEMS = A_ROWS * A_PITCH;
  static constexpr int A_CHUNKS = A_ROWS * (A_K_CONTIG ? BK / 8 : 2);
  using SB = SmemOperand<B_K_CONTIG ? BN : BK, B_K_CONTIG ? BK : BN>;
  static constexpr int B_ELEMS = SB::ELEMS;
  static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
  static constexpr size_t STAGE_BYTES = size_t(STAGE_ELEMS) * sizeof(__half);
  static_assert(BN % WARPS_N == 0 && WN % 16 == 0, "warp N: a multiple of 16 (one ldmatrix.x4 = 2 n8 tiles)");
  static_assert(BK % 16 == 0 && (A_ELEMS * 2) % 16 == 0, "k16 steps; B tile 16 B aligned");
};

struct TcArgs {
  splitk::Args sk;
  int mtiles, ntiles;
};

// A's 16 x BK tile of k-tile k0 (slice coordinates; KS: slice length).
template <class T>
__device__ __forceinline__ void load_a(__half* As, const __half* A, long long lda, int m0, int M, int k0, int KS,
                                       bool vec) {
  const uint32_t s = smem_addr(As);
  for (int i = threadIdx.x; i < T::A_CHUNKS; i += T::THREADS) {
    int r, c, nvalid;
    const __half* src;
    if constexpr (T::A_K_CONTIG) {  // As[m][k]
      r = i / (T::BK / 8);
      c = (i % (T::BK / 8)) * 8;
      src = A + (long long)(m0 + r) * lda + k0 + c;
      nvalid = m0 + r < M ? max(0, min(8, KS - (k0 + c))) : 0;
    } else {  // As[k][m]
      r = i / 2;
      c = (i % 2) * 8;
      src = A + (long long)(k0 + r) * lda + m0 + c;
      nvalid = k0 + r < KS ? max(0, min(8, M - (m0 + c))) : 0;
    }
    const uint32_t dst = s + uint32_t((r * T::A_PITCH + c) * sizeof(__half));
    if (nvalid == 8 && vec)
      cp_async_16(dst, src);
    else
      st_shared_16(dst, nvalid == 0 ? make_uint4(0, 0, 0, 0) : load_partial_16(src, nvalid * 2));
  }
}

template <class T>
__device__ __forceinline__ void compute(const __half* As, const __half* Bs, float (&acc)[T::NI][4]) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int wn = warp * T::WN;
  const int j = lane / 8, l8 = lane % 8;
#pragma unroll
  for (int kk = 0; kk < T::BK; kk += 16) {
    uint32_t a[4];
    if constexpr (T::A_K_CONTIG)
      ldmatrix_x4(a, smem_addr(As + (lane % 16) * T::A_PITCH + kk + (lane / 16) * 8));
    else
      ldmatrix_x4_trans(a, smem_addr(As + (kk + l8 + (j / 2) * 8) * T::A_PITCH + (j % 2) * 8));
#pragma unroll
    for (int nj = 0; nj < T::NI / 2; ++nj) {
      uint32_t r[4];
      if constexpr (T::B_K_CONTIG)
        ldmatrix_x4(r, smem_addr(Bs + T::SB::at(wn + nj * 16 + (j / 2) * 8 + l8, kk + (j % 2) * 8)));
      else
        ldmatrix_x4_trans(r, smem_addr(Bs + T::SB::at(kk + l8 + (j % 2) * 8, wn + nj * 16 + (j / 2) * 8)));
      mma_16816(acc[2 * nj], a, r[0], r[1]);
      mma_16816(acc[2 * nj + 1], a, r[2], r[3]);
    }
  }
}

template <class T, int STAGES, bool MASK_N>
__device__ __forceinline__ void pipe_loop(__half* smem, const __half* A, const __half* B, const KParams& p, int m0,
                                          int n0, int KS, float (&acc)[T::NI][4]) {
  OperandPlan<typename T::SB, T::THREADS, !T::B_K_CONTIG> pb;
  pb.init(B, p.ldb, n0, p.N);
  const int KT = ceil_div(KS, T::BK);
  const bool k_tail = KS % T::BK != 0;
  const uint32_t s0 = smem_addr(smem);
  auto load = [&](int s, int kt) {
    __half* as = smem + s * T::STAGE_ELEMS;
    const uint32_t sb = s0 + uint32_t(s * T::STAGE_BYTES + T::A_ELEMS * sizeof(__half));
    const int k0 = kt * T::BK;
    load_a<T>(as, A, p.lda, m0, p.M, k0, KS, true);
    if (k_tail && kt == KT - 1)
      pb.template load_async<MASK_N, true>(sb, k0, KS);
    else
      pb.template load_async<MASK_N, false>(sb, k0, KS);
    pb.advance();
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int pf = kt + STAGES - 1;
    if (pf < KT) load(pf % STAGES, pf);
    cp_async_commit();
    const __half* as = smem + (kt % STAGES) * T::STAGE_ELEMS;
    compute<T>(as, as + T::A_ELEMS, acc);
  }
  cp_async_wait<0>();
}

template <class T, int STAGES>
__global__ void __launch_bounds__(T::THREADS, 1) gemv_tc_kernel(KParams p, TcArgs g, bool vec) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* smem = reinterpret_cast<__half*>(smem_raw);
  const int S = g.sk.S;
  const int z = blockIdx.z, b = z / S, s = z % S;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * T::BN;
  const int kb = s * g.sk.kc, KS = min(p.K, kb + g.sk.kc) - kb;
  const __half* A = p.A + b * p.sa + (T::A_K_CONTIG ? (long long)kb : (long long)kb * p.lda);
  const __half* B = p.B + b * p.sb + (T::B_K_CONTIG ? (long long)kb : (long long)kb * p.ldb);
  float acc[T::NI][4] = {};

  if (!vec) {  // misaligned: synchronous element loads
    __half* As = smem;
    __half* Bs = smem + T::A_ELEMS;
    for (int k0 = 0; k0 < KS; k0 += T::BK) {
      load_a<T>(As, A, p.lda, m0, p.M, k0, KS, false);
      if constexpr (T::B_K_CONTIG)
        load_operand_scalar<typename T::SB, T::THREADS>(Bs, B, p.ldb, n0, p.N, k0, KS);
      else
        load_operand_scalar<typename T::SB, T::THREADS>(Bs, B, p.ldb, k0, KS, n0, p.N);
      __syncthreads();
      compute<T>(As, Bs, acc);
      __syncthreads();
    }
  } else if (n0 + T::BN <= p.N) {
    pipe_loop<T, STAGES, false>(smem, A, B, p, m0, n0, KS, acc);
  } else {
    pipe_loop<T, STAGES, true>(smem, A, B, p, m0, n0, KS, acc);
  }
  __syncthreads();  // every warp is done with the ring: alias it as the FP32 tile
  float* red = reinterpret_cast<float*>(smem_raw);
  const int lane = threadIdx.x % 32, wn = (threadIdx.x / 32) * T::WN;
#pragma unroll
  for (int ni = 0; ni < T::NI; ++ni)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = lane / 4 + h * 8, col = wn + ni * 8 + (lane % 4) * 2;
      red[row * T::BN + col] = acc[ni][2 * h];
      red[row * T::BN + col + 1] = acc[ni][2 * h + 1];
    }
  const long long tile = ((long long)b * g.mtiles + blockIdx.y) * g.ntiles + blockIdx.x;
  const splitk::Out o{p.C + b * p.sc, p.ldc, m0, n0, min(16, p.M - m0), min(T::BN, p.N - n0), p.alpha, p.beta};
  splitk::finish_tile<16, T::BN, T::THREADS>(red, g.sk, tile, s, o);
}

template <class T, int STAGES>
constexpr size_t smem_bytes() {
  const size_t ring = T::STAGE_BYTES * STAGES, tile = size_t(16) * T::BN * sizeof(float);
  return ring > tile ? ring : tile;
}

template <class T, int STAGES>
cudaError_t launch_layout(const GemmParams& p, int S, cudaStream_t stream) {
  constexpr size_t smem = smem_bytes<T, STAGES>();
  static const cudaError_t configured =
      cudaFuncSetAttribute(gemv_tc_kernel<T, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (configured != cudaSuccess) return configured;
  TcArgs g;
  g.mtiles = ceil_div(p.M, 16);
  g.ntiles = ceil_div(p.N, T::BN);
  const long long tiles = (long long)p.batch * g.mtiles * g.ntiles;
  const splitk::Plan pl = splitk::plan(p.K, S, tiles, 16, T::BN);
  if (const cudaError_t e = splitk::acquire(stream, pl, tiles, 16, T::BN, &g.sk); e != cudaSuccess) return e;
  const dim3 grid(g.ntiles, g.mtiles, unsigned(p.batch) * unsigned(pl.S));
  if (!grid_ok(grid) || (long long)p.batch * pl.S > 65535) return cudaErrorInvalidValue;
  gemv_tc_kernel<T, STAGES><<<grid, T::THREADS, smem, stream>>>(to_kparams(p), g, vec16_ok(p));
  return cudaGetLastError();
}

template <int BN, int BK, int WN, int STAGES, int S>
cudaError_t run(const GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = validate(p); e != cudaSuccess) return e;
  if (is_empty(p)) return cudaSuccess;
  constexpr Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  if (p.layout_a == R && p.layout_b == R) return launch_layout<DecTile<BN, BK, WN, R, R>, STAGES>(p, S, stream);
  if (p.layout_a == R) return launch_layout<DecTile<BN, BK, WN, R, Cm>, STAGES>(p, S, stream);
  if (p.layout_b == R) return launch_layout<DecTile<BN, BK, WN, Cm, R>, STAGES>(p, S, stream);
  return launch_layout<DecTile<BN, BK, WN, Cm, Cm>, STAGES>(p, S, stream);
}

template <int BN, int BK, int WN, int STAGES, int S>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream) {
  return run<BN, BK, WN, STAGES, S>(row_major_params(A, B, C, M, N, K), stream);
}

template <int BN, int BK, int WN, int STAGES, int S>
cudaError_t attrs(cudaFuncAttributes* a) {
  using T = DecTile<BN, BK, WN, Layout::RowMajor, Layout::RowMajor>;
  const cudaError_t e = cudaFuncGetAttributes(a, gemv_tc_kernel<T, STAGES>);
  if (e == cudaSuccess) a->sharedSizeBytes += smem_bytes<T, STAGES>();
  return e;
}

constexpr size_t kN = std::size(GEMV_TC_CONFIGS);
#define AEGIS_TC_ARGS(I)                                                                              \
  GEMV_TC_CONFIGS[I].BN, GEMV_TC_CONFIGS[I].BK, GEMV_TC_CONFIGS[I].WARPS_N, GEMV_TC_CONFIGS[I].STAGES, \
      GEMV_TC_CONFIGS[I].S

std::string name_of(const GemvTcConfig& c) {
  return "gemv_tc_16x" + std::to_string(c.BN) + "x" + std::to_string(c.BK) + "_1x" + std::to_string(c.WARPS_N) +
         "_s" + std::to_string(c.STAGES) + "_k" + std::to_string(c.S);
}

std::string config_of(const GemvTcConfig& c) {
  return "{\"BM\":16,\"BN\":" + std::to_string(c.BN) + ",\"BK\":" + std::to_string(c.BK) +
         ",\"WARPS_N\":" + std::to_string(c.WARPS_N) + ",\"STAGES\":" + std::to_string(c.STAGES) +
         ",\"S\":" + std::to_string(c.S) + ",\"path\":\"tensor_core\"}";
}

template <size_t... I>
std::span<const KernelEntry> make_entries(std::index_sequence<I...>) {
  static const std::array<std::string, kN> names{name_of(GEMV_TC_CONFIGS[I])...};
  static const std::array<std::string, kN> configs{config_of(GEMV_TC_CONFIGS[I])...};
  static const std::array<KernelEntry, kN> entries{KernelEntry{names[I].c_str(), configs[I].c_str(),
                                                               &launch<AEGIS_TC_ARGS(I)>, &attrs<AEGIS_TC_ARGS(I)>,
                                                               &run<AEGIS_TC_ARGS(I)>, 16}...};
  return entries;
}
#undef AEGIS_TC_ARGS

}  // namespace

std::span<const KernelEntry> gemv_tc_entries() { return make_entries(std::make_index_sequence<kN>{}); }

namespace detail {
int gemv_tc_splits(size_t i, const GemmParams& p) {
  const GemvTcConfig& c = GEMV_TC_CONFIGS[i];
  const long long tiles = (long long)p.batch * ceil_div(p.M, 16) * ceil_div(p.N, c.BN);
  return splitk::plan(p.K, c.S, tiles, 16, c.BN).S;
}
}  // namespace detail

}  // namespace aegis::gemm
