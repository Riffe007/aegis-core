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
// Global loads: 16-byte vectors when K and N are multiples of 8 (every vector is
// then wholly in or out of bounds); a scalar fallback handles any other shape.
#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include "aegis/common.cuh"
#include "aegis/gemm.h"

namespace aegis::gemm {
namespace {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_regblock_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                     __half* __restrict__ C, int M, int N, int K, bool vec) {
  constexpr int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
  constexpr int MC = TM / 4, NC = TN / 4;            // float4 groups per thread
  constexpr int MSTRIDE = BM / MC, NSTRIDE = BN / NC;  // spacing between groups
  static_assert(TM % 4 == 0 && TN % 4 == 0, "micro-tile must be float4 groups");
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide evenly");
  static_assert(BK % 8 == 0 && BN % 8 == 0, "16-byte global vectors");

  __shared__ __align__(16) float As[BK][BM];  // transposed: As[k][m]
  __shared__ __align__(16) float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    if (vec) {
      // A: BM rows x BK/8 vectors. Consecutive threads take consecutive rows so
      // the transposed smem stores hit consecutive banks.
      for (int v = tid; v < BM * (BK / 8); v += THREADS) {
        const int r = v % BM, c = (v / BM) * 8;
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (m0 + r < M && k0 + c < K)
          raw = *reinterpret_cast<const uint4*>(A + size_t(m0 + r) * K + k0 + c);
        const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) As[c + j][r] = __half2float(h[j]);
      }
      for (int v = tid; v < BK * (BN / 8); v += THREADS) {
        const int r = v / (BN / 8), c = (v % (BN / 8)) * 8;
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (k0 + r < K && n0 + c < N)
          raw = *reinterpret_cast<const uint4*>(B + size_t(k0 + r) * N + n0 + c);
        const __half2* h = reinterpret_cast<const __half2*>(&raw);
        const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
        const float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
        *reinterpret_cast<float4*>(&Bs[r][c]) = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(&Bs[r][c + 4]) = make_float4(f2.x, f2.y, f3.x, f3.y);
      }
    } else {
      for (int i = tid; i < BM * BK; i += THREADS) {
        const int r = i % BM, c = i / BM;
        As[c][r] = (m0 + r < M && k0 + c < K)
                       ? __half2float(A[size_t(m0 + r) * K + k0 + c]) : 0.f;
      }
      for (int i = tid; i < BK * BN; i += THREADS) {
        const int r = i / BN, c = i % BN;
        Bs[r][c] = (k0 + r < K && n0 + c < N)
                       ? __half2float(B[size_t(k0 + r) * N + n0 + c]) : 0.f;
      }
    }
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

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + (i / 4) * MSTRIDE + ty * 4 + i % 4;
    if (row >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + (j / 4) * NSTRIDE + tx * 4 + j % 4;
      if (col < N) C[size_t(row) * N + col] = __float2half(acc[i][j]);
    }
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t launch(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  const bool vec = K % 8 == 0 && N % 8 == 0 && aligned16(A) && aligned16(B);
  const dim3 block((BM / TM) * (BN / TN));
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  gemm_regblock_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, vec);
  return cudaGetLastError();
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t attrs(cudaFuncAttributes* a) {
  return cudaFuncGetAttributes(a, gemm_regblock_kernel<BM, BN, BK, TM, TN>);
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
             REGBLOCK_CONFIGS[I].TM, REGBLOCK_CONFIGS[I].TN>}...};
  return entries;
}

}  // namespace

std::span<const KernelEntry> regblock_entries() {
  return make_entries(std::make_index_sequence<kN>{});
}

}  // namespace aegis::gemm
