#pragma once
// Deterministic split-K, shared by the decode GEMV family (gemv, gemv_tc) and,
// later, stream-K.
//
// Partition: S CTAs per output tile (TM x TN, one batch); split s covers
// k in [s * kc, min(K, (s + 1) * kc)), kc a multiple of 32, so no split is
// empty and every split start keeps 16-byte alignment of k-contiguous operands.
//
// Reduction, in the kernel, with an ordered ticket (never atomic
// accumulation): each CTA leaves its FP32 TM x TN tile in smem; finish_tile()
//   S == 1: applies the epilogue to C directly;
//   S > 1:  writes the tile to its partial slot in the workspace, fences, and
//           takes a ticket on the tile's counter (atomicAdd on an int: the
//           semaphore, not the data). The CTA that draws S - 1 is the last:
//           it reads all S partial slots in split order 0..S-1 (its own
//           included) and sums them in that order, applies the epilogue, and
//           sets the counter back to 0 (ready for the next launch, so a
//           captured graph replays without a reset). The sum order never
//           depends on which CTA finishes last, so results are bitwise
//           identical across runs.
// Workspace (workspace.h): a counter buffer (tiles x u32, zero on allocation,
// 0 again after every completed launch) and a partials buffer (tiles x S x
// TM x TN FP32), separate cache entries.
//
// Epilogue (as the other families): C = alpha * acc + beta * C; alpha == 0
// contributes nothing from A*B; beta == 0 does not read C. NaN and Inf in the
// partials propagate through the sum.
#include <cuda_fp16.h>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "workspace.h"

namespace aegis::gemm::splitk {

constexpr int kGranule = 32;  // kc is a multiple of this

struct Plan {
  int S = 1;   // effective split count
  int kc = 0;  // K rows per split (a multiple of kGranule, or K when S == 1)
};

// S_req splits of K, for `tiles` output tiles (all batches) of TM x TN.
inline Plan plan(int K, int S_req, long long tiles, int TM, int TN) {
  Plan pl;
  int S = S_req < 1 ? 1 : S_req;
  const long long per_split = tiles * TM * TN * (long long)sizeof(float);
  const long long cap = per_split > 0 ? kSplitKMaxWorkspace / per_split : 1;
  if (S > cap) S = int(cap < 1 ? 1 : cap);
  if (S > 1) {
    const int kc = ceil_div(ceil_div(K, S), kGranule) * kGranule;
    S = ceil_div(K, kc);
    pl.kc = kc;
  }
  pl.S = S;
  if (S == 1) pl.kc = K;
  return pl;
}

struct Args {
  int S;                 // effective splits
  int kc;                // rows per split
  unsigned* counters;    // per tile; nullptr when S == 1
  float* partials;       // tiles x S x TM x TN; nullptr when S == 1
};

// The launch's workspace (S > 1), from the per-stream cache: counters and
// partials in separate buffers. A buffer serves every shape of its size class
// on the stream; partials may hold another shape's bytes, so counters never
// live in a partials buffer (they must read 0 at launch).
inline cudaError_t acquire(cudaStream_t stream, const Plan& pl, long long tiles, int TM, int TN, Args* a) {
  a->S = pl.S;
  a->kc = pl.kc;
  a->counters = nullptr;
  a->partials = nullptr;
  if (pl.S == 1) return cudaSuccess;
  void* cnt = nullptr;
  void* part = nullptr;
  if (const cudaError_t e = detail::stream_workspace(stream, size_t(tiles) * sizeof(unsigned), &cnt,
                                                     detail::WorkspaceUse::SplitKCounters);
      e != cudaSuccess)
    return e;
  if (const cudaError_t e = detail::stream_workspace(stream, size_t(tiles) * pl.S * TM * TN * sizeof(float), &part,
                                                     detail::WorkspaceUse::SplitKPartials);
      e != cudaSuccess)
    return e;
  a->counters = static_cast<unsigned*>(cnt);
  a->partials = static_cast<float*>(part);
  return cudaSuccess;
}

// The output region of one tile.
struct Out {
  __half* C;        // C of this batch
  long long ldc;
  int m0, n0;       // tile origin
  int rows, cols;   // valid extent (<= TM, TN)
  float alpha, beta;
};

__device__ __forceinline__ void store_one(const Out& o, int r, int c, float v) {
  __half* q = o.C + (long long)(o.m0 + r) * o.ldc + o.n0 + c;
  float x = o.alpha == 0.f ? 0.f : o.alpha * v;
  if (o.beta != 0.f) x += o.beta * __half2float(*q);
  *q = __float2half(x);
}

// The CTA's TM x TN FP32 tile is in smem `red` (row-major, pitch TN; written
// by all threads before the call, no barrier needed before it). `tile`: the
// tile's index over all batches; `split`: this CTA's split.
template <int TM, int TN, int THREADS>
__device__ __forceinline__ void finish_tile(const float* red, const Args& a, long long tile, int split, const Out& o) {
  __syncthreads();
  constexpr int ELEMS = TM * TN;
  if (a.S == 1) {
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) {
      const int r = i / TN, c = i % TN;
      if (r < o.rows && c < o.cols) store_one(o, r, c, red[i]);
    }
    return;
  }
  float* slot0 = a.partials + tile * a.S * ELEMS;
  float* mine = slot0 + (long long)split * ELEMS;
  for (int i = threadIdx.x; i < ELEMS; i += THREADS) __stcg(mine + i, red[i]);
  __threadfence();  // the partial is visible device-wide before the ticket
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(a.counters + tile, 1u) == unsigned(a.S - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();  // acquire: the other S - 1 partials are visible
  for (int i = threadIdx.x; i < ELEMS; i += THREADS) {
    const int r = i / TN, c = i % TN;
    if (r >= o.rows || c >= o.cols) continue;
    float v = __ldcg(slot0 + i);
    for (int s = 1; s < a.S; ++s) v += __ldcg(slot0 + (long long)s * ELEMS + i);
    store_one(o, r, c, v);
  }
  if (threadIdx.x == 0) a.counters[tile] = 0;  // the next launch starts from 0
}

}  // namespace aegis::gemm::splitk
