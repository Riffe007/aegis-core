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

#include <atomic>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "hw/device.h"
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

// The split count from the SM count (the "_ksm" configs, S = 0): over
// S in [1, max(1, K / min_rows)], the one that minimizes the busiest SM's K rows,
//   ceil(tiles * S / sms) * ceil(K / S),
// the smallest such S on a tie. Memory-bound decode: an SM's resident CTAs
// share its bandwidth, so the model counts CTAs per SM, not resident slots,
// and a partial last wave costs a whole one. More splits pay only while they
// shorten the busiest SM's rows; min_rows (the caller's pipeline depth) keeps
// each split long enough to amortize its prologue and its share of the
// reduction.
// - Fewer CTAs than SMs is allowed when no larger S lowers the maximum load.
//   On 48 SMs the 896x4864 decode shapes run 14 tiles x 3 = 42 CTAs.
// - Filling a wave of resident slots was the previous auto rule; it measured
//   2-6% slower there (tune c2b937d8adca, LOG 2026-10-18).
// Deterministic per (shape, SM count). 1 if sms < 1 (no device).
inline int sm_load_splits(int sms, long long tiles, int K, int min_rows) {
  if (sms < 1 || tiles < 1 || K < 1) return 1;
  long long s_max = K / (min_rows > 0 ? min_rows : 1);
  if (s_max > 4096) s_max = 4096;
  int best = 1;
  long long best_load = 0;
  for (long long S = 1; S <= (s_max < 1 ? 1 : s_max); ++S) {
    const long long per_sm = (tiles * S + sms - 1) / sms;
    const long long load = per_sm * ((K + S - 1) / S);
    if (S == 1 || load < best_load) {
      best = int(S);
      best_load = load;
    }
  }
  return best;
}

// The SM count sizing decisions use: the device's, or the thread's simulated profile.
inline int sm_count() {
  const hw::DeviceLimits* d = hw::current();
  return d ? d->sm_count : 0;
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

// The CTA's TM x TN FP32 tile is in smem `red` (row-major, pitch TN, 16 B
// aligned; written by all threads before the call, no barrier needed before
// it). `tile`: the tile's index over all batches; `split`: this CTA's split.
// `flag`: one int of the caller's dynamic smem. No static smem here: static
// smem sits before the dynamic region and would shift it off 32 B sectors
// (16 B of it doubled the L2 sectors of every 16 B cp.async; LOG 2026-10-17).
template <int TM, int TN, int THREADS>
__device__ __forceinline__ void finish_tile(const float* red, const Args& a, long long tile, int split, const Out& o,
                                            int* flag) {
  __syncthreads();
  constexpr int ELEMS = TM * TN;
  if (a.S == 1) {
    for (int i = threadIdx.x; i < ELEMS; i += THREADS) {
      const int r = i / TN, c = i % TN;
      if (r < o.rows && c < o.cols) store_one(o, r, c, red[i]);
    }
    return;
  }
  // Whole TM x TN slots (invalid entries too), as float4: TN % 4 == 0.
  static_assert(TN % 4 == 0, "float4 partial slots");
  constexpr int VECS = ELEMS / 4;
  float4* slot0 = reinterpret_cast<float4*>(a.partials + tile * a.S * ELEMS);
  float4* mine = slot0 + (long long)split * VECS;
  const float4* red4 = reinterpret_cast<const float4*>(red);
  for (int i = threadIdx.x; i < VECS; i += THREADS) __stcg(mine + i, red4[i]);
  __threadfence();  // the partial is visible device-wide before the ticket
  __syncthreads();
  if (threadIdx.x == 0) *flag = atomicAdd(a.counters + tile, 1u) == unsigned(a.S - 1);
  __syncthreads();
  if (!*flag) return;
  __threadfence();  // acquire: the other S - 1 partials are visible
  // Sum slots 0..S-1 in that order (v = p0, v += p1, ...). Each thread takes
  // kElems float4s at a time and loads kBatch slots of all of them before any
  // add (kElems x kBatch independent loads in flight: one L2 round trip per
  // group instead of one per element); the adds per element stay in order.
  constexpr int kElems = 4, kBatch = 4;
  for (int i0 = threadIdx.x; i0 < VECS; i0 += THREADS * kElems) {
    float4 v[kElems];
#pragma unroll
    for (int e = 0; e < kElems; ++e) v[e] = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s0 = 0; s0 < a.S; s0 += kBatch) {
      float4 q[kElems][kBatch];
#pragma unroll
      for (int e = 0; e < kElems; ++e)
#pragma unroll
        for (int u = 0; u < kBatch; ++u) {
          const int i = i0 + e * THREADS;
          if (i < VECS && s0 + u < a.S) q[e][u] = __ldcg(slot0 + (long long)(s0 + u) * VECS + i);
        }
#pragma unroll
      for (int e = 0; e < kElems; ++e)
#pragma unroll
        for (int u = 0; u < kBatch; ++u) {
          if (s0 + u >= a.S) break;
          if (s0 + u == 0) {
            v[e] = q[e][0];
          } else {
            v[e].x += q[e][u].x; v[e].y += q[e][u].y; v[e].z += q[e][u].z; v[e].w += q[e][u].w;
          }
        }
    }
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      const int i = i0 + e * THREADS;
      if (i >= VECS) break;
      const int r = (i * 4) / TN, c = (i * 4) % TN;
      if (r >= o.rows || c >= o.cols) continue;
      const float vs[4] = {v[e].x, v[e].y, v[e].z, v[e].w};
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (c + j < o.cols) store_one(o, r, c + j, vs[j]);
    }
  }
  if (threadIdx.x == 0) a.counters[tile] = 0;  // the next launch starts from 0
}

}  // namespace aegis::gemm::splitk
