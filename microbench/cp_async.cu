// Probe: global->shared copy throughput, cp.async.cg 16 B (PTX ISA 9.4
// 9.7.10.28.3.1, sm_80+) vs. plain ld.global.v4 + st.shared.v4.
//
// Each block streams its slice of a buffer through a 2-tile smem ring, 16 KiB
// per tile (256 threads x 4 x 16 B). Working sets: 512 MiB (DRAM) and 1/2 L2
// (L2-resident after the first pass). Bytes moved / event time = GB/s.
// Timed loops: 4 LDGSTS per tile (cp.async); 4 LDG.128 + 4 STS.128 (ld_st).
#include <cstdint>

#include "common.cuh"

namespace {
constexpr int kThreads = 256;
constexpr int kVecs = 4;                                  // 16 B vectors per thread per tile
constexpr int kTileBytes = kThreads * kVecs * 16;         // 16 KiB
}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
mb_copy_cp_async(const uint4* src, size_t vecs_per_block, int passes, uint32_t* sink) {
  __shared__ __align__(16) uint4 tile[2][kThreads * kVecs];
  const uint4* my = src + blockIdx.x * vecs_per_block;
  const size_t tiles = vecs_per_block / (kThreads * kVecs);
#pragma unroll 1
  for (int p = 0; p < passes; ++p) {
#pragma unroll 1
    for (size_t t = 0; t < tiles; ++t) {
      uint4* dst = tile[t & 1];
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const int idx = v * kThreads + threadIdx.x;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst + idx))),
                       "l"(my + t * kThreads * kVecs + idx));
      }
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 1;" ::: "memory");  // keep one tile in flight
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  __syncthreads();
  sink[blockIdx.x * kThreads + threadIdx.x] = tile[0][threadIdx.x].x ^ tile[1][threadIdx.x].y;
}

extern "C" __global__ void __launch_bounds__(kThreads)
mb_copy_ld_st(const uint4* src, size_t vecs_per_block, int passes, uint32_t* sink) {
  __shared__ __align__(16) uint4 tile[2][kThreads * kVecs];
  const uint4* my = src + blockIdx.x * vecs_per_block;
  const size_t tiles = vecs_per_block / (kThreads * kVecs);
#pragma unroll 1
  for (int p = 0; p < passes; ++p) {
#pragma unroll 1
    for (size_t t = 0; t < tiles; ++t) {
      uint4* dst = tile[t & 1];
      uint4 r[kVecs];
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const uint4* g = my + t * kThreads * kVecs + v * kThreads + threadIdx.x;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[v].x), "=r"(r[v].y), "=r"(r[v].z), "=r"(r[v].w) : "l"(g));
      }
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst + v * kThreads + threadIdx.x));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::"r"(s), "r"(r[v].x), "r"(r[v].y), "r"(r[v].z), "r"(r[v].w) : "memory");
      }
    }
  }
  __syncthreads();
  sink[blockIdx.x * kThreads + threadIdx.x] = tile[0][threadIdx.x].x ^ tile[1][threadIdx.x].y;
}

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_copy_cp_async", {{"LDGSTS", kVecs}});
  mb::prove("mb_copy_ld_st", {{"LDG", kVecs}, {"STS", kVecs}});

  const int sms = dev.prop.multiProcessorCount;
  const int blocks = sms * 3;  // 32 KiB smem per block
  uint32_t* sink;
  MB_CHECK(aegis::budget::gpu_malloc(&sink, size_t(blocks) * kThreads * sizeof(uint32_t)));

  struct Set { const char* name; size_t bytes; int passes; };
  const size_t l2_half = size_t(dev.prop.l2CacheSize) / 2;
  const Set sets[] = {{"dram", size_t(512) << 20, 8}, {"l2", l2_half, 200}};
  for (const Set& s : sets) {
    const size_t per_block = (s.bytes / blocks) / kTileBytes * kTileBytes;  // whole tiles
    const size_t total = per_block * blocks;
    uint4* src;
    MB_CHECK(aegis::budget::gpu_malloc(&src, total));
    MB_CHECK(cudaMemset(src, 0x5a, total));
    for (const char* mode : {"cp_async", "ld_st"}) {
      auto k = std::string(mode) == "cp_async" ? mb_copy_cp_async : mb_copy_ld_st;
      k<<<blocks, kThreads>>>(src, per_block / 16, 1, sink);  // warm-up (and fill L2 for "l2")
      const float ms = mb::time_ms([&] { k<<<blocks, kThreads>>>(src, per_block / 16, s.passes, sink); });
      const double gbps = double(total) * s.passes / (ms * 1e-3) / 1e9;
      mb::emit(dev, "gmem_to_smem_copy",
               std::string("{\"mode\":\"") + mode + "\",\"working_set\":\"" + s.name +
                   "\",\"working_set_bytes\":" + std::to_string(total) +
                   ",\"blocks\":" + std::to_string(blocks) + ",\"vec_bytes\":16}",
               gbps, "GB/s");
    }
    MB_CHECK(aegis::budget::gpu_free(src));
  }
  return 0;
}
