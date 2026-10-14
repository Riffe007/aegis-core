// Probe: DRAM bandwidth, two streams, each repeated so the spread is recorded.
//
//   memcpy_d2d  cudaMemcpyAsync D2D of 512 MiB between two buffers far larger
//               than L2 (the aegis_manifest method); bandwidth = (read + write
//               bytes) / time. No kernel of ours, so no SASS proof.
//   read_v4     a read-only stream: every thread issues 4 independent 128-bit
//               global loads per iteration (ld.global.v4.u32) over a 512 MiB
//               buffer and folds them into an XOR reduction stored once per
//               thread; bandwidth = bytes read / time. SASS proof: the timed
//               loop holds exactly 4 LDG.E.128.
// One rep = the median of 20 timed runs after 3 warm-ups; kReps reps are
// emitted plus their median.
//
// Output: probe "dram_bandwidth", param {"method","bytes","rep"} (or
// "stat":"median"), GB/s.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "common.cuh"

namespace {
constexpr size_t kBytes = size_t(512) << 20;
constexpr int kRuns = 20, kWarmup = 3, kReps = 5;
constexpr int kThreads = 512, kBlocksPerSm = 4;  // 2048 threads/SM
}  // namespace

__device__ __forceinline__ uint32_t ld_v4_xor(const uint4* p) {
  uint32_t x, y, z, w;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(p));
  return x ^ y ^ z ^ w;
}

extern "C" __global__ void mb_dram_read_v4(const uint4* __restrict__ p, size_t n, uint32_t* sink) {
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll 1
  for (; i + 3 * stride < n; i += 4 * stride) {  // timed loop: 4 independent loads
    a0 ^= ld_v4_xor(p + i);
    a1 ^= ld_v4_xor(p + i + stride);
    a2 ^= ld_v4_xor(p + i + 2 * stride);
    a3 ^= ld_v4_xor(p + i + 3 * stride);
  }
#pragma unroll 1
  for (; i < n; i += stride) a0 ^= ld_v4_xor(p + i);  // tail
  sink[size_t(blockIdx.x) * blockDim.x + threadIdx.x] = a0 ^ a1 ^ a2 ^ a3;
}

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_dram_read_v4", {{"LDG.E.128", 4}});

  void *src = nullptr, *dst = nullptr;
  MB_CHECK(cudaMalloc(&src, kBytes));
  MB_CHECK(cudaMalloc(&dst, kBytes));
  MB_CHECK(cudaMemset(src, 1, kBytes));
  const int blocks = dev.prop.multiProcessorCount * kBlocksPerSm;
  uint32_t* sink = nullptr;
  MB_CHECK(cudaMalloc(&sink, size_t(blocks) * kThreads * sizeof(uint32_t)));
  cudaEvent_t a, b;
  MB_CHECK(cudaEventCreate(&a));
  MB_CHECK(cudaEventCreate(&b));

  struct Stream {
    const char* method;
    double bytes_moved;  // per run
    std::function<void()> run;
  };
  const Stream streams[] = {
      {"memcpy_d2d", 2.0 * double(kBytes),
       [&] { MB_CHECK(cudaMemcpyAsync(dst, src, kBytes, cudaMemcpyDeviceToDevice)); }},
      {"read_v4", double(kBytes),
       [&] {
         mb_dram_read_v4<<<blocks, kThreads>>>(static_cast<const uint4*>(src), kBytes / sizeof(uint4), sink);
         MB_CHECK(cudaGetLastError());
       }},
  };
  for (const Stream& s : streams) {
    for (int i = 0; i < kWarmup; ++i) s.run();
    MB_CHECK(cudaDeviceSynchronize());
    const std::string head = std::string("{\"method\":\"") + s.method + "\",\"bytes\":" + std::to_string(kBytes);
    std::vector<double> reps;
    for (int r = 0; r < kReps; ++r) {
      std::vector<float> ms(kRuns);
      for (int i = 0; i < kRuns; ++i) {
        MB_CHECK(cudaEventRecord(a));
        s.run();
        MB_CHECK(cudaEventRecord(b));
        MB_CHECK(cudaEventSynchronize(b));
        MB_CHECK(cudaEventElapsedTime(&ms[size_t(i)], a, b));
      }
      std::sort(ms.begin(), ms.end());
      const double gbps = s.bytes_moved / (double(ms[ms.size() / 2]) * 1e-3) / 1e9;
      reps.push_back(gbps);
      mb::emit(dev, "dram_bandwidth", head + ",\"rep\":" + std::to_string(r) + "}", gbps, "GB/s");
    }
    std::sort(reps.begin(), reps.end());
    mb::emit(dev, "dram_bandwidth", head + ",\"stat\":\"median\"}", reps[reps.size() / 2], "GB/s");
  }
  return 0;
}
