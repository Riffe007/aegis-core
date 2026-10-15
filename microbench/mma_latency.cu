// Probe: mma.sync m16n8k16 (f16 in, f32 accumulate) dependent-chain latency.
//
// One warp; every MMA accumulates into the previous one's result, so each
// waits for the last to finish. cycles / MMAs = issue-to-result latency.
// Timed loop: 8 dependent HMMAs per iteration. Rows: cycles, with the SM clock
// and whether clocks were locked (clock_lock.cuh).
#include "clock_lock.cuh"
#include "common.cuh"
#include "mma_common.cuh"

namespace {
constexpr int kUnroll = 8;
}

extern "C" __global__ void mb_mma_latency(uint32_t seed, int iters, float* sink, long long* cycles) {
  const uint32_t a[4] = {seed, seed ^ 1u, seed ^ 2u, seed ^ 3u};
  const uint32_t b0 = seed ^ 4u, b1 = seed ^ 5u;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  __syncwarp();
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; i += kUnroll) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) mma_16816(d, a, b0, b1);
  }
  const long long t1 = mb::clock64_volatile();
  sink[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
  if (threadIdx.x == 0) *cycles = t1 - t0;
}

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_mma_latency", {{"HMMA", kUnroll}});
  float* sink;
  long long* dcycles;
  MB_CHECK(aegis::budget::gpu_malloc(&sink, 32 * sizeof(float)));
  MB_CHECK(aegis::budget::gpu_malloc(&dcycles, sizeof(long long)));
  const int iters = 1 << 16;
  mb::ClockLock clk;  // request locked clocks; recorded on every row
  std::vector<double> per_mma;
  mb_mma_latency<<<1, 32>>>(0x3c003c00u, iters, sink, dcycles);  // warm-up
  MB_CHECK(cudaDeviceSynchronize());
  for (int rep = 0; rep < 3; ++rep) {
    mb_mma_latency<<<1, 32>>>(0x3c003c00u, iters, sink, dcycles);
    MB_CHECK(cudaGetLastError());
    long long cycles = 0;
    MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
    per_mma.push_back(double(cycles) / iters);
  }
  clk.finish();
  for (size_t rep = 0; rep < per_mma.size(); ++rep)
    mb::emit_clocked(dev, "mma_latency",
                     "{\"shape\":\"m16n8k16\",\"types\":\"f32.f16.f16.f32\",\"rep\":" + std::to_string(rep) + "}",
                     per_mma[rep], "cycles", clk);
  return 0;
}
