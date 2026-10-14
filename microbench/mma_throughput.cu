// Probe: mma.sync m16n8k16 (f16 in, f32 accumulate) throughput.
//
// One block per SM, W warps per block, each warp keeping C independent
// accumulator chains so HMMAs can overlap. Sweeps W x C; the best point is the
// sustained tensor-core rate this device actually delivers.
// Timed loop: C x 4 independent HMMAs per iteration (4 rounds of C chains).
// Output: probe "mma_throughput": TFLOPS (events) and HMMA/clk/SM (in-kernel
// cycles of block 0).
#include "common.cuh"
#include "mma_common.cuh"

namespace {
constexpr int kRounds = 4;
}

template <int C>
__device__ __forceinline__ void tput_body(uint32_t seed, int iters, float* sink, long long* cycles) {
  const uint32_t a[4] = {seed, seed ^ 1u, seed ^ 2u, seed ^ 3u};
  const uint32_t b0 = seed ^ 4u, b1 = seed ^ 5u;
  float d[C][4] = {};
  __syncthreads();
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int r = 0; r < kRounds; ++r)
#pragma unroll
      for (int c = 0; c < C; ++c) mma_16816(d[c], a, b0, b1);
  }
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

#define MB_TPUT(C)                                                                         \
  extern "C" __global__ void mb_mma_tput_c##C(uint32_t seed, int iters, float* sink,       \
                                              long long* cycles) {                         \
    tput_body<C>(seed, iters, sink, cycles);                                               \
  }
MB_TPUT(1)
MB_TPUT(2)
MB_TPUT(4)
MB_TPUT(8)

int main() {
  const mb::Device dev = mb::device();
  using Kernel = void (*)(uint32_t, int, float*, long long*);
  const std::vector<std::tuple<int, Kernel, const char*>> kernels = {
      {1, mb_mma_tput_c1, "mb_mma_tput_c1"}, {2, mb_mma_tput_c2, "mb_mma_tput_c2"},
      {4, mb_mma_tput_c4, "mb_mma_tput_c4"}, {8, mb_mma_tput_c8, "mb_mma_tput_c8"}};
  for (const auto& [c, k, name] : kernels) mb::prove(name, {{"HMMA", c * kRounds}});

  const int sms = dev.prop.multiProcessorCount;
  float* sink;
  long long* dcycles;
  MB_CHECK(cudaMalloc(&sink, size_t(sms) * 1024 * sizeof(float)));
  MB_CHECK(cudaMalloc(&dcycles, sizeof(long long)));
  const int iters = 4096;

  for (int warps : {1, 2, 4, 8, 16, 32}) {
    for (const auto& [c, k, name] : kernels) {
      const int threads = warps * 32;
      k<<<sms, threads>>>(0x3c003c00u, 16, sink, dcycles);  // warm-up
      const float ms = mb::time_ms([&] { k<<<sms, threads>>>(0x3c003c00u, iters, sink, dcycles); });
      long long cycles = 0;
      MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
      const double mmas = double(sms) * warps * iters * kRounds * c;
      const std::string param = "{\"warps_per_sm\":" + std::to_string(warps) +
                                ",\"chains\":" + std::to_string(c) + ",\"sms\":" + std::to_string(sms) + "}";
      mb::emit(dev, "mma_throughput", param, mmas * kMmaFlops / (ms * 1e-3) / 1e12, "TFLOPS");
      mb::emit(dev, "mma_throughput", param, double(warps) * iters * kRounds * c / cycles, "HMMA/clk/SM");
    }
  }
  return 0;
}
