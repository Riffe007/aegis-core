// Probe: shared-memory bank-conflict sweep, 32-bit loads, stride 1..32 words.
//
// Lane l loads word l*stride; 32 banks of 4 B, so a warp's load is
// gcd(stride, 32)-way conflicted. A uniform per-iteration offset of 32 words
// keeps every lane on the same bank. One block per SM, 8 warps, loads
// XOR-folded so none is dead. The loads are ld.volatile: with plain ld.shared
// and loop-invariant addresses ptxas hoisted all 8 out of the timed loop (the
// SASS proof caught it); volatile forbids that and still compiles to LDS. cycles per warp-wide load ~ conflict degree if
// shared memory is the bottleneck.
// Timed loop: 8 LDS per iteration.
// Output: probe "smem_bank_conflict": cycles per warp load (per SM), with the
// predicted conflict degree in the params.
#include <cstdint>
#include <numeric>

#include "common.cuh"

namespace {
constexpr int kUnroll = 8;
constexpr int kWords = 32 * 32 + kUnroll * 32;
constexpr int kWarps = 8;
}  // namespace

extern "C" __global__ void mb_smem_banks(int stride, int iters, uint32_t* sink, long long* cycles) {
  __shared__ uint32_t smem[kWords];
  for (int i = threadIdx.x; i < kWords; i += blockDim.x) smem[i] = i * 2654435761u;
  __syncthreads();
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) +
                        (threadIdx.x % 32) * stride * 4;
  uint32_t acc = 0;
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t v;
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(base + j * 128));
      acc ^= v;
    }
  }
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_smem_banks", {{"LDS", kUnroll}});
  const int sms = dev.prop.multiProcessorCount;
  uint32_t* sink;
  long long* dcycles;
  MB_CHECK(cudaMalloc(&sink, size_t(sms) * kWarps * 32 * sizeof(uint32_t)));
  MB_CHECK(cudaMalloc(&dcycles, sizeof(long long)));
  const int iters = 1 << 14;
  for (int stride = 1; stride <= 32; ++stride) {
    mb_smem_banks<<<sms, kWarps * 32>>>(stride, 16, sink, dcycles);
    mb_smem_banks<<<sms, kWarps * 32>>>(stride, iters, sink, dcycles);
    MB_CHECK(cudaGetLastError());
    long long cycles = 0;
    MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
    const double warp_loads = double(kWarps) * iters * kUnroll;
    mb::emit(dev, "smem_bank_conflict",
             "{\"stride_words\":" + std::to_string(stride) +
                 ",\"predicted_ways\":" + std::to_string(std::gcd(stride, 32)) +
                 ",\"warps_per_sm\":" + std::to_string(kWarps) + "}",
             double(cycles) / warp_loads, "cycles/warp_load");
  }
  return 0;
}
