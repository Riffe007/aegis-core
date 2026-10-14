// Probe: ldmatrix.sync.aligned.m8n8.x4.shared.b16 throughput (PTX ISA 9.4
// 9.7.16.5.15, sm_75+), the A-fragment load of the tier 3/4 GEMMs.
//
// One block per SM, W warps. Lane l supplies row address l*16 B, so each
// 8-lane phase reads one 128 B line: conflict-free by construction. A uniform
// per-iteration offset (multiple of 512 B) changes the address without
// changing banks. Results are XOR-folded so no load is dead.
// Timed loop: 8 LDSM per iteration.
// Output: probe "ldmatrix_x4_throughput": bytes/clk/SM and GB/s (all SMs).
// `mb_ldmatrix --warps N` runs only that point (for ncu; see profile_ldmatrix.sh).
#include <cstdint>
#include <cstring>
#include <vector>

#include "common.cuh"

namespace {
constexpr int kUnroll = 8;
constexpr int kSmemBytes = 8 * 512;
}  // namespace

extern "C" __global__ void mb_ldmatrix_x4(int iters, uint32_t* sink, long long* cycles) {
  __shared__ __align__(128) uint8_t smem[kSmemBytes];
  for (int i = threadIdx.x; i < kSmemBytes; i += blockDim.x) smem[i] = uint8_t(i);
  __syncthreads();
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + (threadIdx.x % 32) * 16;
  uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                   : "r"(base + ((j * 512) % kSmemBytes)));
      acc0 ^= r0; acc1 ^= r1; acc2 ^= r2; acc3 ^= r3;
    }
  }
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc0 ^ acc1 ^ acc2 ^ acc3;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

int main(int argc, char** argv) {
  std::vector<int> points = {1, 2, 4, 8, 16, 32};
  if (argc == 3 && std::strcmp(argv[1], "--warps") == 0) points = {std::atoi(argv[2])};
  const mb::Device dev = mb::device();
  mb::prove("mb_ldmatrix_x4", {{"LDSM", kUnroll}});
  const int sms = dev.prop.multiProcessorCount;
  uint32_t* sink;
  long long* dcycles;
  MB_CHECK(cudaMalloc(&sink, size_t(sms) * 1024 * sizeof(uint32_t)));
  MB_CHECK(cudaMalloc(&dcycles, sizeof(long long)));
  const int iters = 1 << 14;
  for (int warps : points) {
    mb_ldmatrix_x4<<<sms, warps * 32>>>(16, sink, dcycles);
    const float ms = mb::time_ms([&] { mb_ldmatrix_x4<<<sms, warps * 32>>>(iters, sink, dcycles); });
    long long cycles = 0;
    MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
    const double bytes_per_sm = double(warps) * iters * kUnroll * 512;  // x4: 4 x 8x8 x 2 B
    const std::string param = "{\"warps_per_sm\":" + std::to_string(warps) + "}";
    mb::emit(dev, "ldmatrix_x4_throughput", param, bytes_per_sm / cycles, "B/clk/SM");
    mb::emit(dev, "ldmatrix_x4_throughput", param, bytes_per_sm * sms / (ms * 1e-3) / 1e9, "GB/s");
  }
  return 0;
}
