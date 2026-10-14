// Probe: pointer-chase load latency vs. working set, 4 KiB .. 512 MiB.
//
// One thread follows a random single-cycle permutation of 128-byte-spaced
// nodes (Sattolo), so every load's address depends on the previous load and
// hardware prefetch cannot help. Loads are ld.global.ca (L1-cacheable), so the
// curve shows L1, L2 and DRAM plateaus. Large working sets also pay TLB misses.
//
// Timed loop: 16 dependent LDGs per iteration.
// Output: probe "ptr_chase_latency", param {"working_set_bytes"}, in cycles
// and in ns (via the measured SM clock).
#include <numeric>
#include <random>

#include "common.cuh"

namespace {
constexpr int kUnroll = 16;
constexpr size_t kStride = 128;  // bytes between nodes: one L1/L2 line per node
}  // namespace

__device__ __forceinline__ const unsigned long long* chase(const unsigned long long* p) {
  unsigned long long next;
  asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(next) : "l"(p));
  return reinterpret_cast<const unsigned long long*>(next);
}

extern "C" __global__ void mb_ptr_chase(const unsigned long long* start, int warm, int steps,
                                        unsigned long long* sink, long long* cycles) {
  const unsigned long long* p = start;
#pragma unroll 1
  for (int i = 0; i < warm; ++i) p = chase(p);  // bring the set into cache, same SM
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < steps; i += kUnroll) {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) p = chase(p);
  }
  const long long t1 = mb::clock64_volatile();
  *sink = reinterpret_cast<unsigned long long>(p);  // keeps the chain live
  *cycles = t1 - t0;
}

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_ptr_chase", {{"LDG", kUnroll}});
  const double mhz = mb::sm_clock_mhz();

  unsigned long long* sink;
  long long* dcycles;
  MB_CHECK(cudaMalloc(&sink, sizeof *sink));
  MB_CHECK(cudaMalloc(&dcycles, sizeof *dcycles));
  std::mt19937_64 rng(42);
  const int steps = 1 << 17;

  for (size_t ws = size_t(4) << 10; ws <= size_t(512) << 20; ws *= 2) {
    const size_t nodes = ws / kStride;
    // Sattolo: a uniformly random single cycle through all nodes.
    std::vector<uint32_t> next(nodes);
    std::iota(next.begin(), next.end(), 0u);
    for (size_t i = nodes - 1; i > 0; --i) {
      std::uniform_int_distribution<size_t> pick(0, i - 1);
      std::swap(next[i], next[pick(rng)]);
    }
    char* buf;
    MB_CHECK(cudaMalloc(&buf, ws));
    std::vector<unsigned long long> host(ws / sizeof(unsigned long long), 0);
    const auto base = reinterpret_cast<unsigned long long>(buf);
    for (size_t i = 0; i < nodes; ++i)
      host[i * kStride / sizeof(unsigned long long)] = base + next[i] * kStride;
    MB_CHECK(cudaMemcpy(buf, host.data(), ws, cudaMemcpyHostToDevice));

    const int warm = int(std::min<size_t>(nodes, size_t(1) << 18));
    mb_ptr_chase<<<1, 1>>>(reinterpret_cast<const unsigned long long*>(buf), warm, steps, sink, dcycles);
    MB_CHECK(cudaGetLastError());
    long long cycles = 0;
    MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
    MB_CHECK(cudaFree(buf));

    const double per_load = double(cycles) / steps;
    const std::string param = "{\"working_set_bytes\":" + std::to_string(ws) + "}";
    mb::emit(dev, "ptr_chase_latency", param, per_load, "cycles");
    mb::emit(dev, "ptr_chase_latency", param, per_load / mhz * 1e3, "ns");
  }
  mb::emit(dev, "sm_clock", "{}", mhz, "MHz");
  return 0;
}
