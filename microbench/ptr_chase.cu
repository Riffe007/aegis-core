// Probe: pointer-chase load latency vs. working set, 4 KiB .. 512 MiB.
//
// One thread follows a random single-cycle chain of 128-byte-spaced nodes, so
// every load's address depends on the previous load and prefetch cannot help.
// Loads are ld.global.ca (L1-cacheable), so the curve shows L1, L2 and DRAM.
//
// Two chain layouts over the same set of lines (same cache footprint):
//   cross_page      one random cycle over all lines: consecutive loads land in
//                   different pages, so TLB reach shows up as soon as the set
//                   exceeds it.
//   page_local_64k  chunks of 64 KiB visited in random order, lines inside a
//                   chunk in random order: one new page per 512 loads, so
//                   TLB misses are amortized and the curve tracks cache capacity.
// Each (layout, size) is measured REPEATS times with a fresh permutation (new
// seed) per repeat; the median and p10/p90 across repeats are reported.
//
// Timed loop: 16 dependent LDGs per iteration.
// Output: probe "ptr_chase_latency", param {working_set_bytes, variant, stat,
// repeats}: ns for median/p10/p90, cycles for the median.
#include <numeric>
#include <random>

#include "common.cuh"

namespace {
constexpr int kUnroll = 16;
constexpr size_t kStride = 128;           // bytes between nodes: one line per node
constexpr size_t kChunk = size_t(64) << 10;  // page_local chunk
constexpr int kRepeats = 5;
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

namespace {

// Visit order of node indices (one cycle through all of them).
std::vector<uint32_t> visit_order(size_t nodes, bool page_local, std::mt19937_64& rng) {
  std::vector<uint32_t> order(nodes);
  std::iota(order.begin(), order.end(), 0u);
  if (!page_local) {
    std::shuffle(order.begin(), order.end(), rng);
    return order;
  }
  const size_t per_chunk = std::max<size_t>(1, kChunk / kStride);
  const size_t chunks = (nodes + per_chunk - 1) / per_chunk;
  std::vector<uint32_t> chunk_order(chunks);
  std::iota(chunk_order.begin(), chunk_order.end(), 0u);
  std::shuffle(chunk_order.begin(), chunk_order.end(), rng);
  std::vector<uint32_t> out;
  out.reserve(nodes);
  for (uint32_t c : chunk_order) {
    const size_t lo = c * per_chunk, hi = std::min(nodes, lo + per_chunk);
    std::vector<uint32_t> lines(order.begin() + lo, order.begin() + hi);
    std::shuffle(lines.begin(), lines.end(), rng);
    out.insert(out.end(), lines.begin(), lines.end());
  }
  return out;
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo = size_t(idx), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

}  // namespace

int main() {
  const mb::Device dev = mb::device();
  mb::prove("mb_ptr_chase", {{"LDG", kUnroll}});
  const double mhz = mb::sm_clock_mhz();

  unsigned long long* sink;
  long long* dcycles;
  MB_CHECK(aegis::budget::gpu_malloc(&sink, sizeof *sink));
  MB_CHECK(aegis::budget::gpu_malloc(&dcycles, sizeof *dcycles));
  const int steps = 1 << 17;

  for (const bool page_local : {false, true}) {
    const char* variant = page_local ? "page_local_64k" : "cross_page";
    int ws_index = 0;
    for (size_t ws = size_t(4) << 10; ws <= size_t(512) << 20; ws *= 2, ++ws_index) {
      const size_t nodes = ws / kStride;
      char* buf;
      MB_CHECK(aegis::budget::gpu_malloc(&buf, ws));
      const auto base = reinterpret_cast<unsigned long long>(buf);
      std::vector<unsigned long long> host(ws / sizeof(unsigned long long), 0);
      std::vector<double> cyc;
      for (int rep = 0; rep < kRepeats; ++rep) {
        std::mt19937_64 rng(1000003ull * (rep + 1) + 7919ull * ws_index + (page_local ? 17 : 0));
        const std::vector<uint32_t> order = visit_order(nodes, page_local, rng);
        for (size_t i = 0; i < nodes; ++i) {
          const uint32_t from = order[i], to = order[(i + 1) % nodes];
          host[from * kStride / sizeof(unsigned long long)] = base + to * kStride;
        }
        MB_CHECK(cudaMemcpy(buf, host.data(), ws, cudaMemcpyHostToDevice));
        const int warm = int(std::min<size_t>(nodes, size_t(1) << 18));
        mb_ptr_chase<<<1, 1>>>(reinterpret_cast<const unsigned long long*>(buf + order[0] * kStride),
                               warm, steps, sink, dcycles);
        MB_CHECK(cudaGetLastError());
        long long cycles = 0;
        MB_CHECK(cudaMemcpy(&cycles, dcycles, sizeof cycles, cudaMemcpyDeviceToHost));
        cyc.push_back(double(cycles) / steps);
      }
      MB_CHECK(aegis::budget::gpu_free(buf));
      const std::string head = "{\"working_set_bytes\":" + std::to_string(ws) + ",\"variant\":\"" +
                               variant + "\",\"repeats\":" + std::to_string(kRepeats) + ",\"stat\":";
      const double med = percentile(cyc, 0.5);
      mb::emit(dev, "ptr_chase_latency", head + "\"median\"}", med, "cycles");
      mb::emit(dev, "ptr_chase_latency", head + "\"median\"}", med / mhz * 1e3, "ns");
      mb::emit(dev, "ptr_chase_latency", head + "\"p10\"}", percentile(cyc, 0.1) / mhz * 1e3, "ns");
      mb::emit(dev, "ptr_chase_latency", head + "\"p90\"}", percentile(cyc, 0.9) / mhz * 1e3, "ns");
    }
  }
  mb::emit(dev, "sm_clock", "{}", mhz, "MHz");
  return 0;
}
