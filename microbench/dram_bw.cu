// Probe: DRAM bandwidth, device-to-device copy of 512 MiB (the aegis_manifest
// method), repeated so the spread is recorded.
//
// Each sample: cudaMemcpyAsync D2D of 512 MiB between two buffers far larger
// than L2, bracketed by events; bandwidth = (read + write bytes) / time. One
// rep = the median of 20 copies after 3 warm-ups; kReps reps are emitted plus
// their median. No kernel of ours runs, so there is no SASS proof.
//
// Output: probe "dram_bandwidth", param {"method":"memcpy_d2d","bytes":B,
// "rep":i} (or "stat":"median"), GB/s.
#include <algorithm>
#include <vector>

#include "common.cuh"

namespace {
constexpr size_t kBytes = size_t(512) << 20;
constexpr int kCopies = 20, kWarmup = 3, kReps = 5;
}  // namespace

int main() {
  const mb::Device dev = mb::device();
  void *src = nullptr, *dst = nullptr;
  MB_CHECK(cudaMalloc(&src, kBytes));
  MB_CHECK(cudaMalloc(&dst, kBytes));
  MB_CHECK(cudaMemset(src, 1, kBytes));
  cudaEvent_t a, b;
  MB_CHECK(cudaEventCreate(&a));
  MB_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < kWarmup; ++i) MB_CHECK(cudaMemcpy(dst, src, kBytes, cudaMemcpyDeviceToDevice));

  const std::string head = "{\"method\":\"memcpy_d2d\",\"bytes\":" + std::to_string(kBytes);
  std::vector<double> reps;
  for (int r = 0; r < kReps; ++r) {
    std::vector<float> ms(kCopies);
    for (int i = 0; i < kCopies; ++i) {
      MB_CHECK(cudaEventRecord(a));
      MB_CHECK(cudaMemcpyAsync(dst, src, kBytes, cudaMemcpyDeviceToDevice));
      MB_CHECK(cudaEventRecord(b));
      MB_CHECK(cudaEventSynchronize(b));
      MB_CHECK(cudaEventElapsedTime(&ms[size_t(i)], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double gbps = 2.0 * double(kBytes) / (double(ms[ms.size() / 2]) * 1e-3) / 1e9;
    reps.push_back(gbps);
    mb::emit(dev, "dram_bandwidth", head + ",\"rep\":" + std::to_string(r) + "}", gbps, "GB/s");
  }
  std::sort(reps.begin(), reps.end());
  mb::emit(dev, "dram_bandwidth", head + ",\"stat\":\"median\"}", reps[reps.size() / 2], "GB/s");
  return 0;
}
