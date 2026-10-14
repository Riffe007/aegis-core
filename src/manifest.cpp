// Device manifest JSON (see include/aegis/manifest.h). Errors are returned,
// never fatal, because the C ABI calls this.
#include "aegis/manifest.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace aegis {
namespace {

struct DeviceGuard {  // restore the caller's current device on every path
  int prev = 0;
  bool ok = cudaGetDevice(&prev) == cudaSuccess;
  ~DeviceGuard() { if (ok) cudaSetDevice(prev); }
};

}  // namespace

cudaError_t device_manifest_json(int device, std::string& json) {
  DeviceGuard guard;
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) return err;

  cudaDeviceProp p{};
  int rt = 0, drv = 0;
  if ((err = cudaGetDeviceProperties(&p, device)) != cudaSuccess) return err;
  if ((err = cudaRuntimeGetVersion(&rt)) != cudaSuccess) return err;
  if ((err = cudaDriverGetVersion(&drv)) != cudaSuccess) return err;

  // Measured bandwidth: device-to-device copy of 512 MiB (read + write counted).
  // Unified-memory Jetsons share this bandwidth with the CPU; run with the
  // system otherwise idle.
  const size_t bytes = size_t(512) << 20;
  void *src = nullptr, *dst = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  std::vector<float> ms(10);
  auto measure = [&]() -> cudaError_t {
    cudaError_t e;
    if ((e = cudaMalloc(&src, bytes)) != cudaSuccess) return e;
    if ((e = cudaMalloc(&dst, bytes)) != cudaSuccess) return e;
    if ((e = cudaMemset(src, 1, bytes)) != cudaSuccess) return e;
    if ((e = cudaEventCreate(&e0)) != cudaSuccess) return e;
    if ((e = cudaEventCreate(&e1)) != cudaSuccess) return e;
    for (int i = 0; i < 3; ++i)
      if ((e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)) != cudaSuccess) return e;
    for (auto& t : ms) {
      if ((e = cudaEventRecord(e0)) != cudaSuccess) return e;
      if ((e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice)) != cudaSuccess) return e;
      if ((e = cudaEventRecord(e1)) != cudaSuccess) return e;
      if ((e = cudaEventSynchronize(e1)) != cudaSuccess) return e;
      if ((e = cudaEventElapsedTime(&t, e0, e1)) != cudaSuccess) return e;
    }
    return cudaSuccess;
  };
  err = measure();
  if (e0) cudaEventDestroy(e0);
  if (e1) cudaEventDestroy(e1);
  cudaFree(src);
  cudaFree(dst);
  if (err != cudaSuccess) return err;

  std::sort(ms.begin(), ms.end());
  const double bw_gbps = 2.0 * bytes / (ms[ms.size() / 2] * 1e-3) / 1e9;

  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  const char* fmt =
      "{\n"
      "  \"device_id\": \"%s\",\n"
      "  \"gpu_name\": \"%s\",\n"
      "  \"sm_arch\": %d%d,\n"
      "  \"sm_count\": %d,\n"
      "  \"integrated\": %s,\n"
      "  \"global_mem_bytes\": %zu,\n"
      "  \"l2_bytes\": %d,\n"
      "  \"smem_per_block_optin_bytes\": %zu,\n"
      "  \"smem_per_sm_bytes\": %zu,\n"
      "  \"regs_per_sm\": %d,\n"
      "  \"max_threads_per_sm\": %d,\n"
      "  \"measured_d2d_bw_gbps\": %.1f,\n"
      "  \"cuda_runtime\": %d,\n"
      "  \"cuda_driver\": %d\n"
      "}\n";
  auto print = [&](char* buf, size_t n) {
    return std::snprintf(buf, n, fmt, host, p.name, p.major, p.minor, p.multiProcessorCount,
                         p.integrated ? "true" : "false", p.totalGlobalMem, p.l2CacheSize,
                         p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor,
                         p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor, bw_gbps, rt, drv);
  };
  const int n = print(nullptr, 0);
  json.assign(size_t(n), '\0');
  print(json.data(), json.size() + 1);
  return cudaSuccess;
}

}  // namespace aegis
