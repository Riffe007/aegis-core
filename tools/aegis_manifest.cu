// aegis_manifest: print a JSON description of the local GPU, including a
// MEASURED device-memory bandwidth. Rooflines use this number, not datasheets.
//
//   ./build/aegis_manifest > results/manifest-$(hostname).json
#include <algorithm>
#include <cstdio>
#include <vector>
#include <unistd.h>

#include "aegis/common.cuh"

int main() {
  int dev = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  AEGIS_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int rt = 0, drv = 0;
  AEGIS_CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  AEGIS_CUDA_CHECK(cudaDriverGetVersion(&drv));

  // Measured bandwidth: device-to-device copy of 512 MiB (read + write counted).
  // Unified-memory Jetsons share this bandwidth with the CPU; run with the
  // system otherwise idle.
  const size_t bytes = size_t(512) << 20;
  void *src, *dst;
  AEGIS_CUDA_CHECK(cudaMalloc(&src, bytes));
  AEGIS_CUDA_CHECK(cudaMalloc(&dst, bytes));
  AEGIS_CUDA_CHECK(cudaMemset(src, 1, bytes));
  cudaEvent_t e0, e1;
  AEGIS_CUDA_CHECK(cudaEventCreate(&e0));
  AEGIS_CUDA_CHECK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; ++i) AEGIS_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  std::vector<float> ms(10);
  for (auto& t : ms) {
    AEGIS_CUDA_CHECK(cudaEventRecord(e0));
    AEGIS_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice));
    AEGIS_CUDA_CHECK(cudaEventRecord(e1));
    AEGIS_CUDA_CHECK(cudaEventSynchronize(e1));
    AEGIS_CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
  }
  std::sort(ms.begin(), ms.end());
  const double bw_gbps = 2.0 * bytes / (ms[ms.size() / 2] * 1e-3) / 1e9;

  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  std::printf(
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
      "}\n",
      host, p.name, p.major, p.minor, p.multiProcessorCount,
      p.integrated ? "true" : "false", p.totalGlobalMem, p.l2CacheSize,
      p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor,
      p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor, bw_gbps, rt, drv);

  cudaFree(src);
  cudaFree(dst);
  return 0;
}
