#include "device.h"

#include <mutex>

namespace aegis::hw {
namespace {

thread_local const DeviceLimits* t_sim = nullptr;

struct Cache {
  std::once_flag once[kMaxCachedDevices];
  DeviceLimits limits[kMaxCachedDevices];
  bool ok[kMaxCachedDevices] = {};
};

Cache& cache() {
  static Cache c;
  return c;
}

}  // namespace

cudaError_t query(int dev, DeviceLimits* out) {
  cudaDeviceProp p{};
  if (const cudaError_t e = cudaGetDeviceProperties(&p, dev); e != cudaSuccess) return e;
  DeviceLimits d;
  d.device = dev;
  d.name = p.name;
  d.arch = p.major * 10 + p.minor;
  d.sm_count = p.multiProcessorCount;
  d.smem_per_sm = p.sharedMemPerMultiprocessor;
  d.smem_per_block = p.sharedMemPerBlock;
  d.smem_per_block_optin = p.sharedMemPerBlockOptin;
  d.reserved_smem_per_block = p.reservedSharedMemPerBlock;
  d.regs_per_sm = p.regsPerMultiprocessor;
  d.regs_per_block = p.regsPerBlock;
  d.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
  d.max_blocks_per_sm = p.maxBlocksPerMultiProcessor;
  d.l2_bytes = size_t(p.l2CacheSize);
  d.global_mem_bytes = p.totalGlobalMem;
  d.unified = p.integrated != 0;
  *out = d;
  return cudaSuccess;
}

const DeviceLimits* physical() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  if (dev < 0 || dev >= kMaxCachedDevices) {
    thread_local DeviceLimits uncached;
    if (query(dev, &uncached) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    return &uncached;
  }
  Cache& c = cache();
  std::call_once(c.once[dev], [&] {
    c.ok[dev] = query(dev, &c.limits[dev]) == cudaSuccess;
    if (!c.ok[dev]) cudaGetLastError();
  });
  return c.ok[dev] ? &c.limits[dev] : nullptr;
}

const DeviceLimits* current() { return t_sim ? t_sim : physical(); }

ScopedSimulation::ScopedSimulation(const DeviceLimits& sim) : sim_(sim), prev_(t_sim) {
  sim_.simulated = true;
  t_sim = &sim_;
}

ScopedSimulation::~ScopedSimulation() { t_sim = prev_; }

}  // namespace aegis::hw
