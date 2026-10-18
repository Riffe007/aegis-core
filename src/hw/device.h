#pragma once
// Runtime device limits (aegis-hw step 1). Every launch-sizing decision in Core
// reads these, never a constant tuned on one device:
// - split-K S and grids (splitk.cuh);
// - dynamic smem limits and opt-in;
// - TMA eligibility, the dispatch arch and the GPU budget defaults.
// One arch can come with different SM counts (Orin Nano and AGX Orin are both
// sm_87), so nothing keyed only by arch may stand in for these.
//
// Queried once per device (cudaGetDeviceProperties) and cached for the process.
// Tests can substitute a simulated profile per thread (ScopedSimulation).
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>

namespace aegis::hw {

// Devices cached per process; a device id beyond this is queried on every call.
inline constexpr int kMaxCachedDevices = 64;

struct DeviceLimits {
  int device = -1;
  int arch = 0;                       // major * 10 + minor
  int sm_count = 0;                   // multiProcessorCount
  size_t smem_per_sm = 0;             // sharedMemPerMultiprocessor
  size_t smem_per_block = 0;          // sharedMemPerBlock: the limit without opt-in
  size_t smem_per_block_optin = 0;    // sharedMemPerBlockOptin: with cudaFuncAttributeMaxDynamicSharedMemorySize
  size_t reserved_smem_per_block = 0; // reservedSharedMemPerBlock (runtime-reserved, per resident block)
  int regs_per_sm = 0;                // regsPerMultiprocessor
  int regs_per_block = 0;             // regsPerBlock
  int max_threads_per_sm = 0;         // maxThreadsPerMultiProcessor
  int max_blocks_per_sm = 0;          // maxBlocksPerMultiProcessor
  size_t l2_bytes = 0;                // l2CacheSize
  size_t global_mem_bytes = 0;        // totalGlobalMem
  bool unified = false;               // integrated: device memory is host memory
  bool simulated = false;             // from ScopedSimulation, not the device
};

// Query device `dev` (uncached).
cudaError_t query(int dev, DeviceLimits* out);

// The current device's limits (cached per device), or the calling thread's
// simulated profile while a ScopedSimulation is alive. nullptr if the device
// cannot be queried (the CUDA error is cleared).
const DeviceLimits* current();
// The real current device, ignoring any simulation (for what acts on the device).
const DeviceLimits* physical();

// Tests: while alive, current() on this thread returns `sim` (marked simulated).
// Launches still run on the real device; only sizing decisions see `sim`.
class ScopedSimulation {
 public:
  explicit ScopedSimulation(const DeviceLimits& sim);
  ~ScopedSimulation();
  ScopedSimulation(const ScopedSimulation&) = delete;
  ScopedSimulation& operator=(const ScopedSimulation&) = delete;

 private:
  DeviceLimits sim_;
  const DeviceLimits* prev_;
};

// Per-device once: `fn()` (a cudaError_t, e.g. cudaFuncSetAttribute, which acts on
// the current device) runs the first time per device and its result is returned on
// every later call there. A race may run fn twice on one device; fn must be idempotent.
struct PerDevice {
  std::atomic<int> state[kMaxCachedDevices] = {};  // 0: not yet; else result + 1
};

template <class Fn>
cudaError_t once_per_device(PerDevice& once, Fn&& fn) {
  int dev = 0;
  if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) return e;
  if (dev < 0 || dev >= kMaxCachedDevices) return fn();
  const int s = once.state[dev].load(std::memory_order_acquire);
  if (s != 0) return cudaError_t(s - 1);
  const cudaError_t e = fn();
  once.state[dev].store(int(e) + 1, std::memory_order_release);
  return e;
}

// Sets fn's dynamic smem limit to `bytes`, once per device.
// cudaErrorInvalidValue if `bytes` exceeds the device's opt-in limit.
template <class Kern>
cudaError_t configure_smem(PerDevice& once, Kern* kernel, size_t bytes) {
  return once_per_device(once, [&] {
    const DeviceLimits* d = physical();
    if (d && bytes > d->smem_per_block_optin) return cudaErrorInvalidValue;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  });
}

}  // namespace aegis::hw
