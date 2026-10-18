#pragma once
// In-process GPU allocation budget for every bench / tool / test / microbench
// binary (CLAUDE.md rule 13). GB10 memory is unified: device allocations come
// out of host RAM, and the driver does not charge them to the process's
// cgroup, so a MemoryMax cap does not stop them. This wrapper does:
//
//   aegis::budget::gpu_malloc(&p, bytes)  refuses (cudaErrorMemoryAllocation,
//                                         reason on stderr) when
//     - the process's tracked device bytes + bytes > AEGIS_GPU_BUDGET
//       (default: min(16 GiB, total / 2)), or
//     - cudaMemGetInfo's free memory minus bytes < AEGIS_GPU_MIN_FREE
//       (default: min(16 GiB, total / 4)): the system must keep at least that
//       much free after the allocation (so a call is also refused whenever less
//       than that is free to begin with);
//   total = cudaMemGetInfo's total of the device (16 GiB each on GB10, 128 GB;
//   an 8 GB Orin Nano gets fractions of its own memory, not a 16 GiB floor it
//   could never meet);
//   aegis::budget::gpu_free(p)            frees and untracks.
// Sizes accept a K/M/G/T suffix (binary units), e.g. AEGIS_GPU_BUDGET=4G.
// Only allocations made through these functions count; library code
// (aegis::matmul, the C ABI) does not allocate on the call path.
#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aegis::budget {

inline constexpr size_t kGiB = size_t(1) << 30;
inline constexpr size_t kDefaultCap = 16 * kGiB;  // the defaults never exceed this

// The defaults for a device with `total` bytes (cudaMemGetInfo; 0, not queryable: the cap).
inline size_t default_budget(size_t total) { return total && total / 2 < kDefaultCap ? total / 2 : kDefaultCap; }
inline size_t default_min_free(size_t total) { return total && total / 4 < kDefaultCap ? total / 4 : kDefaultCap; }

// The device's total memory (cudaMemGetInfo), or 0 if it cannot be queried.
inline size_t device_total() {
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return total_b;
}

// Parses "123", "64M", "16G" (binary units). Returns fallback if unset or bad.
inline size_t parse_size(const char* s, size_t fallback) {
  if (!s || !*s) return fallback;
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s || v < 0) return fallback;
  double mul = 1;
  switch (std::toupper(static_cast<unsigned char>(*end))) {
    case 'K': mul = 1024.0; break;
    case 'M': mul = 1024.0 * 1024; break;
    case 'G': mul = double(kGiB); break;
    case 'T': mul = double(kGiB) * 1024; break;
    case '\0': break;
    default: return fallback;
  }
  return size_t(v * mul);
}

struct State {
  std::mutex mu;
  std::unordered_map<void*, size_t> live;
  size_t in_use = 0;
  size_t total = device_total();
  size_t budget = parse_size(std::getenv("AEGIS_GPU_BUDGET"), default_budget(total));
  size_t min_free = parse_size(std::getenv("AEGIS_GPU_MIN_FREE"), default_min_free(total));
};

inline State& state() {
  static State s;
  return s;
}

inline size_t in_use() {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  return s.in_use;
}

inline cudaError_t gpu_malloc(void** p, size_t bytes, const char* what = "device buffer") {
  *p = nullptr;
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mu);
  auto gib = [](size_t b) { return double(b) / double(kGiB); };
  if (s.in_use + bytes > s.budget) {
    std::fprintf(stderr,
                 "aegis gpu budget: refused %.3f GiB for %s: %.3f GiB already in use, budget %.3f GiB "
                 "(AEGIS_GPU_BUDGET)\n",
                 gib(bytes), what, gib(s.in_use), gib(s.budget));
    return cudaErrorMemoryAllocation;
  }
  size_t free_b = 0, total_b = 0;
  const cudaError_t q = cudaMemGetInfo(&free_b, &total_b);
  if (q != cudaSuccess) return q;
  if (free_b < bytes || free_b - bytes < s.min_free) {
    std::fprintf(stderr,
                 "aegis gpu budget: refused %.3f GiB for %s: %.3f GiB free system-wide, must stay >= "
                 "%.3f GiB (AEGIS_GPU_MIN_FREE)\n",
                 gib(bytes), what, gib(free_b), gib(s.min_free));
    return cudaErrorMemoryAllocation;
  }
  const cudaError_t e = cudaMalloc(p, bytes);
  if (e == cudaSuccess) {
    s.live[*p] = bytes;
    s.in_use += bytes;
  }
  return e;
}

template <class T>
cudaError_t gpu_malloc(T** p, size_t bytes, const char* what = "device buffer") {
  return gpu_malloc(reinterpret_cast<void**>(p), bytes, what);
}

inline cudaError_t gpu_free(void* p) {
  if (!p) return cudaSuccess;
  State& s = state();
  {
    std::lock_guard<std::mutex> lock(s.mu);
    if (auto it = s.live.find(p); it != s.live.end()) {
      s.in_use -= it->second;
      s.live.erase(it);
    }
  }
  return cudaFree(p);
}

}  // namespace aegis::budget
