// The in-process GPU allocation budget (src/bench/gpu_budget.h) refuses
// oversize requests before anything is allocated.
//
//   test_gpu_budget default    no env: a 17 GiB request exceeds the default 16 GiB budget
//   test_gpu_budget budget     AEGIS_GPU_BUDGET=64M: 32 MiB ok, +48 MiB refused,
//                              after a free 48 MiB ok; a 1 TiB request refused
//   test_gpu_budget minfree    AEGIS_GPU_MIN_FREE=1000000G: even 1 MiB refused
// ctest sets the environment per mode. Exit 0 iff every expectation holds.
#include <cstdio>
#include <cstring>

#include "bench/gpu_budget.h"

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
  namespace b = aegis::budget;
  const char* mode = argc > 1 ? argv[1] : "default";
  constexpr size_t MiB = size_t(1) << 20;
  void* p = nullptr;
  void* q = nullptr;
  if (std::strcmp(mode, "default") == 0) {
    expect(b::gpu_malloc(&p, 17 * b::kGiB, "17 GiB test") == cudaErrorMemoryAllocation && !p,
           "17 GiB refused by the default 16 GiB budget");
    expect(b::in_use() == 0, "nothing tracked after the refusal");
  } else if (std::strcmp(mode, "budget") == 0) {
    expect(b::gpu_malloc(&p, 32 * MiB, "32 MiB") == cudaSuccess && p, "32 MiB within a 64 MiB budget");
    expect(b::gpu_malloc(&q, 48 * MiB, "48 MiB") == cudaErrorMemoryAllocation && !q, "+48 MiB refused (80 > 64)");
    expect(b::in_use() == 32 * MiB, "32 MiB tracked");
    expect(b::gpu_free(p) == cudaSuccess && b::in_use() == 0, "free untracks");
    expect(b::gpu_malloc(&q, 48 * MiB, "48 MiB") == cudaSuccess && q, "48 MiB ok after the free");
    expect(b::gpu_free(q) == cudaSuccess, "free 48 MiB");
    void* huge = nullptr;
    expect(b::gpu_malloc(&huge, 1024 * b::kGiB, "1 TiB") == cudaErrorMemoryAllocation && !huge,
           "1 TiB refused");
  } else if (std::strcmp(mode, "minfree") == 0) {
    expect(b::gpu_malloc(&p, MiB, "1 MiB") == cudaErrorMemoryAllocation && !p,
           "1 MiB refused when the system-wide free floor cannot be met");
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  std::printf("test_gpu_budget %s: %s\n", mode, failures ? "FAILED" : "all checks passed");
  return failures ? 1 : 0;
}
