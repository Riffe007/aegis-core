// The single list of launchable GEMM impls. Tiers 0-1 are listed by hand;
// templated tiers contribute one entry per row of their config table.
// Also: validate(GemmParams), shared by every run().
#include <vector>

#include "aegis/gemm.h"

namespace aegis::gemm {

cudaError_t validate(const GemmParams& p) {
  if (p.M < 0 || p.N < 0 || p.K < 0 || p.batch < 0) return cudaErrorInvalidValue;
  if (is_empty(p)) return cudaSuccess;  // nothing is read or written
  if (!p.A || !p.B || !p.C) return cudaErrorInvalidValue;
  const bool la_row = p.layout_a == Layout::RowMajor, lb_row = p.layout_b == Layout::RowMajor;
  if ((!la_row && p.layout_a != Layout::ColMajor) || (!lb_row && p.layout_b != Layout::ColMajor))
    return cudaErrorInvalidValue;
  // Leading dimension >= the contiguous extent.
  if (p.lda < (la_row ? p.K : p.M) || p.ldb < (lb_row ? p.N : p.K) || p.ldc < p.N)
    return cudaErrorInvalidValue;
  if (p.batch > 1) {
    // A and B may be shared across the batch (stride 0); C batches must not
    // overlap, or the result would depend on scheduling.
    if (p.stride_a < 0 || p.stride_b < 0) return cudaErrorInvalidValue;
    if (p.stride_c < (long long)(p.M - 1) * p.ldc + p.N) return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

std::span<const KernelEntry> registry() {
  static const std::vector<KernelEntry> all = [] {
    std::vector<KernelEntry> v{
        {"naive", CONFIG_naive, &launch_naive, &attrs_naive, &run_naive},
        {"tiled", CONFIG_tiled, &launch_tiled, &attrs_tiled, nullptr},  // legacy: row-major only
    };
    for (auto tier : {regblock_entries(), mma_entries(), mma_pipe_entries()}) v.insert(v.end(), tier.begin(), tier.end());
#ifdef AEGIS_TMA  // every target architecture is sm_90+ (CMakeLists.txt)
    for (const KernelEntry& e : mma_tma_entries()) v.push_back(e);
#endif
    return v;
  }();
  return all;
}

const KernelEntry* find(std::string_view name) {
  for (const KernelEntry& e : registry())
    if (name == e.name) return &e;
  return nullptr;
}

}  // namespace aegis::gemm
