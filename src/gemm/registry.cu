// The single list of launchable GEMM impls. Tiers 0-1 are listed by hand;
// templated tiers contribute one entry per row of their config table.
// Also: validate(GemmParams), shared by every run().
#include <algorithm>
#include <string>
#include <vector>

#include "aegis/gemm.h"
#include "hw/device.h"

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
        // CUDA cores; static smem from the binary (below). Blocks: naive 16x16, tiled 32x32.
        {"naive", CONFIG_naive, &launch_naive, &attrs_naive, &run_naive, 0, KernelReq{0, false, 0, 256}},
        {"tiled", CONFIG_tiled, &launch_tiled, &attrs_tiled, nullptr, 0, KernelReq{0, false, 0, 1024}},  // legacy: row-major only
    };
    for (auto tier : {regblock_entries(), mma_entries(), mma_pipe_entries()}) v.insert(v.end(), tier.begin(), tier.end());
#ifdef AEGIS_TMA  // every target architecture is sm_90+ (CMakeLists.txt)
    for (const KernelEntry& e : mma_tma_entries()) v.push_back(e);
#endif
    for (auto fam : {gemv_entries(), gemv_tc_entries()}) v.insert(v.end(), fam.begin(), fam.end());
    // Requirements from the compiled binary: registers per thread, and static smem
    // (attrs() also adds the dynamic smem of the row-major NN launch; the declared
    // smem_bytes covers every launch, so the larger of the two stands).
    for (KernelEntry& e : v) {
      cudaFuncAttributes a{};
      if (e.attrs && e.attrs(&a) == cudaSuccess) {
        e.req.regs = a.numRegs;
        e.req.smem_bytes = std::max(e.req.smem_bytes, size_t(a.sharedSizeBytes));
      } else {
        cudaGetLastError();
      }
    }
    return v;
  }();
  return all;
}

namespace detail {
DecodeLaunch gemv_splits(size_t i, const GemmParams& p);     // gemv.cu
DecodeLaunch gemv_tc_splits(size_t i, const GemmParams& p);  // gemv_tc.cu
}  // namespace detail

bool gemv_launch_plan(const char* impl, const GemmParams& p, DecodeLaunch* out) {
  const std::string_view name = impl;
  const std::span<const KernelEntry> cc = gemv_entries(), tc = gemv_tc_entries();
  for (size_t i = 0; i < cc.size(); ++i)
    if (name == cc[i].name) return *out = detail::gemv_splits(i, p), true;
  for (size_t i = 0; i < tc.size(); ++i)
    if (name == tc[i].name) return *out = detail::gemv_tc_splits(i, p), true;
  return false;
}

int gemv_split_count(const char* impl, const GemmParams& p) {
  DecodeLaunch l;
  return gemv_launch_plan(impl, p, &l) ? l.splits : 0;
}

DeviceCaps current_caps() {
  DeviceCaps c;
  const hw::DeviceLimits* d = hw::current();
  if (!d) return c;
  c.arch = d->arch;
#ifdef AEGIS_TMA
  c.tma_built = true;
#endif
  c.smem_per_block_optin = d->smem_per_block_optin;
  c.regs_per_block = d->regs_per_block;
  c.max_threads_per_block = d->max_threads_per_block;
  return c;
}

bool supported(const KernelEntry& e, const DeviceCaps& c, std::string* why) {
  auto no = [&](std::string w) {
    if (why) *why = std::move(w);
    return false;
  };
  const KernelReq& r = e.req;
  if (c.arch < r.min_arch) return no("needs sm_" + std::to_string(r.min_arch) + ", device is sm_" + std::to_string(c.arch));
  if (r.tma && !(c.tma_built && c.arch >= 90)) return no("needs TMA (sm_90+ and a TMA build)");
  if (r.smem_bytes > c.smem_per_block_optin)
    return no("needs " + std::to_string(r.smem_bytes) + " B smem per block, device allows " +
              std::to_string(c.smem_per_block_optin));
  if (r.threads > c.max_threads_per_block)
    return no("needs " + std::to_string(r.threads) + " threads per block, device allows " +
              std::to_string(c.max_threads_per_block));
  if (r.regs > 0 && (long long)r.regs * r.threads > c.regs_per_block)
    return no("needs " + std::to_string(r.regs * r.threads) + " registers per block, device has " +
              std::to_string(c.regs_per_block));
  return true;
}

const KernelEntry* find(std::string_view name) {
  for (const KernelEntry& e : registry())
    if (name == e.name) return &e;
  return nullptr;
}

}  // namespace aegis::gemm
