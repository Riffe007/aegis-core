// Capability-aware dispatch (aegis-hw step 4), ctest hw_dispatch.
//   1. Every registry kernel declares its requirements: threads; regs from the binary;
//      sm_80 for the tensor-core tiers; TMA + sm_90 for tier 5; dynamic-smem tiers a
//      smem size.
//   2. Simulated sm_87 (Orin). Its limits are written into the test:
//      - 163 KiB opt-in smem per block, 64 K registers and 1024 threads per block
//        (CUDA C Programming Guide, compute capability 8.7: a labelled test input,
//        not a measurement);
//      - no TMA.
//      Checks:
//      - no TMA kernel and no kernel over the smem limit is supported;
//      - dispatch over this build's winners (every winners-header shape plus
//        nearest-M shapes) never picks one;
//      - every shape still goes somewhere valid: a supported AEGIS kernel or cuBLASLt.
//   3. A fictional device with 48 KiB of opt-in smem (same arch as this one): the smem
//      filter removes kernels (the decode winners need up to 63 KiB), dispatch
//      avoids them, and aegis::matmul under that simulated profile still matches
//      the FP64 reference on the decode shapes.
#include <array>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "aegis/matmul.h"
#include "bench/gpu_budget.h"
#include "bench/reference.h"
#include "hw/device.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

using aegis::detail::Choice;
using aegis::gemm::DeviceCaps;
using aegis::gemm::KernelEntry;

DeviceCaps sm87_caps() {
  DeviceCaps c;
  c.arch = 87;
  c.tma_built = false;              // an sm_87 build never compiles tier 5 (CMakeLists.txt)
  c.smem_per_block_optin = 166912;  // 163 KiB
  c.regs_per_block = 65536;
  c.max_threads_per_block = 1024;
  return c;
}

void declarations() {
  int tma = 0, bad = 0;
  for (const KernelEntry& e : aegis::gemm::registry()) {
    const std::string n = e.name;
    const auto& r = e.req;
    const bool tc = n.rfind("mma_", 0) == 0 || n.rfind("gemv_tc_", 0) == 0;
    const bool is_tma = n.rfind("mma_tma_", 0) == 0;
    tma += is_tma;
    const bool ok = r.threads > 0 && r.regs > 0 && (!tc || r.min_arch >= 80) && (is_tma == r.tma) &&
                    (!is_tma || r.min_arch >= 90) && (!tc || r.smem_bytes > 0);
    if (!ok) {
      ++bad;
      std::printf("      %s: min_arch %d tma %d smem %zu threads %d regs %d\n", e.name, r.min_arch, int(r.tma),
                  r.smem_bytes, r.threads, r.regs);
    }
  }
  expect(bad == 0, "every registry kernel declares its requirements (" + std::to_string(aegis::gemm::registry().size()) +
                       " kernels, " + std::to_string(tma) + " TMA)");
}

// A choice is valid on `caps`: a supported AEGIS kernel, or cuBLASLt.
bool valid(const Choice& c, const DeviceCaps& caps) {
  if (c.kind == Choice::Kind::Cublaslt) return true;
  return c.kind == Choice::Kind::Aegis && c.kernel && aegis::gemm::supported(*c.kernel, caps) && !c.kernel->req.tma &&
         c.kernel->req.smem_bytes <= caps.smem_per_block_optin;
}

// Every winners-header shape, plus a nearest-M shape between each (N, K)'s rows.
std::vector<std::array<int, 3>> shapes() {
  std::vector<std::array<int, 3>> out;
  for (const auto& w : aegis::detail::winners_table()) {
    out.push_back({w.M, w.N, w.K});
    out.push_back({w.M > 2 ? w.M / 2 + 1 : w.M + 1, w.N, w.K});
  }
  return out;
}

void sim_sm87() {
  const DeviceCaps caps = sm87_caps();
  int tma_rejected = 0, over_smem = 0, wrongly_supported = 0;
  for (const KernelEntry& e : aegis::gemm::registry()) {
    const bool sup = aegis::gemm::supported(e, caps);
    tma_rejected += e.req.tma && !sup;
    over_smem += e.req.smem_bytes > caps.smem_per_block_optin;
    wrongly_supported += sup && (e.req.tma || e.req.smem_bytes > caps.smem_per_block_optin);
  }
  expect(wrongly_supported == 0, "sm_87: no TMA kernel and no kernel over 163 KiB smem is supported (" +
                                     std::to_string(tma_rejected) + " TMA kernels rejected, " +
                                     std::to_string(over_smem) + " over the smem limit)");
  const auto table = aegis::detail::winners_table();
  int aegis_n = 0, lt_n = 0, filtered_shapes = 0, invalid = 0;
  for (const auto& s : shapes()) {
    std::string why;
    const Choice c = aegis::detail::choose(table, s[0], s[1], s[2], caps, &why);
    if (!valid(c, caps)) {
      ++invalid;
      std::printf("      %dx%dx%d: invalid choice\n", s[0], s[1], s[2]);
    }
    aegis_n += c.kind == Choice::Kind::Aegis;
    lt_n += c.kind == Choice::Kind::Cublaslt;
    if (c.filtered) {
      ++filtered_shapes;
      std::printf("      %dx%dx%d -> %s (filtered: %s)\n", s[0], s[1], s[2],
                  c.kind == Choice::Kind::Aegis ? c.kernel->name : "cublaslt", why.c_str());
    }
  }
  expect(invalid == 0 && aegis_n + lt_n == int(shapes().size()) && !table.empty(),
         "sm_87 over this build's winners: " + std::to_string(shapes().size()) + " shapes, " + std::to_string(aegis_n) +
             " AEGIS kernels, " + std::to_string(lt_n) + " cuBLASLt, " + std::to_string(filtered_shapes) +
             " with rows filtered; none TMA or over the smem limit");
}

aegis::bench::Comparison run_once(int M, int N, int K) {
  aegis::bench::HostGemm g;
  g.p = aegis::gemm::row_major_params(nullptr, nullptr, nullptr, M, N, K);
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  g.A.resize(size_t(M) * K); g.B.resize(size_t(K) * N); g.C0.assign(size_t(M) * N, __float2half(0.f));
  for (auto& x : g.A) x = __float2half(d(rng));
  for (auto& x : g.B) x = __float2half(d(rng));
  __half *A, *B, *C;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, g.A.size() * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, g.B.size() * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, g.C0.size() * 2));
  AEGIS_CUDA_CHECK(cudaMemcpy(A, g.A.data(), g.A.size() * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, g.B.data(), g.B.size() * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(aegis::matmul(A, B, C, M, N, K));
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> out(g.C0.size());
  AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), C, out.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref;
  aegis::bench::cpu_reference(g, ref);
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(B); aegis::budget::gpu_free(C);
  return aegis::bench::compare(g, out, ref, aegis::bench::tolerance(g).tol);
}

void tight_smem(const aegis::hw::DeviceLimits& real) {
  aegis::hw::DeviceLimits tight = real;
  tight.name = "tight-smem (simulated)";
  tight.smem_per_block_optin = 48 * 1024;
  aegis::hw::ScopedSimulation sim(tight);
  const DeviceCaps caps = aegis::gemm::current_caps();
  expect(caps.smem_per_block_optin == 48 * 1024 && caps.arch == real.arch, "simulated caps reach current_caps()");
  int removed = 0;
  for (const KernelEntry& e : aegis::gemm::registry()) removed += !aegis::gemm::supported(e, caps);
  expect(removed > 0, "48 KiB opt-in smem: " + std::to_string(removed) + " kernels unsupported");
  constexpr int kDecode[][3] = {{1, 896, 896}, {1, 4864, 896}, {1, 896, 4864},
                                {16, 896, 896}, {16, 4864, 896}, {16, 896, 4864}};
  int filtered = 0;
  for (const auto& s : kDecode) {
    const Choice c = aegis::detail::choose(aegis::detail::winners_table(), s[0], s[1], s[2], caps);
    filtered += c.filtered > 0;
    const std::string chosen = aegis::matmul_choice(s[0], s[1], s[2]);
    const aegis::bench::Comparison cmp = run_once(s[0], s[1], s[2]);
    char buf[200];
    std::snprintf(buf, sizeof buf, "48 KiB device %dx%dx%d -> %s (%d rows filtered), max |err|/tol %.2f", s[0], s[1],
                  s[2], chosen.c_str(), c.filtered, cmp.max_ratio);
    expect(valid(c, caps) && cmp.pass, buf);
  }
  expect(filtered > 0, "the smem filter changed the choice on " + std::to_string(filtered) + " decode shapes");
}

}  // namespace

int main() {
  const aegis::hw::DeviceLimits* real = aegis::hw::physical();
  if (!real) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  declarations();
  sim_sm87();
  tight_smem(*real);
  // The real device, after the simulations: its winners dispatch unfiltered.
  int filtered = 0;
  for (const auto& w : aegis::detail::winners_table())
    filtered += aegis::detail::choose(aegis::detail::winners_table(), w.M, w.N, w.K, aegis::gemm::current_caps()).filtered;
  expect(filtered == 0, "real device: no winner row filtered");
  std::printf("%s: %d failures\n", failures ? "hw_dispatch FAILED" : "hw_dispatch ok", failures);
  return failures ? 1 : 0;
}
