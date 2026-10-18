// ctest tma_paths: the mma_tma family's path choice and correctness.
//   TMA: 16-byte base pointers, ld and batch strides multiples of 8 elements,
//        boundary tiles (M, N, K not multiples of the tile) via TMA's
//        out-of-bounds zero fill.
//   cp.async fallback (tier 4 on the same tile): a misaligned A or B pointer,
//        ld not a multiple of 8, a misaligned batch stride.
// Every case runs every layout pair it lists and is compared with the FP64
// CPU reference under the Production-bar tolerance (src/bench/reference.h).
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "bench/gpu_budget.h"
#include "bench/reference.h"

namespace {

using aegis::gemm::GemmParams;
using aegis::gemm::Layout;

struct Case {
  const char* what;
  int M, N, K;
  Layout la, lb;
  int pad;            // extra ld elements
  int offset_a, offset_b;  // elements added to the base pointers
  int batch;
  long long stride_extra;  // added to each packed batch stride
  float alpha, beta;
  const char* want;   // "tma" or "cp.async (mma_pipe)"
};

bool run_case(const aegis::gemm::KernelEntry& e, const Case& c) {
  GemmParams p;
  p.M = c.M; p.N = c.N; p.K = c.K;
  p.layout_a = c.la; p.layout_b = c.lb;
  p.lda = (c.la == Layout::RowMajor ? c.K : c.M) + c.pad;
  p.ldb = (c.lb == Layout::RowMajor ? c.N : c.K) + c.pad;
  p.ldc = c.N + c.pad;
  p.alpha = c.alpha; p.beta = c.beta;
  p.batch = c.batch;
  if (c.batch > 1) {
    p.stride_a = p.lda * (c.la == Layout::RowMajor ? c.M : c.K) + c.stride_extra;
    p.stride_b = p.ldb * (c.lb == Layout::RowMajor ? c.K : c.N) + c.stride_extra;
    p.stride_c = p.ldc * c.M + c.stride_extra;
  }
  aegis::bench::HostGemm g;
  g.p = p;
  const size_t na = aegis::bench::span_a(p), nb = aegis::bench::span_b(p), nc = aegis::bench::span_c(p);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  g.A.resize(na); g.B.resize(nb); g.C0.resize(nc);
  for (auto& x : g.A) x = __float2half(d(rng));
  for (auto& x : g.B) x = __float2half(d(rng));
  for (auto& x : g.C0) x = __float2half(d(rng));
  __half *A0, *B0, *C;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A0, (na + 8) * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B0, (nb + 8) * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, nc * 2));
  __half* A = A0 + c.offset_a;
  __half* B = B0 + c.offset_b;
  AEGIS_CUDA_CHECK(cudaMemcpy(A, g.A.data(), na * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, g.B.data(), nb * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(C, g.C0.data(), nc * 2, cudaMemcpyHostToDevice));
  p.A = A; p.B = B; p.C = C;
  const std::string path = aegis::gemm::mma_tma_path(p);
  const cudaError_t err = e.run(p, nullptr);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> out(nc);
  AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), C, nc * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref;
  aegis::bench::cpu_reference(g, ref);
  const aegis::bench::Comparison cmp = aegis::bench::compare(g, out, ref, aegis::bench::tolerance(g).tol);
  const bool ok = err == cudaSuccess && cmp.pass && path == c.want;
  std::printf("%-5s %-28s %-48s %4dx%4dx%4d %s%s b%d: path %s (want %s), max |err|/tol %.2f %s\n", ok ? "ok" : "FAIL",
              e.name, c.what, c.M, c.N, c.K, c.la == Layout::RowMajor ? "R" : "C", c.lb == Layout::RowMajor ? "R" : "C",
              c.batch, path.c_str(), c.want, cmp.max_ratio, cmp.pass ? "" : cmp.first_failure.c_str());
  if (err != cudaSuccess) std::printf("      run: %s\n", cudaGetErrorString(err));
  aegis::budget::gpu_free(A0); aegis::budget::gpu_free(B0); aegis::budget::gpu_free(C);
  return ok;
}

}  // namespace

int main() {
  int major = 0, dev = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  AEGIS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  const char* tma = major >= 9 ? "tma" : "cp.async (mma_pipe)";  // below sm_90 everything falls back
  const char* fb = "cp.async (mma_pipe)";
  const Layout R = Layout::RowMajor, Cm = Layout::ColMajor;
  const std::vector<Case> cases = {
      {"aligned, interior tiles", 256, 256, 256, R, R, 0, 0, 0, 1, 0, 1.f, 0.f, tma},
      {"boundary tails, ldb 137 (odd)", 201, 137, 77 + 3, R, R, 0, 0, 0, 1, 0, 1.f, 0.f, fb},
      {"boundary: M, N, K tails, ld +3 (not 8-aligned)", 201, 137, 77, R, R, 3, 0, 0, 1, 0, 1.f, 0.f, fb},
      {"boundary tiles, ld multiple of 8", 200, 136, 72, R, R, 0, 0, 0, 1, 0, 1.f, 0.f, tma},
      {"boundary tiles, K tail, ld not a multiple of 8", 203, 141, 85, R, R, 7, 0, 0, 1, 0, 1.f, 0.f, fb},
      {"M and K tails, N 136, ld multiples of 8", 130, 136, 88, R, R, 0, 0, 0, 1, 0, 1.f, 0.f, tma},
      {"A col, B row, alpha/beta", 192, 200, 136, Cm, R, 0, 0, 0, 1, 0, -0.5f, 0.75f, tma},
      {"A row, B col", 192, 200, 136, R, Cm, 0, 0, 0, 1, 0, 1.f, 0.f, tma},
      {"A col, B col", 192, 200, 136, Cm, Cm, 0, 0, 0, 1, 0, 1.f, 0.f, tma},
      {"batch 3, strides +8, beta 1", 136, 128, 96, R, R, 0, 0, 0, 3, 8, 1.f, 1.f, tma},
      {"misaligned A (+1 element)", 256, 256, 256, R, R, 0, 1, 0, 1, 0, 1.f, 0.f, fb},
      {"misaligned B (+3 elements)", 256, 256, 256, R, R, 0, 0, 3, 1, 0, 1.f, 0.f, fb},
      {"ld not a multiple of 8", 256, 256, 256, R, R, 1, 0, 0, 1, 0, 1.f, 0.f, fb},
      {"batch stride not a multiple of 8", 128, 128, 128, R, R, 0, 0, 0, 2, 4, 1.f, 0.f, fb},
  };
  bool ok = true;
  int n = 0;
  for (const aegis::gemm::KernelEntry& e : aegis::gemm::registry()) {
    if (std::strncmp(e.name, "mma_tma_", 8) != 0) continue;
    ++n;
    for (const Case& c : cases) ok &= run_case(e, c);
  }
  if (n == 0) {
    std::printf("FAIL: no mma_tma impl registered\n");
    return 1;
  }
  return ok ? 0 : 1;
}
