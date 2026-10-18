// Hardware-driven launch sizing (aegis-hw step 1). The decode family's split
// count S and grid come from the detected SM count, not from per-config
// constants tuned on one device.
//   1. splitk::sm_load_splits on hand-worked cases.
//   2. For every decode shape (shapes/decode.txt), the impl aegis::matmul
//      dispatches is planned under simulated profiles of 24, 48 and 96 SMs (the
//      real device otherwise):
//      - its S is the reference argmin of the load model;
//      - the tile grid does not change with the SM count;
//      - grid_z = batch x S;
//      - S changes with the SM count for at least one shape.
//   3. aegis::matmul runs under each simulated profile (sizing only; the kernels
//      run on the real device) and matches the FP64 reference.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "aegis/matmul.h"
#include "bench/gpu_budget.h"
#include "bench/reference.h"
#include "gemm/splitk.cuh"
#include "hw/device.h"

namespace {

bool expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  return ok;
}

// Reference for the model, written independently of splitk.cuh: brute force over
// every S, then splitk::plan's workspace cap and rounding to whole 32-row granules
// (partial tiles of tile_elems floats).
int reference_splits(int sms, long long tiles, int K, int min_rows, long long tile_elems) {
  const long long s_max = std::max(1ll, std::min(4096ll, (long long)K / min_rows));
  long long best = 1;
  double best_load = INFINITY;
  for (long long S = 1; S <= s_max; ++S) {
    const double load = std::ceil(double(tiles * S) / sms) * std::ceil(double(K) / S);
    if (load < best_load) best_load = load, best = S;
  }
  const long long cap = aegis::gemm::kSplitKMaxWorkspace / (tiles * tile_elems * 4);
  if (best > cap) best = std::max(1ll, cap);
  if (best == 1) return 1;
  const long long kc = (K + best - 1) / best;
  const long long kc32 = (kc + 31) / 32 * 32;
  return int((K + kc32 - 1) / kc32);
}

// min rows per split from the impl name: gemv_tc_16x<BN>x<BK>_1x<W>_s<ST>_ksm -> 2 * ST * BK; gemv_* -> 64.
int min_rows_of(const std::string& impl) {
  if (impl.rfind("gemv_tc_", 0) != 0) return 64;
  int bn = 0, bk = 0, w = 0, st = 0;
  if (std::sscanf(impl.c_str(), "gemv_tc_16x%dx%d_1x%d_s%d", &bn, &bk, &w, &st) != 4) return -1;
  return 2 * st * bk;
}

// Partial-tile elements: gemv_tc_16x<BN>...: 16 x BN; gemv_<BN>x...: MT x BN, MT the smallest of 1, 2, 4,
// 8, 16 >= M.
long long tile_elems_of(const std::string& impl, int M) {
  int bn = 0;
  if (impl.rfind("gemv_tc_", 0) == 0) return std::sscanf(impl.c_str(), "gemv_tc_16x%d", &bn) == 1 ? 16ll * bn : -1;
  const int mt = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : 16;
  return std::sscanf(impl.c_str(), "gemv_%d", &bn) == 1 ? (long long)mt * bn : -1;
}

bool model_cases() {
  using aegis::gemm::splitk::sm_load_splits;
  bool ok = true;
  // 14 tiles (896 / 64), K = 4864, min rows 640 (64x64 s5):
  // - 24 SMs -> 5: 70 CTAs, at most 3 per SM x 973 rows;
  // - 48 -> 3: 42 CTAs, one per SM;
  // - 96 -> 6: 84 CTAs, one per SM x 811 rows; S 7 would put 98 CTAs on 96 SMs, 2 x 695.
  ok &= expect(sm_load_splits(24, 14, 4864, 640) == 5, "model: 14 tiles, K 4864, 24 SMs -> S 5");
  ok &= expect(sm_load_splits(48, 14, 4864, 640) == 3, "model: 14 tiles, K 4864, 48 SMs -> S 3");
  ok &= expect(sm_load_splits(96, 14, 4864, 640) == 6, "model: 14 tiles, K 4864, 96 SMs -> S 6");
  // K too short for two ring passes: no split on any device.
  ok &= expect(sm_load_splits(96, 14, 896, 512) == 1, "model: K 896, min rows 512 -> S 1");
  // More tiles than SMs: S 1 leaves a partial second wave (2 x 896 rows on the busiest SM);
  // S 5 spreads 380 CTAs as at most 8 x 180 rows.
  ok &= expect(sm_load_splits(48, 76, 896, 128) == 5, "model: 76 tiles on 48 SMs, K 896, min 128 -> S 5");
  ok &= expect(sm_load_splits(48, 76, 896, 512) == 1, "model: 76 tiles on 48 SMs, K 896, min 512 -> S 1");
  // No device: S 1.
  ok &= expect(sm_load_splits(0, 14, 4864, 640) == 1, "model: 0 SMs -> S 1");
  for (int sms : {8, 16, 24, 48, 96, 132})
    for (long long tiles : {1ll, 7ll, 14ll, 38ll, 76ll, 300ll})
      for (int K : {65, 896, 4864, 16384}) {
        const int got = sm_load_splits(sms, tiles, K, 128);
        const auto pl = aegis::gemm::splitk::plan(K, got, tiles, 16, 64);
        if (pl.S != reference_splits(sms, tiles, K, 128, 16 * 64)) {
          ok &= expect(false, "model vs reference: sms " + std::to_string(sms) + " tiles " +
                                  std::to_string(tiles) + " K " + std::to_string(K));
        }
      }
  return expect(ok, "model: sm_load_splits matches the brute-force reference on 144 cases") && ok;
}

aegis::hw::DeviceLimits with_sms(const aegis::hw::DeviceLimits& real, int sms) {
  aegis::hw::DeviceLimits d = real;
  d.sm_count = sms;
  return d;
}

// matmul under the current (possibly simulated) profile vs the FP64 reference.
bool run_matches_reference(int M, int N, int K, const char* label) {
  aegis::bench::HostGemm g;
  g.p = aegis::gemm::row_major_params(nullptr, nullptr, nullptr, M, N, K);
  std::mt19937 rng(11);
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
  const cudaError_t e = aegis::matmul(A, B, C, M, N, K);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> out(g.C0.size());
  AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), C, out.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref;
  aegis::bench::cpu_reference(g, ref);
  const aegis::bench::Comparison c = aegis::bench::compare(g, out, ref, aegis::bench::tolerance(g).tol);
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(B); aegis::budget::gpu_free(C);
  char buf[160];
  std::snprintf(buf, sizeof buf, "%dx%dx%d %s: matmul %s, max |err|/tol %.2f", M, N, K, label,
                cudaGetErrorString(e), c.max_ratio);
  return expect(e == cudaSuccess && c.pass, buf);
}

}  // namespace

int main() {
  bool ok = model_cases();
  const aegis::hw::DeviceLimits* real = aegis::hw::physical();
  if (!real) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  std::printf("device %d: sm_%d, %d SMs\n", real->device, real->arch, real->sm_count);
  constexpr int kShapes[][3] = {{1, 896, 896},  {1, 4864, 896},  {1, 896, 4864},
                                {16, 896, 896}, {16, 4864, 896}, {16, 896, 4864}};
  constexpr int kSms[] = {24, 48, 96};
  int shapes_that_change = 0;
  for (const auto& s : kShapes) {
    const int M = s[0], N = s[1], K = s[2];
    const std::string impl = aegis::matmul_choice(M, N, K);
    if (impl.rfind("gemv_", 0) != 0) {
      ok &= expect(false, std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) +
                              ": dispatch " + impl + " is not a decode kernel");
      continue;
    }
    const aegis::gemm::GemmParams p = aegis::gemm::row_major_params(nullptr, nullptr, nullptr, M, N, K);
    const int min_rows = min_rows_of(impl);
    std::vector<int> splits;
    unsigned gx0 = 0, gy0 = 0;
    for (int sms : kSms) {
      aegis::hw::ScopedSimulation sim(with_sms(*real, sms));
      const std::string again = aegis::matmul_choice(M, N, K);
      aegis::gemm::DecodeLaunch l;
      const bool planned = aegis::gemm::gemv_launch_plan(impl.c_str(), p, &l);
      const long long tiles = (long long)l.grid_x * l.grid_y;
      const int want = reference_splits(sms, tiles, K, min_rows, tile_elems_of(impl, M));
      if (sms == kSms[0]) gx0 = l.grid_x, gy0 = l.grid_y;
      char buf[256];
      std::snprintf(buf, sizeof buf, "%dx%dx%d on %d SMs: %s, S %d (reference %d), grid %ux%ux%u", M, N, K, sms,
                    impl.c_str(), l.splits, want, l.grid_x, l.grid_y, l.grid_z);
      ok &= expect(planned && again == impl && l.splits == want && l.grid_z == unsigned(l.splits) &&
                       l.grid_x == gx0 && l.grid_y == gy0,
                   buf);
      splits.push_back(l.splits);
      ok &= run_matches_reference(M, N, K, ("simulated " + std::to_string(sms) + " SMs").c_str());
    }
    shapes_that_change += !(splits[0] == splits[1] && splits[1] == splits[2]);
  }
  ok &= expect(shapes_that_change > 0,
               "S follows the SM count on " + std::to_string(shapes_that_change) + " of 6 decode shapes");
  // The real device again (no simulation left behind).
  ok &= expect(!aegis::hw::current()->simulated && aegis::hw::current()->sm_count == real->sm_count,
               "simulation ends with its scope");
  return ok ? 0 : 1;
}
