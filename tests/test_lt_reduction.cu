// cuBLASLt fallback numerics (ctest lt_reduction). aegis::matmul's cuBLASLt path
// (dispatch floor, rows filtered by capability) may run only split-K reductions
// that keep partials in FP32: splitk 1, or CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
// (src/lt_algo.h reduction_ok). INPLACE and OUTPUT_TYPE add FP16 partials.
//   1. Every stored algorithm in this build's winners table complies; the rows
//      that dispatch to cuBLASLt are listed.
//   2. The algorithm each routed shape's plan runs complies: every Cublaslt row
//      (stored algorithm), a nearest-M shape of each (heuristic), and long-K
//      shapes (heuristic). For each, how many of the unmasked heuristic's top 8
//      keep FP16 partials (what the mask removes).
//   3. A stored algorithm with FP16 partials is rejected: the plan says so and
//      runs a compliant one. The violating config comes from the unmasked
//      heuristic when it offers one, else it is constructed.
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <cublasLt.h>

#include "aegis/common.cuh"
#include "aegis/matmul.h"
#include "lt_algo.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

using aegis::detail::LtAlgoConfig;

std::string show(const LtAlgoConfig& c) {
  char b[96];
  std::snprintf(b, sizeof b, "algo %d splitk %d reduction %d", c.id, c.splitk, c.reduction);
  return b;
}

// The unmasked heuristic's top 8 for row-major NN M x N x K (aegis::matmul's
// operand mapping), as configs.
std::vector<LtAlgoConfig> unmasked_top(cublasLtHandle_t lt, int M, int N, int K) {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, N, K, N);
  cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, K, M, K);
  cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, N, M, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  const size_t ws = size_t(32) << 20;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
  cublasLtMatmulHeuristicResult_t res[8] = {};
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 8, res, &n);
  std::vector<LtAlgoConfig> out;
  for (int i = 0; i < n; ++i)
    if (res[i].state == CUBLAS_STATUS_SUCCESS) out.push_back(aegis::lt::read_config(res[i].algo));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(op);
  return out;
}

// A plan for row-major NN M x N x K (pointers aligned to 256 B: only their
// alignment enters the plan key; nothing is launched).
std::string plan(int M, int N, int K, const LtAlgoConfig* stored, LtAlgoConfig* used) {
  const auto p = aegis::gemm::row_major_params(reinterpret_cast<const __half*>(256), reinterpret_cast<const __half*>(512),
                                               reinterpret_cast<__half*>(768), M, N, K);
  return aegis::detail::cublaslt_plan_source(p, nullptr, stored, used);
}

}  // namespace

int main() {
  cublasLtHandle_t lt;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasLtCreate\n");
    return 1;
  }
  const auto table = aegis::detail::winners_table();
  expect(!table.empty(), "winners table for this device (" + std::to_string(table.size()) + " rows)");

  // 1. Stored algorithms.
  int bad_stored = 0, routed = 0;
  for (const auto& w : table) {
    if (w.lt.id < 0) continue;
    const bool ok = aegis::detail::lt_reduction_ok(w.lt);
    bad_stored += !ok;
    if (w.dispatch == aegis::detail::Dispatch::Cublaslt || !ok) {
      routed += w.dispatch == aegis::detail::Dispatch::Cublaslt;
      std::printf("      %dx%dx%d %s: stored %s\n", w.M, w.N, w.K,
                  w.dispatch == aegis::detail::Dispatch::Cublaslt ? "dispatch cublaslt" : "(hint)", show(w.lt).c_str());
    }
  }
  expect(bad_stored == 0, "every stored cuBLASLt algorithm keeps FP32 partials (" + std::to_string(routed) +
                              " rows dispatch to cuBLASLt)");

  // 2. What each routed shape's plan runs.
  struct S {
    int M, N, K;
    const LtAlgoConfig* stored;
    const char* why;
  };
  std::vector<S> shapes;
  for (const auto& w : table)
    if (w.dispatch == aegis::detail::Dispatch::Cublaslt) {
      shapes.push_back({w.M, w.N, w.K, &w.lt, "cublaslt row, stored"});
      shapes.push_back({w.M / 2 + 1, w.N, w.K, nullptr, "nearest-M of a cublaslt row, heuristic"});
    }
  for (const auto& s : std::vector<std::array<int, 3>>{{1, 896, 14000}, {16, 4096, 16384}, {24, 1872, 14747},
                                                       {64, 2048, 14009}, {7, 512, 20000}})
    shapes.push_back({s[0], s[1], s[2], nullptr, "long K, heuristic"});
  int noncompliant = 0;
  for (const S& s : shapes) {
    LtAlgoConfig used;
    const std::string src = plan(s.M, s.N, s.K, s.stored, &used);
    const auto top = unmasked_top(lt, s.M, s.N, s.K);
    int fp16 = 0;
    for (const auto& c : top) fp16 += !aegis::detail::lt_reduction_ok(c);
    const bool ok = !src.empty() && aegis::detail::lt_reduction_ok(used);
    noncompliant += !ok;
    char buf[300];
    std::snprintf(buf, sizeof buf, "%dx%dx%d (%s): plan %s, %s; unmasked top %zu: %d with FP16 partials (rank 1: %s)", s.M,
                  s.N, s.K, s.why, src.empty() ? aegis::matmul_last_error() : src.c_str(), show(used).c_str(), top.size(),
                  fp16, top.empty() ? "none" : show(top[0]).c_str());
    expect(ok, buf);
  }
  expect(noncompliant == 0, "every routed plan keeps FP32 partials");

  // 3. A stored algorithm with FP16 partials is rejected.
  LtAlgoConfig bad;
  int bM = 16, bN = 4096, bK = 16384;
  bool from_heuristic = false;
  for (const auto& s : shapes) {
    for (const auto& c : unmasked_top(lt, s.M, s.N, s.K))
      if (!aegis::detail::lt_reduction_ok(c)) {
        bad = c, bM = s.M, bN = s.N, bK = s.K, from_heuristic = true;
        break;
      }
    if (from_heuristic) break;
  }
  if (!from_heuristic) {  // constructed: the 4096^3 row's algorithm, split 4 ways, FP16 partials
    bad = table.empty() ? LtAlgoConfig{} : table.back().lt;
    bad.splitk = 4;
    bad.reduction = int(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE);
  }
  LtAlgoConfig used;
  const std::string src = plan(bM, bN, bK, &bad, &used);
  expect(src.rfind("heuristic (stored", 0) == 0 && aegis::detail::lt_reduction_ok(used),
         std::string("stored ") + show(bad) + (from_heuristic ? " (offered by the heuristic)" : " (constructed)") + " at " +
             std::to_string(bM) + "x" + std::to_string(bN) + "x" + std::to_string(bK) + " -> " + src + ", runs " +
             show(used));
  cublasLtDestroy(lt);
  std::printf("%s: %d failures\n", failures ? "lt_reduction FAILED" : "lt_reduction ok", failures);
  return failures ? 1 : 0;
}
