// ctest splitk_determinism: the deterministic split-K of the decode GEMV
// family (gemv, gemv_tc; src/gemm/splitk.cuh).
//   1. Every gemv / gemv_tc impl on decode and odd shapes (all layouts,
//      alpha/beta, K tails), 10 runs: bitwise identical output each time, and
//      within the Production-bar tolerance of the FP64 reference.
//   2. The effective split count is what gemv_split_count says, > 1 where the
//      config asks for it, and odd counts occur (7, 3, 5).
//   3. Strict-mode graph capture (cudaStreamCaptureModeGlobal): after one
//      direct call per (impl, shape), 50 calls capture with no workspace
//      allocation, and the replay's output equals the direct call's bit for
//      bit. aegis::matmul on the six decode shapes is captured the same way.
//   4. A first call inside capture (no workspace yet for that stream) returns
//      cudaErrorStreamCaptureUnsupported and leaves the capture valid.
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
#include "gemm/workspace.h"

namespace {

using aegis::gemm::GemmParams;
using aegis::gemm::KernelEntry;
using aegis::gemm::Layout;

struct Shape {
  int M, N, K;
  Layout la, lb;
  int pad;
  float alpha, beta;
};

struct Problem {
  aegis::bench::HostGemm g;
  __half *A = nullptr, *B = nullptr, *C = nullptr;
  GemmParams p;
  explicit Problem(const Shape& s) {
    GemmParams& q = g.p;
    q.M = s.M; q.N = s.N; q.K = s.K;
    q.layout_a = s.la; q.layout_b = s.lb;
    q.lda = (s.la == Layout::RowMajor ? s.K : s.M) + s.pad;
    q.ldb = (s.lb == Layout::RowMajor ? s.N : s.K) + s.pad;
    q.ldc = s.N + s.pad;
    q.alpha = s.alpha; q.beta = s.beta;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    g.A.resize(aegis::bench::span_a(q)); g.B.resize(aegis::bench::span_b(q)); g.C0.resize(aegis::bench::span_c(q));
    for (auto& x : g.A) x = __float2half(d(rng));
    for (auto& x : g.B) x = __float2half(d(rng));
    for (auto& x : g.C0) x = __float2half(d(rng));
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, g.A.size() * 2));
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, g.B.size() * 2));
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, g.C0.size() * 2));
    AEGIS_CUDA_CHECK(cudaMemcpy(A, g.A.data(), g.A.size() * 2, cudaMemcpyHostToDevice));
    AEGIS_CUDA_CHECK(cudaMemcpy(B, g.B.data(), g.B.size() * 2, cudaMemcpyHostToDevice));
    p = q;
    p.A = A; p.B = B; p.C = C;
  }
  ~Problem() {
    aegis::budget::gpu_free(A);
    aegis::budget::gpu_free(B);
    aegis::budget::gpu_free(C);
  }
  void reset_c(cudaStream_t s) {
    AEGIS_CUDA_CHECK(cudaMemcpyAsync(C, g.C0.data(), g.C0.size() * 2, cudaMemcpyHostToDevice, s));
  }
  std::vector<__half> read_c(cudaStream_t s) {
    std::vector<__half> out(g.C0.size());
    AEGIS_CUDA_CHECK(cudaStreamSynchronize(s));
    AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), C, out.size() * 2, cudaMemcpyDeviceToHost));
    return out;
  }
};

bool same_bits(const std::vector<__half>& a, const std::vector<__half>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0;
}

bool is_decode_impl(const char* name) { return std::strncmp(name, "gemv_", 5) == 0; }

const Layout R = Layout::RowMajor, Cm = Layout::ColMajor;

}  // namespace

int main() {
  cudaStream_t stream;
  AEGIS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  bool ok = true;
  int impls = 0, odd_splits = 0, split_cases = 0;

  // 1 + 2: determinism, correctness, split counts.
  const std::vector<Shape> shapes = {
      {1, 896, 4864, R, R, 0, 1.f, 0.f},     {16, 896, 4864, R, R, 0, 1.f, 0.f},
      {16, 896, 896, R, R, 0, 1.f, 0.f},     {1, 4864, 896, R, R, 0, 1.f, 0.f},
      {13, 1000, 4099, Cm, R, 3, -0.5f, 0.75f}, {5, 333, 2053, R, Cm, 1, 2.f, 1.f},
      {9, 136, 7777, Cm, Cm, 0, 1.f, -1.f},   {37, 260, 1031, R, R, 0, 1.f, 0.5f},
  };
  for (const Shape& s : shapes) {
    Problem pr(s);
    std::vector<double> ref;
    aegis::bench::cpu_reference(pr.g, ref);
    const double tol = aegis::bench::tolerance(pr.g).tol;
    for (const KernelEntry& e : aegis::gemm::registry()) {
      if (!is_decode_impl(e.name)) continue;
      if (&s == &shapes[0]) ++impls;
      const int S = aegis::gemm::gemv_split_count(e.name, pr.p);
      if (S > 1) ++split_cases;
      if (S > 1 && S % 2 == 1) ++odd_splits;
      std::vector<__half> first;
      bool det = true, corr = true;
      for (int run = 0; run < 10; ++run) {
        pr.reset_c(stream);
        const cudaError_t err = e.run(pr.p, stream);
        if (err != cudaSuccess) {
          std::printf("FAIL %s %dx%dx%d: run %s\n", e.name, s.M, s.N, s.K, cudaGetErrorString(err));
          ok = false;
          det = false;
          break;
        }
        std::vector<__half> out = pr.read_c(stream);
        if (run == 0) {
          first = out;
          const aegis::bench::Comparison c = aegis::bench::compare(pr.g, out, ref, tol);
          corr = c.pass;
          if (!c.pass) std::printf("FAIL %s %dx%dx%d S=%d: %s\n", e.name, s.M, s.N, s.K, S, c.first_failure.c_str());
        } else if (!same_bits(out, first)) {
          det = false;
          std::printf("FAIL %s %dx%dx%d S=%d: run %d differs from run 0\n", e.name, s.M, s.N, s.K, S, run);
          break;
        }
      }
      ok &= det && corr;
    }
  }
  std::printf("%d impls x %zu shapes x 10 runs: %d split cases (S > 1), %d with odd S\n", impls, shapes.size(),
              split_cases, odd_splits);
  if (impls == 0 || split_cases == 0 || odd_splits == 0) {
    std::printf("FAIL: no split-K coverage\n");
    ok = false;
  }

  // 3: strict-mode capture after one direct call; no allocation inside.
  {
    const std::vector<Shape> cap_shapes = {{1, 896, 4864, R, R, 0, 1.f, 0.f}, {16, 896, 896, R, R, 0, 1.f, 0.f},
                                           {16, 4864, 896, R, R, 0, 1.f, 0.f}, {16, 896, 4864, R, R, 0, 1.f, 0.f},
                                           {1, 896, 896, R, R, 0, 1.f, 0.f}, {1, 4864, 896, R, R, 0, 1.f, 0.f}};
    for (const Shape& s : cap_shapes) {
      Problem pr(s);
      std::vector<const KernelEntry*> targets;
      for (const KernelEntry& e : aegis::gemm::registry())
        if (is_decode_impl(e.name) && aegis::gemm::gemv_split_count(e.name, pr.p) > 1) targets.push_back(&e);
      targets.push_back(nullptr);  // aegis::matmul
      for (const KernelEntry* e : targets) {
        auto call = [&] { return e ? e->run(pr.p, stream) : aegis::matmul(pr.p, stream); };
        const char* name = e ? e->name : "aegis::matmul";
        AEGIS_CUDA_CHECK(call());  // direct: may allocate the stream's workspace
        const std::vector<__half> direct = pr.read_c(stream);
        const size_t allocs = aegis::gemm::detail::stream_workspace_allocations();
        cudaGraph_t graph = nullptr;
        AEGIS_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        cudaError_t err = cudaSuccess;
        for (int i = 0; i < 50 && err == cudaSuccess; ++i) err = call();
        const cudaError_t end = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess || end != cudaSuccess) {
          std::printf("FAIL capture %s %dx%dx%d: call %s, end %s (%s)\n", name, s.M, s.N, s.K,
                      cudaGetErrorString(err), cudaGetErrorString(end), e ? "" : aegis::matmul_last_error());
          ok = false;
          if (graph) cudaGraphDestroy(graph);
          cudaGetLastError();
          continue;
        }
        if (aegis::gemm::detail::stream_workspace_allocations() != allocs) {
          std::printf("FAIL capture %s: allocated inside capture\n", name);
          ok = false;
        }
        cudaGraphExec_t exec = nullptr;
        AEGIS_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        AEGIS_CUDA_CHECK(cudaGraphLaunch(exec, stream));
        const std::vector<__half> replay = pr.read_c(stream);
        if (!same_bits(replay, direct)) {
          std::printf("FAIL capture %s %dx%dx%d: replay differs from the direct call\n", name, s.M, s.N, s.K);
          ok = false;
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
      }
      std::printf("captured %zu targets on %dx%dx%d (50 calls each, global mode)\n", targets.size(), s.M, s.N, s.K);
    }
  }

  // 4: the first call for a stream inside capture.
  {
    cudaStream_t fresh;
    AEGIS_CUDA_CHECK(cudaStreamCreateWithFlags(&fresh, cudaStreamNonBlocking));
    Problem pr({1, 896, 4864, R, R, 0, 1.f, 0.f});
    const KernelEntry* e = aegis::gemm::find("gemv_64x128_v8_k8");
    if (!e || aegis::gemm::gemv_split_count(e->name, pr.p) < 2) {
      std::printf("FAIL: gemv_64x128_v8_k8 missing or not split\n");
      ok = false;
    } else {
      cudaGraph_t graph = nullptr;
      AEGIS_CUDA_CHECK(cudaStreamBeginCapture(fresh, cudaStreamCaptureModeGlobal));
      const cudaError_t err = e->run(pr.p, fresh);
      const cudaError_t end = cudaStreamEndCapture(fresh, &graph);
      if (err != cudaErrorStreamCaptureUnsupported || end != cudaSuccess) {
        std::printf("FAIL first call in capture: run %s (want %s), end capture %s\n", cudaGetErrorString(err),
                    cudaGetErrorString(cudaErrorStreamCaptureUnsupported), cudaGetErrorString(end));
        ok = false;
      } else {
        std::printf("ok   first call in capture: %s, capture still valid\n", cudaGetErrorString(err));
      }
      if (graph) cudaGraphDestroy(graph);
      cudaGetLastError();
    }
    cudaStreamDestroy(fresh);
  }
  cudaStreamDestroy(stream);
  std::printf("%s\n", ok ? "splitk: all passed" : "splitk: FAILED");
  return ok ? 0 : 1;
}
