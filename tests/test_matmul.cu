// aegis::matmul end to end: for shapes that hit each selection path (exact
// winner, nearest-M fallback, cuBLAS fallback, unaligned), compare against
// cuBLAS with the bench tolerance and print which impl was chosen; then the
// dispatch floor's cuBLASLt path (check_floor).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cstring>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "aegis/common.cuh"
#include "bench/gpu_budget.h"
#include "bench/reference.h"
#include "aegis/matmul.h"
#if __has_include("aegis/winners/sm_121.h")
#include "aegis/winners/sm_121.h"
#endif

namespace {

constexpr double kRelTol = 4e-3;  // same as bench_gemm / CLAUDE.md rule 2

bool check(int M, int N, int K) {
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  std::vector<__half> hA(nA), hB(nB), hC(nC), hR(nC);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  for (auto& x : hA) x = __float2half(d(rng));
  for (auto& x : hB) x = __float2half(d(rng));

  __half *A, *B, *C, *R;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, nA * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, nB * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, nC * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&R, nC * 2));
  AEGIS_CUDA_CHECK(cudaMemcpy(A, hA.data(), nA * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, hB.data(), nB * 2, cudaMemcpyHostToDevice));

  AEGIS_CUDA_CHECK(aegis::matmul(A, B, C, M, N, K));
  cublasHandle_t h;
  cublasCreate(&h);
  const float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
               CUDA_R_16F, K, &beta, R, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  AEGIS_CUDA_CHECK(cudaMemcpy(hC.data(), C, nC * 2, cudaMemcpyDeviceToHost));
  AEGIS_CUDA_CHECK(cudaMemcpy(hR.data(), R, nC * 2, cudaMemcpyDeviceToHost));

  double err = 0, ref = 0;
  for (size_t i = 0; i < nC; ++i) {
    const double c = __half2float(hC[i]), r = __half2float(hR[i]);
    err = std::isfinite(c) ? std::max(err, std::abs(c - r)) : INFINITY;
    ref = std::max(ref, std::abs(r));
  }
  const double rel = err / std::max(ref, 1e-12);
  const bool ok = rel <= kRelTol;
  std::printf("%-5s %5dx%5dx%5d  %-30s rel_err=%.3g\n", ok ? "ok" : "FAIL", M, N, K,
              aegis::matmul_choice(M, N, K), rel);
  cublasDestroy(h);
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(B); aegis::budget::gpu_free(C); aegis::budget::gpu_free(R);
  return ok;
}

// cuBLASLt's API log, counted: how many heuristic queries the dispatch made.
int g_heuristic_calls = 0;
void lt_logger(int level, const char* fn, const char*) {
  if (level == 5 && fn && std::strcmp(fn, "cublasLtMatmulAlgoGetHeuristic") == 0) ++g_heuristic_calls;
}

// Dispatch floor (Production bar 3): a shape whose winner is below 80% of
// cublaslt_best runs cuBLASLt. After the first call (which builds the plan),
// calls make no heuristic query, no allocation and no host synchronization:
// 50 of them are captured into one CUDA graph in cudaStreamCaptureModeGlobal,
// where cudaMalloc or any synchronizing call fails the capture
// (capture_rejects_malloc checks that it does), the graph holds kernel nodes
// only, and the call count of cublasLtMatmulAlgoGetHeuristic stays put.
// (Free device memory is no witness: cudaMemGetInfo is system-wide on GB10's
// unified memory and moves with parallel tests.)
bool capture_rejects_malloc() {
  cudaStream_t s;
  AEGIS_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  AEGIS_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  void* p = nullptr;
  const cudaError_t m = cudaMalloc(&p, 1 << 20);
  cudaGraph_t g = nullptr;
  const cudaError_t end = cudaStreamEndCapture(s, &g);
  if (g) cudaGraphDestroy(g);
  if (m == cudaSuccess) cudaFree(p);
  cudaGetLastError();
  cudaStreamDestroy(s);
  const bool ok = m != cudaSuccess || end != cudaSuccess;
  std::printf("%-5s control: cudaMalloc inside a global-mode capture -> %s, end capture %s\n", ok ? "ok" : "FAIL",
              cudaGetErrorString(m), cudaGetErrorString(end));
  return ok;
}

bool check_floor(int M, int N, int K) {
  bool ok = std::strcmp(aegis::matmul_choice(M, N, K), "cublaslt") == 0;
  std::printf("%-5s %5dx%5dx%5d  dispatch %s (want cublaslt)\n", ok ? "ok" : "FAIL", M, N, K,
              aegis::matmul_choice(M, N, K));
  ok &= check(M, N, K);  // correct against cuBLAS (also the first, plan-building call)
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  __half *A, *B, *C;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, nA * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, nB * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, nC * 2));
  AEGIS_CUDA_CHECK(cudaMemset(A, 0, nA * 2));
  AEGIS_CUDA_CHECK(cudaMemset(B, 0, nB * 2));
  cudaStream_t s;
  AEGIS_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int before_plan = g_heuristic_calls;
  AEGIS_CUDA_CHECK(aegis::matmul(A, B, C, M, N, K, s));  // builds this stream's plan
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(s));
  const int heuristics = g_heuristic_calls;
  // The counter must see the plan-building query, or "0 queries" below proves nothing.
  const bool counted = heuristics - before_plan == 1;

  constexpr int kCalls = 50;
  AEGIS_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  cudaError_t call = cudaSuccess;
  for (int i = 0; i < kCalls && call == cudaSuccess; ++i) call = aegis::matmul(A, B, C, M, N, K, s);
  cudaGraph_t g = nullptr;
  const cudaError_t end = cudaStreamEndCapture(s, &g);
  bool captured = call == cudaSuccess && end == cudaSuccess && g;
  size_t nodes = 0, kernels = 0;
  if (captured) {
    AEGIS_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
    std::vector<cudaGraphNode_t> v(nodes);
    AEGIS_CUDA_CHECK(cudaGraphGetNodes(g, v.data(), &nodes));
    for (cudaGraphNode_t n : v) {
      cudaGraphNodeType t;
      AEGIS_CUDA_CHECK(cudaGraphNodeGetType(n, &t));
      kernels += t == cudaGraphNodeTypeKernel;
    }
    captured = nodes >= size_t(kCalls) && kernels == nodes;
    // The captured calls run correctly too.
    cudaGraphExec_t x;
    AEGIS_CUDA_CHECK(cudaGraphInstantiate(&x, g, 0));
    AEGIS_CUDA_CHECK(cudaGraphLaunch(x, s));
    AEGIS_CUDA_CHECK(cudaStreamSynchronize(s));
    AEGIS_CUDA_CHECK(cudaGraphExecDestroy(x));
    AEGIS_CUDA_CHECK(cudaGraphDestroy(g));
  } else {
    cudaGetLastError();
  }
  const int extra_heuristics = g_heuristic_calls - heuristics;
  const bool steady = counted && captured && extra_heuristics == 0;
  std::printf("%-5s %5dx%5dx%5d  steady state: %d calls in one global-mode capture %s (%zu nodes, %zu kernels), "
              "heuristic queries %d (plan build: %d)\n",
              steady ? "ok" : "FAIL", M, N, K, kCalls, captured ? "ok" : "FAILED", nodes, kernels, extra_heuristics,
              heuristics - before_plan);
  if (call != cudaSuccess || end != cudaSuccess)
    std::printf("      capture: matmul %s, end %s (%s)\n", cudaGetErrorString(call), cudaGetErrorString(end),
                aegis::matmul_last_error());
  AEGIS_CUDA_CHECK(cudaStreamDestroy(s));
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(B); aegis::budget::gpu_free(C);
  return ok && steady;
}

// The cuBLASLt plan on general parameters (layouts, padded ld, misaligned
// pointers, alpha/beta, strided batch, NaN) against the FP64 CPU reference
// with the Production-bar tolerance (src/bench/reference.h). `offset_b`
// shifts B by that many elements (misaligned base pointer).
bool check_floor_general(const char* what, aegis::gemm::GemmParams p, int offset_b, bool nan_in_a) {
  using aegis::bench::HostGemm;
  HostGemm g;
  g.p = p;
  const size_t na = aegis::bench::span_a(p), nb = aegis::bench::span_b(p), nc = aegis::bench::span_c(p);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  g.A.resize(na); g.B.resize(nb); g.C0.resize(nc);
  for (auto& x : g.A) x = __float2half(d(rng));
  for (auto& x : g.B) x = __float2half(d(rng));
  for (auto& x : g.C0) x = __float2half(d(rng));
  if (nan_in_a) g.A[0] = __float2half(NAN);
  __half *A, *Bbase, *C;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, na * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&Bbase, (nb + offset_b) * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, nc * 2));
  __half* B = Bbase + offset_b;
  AEGIS_CUDA_CHECK(cudaMemcpy(A, g.A.data(), na * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, g.B.data(), nb * 2, cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(C, g.C0.data(), nc * 2, cudaMemcpyHostToDevice));
  p.A = A; p.B = B; p.C = C;
  const char* choice = aegis::matmul_choice(p.M, p.N, p.K);
  const cudaError_t e = aegis::matmul(p);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> out(nc);
  AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), C, nc * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref;
  aegis::bench::cpu_reference(g, ref);
  const aegis::bench::Comparison c = aegis::bench::compare(g, out, ref, aegis::bench::tolerance(g).tol);
  const bool ok = e == cudaSuccess && std::strcmp(choice, "cublaslt") == 0 && c.pass;
  std::printf("%-5s %5dx%5dx%5d  %s via %s: max |err|/tol %.2f%s%s\n", ok ? "ok" : "FAIL", p.M, p.N, p.K, what, choice,
              c.max_ratio, c.pass ? "" : ", ", c.first_failure.c_str());
  if (e != cudaSuccess) std::printf("      matmul: %s (%s)\n", cudaGetErrorString(e), aegis::matmul_last_error());
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(Bbase); aegis::budget::gpu_free(C);
  return ok;
}

aegis::gemm::GemmParams general(int M, int N, int K, aegis::gemm::Layout la, aegis::gemm::Layout lb, int pad) {
  using aegis::gemm::Layout;
  aegis::gemm::GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.layout_a = la; p.layout_b = lb;
  p.lda = (la == Layout::RowMajor ? K : M) + pad;
  p.ldb = (lb == Layout::RowMajor ? N : K) + pad;
  p.ldc = N + pad;
  return p;
}

}  // namespace

int main() {
  bool ok = true;
#ifdef AEGIS_WINNERS_SM121
  int dev = 0, major = 0, minor = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  AEGIS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  AEGIS_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  if (major * 10 + minor == 121) {
    std::FILE* devnull = std::fopen("/dev/null", "w");  // the callback counts; nothing is printed
    if (!devnull || cublasLtLoggerSetFile(devnull) != CUBLAS_STATUS_SUCCESS ||
        cublasLtLoggerSetCallback(lt_logger) != CUBLAS_STATUS_SUCCESS || cublasLtLoggerSetLevel(5) != CUBLAS_STATUS_SUCCESS) {
      std::printf("FAIL: cannot install the cuBLASLt logger\n");
      return 1;
    }
    ok &= capture_rejects_malloc();
    // The three sub-80% shapes of 2026-10-17 (winners/sm_121.h, dispatch Cublaslt).
    ok &= check_floor(4096, 4096, 4096);
    ok &= check_floor(1, 896, 4864);
    ok &= check_floor(16, 896, 4864);
    ok &= check_floor(8, 896, 4864);  // nearest M (1 or 16) is below the floor too
    {
      using aegis::gemm::Layout;
      auto p = general(16, 896, 4864, Layout::ColMajor, Layout::RowMajor, 3);
      p.alpha = -0.5f; p.beta = 0.75f;
      ok &= check_floor_general("A col, B row, ld +3, alpha -0.5, beta 0.75", p, 0, false);
      p = general(5, 896, 4864, Layout::RowMajor, Layout::ColMajor, 5);
      ok &= check_floor_general("nearest M; A row, B col, ld +5, B misaligned by 2 B", p, 1, false);
      p = general(16, 896, 4864, Layout::ColMajor, Layout::ColMajor, 0);
      p.batch = 3;
      p.stride_a = p.lda * 4864 + 8; p.stride_b = p.ldb * 896 + 8; p.stride_c = p.ldc * 16 + 8;  // col A: lda x K
      p.beta = 1.f;
      ok &= check_floor_general("batch 3, strides +8, both col, beta 1", p, 0, false);
      p = general(1, 896, 4864, Layout::RowMajor, Layout::RowMajor, 0);
      ok &= check_floor_general("NaN in A propagates", p, 0, true);
    }
    // Every other row the header sends to cuBLASLt.
    for (const aegis::detail::Winner& w : kWinners_sm121)
      if (w.dispatch == aegis::detail::Dispatch::Cublaslt && !(w.N == 896 && w.K == 4864 && w.M <= 16) &&
          !(w.M == 4096 && w.N == 4096 && w.K == 4096))
        ok &= check_floor(w.M, w.N, w.K);
    cublasLtLoggerSetLevel(0);
    cublasLtLoggerSetCallback(nullptr);
  }
#endif
  ok &= check(1024, 1024, 1024);  // tuned exact shape
  ok &= check(512, 896, 4864);    // tuned exact shape
  ok &= check(8, 896, 896);       // nearest M among tuned (1, 16, 512)
  ok &= check(300, 4864, 896);    // nearest M among tuned
  ok &= check(100, 200, 300);     // no tuned N,K: cuBLAS
  ok &= check(127, 129, 65);      // unaligned, cuBLAS
  return ok ? 0 : 1;
}
