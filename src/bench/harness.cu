// The AEGIS timing protocol (see harness.h). Moved verbatim from
// bench/bench_gemm.cu; bench_gemm's JSON output is unchanged (checked by diff).
#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "aegis/common.cuh"

#define AEGIS_CUBLAS_CHECK(expr)                                             \
  do {                                                                       \
    cublasStatus_t s_ = (expr);                                              \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__,  \
                   __LINE__);                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace aegis::bench {

void cublas_gemm(cublasHandle_t h, const __half* A, const __half* B, __half* C, int M, int N,
                 int K) {
  const float alpha = 1.f, beta = 0.f;
  AEGIS_CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                  B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta,
                                  C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
}

Problem::Problem(int M_, int N_, int K_, unsigned seed) : M(M_), N(N_), K(K_) {
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  std::vector<__half> hA(nA), hB(nB);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : hA) x = __float2half(dist(rng));
  for (auto& x : hB) x = __float2half(dist(rng));
  AEGIS_CUDA_CHECK(cudaMalloc(&A, nA * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&B, nB * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&C, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&Ref, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMemcpy(A, hA.data(), nA * sizeof(__half), cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, hB.data(), nB * sizeof(__half), cudaMemcpyHostToDevice));
}

Problem::~Problem() {
  cudaFree(Ref);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
}

double percentile(std::vector<float> v, double p) {
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

Harness::Harness() {
  int dev = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  AEGIS_CUDA_CHECK(cudaGetDeviceProperties(&prop_, dev));
  AEGIS_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_));
  AEGIS_CUDA_CHECK(cudaDriverGetVersion(&driver_));
  AEGIS_CUDA_CHECK(cudaStreamCreate(&stream_));
  AEGIS_CUBLAS_CHECK(cublasCreate(&cublas_));
  AEGIS_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
  // L2 flush buffer: 2x L2 so nothing from the previous rep survives.
  flush_bytes_ = std::max<size_t>(size_t(prop_.l2CacheSize) * 2, 1 << 20);
  AEGIS_CUDA_CHECK(cudaMalloc(&flush_, flush_bytes_));
}

Harness::~Harness() {
  cudaFree(flush_);
  cublasDestroy(cublas_);
  cudaStreamDestroy(stream_);
}

void Harness::launch(const gemm::KernelEntry* e, Problem& p) {
  if (e) e->launch(p.A, p.B, p.C, p.M, p.N, p.K, stream_);
  else cublas_gemm(cublas_, p.A, p.B, p.C, p.M, p.N, p.K);
}

void Harness::run_once(const gemm::KernelEntry* e, Problem& p) {
  launch(e, p);
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Check Harness::check(Problem& p) {
  const size_t nC = size_t(p.M) * p.N;
  Check c;
  c.checked = true;
  cublas_gemm(cublas_, p.A, p.B, p.Ref, p.M, p.N, p.K);
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream_));
  std::vector<__half> hC(nC), hRef(nC);
  AEGIS_CUDA_CHECK(cudaMemcpy(hC.data(), p.C, nC * sizeof(__half), cudaMemcpyDeviceToHost));
  AEGIS_CUDA_CHECK(cudaMemcpy(hRef.data(), p.Ref, nC * sizeof(__half), cudaMemcpyDeviceToHost));
  double max_ref = 0;
  c.max_abs_err = 0;
  for (size_t i = 0; i < nC; ++i) {
    const double r = __half2float(hRef[i]), x = __half2float(hC[i]);
    if (!std::isfinite(x)) { c.max_abs_err = INFINITY; break; }
    c.max_abs_err = std::max(c.max_abs_err, std::abs(x - r));
    max_ref = std::max(max_ref, std::abs(r));
  }
  c.rel_err = c.max_abs_err / std::max(max_ref, 1e-12);
  c.pass = c.rel_err <= kRelTol;
  return c;
}

Timing Harness::time(const gemm::KernelEntry* e, Problem& p, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) launch(e, p);
  std::vector<cudaEvent_t> starts(reps), stops(reps);
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventCreate(&starts[i]));
    AEGIS_CUDA_CHECK(cudaEventCreate(&stops[i]));
  }
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaMemsetAsync(flush_, i & 0xff, flush_bytes_, stream_));
    AEGIS_CUDA_CHECK(cudaEventRecord(starts[i], stream_));
    launch(e, p);
    AEGIS_CUDA_CHECK(cudaEventRecord(stops[i], stream_));
  }
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream_));
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventElapsedTime(&ms[i], starts[i], stops[i]));
    cudaEventDestroy(starts[i]);
    cudaEventDestroy(stops[i]);
  }
  Timing t;
  t.reps = reps;
  t.median_us = percentile(ms, 0.5) * 1e3;
  t.p10_us = percentile(ms, 0.1) * 1e3;
  t.p90_us = percentile(ms, 0.9) * 1e3;
  return t;
}

std::string Harness::json_row(const char* impl, const gemm::KernelEntry* e, const Problem& p,
                              const Check& c, const Timing& t, const std::string& extra) const {
  const int M = p.M, N = p.N, K = p.K;
  const double flops = 2.0 * M * N * K;
  // Compulsory traffic only (each operand read once, C written once).
  const double bytes = sizeof(__half) * (double(M) * K + double(K) * N + double(M) * N);
  const double tflops = flops / (t.median_us * 1e-6) / 1e12;
  const double gbps = bytes / (t.median_us * 1e-6) / 1e9;
  const char* config = e ? e->config : R"({"algo":"CUBLAS_GEMM_DEFAULT"})";
  const cudaFuncAttributes attrs = e ? e->attrs() : cudaFuncAttributes{};
  auto print = [&](char* buf, size_t n) {
    return std::snprintf(
        buf, n,
        "{\"op\":\"gemm\",\"impl\":\"%s\",\"dtype\":\"f16f32f16\","
        "\"config\":%s,\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
        "\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,"
        "\"tflops\":%.4f,\"gbps\":%.3f,\"reps\":%d,"
        "\"checked\":%s,\"pass\":%s,\"max_abs_err\":%.6g,\"rel_err\":%.6g,"
        "\"regs\":%d,\"smem_bytes\":%zu,\"local_bytes\":%zu,"
        "\"device\":\"%s\",\"sm_arch\":%d%d,\"cuda_runtime\":%d,\"cuda_driver\":%d%s}",
        impl, config, M, N, K, t.median_us, t.p10_us, t.p90_us, tflops, gbps, t.reps,
        c.checked ? "true" : "false", c.pass ? "true" : "false", c.max_abs_err, c.rel_err,
        e ? attrs.numRegs : -1, e ? attrs.sharedSizeBytes : size_t(0),
        e ? attrs.localSizeBytes : size_t(0), prop_.name, prop_.major, prop_.minor, runtime_,
        driver_, extra.c_str());
  };
  std::string s(size_t(print(nullptr, 0)), '\0');
  print(s.data(), s.size() + 1);
  return s;
}

}  // namespace aegis::bench
