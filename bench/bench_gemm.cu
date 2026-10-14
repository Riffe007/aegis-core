// bench_gemm: time one GEMM implementation on one shape and emit one JSON line.
//
//   bench_gemm --impl tiled --m 4096 --n 4096 --k 4096 [--reps 50] [--warmup 5]
//              [--check] [--seed 42]
//
// Timing protocol (keep it identical across impls, or comparisons are invalid):
//   - warmup launches are not timed
//   - before every timed rep, L2 is flushed by writing a buffer 2x the L2 size
//   - each rep is bracketed by its own cudaEvent pair; median/p10/p90 reported
// Correctness: --check compares against cuBLAS (FP16 in, FP32 compute, FP16 out)
// and exits 1 if max_abs_err / max|ref| exceeds the tolerance in CLAUDE.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "aegis/common.cuh"
#include "aegis/gemm.h"

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    cublasStatus_t s_ = (expr);                                              \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__,  \
                   __LINE__);                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr double kRelTol = 4e-3;

struct Args {
  std::string impl = "tiled";
  int M = 1024, N = 1024, K = 1024;
  int reps = 50, warmup = 5;
  bool check = false;
  unsigned seed = 42;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (k == "--impl") a.impl = next();
    else if (k == "--m") a.M = std::atoi(next());
    else if (k == "--n") a.N = std::atoi(next());
    else if (k == "--k") a.K = std::atoi(next());
    else if (k == "--reps") a.reps = std::atoi(next());
    else if (k == "--warmup") a.warmup = std::atoi(next());
    else if (k == "--seed") a.seed = static_cast<unsigned>(std::atoi(next()));
    else if (k == "--check") a.check = true;
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.M <= 0 || a.N <= 0 || a.K <= 0 || a.reps <= 0 || a.warmup < 0) {
    std::fprintf(stderr, "invalid shape or rep count\n");
    std::exit(2);
  }
  return a;
}

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T.
void cublas_gemm(cublasHandle_t h, const __half* A, const __half* B, __half* C,
                 int M, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta,
                            C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

struct ImplInfo {
  const char* config;
  cudaFuncAttributes attrs;
  bool has_attrs;
};

// Register new impls here (and in gemm.h, sweep.py IMPLS, CMake tests).
bool dispatch(const std::string& impl, cublasHandle_t h, const __half* A,
              const __half* B, __half* C, int M, int N, int K, cudaStream_t s) {
  using namespace aegis::gemm;
  if (impl == "naive") { launch_naive(A, B, C, M, N, K, s); return true; }
  if (impl == "tiled") { launch_tiled(A, B, C, M, N, K, s); return true; }
  if (impl == "cublas") { cublas_gemm(h, A, B, C, M, N, K); return true; }
  return false;
}

ImplInfo info(const std::string& impl) {
  using namespace aegis::gemm;
  if (impl == "naive") return {CONFIG_naive, attrs_naive(), true};
  if (impl == "tiled") return {CONFIG_tiled, attrs_tiled(), true};
  return {R"({"algo":"CUBLAS_GEMM_DEFAULT"})", {}, false};
}

double percentile(std::vector<float> v, double p) {
  std::sort(v.begin(), v.end());
  const double idx = p * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  const int M = a.M, N = a.N, K = a.K;

  int dev = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  AEGIS_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int rt_ver = 0, drv_ver = 0;
  AEGIS_CUDA_CHECK(cudaRuntimeGetVersion(&rt_ver));
  AEGIS_CUDA_CHECK(cudaDriverGetVersion(&drv_ver));

  cudaStream_t stream;
  AEGIS_CUDA_CHECK(cudaStreamCreate(&stream));
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // Inputs: uniform [-1, 1], fixed seed so every impl sees identical data.
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  std::vector<__half> hA(nA), hB(nB);
  std::mt19937 rng(a.seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : hA) x = __float2half(dist(rng));
  for (auto& x : hB) x = __float2half(dist(rng));

  __half *dA, *dB, *dC, *dRef;
  AEGIS_CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(__half), cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(__half), cudaMemcpyHostToDevice));

  // L2 flush buffer: 2x L2 so nothing from the previous rep survives.
  const size_t flush_bytes = std::max<size_t>(size_t(prop.l2CacheSize) * 2, 1 << 20);
  void* dFlush;
  AEGIS_CUDA_CHECK(cudaMalloc(&dFlush, flush_bytes));

  if (!dispatch(a.impl, handle, dA, dB, dC, M, N, K, stream)) {
    std::fprintf(stderr, "unknown impl '%s'\n", a.impl.c_str());
    return 2;
  }
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));

  // Correctness against cuBLAS.
  double max_abs_err = -1, rel_err = -1;
  bool pass = true;
  if (a.check) {
    cublas_gemm(handle, dA, dB, dRef, M, N, K);
    AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__half> hC(nC), hRef(nC);
    AEGIS_CUDA_CHECK(cudaMemcpy(hC.data(), dC, nC * sizeof(__half), cudaMemcpyDeviceToHost));
    AEGIS_CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, nC * sizeof(__half), cudaMemcpyDeviceToHost));
    double max_ref = 0;
    max_abs_err = 0;
    for (size_t i = 0; i < nC; ++i) {
      const double r = __half2float(hRef[i]), c = __half2float(hC[i]);
      if (!std::isfinite(c)) { max_abs_err = INFINITY; break; }
      max_abs_err = std::max(max_abs_err, std::abs(c - r));
      max_ref = std::max(max_ref, std::abs(r));
    }
    rel_err = max_abs_err / std::max(max_ref, 1e-12);
    pass = rel_err <= kRelTol;
  }

  // Timing.
  for (int i = 0; i < a.warmup; ++i) dispatch(a.impl, handle, dA, dB, dC, M, N, K, stream);
  std::vector<cudaEvent_t> starts(a.reps), stops(a.reps);
  for (int i = 0; i < a.reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventCreate(&starts[i]));
    AEGIS_CUDA_CHECK(cudaEventCreate(&stops[i]));
  }
  for (int i = 0; i < a.reps; ++i) {
    AEGIS_CUDA_CHECK(cudaMemsetAsync(dFlush, i & 0xff, flush_bytes, stream));
    AEGIS_CUDA_CHECK(cudaEventRecord(starts[i], stream));
    dispatch(a.impl, handle, dA, dB, dC, M, N, K, stream);
    AEGIS_CUDA_CHECK(cudaEventRecord(stops[i], stream));
  }
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> ms(a.reps);
  for (int i = 0; i < a.reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventElapsedTime(&ms[i], starts[i], stops[i]));
    cudaEventDestroy(starts[i]);
    cudaEventDestroy(stops[i]);
  }

  const double med_us = percentile(ms, 0.5) * 1e3;
  const double p10_us = percentile(ms, 0.1) * 1e3;
  const double p90_us = percentile(ms, 0.9) * 1e3;
  const double flops = 2.0 * M * N * K;
  // Compulsory traffic only (each operand read once, C written once).
  const double bytes = sizeof(__half) * (double(nA) + nB + nC);
  const double tflops = flops / (med_us * 1e-6) / 1e12;
  const double gbps = bytes / (med_us * 1e-6) / 1e9;

  const ImplInfo ii = info(a.impl);
  std::printf(
      "{\"op\":\"gemm\",\"impl\":\"%s\",\"dtype\":\"f16f32f16\","
      "\"config\":%s,\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
      "\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,"
      "\"tflops\":%.4f,\"gbps\":%.3f,\"reps\":%d,"
      "\"checked\":%s,\"pass\":%s,\"max_abs_err\":%.6g,\"rel_err\":%.6g,"
      "\"regs\":%d,\"smem_bytes\":%zu,\"local_bytes\":%zu,"
      "\"device\":\"%s\",\"sm_arch\":%d%d,\"cuda_runtime\":%d,\"cuda_driver\":%d}\n",
      a.impl.c_str(), ii.config, M, N, K, med_us, p10_us, p90_us, tflops, gbps,
      a.reps, a.check ? "true" : "false", pass ? "true" : "false", max_abs_err,
      rel_err, ii.has_attrs ? ii.attrs.numRegs : -1,
      ii.has_attrs ? ii.attrs.sharedSizeBytes : size_t(0),
      ii.has_attrs ? ii.attrs.localSizeBytes : size_t(0), prop.name, prop.major,
      prop.minor, rt_ver, drv_ver);

  if (a.check && !pass) {
    std::fprintf(stderr, "FAIL: %s %dx%dx%d rel_err=%.3g > %.1e\n",
                 a.impl.c_str(), M, N, K, rel_err, kRelTol);
  }

  cudaFree(dFlush); cudaFree(dRef); cudaFree(dC); cudaFree(dB); cudaFree(dA);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  return (a.check && !pass) ? 1 : 0;
}
