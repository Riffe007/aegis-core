// Probe: host-to-completion latency of one GEMM launch, for the M = 1 decode
// winner. This is the single-launch latency that the timing harness (protocol
// v2, steady-state throughput) deliberately does not measure.
//
// One sample = steady_clock before the call, cudaStreamSynchronize, steady_clock
// after: CPU submission + launch + kernel + completion + wake-up. Methods, run
// round-robin (one of each per iteration, so drift hits all alike):
//   runtime      the registry's launch function (<<<>>> + cudaGetLastError)
//   driver       cuLaunchKernel on the same kernel, grid, block, smem and
//                arguments (taken from a captured kernel node)
//   graph        cudaGraphLaunch of an instantiated one-kernel graph
//   matmul       aegis::matmul end to end (winner lookup + launch)
//   null_kernel  an empty <<<1, 32>>> kernel: the floor for any launch
//   sync_only    cudaStreamSynchronize on an idle stream
// Operands stay resident (warm L2), so kernel time is the warm v2 time.
//
// Output: probe "launch_overhead", param {"method","impl","M","N","K",
// "samples","stat"}: us for median / p90 / p99. Host-side probe: no SASS proof.
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "aegis/gemm.h"
#include "aegis/matmul.h"
#include "common.cuh"

namespace {

constexpr int kWarmup = 200, kSamples = 2000;
constexpr const char* kImpl = "mma_pipe_64x64x32_2x2_s4";  // sm_121 winner on all three M = 1 shapes

__global__ void null_kernel() {}

#define CU_CHECK(expr)                                                         \
  do {                                                                         \
    CUresult r_ = (expr);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s_ = nullptr;                                                \
      cuGetErrorString(r_, &s_);                                               \
      std::fprintf(stderr, "CUDA driver error %s at %s:%d\n", s_ ? s_ : "?",   \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}

}  // namespace

int main() {
  const mb::Device dev = mb::device();
  const aegis::gemm::KernelEntry* entry = aegis::gemm::find(kImpl);
  if (!entry) {
    std::fprintf(stderr, "%s is not registered\n", kImpl);
    return 1;
  }
  cudaStream_t s;
  MB_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  struct Shape { int M, N, K; };
  for (const Shape sh : {Shape{1, 896, 896}, Shape{1, 4864, 896}, Shape{1, 896, 4864}}) {
    const size_t nA = size_t(sh.M) * sh.K, nB = size_t(sh.K) * sh.N, nC = size_t(sh.M) * sh.N;
    __half *A, *B, *C;
    MB_CHECK(cudaMalloc(&A, nA * sizeof(__half)));
    MB_CHECK(cudaMalloc(&B, nB * sizeof(__half)));
    MB_CHECK(cudaMalloc(&C, nC * sizeof(__half)));
    MB_CHECK(cudaMemset(A, 0, nA * sizeof(__half)));
    MB_CHECK(cudaMemset(B, 0, nB * sizeof(__half)));
    const char* choice = aegis::matmul_choice(sh.M, sh.N, sh.K);
    if (std::string(choice) != kImpl)
      std::fprintf(stderr, "note: aegis::matmul picks %s for %dx%dx%d\n", choice, sh.M, sh.N, sh.K);

    // Runtime launch once (sets the kernel's smem attribute), then capture one
    // launch to read the kernel node: function, grid, block, smem, arguments.
    MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s));
    MB_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t graph;
    MB_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s));
    MB_CHECK(cudaStreamEndCapture(s, &graph));
    size_t nodes = 1;
    cudaGraphNode_t node;
    MB_CHECK(cudaGraphGetNodes(graph, &node, &nodes));
    cudaKernelNodeParams kp{};
    MB_CHECK(cudaGraphKernelNodeGetParams(node, &kp));
    cudaFunction_t fn;
    MB_CHECK(cudaGetFuncBySymbol(&fn, kp.func));
    cudaGraphExec_t exec;
    MB_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    MB_CHECK(cudaGraphUpload(exec, s));

    const std::vector<std::pair<const char*, std::function<void()>>> methods = {
        {"runtime", [&] { MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s)); }},
        {"driver", [&] {
           CU_CHECK(cuLaunchKernel(reinterpret_cast<CUfunction>(fn), kp.gridDim.x, kp.gridDim.y, kp.gridDim.z,
                                   kp.blockDim.x, kp.blockDim.y, kp.blockDim.z, kp.sharedMemBytes,
                                   reinterpret_cast<CUstream>(s), kp.kernelParams, nullptr));
         }},
        {"graph", [&] { MB_CHECK(cudaGraphLaunch(exec, s)); }},
        {"matmul", [&] { MB_CHECK(aegis::matmul(A, B, C, sh.M, sh.N, sh.K, s)); }},
        {"null_kernel", [&] { null_kernel<<<1, 32, 0, s>>>(); MB_CHECK(cudaGetLastError()); }},
        {"sync_only", [] {}},
    };
    std::vector<std::vector<double>> us(methods.size());
    for (int it = 0; it < kWarmup + kSamples; ++it) {
      for (size_t m = 0; m < methods.size(); ++m) {
        const auto t0 = std::chrono::steady_clock::now();
        methods[m].second();
        MB_CHECK(cudaStreamSynchronize(s));
        const auto t1 = std::chrono::steady_clock::now();
        if (it >= kWarmup) us[m].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }
    for (size_t m = 0; m < methods.size(); ++m) {
      const bool ours = m < 4;
      for (const auto& [stat, p] : {std::pair{"median", 0.5}, {"p90", 0.9}, {"p99", 0.99}}) {
        const std::string param = std::string("{\"method\":\"") + methods[m].first + "\",\"impl\":\"" +
                                  (ours ? kImpl : "-") + "\",\"M\":" + std::to_string(sh.M) +
                                  ",\"N\":" + std::to_string(sh.N) + ",\"K\":" + std::to_string(sh.K) +
                                  ",\"samples\":" + std::to_string(kSamples) + ",\"stat\":\"" + stat + "\"}";
        mb::emit(dev, "launch_overhead", param, pct(us[m], p), "us");
      }
    }
    MB_CHECK(cudaGraphExecDestroy(exec));
    MB_CHECK(cudaGraphDestroy(graph));
    MB_CHECK(cudaFree(C));
    MB_CHECK(cudaFree(B));
    MB_CHECK(cudaFree(A));
  }
  return 0;
}
