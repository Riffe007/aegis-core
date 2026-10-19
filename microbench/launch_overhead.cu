// Probe: host-to-completion latency of one GEMM launch, for the M = 1 decode
// winners. This is the single-launch latency that the timing harness (protocol
// v2, steady-state throughput) deliberately does not measure.
//
// Part 1. One sample = steady_clock before the call, cudaStreamSynchronize,
// steady_clock after: CPU submission + launch + kernel + completion + wake-up.
// Methods, run round-robin (one of each per iteration, so drift hits all alike):
//   runtime      the registry's launch function (<<<>>> + cudaGetLastError)
//   driver       cuLaunchKernel on the same kernel, grid, block, smem and
//                arguments (taken from a captured kernel node)
//   graph        cudaGraphLaunch of an instantiated one-kernel graph
//   matmul       aegis::matmul end to end (winner lookup + launch)
//   null_kernel  an empty <<<1, 32>>> kernel: the floor for any launch
//   sync_only    cudaStreamSynchronize on an idle stream
// Operands stay resident (warm L2), so kernel time is the warm v2 time. The
// kernel is the one aegis::matmul dispatches for the shape (matmul_choice).
//
// Part 2, the host path alone. One sample = steady_clock around the call, no
// synchronization inside it; the stream is drained after every launch, outside
// the sample, so a launch never waits for queue space:
//   host_dispatch        detail::matmul_host_path: validation, profile lookup,
//                        capability-filtered winner choice, cuBLASLt plan lookup
//                        (a cache hit); launches nothing
//   host_enqueue_matmul  aegis::matmul until it returns (dispatch + launch call)
//   host_enqueue_kernel  the chosen kernel's run() until it returns (Aegis shapes)
//   host_clock           two steady_clock reads: the timer's own cost
// for decode shapes, 2048^3 (an Aegis kernel) and 4096^3 (the cuBLASLt floor).
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
constexpr int kHostWarmup = 2000, kHostSamples = 20000;

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

std::string param(const char* method, const std::string& impl, int M, int N, int K, int samples, const char* stat) {
  return std::string("{\"method\":\"") + method + "\",\"impl\":\"" + impl + "\",\"M\":" + std::to_string(M) +
         ",\"N\":" + std::to_string(N) + ",\"K\":" + std::to_string(K) + ",\"samples\":" + std::to_string(samples) +
         ",\"stat\":\"" + stat + "\"}";
}

struct Shape { int M, N, K; };

void end_to_end(const mb::Device& dev, cudaStream_t s) {
  for (const Shape sh : {Shape{1, 896, 896}, Shape{1, 4864, 896}, Shape{1, 896, 4864}}) {
    const size_t nA = size_t(sh.M) * sh.K, nB = size_t(sh.K) * sh.N, nC = size_t(sh.M) * sh.N;
    __half *A, *B, *C;
    MB_CHECK(aegis::budget::gpu_malloc(&A, nA * sizeof(__half)));
    MB_CHECK(aegis::budget::gpu_malloc(&B, nB * sizeof(__half)));
    MB_CHECK(aegis::budget::gpu_malloc(&C, nC * sizeof(__half)));
    MB_CHECK(cudaMemset(A, 0, nA * sizeof(__half)));
    MB_CHECK(cudaMemset(B, 0, nB * sizeof(__half)));
    const std::string impl = aegis::matmul_choice(sh.M, sh.N, sh.K);
    const aegis::gemm::KernelEntry* entry = aegis::gemm::find(impl);
    if (!entry) {
      std::fprintf(stderr, "%dx%dx%d: aegis::matmul runs %s, not a registry kernel\n", sh.M, sh.N, sh.K, impl.c_str());
      std::exit(1);
    }
    // aegis::matmul once (split-K workspace for this stream), a runtime launch
    // (smem attribute), then capture one launch to read the kernel node.
    MB_CHECK(aegis::matmul(A, B, C, sh.M, sh.N, sh.K, s));
    MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s));
    MB_CHECK(cudaStreamSynchronize(s));
    cudaGraph_t graph;
    MB_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s));
    MB_CHECK(cudaStreamEndCapture(s, &graph));
    // A split-K launch captures more than one node; the driver method relaunches
    // only a one-kernel launch.
    size_t nodes = 0;
    MB_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    cudaKernelNodeParams kp{};
    cudaFunction_t fn = nullptr;
    if (nodes == 1) {
      cudaGraphNode_t node;
      MB_CHECK(cudaGraphGetNodes(graph, &node, &nodes));
      cudaGraphNodeType type;
      MB_CHECK(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel) {
        MB_CHECK(cudaGraphKernelNodeGetParams(node, &kp));
        MB_CHECK(cudaGetFuncBySymbol(&fn, kp.func));
      }
    }
    cudaGraphExec_t exec;
    MB_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    MB_CHECK(cudaGraphUpload(exec, s));

    std::vector<std::pair<const char*, std::function<void()>>> methods = {
        {"runtime", [&] { MB_CHECK(entry->launch(A, B, C, sh.M, sh.N, sh.K, s)); }},
        {"graph", [&] { MB_CHECK(cudaGraphLaunch(exec, s)); }},
        {"matmul", [&] { MB_CHECK(aegis::matmul(A, B, C, sh.M, sh.N, sh.K, s)); }},
        {"null_kernel", [&] { null_kernel<<<1, 32, 0, s>>>(); MB_CHECK(cudaGetLastError()); }},
        {"sync_only", [] {}},
    };
    if (fn)
      methods.insert(methods.begin() + 1, {"driver", [&] {
                       CU_CHECK(cuLaunchKernel(reinterpret_cast<CUfunction>(fn), kp.gridDim.x, kp.gridDim.y,
                                               kp.gridDim.z, kp.blockDim.x, kp.blockDim.y, kp.blockDim.z,
                                               kp.sharedMemBytes, reinterpret_cast<CUstream>(s), kp.kernelParams,
                                               nullptr));
                     }});
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
      const std::string name = methods[m].first;
      const bool ours = name != "null_kernel" && name != "sync_only";
      for (const auto& [stat, p] : {std::pair{"median", 0.5}, {"p90", 0.9}, {"p99", 0.99}})
        mb::emit(dev, "launch_overhead", param(methods[m].first, ours ? impl : "-", sh.M, sh.N, sh.K, kSamples, stat),
                 pct(us[m], p), "us");
    }
    MB_CHECK(cudaGraphExecDestroy(exec));
    MB_CHECK(cudaGraphDestroy(graph));
    MB_CHECK(aegis::budget::gpu_free(C));
    MB_CHECK(aegis::budget::gpu_free(B));
    MB_CHECK(aegis::budget::gpu_free(A));
  }
}

void host_path(const mb::Device& dev, cudaStream_t s) {
  using clk = std::chrono::steady_clock;
  for (const Shape sh : {Shape{1, 896, 896}, Shape{16, 4864, 896}, Shape{2048, 2048, 2048}, Shape{4096, 4096, 4096}}) {
    const size_t nA = size_t(sh.M) * sh.K, nB = size_t(sh.K) * sh.N, nC = size_t(sh.M) * sh.N;
    __half *A, *B, *C;
    MB_CHECK(aegis::budget::gpu_malloc(&A, nA * sizeof(__half)));
    MB_CHECK(aegis::budget::gpu_malloc(&B, nB * sizeof(__half)));
    MB_CHECK(aegis::budget::gpu_malloc(&C, nC * sizeof(__half)));
    MB_CHECK(cudaMemset(A, 0, nA * sizeof(__half)));
    MB_CHECK(cudaMemset(B, 0, nB * sizeof(__half)));
    const aegis::gemm::GemmParams p = aegis::gemm::row_major_params(A, B, C, sh.M, sh.N, sh.K);
    const std::string impl = aegis::matmul_choice(sh.M, sh.N, sh.K);
    const aegis::gemm::KernelEntry* entry = aegis::gemm::find(impl);
    MB_CHECK(aegis::matmul(p, s));  // first call: workspace, cuBLASLt plan, smem attributes
    MB_CHECK(cudaStreamSynchronize(s));
    struct Method {
      const char* name;
      std::function<void()> f;
      bool launches;
    };
    std::vector<Method> methods = {
        {"host_dispatch", [&] { aegis::detail::matmul_host_path(p, s); }, false},
        {"host_enqueue_matmul", [&] { MB_CHECK(aegis::matmul(p, s)); }, true},
        {"host_clock", [] {}, false},
    };
    if (entry && entry->run) methods.push_back({"host_enqueue_kernel", [&] { MB_CHECK(entry->run(p, s)); }, true});
    std::vector<std::vector<double>> us(methods.size());
    for (int it = 0; it < kHostWarmup + kHostSamples; ++it) {
      for (size_t m = 0; m < methods.size(); ++m) {
        const auto t0 = clk::now();
        methods[m].f();
        const auto t1 = clk::now();
        if (it >= kHostWarmup) us[m].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (methods[m].launches) MB_CHECK(cudaStreamSynchronize(s));
      }
    }
    for (size_t m = 0; m < methods.size(); ++m)
      for (const auto& [stat, q] : {std::pair{"median", 0.5}, {"p90", 0.9}, {"p99", 0.99}})
        mb::emit(dev, "launch_overhead", param(methods[m].name, impl, sh.M, sh.N, sh.K, kHostSamples, stat),
                 pct(us[m], q), "us");
    MB_CHECK(aegis::budget::gpu_free(C));
    MB_CHECK(aegis::budget::gpu_free(B));
    MB_CHECK(aegis::budget::gpu_free(A));
  }
}

}  // namespace

int main(int argc, char** argv) {
  // --host-only: part 2 alone.
  const bool host_only = argc > 1 && std::string(argv[1]) == "--host-only";
  const mb::Device dev = mb::device();
  cudaStream_t s;
  MB_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  if (!host_only) end_to_end(dev, s);
  host_path(dev, s);
  return 0;
}
