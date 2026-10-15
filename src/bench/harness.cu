// The AEGIS timing protocol (see harness.h), protocol v1 and v2, the timer
// resolution probe and the NVML idle guard.
#include "harness.h"
#include "gpu_budget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

#include <dlfcn.h>

#include <cstring>
#include <type_traits>

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
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, nA * sizeof(__half)));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, nB * sizeof(__half)));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&Ref, nC * sizeof(__half)));
  AEGIS_CUDA_CHECK(cudaMemcpy(A, hA.data(), nA * sizeof(__half), cudaMemcpyHostToDevice));
  AEGIS_CUDA_CHECK(cudaMemcpy(B, hB.data(), nB * sizeof(__half), cudaMemcpyHostToDevice));
}

size_t Problem::footprint_bytes() const {
  return sizeof(__half) * (size_t(M) * K + size_t(K) * N + size_t(M) * N);
}

void Problem::ensure_sets(int k) {
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  while (int(extra_.size()) + 1 < k) {
    BufferSet s;
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&s.A, nA * sizeof(__half)));
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&s.B, nB * sizeof(__half)));
    AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&s.C, nC * sizeof(__half)));
    AEGIS_CUDA_CHECK(cudaMemcpy(s.A, A, nA * sizeof(__half), cudaMemcpyDeviceToDevice));
    AEGIS_CUDA_CHECK(cudaMemcpy(s.B, B, nB * sizeof(__half), cudaMemcpyDeviceToDevice));
    extra_.push_back(s);
  }
}

BufferSet Problem::set(int i) const {
  return i == 0 ? BufferSet{A, B, C} : extra_[size_t(i) - 1];
}

Problem::~Problem() {
  for (const BufferSet& s : extra_) {
    aegis::budget::gpu_free(s.C);
    aegis::budget::gpu_free(s.B);
    aegis::budget::gpu_free(s.A);
  }
  aegis::budget::gpu_free(Ref);
  aegis::budget::gpu_free(C);
  aegis::budget::gpu_free(B);
  aegis::budget::gpu_free(A);
}

namespace {
__global__ void spin_kernel(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}
}  // namespace

// Event pairs around spin kernels of 0..~8 us, so the deltas cross many tick
// boundaries. The tick is the smallest step between distinct deltas. The
// smallest nonzero delta itself is NOT the tick: on GB10 it is the minimum
// event-to-event spacing (~0.7 us with nothing between the events, ~2.2 us
// around a kernel), while distinct deltas step by 32 ns. The coarse 2.048 us
// steps seen in protocol v1 medians come from recording the start event right
// behind the L2-flush memset, not from the timer.
TimerResolution measure_timer_resolution(cudaStream_t stream, int samples) {
  std::vector<cudaEvent_t> ev(2 * size_t(samples));
  for (auto& e : ev) AEGIS_CUDA_CHECK(cudaEventCreate(&e));
  for (int i = 0; i < samples; ++i) {
    AEGIS_CUDA_CHECK(cudaEventRecord(ev[2 * i], stream));
    spin_kernel<<<1, 1, 0, stream>>>(static_cast<long long>(i) * 16);
    AEGIS_CUDA_CHECK(cudaGetLastError());
    AEGIS_CUDA_CHECK(cudaEventRecord(ev[2 * i + 1], stream));
  }
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<long long> ns;
  for (int i = 0; i < samples; ++i) {
    float ms = 0;
    AEGIS_CUDA_CHECK(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]));
    if (ms > 0) ns.push_back(std::llround(double(ms) * 1e6));
  }
  for (auto& e : ev) cudaEventDestroy(e);
  std::sort(ns.begin(), ns.end());
  ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
  TimerResolution r;
  if (!ns.empty()) r.min_delta_ns = double(ns.front());
  for (size_t i = 1; i < ns.size(); ++i)
    r.tick_ns = (r.tick_ns == 0) ? double(ns[i] - ns[i - 1]) : std::min(r.tick_ns, double(ns[i] - ns[i - 1]));
  return r;
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
  // cuBLAS must not allocate while a graph is being captured.
  constexpr size_t kCublasWorkspace = size_t(32) << 20;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&cublas_ws_, kCublasWorkspace));
  AEGIS_CUBLAS_CHECK(cublasSetWorkspace(cublas_, cublas_ws_, kCublasWorkspace));
  AEGIS_CUBLAS_CHECK(cublasLtCreate(&lt_));
  lt_ws_bytes_ = kCublasWorkspace;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&lt_ws_, lt_ws_bytes_));
  // L2 flush buffer: 2x L2 so nothing from the previous rep survives.
  flush_bytes_ = std::max<size_t>(size_t(prop_.l2CacheSize) * 2, 1 << 20);
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&flush_, flush_bytes_));
  timer_ = measure_timer_resolution(stream_);
}

Harness::~Harness() {
  aegis::budget::gpu_free(flush_);
  cublasDestroy(cublas_);
  aegis::budget::gpu_free(cublas_ws_);
  aegis::budget::gpu_free(lt_ws_);
  cublasLtDestroy(lt_);
  cudaStreamDestroy(stream_);
}

void Harness::launch(const Impl& e, Problem& p, int set) {
  const BufferSet s = p.set(set);
  if (e.e) {
    AEGIS_CUDA_CHECK(e.e->launch(s.A, s.B, s.C, p.M, p.N, p.K, stream_));
  } else if (e.plan) {
    const LtPlan& lp = *e.plan;
    const LtAlgo& al = lp.algos[size_t(e.lt)];
    const float alpha = 1.f, beta = 0.f;
    AEGIS_CUBLAS_CHECK(cublasLtMatmul(lt_, lp.op, &alpha, s.B, lp.a, s.A, lp.b, &beta, s.C, lp.c, s.C,
                                      lp.c, &al.algo, lt_ws_, al.workspace, stream_));
  } else {
    cublas_gemm(cublas_, s.A, s.B, s.C, p.M, p.N, p.K);
  }
}

void Harness::run_once(const Impl& e, Problem& p) {
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

Timing Harness::time(const Impl& e, Problem& p, const TimingOptions& o) {
  if (o.protocol == 1) return time_v1(e, p, o);
  if (o.protocol == 2) return time_v2(e, p, o);
  std::fprintf(stderr, "unknown timing protocol %d\n", o.protocol);
  std::exit(2);
}

void Harness::flush_l2(int value) {
  AEGIS_CUDA_CHECK(cudaMemsetAsync(flush_, value & 0xff, flush_bytes_, stream_));
}

Timing Harness::time_v1(const Impl& e, Problem& p, const TimingOptions& o) {
  const int reps = o.reps;
  for (int i = 0; i < o.warmup; ++i) launch(e, p);
  std::vector<cudaEvent_t> starts(reps), stops(reps);
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventCreate(&starts[i]));
    AEGIS_CUDA_CHECK(cudaEventCreate(&stops[i]));
  }
  for (int i = 0; i < reps; ++i) {
    flush_l2(i);
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
  t.protocol_version = 1;
  return t;
}

int Harness::cold_sets(const Problem& p) const {
  const double l2x3 = 3.0 * double(prop_.l2CacheSize);
  return std::max(2, int(std::ceil(l2x3 / double(p.footprint_bytes()))));
}

cudaGraphExec_t Harness::capture(const Impl& e, Problem& p, long long n, int sets) {
  cudaGraph_t graph = nullptr;
  AEGIS_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  for (long long i = 0; i < n; ++i) launch(e, p, int(i % sets));
  AEGIS_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  AEGIS_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  AEGIS_CUDA_CHECK(cudaGraphDestroy(graph));
  AEGIS_CUDA_CHECK(cudaGraphUpload(exec, stream_));
  return exec;
}

float Harness::timed_window(cudaGraphExec_t g, bool flush) {
  cudaEvent_t start, stop;
  AEGIS_CUDA_CHECK(cudaEventCreate(&start));
  AEGIS_CUDA_CHECK(cudaEventCreate(&stop));
  if (flush) flush_l2(0);
  AEGIS_CUDA_CHECK(cudaEventRecord(start, stream_));
  AEGIS_CUDA_CHECK(cudaGraphLaunch(g, stream_));
  AEGIS_CUDA_CHECK(cudaEventRecord(stop, stream_));
  AEGIS_CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  AEGIS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

Harness::Calibrated Harness::calibrate(const Impl& e, Problem& p, const TimingOptions& o) {
  const double target_ms =
      std::max(o.min_window_ms, o.min_window_ticks * timer_.tick_ns * 1e-6);
  // Cache state. cold: launch i uses buffer set i mod K, with K sets covering
  // >= 3x L2, so no launch finds its operands in L2 from an earlier launch;
  // L2 is also flushed before each window (outside it). warm: one set, no
  // flush; the operands stay L2-resident across launches when they fit.
  Calibrated c;
  c.cold = o.cache == CacheMode::Cold;
  c.sets = c.cold ? cold_sets(p) : 1;
  p.ensure_sets(c.sets);

  // Grow the batch until one window reaches the target. Each candidate graph
  // is replayed once untimed (first-replay costs) first.
  for (int iter = 0;; ++iter) {
    c.graph = capture(e, p, c.n, c.sets);
    AEGIS_CUDA_CHECK(cudaGraphLaunch(c.graph, stream_));
    const float ms = timed_window(c.graph, c.cold);
    if (ms >= target_ms || c.n >= o.max_batch || iter >= 16) {
      if (ms < target_ms)
        std::fprintf(stderr, "warning: window %.3f ms < target %.3f ms at N=%lld\n", ms, target_ms, c.n);
      break;
    }
    // Aim 25% above the target; grow at most 64x per step (the first windows
    // of tiny batches are dominated by fixed graph-launch cost).
    const double want = ms > 0 ? std::ceil(c.n * target_ms * 1.25 / ms) : double(c.n) * 64;
    const long long next = std::min<long long>(
        o.max_batch, std::max<long long>(c.n + 1, std::min<long long>(c.n * 64, (long long)want)));
    AEGIS_CUDA_CHECK(cudaGraphExecDestroy(c.graph));
    c.n = next;
  }
  for (int i = 0; i < o.warmup; ++i) AEGIS_CUDA_CHECK(cudaGraphLaunch(c.graph, stream_));
  return c;
}

Timing Harness::time_v2(const Impl& e, Problem& p, const TimingOptions& o) {
  const Calibrated c = calibrate(e, p, o);
  const long long n = c.n;
  const int reps = o.reps;
  std::vector<cudaEvent_t> starts(reps), stops(reps);
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventCreate(&starts[i]));
    AEGIS_CUDA_CHECK(cudaEventCreate(&stops[i]));
  }
  for (int i = 0; i < reps; ++i) {
    if (c.cold) flush_l2(i);  // outside the window
    AEGIS_CUDA_CHECK(cudaEventRecord(starts[i], stream_));
    AEGIS_CUDA_CHECK(cudaGraphLaunch(c.graph, stream_));
    AEGIS_CUDA_CHECK(cudaEventRecord(stops[i], stream_));
  }
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream_));
  std::vector<float> window(reps), per_launch(reps);
  for (int i = 0; i < reps; ++i) {
    AEGIS_CUDA_CHECK(cudaEventElapsedTime(&window[i], starts[i], stops[i]));
    per_launch[i] = window[i] / float(n);
    cudaEventDestroy(starts[i]);
    cudaEventDestroy(stops[i]);
  }
  AEGIS_CUDA_CHECK(cudaGraphExecDestroy(c.graph));

  Timing t;
  t.reps = reps;
  t.median_us = percentile(per_launch, 0.5) * 1e3;
  t.p10_us = percentile(per_launch, 0.1) * 1e3;
  t.p90_us = percentile(per_launch, 0.9) * 1e3;
  t.protocol_version = 2;
  t.batch_n = n;
  t.window_ms = percentile(window, 0.5);
  t.cache = o.cache;
  t.buffer_sets = c.sets;
  return t;
}

Paired Harness::paired(const Impl& a, const Impl& b, Problem& p, const TimingOptions& o, int pairs,
                       unsigned long long seed, int resamples) {
  if (o.protocol != 2) {
    std::fprintf(stderr, "paired windows need timing protocol v2\n");
    std::exit(2);
  }
  // Each side keeps its own calibrated graph (its own N); a pair is one A
  // window then one B window, back to back, so slow drifts (clock, thermal,
  // background load) hit both sides of a pair alike.
  Calibrated ca = calibrate(a, p, o), cb = calibrate(b, p, o);
  Paired r;
  r.pairs = pairs;
  r.ratio.resize(static_cast<size_t>(pairs));
  for (int i = 0; i < pairs; ++i) {
    const double ta = timed_window(ca.graph, ca.cold) / double(ca.n);
    const double tb = timed_window(cb.graph, cb.cold) / double(cb.n);
    r.ratio[size_t(i)] = float(tb / ta);
  }
  AEGIS_CUDA_CHECK(cudaGraphExecDestroy(ca.graph));
  AEGIS_CUDA_CHECK(cudaGraphExecDestroy(cb.graph));
  r.batch_a = ca.n;
  r.batch_b = cb.n;
  r.median = percentile(r.ratio, 0.5);

  // Percentile bootstrap of the median ratio.
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pick(0, pairs - 1);
  std::vector<float> sample(static_cast<size_t>(pairs)), medians(static_cast<size_t>(resamples));
  for (int k = 0; k < resamples; ++k) {
    for (int i = 0; i < pairs; ++i) sample[size_t(i)] = r.ratio[size_t(pick(rng))];
    medians[size_t(k)] = float(percentile(sample, 0.5));
  }
  r.ci_lo = percentile(medians, 0.025);
  r.ci_hi = percentile(medians, 0.975);
  r.tied = r.ci_lo <= 1.0 && r.ci_hi >= 1.0;
  return r;
}

std::string Harness::json_row(const char* impl, const Impl& impl_e, const Problem& p,
                              const Check& c, const Timing& t, const std::string& extra) const {
  const int M = p.M, N = p.N, K = p.K;
  const double flops = 2.0 * M * N * K;
  // Compulsory traffic only (each operand read once, C written once).
  const double bytes = sizeof(__half) * (double(M) * K + double(K) * N + double(M) * N);
  const double tflops = flops / (t.median_us * 1e-6) / 1e12;
  const double gbps = bytes / (t.median_us * 1e-6) / 1e9;
  const gemm::KernelEntry* e = impl_e.e;
  // Baseline configs are constant, so their kernel ids are stable across shapes;
  // the cuBLASLt algorithm actually picked goes into the row's "lt" field.
  const char* config = e ? e->config
                       : impl_e.plan ? R"({"algo":"cublasLt heuristic, fastest passing of top 8"})"
                                     : R"({"algo":"CUBLAS_GEMM_DEFAULT"})";
  cudaFuncAttributes attrs{};
  if (e) AEGIS_CUDA_CHECK(e->attrs(&attrs));
  auto print = [&](char* buf, size_t n) {
    return std::snprintf(
        buf, n,
        "{\"op\":\"gemm\",\"impl\":\"%s\",\"dtype\":\"f16f32f16\","
        "\"config\":%s,\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
        "\"median_us\":%.3f,\"p10_us\":%.3f,\"p90_us\":%.3f,"
        "\"tflops\":%.4f,\"gbps\":%.3f,\"reps\":%d,"
        "\"checked\":%s,\"pass\":%s,\"max_abs_err\":%.6g,\"rel_err\":%.6g,"
        "\"regs\":%d,\"smem_bytes\":%zu,\"local_bytes\":%zu,"
        "\"device\":\"%s\",\"sm_arch\":%d%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,"
        "\"timer_res_ns\":%.1f,\"protocol_version\":%d,\"batch_n\":%lld,\"window_ms\":%.4f,"
        "\"cache_mode\":\"%s\",\"buffer_sets\":%d%s}",
        impl, config, M, N, K, t.median_us, t.p10_us, t.p90_us, tflops, gbps, t.reps,
        c.checked ? "true" : "false", c.pass ? "true" : "false", c.max_abs_err, c.rel_err,
        e ? attrs.numRegs : -1, e ? attrs.sharedSizeBytes : size_t(0),
        e ? attrs.localSizeBytes : size_t(0), prop_.name, prop_.major, prop_.minor, runtime_,
        driver_, timer_.tick_ns, t.protocol_version, t.batch_n, t.window_ms,
        t.protocol_version == 1 ? "flush_per_launch" : cache_mode_name(t.cache), t.buffer_sets,
        extra.c_str());
  };
  std::string s(size_t(print(nullptr, 0)), '\0');
  print(s.data(), s.size() + 1);
  return s;
}

}  // namespace aegis::bench

namespace aegis::bench {

namespace {

// NVML, loaded at run time: Tegra (Jetson) has no libnvidia-ml, and a link-time
// dependency would stop every harness binary from loading there. Only the
// handful of entry points the guard needs, with their documented C types.
struct Nvml {
  using Ret = int;  // nvmlReturn_t; 0 = NVML_SUCCESS
  using Dev = void*;
  struct Util { unsigned gpu, memory; };  // nvmlUtilization_t
  void* lib = nullptr;
  Ret (*init)() = nullptr;
  Ret (*shutdown)() = nullptr;
  Ret (*by_bus)(const char*, Dev*) = nullptr;
  Ret (*by_index)(unsigned, Dev*) = nullptr;
  Ret (*util)(Dev, Util*) = nullptr;
  const char* (*err)(Ret) = nullptr;

  bool load(std::string& error) {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      error = "libnvidia-ml.so.1 not available";
      return false;
    }
    auto sym = [&](auto& f, const char* name) { f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(dlsym(lib, name)); return f != nullptr; };
    if (!(sym(init, "nvmlInit_v2") && sym(shutdown, "nvmlShutdown") &&
          sym(by_bus, "nvmlDeviceGetHandleByPciBusId_v2") && sym(by_index, "nvmlDeviceGetHandleByIndex_v2") &&
          sym(util, "nvmlDeviceGetUtilizationRates") && sym(err, "nvmlErrorString"))) {
      error = "libnvidia-ml.so.1 lacks a required symbol";
      return false;
    }
    return true;
  }
  ~Nvml() { if (lib) dlclose(lib); }
};

IdleSample sample_nvml(double seconds, int interval_ms) {
  IdleSample s;
  s.source = "nvml";
  Nvml nv;
  if (!nv.load(s.error)) return s;
  if (Nvml::Ret r = nv.init(); r != 0) {
    s.error = std::string("nvmlInit: ") + nv.err(r);
    return s;
  }
  Nvml::Dev dev = nullptr;
  int cuda_dev = 0;
  char bus[64] = "";
  Nvml::Ret r = -1;
  if (cudaGetDevice(&cuda_dev) == cudaSuccess && cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev) == cudaSuccess)
    r = nv.by_bus(bus, &dev);
  if (r != 0) r = nv.by_index(0, &dev);  // single-GPU fallback
  if (r != 0) {
    s.error = std::string("nvmlDeviceGetHandle: ") + nv.err(r);
    nv.shutdown();
    return s;
  }
  const int n = std::max(1, int(seconds * 1000 / interval_ms));
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    Nvml::Util u{};
    if (r = nv.util(dev, &u); r != 0) {
      s.error = std::string("nvmlDeviceGetUtilizationRates: ") + nv.err(r);
      nv.shutdown();
      return s;
    }
    sum += u.gpu;
    s.max_pct = std::max(s.max_pct, double(u.gpu));
    ++s.samples;
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
  }
  s.mean_pct = sum / s.samples;
  s.ok = true;
  nv.shutdown();
  return s;
}

// Tegra: the GPU load file (per mille, 0..1000) of the integrated GPU.
IdleSample sample_tegra_sysfs(double seconds, int interval_ms) {
  IdleSample s;
  static const char* const kPaths[] = {
      "/sys/devices/platform/gpu.0/load",                 // older L4T naming
      "/sys/devices/gpu.0/load",
      "/sys/devices/platform/17000000.gpu/load",          // Orin
      "/sys/devices/platform/bus@0/17000000.gpu/load",
  };
  const char* path = nullptr;
  for (const char* p : kPaths)
    if (std::FILE* f = std::fopen(p, "r")) { std::fclose(f); path = p; break; }
  if (!path) {
    s.error = "no Tegra GPU load file in sysfs";
    return s;
  }
  s.source = std::string("sysfs:") + path;
  const int n = std::max(1, int(seconds * 1000 / interval_ms));
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    std::FILE* f = std::fopen(path, "r");
    int permille = -1;
    if (!f || std::fscanf(f, "%d", &permille) != 1 || permille < 0) {
      if (f) std::fclose(f);
      s.error = std::string("cannot read ") + path;
      return s;
    }
    std::fclose(f);
    const double pct = permille / 10.0;
    sum += pct;
    s.max_pct = std::max(s.max_pct, pct);
    ++s.samples;
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
  }
  s.mean_pct = sum / s.samples;
  s.ok = true;
  return s;
}

// Tegra: GR3D_FREQ load from tegrastats ("GR3D_FREQ 12%@[...]" or "GR3D_FREQ 12%@918").
IdleSample sample_tegrastats(double seconds, int interval_ms) {
  IdleSample s;
  s.source = "tegrastats";
  const int n = std::max(1, int(seconds * 1000 / interval_ms));
  const std::string cmd = "timeout " + std::to_string(int(std::ceil(seconds)) + 2) + " tegrastats --interval " +
                          std::to_string(interval_ms) + " 2>/dev/null";
  std::FILE* p = popen(cmd.c_str(), "r");
  if (!p) {
    s.error = "cannot run tegrastats";
    return s;
  }
  char line[4096];
  double sum = 0;
  while (s.samples < n && std::fgets(line, sizeof line, p)) {
    const char* g = std::strstr(line, "GR3D_FREQ ");
    int pct = -1;
    if (!g || std::sscanf(g, "GR3D_FREQ %d%%", &pct) != 1) continue;
    sum += pct;
    s.max_pct = std::max(s.max_pct, double(pct));
    ++s.samples;
  }
  pclose(p);
  if (s.samples == 0) {
    s.error = "tegrastats printed no GR3D_FREQ";
    return s;
  }
  s.mean_pct = sum / s.samples;
  s.ok = true;
  return s;
}

}  // namespace

// NVML where it exists; on Tegra, the sysfs GPU load, then tegrastats GR3D_FREQ.
IdleSample sample_gpu_utilization(double seconds, int interval_ms) {
  // AEGIS_IDLE_SKIP_NVML=1 forces the Tegra paths (for testing them off-Tegra).
  const char* skip = std::getenv("AEGIS_IDLE_SKIP_NVML");
  IdleSample s;
  if (skip && std::strcmp(skip, "1") == 0) s.error = "NVML skipped (AEGIS_IDLE_SKIP_NVML=1)";
  else s = sample_nvml(seconds, interval_ms);
  if (s.ok) return s;
  std::string why = s.error;
  if (IdleSample t = sample_tegra_sysfs(seconds, interval_ms); t.ok) return t;
  else why += "; " + t.error;
  if (IdleSample t = sample_tegrastats(seconds, interval_ms); t.ok) return t;
  else why += "; " + t.error;
  s.error = why;
  return s;
}

double idle_max_pct() {
  double limit = kIdleMaxPct;
  if (const char* e = std::getenv("AEGIS_IDLE_MAX_PCT")) {
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end != e) limit = std::min(limit, v);  // lower only
  }
  return limit;
}

IdleSample idle_guard(bool allow_busy) {
  const double max_pct = idle_max_pct();
  const IdleSample s = sample_gpu_utilization();
  if (s.ok)
    std::fprintf(stderr, "idle guard: GPU utilization mean %.1f%%, max %.0f%% over %d samples (limit %.0f%%, %s)\n",
                 s.mean_pct, s.max_pct, s.samples, max_pct, s.source.c_str());
  else
    std::fprintf(stderr, "idle guard: cannot read GPU utilization (%s)\n", s.error.c_str());
  const bool busy = idle_busy(s, max_pct);
  if (busy && !allow_busy) {
    std::fprintf(stderr,
                 "idle guard: refusing to benchmark a busy GPU. Stop other GPU work (remote desktop "
                 "included; benchmark over SSH) or pass --allow-busy, which marks every row.\n");
    std::exit(3);
  }
  if (busy) std::fprintf(stderr, "idle guard: --allow-busy given; rows are marked allow_busy\n");
  return s;
}

}  // namespace aegis::bench

namespace aegis::bench {

LtPlan::LtPlan(cublasLtHandle_t lt, int M, int N, int K, size_t max_workspace, int max_algos) {
  // Column-major view of row-major C = A*B: C^T (N x M) = B^T (N x K) * A^T (K x M).
  AEGIS_CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  AEGIS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, N, K, N));
  AEGIS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, K, M, K));
  AEGIS_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, N, M, N));
  cublasLtMatmulPreference_t pref = nullptr;
  AEGIS_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  AEGIS_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof max_workspace));
  cublasLtMatmulHeuristicResult_t res[kLtMaxAlgos] = {};
  max_algos = std::min(max_algos, kLtMaxAlgos);
  int returned = 0;
  AEGIS_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, max_algos, res,
                                                    &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  for (int i = 0; i < returned; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
    LtAlgo al;
    al.algo = res[i].algo;
    al.workspace = res[i].workspaceSize;
    al.rank = i;
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&al.algo, CUBLASLT_ALGO_CONFIG_ID, &al.id, sizeof al.id, &written);
    cublasLtMatmulAlgoConfigGetAttribute(&al.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &al.tile, sizeof al.tile,
                                         &written);
    cublasLtMatmulAlgoConfigGetAttribute(&al.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &al.stages,
                                         sizeof al.stages, &written);
    cublasLtMatmulAlgoConfigGetAttribute(&al.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &al.splitk,
                                         sizeof al.splitk, &written);
    algos.push_back(al);
  }
}

LtPlan::~LtPlan() {
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatmulDescDestroy(op);
}

LtBest Harness::cublaslt_best(Problem& p, const TimingOptions& o, int only_rank) {
  LtBest r;
  r.plan = std::make_unique<LtPlan>(lt_, p.M, p.N, p.K, lt_ws_bytes_, kLtMaxAlgos);
  r.candidates = int(r.plan->algos.size());
  for (int i = 0; i < r.candidates; ++i) {
    if (only_rank >= 0 && r.plan->algos[size_t(i)].rank != only_rank) continue;
    const Impl impl(r.plan.get(), i);
    run_once(impl, p);
    const Check c = check(p);
    if (!c.pass) continue;  // never a baseline that fails the tolerance
    ++r.passed;
    const Timing t = time(impl, p, o);
    if (r.best < 0 || t.median_us < r.timing.median_us) {
      r.best = i;
      r.check = c;
      r.timing = t;
    }
  }
  if (r.best >= 0) run_once(r.impl(), p);  // p.C holds the reported algorithm's result
  return r;
}

std::string Harness::lt_json(const LtBest& b) {
  if (b.best < 0) return "";
  const LtAlgo& al = b.plan->algos[size_t(b.best)];
  char buf[256];
  std::snprintf(buf, sizeof buf,
                ",\"lt\":{\"heuristic_rank\":%d,\"algo_id\":%d,\"tile_id\":%d,\"stages_id\":%d,"
                "\"splitk\":%d,\"workspace\":%zu,\"candidates\":%d,\"passed\":%d}",
                al.rank, al.id, al.tile, al.stages, al.splitk, al.workspace, b.candidates, b.passed);
  return buf;
}

}  // namespace aegis::bench
