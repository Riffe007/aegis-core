#pragma once
// AEGIS timing protocol, shared by bench_gemm (one impl, one shape, one
// process) and aegis_tune (every registered config, many shapes, one process).
// Keep it identical across impls, or comparisons are invalid.
//
// Common to every protocol version:
//   - inputs: uniform [-1, 1] from mt19937(seed), A then B, FP16
//   - one untimed run into C, synchronized
//   - --check: cuBLAS (FP16 in, FP32 compute, FP16 out) into Ref; pass if
//     max_abs_err / max|ref| <= kRelTol
//   - the event timer's resolution is measured once at startup (smallest step
//     between distinct event deltas over 1000 samples) and recorded as
//     timer_res_ns
//
// Protocol v2 (default): batched windows.
//   - calibration: N launches are captured into one CUDA Graph, with N grown
//     until one timed window (one graph replay) lasts >= max(1 ms, 500 timer
//     ticks). The graph keeps CPU launch cost out of the window.
//   - warmup windows are not timed
//   - cache_mode=cold (default; tuning): launch i reads and writes buffer set
//     i mod K, K = max(2, ceil(3 x L2 / problem footprint)) copies of A, B, C,
//     so no launch finds its operands in L2 from an earlier launch; before
//     every timed window (outside it), L2 is also flushed by writing a
//     buffer 2x the L2 size. Nothing but the N launches runs inside a window.
//   - cache_mode=warm: one buffer set, no flush; operands that fit stay
//     L2-resident across launches. Recorded for winners, never used to rank.
//   - per-launch time = window / N; median/p10/p90 across `reps` windows;
//     batch_n = N, cache_mode and buffer_sets = K are recorded
// v2 measures steady-state throughput of back-to-back launches. Single-launch
// latency (launch overhead, cold start) is not this harness's job.
//
// Protocol v1 (--protocol 1, for comparison with old rows): before every timed
// rep, the L2 flush memset; each rep is one launch bracketed by its own
// cudaEvent pair. Recording the start event right behind the memset quantizes
// v1 timings to 2.048 us steps on GB10.
//
// The harness is benchmark code: it aborts the process on CUDA/cuBLAS errors.
// The library path (aegis::matmul, the C ABI) never does.
#include <algorithm>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "aegis/gemm.h"

namespace aegis::bench {

inline constexpr double kRelTol = 4e-3;  // CLAUDE.md rule 2

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T. Aborts on error.
void cublas_gemm(cublasHandle_t h, const __half* A, const __half* B, __half* C, int M, int N,
                 int K);

// A, B, C for one launch.
struct BufferSet {
  __half* A = nullptr;
  __half* B = nullptr;
  __half* C = nullptr;
};

// One device-resident GEMM problem with generated inputs.
class Problem {
 public:
  Problem(int M, int N, int K, unsigned seed);
  ~Problem();
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;

  const int M, N, K;
  __half* A = nullptr;
  __half* B = nullptr;
  __half* C = nullptr;
  __half* Ref = nullptr;

  // A + B + C bytes.
  size_t footprint_bytes() const;
  // At least k buffer sets: set 0 is (A, B, C); the others hold copies of A
  // and B and their own C.
  void ensure_sets(int k);
  BufferSet set(int i) const;

 private:
  std::vector<BufferSet> extra_;
};

struct Check {
  bool checked = false;
  double max_abs_err = -1, rel_err = -1;
  bool pass = true;
};

enum class CacheMode { Cold, Warm };
inline const char* cache_mode_name(CacheMode m) { return m == CacheMode::Cold ? "cold" : "warm"; }

struct TimingOptions {
  int protocol = 2;            // 2: batched windows; 1: one flushed launch per rep
  CacheMode cache = CacheMode::Cold;  // v2 only
  int warmup = 3;              // v2: windows; v1: launches
  int reps = 20;               // v2: timed windows; v1: timed launches
  double min_window_ms = 1.0;  // v2: a window lasts >= max(min_window_ms,
  int min_window_ticks = 500;  //     min_window_ticks x timer resolution)
  long long max_batch = 1 << 16;
};

struct Timing {
  int reps = 0;
  double median_us = 0, p10_us = 0, p90_us = 0;  // per launch
  int protocol_version = 1;
  long long batch_n = 1;     // launches per timed window
  double window_ms = 0;      // median window length (v2)
  CacheMode cache = CacheMode::Cold;  // v2
  int buffer_sets = 1;       // v2: K
};

// Paired comparison (protocol v2): `pairs` alternating windows A, B, A, B...
// ratio[i] = per-launch time of B / per-launch time of A in pair i; a
// percentile bootstrap gives the 95% CI of the median ratio. Tied when the CI
// contains 1.0.
inline constexpr int kPairedWindows = 30;
inline constexpr int kBootstrapResamples = 10000;
struct Paired {
  int pairs = 0;
  std::vector<float> ratio;
  double median = 0, ci_lo = 0, ci_hi = 0;
  bool tied = false;
  long long batch_a = 0, batch_b = 0;
};

struct TimerResolution {
  double tick_ns = 0;       // smallest step between distinct deltas: the resolution
  double min_delta_ns = 0;  // smallest nonzero delta: event spacing, not resolution
};

// cuBLASLt plan for one shape: the operation, its layouts (row-major C = A*B
// as column-major C^T = B^T * A^T, like cublas_gemm), and up to `max_algos`
// heuristic algorithms that fit the workspace.
struct LtAlgo {
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  int rank = 0;  // position in the heuristic's list
  int id = -1, tile = -1, stages = -1, splitk = -1;
};

class LtPlan {
 public:
  LtPlan(cublasLtHandle_t lt, int M, int N, int K, size_t max_workspace, int max_algos);
  ~LtPlan();
  LtPlan(const LtPlan&) = delete;
  LtPlan& operator=(const LtPlan&) = delete;

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  std::vector<LtAlgo> algos;
};

// What a launch runs: a registered kernel, cuBLASLt algorithm `lt` of `plan`,
// or (neither) cuBLAS.
struct Impl {
  const gemm::KernelEntry* e = nullptr;
  const LtPlan* plan = nullptr;
  int lt = -1;
  Impl(const gemm::KernelEntry* k = nullptr) : e(k) {}  // NOLINT: implicit on purpose
  Impl(const LtPlan* p, int i) : plan(p), lt(i) {}
};

// The cublaslt_best baseline: every heuristic algorithm is checked against
// cuBLAS and timed under the same protocol; the fastest passing one is the row.
inline constexpr const char* kLtBestImpl = "cublaslt_best";
inline constexpr int kLtMaxAlgos = 8;
struct LtBest {
  std::unique_ptr<LtPlan> plan;
  int best = -1;  // index into plan->algos; -1: no algorithm passed
  int candidates = 0, passed = 0;
  Check check;
  Timing timing;
  Impl impl() const { return Impl(plan.get(), best); }
};

// Linear interpolation between order statistics.
double percentile(std::vector<float> v, double p);

class Harness {
 public:
  Harness();  // current device: properties, stream, cuBLAS handle, L2 flush buffer
  ~Harness();
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

  const cudaDeviceProp& prop() const { return prop_; }
  cudaStream_t stream() const { return stream_; }
  cublasHandle_t cublas() const { return cublas_; }
  // cudaEvent timer resolution (measured in the constructor).
  const TimerResolution& timer() const { return timer_; }

  // Launch `e` on buffer set `set` of p, asynchronously.
  // Aborts on error.
  void launch(const Impl& e, Problem& p, int set = 0);
  // The untimed run, synchronized.
  void run_once(const Impl& e, Problem& p);
  // cuBLAS into p.Ref, compared with p.C (call after run_once).
  Check check(Problem& p);
  Timing time(const Impl& e, Problem& p, const TimingOptions& o);
  // cublaslt_best on p: run, check and time every heuristic algorithm under
  // `o`; leaves the best one's result in p.C.
  // only_rank >= 0: run just the algorithm at that heuristic rank (no selection;
  // for profiling one algorithm, e.g. under ncu where selection timings lie).
  LtBest cublaslt_best(Problem& p, const TimingOptions& o, int only_rank = -1);
  // Paired tie test of b against a (see Paired); bootstrap seeded with `seed`.
  Paired paired(const Impl& a, const Impl& b, Problem& p, const TimingOptions& o,
                int pairs = kPairedWindows, unsigned long long seed = 0,
                int resamples = kBootstrapResamples);
  // ,"lt":{...}: which algorithm cublaslt_best picked, for the row's extra.
  static std::string lt_json(const LtBest& b);

  // One JSON row (no trailing newline). `extra` (e.g. ,"k":v) is appended
  // inside the object after the standard fields; bench_gemm passes "".
  std::string json_row(const char* impl, const Impl& e, const Problem& p,
                       const Check& c, const Timing& t, const std::string& extra = "") const;

 private:
  Timing time_v1(const Impl& e, Problem& p, const TimingOptions& o);
  // A calibrated v2 measurement: N launches in one graph (caller destroys it).
  struct Calibrated {
    cudaGraphExec_t graph = nullptr;
    long long n = 1;
    int sets = 1;
    bool cold = true;
  };
  Calibrated calibrate(const Impl& e, Problem& p, const TimingOptions& o);
  Timing time_v2(const Impl& e, Problem& p, const TimingOptions& o);

  int cold_sets(const Problem& p) const;
  // N launches (launch i on set i mod `sets`) captured into one instantiated graph.
  cudaGraphExec_t capture(const Impl& e, Problem& p, long long n, int sets);
  // Optionally flush L2, then one timed replay of `g`, synchronized; ms.
  float timed_window(cudaGraphExec_t g, bool flush);
  void flush_l2(int value);

  cudaDeviceProp prop_{};
  int runtime_ = 0, driver_ = 0;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  void* cublas_ws_ = nullptr;  // explicit workspace: no allocation during graph capture
  cublasLtHandle_t lt_ = nullptr;
  void* lt_ws_ = nullptr;
  size_t lt_ws_bytes_ = 0;
  void* flush_ = nullptr;
  size_t flush_bytes_ = 0;
  TimerResolution timer_{};
};

// GPU utilization over a sampling period: NVML (dlopen; device matched to the
// current CUDA device by PCI bus id) where it exists, else on Tegra the sysfs
// GPU load file, else tegrastats GR3D_FREQ. Same 2 s / 2% rule for all.
struct IdleSample {
  bool ok = false;     // a source answered; if false, `error` says why
  std::string source;  // "nvml", "sysfs:<path>" (Tegra GPU load) or "tegrastats"
  double mean_pct = 0, max_pct = 0;
  int samples = 0;
  std::string error;
};
IdleSample sample_gpu_utilization(double seconds = 2.0, int interval_ms = 100);

// Idle guard, run before any sweep, tune, paired A/B or microbench run: samples
// utilization for 2 s and exits the process (status 3) if the mean is above the
// limit (or no source can tell) unless allow_busy. Returns the sample; logs it
// on stderr. The limit is kIdleMaxPct; AEGIS_IDLE_MAX_PCT can only lower it
// (tests force the busy path with -1), never raise it.
inline constexpr double kIdleMaxPct = 2.0;
double idle_max_pct();
// The guard's decision: busy when no source answered or the mean exceeds limit.
inline bool idle_busy(const IdleSample& s, double limit) { return !s.ok || s.mean_pct > limit; }
IdleSample idle_guard(bool allow_busy);

// In-run idle watch: the idle guard keeps sampling while a run executes.
// "Ours" is the process tree of pid `root`; everything else is foreign. Polls
// NVML every 250 ms: device utilization, per-process SM samples
// (nvmlDeviceGetProcessUtilization) and compute contexts
// (nvmlDeviceGetComputeRunningProcesses). First a 2 s baseline, before the
// run's GPU work (the caller waits for `ready`): device D0 and foreign
// per-process sum F0 give the per-process -> device scale k = D0 / F0 (<= 1).
// Then per `window_ms` window: foreign = min(device - ours, k * foreign
// per-process sum), window means. A window is over when foreign > `limit` or
// any foreign process holds a compute context; any such window makes the run
// disturbed, and each is logged on stderr as it happens. Stops when
// `stop(timeout_ms)` returns true (called every poll).
// No NVML (Tegra): ok = false, disturbed unknown.
struct WatchReport {
  bool ok = false;
  std::string source, error;
  double limit_pct = 0;
  double baseline_device_pct = 0, baseline_foreign_pp = 0, scale = 1;  // D0, F0, k
  int windows = 0, windows_over = 0;
  double max_other_pct = 0;  // the worst window's foreign share (device level)
  bool foreign_compute = false;     // a foreign compute context was seen
  bool worst_has_compute = false;
  std::string worst;  // the worst window: "name (pid P) ... at <UTC time>"
  bool disturbed() const { return ok && windows_over > 0; }
};
WatchReport watch_other_utilization(int root, double limit, int window_ms, bool (*stop)(int timeout_ms),
                                    void (*ready)(const WatchReport&));

// The watch's arithmetic (tests/test_idle_rule.cpp). Per-process -> device scale
// from the baseline: k = D0 / F0, D0 floored at 0.5%, at most 1; 1 when F0 = 0.
inline double watch_scale(double d0, double f0) {
  return f0 > 0 ? std::min(1.0, std::max(d0, 0.5) / f0) : 1.0;
}
// A window's foreign share (device level): min(device - ours, k * foreign_pp).
inline double foreign_share(double device, double ours, double foreign_pp, double k) {
  return std::min(std::max(0.0, device - ours), k * foreign_pp);
}

// Event pairs around `samples` spin kernels of increasing length. Aborts on error.
TimerResolution measure_timer_resolution(cudaStream_t stream, int samples = 1000);

}  // namespace aegis::bench
