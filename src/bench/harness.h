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
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

struct TimerResolution {
  double tick_ns = 0;       // smallest step between distinct deltas: the resolution
  double min_delta_ns = 0;  // smallest nonzero delta: event spacing, not resolution
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

  // Launch `e` (nullptr = cuBLAS) on buffer set `set` of p, asynchronously.
  // Aborts on error.
  void launch(const gemm::KernelEntry* e, Problem& p, int set = 0);
  // The untimed run, synchronized.
  void run_once(const gemm::KernelEntry* e, Problem& p);
  // cuBLAS into p.Ref, compared with p.C (call after run_once).
  Check check(Problem& p);
  Timing time(const gemm::KernelEntry* e, Problem& p, const TimingOptions& o);

  // One JSON row (no trailing newline). `extra` (e.g. ,"k":v) is appended
  // inside the object after the standard fields; bench_gemm passes "".
  std::string json_row(const char* impl, const gemm::KernelEntry* e, const Problem& p,
                       const Check& c, const Timing& t, const std::string& extra = "") const;

 private:
  Timing time_v1(const gemm::KernelEntry* e, Problem& p, const TimingOptions& o);
  Timing time_v2(const gemm::KernelEntry* e, Problem& p, const TimingOptions& o);

  int cold_sets(const Problem& p) const;
  // N launches (launch i on set i mod `sets`) captured into one instantiated graph.
  cudaGraphExec_t capture(const gemm::KernelEntry* e, Problem& p, long long n, int sets);
  // Optionally flush L2, then one timed replay of `g`, synchronized; ms.
  float timed_window(cudaGraphExec_t g, bool flush);
  void flush_l2(int value);

  cudaDeviceProp prop_{};
  int runtime_ = 0, driver_ = 0;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  void* cublas_ws_ = nullptr;  // explicit workspace: no allocation during graph capture
  void* flush_ = nullptr;
  size_t flush_bytes_ = 0;
  TimerResolution timer_{};
};

// GPU utilization over a sampling period, from NVML (device matched to the
// current CUDA device by PCI bus id).
struct IdleSample {
  bool ok = false;     // NVML answered; if false, `error` says why
  double mean_pct = 0, max_pct = 0;
  int samples = 0;
  std::string error;
};
IdleSample sample_gpu_utilization(double seconds = 2.0, int interval_ms = 100);

// Idle guard, run before any sweep or tune: samples utilization for 2 s and
// exits the process (status 3) if the mean is above `max_pct` (or NVML cannot
// tell) unless allow_busy. Returns the sample; logs it on stderr.
inline constexpr double kIdleMaxPct = 2.0;
IdleSample idle_guard(bool allow_busy, double max_pct = kIdleMaxPct);

// Event pairs around `samples` spin kernels of increasing length. Aborts on error.
TimerResolution measure_timer_resolution(cudaStream_t stream, int samples = 1000);

}  // namespace aegis::bench
