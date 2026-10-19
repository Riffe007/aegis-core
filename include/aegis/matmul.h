#pragma once
// aegis::matmul: one entry point that runs the tuned winner for this device.
//
//   C[M,N] = A[M,K] * B[K,N], row-major, FP16 in, FP32 accumulate, FP16 out;
//   or, general: matmul(GemmParams) (layouts, ld, alpha/beta, strided batch;
//   see aegis/gemm.h). M, N, K or batch == 0: success, nothing launched.
//
// Selection, against the winners compiled in for the device's sm_arch
// (include/aegis/winners/sm_<arch>.h, written by aegis-perf emit-header):
//   1. exact (M, N, K) match;
//   2. else the winner with the same N and K whose M is nearest in log2
//      (ties go to the larger M);
//   3. else cuBLAS (also if the winner's impl is no longer registered or has
//      no general entry).
// Dispatch floor (Production bar 3): a winner whose paired CI of
// winner / cublaslt_best lies entirely below 0.80 (upper bound < 0.80; without
// a paired CI, the unpaired median ratio) is emitted with dispatch Cublaslt,
// and its shapes (exact or nearest-M) run cuBLASLt instead of the kernel:
// the stored algorithm cublaslt_best picked at the winner's exact shape (used
// only when every operand is at least 16 B aligned and meets the algorithm's
// alignment caps), else the heuristic's first algorithm (workspace limit 32 MiB,
// as cublaslt_best). Either way only split-K reductions with FP32 partials
// (splitk 1 or COMPUTE_TYPE; never INPLACE or OUTPUT_TYPE, which add FP16
// partials); no such algorithm: cuBLAS, whose reductions are FP32 too. Chosen
// per (shape, layouts, ld, batch, strides, pointer alignment, stream), picked
// on the first such call on a host thread and cached with its descriptors and
// workspace (split-K algorithms need one; it comes from the per-stream
// workspace cache, src/gemm/workspace.h, allocated on the first call that needs
// its size class on the stream, so make that first call outside stream
// capture). Later calls only launch: no heuristic query, no allocation, no
// host synchronization. A plan, and so its workspace, belongs to one stream.
// The decode GEMV family (gemv_*, gemv_tc_*: deterministic split-K) takes its
// partials and counters from the same cache, with the same first-call rule.
// Winners are tuned on row-major NN; other layouts run the same config.
// Asynchronous on `stream`, like the per-impl launchers. Never exits the
// process: returns cudaSuccess or the error (a launch-configuration error, or
// cudaErrorUnknown when cuBLAS fails); matmul_last_error() describes the last
// failure on this host thread.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

#include "aegis/gemm.h"

namespace aegis {

namespace detail {
enum class Dispatch { Aegis, Cublaslt };
// The cuBLASLt algorithm cublaslt_best picked for the winner's shape in its
// tune run (every CUBLASLT_ALGO_CONFIG_* attribute; src/lt_algo.h rebuilds
// it). id < 0: none recorded.
struct LtAlgoConfig {
  int id = -1;
  int tile = 0, stages = 0, splitk = 1, reduction = 0, swizzle = 0, custom = 0, inner_shape = 0, cluster_shape = 0;
};
struct Winner {
  int M, N, K;
  const char* impl;  // aegis::gemm registry name (the tuned config, also when not dispatched)
  Dispatch dispatch = Dispatch::Aegis;
  double pct_of_cublaslt = 0;      // winner median vs cublaslt_best (unpaired), %
  double lt_ci_lo = 0, lt_ci_hi = 0;  // paired CI of cublaslt_best / winner time; 0 = none
  LtAlgoConfig lt = {};               // cublaslt_best's algorithm at this exact shape
};
inline constexpr double kDispatchFloor = 0.80;  // Production bar 3
// For tests. The winner aegis::matmul would use for this shape (exact or
// nearest M), or nullptr.
const Winner* winner_for(int M, int N, int K);

// Capability-aware selection (aegis-hw step 4). Of `table`'s rows for (N, K),
// only those this device can run take part (gemm::supported on the row's kernel;
// a Cublaslt row always can); among them the nearest M (exact, then the
// smallest |log2 M ratio|, then the larger M). Every row of (N, K) filtered
// out: cuBLASLt, with the nearest such row as the stored-algorithm hint. No
// row for (N, K): cuBLAS. No allocation unless `why` is given.
struct Choice {
  enum class Kind { Aegis, Cublaslt, Cublas } kind = Kind::Cublas;
  const Winner* winner = nullptr;             // the row chosen (or the hint); nullptr for cuBLAS
  const gemm::KernelEntry* kernel = nullptr;  // Kind::Aegis
  int filtered = 0;                           // rows of (N, K) skipped for unmet requirements
};
Choice choose(std::span<const Winner> table, int M, int N, int K, const gemm::DeviceCaps& caps,
              std::string* why = nullptr);  // why: "impl: reason; ..." per skipped row
// The winners table of the current device's arch (hw::current; empty if none).
std::span<const Winner> winners_table();
// Which algorithm this thread's cuBLASLt plan for (p, stream, stored) runs:
// "stored", "heuristic", or "heuristic (stored algo N rejected: ...)"; builds
// the plan if needed ("" and matmul_last_error() on failure). aegis::matmul
// passes the exact-shape winner's lt; a test may pass any config. `config`:
// the attributes of the algorithm the plan runs.
std::string cublaslt_plan_source(const gemm::GemmParams& p, cudaStream_t stream, const LtAlgoConfig* stored,
                                 LtAlgoConfig* config = nullptr);
// The cuBLASLt path runs only split-K reductions that keep partials in FP32:
// splitk 1, or CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE (src/lt_algo.h). A stored
// algorithm that fails this is rejected and the (masked) heuristic used.
bool lt_reduction_ok(const LtAlgoConfig& c);
// For tests and the fuzzer: aegis::matmul's argument handling (validate, empty,
// alpha == 0) with the GEMM forced onto one path: Cublaslt (the floor / filtered
// path, `stored` as the algorithm hint), Cublas (the no-winner fallback), or
// Aegis (plain aegis::matmul).
cudaError_t matmul_forced(const gemm::GemmParams& p, Choice::Kind kind, const LtAlgoConfig* stored,
                          cudaStream_t stream);
}  // namespace detail

cudaError_t matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream = nullptr);

// General form. Invalid parameters: cudaErrorInvalidValue (see gemm::validate).
cudaError_t matmul(const gemm::GemmParams& p, cudaStream_t stream = nullptr);

// Text of the last matmul() failure on this host thread ("" if none yet).
const char* matmul_last_error();

// The impl matmul() would run for this shape on the current device:
// a registry name, "cublaslt" (dispatch floor) or "cublas". For logging and tests.
const char* matmul_choice(int M, int N, int K);

}  // namespace aegis
