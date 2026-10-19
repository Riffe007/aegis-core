// aegis::matmul dispatch. See include/aegis/matmul.h for the selection rules.
#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "aegis/gemm.h"
#include "aegis/matmul.h"
#include "gemm/workspace.h"
#include "hw/device.h"
#include "hw/profile.h"
#include "lt_algo.h"

// One generated table per arch; a missing file just means no tuned winners.
#if __has_include("aegis/winners/sm_121.h")
#include "aegis/winners/sm_121.h"
#endif
#if __has_include("aegis/winners/sm_87.h")
#include "aegis/winners/sm_87.h"
#endif

namespace aegis {
namespace {

std::span<const detail::Winner> winners_for(int arch) {
  switch (arch) {
#ifdef AEGIS_WINNERS_SM121
    case 121: return kWinners_sm121;
#endif
#ifdef AEGIS_WINNERS_SM87
    case 87: return kWinners_sm87;
#endif
    default: return {};
  }
}

thread_local std::string last_error;

cudaError_t fail(cudaError_t e, const std::string& what) {
  last_error = what + ": " + cudaGetErrorString(e);
  return e;
}

// 0 if the device cannot be queried (then only cuBLAS is used).
// Cached per device (hw::current; a test's simulated profile included).
int current_arch() {
  const hw::DeviceLimits* d = hw::current();
  return d ? d->arch : 0;
}

// The current device's choice (capabilities of hw::current, a simulation included).
detail::Choice choice_for(int M, int N, int K) {
  return detail::choose(winners_for(current_arch()), M, N, K, gemm::current_caps());
}

cudaError_t cublas_fail(const char* what, cublasStatus_t s) {
  last_error = std::string(what) + " failed: cuBLAS status " + std::to_string(int(s));
  return cudaErrorUnknown;
}

// General cuBLAS call for GemmParams. Row-major C (M x N, ldc) is column-major
// C^T (N x M, ldc) = op(B) * op(A): a row-major operand read as column-major is
// already transposed (op N); a column-major one needs op T.
cudaError_t cublas_gemm_general(const gemm::GemmParams& p, cudaStream_t stream) {
  thread_local cublasHandle_t h = nullptr;
  if (!h) {
    cublasStatus_t s = cublasCreate(&h);
    if (s != CUBLAS_STATUS_SUCCESS) {
      h = nullptr;
      return cublas_fail("cublasCreate", s);
    }
    // FP16 output: by default cuBLAS may reduce split-K partials in FP16. That broke
    // the tolerance at 24x1872x14747 CR, misaligned (fuzz_decode seed 41 case 1360:
    // 4.2 ulp, 2.15x tol). Partials must stay FP32, like our own split-K.
    s = cublasSetMathMode(h, cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (s != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(h);
      h = nullptr;
      return cublas_fail("cublasSetMathMode", s);
    }
  }
  cublasStatus_t s = cublasSetStream(h, stream);
  if (s != CUBLAS_STATUS_SUCCESS) return cublas_fail("cublasSetStream", s);
  constexpr long long kIntMax = 2147483647;
  if (p.lda > kIntMax || p.ldb > kIntMax || p.ldc > kIntMax) {
    last_error = "cuBLAS fallback: leading dimension exceeds int";
    return cudaErrorInvalidValue;
  }
  const cublasOperation_t op_b = p.layout_b == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t op_a = p.layout_a == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  s = cublasGemmStridedBatchedEx(h, op_b, op_a, p.N, p.M, p.K, &p.alpha, p.B, CUDA_R_16F, int(p.ldb),
                                 p.stride_b, p.A, CUDA_R_16F, int(p.lda), p.stride_a, &p.beta, p.C,
                                 CUDA_R_16F, int(p.ldc), p.stride_c, p.batch, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT);
  return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cublas_fail("cublasGemmStridedBatchedEx", s);
}

// The dispatch floor's cuBLASLt path (see matmul.h). Same operand mapping as
// cublas_gemm_general: column-major C^T (N x M) = op(B) * op(A).
// Everything that decides which algorithm is valid is in the key: sizes,
// layouts, leading dimensions, batch and strides, the device, and each
// operand's alignment (its pointer and, batched, its byte stride). The stream
// is in the key too, so a plan is never shared between streams.
struct LtKey {
  int dev, M, N, K, batch;
  gemm::Layout la, lb;
  long long lda, ldb, ldc, sa, sb, sc;
  unsigned align_a, align_b, align_c;
  cudaStream_t stream;
  const detail::LtAlgoConfig* stored;  // the winner's recorded algorithm (exact shape), or nullptr
  bool operator==(const LtKey&) const = default;
};

struct LtPlan {
  LtKey key;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;  // cuBLASLt's A (our B), B (our A), C
  cublasLtMatmulAlgo_t algo{};
  void* workspace = nullptr;  // from this stream's workspace cache (split-K partials of the algorithm)
  size_t workspace_bytes = 0;
  std::string source;  // "stored", "heuristic", or "heuristic (stored ... rejected: ...)"
  detail::LtAlgoConfig config;  // the algorithm's config attributes (lt::read_config)
  mutable bool use_cublas = false;  // no cuBLASLt algorithm with an FP32 split-K reduction: cuBLAS runs it
};

// The heuristic's workspace limit: the same as cublaslt_best's (src/bench/harness.cu),
// so the algorithm picked here is rank 0 of the list cublaslt_best times.
constexpr size_t kLtMaxWorkspace = size_t(32) << 20;

// Largest power of two (<= 256) dividing the address and, if batched, the byte stride.
unsigned alignment(const void* p, long long stride_elems, int batch) {
  unsigned long long v = reinterpret_cast<unsigned long long>(p) | 256ull;
  if (batch > 1) v |= static_cast<unsigned long long>(stride_elems) * sizeof(__half);
  return static_cast<unsigned>(v & (~v + 1));
}

// Does every operand meet `algo`'s minimum alignment (CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_*,
// and 16 B)? For stored algorithms only; the heuristic takes the alignments as preferences.
// cuBLASLt's A is our B, its B our A; C and D are our C. An operand's alignment is
// its pointer's and batch stride's (the key) and its leading dimension's in bytes.
bool aligned_for(const cublasLtMatmulAlgo_t& algo, const LtKey& k, const gemm::GemmParams& p) {
  auto with_ld = [](unsigned a, long long ld) { return std::min(a, alignment(nullptr, ld, 2)); };
  const unsigned have[] = {with_ld(k.align_b, p.ldb), with_ld(k.align_a, p.lda), with_ld(k.align_c, p.ldc),
                           with_ld(k.align_c, p.ldc)};
  const cublasLtMatmulAlgoCapAttributes_t cap[] = {
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
  for (int i = 0; i < 4; ++i) {
    uint32_t need = 0;
    size_t w = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, cap[i], &need, sizeof need, &w) != CUBLAS_STATUS_SUCCESS) return false;
    // At least 16 B as well: the tune ran the algorithm on 256 B-aligned, packed
    // operands, and the caps are not enough (algo 67 reports 2 B for C, then fails
    // with C at 6 B: fuzz --large-k seed 62 case 129).
    if (have[i] < std::max<uint32_t>(need, 16)) return false;
  }
  return true;
}

cudaError_t cublaslt_fail(const char* what, cublasStatus_t s) {
  last_error = std::string("cuBLASLt fallback: ") + what + " failed: cuBLAS status " + std::to_string(int(s));
  return cudaErrorUnknown;
}

// Build the plan for `key`: descriptors, the algorithm and its workspace (one
// cudaMalloc per plan, only if the algorithm needs one). The algorithm: the
// one cublaslt_best picked at this exact shape in the tune (key.stored,
// rebuilt from its config attributes and accepted by cublasLtMatmulAlgoCheck
// for these descriptors); else the heuristic's first choice ("rank 1" of its
// list). A stored algorithm that cannot be rebuilt or is rejected (another
// driver or cuBLASLt version, another layout or alignment) is logged on
// stderr once per plan, and the heuristic's choice is used.
cudaError_t make_lt_plan(cublasLtHandle_t lt, const gemm::GemmParams& p, LtPlan& plan) {
  const LtKey& k = plan.key;
  cublasStatus_t s = cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (s != CUBLAS_STATUS_SUCCESS) return cublaslt_fail("cublasLtMatmulDescCreate", s);
  const cublasOperation_t op_b = p.layout_b == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t op_a = p.layout_a == gemm::Layout::RowMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &op_b, sizeof op_b);
  cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &op_a, sizeof op_a);
  // Stored shapes (column-major): our B is N x K (row-major) or K x N; our A is K x M or M x K.
  const bool rb = op_b == CUBLAS_OP_N, ra = op_a == CUBLAS_OP_N;
  auto layout = [&](cublasLtMatrixLayout_t* l, uint64_t rows, uint64_t cols, long long ld, long long stride) {
    cublasStatus_t st = cublasLtMatrixLayoutCreate(l, CUDA_R_16F, rows, cols, ld);
    if (st != CUBLAS_STATUS_SUCCESS || k.batch == 1) return st;
    const int32_t n = k.batch;
    const int64_t off = stride;
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &n, sizeof n);
    return cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &off, sizeof off);
  };
  if ((s = layout(&plan.a, rb ? p.N : p.K, rb ? p.K : p.N, p.ldb, p.stride_b)) != CUBLAS_STATUS_SUCCESS ||
      (s = layout(&plan.b, ra ? p.K : p.M, ra ? p.M : p.K, p.lda, p.stride_a)) != CUBLAS_STATUS_SUCCESS ||
      (s = layout(&plan.c, p.N, p.M, p.ldc, p.stride_c)) != CUBLAS_STATUS_SUCCESS)
    return cublaslt_fail("cublasLtMatrixLayoutCreate", s);
  // The per-stream workspace cache (src/gemm/workspace.h), shared by every plan
  // on this stream; allocated on the first call that needs the size class.
  auto use_workspace = [&](size_t bytes) -> cudaError_t {
    if (bytes == 0) return cudaSuccess;
    if (const cudaError_t e =
            gemm::detail::stream_workspace(k.stream, bytes, &plan.workspace, gemm::detail::WorkspaceUse::CublasLt);
        e != cudaSuccess) {
      plan.workspace = nullptr;
      return fail(e, "cuBLASLt fallback: workspace (" + std::to_string(bytes) +
                         " B; make the first call for this shape outside stream capture)");
    }
    plan.workspace_bytes = bytes;
    return cudaSuccess;
  };
  if (k.stored) {
    cublasLtMatmulHeuristicResult_t check{};
    cublasStatus_t st = lt::rebuild(lt, *k.stored, &plan.algo);
    const char* stage = "rebuild";
    if (st == CUBLAS_STATUS_SUCCESS) {
      stage = "cublasLtMatmulAlgoCheck";
      st = cublasLtMatmulAlgoCheck(lt, plan.op, plan.a, plan.b, plan.c, plan.c, &plan.algo, &check);
      if (st == CUBLAS_STATUS_SUCCESS && check.workspaceSize > kLtMaxWorkspace) {
        stage = "workspace limit";
        st = CUBLAS_STATUS_NOT_SUPPORTED;
      }
      // cublasLtMatmulAlgoCheck sees no pointers: an algorithm it accepts can still
      // fail in cublasLtMatmul (status 15) on a misaligned operand. Each operand's
      // alignment (pointer, batch stride, leading dimension) against the algorithm's minimum.
      if (st == CUBLAS_STATUS_SUCCESS && !aligned_for(plan.algo, k, p)) {
        stage = "operand alignment below the algorithm's minimum";
        st = CUBLAS_STATUS_NOT_SUPPORTED;
      }
      if (st == CUBLAS_STATUS_SUCCESS && !lt::reduction_ok(lt::read_config(plan.algo))) {
        stage = "split-K reduction scheme (partials in FP16)";
        st = CUBLAS_STATUS_NOT_SUPPORTED;
      }
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
      plan.source = "stored";
      plan.config = lt::read_config(plan.algo);
      return use_workspace(check.workspaceSize);
    }
    plan.source = "heuristic (stored algo " + std::to_string(k.stored->id) + " rejected: " + stage +
                  ", cuBLAS status " + std::to_string(int(st)) + ")";
    std::fprintf(stderr, "aegis::matmul: %dx%dx%d: %s; using the heuristic's first choice\n", p.M, p.N, p.K,
                 plan.source.c_str());
  } else {
    plan.source = "heuristic";
  }
  cublasLtMatmulPreference_t pref = nullptr;
  if ((s = cublasLtMatmulPreferenceCreate(&pref)) != CUBLAS_STATUS_SUCCESS)
    return cublaslt_fail("cublasLtMatmulPreferenceCreate", s);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kLtMaxWorkspace,
                                       sizeof kLtMaxWorkspace);
  // cuBLASLt's A is our B and vice versa; C and D are the same buffer.
  const uint32_t al[] = {k.align_b, k.align_a, k.align_c, k.align_c};
  const cublasLtMatmulPreferenceAttributes_t which[] = {
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
  for (int i = 0; i < 4; ++i) cublasLtMatmulPreferenceSetAttribute(pref, which[i], &al[i], sizeof al[i]);
  // Only FP32 split-K reductions (lt::kReductionMask); the result is checked
  // again below, so a heuristic that ignored the mask could not slip one through.
  const uint32_t mask = lt::kReductionMask;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof mask);
  constexpr int kAsk = 8;
  cublasLtMatmulHeuristicResult_t res[kAsk] = {};
  int found = 0;
  s = cublasLtMatmulAlgoGetHeuristic(lt, plan.op, plan.a, plan.b, plan.c, plan.c, pref, kAsk, res, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  if (s != CUBLAS_STATUS_SUCCESS) return cublaslt_fail("cublasLtMatmulAlgoGetHeuristic", s);
  for (int i = 0; i < found; ++i) {
    if (res[i].state != CUBLAS_STATUS_SUCCESS || !lt::reduction_ok(lt::read_config(res[i].algo))) continue;
    plan.algo = res[i].algo;
    plan.config = lt::read_config(plan.algo);
    return use_workspace(res[i].workspaceSize);
  }
  // None of the heuristic's algorithms keeps FP32 partials (seen with misaligned
  // operands at K > 14000): the plan runs cuBLAS, whose reductions are FP32 too
  // (CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION). Decided once per plan.
  plan.use_cublas = true;
  plan.source = "cublas (" + std::to_string(found) + " heuristic algorithms, none with an FP32 split-K reduction)";
  return cudaSuccess;
}

void destroy(LtPlan& plan) {
  if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
  if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
  if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
  if (plan.op) cublasLtMatmulDescDestroy(plan.op);  // the workspace belongs to the stream cache
}

// This thread's plan for (p, stream, stored), built on first use.
cudaError_t lt_plan(const gemm::GemmParams& p, cudaStream_t stream, const detail::LtAlgoConfig* stored,
                    cublasLtHandle_t* handle, const LtPlan** out) {
  // Per host thread, like the cuBLAS handle; never destroyed (process lifetime).
  thread_local cublasLtHandle_t lt = nullptr;
  thread_local std::vector<LtPlan> plans;
  if (!lt) {
    const cublasStatus_t s = cublasLtCreate(&lt);
    if (s != CUBLAS_STATUS_SUCCESS) {
      lt = nullptr;
      return cublaslt_fail("cublasLtCreate", s);
    }
  }
  *handle = lt;
  int dev = 0;
  if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) return fail(e, "cuBLASLt fallback: cudaGetDevice");
  const LtKey key{dev,          p.M,          p.N,
                  p.K,          p.batch,      p.layout_a,
                  p.layout_b,   p.lda,        p.ldb,
                  p.ldc,        p.batch > 1 ? p.stride_a : 0,
                  p.batch > 1 ? p.stride_b : 0,
                  p.batch > 1 ? p.stride_c : 0,
                  alignment(p.A, p.stride_a, p.batch),
                  alignment(p.B, p.stride_b, p.batch),
                  alignment(p.C, p.stride_c, p.batch),
                  stream,
                  stored};
  for (const LtPlan& q : plans)
    if (q.key == key) {
      *out = &q;
      return cudaSuccess;
    }
  LtPlan fresh;
  fresh.key = key;
  if (const cudaError_t e = make_lt_plan(lt, p, fresh); e != cudaSuccess) {
    destroy(fresh);
    return e;
  }
  plans.push_back(fresh);
  *out = &plans.back();
  return cudaSuccess;
}

// The winner's recorded algorithm when the call is the problem it was picked
// for: that exact M, N, K, row-major NN, packed, one batch (how aegis_tune
// times cublaslt_best). Anything else asks the heuristic.
const detail::LtAlgoConfig* stored_for(const detail::Winner* w, const gemm::GemmParams& p) {
  const bool tuned_form = p.layout_a == gemm::Layout::RowMajor && p.layout_b == gemm::Layout::RowMajor &&
                          p.lda == p.K && p.ldb == p.N && p.ldc == p.N && p.batch == 1;
  return w && w->lt.id >= 0 && tuned_form && w->M == p.M && w->N == p.N && w->K == p.K ? &w->lt : nullptr;
}

cudaError_t cublaslt_gemm(const gemm::GemmParams& p, cudaStream_t stream, const detail::LtAlgoConfig* stored) {
  cublasLtHandle_t lt = nullptr;
  const LtPlan* plan = nullptr;
  if (const cudaError_t e = lt_plan(p, stream, stored, &lt, &plan); e != cudaSuccess) return e;
  if (plan->use_cublas) return cublas_gemm_general(p, stream);
  const cublasStatus_t s = cublasLtMatmul(lt, plan->op, &p.alpha, p.B, plan->a, p.A, plan->b, &p.beta, p.C, plan->c,
                                          p.C, plan->c, &plan->algo, plan->workspace, plan->workspace_bytes, stream);
  if (s == CUBLAS_STATUS_NOT_SUPPORTED) {
    // Refused before launch (nothing written): the plan runs cuBLAS from now on. A
    // safety net; aligned_for and the heuristic's alignment preferences should prevent it.
    plan->use_cublas = true;
    std::fprintf(stderr, "aegis::matmul: %dx%dx%d: cublasLtMatmul refused the plan's algorithm (%s); using cuBLAS\n",
                 p.M, p.N, p.K, plan->source.c_str());
    return cublas_gemm_general(p, stream);
  }
  return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cublaslt_fail("cublasLtMatmul", s);
}

// alpha == 0: C = beta * C (beta == 0: C = 0 without reading C), as in BLAS.
// cuBLAS/cuBLASLt do not apply this consistently (some algorithms propagate
// NaN from A*B), so aegis::matmul handles it before any GEMM is chosen.
__global__ void scale_c_kernel(__half* C, int M, int N, long long ldc, long long stride_c, float beta) {
  __half* c = C + blockIdx.z * stride_c;
  const long long total = (long long)M * N;
  for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
       t += (long long)gridDim.x * blockDim.x) {
    __half* q = c + (t / N) * ldc + t % N;
    *q = __float2half(beta == 0.f ? 0.f : beta * __half2float(*q));
  }
}

cudaError_t scale_c(const gemm::GemmParams& p, cudaStream_t stream) {
  const long long total = (long long)p.M * p.N;
  const int blocks = int(std::min<long long>((total + 255) / 256, 1024));
  scale_c_kernel<<<dim3(blocks, 1, p.batch), 256, 0, stream>>>(p.C, p.M, p.N, p.ldc, p.stride_c, p.beta);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? e : fail(e, "aegis::matmul: scale C (alpha == 0)");
}

}  // namespace

cudaError_t matmul(const gemm::GemmParams& p, cudaStream_t stream) {
  if (const cudaError_t e = gemm::validate(p); e != cudaSuccess)
    return fail(e, "aegis::matmul: invalid GEMM parameters");
  if (gemm::is_empty(p)) return cudaSuccess;
  hw::current_profile();  // cached; the first call on an unmeasured device prints the warning
  if (p.alpha == 0.f) return scale_c(p, stream);  // BLAS: A*B does not contribute
  const detail::Choice c = choice_for(p.M, p.N, p.K);
  using Kind = detail::Choice::Kind;
  if (c.kind == Kind::Cublaslt) return cublaslt_gemm(p, stream, stored_for(c.winner, p));  // floor or filtered
  if (c.kind == Kind::Aegis && c.kernel->run) {
    const cudaError_t err = c.kernel->run(p, stream);
    return err == cudaSuccess ? err : fail(err, std::string("launch ") + c.kernel->name);
  }
  return cublas_gemm_general(p, stream);
}

cudaError_t matmul(const __half* A, const __half* B, __half* C, int M, int N, int K,
                   cudaStream_t stream) {
  return matmul(gemm::row_major_params(A, B, C, M, N, K), stream);
}

const char* matmul_last_error() { return last_error.c_str(); }

const char* matmul_choice(int M, int N, int K) {
  const detail::Choice c = choice_for(M, N, K);
  using Kind = detail::Choice::Kind;
  if (c.kind == Kind::Cublaslt) return "cublaslt";
  return c.kind == Kind::Aegis && c.kernel->run ? c.kernel->name : "cublas";
}

}  // namespace aegis

namespace aegis::detail {

namespace {
// gemm::find for a winner row, once per row (the tables are static; find scans the registry).
const gemm::KernelEntry* find_cached(const Winner& w) {
  static std::mutex mu;
  static std::unordered_map<const Winner*, const gemm::KernelEntry*> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = cache.find(&w);
  if (it != cache.end()) return it->second;
  return cache[&w] = gemm::find(w.impl);
}
}  // namespace

Choice choose(std::span<const Winner> table, int M, int N, int K, const gemm::DeviceCaps& caps, std::string* why) {
  Choice c;
  const Winner* best = nullptr;     // nearest usable row
  const Winner* hint = nullptr;     // nearest row of (N, K), usable or not
  const gemm::KernelEntry* best_kernel = nullptr;
  double best_dist = INFINITY, hint_dist = INFINITY;
  auto nearer = [M](const Winner& w, const Winner* cur, double cur_dist, double* dist) {
    *dist = w.M == M ? -1.0 : std::abs(std::log2(double(w.M)) - std::log2(double(M)));
    return !cur || *dist < cur_dist || (*dist == cur_dist && w.M > cur->M);
  };
  for (const Winner& w : table) {
    if (w.N != N || w.K != K) continue;
    double d;
    if (nearer(w, hint, hint_dist, &d)) hint = &w, hint_dist = d;
    const gemm::KernelEntry* e = nullptr;
    if (w.dispatch != Dispatch::Cublaslt) {
      e = find_cached(w);
      std::string reason;
      if (!e || !gemm::supported(*e, caps, why ? &reason : nullptr)) {
        ++c.filtered;
        if (why) *why += std::string(why->empty() ? "" : "; ") + w.impl + ": " + (e ? reason : "not in this build");
        continue;
      }
    }
    if (nearer(w, best, best_dist, &d)) best = &w, best_dist = d, best_kernel = e;
  }
  if (best) {
    c.winner = best;
    c.kind = best->dispatch == Dispatch::Cublaslt ? Choice::Kind::Cublaslt : Choice::Kind::Aegis;
    c.kernel = best_kernel;
  } else if (hint) {
    c.winner = hint;  // every row filtered out: cuBLASLt, the row's tuned algorithm as the hint
    c.kind = Choice::Kind::Cublaslt;
  }
  return c;
}

std::span<const Winner> winners_table() { return winners_for(current_arch()); }

const Winner* winner_for(int M, int N, int K) { return choice_for(M, N, K).winner; }

std::string cublaslt_plan_source(const gemm::GemmParams& p, cudaStream_t stream, const LtAlgoConfig* stored,
                                 LtAlgoConfig* config) {
  cublasLtHandle_t lt = nullptr;
  const LtPlan* plan = nullptr;
  if (lt_plan(p, stream, stored, &lt, &plan) != cudaSuccess) return std::string();
  if (config) *config = plan->config;
  return plan->source;
}

bool lt_reduction_ok(const LtAlgoConfig& c) { return lt::reduction_ok(c); }

Choice::Kind matmul_host_path(const gemm::GemmParams& p, cudaStream_t stream) {
  if (gemm::validate(p) != cudaSuccess || gemm::is_empty(p)) return Choice::Kind::Cublas;
  hw::current_profile();
  const Choice c = choice_for(p.M, p.N, p.K);
  if (c.kind == Choice::Kind::Cublaslt) {
    cublasLtHandle_t lt = nullptr;
    const LtPlan* plan = nullptr;
    lt_plan(p, stream, stored_for(c.winner, p), &lt, &plan);
  }
  return c.kind;
}

cudaError_t matmul_forced(const gemm::GemmParams& p, Choice::Kind kind, const LtAlgoConfig* stored, cudaStream_t stream) {
  if (const cudaError_t e = gemm::validate(p); e != cudaSuccess)
    return fail(e, "aegis::matmul: invalid GEMM parameters");
  if (gemm::is_empty(p)) return cudaSuccess;
  if (p.alpha == 0.f) return scale_c(p, stream);
  switch (kind) {
    case Choice::Kind::Cublaslt: return cublaslt_gemm(p, stream, stored);
    case Choice::Kind::Cublas: return cublas_gemm_general(p, stream);
    default: return matmul(p, stream);
  }
}

}  // namespace aegis::detail
