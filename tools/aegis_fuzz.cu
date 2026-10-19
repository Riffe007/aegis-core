// aegis_fuzz: randomized GEMM correctness against a reference (Production bar).
//
//   aegis_fuzz [--cases 2000] [--seed 1] [--case I] [--impls naive,regblock_,mma_,mma_pipe_,matmul]
//              [--max-dim 4096] [--calibrate] [--verbose] [--corrupt] [--poison] [--cpu-ref-only]
//              [--explain] [--decode] [--large-k]
//
// Case i draws everything from mt19937_64(seed * 1000003 + i), so a failure
// is reproduced alone with --seed S --case I. Per case:
//   impl     a registered impl with a general entry (run) whose name starts
//            with one of --impls, or "matmul" (aegis::matmul dispatch), or
//            "matmul_lt" / "matmul_cublas": aegis::matmul's argument handling
//            with the GEMM forced onto its cuBLASLt path (dispatch floor /
//            filtered rows; half the cases pass a stored algorithm of a
//            Cublaslt winner as the hint, half ask the heuristic) or its
//            cuBLAS fallback (no winner)
//   shape    each of M, N, K in 1..max-dim: 30% a prime, 25% 1..64, 20% a
//            multiple of 8, else log-uniform; M*N*K*batch capped at 2^32
//            (naive: 2^27)
//            --decode: M in 1..16 (85%) or 17..40, K drawn the same way up to
//            16384 (GEMV and split-K: slices, odd split counts, K tails)
//            --large-k: K in 14000..20000 (40% a prime), M 1..64, N up to
//            max-dim, M*N*K <= 2^30; every case against the FP64 CPU reference
//            (long-K split-K reductions: partials must stay in FP32)
//   layouts  A, B each row- or column-major
//   ld       minimal, or padded by 1..17 elements (50%)
//   pointers offset by 1..7 elements (20% per operand): misaligned paths
//   alpha    1, 0, -1, 0.5, -2.25 or uniform(-3, 3); beta 0, 1, -1, 0.5 or
//            uniform(-2, 2); beta == 0 fills C with NaN (C must not be read)
//   batch    2..4 with padded strides (15%)
//   data     uniform(-s, s), s in {1, 0.25, 4}; 5% of cases inject 1-3 of
//            NaN / +Inf / -Inf into A or B
// Reference: FP64 CPU when M*N*K*batch <= 2^24, else cuBLASLt (2x tolerance),
// and a case that fails against cuBLASLt is decided by FP64 (1x tolerance);
// see src/bench/reference.h. Exit 1 on any failure.
//
// --poison: every element outside the logical matrices (ld padding, batch
// gaps, the tail) is NaN, plus NaN guard regions around each allocation; with
// beta == 0 all of C is NaN anyway. Must pass with 0 failures: no kernel may
// read past K, M or N into a product, and beta == 0 must not read C.
//
// --cpu-ref-only: no cuBLAS / cuBLASLt kernel ever runs: every case is checked
// against the FP64 CPU reference whatever its size, and aegis::matmul cases
// (whose fallback is cuBLAS) are skipped. The case stream is the same as in
// the default mode: case I with seed S is the identical problem in both. For
// compute-sanitizer synccheck (tools/sanitize.sh) and for checking a
// cuBLASLt-reference failure against FP64.
//
// --explain (with --case I): FP64 CPU reference, our result and cuBLASLt's,
// each element's error in FP16 ulps; prints the worst elements of each and the
// tolerance terms.
//
// --corrupt: self-test of the checker. Each result gets one element pushed
// 4x past the allowed error (or a NaN expectation replaced by 0); the run
// passes only if every case is then reported as a failure.
//
// --calibrate: also run cuBLASLt against the CPU reference on the CPU-sized
// cases and print, per impl, the largest err / (sqrt(K)*eps*max|A|*max|B|*|alpha|
// + eps*|beta|*max|C0|), i.e. the observed c (no NaN/Inf injection).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "aegis/common.cuh"
#include "aegis/gemm.h"
#include "aegis/matmul.h"
#include "bench/build_info.h"
#include "bench/gpu_budget.h"
#include "bench/reference.h"

namespace {

using aegis::bench::HostGemm;
using aegis::gemm::GemmParams;
using aegis::gemm::KernelEntry;
using aegis::gemm::Layout;

constexpr int kDecodeMaxK = 16384;
constexpr int kLargeKMin = 14000, kLargeKMax = 20000;

struct Args {
  long long cases = 2000;
  unsigned long long seed = 1;
  long long only_case = -1;
  std::vector<std::string> impls = {"naive", "regblock_", "mma_", "mma_pipe_", "matmul"};
  int max_dim = 4096;
  bool calibrate = false, verbose = false, corrupt = false, poison = false, cpu_ref_only = false, explain = false;
  bool decode = false;  // --decode: M 1..16 (85%) or 17..40, K up to kDecodeMaxK
  bool large_k = false;  // --large-k: K in kLargeKMin..kLargeKMax, FP64 reference
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    const size_t j = s.find(',', i);
    const std::string t = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!t.empty()) out.push_back(t);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (k == "--cases") a.cases = std::stoll(next());
    else if (k == "--seed") a.seed = std::stoull(next());
    else if (k == "--case") a.only_case = std::stoll(next());
    else if (k == "--impls") a.impls = split(next());
    else if (k == "--max-dim") a.max_dim = std::stoi(next());
    else if (k == "--calibrate") a.calibrate = true;
    else if (k == "--verbose") a.verbose = true;
    else if (k == "--corrupt") a.corrupt = true;
    else if (k == "--poison") a.poison = true;
    else if (k == "--cpu-ref-only") a.cpu_ref_only = true;
    else if (k == "--explain") a.explain = true;
    else if (k == "--decode") a.decode = true;
    else if (k == "--large-k") a.large_k = true;
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.explain && (a.cpu_ref_only || a.only_case < 0)) {
    std::fprintf(stderr, "--explain needs --case I and runs cuBLASLt (not with --cpu-ref-only)\n");
    std::exit(2);
  }
  return a;
}

std::vector<int> primes_upto(int n) {
  std::vector<bool> comp(size_t(n) + 1, false);
  std::vector<int> p;
  for (int i = 2; i <= n; ++i) {
    if (comp[size_t(i)]) continue;
    p.push_back(i);
    for (long long j = (long long)i * i; j <= n; j += i) comp[size_t(j)] = true;
  }
  return p;
}

// A target: a registry entry with run(), or aegis::matmul (entry == nullptr),
// possibly forced onto one of its library paths.
struct Target {
  std::string name;
  const KernelEntry* entry;
  aegis::detail::Choice::Kind forced = aegis::detail::Choice::Kind::Aegis;  // Aegis: plain matmul
  bool library() const { return !entry; }  // aegis::matmul in some form (cuBLAS-backed paths)
};

int draw_dim(std::mt19937_64& rng, int max_dim, const std::vector<int>& primes) {
  std::uniform_real_distribution<double> u(0, 1);
  const double r = u(rng);
  if (r < 0.30) {
    const auto hi = std::upper_bound(primes.begin(), primes.end(), max_dim);
    return primes[std::uniform_int_distribution<size_t>(0, size_t(hi - primes.begin()) - 1)(rng)];
  }
  if (r < 0.55) return std::uniform_int_distribution<int>(1, std::min(64, max_dim))(rng);
  const int d = int(std::exp(u(rng) * std::log(double(max_dim))));  // log-uniform 1..max
  if (r < 0.75) return std::max(8, d / 8 * 8);
  return std::max(1, std::min(max_dim, d));
}

struct Case {
  Target target;
  HostGemm g;
  int off_a = 0, off_b = 0, off_c = 0;  // element offsets of the device pointers
  bool injected = false;
  std::string describe() const {
    const GemmParams& p = g.p;
    char buf[400];
    std::snprintf(buf, sizeof buf,
                  "%s %dx%dx%d %c%c lda=%lld ldb=%lld ldc=%lld alpha=%g beta=%g batch=%d strides=%lld/%lld/%lld "
                  "offsets=%d/%d/%d%s",
                  target.name.c_str(), p.M, p.N, p.K, p.layout_a == Layout::RowMajor ? 'R' : 'C',
                  p.layout_b == Layout::RowMajor ? 'R' : 'C', p.lda, p.ldb, p.ldc, p.alpha, p.beta, p.batch,
                  p.stride_a, p.stride_b, p.stride_c, off_a, off_b, off_c, injected ? " +NaN/Inf" : "");
    return buf;
  }
};

// The case depends only on (seed, case, targets, max_dim, calibrate): never on
// the reference mode, so --cpu-ref-only and --explain reproduce a case exactly.
Case make_case(std::mt19937_64& rng, const std::vector<Target>& targets, int max_dim,
               const std::vector<int>& primes, bool calibrate, bool decode, const std::vector<int>& kprimes,
               bool large_k) {
  std::uniform_real_distribution<double> u(0, 1);
  Case c;
  c.target = targets[std::uniform_int_distribution<size_t>(0, targets.size() - 1)(rng)];
  GemmParams& p = c.g.p;
  p.batch = u(rng) < 0.15 ? std::uniform_int_distribution<int>(2, 4)(rng) : 1;
  const double cap = c.target.name == "naive" ? std::ldexp(1.0, 27) : std::ldexp(1.0, 32);
  for (;;) {
    if (large_k) {  // long-K reductions: split-K partial sums over 14000+ products
      if (u(rng) < 0.4) {
        const auto lo = std::lower_bound(kprimes.begin(), kprimes.end(), kLargeKMin);
        p.K = *(lo + std::uniform_int_distribution<long>(0, long(kprimes.end() - lo) - 1)(rng));
      } else {
        p.K = std::uniform_int_distribution<int>(kLargeKMin, kLargeKMax)(rng);
      }
      p.M = u(rng) < 0.6 ? std::uniform_int_distribution<int>(1, 16)(rng) : std::uniform_int_distribution<int>(17, 64)(rng);
      p.N = draw_dim(rng, max_dim, primes);
      if (double(p.M) * p.N * p.K * p.batch <= std::ldexp(1.0, 30)) break;
      continue;
    }
    if (decode) {  // GEMV and split-K paths: small M, long K (odd split slices, K tails)
      p.M = u(rng) < 0.85 ? std::uniform_int_distribution<int>(1, 16)(rng) : std::uniform_int_distribution<int>(17, 40)(rng);
      p.N = draw_dim(rng, max_dim, primes);
      p.K = draw_dim(rng, kDecodeMaxK, kprimes);
    } else {
      p.M = draw_dim(rng, max_dim, primes);
      p.N = draw_dim(rng, max_dim, primes);
      p.K = draw_dim(rng, max_dim, primes);
    }
    if (double(p.M) * p.N * p.K * p.batch <= cap) break;
  }
  p.layout_a = u(rng) < 0.5 ? Layout::RowMajor : Layout::ColMajor;
  p.layout_b = u(rng) < 0.5 ? Layout::RowMajor : Layout::ColMajor;
  auto pad = [&] { return u(rng) < 0.5 ? std::uniform_int_distribution<int>(1, 17)(rng) : 0; };
  p.lda = (p.layout_a == Layout::RowMajor ? p.K : p.M) + pad();
  p.ldb = (p.layout_b == Layout::RowMajor ? p.N : p.K) + pad();
  p.ldc = p.N + pad();
  const double alphas[] = {1, 0, -1, 0.5, -2.25};
  const double betas[] = {0, 1, -1, 0.5};
  const double ra = u(rng), rbeta = u(rng);
  p.alpha = float(ra < 0.8 ? alphas[std::uniform_int_distribution<int>(0, 4)(rng)] : u(rng) * 6 - 3);
  p.beta = float(rbeta < 0.8 ? betas[std::uniform_int_distribution<int>(0, 3)(rng)] : u(rng) * 4 - 2);
  if (p.batch > 1) {
    const long long one_a = p.layout_a == Layout::RowMajor ? (long long)p.M * p.lda : (long long)p.K * p.lda;
    const long long one_b = p.layout_b == Layout::RowMajor ? (long long)p.K * p.ldb : (long long)p.N * p.ldb;
    p.stride_a = one_a + std::uniform_int_distribution<int>(0, 31)(rng);
    p.stride_b = one_b + std::uniform_int_distribution<int>(0, 31)(rng);
    p.stride_c = (long long)p.M * p.ldc + std::uniform_int_distribution<int>(0, 31)(rng);
  }
  auto offset = [&] { return u(rng) < 0.2 ? std::uniform_int_distribution<int>(1, 7)(rng) : 0; };
  c.off_a = offset();
  c.off_b = offset();
  c.off_c = offset();

  const double scales[] = {1, 0.25, 4};
  const double sa = scales[std::uniform_int_distribution<int>(0, 2)(rng)];
  const double sb = scales[std::uniform_int_distribution<int>(0, 2)(rng)];
  auto fill = [&](std::vector<__half>& v, size_t n, double s) {
    v.resize(n);
    std::uniform_real_distribution<float> d(float(-s), float(s));
    for (auto& x : v) x = __float2half(d(rng));
  };
  fill(c.g.A, aegis::bench::span_a(p), sa);
  fill(c.g.B, aegis::bench::span_b(p), sb);
  if (p.beta != 0.f) {
    fill(c.g.C0, aegis::bench::span_c(p), 1.0);
  } else {
    c.g.C0.assign(aegis::bench::span_c(p), __float2half(NAN));  // must not be read
  }
  if (!calibrate && u(rng) < 0.05) {  // NaN / Inf injection at valid positions
    c.injected = true;
    const int n = std::uniform_int_distribution<int>(1, 3)(rng);
    const float specials[] = {NAN, INFINITY, -INFINITY};
    for (int t = 0; t < n; ++t) {
      const float s = specials[std::uniform_int_distribution<int>(0, 2)(rng)];
      const int b = std::uniform_int_distribution<int>(0, p.batch - 1)(rng);
      if (u(rng) < 0.5) {
        const int i = std::uniform_int_distribution<int>(0, p.M - 1)(rng);
        const int k = std::uniform_int_distribution<int>(0, p.K - 1)(rng);
        c.g.A[size_t(b * p.stride_a +
                     (p.layout_a == Layout::RowMajor ? i * p.lda + k : i + k * p.lda))] = __float2half(s);
      } else {
        const int k = std::uniform_int_distribution<int>(0, p.K - 1)(rng);
        const int j = std::uniform_int_distribution<int>(0, p.N - 1)(rng);
        c.g.B[size_t(b * p.stride_b + (p.layout_b == Layout::RowMajor ? k * p.ldb + j : k + j * p.ldb))] =
            __float2half(s);
      }
    }
  }
  return c;
}

// --poison: every element outside the logical matrices (ld padding, gaps
// between batches, the tail after the last row) becomes NaN on the host, and
// the device allocations get NaN guard regions before and after. Any read of
// a non-logical element that reached a product or the output would show up as
// NaN in a checked element (NaN * 0 = NaN).
void poison_outside(std::vector<__half>& v, bool row_major, int rows, int cols, long long ld, long long stride,
                    int batch) {
  std::vector<__half> p(v.size(), __float2half(NAN));
  for (int b = 0; b < batch; ++b)
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const size_t i = size_t(b * stride + (row_major ? r * ld + c : r + c * ld));
        p[i] = v[i];
      }
  v.swap(p);
}

constexpr int kGuard = 64;  // NaN elements before and after each poisoned allocation

template <class T>
T* device_copy(const std::vector<T>& h, int offset, T** base, bool poison) {
  const size_t n = h.size() + 16 + (poison ? 2 * kGuard : 0);
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(base, n * sizeof(T)));
  if (poison) AEGIS_CUDA_CHECK(cudaMemset(*base, 0x7e, n * sizeof(T)));  // 0x7e7e: an FP16 NaN
  T* p = *base + (poison ? kGuard : 0) + offset;
  if (!h.empty()) AEGIS_CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

// FP16 ulp at |x|: 2^(e-10) for |x| in [2^e, 2^(e+1)), 2^-24 for subnormals.
double ulp_fp16(double x) {
  const double ax = std::fabs(x);
  if (ax < std::ldexp(1.0, -14)) return std::ldexp(1.0, -24);
  return std::ldexp(1.0, std::ilogb(ax) - 10);
}

// --explain: our result and cuBLASLt's, both against the FP64 CPU reference, at
// the worst elements, in FP16 ulps of the reference, plus the tolerance terms.
void explain(const Case& c, const GemmParams& dp, const std::vector<__half>& out, const std::vector<double>& ref,
             const aegis::bench::Tolerance& tol, aegis::bench::LtReference& lt, cudaStream_t stream) {
  const GemmParams& p = c.g.p;
  AEGIS_CUDA_CHECK(cudaMemcpy(dp.C, c.g.C0.data(), c.g.C0.size() * sizeof(__half), cudaMemcpyHostToDevice));
  lt.run(dp, stream);
  AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__half> lo(c.g.C0.size());
  AEGIS_CUDA_CHECK(cudaMemcpy(lo.data(), dp.C, lo.size() * sizeof(__half), cudaMemcpyDeviceToHost));
  struct At { double score = -1; int b = 0, i = 0, j = 0; };
  At ours_worst, lt_worst, apart;
  double sum_ours = 0, sum_lt = 0;
  long long n = 0;
  for (int b = 0; b < p.batch; ++b)
    for (int i = 0; i < p.M; ++i)
      for (int j = 0; j < p.N; ++j) {
        const double r = ref[(size_t(b) * p.M + i) * p.N + j];
        if (!std::isfinite(r)) continue;
        const size_t oi = size_t(b * p.stride_c + i * p.ldc + j);
        const double o = __half2float(out[oi]), l = __half2float(lo[oi]);
        const double eo = std::fabs(o - r), el = std::fabs(l - r), d = std::fabs(o - l);
        sum_ours += eo, sum_lt += el, ++n;
        if (eo > ours_worst.score) ours_worst = {eo, b, i, j};
        if (el > lt_worst.score) lt_worst = {el, b, i, j};
        if (d > apart.score) apart = {d, b, i, j};
      }
  const double t1 = aegis::bench::kToleranceC * std::sqrt(double(p.K)) * aegis::bench::kEpsFp16 * tol.max_a *
                    tol.max_b * std::fabs(double(p.alpha));
  const double t2 = aegis::bench::kEpsFp16 * std::fabs(double(p.beta)) * tol.max_c0;
  std::printf("explain: %s\n", c.describe().c_str());
  std::printf("  tolerance (c = %.2f): %.6g = %.6g (c*sqrt(K)*eps*max|A|*max|B|*|alpha|; max|A| %g, max|B| %g)"
              " + %.6g (eps*|beta|*max|C0|; max|C0| %g); vs cuBLASLt 2x = %.6g\n",
              aegis::bench::kToleranceC, tol.tol, t1, tol.max_a, tol.max_b, t2, tol.max_c0, 2 * tol.tol);
  auto show = [&](const char* what, const At& at) {
    const size_t oi = size_t(at.b * p.stride_c + at.i * p.ldc + at.j);
    const double r = ref[(size_t(at.b) * p.M + at.i) * p.N + at.j];
    const double o = __half2float(out[oi]), l = __half2float(lo[oi]), u = ulp_fp16(r);
    const double c0 = p.beta != 0.f ? __half2float(c.g.C0[oi]) : 0.0;
    std::printf("  %-22s batch %d row %d col %d: FP64 %.9g (alpha*AB %.9g, beta*C0 %.9g; ulp %.3g)\n"
                "  %-22s   ours %.9g: err %.6g = %.2f ulp (%.2f x tol);  cuBLASLt %.9g: err %.6g = %.2f ulp (%.2f x tol)\n",
                what, at.b, at.i, at.j, r, r - double(p.beta) * c0, double(p.beta) * c0, u, "", o, std::fabs(o - r),
                std::fabs(o - r) / u, std::fabs(o - r) / tol.tol, l, std::fabs(l - r), std::fabs(l - r) / u,
                std::fabs(l - r) / tol.tol);
  };
  show("worst ours vs FP64:", ours_worst);
  show("worst cuBLASLt vs FP64:", lt_worst);
  show("largest ours-cuBLASLt:", apart);
  std::printf("  mean |err| vs FP64 over %lld elements: ours %.3g, cuBLASLt %.3g\n", n, sum_ours / std::max(1LL, n),
              sum_lt / std::max(1LL, n));
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  std::vector<Target> targets;
  for (const KernelEntry& e : aegis::gemm::registry()) {
    if (!e.run) continue;
    for (const std::string& pre : a.impls)
      if (pre != "matmul" && std::string(e.name).rfind(pre, 0) == 0) { targets.push_back({e.name, &e}); break; }
  }
  // aegis::matmul stays a target in every mode (the case stream must not depend
  // on the mode); --cpu-ref-only skips its cases (its fallback is cuBLAS).
  auto listed = [&](const char* n) { return std::find(a.impls.begin(), a.impls.end(), n) != a.impls.end(); };
  using Kind = aegis::detail::Choice::Kind;
  if (listed("matmul")) targets.push_back({"matmul", nullptr});
  if (listed("matmul_lt")) targets.push_back({"matmul_lt", nullptr, Kind::Cublaslt});
  if (listed("matmul_cublas")) targets.push_back({"matmul_cublas", nullptr, Kind::Cublas});
  // Stored-algorithm hints for matmul_lt: the Cublaslt winners' recorded algorithms.
  std::vector<const aegis::detail::LtAlgoConfig*> stored;
  for (const auto& w : aegis::detail::winners_table())
    if (w.dispatch == aegis::detail::Dispatch::Cublaslt && w.lt.id >= 0) stored.push_back(&w.lt);
  if (targets.empty()) { std::fprintf(stderr, "no targets\n"); return 2; }
  const std::vector<int> primes = primes_upto(a.max_dim);
  const std::vector<int> kprimes =
      a.large_k ? primes_upto(kLargeKMax) : a.decode ? primes_upto(kDecodeMaxK) : std::vector<int>{};
  aegis::bench::LtReference lt;
  cudaStream_t stream;
  AEGIS_CUDA_CHECK(cudaStreamCreate(&stream));

  std::fprintf(stderr, "aegis_fuzz: seed %llu, %lld cases, %zu targets, c = %.2f%s\n", a.seed,
               a.only_case >= 0 ? 1LL : a.cases,
               targets.size(), aegis::bench::kToleranceC, a.calibrate ? ", calibrating" : "");
  const auto t0 = std::chrono::steady_clock::now();
  long long failures = 0, cpu_ref = 0, lt_ref = 0, injected = 0, misaligned = 0, skipped = 0, lt_overruled = 0;
  std::map<std::string, double> worst_c;  // calibrate: max err / (c=1 tolerance) per impl
  const long long first = a.only_case >= 0 ? a.only_case : 0;
  const long long last = a.only_case >= 0 ? a.only_case + 1 : a.cases;
  for (long long ci = first; ci < last; ++ci) {
    std::mt19937_64 rng(a.seed * 1000003ull + (unsigned long long)ci);
    Case c = make_case(rng, targets, a.max_dim, primes, a.calibrate, a.decode, kprimes, a.large_k);
    // matmul_lt: a stored hint on odd cases (drawn after the case, so the stream is unchanged).
    const aegis::detail::LtAlgoConfig* hint =
        c.target.forced == Kind::Cublaslt && !stored.empty() && ci % 2 ? stored[size_t(ci / 2) % stored.size()] : nullptr;
    if (a.cpu_ref_only && c.target.library()) {
      ++skipped;
      continue;
    }
    GemmParams& hp = c.g.p;
    injected += c.injected;
    misaligned += (c.off_a || c.off_b || c.off_c) ? 1 : 0;

    __half *ba, *bb, *bc;
    GemmParams dp = hp;
    if (a.poison) {
      poison_outside(c.g.A, hp.layout_a == Layout::RowMajor, hp.M, hp.K, hp.lda, hp.stride_a, hp.batch);
      poison_outside(c.g.B, hp.layout_b == Layout::RowMajor, hp.K, hp.N, hp.ldb, hp.stride_b, hp.batch);
      if (hp.beta != 0.f) poison_outside(c.g.C0, true, hp.M, hp.N, hp.ldc, hp.stride_c, hp.batch);
    }
    dp.A = device_copy(c.g.A, c.off_a, &ba, a.poison);
    dp.B = device_copy(c.g.B, c.off_b, &bb, a.poison);
    dp.C = device_copy(c.g.C0, c.off_c, &bc, a.poison);
    const cudaError_t e = c.target.entry                    ? c.target.entry->run(dp, stream)
                          : c.target.forced == Kind::Aegis ? aegis::matmul(dp, stream)
                                                           : aegis::detail::matmul_forced(dp, c.target.forced, hint, stream);
    const cudaError_t s = e == cudaSuccess ? cudaStreamSynchronize(stream) : e;
    std::vector<__half> out(c.g.C0.size());
    if (s == cudaSuccess && !out.empty())
      AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), dp.C, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    const double macs = double(hp.M) * hp.N * hp.K * hp.batch;
    // alpha == 0 (C = beta*C) is always checked on the CPU: cuBLASLt applies the
    // BLAS convention inconsistently there (some algorithms propagate NaN from A*B).
    // --cpu-ref-only and --explain: the FP64 CPU reference for every case, any size.
    const bool use_cpu = a.cpu_ref_only || a.explain || a.large_k || macs <= std::ldexp(1.0, 24) || hp.alpha == 0.f;
    std::vector<double> ref;
    std::vector<__half> lt_out;
    if (use_cpu) {
      ++cpu_ref;
      aegis::bench::cpu_reference(c.g, ref);
    } else {
      ++lt_ref;
      AEGIS_CUDA_CHECK(cudaMemcpy(dp.C, c.g.C0.data(), c.g.C0.size() * sizeof(__half), cudaMemcpyHostToDevice));
      lt.run(dp, stream);
      AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
      lt_out.resize(c.g.C0.size());
      AEGIS_CUDA_CHECK(cudaMemcpy(lt_out.data(), dp.C, lt_out.size() * sizeof(__half), cudaMemcpyDeviceToHost));
      ref.resize(size_t(hp.batch) * hp.M * hp.N);
      for (int b = 0; b < hp.batch; ++b)
        for (int i = 0; i < hp.M; ++i)
          for (int j = 0; j < hp.N; ++j)
            ref[(size_t(b) * hp.M + i) * hp.N + j] = __half2float(lt_out[size_t(b * hp.stride_c + i * hp.ldc + j)]);
    }
    const aegis::bench::Tolerance tol = aegis::bench::tolerance(c.g);
    const double allowed = use_cpu ? tol.tol : 2 * tol.tol;
    bool ok = s == cudaSuccess;
    aegis::bench::Comparison cmp;
    if (ok && a.corrupt && !out.empty()) {  // push one element past the bound
      const size_t at = std::uniform_int_distribution<size_t>(0, ref.size() - 1)(rng);
      const int b = int(at / (size_t(hp.M) * hp.N)), i = int(at / hp.N % hp.M), j = int(at % hp.N);
      const size_t oi = size_t(b * hp.stride_c + i * hp.ldc + j);
      const double r = ref[at];
      out[oi] = __float2half(std::isfinite(r) ? float(r + 4 * allowed + 4 * std::fabs(r) * aegis::bench::kEpsFp16 + 1) : 0.f);
    }
    if (ok) {
      cmp = aegis::bench::compare(c.g, out, ref, allowed);
      ok = cmp.pass;
    }
    // cuBLASLt is a proxy for the exact result, and it can itself be off by more
    // than the tolerance (seed 20261014 case 24632: 2.57 FP16 ulp from FP64
    // where ours was 0.43). A cuBLASLt-reference failure is therefore decided by
    // the FP64 CPU reference at 1x tolerance; the tolerance is never widened.
    if (!ok && !use_cpu && s == cudaSuccess && !a.corrupt) {
      std::vector<double> exact;
      aegis::bench::cpu_reference(c.g, exact);
      const aegis::bench::Comparison fp64 = aegis::bench::compare(c.g, out, exact, tol.tol);
      if (fp64.pass) {
        ++lt_overruled;
        std::printf("note case %lld (--seed %llu --case %lld): %s\n  vs cuBLASLt: %s; vs FP64: pass (max %.2f x tol): "
                    "cuBLASLt reference error\n",
                    ci, a.seed, ci, c.describe().c_str(), cmp.first_failure.c_str(), fp64.max_ratio);
        ok = true;
      } else {
        cmp = fp64;  // report the FP64 verdict
      }
    }
    if (a.explain && s == cudaSuccess) explain(c, dp, out, ref, tol, lt, stream);
    if (a.corrupt) ok = !ok;  // the corrupted result must be caught
    if (a.calibrate && s == cudaSuccess && use_cpu) {
      const double unit = aegis::bench::tolerance(c.g, 1.0).tol;  // c = 1
      auto observe = [&](const std::string& who, const std::vector<__half>& o) {
        const aegis::bench::Comparison k = aegis::bench::compare(c.g, o, ref, unit);
        double& w = worst_c[who];
        if (k.max_ratio > w) w = k.max_ratio;
      };
      observe(c.target.name.rfind("mma_pipe_", 0) == 0 ? "mma_pipe_*" :
              c.target.name.rfind("mma_", 0) == 0 ? "mma_*" :
              c.target.name.rfind("regblock_", 0) == 0 ? "regblock_*" : c.target.name, out);
      // cuBLASLt against the same FP64 reference.
      AEGIS_CUDA_CHECK(cudaMemcpy(dp.C, c.g.C0.data(), c.g.C0.size() * sizeof(__half), cudaMemcpyHostToDevice));
      lt.run(dp, stream);
      AEGIS_CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<__half> lo(c.g.C0.size());
      AEGIS_CUDA_CHECK(cudaMemcpy(lo.data(), dp.C, lo.size() * sizeof(__half), cudaMemcpyDeviceToHost));
      observe("cublasLt", lo);
    }
    if (!ok) {
      ++failures;
      std::printf("FAIL case %lld (--seed %llu --case %lld): %s\n  %s%s\n", ci, a.seed, ci, c.describe().c_str(),
                  s != cudaSuccess ? (std::string(cudaGetErrorString(s)) + (c.target.library() ? std::string(": ") + aegis::matmul_last_error() : "")).c_str()
                                  : cmp.first_failure.c_str(),
                  use_cpu ? "" : (!a.corrupt && s == cudaSuccess ? " [FP64 reference, after cuBLASLt]" : " [cuBLASLt reference]"));
      if (c.target.forced == Kind::Cublaslt) {
        aegis::detail::LtAlgoConfig used;
        const std::string src = aegis::detail::cublaslt_plan_source(dp, stream, hint, &used);
        std::printf("  cuBLASLt plan: %s, algo %d splitk %d reduction %d (hint: %s)\n", src.c_str(), used.id, used.splitk,
                    used.reduction, hint ? ("stored algo " + std::to_string(hint->id)).c_str() : "none");
      }
    } else if (a.verbose) {
      std::printf("ok   case %lld: %s  max_ratio %.3f%s\n", ci, c.describe().c_str(), cmp.max_ratio,
                  use_cpu ? "" : " [lt]");
    }
    aegis::budget::gpu_free(ba);
    aegis::budget::gpu_free(bb);
    aegis::budget::gpu_free(bc);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // The build, for the perf DB (aegis-perf ingest-fuzz); before the summary line, which stays last.
  std::printf("aegis_fuzz build: {%s}\n", aegis::bench::build_json().c_str());
  std::printf("aegis_fuzz: %lld cases, %lld %s (%lld CPU-FP64 reference, %lld cuBLASLt; %lld with NaN/Inf, "
              "%lld misaligned%s), %.1f s, seed %llu\n",
              last - first, failures, a.corrupt ? "corruptions NOT caught" : "failures", cpu_ref, lt_ref, injected,
              misaligned,
              (std::string(skipped ? ", " + std::to_string(skipped) + " aegis::matmul cases skipped" : "") +
               (lt_overruled ? ", " + std::to_string(lt_overruled) + " cuBLASLt-reference failures overruled by FP64" : ""))
                  .c_str(),
              secs, a.seed);
  if (a.calibrate) {
    std::printf("observed c (max err / (sqrt(K)*eps*max|A|*max|B|*|alpha| + eps*|beta|*max|C0|)), CPU-reference cases:\n");
    for (const auto& [who, w] : worst_c) std::printf("  %-12s %.4f\n", who.c_str(), w);
  }
  return failures ? 1 : 0;
}
