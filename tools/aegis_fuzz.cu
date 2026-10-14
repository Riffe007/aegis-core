// aegis_fuzz: randomized GEMM correctness against a reference (Production bar).
//
//   aegis_fuzz [--cases 2000] [--seed 1] [--case I] [--impls naive,regblock_,mma_,mma_pipe_,matmul]
//              [--max-dim 4096] [--calibrate] [--verbose] [--corrupt]
//
// Case i draws everything from mt19937_64(seed * 1000003 + i), so a failure
// is reproduced alone with --seed S --case I. Per case:
//   impl     a registered impl with a general entry (run) whose name starts
//            with one of --impls, or "matmul" (aegis::matmul dispatch)
//   shape    each of M, N, K in 1..max-dim: 30% a prime, 25% 1..64, 20% a
//            multiple of 8, else log-uniform; M*N*K*batch capped at 2^32
//            (naive: 2^27)
//   layouts  A, B each row- or column-major
//   ld       minimal, or padded by 1..17 elements (50%)
//   pointers offset by 1..7 elements (20% per operand): misaligned paths
//   alpha    1, 0, -1, 0.5, -2.25 or uniform(-3, 3); beta 0, 1, -1, 0.5 or
//            uniform(-2, 2); beta == 0 fills C with NaN (C must not be read)
//   batch    2..4 with padded strides (15%)
//   data     uniform(-s, s), s in {1, 0.25, 4}; 5% of cases inject 1-3 of
//            NaN / +Inf / -Inf into A or B
// Reference: FP64 CPU when M*N*K*batch <= 2^24, else cuBLASLt (2x tolerance);
// see src/bench/reference.h. Exit 1 on any failure.
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
#include "bench/reference.h"

namespace {

using aegis::bench::HostGemm;
using aegis::gemm::GemmParams;
using aegis::gemm::KernelEntry;
using aegis::gemm::Layout;

struct Args {
  long long cases = 2000;
  unsigned long long seed = 1;
  long long only_case = -1;
  std::vector<std::string> impls = {"naive", "regblock_", "mma_", "mma_pipe_", "matmul"};
  int max_dim = 4096;
  bool calibrate = false, verbose = false, corrupt = false;
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
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
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

// A target: a registry entry with run(), or aegis::matmul (entry == nullptr).
struct Target {
  std::string name;
  const KernelEntry* entry;
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

Case make_case(std::mt19937_64& rng, const std::vector<Target>& targets, int max_dim,
               const std::vector<int>& primes, bool calibrate) {
  std::uniform_real_distribution<double> u(0, 1);
  Case c;
  c.target = targets[std::uniform_int_distribution<size_t>(0, targets.size() - 1)(rng)];
  GemmParams& p = c.g.p;
  p.batch = u(rng) < 0.15 ? std::uniform_int_distribution<int>(2, 4)(rng) : 1;
  const double cap = c.target.name == "naive" ? std::ldexp(1.0, 27) : std::ldexp(1.0, 32);
  for (;;) {
    p.M = draw_dim(rng, max_dim, primes);
    p.N = draw_dim(rng, max_dim, primes);
    p.K = draw_dim(rng, max_dim, primes);
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

template <class T>
T* device_copy(const std::vector<T>& h, int offset, T** base) {
  AEGIS_CUDA_CHECK(cudaMalloc(base, (h.size() + 16) * sizeof(T)));
  T* p = *base + offset;
  if (!h.empty()) AEGIS_CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
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
  if (std::find(a.impls.begin(), a.impls.end(), "matmul") != a.impls.end()) targets.push_back({"matmul", nullptr});
  if (targets.empty()) { std::fprintf(stderr, "no targets\n"); return 2; }
  const std::vector<int> primes = primes_upto(a.max_dim);
  aegis::bench::LtReference lt;
  cudaStream_t stream;
  AEGIS_CUDA_CHECK(cudaStreamCreate(&stream));

  std::fprintf(stderr, "aegis_fuzz: seed %llu, %lld cases, %zu targets, c = %.2f%s\n", a.seed,
               a.only_case >= 0 ? 1LL : a.cases,
               targets.size(), aegis::bench::kToleranceC, a.calibrate ? ", calibrating" : "");
  const auto t0 = std::chrono::steady_clock::now();
  long long failures = 0, cpu_ref = 0, lt_ref = 0, injected = 0, misaligned = 0;
  std::map<std::string, double> worst_c;  // calibrate: max err / (c=1 tolerance) per impl
  const long long first = a.only_case >= 0 ? a.only_case : 0;
  const long long last = a.only_case >= 0 ? a.only_case + 1 : a.cases;
  for (long long ci = first; ci < last; ++ci) {
    std::mt19937_64 rng(a.seed * 1000003ull + (unsigned long long)ci);
    Case c = make_case(rng, targets, a.max_dim, primes, a.calibrate);
    GemmParams& hp = c.g.p;
    injected += c.injected;
    misaligned += (c.off_a || c.off_b || c.off_c) ? 1 : 0;

    __half *ba, *bb, *bc;
    GemmParams dp = hp;
    dp.A = device_copy(c.g.A, c.off_a, &ba);
    dp.B = device_copy(c.g.B, c.off_b, &bb);
    dp.C = device_copy(c.g.C0, c.off_c, &bc);
    const cudaError_t e = c.target.entry ? c.target.entry->run(dp, stream) : aegis::matmul(dp, stream);
    const cudaError_t s = e == cudaSuccess ? cudaStreamSynchronize(stream) : e;
    std::vector<__half> out(c.g.C0.size());
    if (s == cudaSuccess && !out.empty())
      AEGIS_CUDA_CHECK(cudaMemcpy(out.data(), dp.C, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    const double macs = double(hp.M) * hp.N * hp.K * hp.batch;
    // alpha == 0 (C = beta*C) is always checked on the CPU: cuBLASLt applies the
    // BLAS convention inconsistently there (some algorithms propagate NaN from A*B).
    const bool use_cpu = macs <= std::ldexp(1.0, 24) || hp.alpha == 0.f;
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
                  s != cudaSuccess ? cudaGetErrorString(s) : cmp.first_failure.c_str(),
                  use_cpu ? "" : " [cuBLASLt reference]");
    } else if (a.verbose) {
      std::printf("ok   case %lld: %s  max_ratio %.3f%s\n", ci, c.describe().c_str(), cmp.max_ratio,
                  use_cpu ? "" : " [lt]");
    }
    cudaFree(ba);
    cudaFree(bb);
    cudaFree(bc);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("aegis_fuzz: %lld cases, %lld %s (%lld CPU-FP64 reference, %lld cuBLASLt; %lld with NaN/Inf, "
              "%lld misaligned), %.1f s, seed %llu\n",
              last - first, failures, a.corrupt ? "corruptions NOT caught" : "failures", cpu_ref, lt_ref, injected,
              misaligned, secs, a.seed);
  if (a.calibrate) {
    std::printf("observed c (max err / (sqrt(K)*eps*max|A|*max|B|*|alpha| + eps*|beta|*max|C0|)), CPU-reference cases:\n");
    for (const auto& [who, w] : worst_c) std::printf("  %-12s %.4f\n", who.c_str(), w);
  }
  return failures ? 1 : 0;
}
