// bench_gemm: time one GEMM implementation on one shape and emit one JSON line.
//
//   bench_gemm --impl tiled --m 4096 --n 4096 --k 4096 [--reps 20] [--warmup 3]
//              [--protocol 2] [--cache cold|warm] [--check] [--seed 42]
//   bench_gemm --impl cublaslt_best ...  fastest passing cuBLASLt heuristic algorithm (of 8)
//   bench_gemm --list                  registered impls as JSON lines
//   bench_gemm --idle-check [--allow-busy]   the idle guard alone (aegis-perf sweep
//                                      runs it once before the sweep); exit 3 = busy
//
// --allow-busy --idle-util-pct U (passed by aegis-perf sweep after a busy idle
// check) adds ,"idle_util_pct":U,"allow_busy":true to the row.
//
// The timing protocol and JSON row live in src/bench/harness.{h,cu}, shared
// with aegis_tune. --check exits 1 if max_abs_err / max|ref| exceeds the
// tolerance in CLAUDE.md.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "aegis/gemm.h"
#include "bench/harness.h"

namespace {

struct Args {
  std::string impl = "tiled";
  int M = 1024, N = 1024, K = 1024;
  int reps = -1, warmup = -1;  // default: per protocol (v2 20 windows / 3; v1 50 / 5)
  int protocol = 2;
  std::string cache = "cold";
  bool check = false;
  bool list = false;
  bool idle_check = false, allow_busy = false;
  double idle_util_pct = -1;  // < 0: not recorded
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
    else if (k == "--protocol") a.protocol = std::atoi(next());
    else if (k == "--cache") a.cache = next();
    else if (k == "--seed") a.seed = static_cast<unsigned>(std::atoi(next()));
    else if (k == "--check") a.check = true;
    else if (k == "--list") a.list = true;
    else if (k == "--idle-check") a.idle_check = true;
    else if (k == "--allow-busy") a.allow_busy = true;
    else if (k == "--idle-util-pct") a.idle_util_pct = std::atof(next());
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.reps < 0) a.reps = a.protocol == 1 ? 50 : 20;
  if (a.warmup < 0) a.warmup = a.protocol == 1 ? 5 : 3;
  if (a.M <= 0 || a.N <= 0 || a.K <= 0 || a.reps <= 0 || a.warmup < 0 ||
      (a.protocol != 1 && a.protocol != 2) || (a.cache != "cold" && a.cache != "warm")) {
    std::fprintf(stderr, "invalid shape, rep count, protocol or cache mode\n");
    std::exit(2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);

  // --list: one JSON line per registered impl (cuBLAS is always available too).
  if (a.list) {
    for (const auto& e : aegis::gemm::registry())
      std::printf("{\"impl\":\"%s\",\"config\":%s}\n", e.name, e.config);
    return 0;
  }
  if (a.idle_check) {
    const aegis::bench::IdleSample s = aegis::bench::idle_guard(a.allow_busy);  // exits 3 if busy
    std::printf("{\"ok\":%s,\"mean_pct\":%.2f,\"max_pct\":%.0f,\"samples\":%d}\n",
                s.ok ? "true" : "false", s.mean_pct, s.max_pct, s.samples);
    return 0;
  }
  // Resolved once, before timing, so lookup cost never lands in a timed rep.
  const aegis::gemm::KernelEntry* entry = aegis::gemm::find(a.impl);
  const bool lt_best = a.impl == aegis::bench::kLtBestImpl;
  if (!entry && a.impl != "cublas" && !lt_best) {
    std::fprintf(stderr, "unknown impl '%s'\n", a.impl.c_str());
    return 2;
  }

  aegis::bench::Harness h;
  aegis::bench::Problem p(a.M, a.N, a.K, a.seed);
  aegis::bench::TimingOptions opt;
  opt.protocol = a.protocol;
  opt.cache = a.cache == "warm" ? aegis::bench::CacheMode::Warm : aegis::bench::CacheMode::Cold;
  opt.warmup = a.warmup;
  opt.reps = a.reps;
  std::string extra;
  if (a.idle_util_pct >= 0) {
    char buf[96];
    std::snprintf(buf, sizeof buf, ",\"idle_util_pct\":%.1f,\"allow_busy\":%s", a.idle_util_pct,
                  a.allow_busy ? "true" : "false");
    extra = buf;
  }

  // cublaslt_best: every heuristic algorithm is run, checked and timed; the
  // fastest passing one is the row (always checked).
  if (lt_best) {
    const aegis::bench::LtBest b = h.cublaslt_best(p, opt);
    if (b.best < 0) {
      std::fprintf(stderr, "FAIL: no cuBLASLt algorithm passed (%d candidates) for %dx%dx%d\n",
                   b.candidates, a.M, a.N, a.K);
      return 1;
    }
    std::printf("%s\n",
                h.json_row(a.impl.c_str(), b.impl(), p, b.check, b.timing, extra + h.lt_json(b)).c_str());
    return 0;
  }

  h.run_once(entry, p);
  const aegis::bench::Check c = a.check ? h.check(p) : aegis::bench::Check{};
  const aegis::bench::Timing t = h.time(entry, p, opt);
  std::printf("%s\n", h.json_row(a.impl.c_str(), entry, p, c, t, extra).c_str());

  if (c.checked && !c.pass) {
    std::fprintf(stderr, "FAIL: %s %dx%dx%d rel_err=%.3g > %.1e\n",
                 a.impl.c_str(), a.M, a.N, a.K, c.rel_err, aegis::bench::kRelTol);
  }
  return (c.checked && !c.pass) ? 1 : 0;
}
