// bench_gemm: time one GEMM implementation on one shape and emit one JSON line.
//
//   bench_gemm --impl tiled --m 4096 --n 4096 --k 4096 [--reps 50] [--warmup 5]
//              [--check] [--seed 42]
//   bench_gemm --list                  registered impls as JSON lines
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
  int reps = 50, warmup = 5;
  bool check = false;
  bool list = false;
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
    else if (k == "--seed") a.seed = static_cast<unsigned>(std::atoi(next()));
    else if (k == "--check") a.check = true;
    else if (k == "--list") a.list = true;
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.M <= 0 || a.N <= 0 || a.K <= 0 || a.reps <= 0 || a.warmup < 0) {
    std::fprintf(stderr, "invalid shape or rep count\n");
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
  // Resolved once, before timing, so lookup cost never lands in a timed rep.
  const aegis::gemm::KernelEntry* entry = aegis::gemm::find(a.impl);
  if (!entry && a.impl != "cublas") {
    std::fprintf(stderr, "unknown impl '%s'\n", a.impl.c_str());
    return 2;
  }

  aegis::bench::Harness h;
  aegis::bench::Problem p(a.M, a.N, a.K, a.seed);
  h.run_once(entry, p);
  const aegis::bench::Check c = a.check ? h.check(p) : aegis::bench::Check{};
  const aegis::bench::Timing t = h.time(entry, p, a.warmup, a.reps);
  std::printf("%s\n", h.json_row(a.impl.c_str(), entry, p, c, t).c_str());

  if (c.checked && !c.pass) {
    std::fprintf(stderr, "FAIL: %s %dx%dx%d rel_err=%.3g > %.1e\n",
                 a.impl.c_str(), a.M, a.N, a.K, c.rel_err, aegis::bench::kRelTol);
  }
  return (c.checked && !c.pass) ? 1 : 0;
}
