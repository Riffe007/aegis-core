// aegis_tune: brute-force autotuner, in one process.
//
//   aegis_tune --shapes shapes/default.txt [--seed S] [--prefix regblock_,mma_,mma_pipe_]
//              [--reps 20] [--cublas-reps 50] [--warmup 3] [--data-seed 42] [--protocol 2]
//
// For each shape (from a file, "M N K" per line, # comments) it allocates the
// problem once, then runs cuBLAS and every registered config whose name starts
// with one of the prefixes, in an order shuffled per shape (seeded; the seed is
// logged) so neither ordering nor thermal drift favours a config. Each config
// gets the bench_gemm protocol (src/bench/harness.h): untimed run, --check
// against cuBLAS, then timing (protocol v2 by default: batched windows; --reps
// is the number of timed windows). Configs failing the check are
// not timed and not emitted; they are listed on stderr.
//
// stdout: one JSON row per timed config, bench_gemm's schema plus
//   ,"tune":{"seed":S,"shape_seed":s,"order":i}
// One CUDA context, one stream, one cuBLAS handle for the whole run.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "aegis/gemm.h"
#include "bench/harness.h"

namespace {

struct Args {
  std::string shapes;
  unsigned long long seed = 0;
  bool seed_given = false;
  std::vector<std::string> prefixes = {"regblock_", "mma_", "mma_pipe_"};
  int reps = 20, cublas_reps = 50, warmup = 3;
  unsigned data_seed = 42;
  int protocol = 2;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty()) out.push_back(item);
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
    if (k == "--shapes") a.shapes = next();
    else if (k == "--seed") { a.seed = std::stoull(next()); a.seed_given = true; }
    else if (k == "--prefix") a.prefixes = split(next(), ',');
    else if (k == "--reps") a.reps = std::stoi(next());
    else if (k == "--cublas-reps") a.cublas_reps = std::stoi(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--data-seed") a.data_seed = static_cast<unsigned>(std::stoul(next()));
    else if (k == "--protocol") a.protocol = std::stoi(next());
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.shapes.empty()) { std::fprintf(stderr, "--shapes FILE is required\n"); std::exit(2); }
  if (a.reps <= 0 || a.cublas_reps <= 0 || a.warmup < 0) { std::fprintf(stderr, "invalid reps\n"); std::exit(2); }
  if (a.protocol != 1 && a.protocol != 2) { std::fprintf(stderr, "--protocol must be 1 or 2\n"); std::exit(2); }
  if (!a.seed_given) a.seed = std::random_device{}();
  return a;
}

struct Shape { int M, N, K; };

std::vector<Shape> read_shapes(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
  std::vector<Shape> v;
  int line_no = 0;
  for (std::string line; std::getline(f, line);) {
    ++line_no;
    line = line.substr(0, line.find('#'));
    std::istringstream ss(line);
    Shape s{};
    if (!(ss >> s.M)) continue;  // blank or comment
    if (!(ss >> s.N >> s.K) || s.M <= 0 || s.N <= 0 || s.K <= 0) {
      std::fprintf(stderr, "%s:%d: expected \"M N K\" with positive ints\n", path.c_str(), line_no);
      std::exit(2);
    }
    v.push_back(s);
  }
  if (v.empty()) { std::fprintf(stderr, "%s: no shapes\n", path.c_str()); std::exit(2); }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  const std::vector<Shape> shapes = read_shapes(a.shapes);

  // Candidates: nullptr = cuBLAS (the reference row), then matching configs.
  std::vector<const aegis::gemm::KernelEntry*> candidates = {nullptr};
  for (const auto& e : aegis::gemm::registry())
    for (const auto& p : a.prefixes)
      if (std::string(e.name).rfind(p, 0) == 0) { candidates.push_back(&e); break; }

  std::fprintf(stderr, "aegis_tune: seed %llu, %zu shapes x %zu candidates (incl. cuBLAS)\n",
               a.seed, shapes.size(), candidates.size());
  const auto t0 = std::chrono::steady_clock::now();
  aegis::bench::Harness h;
  std::fprintf(stderr, "aegis_tune: event timer resolution %.0f ns (min event spacing %.0f ns)\n",
               h.timer().tick_ns, h.timer().min_delta_ns);
  int timed = 0;
  std::vector<std::string> skipped;

  for (size_t si = 0; si < shapes.size(); ++si) {
    const Shape s = shapes[si];
    const unsigned long long shape_seed = a.seed + si;
    std::vector<const aegis::gemm::KernelEntry*> order = candidates;
    std::shuffle(order.begin(), order.end(), std::mt19937_64(shape_seed));
    aegis::bench::Problem p(s.M, s.N, s.K, a.data_seed);

    for (size_t oi = 0; oi < order.size(); ++oi) {
      const aegis::gemm::KernelEntry* e = order[oi];
      const char* name = e ? e->name : "cublas";
      h.run_once(e, p);
      const aegis::bench::Check c = h.check(p);
      if (!c.pass) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s %dx%dx%d rel_err=%.3g", name, s.M, s.N, s.K, c.rel_err);
        skipped.emplace_back(buf);
        std::fprintf(stderr, "  SKIP %s (fails --check)\n", buf);
        continue;
      }
      aegis::bench::TimingOptions opt;
      opt.protocol = a.protocol;
      opt.warmup = a.warmup;
      opt.reps = e ? a.reps : a.cublas_reps;
      const aegis::bench::Timing t = h.time(e, p, opt);
      const std::string extra = ",\"tune\":{\"seed\":" + std::to_string(a.seed) +
                                ",\"shape_seed\":" + std::to_string(shape_seed) +
                                ",\"order\":" + std::to_string(oi) + "}";
      std::printf("%s\n", h.json_row(name, e, p, c, t, extra).c_str());
      std::fflush(stdout);
      ++timed;
      std::fprintf(stderr, "  %-28s %5dx%5dx%5d  %10.2f us  N=%lld\n", name, s.M, s.N, s.K, t.median_us,
                   t.batch_n);
    }
  }

  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "aegis_tune: %d timed, %zu skipped (failed --check), %.1f s wall, seed %llu\n",
               timed, skipped.size(), secs, a.seed);
  for (const auto& s : skipped) std::fprintf(stderr, "  skipped: %s\n", s.c_str());
  return 0;
}
