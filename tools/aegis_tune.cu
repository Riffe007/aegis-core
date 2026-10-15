// aegis_tune: brute-force autotuner, in one process.
//
//   aegis_tune --shapes shapes/default.txt [--seed S] [--prefix regblock_,mma_,mma_pipe_]
//              [--reps 20] [--cublas-reps 50] [--warmup 3] [--data-seed 42] [--protocol 2]
//              [--warm-top 2] [--allow-busy] [--no-paired] [--pair-with IMPL]
//
// For each shape (from a file, "M N K" per line, # comments) it allocates the
// problem once, then runs the baselines (cuBLAS; cublaslt_best: the fastest
// passing of up to 8 cuBLASLt heuristic algorithms) and every registered config whose name starts
// with one of the prefixes, in an order shuffled per shape (seeded; the seed is
// logged) so neither ordering nor thermal drift favours a config. Each config
// gets the bench_gemm protocol (src/bench/harness.h): untimed run, --check
// against cuBLAS, then timing (protocol v2 by default: batched windows; --reps
// is the number of timed windows). Configs are ranked on cache_mode=cold; then
// cuBLAS and the --warm-top fastest configs are timed again with cache_mode=warm
// (recorded for winners, not used to rank). Configs failing the check are
// not timed and not emitted; they are listed on stderr.
//
// Paired tie test (protocol v2): per shape, the fastest cold config is compared
// with the runner-up and with cublaslt_best over 30 alternating windows
// (A, B, A, B...); a bootstrap gives the 95% CI of the median ratio B/A. One
// line per comparison: {"kind":"paired", ..., "ratio_ci_lo", "ratio_ci_hi"}.
//
// Idle guard: before anything else, GPU utilization is sampled for 2 s (NVML);
// above 2% the tuner refuses to start unless --allow-busy is given.
//
// stdout: one JSON row per timed config, bench_gemm's schema plus
//   ,"idle_util_pct":U,"allow_busy":B,"tune":{"seed":S,"shape_seed":s,"order":i}
// (allow_busy is true when the guard found the GPU busy and --allow-busy let it run).
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
  int warm_top = 2;  // warm rows for cuBLAS and the N fastest cold configs per shape
  bool allow_busy = false;
  bool paired = true;  // --no-paired skips the paired tie tests
  std::string pair_with;  // --pair-with IMPL: also pair the fastest config with IMPL
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
    else if (k == "--warm-top") a.warm_top = std::stoi(next());
    else if (k == "--allow-busy") a.allow_busy = true;
    else if (k == "--no-paired") a.paired = false;
    else if (k == "--pair-with") a.pair_with = next();
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
  const aegis::bench::IdleSample idle = aegis::bench::idle_guard(a.allow_busy);
  const bool busy = aegis::bench::idle_busy(idle, aegis::bench::idle_max_pct());
  char util[32] = "null";
  if (idle.ok) std::snprintf(util, sizeof util, "%.1f", idle.mean_pct);
  const std::string idle_json = std::string(",\"idle_util_pct\":") + util + ",\"idle_source\":\"" +
                                (idle.ok ? idle.source : "none") + "\",\"allow_busy\":" + (busy ? "true" : "false");

  // Candidates: the two baselines (cuBLAS, cublaslt_best), then matching configs.
  struct Cand { const aegis::gemm::KernelEntry* e; bool lt; };
  std::vector<Cand> candidates = {{nullptr, false}, {nullptr, true}};
  for (const auto& e : aegis::gemm::registry())
    for (const auto& p : a.prefixes)
      if (std::string(e.name).rfind(p, 0) == 0) { candidates.push_back({&e, false}); break; }

  std::fprintf(stderr, "aegis_tune: seed %llu, %zu shapes x %zu candidates (incl. cuBLAS, cublaslt_best)\n",
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
    std::vector<Cand> order = candidates;
    std::shuffle(order.begin(), order.end(), std::mt19937_64(shape_seed));
    aegis::bench::Problem p(s.M, s.N, s.K, a.data_seed);
    aegis::bench::LtBest lt;  // this shape's cublaslt_best, reused for its warm row

    // Pass 1, cold (ranks the configs). Pass 2, warm: the baselines and the
    // --warm-top fastest cold configs (the winner and its tie candidate).
    struct Done {
      const char* name;
      aegis::bench::Impl impl;
      bool ours;
      aegis::bench::Check c;
      double cold_us;
      size_t order;
    };
    std::vector<Done> done;
    auto options = [&](bool ours, aegis::bench::CacheMode cache) {
      aegis::bench::TimingOptions opt;
      opt.protocol = a.protocol;
      opt.cache = cache;
      opt.warmup = a.warmup;
      opt.reps = ours ? a.reps : a.cublas_reps;
      return opt;
    };
    auto emit = [&](const Done& d, const aegis::bench::Timing& t, aegis::bench::CacheMode cache) {
      const std::string extra = idle_json + (d.impl.plan ? h.lt_json(lt) : "") +
                                ",\"tune\":{\"seed\":" + std::to_string(a.seed) +
                                ",\"shape_seed\":" + std::to_string(shape_seed) +
                                ",\"order\":" + std::to_string(d.order) + "}";
      std::printf("%s\n", h.json_row(d.name, d.impl, p, d.c, t, extra).c_str());
      std::fflush(stdout);
      ++timed;
      std::fprintf(stderr, "  %-28s %5dx%5dx%5d  %10.2f us  N=%-6lld %s\n", d.name, s.M, s.N, s.K,
                   t.median_us, t.batch_n, a.protocol == 1 ? "v1" : aegis::bench::cache_mode_name(cache));
    };
    auto fail = [&](const char* name, double rel_err) {
      char buf[256];
      std::snprintf(buf, sizeof buf, "%s %dx%dx%d rel_err=%.3g", name, s.M, s.N, s.K, rel_err);
      skipped.emplace_back(buf);
      std::fprintf(stderr, "  SKIP %s (fails --check)\n", buf);
    };

    for (size_t oi = 0; oi < order.size(); ++oi) {
      const Cand cand = order[oi];
      if (cand.lt) {  // checks and times each heuristic algorithm, keeps the fastest passing one
        lt = h.cublaslt_best(p, options(false, aegis::bench::CacheMode::Cold));
        if (lt.best < 0) { fail(aegis::bench::kLtBestImpl, -1); continue; }
        Done d{aegis::bench::kLtBestImpl, lt.impl(), false, lt.check, lt.timing.median_us, oi};
        emit(d, lt.timing, aegis::bench::CacheMode::Cold);
        done.push_back(d);
        continue;
      }
      Done d{cand.e ? cand.e->name : "cublas", cand.e, cand.e != nullptr, {}, 0, oi};
      h.run_once(d.impl, p);
      d.c = h.check(p);
      if (!d.c.pass) { fail(d.name, d.c.rel_err); continue; }
      const aegis::bench::Timing t = h.time(d.impl, p, options(d.ours, aegis::bench::CacheMode::Cold));
      d.cold_us = t.median_us;
      emit(d, t, aegis::bench::CacheMode::Cold);
      done.push_back(d);
    }
    // Paired tie tests (v2): the fastest cold config against the runner-up and
    // against cublaslt_best, kPairedWindows alternating windows each. When the
    // runner-up can be dispatched (tie, or faster in the paired windows: CI not
    // entirely above 1.0), it is also paired with cublaslt_best, so the winner's
    // baseline CI is always of the config aegis::matmul runs.
    if (a.protocol == 2 && a.paired) {
      std::vector<const Done*> ours;
      const Done* ltd = nullptr;
      for (const Done& d : done) {
        if (d.ours) ours.push_back(&d);
        else if (d.impl.plan) ltd = &d;
      }
      std::sort(ours.begin(), ours.end(), [](const Done* x, const Done* y) { return x->cold_us < y->cold_us; });
      auto pair_row = [&](const Done& w, const Done& o, const char* role) {
        const aegis::bench::Paired r =
            h.paired(w.impl, o.impl, p, options(true, aegis::bench::CacheMode::Cold),
                     aegis::bench::kPairedWindows, shape_seed);
        std::printf(
            "{\"kind\":\"paired\",\"op\":\"gemm\",\"dtype\":\"f16f32f16\",\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
            "\"a\":\"%s\",\"b\":\"%s\",\"role\":\"%s\",\"pairs\":%d,\"ratio_median\":%.5f,"
            "\"ratio_ci_lo\":%.5f,\"ratio_ci_hi\":%.5f,\"tied\":%s,\"batch_a\":%lld,\"batch_b\":%lld,"
            "\"bootstrap_resamples\":%d,\"bootstrap_seed\":%llu,\"protocol_version\":2,\"cache_mode\":\"cold\"%s}\n",
            s.M, s.N, s.K, w.name, o.name, role, r.pairs, r.median, r.ci_lo, r.ci_hi, r.tied ? "true" : "false",
            r.batch_a, r.batch_b, aegis::bench::kBootstrapResamples, shape_seed, idle_json.c_str());
        std::fflush(stdout);
        std::fprintf(stderr, "  paired %-28s vs %-28s ratio %.4f [%.4f, %.4f] %s\n", w.name, o.name, r.median,
                     r.ci_lo, r.ci_hi, r.tied ? "TIED" : "");
        return r;
      };
      bool runner_up_can_win = false;
      if (ours.size() >= 2) runner_up_can_win = !(pair_row(*ours[0], *ours[1], "runner_up").ci_lo > 1.0);
      if (!ours.empty() && ltd) pair_row(*ours[0], *ltd, "cublaslt_best");
      if (runner_up_can_win && ltd) pair_row(*ours[1], *ltd, "cublaslt_best");
      if (!a.pair_with.empty() && !ours.empty())  // e.g. the previous winner
        for (const Done* d : ours)
          if (d->name == a.pair_with && d != ours[0] && (ours.size() < 2 || d != ours[1]))
            pair_row(*ours[0], *d, "reference");
    }
    if (a.protocol == 2 && a.warm_top > 0) {
      std::vector<Done> warm;
      for (const Done& d : done) if (d.ours) warm.push_back(d);
      std::sort(warm.begin(), warm.end(), [](const Done& x, const Done& y) { return x.cold_us < y.cold_us; });
      warm.resize(std::min(warm.size(), size_t(a.warm_top)));
      for (const Done& d : done) if (!d.ours) warm.insert(warm.begin(), d);  // baselines first
      for (const Done& d : warm)
        emit(d, h.time(d.impl, p, options(d.ours, aegis::bench::CacheMode::Warm)), aegis::bench::CacheMode::Warm);
    }
  }

  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "aegis_tune: %d timed, %zu skipped (failed --check), %.1f s wall, seed %llu\n",
               timed, skipped.size(), secs, a.seed);
  for (const auto& s : skipped) std::fprintf(stderr, "  skipped: %s\n", s.c_str());
  return 0;
}
