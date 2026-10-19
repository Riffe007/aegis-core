// CPU thread pool (src/cpu/thread_pool.h), ctest thread_pool.
//   1. Sizing from discovery, on recorded sysfs (tests/fixtures/hw/):
//      - spark-gb10: performance = the 10 Cortex-X925 (5-9, 15-19); efficiency =
//        the 10 Cortex-A725 (0-4, 10-14);
//      - orin-nano-synthetic (one core class): efficiency falls back to it;
//      - SMT siblings count once (a hand-built CpuInfo with physical_cpus).
//   2. On this machine: one worker per physical performance core, every worker
//      pinned, and every task / parallel_for index runs on its worker's core.
//      An efficiency pool runs on the efficiency cores.
//   3. parallel_for: each index exactly once, repeated; submit: a full ring
//      refuses (false), every accepted task runs, wait_idle waits for all.
//   4. No allocation on the submit path: global operator new is counted, and
//      submit / parallel_for / wait_idle after construction allocate nothing.
#include <sched.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <set>
#include <string>
#include <vector>

#include "aegis/hw.h"
#include "cpu/thread_pool.h"

namespace {
std::atomic<long> g_allocs{0};
}  // namespace

void* operator new(size_t n) {
  g_allocs.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

std::string list(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
  return s;
}

aegis::hw::CpuInfo fixture(const char* name) {
  aegis::hw::DiscoverOptions o;
  o.root = std::string(AEGIS_SOURCE_DIR) + "/tests/fixtures/hw/" + name;
  o.cuda = o.nvml = o.video = o.auxv = false;
  return aegis::hw::discover(o).cpu;
}

using aegis::cpu::CoreClass;

void sizing() {
  const auto spark = fixture("spark-gb10");
  bool fb = true;
  const auto perf = aegis::cpu::select_cpus(spark, CoreClass::Performance, &fb);
  expect(list(perf) == "5,6,7,8,9,15,16,17,18,19" && !fb, "spark-gb10 performance: " + list(perf) + " (the X925s)");
  const auto eff = aegis::cpu::select_cpus(spark, CoreClass::Efficiency, &fb);
  expect(list(eff) == "0,1,2,3,4,10,11,12,13,14" && !fb, "spark-gb10 efficiency: " + list(eff) + " (the A725s)");

  const auto orin = fixture("orin-nano-synthetic");
  const auto operf = aegis::cpu::select_cpus(orin, CoreClass::Performance, &fb);
  const auto oeff = aegis::cpu::select_cpus(orin, CoreClass::Efficiency, &fb);
  expect(!operf.empty() && oeff == operf && fb,
         "orin-nano-synthetic: one class (" + list(operf) + "); efficiency falls back to it");

  aegis::hw::CpuInfo smt;  // 2 cores x 2 threads, performance; 1 efficiency core
  aegis::hw::CoreType p;
  p.performance = true;
  p.cpus = {0, 1, 2, 3};
  p.physical_cpus = {0, 2};
  aegis::hw::CoreType e;
  e.cpus = {4};
  e.physical_cpus = {4};
  smt.core_types = {p, e};
  expect(list(aegis::cpu::select_cpus(smt, CoreClass::Performance)) == "0,2", "SMT siblings count once: 0,2");

  aegis::cpu::PoolOptions opt;
  opt.pin = false;
  aegis::cpu::ThreadPool pool(spark, opt);
  expect(pool.size() == 10 && pool.pinned() == 0, "pool from the spark fixture: 10 workers, unpinned on request");
}

void live() {
  aegis::hw::DiscoverOptions o;
  o.cuda = o.nvml = o.video = false;
  const auto info = aegis::hw::discover(o).cpu;
  const auto want = aegis::cpu::select_cpus(info, CoreClass::Performance);
  aegis::cpu::ThreadPool pool;
  expect(pool.size() == int(want.size()) && pool.size() > 0 && pool.pinned() == pool.size(),
         "this machine: " + std::to_string(pool.size()) + " workers on the physical performance cores " +
             list(std::vector<int>(pool.cpus().begin(), pool.cpus().end())) + ", " + std::to_string(pool.pinned()) +
             " pinned");
  // Where each index ran: (worker, cpu). The caller (-1) runs some indices too.
  constexpr int kN = 4096;
  std::vector<int> worker(kN), cpu(kN);
  for (int rep = 0; rep < 20; ++rep)
    pool.parallel_for(kN, [&](int i) {
      worker[size_t(i)] = pool.worker_index();
      cpu[size_t(i)] = sched_getcpu();
      volatile double x = 0;
      for (int k = 0; k < 2000; ++k) x = x + k;  // enough work that workers take part
    });
  int off_core = 0, by_workers = 0;
  std::set<int> workers_seen;
  for (int i = 0; i < kN; ++i)
    if (worker[size_t(i)] >= 0) {
      ++by_workers;
      workers_seen.insert(worker[size_t(i)]);
      off_core += cpu[size_t(i)] != pool.cpus()[size_t(worker[size_t(i)])];
    }
  expect(off_core == 0 && by_workers > 0, "every index a worker ran ran on that worker's core (" +
                                              std::to_string(by_workers) + " of " + std::to_string(kN) + " by " +
                                              std::to_string(workers_seen.size()) + " workers, last rep)");

  aegis::cpu::PoolOptions eo;
  eo.cores = CoreClass::Efficiency;
  aegis::cpu::ThreadPool bg(eo);
  const auto eff = aegis::cpu::select_cpus(info, CoreClass::Efficiency);
  std::atomic<int> wrong{0}, ran{0};
  struct Ctx {
    aegis::cpu::ThreadPool* pool;
    std::atomic<int>* wrong;
    std::atomic<int>* ran;
  } ctx{&bg, &wrong, &ran};
  for (int i = 0; i < 200; ++i)
    while (!bg.submit(
        [](void* c) {
          auto* x = static_cast<Ctx*>(c);
          const int w = x->pool->worker_index();
          if (w < 0 || sched_getcpu() != x->pool->cpus()[size_t(w)]) x->wrong->fetch_add(1);
          x->ran->fetch_add(1);
        },
        &ctx)) {
    }
  bg.wait_idle();
  expect(bg.size() == int(eff.size()) && ran == 200 && wrong == 0,
         "efficiency pool: " + std::to_string(bg.size()) + " workers on " +
             list(std::vector<int>(bg.cpus().begin(), bg.cpus().end())) + "; 200 tasks, each on its worker's core" +
             (bg.efficiency_fallback() ? " (fallback: one core class)" : ""));
}

void semantics() {
  aegis::cpu::PoolOptions opt;
  opt.queue_capacity = 8;
  opt.max_threads = 4;
  aegis::cpu::ThreadPool pool(opt);
  expect(pool.size() <= 4 && pool.size() > 0, "max_threads caps the pool (" + std::to_string(pool.size()) + ")");

  // parallel_for: every index once, many times over; n = 0 and n = 1.
  std::vector<std::atomic<int>> hits(10007);
  for (int rep = 0; rep < 200; ++rep) pool.parallel_for(10007, [&](int i) { hits[size_t(i)].fetch_add(1); });
  int bad = 0;
  for (auto& h : hits) bad += h.load() != 200;
  int one = 0;
  pool.parallel_for(0, [&](int) { ++one; });
  pool.parallel_for(1, [&](int) { ++one; });
  expect(bad == 0 && one == 1, "parallel_for: 200 x 10007 indices, each exactly once per call; n = 0 runs nothing");

  // submit: a blocked ring refuses, then everything accepted runs.
  struct Gate {
    std::atomic<bool> open{false};
    std::atomic<int> done{0};
  } gate;
  auto blocker = [](void* g) {
    auto* x = static_cast<Gate*>(g);
    while (!x->open.load()) std::this_thread::yield();
    x->done.fetch_add(1);
  };
  int accepted = 0;
  bool refused = false;
  for (int i = 0; i < 100; ++i) {
    if (!pool.submit(blocker, &gate)) {
      refused = true;
      break;
    }
    ++accepted;
  }
  gate.open = true;
  pool.wait_idle();
  expect(refused && accepted <= pool.size() + 8 && gate.done == accepted,
         "submit: the 8-slot ring refused at " + std::to_string(accepted) + " outstanding tasks (" +
             std::to_string(pool.size()) + " running + 8 queued at most); all accepted ran");

  // No allocation on the submit path.
  std::atomic<int> sum{0};
  auto add = [](void* s) { static_cast<std::atomic<int>*>(s)->fetch_add(1); };
  const long before = g_allocs.load();
  for (int rep = 0; rep < 1000; ++rep) {
    while (!pool.submit(add, &sum)) {
    }
    pool.parallel_for(64, [&](int) { sum.fetch_add(1); });
  }
  pool.wait_idle();
  const long allocs = g_allocs.load() - before;
  expect(allocs == 0 && sum == 1000 * 65,
         "no allocation: 1000 submit + 1000 parallel_for + wait_idle allocated " + std::to_string(allocs) + " times");
}

}  // namespace

int main() {
  sizing();
  live();
  semantics();
  std::printf("%s: %d failures\n", failures ? "thread_pool FAILED" : "thread_pool ok", failures);
  return failures ? 1 : 0;
}
