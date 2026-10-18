#pragma once
// Shared harness for the AEGIS microbenchmarks (one binary per probe).
//
// Output: one JSON line per measurement on stdout,
//   {"probe":..., "param":{...}, "value":..., "unit":..., "device":..., "sm_arch":...,
//    "cuda_runtime":..., "cuda_driver":...}
// ingested into the perf DB `microbench` table (aegis-perf ingest-microbench).
//
// Proof: every probe disassembles its own timed kernel with cuobjdump, prints
// the timed loop (the backward-branch loop with the most instances of the
// measured instruction; ties go to the larger loop) to stderr, and checks that
// the loop body contains exactly the expected number of measured instructions.
// A mismatch means the compiler eliminated, duplicated or restructured the
// measured work, and the binary exits 3 without trusting its numbers.
//
// Timed kernels are extern "C" so cuobjdump -fun can find them by name.
#include <cuda_runtime.h>

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "hw/profile.h"        // the device profile label on every row (aegis-hw step 3)
#include "bench/gpu_budget.h"  // every probe allocates through the GPU budget (CLAUDE.md rule 13)

#define MB_CHECK(expr)                                                         \
  do {                                                                         \
    cudaError_t mb_err_ = (expr);                                              \
    if (mb_err_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error: %s at %s:%d in %s\n",                  \
                   cudaGetErrorString(mb_err_), __FILE__, __LINE__, #expr);    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace mb {

__device__ __forceinline__ long long clock64_volatile() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)::"memory");
  return t;
}

struct Device {
  int id = 0;
  cudaDeviceProp prop{};
  int sm_arch = 0, runtime = 0, driver = 0;
};

inline Device device() {
  Device d;
  MB_CHECK(cudaGetDevice(&d.id));
  MB_CHECK(cudaGetDeviceProperties(&d.prop, d.id));
  MB_CHECK(cudaRuntimeGetVersion(&d.runtime));
  MB_CHECK(cudaDriverGetVersion(&d.driver));
  d.sm_arch = d.prop.major * 10 + d.prop.minor;
  return d;
}

// param: a JSON object literal, e.g. R"({"stride":4})".
inline void emit(const Device& d, const char* probe, const std::string& param, double value,
                 const char* unit) {
  std::printf(
      "{\"probe\":\"%s\",\"param\":%s,\"value\":%.6g,\"unit\":\"%s\",\"device\":\"%s\","
      "\"sm_arch\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"profile\":\"%s\"}\n",
      probe, param.c_str(), value, unit, d.prop.name, d.sm_arch, d.runtime, d.driver,
      aegis::hw::current_profile().label());
  std::fflush(stdout);
}

// --- SASS proof ------------------------------------------------------------

struct Loop {
  std::vector<std::string> lines;  // raw SASS lines of the loop body
  std::map<std::string, int> ops;  // full opcode (with modifiers) -> count
  int count(const std::string& prefix) const {
    int n = 0;
    for (const auto& [op, c] : ops)
      if (op.compare(0, prefix.size(), prefix) == 0) n += c;
    return n;
  }
};

inline std::string run(const std::string& cmd) {
  std::string out;
  std::unique_ptr<FILE, int (*)(FILE*)> p(popen(cmd.c_str(), "r"), pclose);
  if (!p) return out;
  char buf[4096];
  while (size_t n = std::fread(buf, 1, sizeof buf, p.get())) out.append(buf, n);
  return out;
}

// This executable's real path (cuobjdump cannot read /proc/self/exe itself).
inline std::string self_path() {
  char buf[PATH_MAX] = {0};
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  return n > 0 ? std::string(buf, size_t(n)) : std::string("/proc/self/exe");
}

// The loop (backward BRA) in `kernel`'s SASS inside this executable with the
// most instructions starting with `measured`; ties go to the larger loop.
inline Loop timed_loop(const char* kernel, const std::string& measured) {
  const std::string sass = run(std::string("cuobjdump -sass -fun ") + kernel + " '" +
                               self_path() + "' 2>/dev/null");
  static const std::regex ins(R"(/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;)");
  static const std::regex bra(R"(\bBRA\b.*?(0x[0-9a-f]+))");
  std::vector<std::pair<long, std::string>> code;
  std::smatch m;
  size_t pos = 0;
  while (pos < sass.size()) {
    const size_t eol = std::min(sass.find('\n', pos), sass.size());
    const std::string line = sass.substr(pos, eol - pos);
    if (std::regex_search(line, m, ins)) code.emplace_back(std::stol(m[1], nullptr, 16), m[2]);
    pos = eol + 1;
  }
  Loop best;
  for (const auto& [addr, text] : code) {
    if (!std::regex_search(text, m, bra)) continue;
    const long target = std::stol(m[1], nullptr, 16);
    if (target > addr) continue;
    Loop l;
    for (const auto& [a, t] : code)
      if (a >= target && a <= addr) {
        l.lines.push_back(t);
        std::string op = std::regex_replace(t, std::regex(R"(^@!?U?P\w+\s+)"), "");
        op = op.substr(0, op.find(' '));
        ++l.ops[op];
      }
    const int lc = l.count(measured), bc = best.count(measured);
    if (lc > bc || (lc == bc && l.lines.size() > best.lines.size())) best = std::move(l);
  }
  return best;
}

// Print the timed loop to stderr and check exact counts of measured opcodes
// (by prefix, e.g. {"HMMA", 8}). Exits 3 on mismatch.
inline void prove(const char* kernel, const std::vector<std::pair<std::string, int>>& expect) {
  const Loop l = timed_loop(kernel, expect.empty() ? std::string() : expect.front().first);
  std::fprintf(stderr, "=== SASS timed loop of %s (%zu instructions) ===\n", kernel, l.lines.size());
  for (const auto& t : l.lines) std::fprintf(stderr, "    %s\n", t.c_str());
  bool ok = !l.lines.empty();
  for (const auto& [prefix, want] : expect) {
    const int got = l.count(prefix);
    std::fprintf(stderr, "    check %-14s expected %3d found %3d  %s\n", prefix.c_str(), want, got,
                 got == want ? "ok" : "MISMATCH");
    ok &= got == want;
  }
  if (!ok) {
    std::fprintf(stderr, "SASS proof FAILED for %s: measured instructions are not as written\n",
                 kernel);
    std::exit(3);
  }
}

// --- Clock -----------------------------------------------------------------

__global__ void spin_kernel(long long cycles, long long* out) {
  const long long t0 = clock64_volatile();
  long long t = t0;
  while (t - t0 < cycles) t = clock64_volatile();
  *out = t - t0;
}

// Effective SM clock in MHz under load: spin a known number of cycles, time
// it with events. Datasheet clocks are not used.
inline double sm_clock_mhz() {
  long long* d;
  MB_CHECK(aegis::budget::gpu_malloc(&d, sizeof(long long)));
  spin_kernel<<<1, 1>>>(1000000, d);  // warm up clocks
  cudaEvent_t a, b;
  MB_CHECK(cudaEventCreate(&a));
  MB_CHECK(cudaEventCreate(&b));
  MB_CHECK(cudaEventRecord(a));
  spin_kernel<<<1, 1>>>(200000000, d);
  MB_CHECK(cudaEventRecord(b));
  MB_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  long long cycles = 0;
  MB_CHECK(cudaEventElapsedTime(&ms, a, b));
  MB_CHECK(cudaMemcpy(&cycles, d, sizeof cycles, cudaMemcpyDeviceToHost));
  aegis::budget::gpu_free(d);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return cycles / (ms * 1e3);
}

// Elapsed ms of `launch()` bracketed by events.
template <class F>
float time_ms(F&& launch) {
  cudaEvent_t a, b;
  MB_CHECK(cudaEventCreate(&a));
  MB_CHECK(cudaEventCreate(&b));
  MB_CHECK(cudaEventRecord(a));
  launch();
  MB_CHECK(cudaGetLastError());
  MB_CHECK(cudaEventRecord(b));
  MB_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  MB_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms;
}

}  // namespace mb
