// Probe: time-to-throttle profile under sustained full load (aegis-hw step 7,
// sustained mode). Not part of run_all.sh: it runs for minutes.
//
//   mb_throttle_profile [--seconds 300] [--n 4096] [--samples FILE] [--profile FILE]
//   mb_throttle_profile --refit SAMPLES   (the profile of a recording; no GPU)
//
// Load: aegis::matmul N x N x N (FP16), back to back on one stream, drained every
// 32 launches so the queue stays short. Telemetry (src/hw/telemetry.h) samples
// every 100 ms: temperature, SM clock, power, clock-event reasons. Throughput is
// counted per 10 s window (launches completed), so a clock that sags shows up as
// work lost, not only as a lower MHz.
//
// Output:
//   --samples FILE  one JSON line per telemetry sample plus the window counts
//   stdout          probe "throttle_profile" rows (microbench format): ambient,
//                   max and steady temperature, time to throttle (-1: none),
//                   tau, sustained duty, SM clock min/max, mean power, throughput
//                   in the first and last 10 s windows
//   --profile FILE  hw::to_json (aegis-throttle/1) plus reasons seen, launches, wall
//                   time and SM count (also on stderr)
// Reporting and sizing only: nothing here changes clocks or power limits.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "aegis/matmul.h"
#include "common.cuh"
#include "hw/json_lite.h"
#include "hw/telemetry.h"

// --refit FILE: the profile of a recorded --samples file, no GPU (the fit
// criteria changed, or to inspect a recording).
int refit(const std::string& path) {
  std::ifstream f(path);
  std::vector<aegis::hw::TelemetrySample> load;
  for (std::string line; std::getline(f, line);) {
    aegis::hw::json::Value v;
    if (!aegis::hw::json::Parser(line).parse(&v) || !v["t_s"].is_number()) continue;
    aegis::hw::TelemetrySample s;
    s.t_s = v["t_s"].num;
    s.temp_c = v["temp_c"].number_or(-1);
    s.sm_mhz = int(v["sm_mhz"].number_or(-1));
    s.power_w = v["power_w"].number_or(-1);
    s.reasons_known = true;  // recorded reasons are names; any throttle name marks the sample
    const std::string r = v["reasons"].is_string() ? v["reasons"].str : "";
    for (const char* n : {"sw_power_cap", "hw_slowdown", "sync_boost", "sw_thermal_slowdown", "hw_thermal_slowdown",
                          "hw_power_brake_slowdown"})
      if (r.find(n) != std::string::npos) s.reasons |= aegis::hw::kThrottleReasons;
    load.push_back(s);
  }
  std::printf("%s\n", aegis::hw::to_json(aegis::hw::fit_throttle_profile(load)).c_str());
  return load.empty() ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--refit") return refit(argv[2]);
  double seconds = 300;
  int n = 4096;
  std::string samples_path, profile_path;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--seconds" && i + 1 < argc) seconds = std::stod(argv[++i]);
    else if (k == "--n" && i + 1 < argc) n = std::stoi(argv[++i]);
    else if (k == "--samples" && i + 1 < argc) samples_path = argv[++i];
    else if (k == "--profile" && i + 1 < argc) profile_path = argv[++i];
    else {
      std::fprintf(stderr, "usage: mb_throttle_profile [--seconds 300] [--n 4096] [--samples FILE] [--profile FILE]\n");
      return 2;
    }
  }
  const mb::Device dev = mb::device();
  aegis::hw::TelemetryOptions to;
  to.device = dev.id;
  aegis::hw::Telemetry tel(to);
  if (!tel.ok()) {
    std::fprintf(stderr, "telemetry unavailable: %s\n", tel.source().c_str());
    return 1;
  }
  std::fprintf(stderr, "throttle_profile: %s, %.0f s of aegis::matmul %d^3; telemetry %s (%s)\n", dev.prop.name, seconds,
               n, tel.source().c_str(), tel.detail().c_str());
  const size_t elems = size_t(n) * n;
  __half *A, *B, *C;
  MB_CHECK(aegis::budget::gpu_malloc(&A, elems * sizeof(__half)));
  MB_CHECK(aegis::budget::gpu_malloc(&B, elems * sizeof(__half)));
  MB_CHECK(aegis::budget::gpu_malloc(&C, elems * sizeof(__half)));
  MB_CHECK(cudaMemset(A, 0x3c, elems * sizeof(__half)));  // 1.0-ish halves: real math, finite
  MB_CHECK(cudaMemset(B, 0x1c, elems * sizeof(__half)));
  cudaStream_t s;
  MB_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  MB_CHECK(aegis::matmul(A, B, C, n, n, n, s));  // plan, workspace
  MB_CHECK(cudaStreamSynchronize(s));

  // Idle baseline: one second before the load.
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  const size_t start = tel.mark();
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  std::vector<long long> per_window;  // launches completed per 10 s window
  long long launches = 0, window_launches = 0;
  auto window_end = t0 + std::chrono::seconds(10);
  while (std::chrono::duration<double>(clk::now() - t0).count() < seconds) {
    for (int i = 0; i < 32; ++i) MB_CHECK(aegis::matmul(A, B, C, n, n, n, s));
    MB_CHECK(cudaStreamSynchronize(s));
    launches += 32;
    window_launches += 32;
    if (clk::now() >= window_end) {
      per_window.push_back(window_launches);
      window_launches = 0;
      window_end += std::chrono::seconds(10);
    }
  }
  const size_t end = tel.mark();
  const double wall = std::chrono::duration<double>(clk::now() - t0).count();
  const std::vector<aegis::hw::TelemetrySample> load = tel.samples(start, end);
  const aegis::hw::TelemetrySummary sum = tel.summary(start, end);
  aegis::hw::ThrottleProfile p = aegis::hw::fit_throttle_profile(load);
  p.device = dev.prop.name;

  if (!samples_path.empty()) {
    std::ofstream f(samples_path);
    for (const auto& x : load)
      f << "{\"t_s\":" << x.t_s << ",\"temp_c\":" << x.temp_c << ",\"sm_mhz\":" << x.sm_mhz << ",\"power_w\":"
        << x.power_w << ",\"reasons\":\"" << aegis::hw::reason_names(x.reasons) << "\"}\n";
    for (size_t w = 0; w < per_window.size(); ++w)
      f << "{\"window\":" << w << ",\"seconds\":10,\"launches\":" << per_window[w] << "}\n";
  }
  const std::string base = "{\"n\":" + std::to_string(n) + ",\"seconds\":" + std::to_string(int(seconds)) +
                           ",\"impl\":\"" + aegis::matmul_choice(n, n, n) + "\",\"stat\":\"";
  auto row = [&](const char* stat, double v, const char* unit) { mb::emit(dev, "throttle_profile", base + stat + "\"}", v, unit); };
  row("ambient_c", p.ambient_c, "C");
  row("max_temp_c", sum.max_temp_c, "C");
  row("steady_c_fit", p.steady_c, "C");
  row("tau_s_fit", p.tau_s, "s");
  row("time_to_throttle_s", p.time_to_throttle_s, "s");
  row("throttle_c", p.throttle_c, "C");
  row("sustained_duty", aegis::hw::sustained_duty(p), "fraction");
  row("sm_clock_min_mhz", sum.min_sm_mhz, "MHz");
  row("sm_clock_max_mhz", sum.max_sm_mhz, "MHz");
  row("mean_power_w", sum.mean_power_w, "W");
  row("samples", double(load.size()), "count");
  if (!per_window.empty()) {
    row("first_window_tflops", 2.0 * n * n * double(n) * per_window.front() / 10.0 / 1e12, "TFLOP/s");
    row("last_window_tflops", 2.0 * n * n * double(n) * per_window.back() / 10.0 / 1e12, "TFLOP/s");
  }
  // The profile itself (aegis-throttle/1): --profile FILE, and stderr.
  char tail[256];
  std::snprintf(tail, sizeof tail, ",\"reasons_seen\":\"%s\",\"launches\":%lld,\"wall_s\":%.1f,\"sm_count\":%d}",
                aegis::hw::reason_names(sum.reasons).c_str(), launches, wall, dev.prop.multiProcessorCount);
  std::string prof = aegis::hw::to_json(p);
  prof.pop_back();
  prof += tail;
  std::fprintf(stderr, "%s\n", prof.c_str());
  if (!profile_path.empty()) std::ofstream(profile_path) << prof << "\n";
  MB_CHECK(aegis::budget::gpu_free(C));
  MB_CHECK(aegis::budget::gpu_free(B));
  MB_CHECK(aegis::budget::gpu_free(A));
  return 0;
}
