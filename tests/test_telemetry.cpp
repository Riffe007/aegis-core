// Telemetry and sustained mode (src/hw/telemetry.h), ctest telemetry.
//   1. reason_names: nvml.h bit values, throttle subset.
//   2. sysfs paths on recorded / synthetic trees, NVML off:
//      - orin-nano-synthetic: temperature = the hottest zone (49.0 C), SM clock from
//        the GPU devfreq cur_freq (306 MHz), no power, no reasons;
//      - spark-gb10 (recorded): temperature from its zones, no SM clock.
//   3. This machine: the 100 ms sampler takes ~10 samples a second; a summary over
//      an interval shorter than one period uses the latest sample; quantities NVML
//      cannot read stay -1.
//   4. Sustained mode on synthetic first-order heating curves: the fit recovers
//      ambient, steady and tau; the duty keeps the projected steady temperature
//      margin_c below the throttle point; no throttle -> duty 1; JSON round trip.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "aegis/hw.h"
#include "hw/telemetry.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

std::string fx(const char* name) { return std::string(AEGIS_SOURCE_DIR) + "/tests/fixtures/hw/" + name; }

using namespace aegis::hw;

void names() {
  expect(reason_names(0) == "", "no reasons: empty");
  expect(reason_names(kReasonSwPowerCap | kReasonHwThermalSlowdown) == "sw_power_cap|hw_thermal_slowdown",
         "sw_power_cap|hw_thermal_slowdown");
  expect(reason_names(0x1000) == "0x1000", "unknown bits are printed in hex");
  expect((kThrottleReasons & (kReasonGpuIdle | kReasonAppClocksSetting | kReasonDisplayClockSetting)) == 0,
         "idle and clock-setting reasons are not throttling");
}

void sysfs() {
  TelemetryOptions o;
  o.nvml = false;
  o.interval_ms = 0;  // no thread: sample_now() only
  o.root = fx("orin-nano-synthetic");
  Telemetry orin(o);
  const TelemetrySample s = orin.sample_now();
  expect(orin.ok() && orin.source() == "sysfs" && s.temp_c == 49.0 && s.sm_mhz == 306 && s.mem_mhz < 0 &&
             s.power_w < 0 && !s.reasons_known,
         "orin-nano-synthetic, no NVML: " + orin.detail() + "; " + std::to_string(s.temp_c) + " C, " +
             std::to_string(s.sm_mhz) + " MHz");
  o.root = fx("spark-gb10");
  Telemetry spark(o);
  const TelemetrySample t = spark.sample_now();
  expect(spark.ok() && t.temp_c > 0 && t.sm_mhz < 0, "spark-gb10 fixture, no NVML: temperature " +
                                                         std::to_string(t.temp_c) + " C from its thermal zones");
  o.root = "/nonexistent";
  Telemetry none(o);
  expect(!none.ok() && none.source().rfind("unavailable", 0) == 0, "no sensor at all: " + none.source());
}

void live() {
  Telemetry tel;  // NVML where present, else this machine's sysfs
  const size_t m0 = tel.mark();
  std::this_thread::sleep_for(std::chrono::milliseconds(1050));
  const size_t m1 = tel.mark();
  const TelemetrySummary s = tel.summary(m0, m1);
  expect(tel.ok() && m1 - m0 >= 8 && m1 - m0 <= 12 && s.samples == int(m1 - m0) && s.max_temp_c > 0,
         "live (" + tel.source() + "): " + std::to_string(m1 - m0) + " samples in 1.05 s, max " +
             std::to_string(s.max_temp_c) + " C, SM " + std::to_string(s.min_sm_mhz) + "-" +
             std::to_string(s.max_sm_mhz) + " MHz, power " + std::to_string(s.mean_power_w) + " W, reasons '" +
             reason_names(s.reasons) + "'");
  const size_t m2 = tel.mark();
  const TelemetrySummary z = tel.summary(m2, m2);
  expect(z.samples == 0 && z.max_temp_c > 0, "an interval shorter than one period: the latest sample stands in");
  std::printf("      detail: %s\n", tel.detail().c_str());
  // The sampler runs on the efficiency cores (non-performance classes, discovery).
  aegis::hw::DiscoverOptions d;
  d.cuda = d.nvml = d.video = false;
  std::vector<int> eff;
  for (const auto& t : aegis::hw::discover(d).cpu.core_types)
    if (!t.performance) eff.insert(eff.end(), t.cpus.begin(), t.cpus.end());
  std::sort(eff.begin(), eff.end());
  expect(tel.sampler_cpus().empty(), "default: the sampler is not pinned");
  TelemetryOptions po;
  po.pin_efficiency = true;
  Telemetry pinned(po);
  expect(pinned.sampler_cpus() == eff, "pin_efficiency: sampler on the " + std::to_string(eff.size()) + " efficiency cpus");
  TelemetryOptions o;
  o.pin_efficiency = true;
  o.root = fx("spark-gb10");
  o.nvml = false;
  o.interval_ms = 0;
  Telemetry fixture(o);
  std::string cpus;
  for (int c : fixture.sampler_cpus()) cpus += (cpus.empty() ? "" : ",") + std::to_string(c);
  expect(cpus == "0,1,2,3,4,10,11,12,13,14", "spark-gb10 fixture: sampler cpus " + cpus + " (the A725s)");
}

std::vector<TelemetrySample> curve(double ambient, double steady, double tau, double seconds, double throttle_c) {
  std::vector<TelemetrySample> v;
  for (double t = 0; t <= seconds; t += 0.1) {
    TelemetrySample s;
    s.t_s = 100 + t;  // the sampler's clock does not start at the load
    s.temp_c = std::round((steady - (steady - ambient) * std::exp(-t / tau)) * 1.0);  // NVML reports whole C
    s.reasons_known = true;
    if (throttle_c > 0 && s.temp_c >= throttle_c) s.reasons = kReasonSwThermalSlowdown;
    v.push_back(s);
  }
  return v;
}

void sustained() {
  // Throttles at 85 C on the way to 95 C, from 40 C, tau 60 s.
  ThrottleProfile p = fit_throttle_profile(curve(40, 95, 60, 300, 85));
  p.device = "synthetic";
  const double t_expect = -60 * std::log((95 - 84.5) / 55.0);  // first sample at 85 C (rounded)
  expect(std::fabs(p.ambient_c - 40) < 1e-9 && p.throttle_c == 85 && std::fabs(p.time_to_throttle_s - t_expect) < 1.0,
         "throttle at " + std::to_string(p.time_to_throttle_s) + " s, " + std::to_string(p.throttle_c) + " C");
  expect(std::fabs(p.steady_c - 95) < 2.5 && std::fabs(p.tau_s - 60) < 8,
         "fit: steady " + std::to_string(p.steady_c) + " C, tau " + std::to_string(p.tau_s) + " s (truth 95, 60)");
  const double d = sustained_duty(p);
  const double d_expect = (85 - 3 - 40) / (p.steady_c - 40);
  expect(std::fabs(d - d_expect) < 1e-9 && d > 0.6 && d < 0.85,
         "duty " + std::to_string(d) + ": steady 40 + d x rise stays 3 C under 85 C");
  expect(sustained_concurrency(p, 8) == int(std::floor(d * 8)) && sustained_concurrency(p, 1) == 1,
         "concurrency: 8 requested -> " + std::to_string(sustained_concurrency(p, 8)) + "; never below 1");

  const ThrottleProfile cool = fit_throttle_profile(curve(40, 60, 30, 300, 0));
  expect(cool.time_to_throttle_s < 0 && sustained_duty(cool) == 1.0 && sustained_concurrency(cool, 8) == 8,
         "no throttle in the recording: duty 1, no cap");

  // GB10's case: no reason reported, the SM clock sags 5% from 70 C on.
  auto sag = curve(40, 95, 60, 300, 0);
  for (auto& x : sag) x.sm_mhz = x.temp_c >= 70 ? 2400 : 2528;
  const ThrottleProfile g = fit_throttle_profile(sag);
  expect(g.cause == "sm_clock_sag" && g.throttle_c == 70 && g.clock_ref_mhz == 2528 && sustained_duty(g) < 1,
         "clock sag without a reason: throttle at " + std::to_string(g.time_to_throttle_s) + " s, " +
             std::to_string(g.throttle_c) + " C (" + g.cause + "), duty " + std::to_string(sustained_duty(g)));
  auto blip = curve(40, 60, 30, 300, 0);
  for (size_t i = 0; i < blip.size(); ++i) blip[i].sm_mhz = (i >= 500 && i < 530) ? 2229 : 2405;
  expect(fit_throttle_profile(blip).cause.empty(), "a 3 s clock dip is not a throttle (10 s median)");
  auto boost = curve(40, 60, 30, 300, 0);  // a few boost samples at the start set no reference
  for (size_t i = 0; i < boost.size(); ++i) boost[i].sm_mhz = i < 5 ? 2528 : 2405;
  expect(fit_throttle_profile(boost).cause.empty() && fit_throttle_profile(boost).clock_ref_mhz == 2405,
         "boost samples at the start: reference is the 10 s median (2405), no throttle");
  expect(p.cause == "clock_event_reasons", "reason-marked throttle: cause clock_event_reasons");

  ThrottleProfile back;
  expect(from_json(to_json(p), &back) && back.device == "synthetic" && std::fabs(back.steady_c - p.steady_c) < 0.01 &&
             std::fabs(sustained_duty(back) - d) < 1e-3 && back.cause == p.cause,
         "JSON round trip: " + to_json(p));
  expect(!from_json("{not json", &back), "malformed JSON is refused");
}

}  // namespace

int main() {
  names();
  sysfs();
  live();
  sustained();
  std::printf("%s: %d failures\n", failures ? "telemetry FAILED" : "telemetry ok", failures);
  return failures ? 1 : 0;
}
