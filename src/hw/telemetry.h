#pragma once
// Telemetry (aegis-hw step 7): an opt-in sampler thread for reporting and
// sizing. It reads; Core never changes clocks or power limits.
//
// Every interval (default 100 ms):
//   temperature  NVML GPU sensor; else the hottest sysfs thermal zone
//   SM / memory clock, MHz  NVML; else the Tegra GPU devfreq cur_freq (SM only)
//   power, W     NVML power usage, where the device reports it (-1 otherwise)
//   reasons      NVML clock-event reasons (nvmlDeviceGetCurrentClocksEventReasons,
//                bit values of nvml.h nvmlClocksEventReason*)
// Samples go into a ring of fixed capacity (allocated once). Users bracket a
// measurement with mark() and summarize the samples taken inside it.
//
// Sustained mode: a time-to-throttle profile (tools: aegis_throttle_profile)
// records temperature and clocks under continuous full load. sustained_duty()
// turns it into the largest fraction of time the device can be kept busy while
// its projected steady temperature stays below the throttle point; reporting and
// sizing only (callers cap concurrent work with sustained_concurrency()).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aegis::hw {

// nvml.h nvmlClocksEventReason* bits.
enum ClockEventReason : uint64_t {
  kReasonGpuIdle = 0x1,
  kReasonAppClocksSetting = 0x2,
  kReasonSwPowerCap = 0x4,
  kReasonHwSlowdown = 0x8,
  kReasonSyncBoost = 0x10,
  kReasonSwThermalSlowdown = 0x20,
  kReasonHwThermalSlowdown = 0x40,
  kReasonHwPowerBrakeSlowdown = 0x80,
  kReasonDisplayClockSetting = 0x100,
};
// The reasons that slow the clocks down under load: power cap, thermal and
// hardware slowdowns, power brake, sync boost. Idle and clock-setting reasons
// are states, not throttling.
inline constexpr uint64_t kThrottleReasons = kReasonSwPowerCap | kReasonHwSlowdown | kReasonSyncBoost |
                                             kReasonSwThermalSlowdown | kReasonHwThermalSlowdown |
                                             kReasonHwPowerBrakeSlowdown;
// "sw_power_cap|hw_thermal_slowdown", "" for none.
std::string reason_names(uint64_t bits);

struct TelemetrySample {
  double t_s = 0;         // since the sampler started
  double temp_c = -1;     // -1: unknown
  int sm_mhz = -1, mem_mhz = -1;
  double power_w = -1;
  uint64_t reasons = 0;
  bool reasons_known = false;
};

struct TelemetrySummary {
  int samples = 0;              // samples inside the interval (0: the latest one before it is used)
  double max_temp_c = -1;
  uint64_t reasons = 0;         // OR over the samples, all bits
  uint64_t throttle = 0;        // reasons & kThrottleReasons
  int min_sm_mhz = -1, max_sm_mhz = -1;
  double mean_power_w = -1;
};

struct TelemetryOptions {
  int interval_ms = 100;
  int device = 0;          // CUDA device (NVML handle by its PCI bus id)
  std::string root = "/";  // sysfs prefix for the fallback (fixtures in tests)
  bool nvml = true;        // false: the sysfs path only
  size_t capacity = size_t(1) << 15;  // ring: 54 min at 100 ms
  // Pin the sampler thread to the efficiency cores (every core class but the
  // performance one, from hardware discovery: the Cortex-A725s on the Spark), away
  // from the threads that launch work. One core class only: not pinned. Off by
  // default: unpinned at 100 ms already costs <= 0.28% (paired-ab id 63, decode).
  bool pin_efficiency = false;
};

class Telemetry {
 public:
  explicit Telemetry(const TelemetryOptions& opt = {});
  ~Telemetry();
  Telemetry(const Telemetry&) = delete;
  Telemetry& operator=(const Telemetry&) = delete;

  // "nvml", "sysfs", or "unavailable: <why>"; per quantity in detail().
  const std::string& source() const { return source_; }
  const std::string& detail() const { return detail_; }
  const std::vector<int>& sampler_cpus() const { return sampler_cpus_; }  // empty: not pinned
  bool ok() const { return ok_; }

  size_t mark() const { return taken_.load(std::memory_order_acquire); }  // index of the next sample
  TelemetrySummary summary(size_t from, size_t to) const;
  std::vector<TelemetrySample> samples(size_t from, size_t to) const;
  // One sample now, on the calling thread (also what the sampler thread runs).
  TelemetrySample sample_now();

 private:
  struct Impl;
  void loop();
  TelemetryOptions opt_;
  Impl* impl_ = nullptr;
  std::string source_, detail_;
  std::vector<int> sampler_cpus_;
  bool ok_ = false;
  mutable std::mutex mu_;
  std::vector<TelemetrySample> ring_;
  std::atomic<size_t> taken_{0};
  std::atomic<bool> stop_{false};
  std::thread thread_;
  double t0_ = 0;
};

// Sustained mode.
struct ThrottleProfile {
  std::string device;
  double ambient_c = -1;          // temperature at the start of the load
  double throttle_c = -1;         // temperature at the first throttle sample; -1: none seen
  double time_to_throttle_s = -1; // -1: no throttle within the recording
  double recorded_s = 0;
  double steady_c = -1;           // temperature the full load approaches (fit)
  double tau_s = -1;              // first-order time constant (fit)
  double margin_c = 3;            // stay this far below the throttle point
  // What marked the throttle point: "clock_event_reasons" (NVML reported a throttle
  // reason), "sm_clock_sag" (the 10 s rolling median of the SM clock fell below
  // (1 - clock_sag) x the first 10 s median; GB10 sags with no reason reported),
  // or "" (none).
  std::string cause;
  double clock_ref_mhz = -1;      // median SM clock over the first 10 s of load
  double clock_sag = 0.03;
};
// From (t, temperature, throttled) samples of a continuous full load: ambient,
// first throttle, and a first-order fit T(t) = steady - (steady - ambient) e^{-t/tau}
// over the whole recording.
// The throttle point is the first sample with a throttle reason, or the first
// sample after the first 10 s whose next 10 s have a median SM clock below
// (1 - clock_sag) x the first 10 s median, whichever comes first.
ThrottleProfile fit_throttle_profile(const std::vector<TelemetrySample>& load, uint64_t throttle_mask = kThrottleReasons,
                                     double clock_sag = 0.03);
// The largest busy fraction in [0, 1] whose steady temperature, ambient + d x
// (steady - ambient), stays margin_c below the throttle point. 1 when the profile
// never throttled.
double sustained_duty(const ThrottleProfile& p);
// Concurrent work items out of `requested` that keep the device under its throttle
// point: max(1, floor(duty x requested)).
int sustained_concurrency(const ThrottleProfile& p, int requested);
std::string to_json(const ThrottleProfile& p);
bool from_json(const std::string& json, ThrottleProfile* p);

}  // namespace aegis::hw
