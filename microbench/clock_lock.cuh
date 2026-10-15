// Locked SM clocks for latency probes (CLAUDE.md rule 8: latencies are stored
// in cycles, with the SM clock they were measured at).
//
// A latency probe requests locked clocks and records whether it got them on
// every row it emits:
//   - AEGIS_CLOCKS_LOCKED_MHZ=F: the caller locked the clocks at F MHz itself
//     (sudo nvidia-smi -lgc F,F; microbench/run_latency_locked.sh). The probe
//     does not touch them; it verifies the measured SM clock before and after.
//   - otherwise: NVML nvmlDeviceSetGpuLockedClocks(F, F), F = AEGIS_LOCK_SM_MHZ
//     or the default applications clock, reset when the probe exits. Without
//     root NVML refuses; the row then says clocks_locked:false and why.
// In both cases the lock counts only if the SM clock measured before and after
// the probe (mb::sm_clock_mhz, a timed spin) is within kLockTolerance of F.
//
// Rows: emit_clocked() appends "sm_clock_mhz" (mean of before/after),
// "clocks_locked" and "clock_lock" (what was requested, what happened).
#pragma once

#include <dlfcn.h>

#include <cmath>
#include <string>

#include "common.cuh"

namespace mb {

inline constexpr double kLockTolerance = 0.02;

class ClockLock {
 public:
  ClockLock() {
    if (const char* e = std::getenv("AEGIS_CLOCKS_LOCKED_MHZ")) {
      target_ = std::atof(e);
      external_ = true;
      detail_ = "locked externally at " + fmt(target_) + " MHz (AEGIS_CLOCKS_LOCKED_MHZ)";
    } else {
      request_nvml();
    }
    before_ = sm_clock_mhz();
  }
  ~ClockLock() {
    if (we_set_ && reset_) reset_(dev_);
    if (shutdown_) shutdown_();
    if (lib_) dlclose(lib_);
  }
  ClockLock(const ClockLock&) = delete;
  ClockLock& operator=(const ClockLock&) = delete;

  // Call after the measurements: samples the clock again and settles `locked`.
  void finish() {
    after_ = sm_clock_mhz();
    const auto near = [&](double x) { return target_ > 0 && std::fabs(x / target_ - 1.0) <= kLockTolerance; };
    locked_ = (external_ || we_set_) && near(before_) && near(after_);
    detail_ += "; measured " + fmt(before_) + " / " + fmt(after_) + " MHz before / after";
    if ((external_ || we_set_) && !locked_) detail_ += " (not at the target: not counted as locked)";
  }
  double mhz() const { return (before_ + (after_ > 0 ? after_ : before_)) / (after_ > 0 ? 2.0 : 1.0); }
  bool locked() const { return locked_; }
  const std::string& detail() const { return detail_; }

 private:
  using Ret = int;
  using Dev = void*;
  static std::string fmt(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%.0f", x);
    return b;
  }
  void request_nvml() {
    lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib_) {
      detail_ = "lock not requested: no libnvidia-ml";
      return;
    }
    auto sym = [&](auto& f, const char* n) { return (f = reinterpret_cast<std::decay_t<decltype(f)>>(dlsym(lib_, n))); };
    Ret (*init)() = nullptr;
    Ret (*by_bus)(const char*, Dev*) = nullptr;
    Ret (*app_clock)(Dev, int, unsigned*) = nullptr;
    Ret (*set)(Dev, unsigned, unsigned) = nullptr;
    const char* (*err)(Ret) = nullptr;
    if (!(sym(init, "nvmlInit_v2") && sym(shutdown_, "nvmlShutdown") &&
          sym(by_bus, "nvmlDeviceGetHandleByPciBusId_v2") && sym(app_clock, "nvmlDeviceGetDefaultApplicationsClock") &&
          sym(set, "nvmlDeviceSetGpuLockedClocks") && sym(reset_, "nvmlDeviceResetGpuLockedClocks") &&
          sym(err, "nvmlErrorString"))) {
      detail_ = "lock not requested: NVML symbols missing";
      return;
    }
    if (Ret r = init(); r != 0) {
      shutdown_ = nullptr;
      detail_ = std::string("lock not requested: nvmlInit: ") + err(r);
      return;
    }
    char bus[32] = {};
    int cur = 0;
    MB_CHECK(cudaGetDevice(&cur));
    MB_CHECK(cudaDeviceGetPCIBusId(bus, sizeof bus, cur));
    if (Ret r = by_bus(bus, &dev_); r != 0) {
      detail_ = std::string("lock not requested: nvmlDeviceGetHandleByPciBusId: ") + err(r);
      return;
    }
    unsigned f = 0;
    if (const char* e = std::getenv("AEGIS_LOCK_SM_MHZ")) f = unsigned(std::atoi(e));
    else if (Ret r = app_clock(dev_, 0 /* NVML_CLOCK_GRAPHICS */, &f); r != 0) f = 0;
    if (f == 0) {
      detail_ = "lock not requested: no target clock (set AEGIS_LOCK_SM_MHZ)";
      return;
    }
    target_ = f;
    if (Ret r = set(dev_, f, f); r != 0) {
      detail_ = "lock requested at " + fmt(f) + " MHz via NVML: " + err(r);
      return;
    }
    we_set_ = true;
    detail_ = "locked at " + fmt(f) + " MHz via NVML (reset on exit)";
  }

  void* lib_ = nullptr;
  Dev dev_ = nullptr;
  Ret (*shutdown_)() = nullptr;
  Ret (*reset_)(Dev) = nullptr;
  double target_ = 0, before_ = 0, after_ = 0;
  bool external_ = false, we_set_ = false, locked_ = false;
  std::string detail_;
};

// emit() plus the clock the value was measured at.
inline void emit_clocked(const Device& d, const char* probe, const std::string& param, double value,
                         const char* unit, const ClockLock& clk) {
  std::string detail = clk.detail();
  for (char& c : detail)
    if (c == '"' || c == '\\') c = '\'';
  std::printf(
      "{\"probe\":\"%s\",\"param\":%s,\"value\":%.6g,\"unit\":\"%s\",\"device\":\"%s\","
      "\"sm_arch\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"sm_clock_mhz\":%.2f,\"clocks_locked\":%s,"
      "\"clock_lock\":\"%s\"}\n",
      probe, param.c_str(), value, unit, d.prop.name, d.sm_arch, d.runtime, d.driver, clk.mhz(),
      clk.locked() ? "true" : "false", detail.c_str());
  std::fflush(stdout);
}

}  // namespace mb
