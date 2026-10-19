// Telemetry sampler and sustained-mode sizing (telemetry.h).
#include "telemetry.h"

#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <cuda_runtime.h>

#include "aegis/hw.h"
#include "json_lite.h"

namespace aegis::hw {

namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool read_ll(const std::string& path, long long* v) {
  std::ifstream f(path);
  return f && (f >> *v);
}

std::vector<std::string> list_dir(const std::string& path) {
  std::vector<std::string> out;
  if (DIR* d = opendir(path.c_str())) {
    while (dirent* e = readdir(d))
      if (e->d_name[0] != '.') out.push_back(e->d_name);
    closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string join(const std::string& root, const std::string& path) {
  return root == "/" || root.empty() ? path : root + path;
}

}  // namespace

std::string reason_names(uint64_t bits) {
  static const std::pair<uint64_t, const char*> names[] = {
      {kReasonGpuIdle, "gpu_idle"},
      {kReasonAppClocksSetting, "applications_clocks_setting"},
      {kReasonSwPowerCap, "sw_power_cap"},
      {kReasonHwSlowdown, "hw_slowdown"},
      {kReasonSyncBoost, "sync_boost"},
      {kReasonSwThermalSlowdown, "sw_thermal_slowdown"},
      {kReasonHwThermalSlowdown, "hw_thermal_slowdown"},
      {kReasonHwPowerBrakeSlowdown, "hw_power_brake_slowdown"},
      {kReasonDisplayClockSetting, "display_clock_setting"},
  };
  std::string s;
  uint64_t known = 0;
  for (const auto& [bit, name] : names) {
    known |= bit;
    if (bits & bit) s += (s.empty() ? "" : "|") + std::string(name);
  }
  if (bits & ~known) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)(bits & ~known));
    s += (s.empty() ? "" : "|") + std::string(buf);
  }
  return s;
}

// NVML by dlopen (Tegra has none); nvml.h types as plain ints, 0 = success.
struct Telemetry::Impl {
  void* lib = nullptr;
  using Dev = void*;
  int (*init)() = nullptr;
  int (*shutdown)() = nullptr;
  int (*by_bus)(const char*, Dev*) = nullptr;
  int (*by_index)(unsigned, Dev*) = nullptr;
  int (*temp)(Dev, int, unsigned*) = nullptr;               // nvmlDeviceGetTemperature (sensor 0: GPU)
  int (*clock)(Dev, int, unsigned*) = nullptr;              // nvmlDeviceGetClockInfo (1: SM, 2: MEM)
  int (*power)(Dev, unsigned*) = nullptr;                   // nvmlDeviceGetPowerUsage, mW
  int (*reasons)(Dev, unsigned long long*) = nullptr;       // nvmlDeviceGetCurrentClocksEventReasons
  Dev dev = nullptr;
  bool nvml = false, has_temp = false, has_sm = false, has_mem = false, has_power = false, has_reasons = false;
  std::vector<std::string> zones;  // sysfs thermal zone temp files
  std::string devfreq_cur;         // Tegra GPU devfreq cur_freq (Hz)

  ~Impl() {
    if (nvml && shutdown) shutdown();
    if (lib) dlclose(lib);
  }
};

Telemetry::Telemetry(const TelemetryOptions& opt) : opt_(opt), impl_(new Impl) {
  Impl& m = *impl_;
  std::string notes;
  if (opt.nvml && (m.lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL))) {
    auto sym = [&](auto& f, const char* n) {
      f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(dlsym(m.lib, n));
      return f != nullptr;
    };
    if (sym(m.init, "nvmlInit_v2") && sym(m.shutdown, "nvmlShutdown") && sym(m.by_bus, "nvmlDeviceGetHandleByPciBusId_v2") &&
        sym(m.by_index, "nvmlDeviceGetHandleByIndex_v2") && m.init() == 0) {
      m.nvml = true;
      char bus[64] = "";
      int r = -1;
      if (cudaDeviceGetPCIBusId(bus, sizeof bus, opt.device) == cudaSuccess) r = m.by_bus(bus, &m.dev);
      else cudaGetLastError();
      if (r != 0) r = m.by_index(unsigned(opt.device), &m.dev);
      if (r == 0) {
        sym(m.temp, "nvmlDeviceGetTemperature");
        sym(m.clock, "nvmlDeviceGetClockInfo");
        sym(m.power, "nvmlDeviceGetPowerUsage");
        if (!sym(m.reasons, "nvmlDeviceGetCurrentClocksEventReasons")) sym(m.reasons, "nvmlDeviceGetCurrentClocksThrottleReasons");
        unsigned u = 0;
        unsigned long long ull = 0;
        m.has_temp = m.temp && m.temp(m.dev, 0, &u) == 0;
        m.has_sm = m.clock && m.clock(m.dev, 1, &u) == 0;
        m.has_mem = m.clock && m.clock(m.dev, 2, &u) == 0;
        m.has_power = m.power && m.power(m.dev, &u) == 0;
        m.has_reasons = m.reasons && m.reasons(m.dev, &ull) == 0;
      } else {
        notes += "nvml: no device handle; ";
      }
    } else {
      notes += "nvml: init failed; ";
    }
  } else if (opt.nvml) {
    notes += "nvml: libnvidia-ml.so.1 not found; ";
  }
  // sysfs: every thermal zone (the hottest is the temperature without NVML) and
  // the Tegra GPU devfreq node.
  const std::string tz = join(opt.root, "/sys/class/thermal");
  for (const std::string& d : list_dir(tz))
    if (d.compare(0, 12, "thermal_zone") == 0) m.zones.push_back(tz + "/" + d + "/temp");
  const std::string df = join(opt.root, "/sys/class/devfreq");
  for (const std::string& d : list_dir(df))
    if (d.find("gpu") != std::string::npos || d.find("ga10b") != std::string::npos || d.find("gv11b") != std::string::npos) {
      m.devfreq_cur = df + "/" + d + "/cur_freq";
      break;
    }
  auto src = [&](bool nv, bool sys) { return nv ? "nvml" : sys ? "sysfs" : "unavailable"; };
  detail_ = notes + std::string("temperature ") + src(m.has_temp, !m.zones.empty()) + ", sm clock " +
            src(m.has_sm, !m.devfreq_cur.empty()) + ", mem clock " + src(m.has_mem, false) + ", power " +
            src(m.has_power, false) + ", clock-event reasons " + src(m.has_reasons, false);
  ok_ = m.has_temp || m.has_sm || !m.zones.empty() || !m.devfreq_cur.empty();
  source_ = !ok_ ? "unavailable: no NVML quantity and no sysfs sensor" : m.nvml && (m.has_temp || m.has_sm) ? "nvml" : "sysfs";
  ring_.resize(std::max<size_t>(1, opt.capacity));
  if (opt.pin_efficiency) {  // the non-performance core classes, from discovery
    DiscoverOptions d;
    d.root = opt.root;
    d.cuda = d.nvml = d.video = false;
    for (const CoreType& t : discover(d).cpu.core_types)
      if (!t.performance) sampler_cpus_.insert(sampler_cpus_.end(), t.cpus.begin(), t.cpus.end());
    std::sort(sampler_cpus_.begin(), sampler_cpus_.end());
  }
  detail_ += sampler_cpus_.empty() ? "; sampler not pinned"
                                   : "; sampler pinned to " + std::to_string(sampler_cpus_.size()) + " efficiency cpus";
  t0_ = now_s();
  if (ok_ && opt.interval_ms > 0) thread_ = std::thread([this] { loop(); });
}

Telemetry::~Telemetry() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  delete impl_;
}

TelemetrySample Telemetry::sample_now() {
  Impl& m = *impl_;
  TelemetrySample s;
  s.t_s = now_s() - t0_;
  unsigned u = 0;
  unsigned long long r = 0;
  if (m.has_temp && m.temp(m.dev, 0, &u) == 0) {
    s.temp_c = u;
  } else {
    for (const std::string& z : m.zones) {
      long long mc;
      if (read_ll(z, &mc)) s.temp_c = std::max(s.temp_c, double(mc) / 1000.0);
    }
  }
  if (m.has_sm && m.clock(m.dev, 1, &u) == 0) {
    s.sm_mhz = int(u);
  } else if (!m.devfreq_cur.empty()) {
    long long hz;
    if (read_ll(m.devfreq_cur, &hz)) s.sm_mhz = int(hz / 1000000);
  }
  if (m.has_mem && m.clock(m.dev, 2, &u) == 0) s.mem_mhz = int(u);
  if (m.has_power && m.power(m.dev, &u) == 0) s.power_w = u / 1000.0;
  if (m.has_reasons && m.reasons(m.dev, &r) == 0) {
    s.reasons = r;
    s.reasons_known = true;
  }
  return s;
}

void Telemetry::loop() {
  if (!sampler_cpus_.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : sampler_cpus_) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  }
  using clk = std::chrono::steady_clock;
  auto next = clk::now();
  while (!stop_.load(std::memory_order_relaxed)) {
    const TelemetrySample s = sample_now();
    {
      std::lock_guard<std::mutex> l(mu_);
      const size_t i = taken_.load(std::memory_order_relaxed);
      ring_[i % ring_.size()] = s;
      taken_.store(i + 1, std::memory_order_release);
    }
    next += std::chrono::milliseconds(opt_.interval_ms);
    // Sleep in short steps so the destructor never waits a whole interval.
    while (!stop_.load(std::memory_order_relaxed) && clk::now() < next)
      std::this_thread::sleep_for(std::min<clk::duration>(next - clk::now(), std::chrono::milliseconds(10)));
  }
}

std::vector<TelemetrySample> Telemetry::samples(size_t from, size_t to) const {
  std::lock_guard<std::mutex> l(mu_);
  const size_t n = taken_.load(std::memory_order_relaxed);
  to = std::min(to, n);
  from = std::max(from, n > ring_.size() ? n - ring_.size() : 0);
  std::vector<TelemetrySample> out;
  for (size_t i = from; i < to; ++i) out.push_back(ring_[i % ring_.size()]);
  return out;
}

TelemetrySummary Telemetry::summary(size_t from, size_t to) const {
  TelemetrySummary sum;
  std::lock_guard<std::mutex> l(mu_);
  const size_t n = taken_.load(std::memory_order_relaxed);
  to = std::min(to, n);
  const size_t oldest = n > ring_.size() ? n - ring_.size() : 0;
  from = std::max(from, oldest);
  const size_t inside = to > from ? to - from : 0;
  // An interval shorter than one sample period: the latest sample before its end
  // stands in for it (samples stays 0).
  if (from >= to && to > oldest) from = to - 1;
  double power = 0;
  int powered = 0;
  for (size_t i = from; i < to; ++i) {
    const TelemetrySample& s = ring_[i % ring_.size()];
    sum.max_temp_c = std::max(sum.max_temp_c, s.temp_c);
    sum.reasons |= s.reasons;
    if (s.sm_mhz >= 0) {
      sum.min_sm_mhz = sum.min_sm_mhz < 0 ? s.sm_mhz : std::min(sum.min_sm_mhz, s.sm_mhz);
      sum.max_sm_mhz = std::max(sum.max_sm_mhz, s.sm_mhz);
    }
    if (s.power_w >= 0) power += s.power_w, ++powered;
  }
  sum.samples = int(inside);
  sum.throttle = sum.reasons & kThrottleReasons;
  sum.mean_power_w = powered ? power / powered : -1;
  return sum;
}

// ---------------------------------------------------------------------------
// Sustained mode

ThrottleProfile fit_throttle_profile(const std::vector<TelemetrySample>& load, uint64_t mask, double clock_sag) {
  ThrottleProfile p;
  p.clock_sag = clock_sag;
  if (load.empty()) return p;
  const double t0 = load.front().t_s;
  p.recorded_s = load.back().t_s - t0;
  // Ambient: the first sample's temperature (the load starts right after it).
  for (const TelemetrySample& s : load)
    if (s.temp_c >= 0) {
      p.ambient_c = s.temp_c;
      break;
    }
  size_t onset = load.size();
  for (size_t i = 0; i < load.size(); ++i)
    if (load[i].reasons_known && (load[i].reasons & mask)) {
      onset = i;
      p.cause = "clock_event_reasons";
      break;
    }
  // Clock sag: the 10 s rolling median of the SM clock against the median of the
  // first 10 s of load (medians: GB10 boosts and dips for single samples).
  auto median_in = [&](double from_s, double to_s, bool* full) {
    std::vector<int> v;
    for (const TelemetrySample& s : load)
      if (s.t_s - t0 >= from_s && s.t_s - t0 < to_s && s.sm_mhz > 0) v.push_back(s.sm_mhz);
    if (full) *full = !load.empty() && load.back().t_s - t0 >= to_s;
    if (v.empty()) return -1.0;
    std::nth_element(v.begin(), v.begin() + long(v.size() / 2), v.end());
    return double(v[v.size() / 2]);
  };
  p.clock_ref_mhz = median_in(0, 10, nullptr);
  if (p.clock_ref_mhz > 0) {
    const double lim = p.clock_ref_mhz * (1 - clock_sag);
    for (size_t i = 0; i < std::min(onset, load.size()); ++i) {
      const double t = load[i].t_s - t0;
      if (t < 10) continue;
      bool full = false;
      const double m = median_in(t, t + 10, &full);
      if (!full) break;
      if (m > 0 && m < lim) {
        // The throttle starts at the first slow sample of that window.
        onset = i;
        while (onset < load.size() && !(load[onset].sm_mhz > 0 && load[onset].sm_mhz < lim)) ++onset;
        p.cause = "sm_clock_sag";
        break;
      }
    }
  }
  if (onset < load.size()) {
    p.time_to_throttle_s = load[onset].t_s - t0;
    p.throttle_c = load[onset].temp_c;
  }
  // First-order fit over the whole recording: steady by a scan, tau by least
  // squares of ln((steady - T) / (steady - ambient)) = -t / tau. The whole curve,
  // not only up to the throttle point: GB10 sags its clock a few percent and keeps
  // heating (run 1, 2026-10-18: onset 79 C, 86 C reached), so a fit that stops at
  // the onset underestimates the steady temperature and overstates the duty.
  std::vector<std::pair<double, double>> pts;
  for (const TelemetrySample& s : load)
    if (s.temp_c >= 0) pts.push_back({s.t_s - t0, s.temp_c});
  if (pts.size() < 4 || p.ambient_c < 0) return p;
  double tmax = p.ambient_c;
  for (const auto& q : pts) tmax = std::max(tmax, q.second);
  if (tmax - p.ambient_c < 0.5) {  // no measurable rise
    p.steady_c = tmax;
    p.tau_s = 0;
    return p;
  }
  double best_err = INFINITY;
  for (double steady = tmax + 0.1; steady <= tmax + 60; steady += 0.1) {
    double sxy = 0, sxx = 0;
    for (const auto& [t, T] : pts) {
      const double y = std::log((steady - T) / (steady - p.ambient_c));
      sxy += t * y, sxx += t * t;
    }
    if (sxx <= 0 || sxy >= 0) continue;
    const double tau = -sxx / sxy;
    double err = 0;
    for (const auto& [t, T] : pts) {
      const double f = steady - (steady - p.ambient_c) * std::exp(-t / tau);
      err += (f - T) * (f - T);
    }
    if (err < best_err) best_err = err, p.steady_c = steady, p.tau_s = tau;
  }
  return p;
}

double sustained_duty(const ThrottleProfile& p) {
  if (p.time_to_throttle_s < 0 || p.throttle_c < 0 || p.steady_c < 0 || p.ambient_c < 0) return 1.0;
  const double rise = p.steady_c - p.ambient_c;
  if (rise <= 0) return 1.0;
  return std::clamp((p.throttle_c - p.margin_c - p.ambient_c) / rise, 0.0, 1.0);
}

int sustained_concurrency(const ThrottleProfile& p, int requested) {
  return std::max(1, int(std::floor(sustained_duty(p) * requested)));
}

std::string to_json(const ThrottleProfile& p) {
  char buf[640];
  std::snprintf(buf, sizeof buf,
                "{\"schema\":\"aegis-throttle/1\",\"device\":\"%s\",\"ambient_c\":%.2f,\"throttle_c\":%.2f,"
                "\"time_to_throttle_s\":%.2f,\"recorded_s\":%.2f,\"steady_c\":%.2f,\"tau_s\":%.2f,\"margin_c\":%.2f,"
                "\"cause\":\"%s\",\"clock_ref_mhz\":%.0f,\"clock_sag\":%.3f,\"sustained_duty\":%.4f}",
                p.device.c_str(), p.ambient_c, p.throttle_c, p.time_to_throttle_s, p.recorded_s, p.steady_c, p.tau_s,
                p.margin_c, p.cause.c_str(), p.clock_ref_mhz, p.clock_sag, sustained_duty(p));
  return buf;
}

bool from_json(const std::string& text, ThrottleProfile* p) {
  json::Value v;
  if (!json::Parser(text).parse(&v) || v.kind != json::Value::Kind::Object) return false;
  ThrottleProfile q;
  q.device = v["device"].is_string() ? v["device"].str : "";
  q.ambient_c = v["ambient_c"].number_or(-1);
  q.throttle_c = v["throttle_c"].number_or(-1);
  q.time_to_throttle_s = v["time_to_throttle_s"].number_or(-1);
  q.recorded_s = v["recorded_s"].number_or(0);
  q.steady_c = v["steady_c"].number_or(-1);
  q.tau_s = v["tau_s"].number_or(-1);
  q.margin_c = v["margin_c"].number_or(3);
  q.cause = v["cause"].is_string() ? v["cause"].str : "";
  q.clock_ref_mhz = v["clock_ref_mhz"].number_or(-1);
  q.clock_sag = v["clock_sag"].number_or(0.03);
  *p = q;
  return true;
}

}  // namespace aegis::hw
