// Hardware discovery (include/aegis/hw.h). Host code only: CUDA runtime, NVML
// and libnvcuvid are reached at run time (dlopen), sysfs / procfs through
// DiscoverOptions::root so that tests can use recorded trees.
#include "aegis/hw.h"

#include <cuda_runtime.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <sstream>

#include "device.h"
#include "profile.h"

namespace aegis::hw {
namespace {

// ---------------------------------------------------------------------------
// Files under the root.

std::string join(const std::string& root, const std::string& path) {
  if (root.empty() || root == "/") return path;
  return (root.back() == '/' ? root.substr(0, root.size() - 1) : root) + path;
}

std::optional<std::string> read_file(const std::string& root, const std::string& path) {
  std::FILE* f = std::fopen(join(root, path).c_str(), "r");
  if (!f) return std::nullopt;
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

std::string trim(std::string s) {
  const char* ws = " \t\r\n";
  s.erase(0, s.find_first_not_of(ws));
  const size_t e = s.find_last_not_of(ws);
  s.erase(e == std::string::npos ? 0 : e + 1);
  // device-tree strings end in NUL
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

std::optional<long long> read_ll(const std::string& root, const std::string& path, int base = 10) {
  const auto s = read_file(root, path);
  if (!s) return std::nullopt;
  const std::string t = trim(*s);
  if (t.empty()) return std::nullopt;
  char* end = nullptr;
  const long long v = std::strtoll(t.c_str(), &end, base);
  if (end == t.c_str()) return std::nullopt;
  return v;
}

std::vector<std::string> list_dir(const std::string& root, const std::string& path) {
  std::vector<std::string> out;
  DIR* d = opendir(join(root, path).c_str());
  if (!d) return out;
  while (dirent* e = readdir(d))
    if (e->d_name[0] != '.') out.emplace_back(e->d_name);
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

bool exists(const std::string& root, const std::string& path) {
  struct stat st;
  return stat(join(root, path).c_str(), &st) == 0;
}

// "0-3,8,10-11" -> {0,1,2,3,8,10,11}
std::vector<int> parse_cpulist(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(trim(s));
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) continue;
    const size_t dash = part.find('-');
    const int a = std::atoi(part.c_str());
    const int b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
    for (int i = a; i <= b; ++i) out.push_back(i);
  }
  return out;
}

// ---------------------------------------------------------------------------
// CPU

struct CoreName {
  uint32_t implementer, part;
  const char* name;
};
// MIDR_EL1 implementer / part numbers (Arm and NVIDIA technical reference manuals).
constexpr CoreName kCores[] = {
    {0x41, 0xd05, "Cortex-A55"},  {0x41, 0xd0b, "Cortex-A76"},  {0x41, 0xd0c, "Neoverse-N1"},
    {0x41, 0xd40, "Neoverse-V1"}, {0x41, 0xd41, "Cortex-A78"},  {0x41, 0xd42, "Cortex-A78AE"},
    {0x41, 0xd44, "Cortex-X1"},   {0x41, 0xd46, "Cortex-A510"}, {0x41, 0xd47, "Cortex-A710"},
    {0x41, 0xd48, "Cortex-X2"},   {0x41, 0xd49, "Neoverse-N2"}, {0x41, 0xd4d, "Cortex-A715"},
    {0x41, 0xd4e, "Cortex-X3"},   {0x41, 0xd4f, "Neoverse-V2"}, {0x41, 0xd80, "Cortex-A520"},
    {0x41, 0xd81, "Cortex-A720"}, {0x41, 0xd82, "Cortex-X4"},   {0x41, 0xd85, "Cortex-X925"},
    {0x41, 0xd87, "Cortex-A725"}, {0x4e, 0x004, "Carmel"},
};

struct CpuRaw {
  int cpu = -1;
  int cluster = -1;
  int core_first = -1;  // lowest logical CPU of its physical core
  uint32_t implementer = 0, part = 0;
  int capacity = -1;
  long long min_khz = -1, max_khz = -1;
};

// "processor : N" blocks of /proc/cpuinfo: implementer and part per cpu, and the
// first "Features" line.
void parse_cpuinfo(const std::string& text, std::map<int, std::pair<uint32_t, uint32_t>>* midr,
                   std::string* features) {
  std::stringstream ss(text);
  std::string line;
  int cpu = -1;
  while (std::getline(ss, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
    if (key == "processor") cpu = std::atoi(val.c_str());
    else if (key == "CPU implementer" && cpu >= 0) (*midr)[cpu].first = uint32_t(std::strtoul(val.c_str(), nullptr, 0));
    else if (key == "CPU part" && cpu >= 0) (*midr)[cpu].second = uint32_t(std::strtoul(val.c_str(), nullptr, 0));
    else if (key == "Features" && features->empty()) *features = val;
  }
}

CpuIsa isa_from_features(const std::string& features) {
  CpuIsa isa;
  std::stringstream ss(features);
  std::string f;
  bool fphp = false, asimdhp = false;
  while (ss >> f) {
    if (f == "asimd") isa.neon = true;
    else if (f == "fphp") fphp = true;
    else if (f == "asimdhp") asimdhp = true;
    else if (f == "asimddp") isa.dotprod = true;
    else if (f == "sve") isa.sve = true;
    else if (f == "sve2") isa.sve2 = true;
    else if (f == "i8mm") isa.i8mm = true;
    else if (f == "bf16") isa.bf16 = true;
  }
  isa.fp16 = fphp && asimdhp;
  isa.source = "cpuinfo";
  return isa;
}

CpuIsa isa_from_auxv() {
  CpuIsa isa;
#if defined(__aarch64__)
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1 << 14)
#endif
  isa.hwcap = getauxval(AT_HWCAP);
  isa.hwcap2 = getauxval(AT_HWCAP2);
  isa.neon = isa.hwcap & HWCAP_ASIMD;
  isa.fp16 = (isa.hwcap & HWCAP_FPHP) && (isa.hwcap & HWCAP_ASIMDHP);
  isa.dotprod = isa.hwcap & HWCAP_ASIMDDP;
  isa.sve = isa.hwcap & HWCAP_SVE;
  isa.sve2 = isa.hwcap2 & HWCAP2_SVE2;
  isa.i8mm = isa.hwcap2 & HWCAP2_I8MM;
  isa.bf16 = isa.hwcap2 & HWCAP2_BF16;
  isa.source = "getauxval";
#else
  isa.source = "unavailable: not aarch64";
#endif
  return isa;
}

CpuInfo discover_cpu(const DiscoverOptions& opt) {
  CpuInfo c;
  if (opt.root == "/") {
    utsname u{};
    if (uname(&u) == 0) c.arch = u.machine;
  } else if (const auto m = read_file(opt.root, "/uname-m")) {
    c.arch = trim(*m);  // fixtures record `uname -m` here
  }
  std::map<int, std::pair<uint32_t, uint32_t>> midr_info;
  std::string features;
  if (const auto ci = read_file(opt.root, "/proc/cpuinfo")) parse_cpuinfo(*ci, &midr_info, &features);

  const std::string base = "/sys/devices/system/cpu";
  std::vector<int> cpus;
  if (const auto online = read_file(opt.root, base + "/online")) cpus = parse_cpulist(*online);
  if (cpus.empty())
    for (const std::string& d : list_dir(opt.root, base))
      if (d.size() > 3 && d.compare(0, 3, "cpu") == 0 && std::isdigit(static_cast<unsigned char>(d[3])))
        cpus.push_back(std::atoi(d.c_str() + 3));
  std::sort(cpus.begin(), cpus.end());
  c.logical_cpus = int(cpus.size());

  std::vector<CpuRaw> raw;
  for (int cpu : cpus) {
    const std::string d = base + "/cpu" + std::to_string(cpu);
    CpuRaw r;
    r.cpu = cpu;
    r.cluster = int(read_ll(opt.root, d + "/topology/cluster_id").value_or(-1));
    if (r.cluster < 0) r.cluster = int(read_ll(opt.root, d + "/topology/physical_package_id").value_or(-1));
    // SMT: the core's first logical CPU represents it (core_cpus_list, older kernels
    // thread_siblings_list; neither: its own core).
    r.core_first = cpu;
    for (const char* f : {"/topology/core_cpus_list", "/topology/thread_siblings_list"})
      if (const auto l = read_file(opt.root, d + f)) {
        const std::vector<int> sib = parse_cpulist(*l);
        if (!sib.empty()) r.core_first = *std::min_element(sib.begin(), sib.end());
        break;
      }
    r.capacity = int(read_ll(opt.root, d + "/cpu_capacity").value_or(-1));
    r.min_khz = read_ll(opt.root, d + "/cpufreq/cpuinfo_min_freq").value_or(-1);
    r.max_khz = read_ll(opt.root, d + "/cpufreq/cpuinfo_max_freq").value_or(-1);
    if (const auto m = read_ll(opt.root, d + "/regs/identification/midr_el1", 16)) {
      r.implementer = uint32_t((*m >> 24) & 0xff);
      r.part = uint32_t((*m >> 4) & 0xfff);
    } else if (midr_info.count(cpu)) {
      r.implementer = midr_info[cpu].first;
      r.part = midr_info[cpu].second;
    }
    raw.push_back(r);
  }

  std::map<int, CpuCluster> clusters;
  std::map<std::pair<uint32_t, uint32_t>, CoreType> types;
  for (const CpuRaw& r : raw) {
    CpuCluster& cl = clusters[r.cluster];
    cl.id = r.cluster;
    cl.cpus.push_back(r.cpu);
    CoreType& t = types[{r.implementer, r.part}];
    t.implementer = r.implementer;
    t.part = r.part;
    t.cpus.push_back(r.cpu);
    if (r.core_first == r.cpu) t.physical_cpus.push_back(r.cpu);
    t.capacity = std::max(t.capacity, r.capacity);
    t.max_khz = std::max<int64_t>(t.max_khz, r.max_khz);
    if (r.min_khz >= 0) t.min_khz = t.min_khz < 0 ? r.min_khz : std::min<int64_t>(t.min_khz, r.min_khz);
  }
  for (auto& [k, cl] : clusters) c.clusters.push_back(cl);
  for (auto& [k, t] : types) {
    t.name = core_name(t.implementer, t.part);
    if (t.name.empty()) {
      char buf[48];
      std::snprintf(buf, sizeof buf, "arm:0x%02x/0x%03x", t.implementer, t.part);
      t.name = buf;
    }
    c.core_types.push_back(t);
  }
  // Performance cores: the class with the highest max frequency, then capacity.
  auto rank = [](const CoreType& t) { return std::make_pair(t.max_khz, t.capacity); };
  if (!c.core_types.empty()) {
    const auto best = rank(*std::max_element(c.core_types.begin(), c.core_types.end(),
                                             [&](const CoreType& a, const CoreType& b) { return rank(a) < rank(b); }));
    for (CoreType& t : c.core_types) t.performance = rank(t) == best;
  }

  if (opt.auxv && opt.root == "/") {
    c.isa = isa_from_auxv();
  } else if (!features.empty()) {
    c.isa = isa_from_features(features);
  } else {
    c.isa.source = "unavailable: no getauxval (fixture) and no cpuinfo Features";
  }
  return c;
}

// ---------------------------------------------------------------------------
// Memory, thermal

MemoryInfo discover_memory(const DiscoverOptions& opt) {
  MemoryInfo m;
  const auto text = read_file(opt.root, "/proc/meminfo");
  if (!text) {
    m.source = "unavailable: no /proc/meminfo";
    return m;
  }
  std::stringstream ss(*text);
  std::string key;
  long long kb;
  std::string unit;
  while (ss >> key >> kb) {
    std::getline(ss, unit);
    if (key == "MemTotal:") m.total_bytes = kb * 1024;
    else if (key == "MemAvailable:") m.available_bytes = kb * 1024;
  }
  m.source = "/proc/meminfo";
  return m;
}

std::vector<ThermalZone> discover_thermal(const DiscoverOptions& opt) {
  std::vector<ThermalZone> out;
  const std::string base = "/sys/class/thermal";
  for (const std::string& d : list_dir(opt.root, base)) {
    if (d.compare(0, 12, "thermal_zone") != 0) continue;
    ThermalZone z;
    z.index = std::atoi(d.c_str() + 12);
    z.type = trim(read_file(opt.root, base + "/" + d + "/type").value_or(""));
    const auto t = read_ll(opt.root, base + "/" + d + "/temp");
    if (!t) continue;  // a zone whose sensor cannot be read
    z.temp_c = double(*t) / 1000.0;
    out.push_back(z);
  }
  std::sort(out.begin(), out.end(), [](const ThermalZone& a, const ThermalZone& b) { return a.index < b.index; });
  return out;
}

// ---------------------------------------------------------------------------
// NVML (dlopen; nvml.h types as plain ints: nvmlReturn_t 0 = success)

struct Nvml {
  void* lib = nullptr;
  using Dev = void*;
  int (*init)() = nullptr;
  int (*shutdown)() = nullptr;
  const char* (*err)(int) = nullptr;
  int (*by_bus)(const char*, Dev*) = nullptr;
  int (*by_index)(unsigned, Dev*) = nullptr;
  int (*name)(Dev, char*, unsigned) = nullptr;
  int (*max_clock)(Dev, int, unsigned*) = nullptr;        // nvmlClockType_t: 1 SM, 2 MEM
  int (*power_limit)(Dev, unsigned*) = nullptr;           // enforced limit, mW
  int (*enc_capacity)(Dev, int, unsigned*) = nullptr;     // nvmlEncoderType_t: 0 H264, 1 HEVC, 2 AV1
  int (*dec_util)(Dev, unsigned*, unsigned*) = nullptr;
  std::string error;

  bool load() {
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      error = "libnvidia-ml.so.1 not found";
      return false;
    }
    auto sym = [&](auto& fn, const char* s) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, s));
      return fn != nullptr;
    };
    if (!(sym(init, "nvmlInit_v2") && sym(shutdown, "nvmlShutdown") && sym(err, "nvmlErrorString") &&
          sym(by_bus, "nvmlDeviceGetHandleByPciBusId_v2") && sym(by_index, "nvmlDeviceGetHandleByIndex_v2") &&
          sym(name, "nvmlDeviceGetName") && sym(max_clock, "nvmlDeviceGetMaxClockInfo"))) {
      error = "libnvidia-ml.so.1 lacks a required entry point";
      return false;
    }
    sym(power_limit, "nvmlDeviceGetEnforcedPowerLimit");
    sym(enc_capacity, "nvmlDeviceGetEncoderCapacity");
    sym(dec_util, "nvmlDeviceGetDecoderUtilization");
    return true;
  }
  ~Nvml() {
    if (lib) dlclose(lib);
  }
};

// ---------------------------------------------------------------------------
// libnvcuvid: cuvidGetDecoderCaps. The struct is CUVIDDECODECAPS from the Video
// Codec SDK's cuviddec.h (not installed here; declared from the SDK's
// documentation), padded on both ends of what the library may write. Results are
// checked for plausibility, so a layout mismatch reads as "unverified", never as
// a codec list.
struct CuvidDecodeCaps {
  int eCodecType;           // cudaVideoCodec
  int eChromaFormat;        // cudaVideoChromaFormat (1: 4:2:0)
  unsigned nBitDepthMinus8;
  unsigned reserved1[3];
  unsigned char bIsSupported;
  unsigned char nNumNVDECs;
  unsigned short nOutputFormatMask;
  unsigned nMaxWidth, nMaxHeight, nMaxMBCount;
  unsigned short nMinWidth, nMinHeight;
  unsigned char bIsHistogramSupported, nCounterBitDepth;
  unsigned short nMaxHistogramBins;
  unsigned reserved3[10];
  unsigned char slack[256];  // in case the installed library's struct is larger
};

VideoEngine nvdec_from_nvcuvid(int device) {
  VideoEngine e;
  void* lib = dlopen("libnvcuvid.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    e.source = "unavailable: libnvcuvid.so.1 not found";
    return e;
  }
  using Fn = int (*)(CuvidDecodeCaps*);
  auto caps_fn = reinterpret_cast<Fn>(dlsym(lib, "cuvidGetDecoderCaps"));
  if (!caps_fn) {
    dlclose(lib);
    e.source = "unavailable: libnvcuvid.so.1 lacks cuvidGetDecoderCaps";
    return e;
  }
  // The runtime's primary context, current on this thread.
  if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    cudaGetLastError();
    dlclose(lib);
    e.source = "unavailable: no CUDA context for cuvidGetDecoderCaps";
    return e;
  }
  struct Codec {
    int id;
    const char* name;
  };
  // cudaVideoCodec values (cuviddec.h).
  constexpr Codec kCodecs[] = {{1, "MPEG2"}, {2, "MPEG4"}, {3, "VC1"}, {4, "H264"},
                               {8, "HEVC"},  {9, "VP8"},   {10, "VP9"}, {11, "AV1"}};
  bool plausible = true, any_answer = false;
  int engines = -1;
  for (const Codec& c : kCodecs) {
    CuvidDecodeCaps caps;
    std::memset(&caps, 0, sizeof caps);
    caps.eCodecType = c.id;
    caps.eChromaFormat = 1;
    caps.nBitDepthMinus8 = 0;
    if (caps_fn(&caps) != 0) continue;
    any_answer = true;
    if (caps.bIsSupported > 1) plausible = false;
    if (caps.bIsSupported == 1) {
      if (caps.nMaxWidth < 16 || caps.nMaxWidth > 65536 || caps.nMaxHeight < 16 || caps.nMaxHeight > 65536)
        plausible = false;
      e.codecs.push_back(c.name);
      engines = std::max(engines, int(caps.nNumNVDECs));
    }
  }
  dlclose(lib);
  if (!any_answer) {
    e.codecs.clear();
    e.source = "unavailable: cuvidGetDecoderCaps failed for every codec";
  } else if (!plausible) {
    e.codecs.clear();
    e.source = "unverified: implausible cuvidGetDecoderCaps results (struct layout mismatch?)";
  } else {
    e.present = !e.codecs.empty();
    e.engines = engines > 0 ? engines : -1;
    e.source = "nvcuvid";
  }
  return e;
}

// Tegra: engine device nodes (presence only).
VideoEngine engine_from_devnodes(const DiscoverOptions& opt, std::initializer_list<const char*> nodes) {
  VideoEngine e;
  for (const char* n : nodes) {
    if (exists(opt.root, n)) {
      e.present = true;
      e.source = std::string("devnode:") + n + " (codecs need the Jetson Multimedia API)";
      return e;
    }
  }
  e.source = "unavailable: no NVML and no engine device node";
  return e;
}

// ---------------------------------------------------------------------------
// GPU

// Tegra sysfs: the GPU's devfreq node (17000000.gpu on Orin, *.ga10b / *.gv11b on
// older L4T). Returns the node directory under /sys/class/devfreq, or "".
std::string tegra_gpu_devfreq(const DiscoverOptions& opt) {
  for (const std::string& d : list_dir(opt.root, "/sys/class/devfreq"))
    if (d.find("gpu") != std::string::npos || d.find("ga10b") != std::string::npos ||
        d.find("gv11b") != std::string::npos)
      return "/sys/class/devfreq/" + d;
  return "";
}

int arch_from_compatible(const std::string& compat) {
  if (compat.find("ga10b") != std::string::npos) return 87;  // Orin
  if (compat.find("gv11b") != std::string::npos) return 72;  // Xavier
  return 0;
}

void gpu_sysfs_clocks(const DiscoverOptions& opt, GpuInfo* g) {
  const std::string df = tegra_gpu_devfreq(opt);
  if (df.empty()) {
    if (g->clock_source.empty()) g->clock_source = "unavailable: no NVML and no GPU devfreq node";
    return;
  }
  if (const auto hz = read_ll(opt.root, df + "/max_freq")) {
    g->max_sm_clock_mhz = int(*hz / 1000000);
    g->clock_source = "sysfs:" + df + "/max_freq";
  }
}

GpuInfo discover_gpu(const DiscoverOptions& opt, VideoInfo* video) {
  GpuInfo g;
  g.device = opt.device;
#if defined(AEGIS_TMA)
  g.tma_built = true;
#endif
  cudaDeviceProp prop{};
  bool have_prop = false;
  if (opt.cuda) {
    int n = 0;
    if (cudaGetDeviceCount(&n) == cudaSuccess && opt.device >= 0 && opt.device < n) {
      DeviceLimits d;
      if (query(opt.device, &d) == cudaSuccess && cudaGetDeviceProperties(&prop, opt.device) == cudaSuccess) {
        have_prop = true;
        g.present = true;
        g.source = "cuda";
        g.name = prop.name;
        g.arch = d.arch;
        g.sm_count = d.sm_count;
        g.smem_per_sm = int64_t(d.smem_per_sm);
        g.smem_per_block = int64_t(d.smem_per_block);
        g.smem_per_block_optin = int64_t(d.smem_per_block_optin);
        g.regs_per_sm = d.regs_per_sm;
        g.l2_bytes = int64_t(d.l2_bytes);
        g.memory_bytes = int64_t(d.global_mem_bytes);
        g.unified = d.unified;
      }
    }
    cudaGetLastError();
  }
  if (!g.present) {
    // No CUDA (or not asked): what Tegra sysfs says about the GPU.
    const std::string df = tegra_gpu_devfreq(opt);
    if (!df.empty()) {
      g.present = true;
      g.source = "sysfs";
      const std::string model = trim(read_file(opt.root, "/proc/device-tree/model").value_or(""));
      g.arch = arch_from_compatible(read_file(opt.root, df + "/device/of_node/compatible").value_or(""));
      g.name = model.empty() ? df.substr(df.rfind('/') + 1) : model + " GPU";
      g.unified = true;  // Tegra iGPU
    } else {
      g.source = "none";
    }
  }
  g.tma_hardware = g.arch >= 90;

  Nvml nv;
  void* dev = nullptr;
  if (opt.nvml && g.present && g.source == "cuda") {
    if (!nv.load()) {
      g.nvml_error = nv.error;
    } else if (const int r = nv.init(); r != 0) {
      g.nvml_error = std::string("nvmlInit: ") + nv.err(r);
    } else {
      char bus[32] = {};
      std::snprintf(bus, sizeof bus, "%08x:%02x:%02x.0", unsigned(prop.pciDomainID), unsigned(prop.pciBusID),
                    unsigned(prop.pciDeviceID));
      int r2 = have_prop ? nv.by_bus(bus, &dev) : -1;
      if (r2 != 0) r2 = nv.by_index(unsigned(opt.device), &dev);
      if (r2 != 0) {
        g.nvml_error = std::string("nvmlDeviceGetHandle: ") + nv.err(r2);
        dev = nullptr;
        nv.shutdown();
      }
    }
  } else if (!opt.nvml) {
    g.nvml_error = "not queried (DiscoverOptions::nvml = false)";
  }
  if (dev) {
    g.nvml = true;
    char name[96] = {};
    if (nv.name(dev, name, sizeof name) == 0 && name[0]) g.name = name;
    unsigned mhz = 0;
    if (const int r = nv.max_clock(dev, 1, &mhz); r == 0) {
      g.max_sm_clock_mhz = int(mhz);
      g.clock_source = "nvml";
    } else {
      g.clock_source = std::string("unavailable: nvmlDeviceGetMaxClockInfo(SM): ") + nv.err(r);
    }
    if (nv.max_clock(dev, 2, &mhz) == 0) g.max_mem_clock_mhz = int(mhz);
    unsigned mw = 0;
    if (!nv.power_limit) {
      g.power_source = "unavailable: no nvmlDeviceGetEnforcedPowerLimit";
    } else if (const int r = nv.power_limit(dev, &mw); r == 0) {
      g.power_limit_w = mw / 1000.0;
      g.power_source = "nvml";
    } else {
      g.power_source = std::string("unavailable: nvmlDeviceGetEnforcedPowerLimit: ") + nv.err(r);
    }
    if (opt.video) {
      VideoEngine& enc = video->nvenc;
      if (nv.enc_capacity) {
        constexpr std::pair<int, const char*> kEnc[] = {{0, "H264"}, {1, "HEVC"}, {2, "AV1"}};
        bool answered = false;
        for (const auto& [id, cname] : kEnc) {
          unsigned cap = 0;
          if (nv.enc_capacity(dev, id, &cap) == 0) {
            answered = true;
            if (cap > 0) enc.codecs.push_back(cname);
          }
        }
        enc.present = !enc.codecs.empty();
        enc.source = answered ? "nvml" : "unavailable: nvmlDeviceGetEncoderCapacity not supported";
      } else {
        enc.source = "unavailable: no nvmlDeviceGetEncoderCapacity";
      }
    }
    nv.shutdown();
  }
  if (g.max_sm_clock_mhz < 0) gpu_sysfs_clocks(opt, &g);
  if (g.power_source.empty()) g.power_source = "unavailable: no NVML (Tegra power limits are not in sysfs)";
  if (g.clock_source.empty()) g.clock_source = "unavailable";

  if (opt.video) {
    if (g.source == "cuda") video->nvdec = nvdec_from_nvcuvid(opt.device);
    const bool tegra = !tegra_gpu_devfreq(opt).empty();
    if (tegra && !video->nvdec.present)
      video->nvdec = engine_from_devnodes(opt, {"/dev/v4l2-nvdec", "/dev/nvhost-nvdec", "/dev/nvhost-nvdec1"});
    if (tegra && video->nvenc.source.empty())
      video->nvenc = engine_from_devnodes(opt, {"/dev/v4l2-nvenc", "/dev/nvhost-msenc", "/dev/nvhost-nvenc1"});
    if (video->nvenc.source.empty()) video->nvenc.source = "unavailable: " + (g.nvml_error.empty() ? "no NVML" : g.nvml_error);
    if (video->nvdec.source.empty()) video->nvdec.source = "unavailable";
  } else {
    video->nvdec.source = video->nvenc.source = "not queried (DiscoverOptions::video = false)";
  }
  return g;
}

// ---------------------------------------------------------------------------
// JSON

std::string esc(const std::string& s) {
  std::string o = "\"";
  for (unsigned char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default:
        if (ch < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", ch);
          o += b;
        } else {
          o += char(ch);
        }
    }
  }
  return o + "\"";
}

std::string b(bool v) { return v ? "true" : "false"; }

std::string ints(const std::vector<int>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
  return o + "]";
}

std::string strs(const std::vector<std::string>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + esc(v[i]);
  return o + "]";
}

std::string num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.3f", v);
  return buf;
}

std::string engine_json(const VideoEngine& e) {
  return "{\"present\":" + b(e.present) + ",\"engines\":" + std::to_string(e.engines) + ",\"codecs\":" +
         strs(e.codecs) + ",\"source\":" + esc(e.source) + "}";
}

}  // namespace

std::string core_name(uint32_t implementer, uint32_t part) {
  for (const CoreName& c : kCores)
    if (c.implementer == implementer && c.part == part) return c.name;
  return "";
}

HwInfo discover(const DiscoverOptions& opt) {
  HwInfo h;
  if (opt.root == "/") {
    char host[256] = {};
    if (gethostname(host, sizeof host - 1) == 0) h.host = host;
  } else {
    h.host = trim(read_file(opt.root, "/etc/hostname").value_or(""));
  }
  h.cpu = discover_cpu(opt);
  h.memory = discover_memory(opt);
  h.thermal = discover_thermal(opt);
  h.gpu = discover_gpu(opt, &h.video);
  if (h.gpu.source == "cuda" && cudaSetDevice(opt.device) == cudaSuccess) {
    const Profile& p = current_profile();
    h.profile.label = p.label();
    h.profile.source = p.source;
    h.profile.mismatch = p.mismatch;
  }
  cudaGetLastError();
  return h;
}

std::string to_json(const HwInfo& h) {
  const GpuInfo& g = h.gpu;
  std::string o = "{\"schema\":\"aegis-hw/1\",\"host\":" + esc(h.host) + ",\n";
  o += " \"gpu\":{\"present\":" + b(g.present) + ",\"device\":" + std::to_string(g.device) +
       ",\"name\":" + esc(g.name) + ",\"arch\":" + std::to_string(g.arch) +
       ",\"sm_count\":" + std::to_string(g.sm_count) + ",\"smem_per_sm\":" + std::to_string(g.smem_per_sm) +
       ",\"smem_per_block\":" + std::to_string(g.smem_per_block) +
       ",\"smem_per_block_optin\":" + std::to_string(g.smem_per_block_optin) +
       ",\"regs_per_sm\":" + std::to_string(g.regs_per_sm) + ",\"l2_bytes\":" + std::to_string(g.l2_bytes) +
       ",\"memory_bytes\":" + std::to_string(g.memory_bytes) + ",\"unified\":" + b(g.unified) +
       ",\"tma_hardware\":" + b(g.tma_hardware) + ",\"tma_built\":" + b(g.tma_built) +
       ",\"max_sm_clock_mhz\":" + std::to_string(g.max_sm_clock_mhz) +
       ",\"max_mem_clock_mhz\":" + std::to_string(g.max_mem_clock_mhz) +
       ",\"power_limit_w\":" + (g.power_limit_w < 0 ? std::string("null") : num(g.power_limit_w)) +
       ",\"source\":" + esc(g.source) + ",\"clock_source\":" + esc(g.clock_source) +
       ",\"power_source\":" + esc(g.power_source) + ",\"nvml\":" + b(g.nvml) +
       ",\"nvml_error\":" + esc(g.nvml_error) + "},\n";
  const CpuInfo& c = h.cpu;
  o += " \"cpu\":{\"arch\":" + esc(c.arch) + ",\"logical_cpus\":" + std::to_string(c.logical_cpus) + ",\"clusters\":[";
  for (size_t i = 0; i < c.clusters.size(); ++i)
    o += std::string(i ? "," : "") + "{\"id\":" + std::to_string(c.clusters[i].id) + ",\"cpus\":" + ints(c.clusters[i].cpus) + "}";
  o += "],\"core_types\":[";
  for (size_t i = 0; i < c.core_types.size(); ++i) {
    const CoreType& t = c.core_types[i];
    char midr[32];
    std::snprintf(midr, sizeof midr, "0x%02x/0x%03x", t.implementer, t.part);
    o += std::string(i ? "," : "") + "{\"name\":" + esc(t.name) + ",\"midr\":" + esc(midr) + ",\"cpus\":" + ints(t.cpus) + ",\"physical_cpus\":" + ints(t.physical_cpus) +
         ",\"min_khz\":" + std::to_string(t.min_khz) + ",\"max_khz\":" + std::to_string(t.max_khz) +
         ",\"capacity\":" + std::to_string(t.capacity) + ",\"performance\":" + b(t.performance) + "}";
  }
  char hw[64];
  std::snprintf(hw, sizeof hw, "\"0x%llx\",\"hwcap2\":\"0x%llx\"", (unsigned long long)c.isa.hwcap,
                (unsigned long long)c.isa.hwcap2);
  o += "],\"isa\":{\"neon\":" + b(c.isa.neon) + ",\"fp16\":" + b(c.isa.fp16) + ",\"dotprod\":" + b(c.isa.dotprod) +
       ",\"sve\":" + b(c.isa.sve) + ",\"sve2\":" + b(c.isa.sve2) + ",\"i8mm\":" + b(c.isa.i8mm) +
       ",\"bf16\":" + b(c.isa.bf16) + ",\"hwcap\":" + hw + ",\"source\":" + esc(c.isa.source) + "}},\n";
  o += " \"profile\":{\"label\":" + esc(h.profile.label) + ",\"source\":" + esc(h.profile.source) +
       ",\"mismatch\":" + esc(h.profile.mismatch) + "},\n";
  o += " \"memory\":{\"total_bytes\":" + std::to_string(h.memory.total_bytes) +
       ",\"available_bytes\":" + std::to_string(h.memory.available_bytes) + ",\"source\":" + esc(h.memory.source) + "},\n";
  o += " \"thermal\":[";
  for (size_t i = 0; i < h.thermal.size(); ++i)
    o += std::string(i ? "," : "") + "{\"zone\":" + std::to_string(h.thermal[i].index) + ",\"type\":" +
         esc(h.thermal[i].type) + ",\"temp_c\":" + num(h.thermal[i].temp_c) + "}";
  o += "],\n \"video\":{\"nvdec\":" + engine_json(h.video.nvdec) + ",\"nvenc\":" + engine_json(h.video.nvenc) + "}}\n";
  return o;
}

}  // namespace aegis::hw
