#pragma once
// Hardware discovery (aegis-hw step 2): what Core runs on, as a struct and as JSON.
//
// Sources, per field (every result says where its data came from):
//   GPU    cudaDeviceProp; NVML (dlopen libnvidia-ml.so.1) for name, max clocks and the
//          power limit; without NVML (Tegra, or no driver library) the Tegra sysfs
//          fallback: devfreq for the max GPU clock, the device-tree compatible string
//          for the GPU when CUDA is not queried.
//   CPU    /sys/devices/system/cpu: cluster topology, per-core MIDR, cpu_capacity,
//          cpufreq min/max. ISA features from getauxval(AT_HWCAP / AT_HWCAP2) on the
//          live system, from /proc/cpuinfo "Features" under a fixture root.
//   Memory /proc/meminfo; the GPU's totalGlobalMem.
//   Thermal /sys/class/thermal/thermal_zone*.
//   Video  NVENC: NVML encoder capacity per codec. NVDEC: libnvcuvid's
//          cuvidGetDecoderCaps per codec (dlopen). Tegra: the engines' device nodes
//          (presence only; codecs need the Multimedia API).
// A field that no source could read is reported as unknown (-1 / empty / "unavailable"),
// never guessed.
//
// Paths are read under DiscoverOptions::root, so tests run discovery against
// recorded sysfs trees (tests/fixtures/hw/).
#include <cstdint>
#include <string>
#include <vector>

namespace aegis::hw {

struct GpuInfo {
  bool present = false;
  int device = -1;
  std::string name;              // NVML or cudaDeviceProp; Tegra fixture: device-tree model
  int arch = 0;                  // major * 10 + minor (0: unknown)
  int sm_count = -1;
  int64_t smem_per_sm = -1, smem_per_block = -1, smem_per_block_optin = -1;
  int regs_per_sm = -1;
  int64_t l2_bytes = -1;
  int64_t memory_bytes = -1;     // totalGlobalMem
  bool unified = false;          // integrated: device memory is host memory
  bool tma_hardware = false;     // arch >= 90 (cp.async.bulk.tensor)
  bool tma_built = false;        // this build compiled the TMA tier (AEGIS_TMA)
  int max_sm_clock_mhz = -1;
  int max_mem_clock_mhz = -1;
  double power_limit_w = -1;     // enforced limit; -1: not readable
  std::string source;            // "cuda", "sysfs", "none"
  std::string clock_source;      // "nvml", "sysfs:<path>", "unavailable: <why>"
  std::string power_source;
  bool nvml = false;             // NVML loaded and answered for this device
  std::string nvml_error;
};

struct CoreType {
  std::string name;              // "Cortex-X925", ..., or "arm:0x41/0xd85" when unknown
  uint32_t implementer = 0, part = 0;
  std::vector<int> cpus;
  int64_t min_khz = -1, max_khz = -1;  // cpufreq cpuinfo_{min,max}_freq
  int capacity = -1;                   // cpu_capacity (max over its cpus)
  bool performance = false;            // the highest max_khz (then capacity) class
};

struct CpuCluster {
  int id = -1;                   // topology/cluster_id
  std::vector<int> cpus;
};

struct CpuIsa {
  bool neon = false, fp16 = false, dotprod = false, sve = false, sve2 = false, i8mm = false, bf16 = false;
  uint64_t hwcap = 0, hwcap2 = 0;  // raw auxv words (0 when read from cpuinfo)
  std::string source;              // "getauxval", "cpuinfo", "unavailable: <why>"
};

struct CpuInfo {
  std::string arch;              // "aarch64", ...
  int logical_cpus = 0;
  std::vector<CpuCluster> clusters;
  std::vector<CoreType> core_types;
  CpuIsa isa;
};

struct MemoryInfo {
  int64_t total_bytes = -1, available_bytes = -1;
  std::string source;
};

struct ThermalZone {
  int index = -1;
  std::string type;
  double temp_c = 0;
};

struct VideoEngine {
  bool present = false;
  int engines = -1;              // count when known
  std::vector<std::string> codecs;
  std::string source;            // "nvml", "nvcuvid", "devnode:<path>", "unavailable: <why>"
};

struct VideoInfo {
  VideoEngine nvdec, nvenc;
};

struct HwInfo {
  std::string host;
  GpuInfo gpu;
  CpuInfo cpu;
  MemoryInfo memory;
  std::vector<ThermalZone> thermal;
  VideoInfo video;
};

struct DiscoverOptions {
  std::string root = "/";        // sysfs / procfs / dev prefix (a fixture directory in tests)
  int device = 0;                // CUDA device
  bool cuda = true;              // query the CUDA runtime (false: sysfs only)
  bool nvml = true;              // load NVML (false: the no-NVML path)
  bool video = true;             // query NVENC / NVDEC
  bool auxv = true;              // ISA from getauxval (false: from <root>/proc/cpuinfo)
};

// Never fails as a whole: what cannot be read is marked unknown with its reason.
HwInfo discover(const DiscoverOptions& opt = {});
std::string to_json(const HwInfo& info);

// Core name for an (implementer, part) MIDR pair, or "" when not in the table.
std::string core_name(uint32_t implementer, uint32_t part);

}  // namespace aegis::hw
