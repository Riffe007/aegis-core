// Hardware discovery (include/aegis/hw.h), ctest hw_discover.
//   1. Synthetic Orin Nano fixture (tests/fixtures/hw/orin-nano-synthetic): the Tegra
//      path with no CUDA and no NVML. Clusters, core type, ISA from cpuinfo, GPU from
//      devfreq + device tree, NVDEC from its device node, no NVENC, memory, thermal.
//   2. Recorded Spark fixture (tests/fixtures/hw/spark-gb10): the CPU part parsed from
//      the recording equals live discovery, and the ISA from cpuinfo "Features" equals
//      the ISA from getauxval (both paths agree on the same machine).
//   3. Live: GPU fields equal cudaDeviceProp; the no-NVML path keeps the CUDA fields
//      and marks clocks / power / NVENC unavailable; no CUDA leaves the GPU absent here.
//   4. An empty root: everything unknown, nothing invented, no crash.
//   5. JSON: every top-level key, balanced braces.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "aegis/hw.h"

#ifndef AEGIS_FIXTURE_DIR
#error "AEGIS_FIXTURE_DIR (tests/fixtures/hw) must be defined by CMake"
#endif

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

using aegis::hw::DiscoverOptions;
using aegis::hw::HwInfo;

DiscoverOptions fixture(const char* name) {
  DiscoverOptions o;
  o.root = std::string(AEGIS_FIXTURE_DIR) + "/" + name;
  o.cuda = false;
  o.nvml = false;
  o.auxv = false;
  return o;
}

bool balanced(const std::string& j) {
  int depth = 0;
  bool in_str = false;
  for (size_t i = 0; i < j.size(); ++i) {
    const char c = j[i];
    if (in_str) {
      if (c == '\\') ++i;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') ++depth;
    else if (c == '}' || c == ']') --depth;
    if (depth < 0) return false;
  }
  return depth == 0 && !in_str;
}

void orin_nano() {
  const HwInfo h = aegis::hw::discover(fixture("orin-nano-synthetic"));
  const auto& c = h.cpu;
  expect(c.arch == "aarch64" && c.logical_cpus == 6, "orin: 6 aarch64 cpus");
  expect(c.clusters.size() == 2 && c.clusters[0].cpus == std::vector<int>{0, 1, 2, 3} &&
             c.clusters[1].cpus == std::vector<int>{4, 5},
         "orin: clusters {0-3} {4-5}");
  expect(c.core_types.size() == 1 && c.core_types[0].name == "Cortex-A78AE" && c.core_types[0].performance &&
             c.core_types[0].min_khz == 115200 && c.core_types[0].max_khz == 1510400,
         "orin: one core type, Cortex-A78AE, 115200-1510400 kHz, performance");
  expect(c.isa.source == "cpuinfo" && c.isa.neon && c.isa.fp16 && c.isa.dotprod && !c.isa.sve && !c.isa.sve2 &&
             !c.isa.i8mm && !c.isa.bf16,
         "orin: ISA from cpuinfo: NEON, fp16, dotprod; no SVE, SVE2, i8mm, bf16");
  const auto& g = h.gpu;
  expect(g.present && g.source == "sysfs" && g.arch == 87 && g.unified && !g.tma_hardware,
         "orin: GPU from sysfs: ga10b -> sm_87, unified, no TMA");
  expect(g.max_sm_clock_mhz == 624 && g.clock_source == "sysfs:/sys/class/devfreq/17000000.gpu/max_freq",
         "orin: max GPU clock 624 MHz from devfreq");
  expect(!g.nvml && g.power_limit_w < 0 && g.power_source.rfind("unavailable", 0) == 0,
         "orin: no NVML: power limit unavailable, not guessed");
  expect(g.sm_count == -1 && g.smem_per_sm == -1 && g.l2_bytes == -1, "orin: no CUDA: SM count, smem, L2 unknown");
  expect(h.video.nvdec.present && h.video.nvdec.source.rfind("devnode:/dev/v4l2-nvdec", 0) == 0 &&
             h.video.nvdec.codecs.empty(),
         "orin: NVDEC present from its device node, codecs unknown");
  expect(!h.video.nvenc.present, "orin: no NVENC");
  expect(h.memory.total_bytes == 7802616ll * 1024 && h.memory.available_bytes == 6512304ll * 1024,
         "orin: memory from meminfo");
  expect(h.thermal.size() == 9 && h.thermal[0].type == "cpu-thermal" && h.thermal[8].type == "tj-thermal" &&
             h.thermal[1].temp_c == 45.5,
         "orin: 9 thermal zones");
  expect(h.host == "orin-nano-synthetic", "orin: host from etc/hostname");
  const std::string j = aegis::hw::to_json(h);
  expect(balanced(j) && j.find("\"schema\":\"aegis-hw/1\"") != std::string::npos, "orin: JSON well formed");
}

void spark_recorded_vs_live() {
  const HwInfo rec = aegis::hw::discover(fixture("spark-gb10"));
  DiscoverOptions live_opt;
  live_opt.cuda = false;
  live_opt.video = false;
  const HwInfo live = aegis::hw::discover(live_opt);
  // Only meaningful on the machine it was recorded on.
  if (live.host != rec.host) {
    std::printf("skip  spark fixture: recorded on %s, running on %s (parse-only checks)\n", rec.host.c_str(),
                live.host.c_str());
  } else {
    bool same_types = rec.cpu.core_types.size() == live.cpu.core_types.size();
    for (size_t i = 0; same_types && i < rec.cpu.core_types.size(); ++i)
      same_types = rec.cpu.core_types[i].name == live.cpu.core_types[i].name &&
                   rec.cpu.core_types[i].cpus == live.cpu.core_types[i].cpus &&
                   rec.cpu.core_types[i].max_khz == live.cpu.core_types[i].max_khz &&
                   rec.cpu.core_types[i].performance == live.cpu.core_types[i].performance;
    expect(same_types && rec.cpu.clusters.size() == live.cpu.clusters.size() &&
               rec.cpu.logical_cpus == live.cpu.logical_cpus,
           "spark: recorded CPU topology equals live discovery");
    const auto& a = rec.cpu.isa;
    const auto& b = live.cpu.isa;
    expect(a.source == "cpuinfo" && b.source == "getauxval" && a.neon == b.neon && a.fp16 == b.fp16 &&
               a.dotprod == b.dotprod && a.sve == b.sve && a.sve2 == b.sve2 && a.i8mm == b.i8mm && a.bf16 == b.bf16,
           "spark: ISA from cpuinfo Features equals ISA from getauxval");
    expect(rec.memory.total_bytes == live.memory.total_bytes, "spark: recorded MemTotal equals live");
  }
  // Parse-only expectations of the recording (GB10: 10 Cortex-X925 + 10 Cortex-A725, two clusters).
  int perf = 0;
  for (const auto& t : rec.cpu.core_types) perf += t.performance ? int(t.cpus.size()) : 0;
  expect(rec.cpu.clusters.size() == 2 && rec.cpu.core_types.size() == 2 && perf == 10,
         "spark fixture: 2 clusters, 2 core types, 10 performance cores");
  expect(rec.cpu.isa.sve2 && rec.cpu.isa.i8mm && rec.cpu.isa.bf16, "spark fixture: SVE2, i8mm, bf16");
  expect(!rec.gpu.present && rec.gpu.source == "none", "spark fixture without CUDA: no GPU (no Tegra devfreq)");
}

void live_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
    cudaGetLastError();
    expect(false, "live: no CUDA device");
    return;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    expect(false, "live: cudaGetDeviceProperties");
    return;
  }
  const HwInfo h = aegis::hw::discover();
  const auto& g = h.gpu;
  expect(g.present && g.source == "cuda" && g.arch == p.major * 10 + p.minor && g.sm_count == p.multiProcessorCount &&
             g.smem_per_sm == int64_t(p.sharedMemPerMultiprocessor) &&
             g.smem_per_block_optin == int64_t(p.sharedMemPerBlockOptin) && g.regs_per_sm == p.regsPerMultiprocessor &&
             g.l2_bytes == p.l2CacheSize && g.memory_bytes == int64_t(p.totalGlobalMem) &&
             g.unified == (p.integrated != 0) && g.tma_hardware == (p.major >= 9),
         "live: GPU fields equal cudaDeviceProp (sm_" + std::to_string(g.arch) + ", " + std::to_string(g.sm_count) +
             " SMs)");
  expect(g.nvml ? g.clock_source == "nvml" && g.max_sm_clock_mhz > 0 : !g.nvml_error.empty(),
         "live: NVML " + std::string(g.nvml ? "answered (max SM clock " + std::to_string(g.max_sm_clock_mhz) + " MHz)"
                                            : "unavailable: " + g.nvml_error));
  expect(g.power_limit_w > 0 || g.power_source.rfind("unavailable", 0) == 0,
         "live: power limit " + (g.power_limit_w > 0 ? std::to_string(g.power_limit_w) + " W" : g.power_source));
  const std::string j = aegis::hw::to_json(h);
  bool keys = balanced(j);
  for (const char* k : {"\"gpu\":", "\"cpu\":", "\"memory\":", "\"thermal\":", "\"video\":", "\"nvdec\":",
                        "\"nvenc\":", "\"isa\":", "\"core_types\":"})
    keys = keys && j.find(k) != std::string::npos;
  expect(keys, "live: JSON has every section and is balanced");

  DiscoverOptions no_nvml;
  no_nvml.nvml = false;
  const HwInfo nn = aegis::hw::discover(no_nvml);
  expect(nn.gpu.present && nn.gpu.sm_count == g.sm_count && !nn.gpu.nvml &&
             (nn.gpu.max_sm_clock_mhz < 0 ? nn.gpu.clock_source.rfind("unavailable", 0) == 0
                                          : nn.gpu.clock_source.rfind("sysfs:", 0) == 0) &&
             nn.gpu.power_source.rfind("unavailable", 0) == 0 && !nn.video.nvenc.present,
         "live, no NVML: CUDA fields kept; clocks from sysfs or unavailable; power and NVENC unavailable");

  DiscoverOptions no_cuda;
  no_cuda.cuda = false;
  no_cuda.nvml = false;
  no_cuda.video = false;
  const HwInfo nc = aegis::hw::discover(no_cuda);
  expect(nc.gpu.source == (nc.gpu.present ? "sysfs" : "none"), "live, no CUDA: GPU from sysfs or absent");
}

void empty_root() {
  DiscoverOptions o;
  o.root = std::string(AEGIS_FIXTURE_DIR) + "/does-not-exist";
  o.cuda = o.nvml = o.auxv = false;
  const HwInfo h = aegis::hw::discover(o);
  expect(h.cpu.logical_cpus == 0 && h.cpu.core_types.empty() && h.memory.total_bytes == -1 && h.thermal.empty() &&
             !h.gpu.present && h.cpu.isa.source.rfind("unavailable", 0) == 0 && balanced(aegis::hw::to_json(h)),
         "empty root: everything unknown, valid JSON");
}

}  // namespace

int main() {
  expect(aegis::hw::core_name(0x41, 0xd85) == "Cortex-X925" && aegis::hw::core_name(0x41, 0xd42) == "Cortex-A78AE" &&
             aegis::hw::core_name(0x41, 0xfff).empty(),
         "core names from MIDR");
  orin_nano();
  spark_recorded_vs_live();
  live_gpu();
  empty_root();
  std::printf("%s: %d failures\n", failures ? "hw_discover FAILED" : "hw_discover ok", failures);
  return failures ? 1 : 0;
}
