// Device profiles (src/hw/profile.h), ctest hw_profile.
//   1. The embedded arch file matches the real device: measured, with its roofs.
//   2. Mismatches give an unmeasured profile with the reason: SM count (a simulated
//      24-SM device of the same arch), device name, L2 size, another arch (no file),
//      invalid JSON. Unmeasured identity comes from the device query; every measured
//      quantity is -1.
//   3. current_profile(): the real device is measured; under a simulated sm_87
//      profile it is unmeasured (only a real device prints the warning).
//   4. With AEGIS_PROFILE_EMBEDDED=0 (a child process: the profile is cached per
//      process): the real device is unmeasured, the warning is printed exactly
//      once across repeated current_profile() and aegis::matmul calls, and a
//      bench row is labelled "profile":"unmeasured".
#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "aegis/common.cuh"
#include "aegis/matmul.h"
#include "bench/gpu_budget.h"
#include "hw/device.h"
#include "hw/profile.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%-5s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

bool unknown_roofs(const aegis::hw::Profile& p) {
  return p.read_bw_gbps == -1 && p.copy_bw_gbps == -1 && p.mma_peak_tflops == -1 && p.dram_latency_cycles == -1 &&
         p.l2_latency_cycles == -1;
}

void matching(const aegis::hw::DeviceLimits& real) {
  using aegis::hw::match_profile;
  const std::string text = aegis::hw::embedded_profile(real.arch);
  if (text.empty()) {
    std::printf("skip  no embedded profile for sm_%d (matching checks need one)\n", real.arch);
    return;
  }
  const aegis::hw::Profile p = match_profile(real, text, "embedded");
  expect(p.measured && p.mismatch.empty() && p.read_bw_gbps > 0 && p.mma_peak_tflops > 0 && p.dram_latency_cycles > 0,
         "embedded sm_" + std::to_string(real.arch) + " profile matches the device: measured, read_bw " +
             std::to_string(p.read_bw_gbps) + " GB/s");

  aegis::hw::DeviceLimits d = real;
  d.sm_count = 24;
  aegis::hw::Profile q = match_profile(d, text, "embedded");
  expect(!q.measured && q.mismatch.find("sm_count") != std::string::npos && q.sm_count == 24 && unknown_roofs(q),
         "24-SM device of the same arch: unmeasured (" + q.mismatch + ")");
  d = real;
  d.name = "Some Other GPU";
  q = match_profile(d, text, "embedded");
  expect(!q.measured && q.mismatch.find("device") != std::string::npos && q.device == "Some Other GPU",
         "another device name: unmeasured");
  d = real;
  d.l2_bytes = real.l2_bytes / 2;
  q = match_profile(d, text, "embedded");
  expect(!q.measured && q.mismatch.find("l2_bytes") != std::string::npos, "another L2 size: unmeasured");
  q = match_profile(real, "{not json", "bad");
  expect(!q.measured && q.mismatch.find("not valid JSON") != std::string::npos && unknown_roofs(q),
         "invalid JSON: unmeasured");

}

void current(const aegis::hw::DeviceLimits& real) {
  const aegis::hw::Profile& p = aegis::hw::current_profile();
  const bool has_file = !aegis::hw::embedded_profile(real.arch).empty();
  expect(p.measured == has_file, std::string("real device: ") + p.label() + " (" +
                                     (p.measured ? p.source : p.mismatch) + ")");
  aegis::hw::DeviceLimits nano = real;
  nano.name = "Orin (simulated)";
  nano.arch = 87;
  nano.sm_count = 8;
  {
    aegis::hw::ScopedSimulation sim(nano);
    const aegis::hw::Profile& s = aegis::hw::current_profile();
    expect(!s.measured && s.arch == 87 && s.sm_count == 8 && s.mismatch.find("sm_87") != std::string::npos,
           "simulated sm_87: unmeasured (" + s.mismatch + ")");
  }
  expect(aegis::hw::current_profile().measured == has_file, "real device again after the simulation");
}

// Child: AEGIS_PROFILE_EMBEDDED=0. stderr is redirected to a file and the
// warnings counted.
int child() {
  char path[] = "/tmp/aegis_profile_warnXXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) return 2;
  std::fflush(stderr);
  const int saved = dup(2);
  dup2(fd, 2);
  const aegis::hw::Profile& p = aegis::hw::current_profile();
  aegis::hw::current_profile();
  aegis::hw::current_profile();
  __half *A, *B, *C;
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&A, 16 * 64 * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&B, 64 * 64 * 2));
  AEGIS_CUDA_CHECK(aegis::budget::gpu_malloc(&C, 16 * 64 * 2));
  AEGIS_CUDA_CHECK(cudaMemset(A, 0, 16 * 64 * 2));
  AEGIS_CUDA_CHECK(cudaMemset(B, 0, 64 * 64 * 2));
  const cudaError_t e1 = aegis::matmul(A, B, C, 16, 64, 64);
  const cudaError_t e2 = aegis::matmul(A, B, C, 16, 64, 64);
  AEGIS_CUDA_CHECK(cudaDeviceSynchronize());
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::FILE* f = std::fopen(path, "r");
  std::string text;
  char buf[512];
  while (f && std::fgets(buf, sizeof buf, f)) text += buf;
  if (f) std::fclose(f);
  unlink(path);
  int warnings = 0;
  for (size_t at = 0; (at = text.find("aegis: unmeasured device", at)) != std::string::npos; ++at) ++warnings;
  expect(!p.measured && p.mismatch.find("AEGIS_PROFILE_EMBEDDED=0") != std::string::npos,
         "embedded profiles off: the real device is unmeasured");
  expect(warnings == 1 && text.find("run aegis-perf calibrate") != std::string::npos,
         "the warning is printed once (" + std::to_string(warnings) + ") across 3 lookups and 2 matmul calls");
  expect(e1 == cudaSuccess && e2 == cudaSuccess, "matmul still runs on an unmeasured device");
  aegis::budget::gpu_free(A); aegis::budget::gpu_free(B); aegis::budget::gpu_free(C);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--child") == 0) return child();
  const aegis::hw::DeviceLimits* real = aegis::hw::physical();
  if (!real) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  matching(*real);
  current(*real);
  // The unmeasured real device needs a fresh process (the profile is cached per process).
  const std::string self = argv[0];
  const std::string cmd = "AEGIS_PROFILE_EMBEDDED=0 '" + self + "' --child";
  const int rc = std::system(cmd.c_str());
  expect(rc == 0, "child with AEGIS_PROFILE_EMBEDDED=0");
  std::printf("%s: %d failures\n", failures ? "hw_profile FAILED" : "hw_profile ok", failures);
  return failures ? 1 : 0;
}
