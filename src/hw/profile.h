#pragma once
// Device profiles (aegis-hw step 3): the measured machine parameters of
// include/aegis/arch/sm_<arch>.json (aegis-perf emit-arch), used only when the file
// describes the detected device. A file matches when its sm_arch, sm_count,
// device name and l2_bytes all equal the device's. One arch can span devices
// (sm_87: Orin Nano, AGX Orin), so arch alone never matches.
//
// Sources, in order:
//   1. $AEGIS_PROFILE_DIR/sm_<arch>.json, if the variable is set and the file exists
//      (e.g. fresh `aegis-perf calibrate` output before a rebuild);
//   2. the arch files compiled into the library (CMake embeds include/aegis/arch/*.json).
//      AEGIS_PROFILE_EMBEDDED=0 turns these off (tests; calibrate checking a fresh file).
// No match: an unmeasured profile. Its identity fields come from the device query
// (discovery); every measured quantity (bandwidth, latency, peak) is unknown (-1),
// never a datasheet value (CLAUDE.md rule 8). The first time a device resolves to
// an unmeasured profile, the process prints one warning to stderr:
//   aegis: unmeasured device ...: run aegis-perf calibrate
// Measurement rows record the label ("measured" / "unmeasured", JSON "profile").
#include <cstdint>
#include <string>

#include "device.h"

namespace aegis::hw {

struct Profile {
  bool measured = false;
  std::string source;    // the file used, or "defaults: device query"
  std::string mismatch;  // why no file matched ("" when measured)
  // Identity (the device's when unmeasured).
  std::string device;
  int arch = 0;
  int sm_count = -1;
  int64_t l2_bytes = -1;
  // Measured quantities; -1: unknown (unmeasured profile, or absent from the file).
  double read_bw_gbps = -1;
  double copy_bw_gbps = -1;
  double mma_peak_tflops = -1;
  double dram_latency_cycles = -1;
  double l2_latency_cycles = -1;
  double latency_ref_clock_mhz = -1;

  const char* label() const { return measured ? "measured" : "unmeasured"; }
};

// One arch file against device `d`: measured if it matches, else the device's
// unmeasured defaults with the reason in `mismatch`.
Profile match_profile(const DeviceLimits& d, const std::string& json_text, const std::string& source);

// The unmeasured profile of `d` (identity from the device query).
Profile unmeasured_profile(const DeviceLimits& d, const std::string& why);

// The arch file compiled into the library for `arch`, or "" if none.
std::string embedded_profile(int arch);

// The current device's profile (hw::current(): a simulated profile included, which
// is resolved on every call and never cached). Cached per real device; the
// one-time warning is printed when it is unmeasured.
const Profile& current_profile();

}  // namespace aegis::hw
