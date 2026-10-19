# orin-nano-synthetic

**Synthetic, not recorded.** No Orin Nano was connected when this was written
(2026-10-18). The files follow the L4T (JetPack 6) sysfs layout that
`src/hw/discover.cpp` reads. The values are illustrative of an Orin Nano 8 GB:
- 6 Cortex-A78AE (MIDR 0x41/0xd42) in clusters of 4 + 2;
- cpufreq 115.2-1510.4 MHz;
- GPU devfreq node `17000000.gpu` (`nvidia,ga10b`), max 624.75 MHz, cur 306 MHz
  (`cur_freq`, read by telemetry);
- an NVDEC node (`/dev/v4l2-nvdec`), no NVENC node.

They exist so the Tegra and missing-NVML paths have tests before the device is
available. Replace this directory with `tools/record_hw_fixture.sh` output from
a real Orin Nano when one is connected, and keep the tests' expectations in
step with what it records.
