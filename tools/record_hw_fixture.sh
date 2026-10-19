#!/usr/bin/env bash
# Record the sysfs / procfs / dev files hardware discovery reads (include/aegis/hw.h)
# into a fixture tree, for tests (tests/fixtures/hw/<name>/).
#
#   tools/record_hw_fixture.sh DEST
#
# Files are copied dereferenced (cpufreq is a symlink into policyN); device nodes
# become empty files of the same name (discovery only tests their presence);
# `uname -m` goes to DEST/uname-m. Nothing GPU-side is recorded: CUDA and NVML
# answer at run time, and Tegra's GPU devfreq node is copied when present.
set -euo pipefail
dest=${1:?usage: record_hw_fixture.sh DEST}
mkdir -p "$dest"

copy() {  # copy one file if readable, keeping its path under $dest
  local src=$1
  [[ -r "$src" && -f "$src" ]] || return 0
  mkdir -p "$dest$(dirname "$src")"
  cat "$src" > "$dest$src" 2>/dev/null || rm -f "$dest$src"
}

cpu=/sys/devices/system/cpu
copy $cpu/online
for d in $cpu/cpu[0-9]*; do
  copy "$d/topology/cluster_id"
  copy "$d/topology/physical_package_id"
  copy "$d/topology/core_cpus_list"
  copy "$d/topology/thread_siblings_list"
  copy "$d/cpu_capacity"
  copy "$d/cpufreq/cpuinfo_min_freq"
  copy "$d/cpufreq/cpuinfo_max_freq"
  copy "$d/regs/identification/midr_el1"
done
copy /proc/cpuinfo
copy /proc/meminfo
copy /proc/device-tree/model
for z in /sys/class/thermal/thermal_zone*; do
  copy "$z/type"
  copy "$z/temp"
done
for f in /sys/class/devfreq/*; do
  [[ -e "$f" ]] || continue
  copy "$f/max_freq"
  copy "$f/min_freq"
  copy "$f/device/of_node/compatible"
done
for n in /dev/v4l2-nvdec /dev/v4l2-nvenc /dev/nvhost-nvdec /dev/nvhost-nvdec1 /dev/nvhost-msenc /dev/nvhost-nvenc1; do
  if [[ -e "$n" ]]; then mkdir -p "$dest/dev"; : > "$dest$n"; fi
done
uname -m > "$dest/uname-m"
mkdir -p "$dest/etc"
hostname > "$dest/etc/hostname"
echo "recorded $(find "$dest" -type f | wc -l) files into $dest"
