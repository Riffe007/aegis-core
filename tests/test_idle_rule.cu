// ctest idle_rule: the in-run idle watch's arithmetic (src/bench/harness.h
// watch_scale, foreign_share) on readings measured on GB10 (LOG 2026-10-15).
// Limit 2% (kIdleMaxPct).
#include <cmath>
#include <cstdio>

#include "bench/harness.h"

using aegis::bench::foreign_share;
using aegis::bench::kIdleMaxPct;
using aegis::bench::watch_scale;

static int failures = 0;
static void expect(bool ok, const char* what, double got) {
  std::printf("%s %-62s %.3f\n", ok ? "ok  " : "FAIL", what, got);
  failures += !ok;
}

int main() {
  // Idle desktop: device ~1%, per-process samples summed ~5% (Xorg 3.9%, gnome-shell,
  // Firefox). The old rule (sum of foreign per-process) read 5% > 2%: every row disturbed.
  const double k = watch_scale(1.0, 5.0);
  expect(std::fabs(k - 0.2) < 1e-12, "baseline D0 1%, F0 5: k = 0.2", k);
  double f = foreign_share(1.0, 0.0, 5.0, k);
  expect(f <= kIdleMaxPct, "idle desktop, run idle: clean (old rule: 5% disturbed)", f);

  // Our full load ramping (per-process 59 while the device reads 96) on the same desktop.
  f = foreign_share(96.0, 59.0, 5.0, k);
  expect(f <= kIdleMaxPct, "our 96% load, per-process lag, idle desktop: clean", f);

  // The desktop gets busy (e.g. video in Firefox): foreign per-process 40.
  f = foreign_share(30.0, 0.0, 40.0, k);
  expect(f > kIdleMaxPct, "foreign per-process 40 while the run is idle: disturbed", f);
  f = foreign_share(96.0, 59.0, 40.0, k);
  expect(f > kIdleMaxPct, "foreign per-process 40 during our load: disturbed", f);

  // Nothing foreign at baseline: per-process taken at face value (k = 1).
  expect(watch_scale(0.0, 0.0) == 1.0, "F0 = 0: k = 1", watch_scale(0.0, 0.0));
  f = foreign_share(10.0, 0.0, 8.0, watch_scale(0.0, 0.0));
  expect(f > kIdleMaxPct, "k = 1, foreign 8 on an otherwise idle device: disturbed", f);
  // A device at 0% with foreign samples: D0 floored at 0.5%, so foreign use still counts.
  const double k0 = watch_scale(0.0, 5.0);
  expect(std::fabs(k0 - 0.1) < 1e-12, "D0 0%, F0 5: k = 0.1 (floor 0.5%)", k0);
  f = foreign_share(40.0, 0.0, 50.0, k0);
  expect(f > kIdleMaxPct, "then foreign per-process 50: disturbed", f);
  // k never inflates per-process numbers.
  expect(watch_scale(9.0, 3.0) == 1.0, "D0 > F0: k capped at 1", watch_scale(9.0, 3.0));
  // Device below ours (per-process lag after our load ends): no negative share.
  f = foreign_share(5.0, 80.0, 0.0, 1.0);
  expect(f == 0.0, "device < ours: 0", f);
  return failures ? 1 : 0;
}
