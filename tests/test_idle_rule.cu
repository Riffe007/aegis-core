// ctest idle_rule: the in-run idle watch's arithmetic (src/bench/harness.h
// watch_scale, foreign_share, SettleFilter) on readings measured on GB10
// (LOG 2026-10-15, 2026-10-17).
// Limit 2% (kIdleMaxPct).
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "bench/harness.h"

using aegis::bench::foreign_share;
using aegis::bench::kIdleMaxPct;
using aegis::bench::kScaleNoBaseline;
using aegis::bench::SettleFilter;
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

  // Nothing foreign at baseline: no measured scale, so k = kScaleNoBaseline (0.25).
  expect(watch_scale(0.0, 0.0) == kScaleNoBaseline, "F0 = 0: k = 0.25", watch_scale(0.0, 0.0));
  // The 2026-10-15 nightly: one window, device 26%, ours 11, gnome-shell 2.1% per process,
  // baseline device 0% with no foreign samples. Old k = 1: 2.1% > 2%, whole run disturbed.
  f = foreign_share(26.0, 11.0, 2.1, watch_scale(0.0, 0.0));
  expect(f <= kIdleMaxPct, "2026-10-15 desktop blip (pp 2.1, no baseline): clean", f);
  expect(foreign_share(26.0, 11.0, 2.1, 1.0) > kIdleMaxPct, "same blip at k = 1 (old rule): disturbed", 2.1);
  // A real foreign load still counts without a baseline: per-process 40 -> 10%.
  f = foreign_share(30.0, 0.0, 40.0, watch_scale(0.0, 0.0));
  expect(f > kIdleMaxPct, "no baseline, foreign per-process 40: disturbed", f);
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

  // Our process exits (trace 2026-10-17, baseline D0 1, F0 1: k = 1). Polls (device %,
  // ours, foreign per-process), 8 per 2 s window: our 96% load ends after poll 6 of
  // window 1; the device total trails at 83% for 4 polls; gnome-shell's stale
  // per-process average jumps 1 -> 8% at the last poll still showing ours and decays.
  struct P { double device, ours, pp; };
  const std::vector<std::vector<P>> trace = {
      {{96, 56, 1}, {96, 58, 1}, {96, 64, 1}, {96, 65, 1}, {96, 67, 1}, {96, 69, 8}, {83, 0, 7}, {83, 0, 6}},
      {{83, 0, 5}, {83, 0, 5}, {1, 0, 4}, {1, 0, 4}, {1, 0, 4}, {1, 0, 4}, {1, 0, 3}, {1, 0, 3}},
      {{1, 0, 3}, {1, 0, 3}, {1, 0, 3}, {1, 0, 3}, {1, 0, 3}, {1, 0, 2}, {1, 0, 2}, {1, 0, 2}},
  };
  // window shares with the settle filter (now) or without it (before)
  auto shares = [](const std::vector<std::vector<P>>& t, double k, bool settle) {
    SettleFilter s;
    std::vector<double> out;
    for (const auto& win : t) {
      std::vector<double> ours;
      for (const P& p : win) ours.push_back(p.ours);
      const std::vector<char> keep = settle ? s.mask(ours) : std::vector<char>(win.size(), 1);
      double d = 0, o = 0, f = 0;
      int n = 0;
      for (size_t i = 0; i < win.size(); ++i)
        if (keep[i]) d += win[i].device, o += win[i].ours, f += win[i].pp, ++n;
      out.push_back(n ? foreign_share(d / n, o / n, f / n, k) : 0.0);
    }
    return out;
  };
  const std::vector<double> before = shares(trace, 1.0, false), now = shares(trace, 1.0, true);
  expect(before[0] > kIdleMaxPct && before[1] > kIdleMaxPct, "exit tail, no settle: windows 1, 2 disturbed",
         std::max(before[0], before[1]));
  for (int i = 0; i < 3; ++i) {
    char what[80];
    std::snprintf(what, sizeof what, "exit tail, settled: window %d clean", i + 1);
    expect(now[i] <= kIdleMaxPct, what, now[i]);
  }
  // Settling hides only the tail: a foreign load that is still there 2 s after our
  // exit is counted (device 30%, per-process 40 from window 2 on).
  std::vector<std::vector<P>> busy = trace;
  for (size_t w = 1; w < busy.size(); ++w)
    for (P& p : busy[w]) p = {30, 0, 40};
  expect(shares(busy, 1.0, true)[2] > kIdleMaxPct, "foreign load persisting past the tail: disturbed",
         shares(busy, 1.0, true)[2]);
  // Our steady load with no drop never settles: a foreign load during it counts.
  std::vector<std::vector<P>> during = {{{96, 60, 40}, {96, 62, 40}, {96, 64, 40}, {96, 66, 40}}};
  expect(shares(during, 1.0, true)[0] > kIdleMaxPct, "foreign per-process 40 during our steady load: disturbed",
         shares(during, 1.0, true)[0]);
  return failures ? 1 : 0;
}
