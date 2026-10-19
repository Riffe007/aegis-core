#!/usr/bin/env bash
# ctest kernel_log_guard: tools/kernel_log_guard.sh and the nightly's use of it,
# on kernel-log lines recorded on the Spark (tests/fixtures/kernel_log/,
# `journalctl -k -o short-unix`):
#   profiler.log  ncu's NVRM refcnt messages, 2026-10-17 19:44:19-20 (+ a
#                 Bluetooth line)
#   xid.log       an Xid 43 (bench_gemm, 2026-10-14)
#   nvrm_oom.log  an NVRM out-of-memory check failure (2026-10-14)
#   no_gpu.log    kernel lines that are not GPU lines
# Session logs are written here, around the recorded times.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FX="$ROOT/tests/fixtures/kernel_log"
GUARD="$ROOT/tools/kernel_log_guard.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fails=0
check() {  # NAME EXPECT(clean|blocked) PATTERN LOG SESSIONS
  local name="$1" want="$2" pat="$3" log="$4" ses="$5" out rc
  out="$("$GUARD" --log "$log" --sessions "$ses" 2>&1)"; rc=$?
  local got=clean; ((rc == 1)) && got=blocked; ((rc > 1)) && got="error($rc)"
  if [[ "$got" == "$want" && "$out" == *$pat* ]]; then
    echo "ok    $name: $out"
  else
    echo "FAIL  $name: want $want /$pat/, got $got: $out"; fails=$((fails + 1))
  fi
}
session() { printf 'begin %s pid=%s ncu --set full\n' "$2" "$1"; [[ -n "${3:-}" ]] && printf 'end %s pid=%s rc=0\n' "$3" "$1"; }

session 101 1792291450.0 1792291458.0 > "$T/inside.ses"      # ends 1.1 s before the first line: within the 5 s slack
session 102 1792291300.0 1792291400.0 > "$T/before.ses"      # ends a minute earlier
session 103 1792291455.0 > "$T/unclosed.ses"                 # killed: no end line
session 104 1792014000.0 1792014300.0 > "$T/oct14.ses"       # covers the Xid and the OOM line
{ session 105 1792291300.0 1792291400.0; session 106 1792291456.0 1792291470.0; } > "$T/two.ses"
: > "$T/none.ses"

check "profiler lines inside a session" clean "5 profiler lines inside logged sessions ignored" "$FX/profiler.log" "$T/inside.ses"
check "profiler lines, the second of two sessions" clean "5 profiler lines" "$FX/profiler.log" "$T/two.ses"
check "profiler lines, unclosed session" clean "5 profiler lines" "$FX/profiler.log" "$T/unclosed.ses"
check "profiler lines, no session logged" blocked "outside every logged profiler session" "$FX/profiler.log" "$T/none.ses"
check "profiler lines after the session" blocked "outside every logged profiler session" "$FX/profiler.log" "$T/before.ses"
check "Xid inside a session" blocked "Xid (PCI:000f:01:00): 43" "$FX/xid.log" "$T/oct14.ses"
check "other NVRM line inside a session" blocked "Out of memory" "$FX/nvrm_oom.log" "$T/oct14.ses"
cat "$FX/profiler.log" "$FX/xid.log" > "$T/mixed.log"
check "profiler lines then an Xid" blocked "Xid" "$T/mixed.log" "$T/two.ses"
check "no GPU lines" clean "clean: 0 GPU lines" "$FX/no_gpu.log" "$T/none.ses"

# profiler_session.sh writes a begin and an end line (with the exit status), once when nested.
AEGIS_PROFILER_SESSIONS="$T/written.ses" "$ROOT/tools/profiler_session.sh" \
  "$ROOT/tools/profiler_session.sh" bash -c 'exit 3'; rc=$?
if [[ $rc -eq 3 && "$(grep -c '^begin ' "$T/written.ses")" == 1 && "$(grep -c '^end .* rc=3$' "$T/written.ses")" == 1 ]]; then
  echo "ok    profiler_session.sh logs one window, rc 3"
else
  echo "FAIL  profiler_session.sh: rc $rc, log: $(tr '\n' '|' < "$T/written.ses")"; fails=$((fails + 1))
fi

# gpu_lock.sh runs a profiler command through profiler_session.sh, with or without the lock held.
mkdir -p "$T/bin"; printf '#!/bin/sh\nexit 0\n' > "$T/bin/ncu"; chmod +x "$T/bin/ncu"
env -u AEGIS_GPU_LOCK_HELD -u AEGIS_PROFILER_SESSION AEGIS_GPU_LOCK="$T/gpu.lock" AEGIS_PROFILER_SESSIONS="$T/lock.ses" \
  "$ROOT/tools/gpu_lock.sh" "$T/bin/ncu" --version
env -u AEGIS_PROFILER_SESSION AEGIS_GPU_LOCK_HELD=1 AEGIS_PROFILER_SESSIONS="$T/lock.ses" "$ROOT/tools/gpu_lock.sh" "$T/bin/ncu" -v
env -u AEGIS_GPU_LOCK_HELD -u AEGIS_PROFILER_SESSION AEGIS_GPU_LOCK="$T/gpu.lock" AEGIS_PROFILER_SESSIONS="$T/lock.ses" \
  "$ROOT/tools/gpu_lock.sh" true
if [[ "$(grep -c '^begin .*bin/ncu' "$T/lock.ses")" == 2 && "$(grep -c '^end ' "$T/lock.ses")" == 2 ]]; then
  echo "ok    gpu_lock.sh logs ncu sessions (lock taken, lock held), not other commands"
else
  echo "FAIL  gpu_lock.sh session log: $(tr '\n' '|' < "$T/lock.ses" 2>/dev/null)"; fails=$((fails + 1))
fi

# The nightly: a blocked kernel log is a loud skip; a clean one passes the guard.
nightly() {  # LOG SESSIONS -> the summary line
  rm -f "$T/nightly.log"
  env -u AEGIS_GPU_LOCK_HELD AEGIS_GPU_LOCK="$T/gpu.lock" AEGIS_NIGHTLY_SUMMARY="$T/nightly.log" \
    AEGIS_NIGHTLY_DIR="$T/nightly" AEGIS_KERNEL_LOG="$1" AEGIS_PROFILER_SESSIONS="$2" \
    "$ROOT/tools/nightly.sh" --preflight-only
  echo "rc=$? $(cat "$T/nightly.log" 2>/dev/null)"
}
out="$(nightly "$FX/profiler.log" "$T/none.ses")"
if [[ "$out" == "rc=0 "*"status=SKIPPED reason=\"kernel log since "*"blocked: profiler-pattern line outside"* ]] &&
   grep -q '== NIGHTLY SKIPPED: kernel log' "$T"/nightly/*.log; then
  echo "ok    nightly, profiler lines without a session: ${out#rc=0 }"
else
  echo "FAIL  nightly blocked path: $out"; fails=$((fails + 1))
fi
out="$(nightly "$FX/profiler.log" "$T/inside.ses")"
if [[ "$out" == "rc=0 "*"status=PREFLIGHT_OK kernel log: clean: 5 GPU lines, 5 profiler lines"* ]]; then
  echo "ok    nightly, profiler lines inside a session: ${out#rc=0 }"
else
  echo "FAIL  nightly clean path: $out"; fails=$((fails + 1))
fi
out="$(nightly "$FX/xid.log" "$T/oct14.ses")"
if [[ "$out" == *"status=SKIPPED reason="*"blocked: "*"Xid"* ]]; then
  echo "ok    nightly, an Xid inside a session: skipped"
else
  echo "FAIL  nightly Xid path: $out"; fails=$((fails + 1))
fi

echo "kernel_log_guard: $fails failures"
exit $((fails > 0))
