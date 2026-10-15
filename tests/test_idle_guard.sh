#!/usr/bin/env bash
# ctest idle_guard_paths: the blocking idle guard in the real binaries.
#   tests/test_idle_guard.sh BENCH_GEMM AEGIS_TUNE
# AEGIS_IDLE_MAX_PCT=-1 lowers the limit below any reading, so the busy path is
# deterministic; the variable can only lower the limit (checked below).
set -uo pipefail
BENCH="$1"; TUNE="$2"
fail() { echo "FAIL: $*" >&2; exit 1; }
shapes="$(mktemp)"; trap 'rm -f "$shapes"' EXIT
echo "64 64 64" > "$shapes"

# 1. Busy, no --allow-busy: bench_gemm's guard aborts with exit 3.
AEGIS_IDLE_MAX_PCT=-1 "$BENCH" --idle-check >/dev/null 2>&1; rc=$?
[[ $rc -eq 3 ]] || fail "busy --idle-check exited $rc, want 3"

# 2. Busy with --allow-busy: runs, and reports busy (rows get marked).
out="$(AEGIS_IDLE_MAX_PCT=-1 "$BENCH" --idle-check --allow-busy 2>/dev/null)" || fail "allow-busy --idle-check failed"
grep -q '"busy":true' <<<"$out" || fail "allow-busy reading not marked busy: $out"

# 3. aegis_tune aborts before measuring anything.
out="$(AEGIS_IDLE_MAX_PCT=-1 "$TUNE" --shapes "$shapes" 2>/dev/null)"; rc=$?
[[ $rc -eq 3 ]] || fail "busy aegis_tune exited $rc, want 3"
[[ -z "$out" ]] || fail "busy aegis_tune printed rows: $out"

# 4. The variable cannot raise the limit above 2%.
out="$(AEGIS_IDLE_MAX_PCT=50 "$BENCH" --idle-check --allow-busy 2>/dev/null)" || fail "--idle-check failed"
grep -q '"limit_pct":2.00' <<<"$out" || fail "AEGIS_IDLE_MAX_PCT raised the limit: $out"

# 5. Rows carry the reading: --idle-util-pct/--idle-source/--allow-busy reach the JSON row.
out="$("$BENCH" --impl naive --m 8 --n 8 --k 8 --reps 1 --warmup 0 --idle-util-pct 0.4 --idle-source nvml 2>/dev/null | tail -1)"
grep -q '"idle_util_pct":0.4,"allow_busy":false,"idle_source":"nvml"' <<<"$out" || fail "row lacks the idle reading: $out"
echo "idle guard paths ok"
