#!/usr/bin/env bash
# Kernel-log guard (Benchmark policy "Before any run"; tools/nightly.sh): are
# there new GPU errors (Xid / NVRM lines) in the kernel log?
#
#   tools/kernel_log_guard.sh --since TIME [--log FILE] [--sessions FILE]
#
# Lines come from `journalctl -k -o short-unix --since TIME` (--log FILE: the
# same format from a file, for tests). An NVRM line is ignored only if BOTH
#   1. it matches a known profiler pattern (below), AND
#   2. its time falls inside a logged profiler session window
#      (results/profiler_sessions.log, written by tools/profiler_session.sh;
#      AEGIS_PROFILER_SESSIONS or --sessions overrides), widened by SLACK
#      seconds on each side (the driver logs a moment after the process exits).
# Everything else blocks: any Xid, any other NVRM line, and a profiler-pattern
# line outside every window. A session with no end line (the wrapper was
# killed) covers UNCLOSED seconds after its begin.
#
# Output: one line. Exit 0: clean ("clean: N lines, M profiler lines inside
# sessions ignored"); 1: blocked ("blocked: <the first blocking line>");
# 2: usage or the log could not be read.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SINCE="" LOGFILE="" SESSIONS="${AEGIS_PROFILER_SESSIONS:-$ROOT/results/profiler_sessions.log}"
SLACK="${AEGIS_PROFILER_SLACK_S:-5}" UNCLOSED=3600
while [[ $# -gt 0 ]]; do
  case "$1" in
    --since) SINCE="$2"; shift 2 ;;
    --log) LOGFILE="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    *) echo "usage: kernel_log_guard.sh --since TIME [--log FILE] [--sessions FILE]" >&2; exit 2 ;;
  esac
done
[[ -n "$SINCE" || -n "$LOGFILE" ]] || { echo "kernel_log_guard: --since or --log required" >&2; exit 2; }

# Known messages the driver logs while Nsight Compute holds the GPU's
# performance counters (1,072 such lines 2026-10-14..17, each during an ncu
# session; LOG 2026-10-18). Extended only with evidence.
PROFILER_RE='NVRM: refcntRequestReference_IMPL: Failed to enter state 1 [(]current state: 0, status: 0x00000056[)]'

if [[ -n "$LOGFILE" ]]; then
  lines="$(cat "$LOGFILE")" || exit 2
else
  lines="$(journalctl -k -o short-unix --since "$SINCE" --no-pager -q 2>/dev/null)" || exit 2
fi
sessions="$(cat "$SESSIONS" 2>/dev/null || true)"

awk -v re="$PROFILER_RE" -v slack="$SLACK" -v unclosed="$UNCLOSED" '
  FNR == NR {  # sessions: begin/end lines, matched by pid
    if ($1 == "begin") { pid = $3; b[pid] = $2; order[++n] = pid; e[pid] = "" }
    else if ($1 == "end" && ($3 in b)) e[$3] = $2
    next
  }
  /Xid|NVRM/ {
    total++
    t = $1 + 0
    if ($0 ~ re) {
      for (i = 1; i <= n; i++) {
        p = order[i]
        hi = e[p] == "" ? b[p] + unclosed : e[p] + 0
        if (t >= b[p] - slack && t <= hi + slack) { ignored++; next }
      }
      print "blocked: profiler-pattern line outside every logged profiler session: " $0
      blocked = 1
      exit 1
    }
    print "blocked: " $0
    blocked = 1
    exit 1
  }
  END {
    if (!blocked) printf "clean: %d GPU lines, %d profiler lines inside logged sessions ignored\n", total, ignored
  }
' <(printf '%s\n' "$sessions") <(printf '%s\n' "$lines") | {
  read -r out
  echo "$out"
  [[ "$out" == clean:* ]]
}
