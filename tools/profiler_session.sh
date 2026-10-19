#!/usr/bin/env bash
# Run a profiler command and log its session window, so the nightly kernel-log
# guard (tools/kernel_log_guard.sh) can tell the profiler's own NVRM messages
# from GPU faults.
#
#   tools/profiler_session.sh CMD [ARGS...]
#
# Appends to results/profiler_sessions.log (AEGIS_PROFILER_SESSIONS overrides):
#   begin <unix time> pid=<pid> <command>
#   end <unix time> pid=<pid> rc=<exit status>
# Nested calls (a wrapper under tools/gpu_lock.sh, which also logs profiler
# commands) log once: AEGIS_PROFILER_SESSION is set for the children.
# Exit status: CMD's.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${AEGIS_PROFILER_SESSIONS:-$ROOT/results/profiler_sessions.log}"
[[ $# -ge 1 ]] || { echo "usage: profiler_session.sh CMD [ARGS...]" >&2; exit 2; }
if [[ -n "${AEGIS_PROFILER_SESSION:-}" ]]; then exec "$@"; fi
mkdir -p "$(dirname "$LOG")"
printf 'begin %s pid=%d %s\n' "$(date +%s.%N)" $$ "$*" >> "$LOG"
export AEGIS_PROFILER_SESSION=$$
# The end line is written whatever happens to CMD (exit, signal).
rc=1
trap 'printf "end %s pid=%d rc=%d\n" "$(date +%s.%N)" $$ "$rc" >> "$LOG"' EXIT
"$@"
rc=$?
exit $rc
