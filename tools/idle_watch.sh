# The idle guard during a run, for bash scripts (source this file):
#
#   source "$ROOT/tools/idle_watch.sh"
#   idle_watch_start "$ROOT/build/bench_gemm" OUT.watch.json   # after the idle guard
#   ... the run (children of this shell are the run's own processes) ...
#   idle_watch_stop                                             # report in OUT.watch.json
#   aegis-perf ingest-microbench ... --watch OUT.watch.json     # marks rows disturbed
#
# bench_gemm --idle-watch --root $$ samples GPU use by processes outside this
# shell's process tree every 2 s until its stdin (a FIFO held open here)
# closes, logs every window above the limit on stderr, then writes its report.

idle_watch_start() {
  local bench="$1" out="$2" fifo
  fifo="$(mktemp -u "${TMPDIR:-/tmp}/aegis-idle-watch.XXXXXX")"
  mkfifo "$fifo"
  "$bench" --idle-watch --root $$ < "$fifo" > "$out" &
  IDLE_WATCH_PID=$!
  exec {IDLE_WATCH_FD}> "$fifo"
  rm -f "$fifo"
  IDLE_WATCH_OUT="$out"
}

idle_watch_stop() {
  [[ -n "${IDLE_WATCH_PID:-}" ]] || return 0
  exec {IDLE_WATCH_FD}>&-
  wait "$IDLE_WATCH_PID" || echo "idle watch: watcher failed; the run stays unwatched" >&2
  IDLE_WATCH_PID=
  if [[ "$(jq -r .disturbed "$IDLE_WATCH_OUT" 2>/dev/null)" == true ]]; then
    echo "idle watch: run DISTURBED: $(jq -r '"\(.windows_over) of \(.windows) windows over \(.limit_pct)%, busiest \(.max_other_pct)% (\(.worst))"' "$IDLE_WATCH_OUT")" >&2
  fi
}
