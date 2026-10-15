# The idle guard during a run, for bash scripts (source this file):
#
#   source "$ROOT/tools/idle_watch.sh"
#   idle_watch_start "$ROOT/build/bench_gemm" OUT.watch.json   # after the idle guard
#   ... the run (children of this shell are the run's own processes) ...
#   idle_watch_stop                                             # report in OUT.watch.json
#   aegis-perf ingest-microbench ... --watch OUT.watch.json     # marks rows disturbed
#
# bench_gemm --idle-watch --root $$ takes a 2 s baseline (idle_watch_start
# returns after its ready line), then every 2 s estimates the device-level GPU
# use of processes outside this shell's process tree and watches for foreign
# compute contexts, until its stdin (a FIFO held open here) closes; it logs
# every window over the limit on stderr, then writes its report (last line).
# Rule: CLAUDE.md, Benchmark policy, "Idle guard during the run".

idle_watch_start() {
  local bench="$1" out="$2" fifo
  fifo="$(mktemp -u "${TMPDIR:-/tmp}/aegis-idle-watch.XXXXXX")"
  mkfifo "$fifo"
  "$bench" --idle-watch --root $$ < "$fifo" > "$out" &
  IDLE_WATCH_PID=$!
  exec {IDLE_WATCH_FD}> "$fifo"
  rm -f "$fifo"
  IDLE_WATCH_OUT="$out"
  # The watcher takes a 2 s baseline first; the run starts after its ready line.
  local i
  for ((i = 0; i < 100; ++i)); do
    grep -q '"ready"' "$out" 2>/dev/null && return 0
    kill -0 "$IDLE_WATCH_PID" 2>/dev/null || break
    sleep 0.1
  done
  echo "idle watch: no ready line from the watcher; the run stays unwatched" >&2
}

idle_watch_stop() {
  [[ -n "${IDLE_WATCH_PID:-}" ]] || return 0
  exec {IDLE_WATCH_FD}>&-
  wait "$IDLE_WATCH_PID" || echo "idle watch: watcher failed; the run stays unwatched" >&2
  IDLE_WATCH_PID=
  if [[ "$(tail -1 "$IDLE_WATCH_OUT" | jq -r .disturbed 2>/dev/null)" == true ]]; then
    echo "idle watch: run DISTURBED: $(tail -1 "$IDLE_WATCH_OUT" | jq -r '"\(.windows_over) of \(.windows) windows over \(.limit_pct)%, worst \(.max_other_pct)% foreign (\(.worst))"')" >&2
  fi
}
