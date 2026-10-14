#!/usr/bin/env bash
# compute-sanitizer over the ctest suite (Production bar: clean under memcheck,
# racecheck, initcheck and synccheck), one tool per invocation, never nested.
#
#   tools/sanitize.sh [TOOL ...]       default: memcheck racecheck initcheck synccheck
#   JOBS=N FUZZ_CASES=100 tools/sanitize.sh memcheck
#
# Every sanitized process runs through tools/gpu_job.sh (oom_score_adj 1000 +
# MemoryMax). Parallelism: JOBS if set, else 2 for memcheck and 1 for the
# others; never more than 2 for memcheck or 1 for racecheck / synccheck.
# Never pass --num-cuda-barriers (CLAUDE.md rule 12).
#
# Scope (CLAUDE.md Benchmark policy):
#   - all tools skip sm87_gate and the gpu_budget_* ctests (no kernels; the
#     budget tests also need the ctest environment) and the memcheck_* ctests (they
#     already run compute-sanitizer; nesting is not supported);
#   - fuzz tests run FUZZ_CASES cases (default 100) with dims <= 512;
#   - synccheck runs only our kernels, with no cuBLAS / cuBLASLt kernel at all:
#     bench_gemm tests with --check none, fuzz tests with --cpu-ref-only, and
#     the cuBLAS-using tests (capi_*, matmul_dispatch) left out. cuBLAS kernels
#     use cuda::barrier, whose tracking overflows synccheck's table.
# Logs: artifacts/sanitizer/<tool>/<test>.log. Exits 1 if any run reported an
# error or ended without a sanitizer summary.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
SAN="${SANITIZER:-$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)}"
FUZZ_CASES="${FUZZ_CASES:-100}"
TOOLS=("$@")
[[ ${#TOOLS[@]} -eq 0 ]] && TOOLS=(memcheck racecheck initcheck synccheck)

mapfile -t ALL < <(ctest --test-dir "$BUILD" --show-only=json-v1 |
  jq -r '.tests[] | select(.name != "sm87_gate" and (.name | startswith("memcheck_") | not)
                           and (.name | startswith("gpu_budget_") | not))
         | .name + "\t" + (.command | @sh)')

status=0
for tool in "${TOOLS[@]}"; do
  case "$tool" in
    memcheck) max=2 ;;
    *) max=1 ;;
  esac
  jobs="${JOBS:-$max}"
  (( jobs > max )) && jobs=$max
  out="$ROOT/artifacts/sanitizer/$tool"
  rm -rf "$out"
  mkdir -p "$out"
  # Per-tool command rewriting (commands are shell-quoted per argument).
  rewrite=(-e "s/'--cases' '[0-9]+'/'--cases' '$FUZZ_CASES' '--max-dim' '512'/")
  tests=("${ALL[@]}")
  if [[ "$tool" == synccheck ]]; then
    rewrite+=(-e "s/'--check'/'--check' 'none'/" -e "s/(aegis_fuzz')/\\1 '--cpu-ref-only'/")
    mapfile -t tests < <(printf '%s\n' "${ALL[@]}" | grep -vE '^(capi_|matmul_dispatch)')
  fi
  printf '%s\n' "${tests[@]}" | sed -E "${rewrite[@]}" |
    xargs -P "$jobs" -d '\n' -I{} bash -c '
      name="${1%%$(printf "\t")*}"; cmd="${1#*$(printf "\t")}"
      eval "\"$5/tools/gpu_job.sh\" \"$2\" --tool $3 --error-exitcode 99 --print-limit 20 $cmd" > "$4/$name.log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then echo "FAIL $3 $name (exit $rc)"; fi
    ' _ {} "$SAN" "$tool" "$out" "$ROOT"
  # memcheck/initcheck/synccheck end with "ERROR SUMMARY: N errors"; racecheck
  # with "RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)".
  bad=$(grep -l -E "ERROR SUMMARY: [1-9]|RACECHECK SUMMARY: [1-9]|========= (Invalid|Race|Uninitialized|Barrier|Program hit|Warning: Detected overflow)" "$out"/*.log 2>/dev/null | wc -l)
  errs=$(grep -h -oE "ERROR SUMMARY: [0-9]+|RACECHECK SUMMARY: [0-9]+" "$out"/*.log | awk '{s += $3} END {print s + 0}')
  missing=$(grep -L -E "ERROR SUMMARY|RACECHECK SUMMARY" "$out"/*.log 2>/dev/null | wc -l)
  [[ $missing -ne 0 ]] && echo "$tool: $missing logs without a sanitizer summary (crashed or killed?)"
  echo "$tool: ${#tests[@]} tests (JOBS=$jobs), $bad with reports, $errs errors total (logs: $out)"
  [[ $bad -ne 0 || $missing -ne 0 ]] && status=1
done
exit $status
