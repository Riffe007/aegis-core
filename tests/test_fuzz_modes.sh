#!/usr/bin/env bash
# ctest fuzz_modes_same_cases: the case generator does not depend on the
# reference mode. Case I of seed S is the same problem in the default mode and
# under --cpu-ref-only (which skips only the aegis::matmul cases).
#   tests/test_fuzz_modes.sh AEGIS_FUZZ
set -uo pipefail
FUZZ="$1"
cases() { "$FUZZ" --cases 150 --seed 11 --max-dim 384 --verbose "$@" | sed -n 's/^\(ok  \|FAIL\) case \([0-9]*\)[:(].*: \(.*\)/\2 \3/p; s/^ok   case \([0-9]*\): \(.*\)  max_ratio.*/\1 \2/p'; }
def="$(cases)" || true
cpu="$(cases --cpu-ref-only)" || true
[[ -n "$cpu" ]] || { echo "FAIL: no cases listed" >&2; exit 1; }
diff <(grep -v ' matmul ' <<<"$def") <(echo "$cpu") || { echo "FAIL: --cpu-ref-only generated different cases" >&2; exit 1; }
echo "fuzz modes: $(wc -l <<<"$cpu") identical cases"
