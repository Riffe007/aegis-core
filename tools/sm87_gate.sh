#!/usr/bin/env bash
# sm_87 compile gate (ctest sm87_gate; CMake option AEGIS_SM87_GATE).
#
#   tools/sm87_gate.sh OBJECT_LIST
#
# OBJECT_LIST names the objects of the aegis_sm87_gate library, one per line.
# For every kernel compiled for sm_87 (cuobjdump -res-usage), fail if it
#   - spills or uses local memory: STACK > 0 or LOCAL > 0
#   - uses more than 48 KB of static shared memory (SHARED > 49152); dynamic
#     shared memory is opted in at launch and is not counted here
# and fail if an object is missing or holds no sm_87 code. Prints one line per
# kernel. Nothing runs on an sm_87 device.
set -euo pipefail
LIST="${1:?usage: sm87_gate.sh OBJECT_LIST}"
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)}"
STATIC_SMEM_MAX=49152
fail=0 kernels=0
while IFS= read -r obj; do
  [[ -z "$obj" ]] && continue
  if [[ ! -f "$obj" ]]; then
    echo "FAIL missing object (did it compile for sm_87?): $obj"
    fail=1
    continue
  fi
  out="$("$CUOBJDUMP" -res-usage "$obj" 2>&1)" || { echo "FAIL cuobjdump: $obj"; fail=1; continue; }
  if ! grep -q "arch = sm_87" <<<"$out"; then
    echo "FAIL no sm_87 code in $obj"
    fail=1
    continue
  fi
  # Function name line, then its resource line.
  while IFS='|' read -r fn res; do
    kernels=$((kernels + 1))
    reg=$(sed -n 's/.*REG:\([0-9]*\).*/\1/p' <<<"$res")
    stack=$(sed -n 's/.*STACK:\([0-9]*\).*/\1/p' <<<"$res")
    shared=$(sed -n 's/.*SHARED:\([0-9]*\).*/\1/p' <<<"$res")
    local_=$(sed -n 's/.*LOCAL:\([0-9]*\).*/\1/p' <<<"$res")
    why=""
    (( stack > 0 || local_ > 0 )) && why+=" spill/local (STACK:$stack LOCAL:$local_)"
    (( shared > STATIC_SMEM_MAX )) && why+=" static smem $shared > $STATIC_SMEM_MAX"
    name=$(c++filt <<<"$fn" | cut -c1-90)
    if [[ -n "$why" ]]; then
      echo "FAIL sm_87 $name:$why"
      fail=1
    else
      printf "ok   sm_87 REG:%-3s STACK:%s SHARED:%-6s LOCAL:%s  %s\n" "$reg" "$stack" "$shared" "$local_" "$name"
    fi
  done < <(awk '/^ Function /{fn=$2; sub(/:$/,"",fn); next} fn && /REG:/{print fn "|" $0; fn=""}' <<<"$out")
done < "$LIST"
if (( kernels == 0 )); then
  echo "FAIL no sm_87 kernels found"
  exit 1
fi
echo "sm87_gate: $kernels sm_87 kernels checked, $([[ $fail == 0 ]] && echo 'all pass' || echo 'FAILURES')"
exit $fail
