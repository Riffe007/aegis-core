#!/usr/bin/env bash
# Dump SASS and per-kernel resource usage for every kernel in bench_gemm.
# Output: artifacts/sass/sm_<arch>/<kernel>.sass, plus resource-usage.txt and
# an instruction-mix summary (HMMA / LDSM / LDGSTS / LDG / STS counts).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/bench_gemm"
[[ -x "$BIN" ]] || { echo "build/bench_gemm not found; build first" >&2; exit 1; }

ARCH="$(cuobjdump -lelf "$BIN" | grep -o 'sm_[0-9]*a\?' | sort -u | head -1)"
OUT="$ROOT/artifacts/sass/$ARCH"
mkdir -p "$OUT"

cuobjdump -res-usage "$BIN" > "$OUT/resource-usage.txt" 2>&1 || true

# Kernel symbols that belong to us (skip cuBLAS, which is a shared library anyway).
mapfile -t KERNELS < <(cuobjdump -sass "$BIN" | grep -oP '(?<=Function : )\S+' | grep -i aegis | sort -u)

printf "%-60s %6s %6s %6s %6s %6s\n" kernel HMMA LDSM LDGSTS LDG STS > "$OUT/instruction-mix.txt"
for k in "${KERNELS[@]}"; do
  pretty="$(echo "$k" | c++filt | sed 's/[^A-Za-z0-9_]/_/g' | cut -c1-120)"
  cuobjdump -sass -fun "$k" "$BIN" > "$OUT/$pretty.sass"
  count() { grep -cE "\\b$1" "$OUT/$pretty.sass" || true; }
  printf "%-60s %6s %6s %6s %6s %6s\n" "$(echo "$k" | c++filt | cut -c1-60)" \
    "$(count HMMA)" "$(count LDSM)" "$(count LDGSTS)" "$(count 'LDG\.')" "$(count 'STS')" \
    >> "$OUT/instruction-mix.txt"
done

sha256sum "$OUT"/*.sass > "$OUT/SHA256SUMS"
cat "$OUT/instruction-mix.txt"
echo "SASS written to $OUT"
