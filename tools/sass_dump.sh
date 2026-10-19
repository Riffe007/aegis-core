#!/usr/bin/env bash
# Dump SASS and per-kernel resource usage for every kernel in bench_gemm.
# Output (artifacts/sass/sm_<arch>/):
#   <name>__<id>.sass     one listing per kernel symbol. <name> is the readable
#                         template name (cut to 80 characters); <id> is the first
#                         16 hex digits of sha1(mangled symbol), so instantiations
#                         that differ only past the cut (e.g. the Layout template
#                         arguments: row- vs column-major A and B) never share a
#                         file. Before 2026-10-18 the cut name alone was used, and
#                         a config's layouts overwrote one another.
#   index.tsv             id, file, layout_a, layout_b, name, mangled symbol: one
#                         row per listing. Layouts are the (aegis::gemm::Layout)
#                         template arguments in order (0 = row-major, as
#                         bench_gemm measures; "-" for kernels without them).
#   BUILD                 bench_gemm --build-info: the commit and dirty flag of
#                         the build the listings came from.
#   resource-usage.txt    cuobjdump -res-usage.
#   instruction-mix.txt   HMMA / LDSM / LDGSTS / LDG / STS counts per listing,
#                         keyed by id, with the full template name (never cut).
#   SHA256SUMS
# AEGIS_SASS_BIN / AEGIS_SASS_OUT override the binary and the output directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${AEGIS_SASS_BIN:-$ROOT/build/bench_gemm}"
[[ -x "$BIN" ]] || { echo "$BIN not found; build first" >&2; exit 1; }

ARCH="$(cuobjdump -lelf "$BIN" | grep -o 'sm_[0-9]*a\?' | sort -u | head -1)"
OUT="${AEGIS_SASS_OUT:-$ROOT/artifacts/sass/$ARCH}"
mkdir -p "$OUT"
rm -f "$OUT"/*.sass "$OUT/index.tsv"

cuobjdump -res-usage "$BIN" > "$OUT/resource-usage.txt" 2>&1 || true
"$BIN" --build-info > "$OUT/BUILD" 2>/dev/null || echo '{"build_sha":null,"build_dirty":true}' > "$OUT/BUILD"

# Kernel symbols that belong to us (skip cuBLAS, which is a shared library anyway).
mapfile -t KERNELS < <(cuobjdump -sass "$BIN" | grep -oP '(?<=Function : )\S+' | grep -i aegis | sort -u)

# The template name: drop the return type, the anonymous namespace and the
# parameter list (after the '>' closing the template arguments, or from the
# first '(' of a non-template kernel); keep every template argument.
name_of() {
  local d="$1"
  d="${d#void }"
  d="${d//"(anonymous namespace)::"/}"
  if [[ "$d" == *">("* ]]; then d="${d%%">("*}>"; else d="${d%%"("*}"; fi
  printf '%s' "$d"
}

printf "%-16s %6s %6s %6s %6s %6s  %s\n" id HMMA LDSM LDGSTS LDG STS kernel > "$OUT/instruction-mix.txt"
printf "id\tfile\tlayout_a\tlayout_b\tname\tsymbol\n" > "$OUT/index.tsv"
for k in "${KERNELS[@]}"; do
  name="$(name_of "$(echo "$k" | c++filt)")"
  id="$(printf '%s' "$k" | sha1sum | cut -c1-16)"
  file="$(printf '%s' "$name" | sed 's/[^A-Za-z0-9_]/_/g' | cut -c1-80)__$id.sass"
  [[ ! -e "$OUT/$file" ]] || { echo "sass_dump: $file written twice (id collision)" >&2; exit 1; }
  # cuobjdump warns once per cubin that lacks the symbol; the listing comes from the one that has it.
  cuobjdump -sass -fun "$k" "$BIN" > "$OUT/$file" 2>/dev/null
  grep -q "Function : $k" "$OUT/$file" || { echo "sass_dump: no SASS for $k" >&2; exit 1; }
  mapfile -t lay < <(grep -oP '\(aegis::gemm::Layout\)\K[0-9]+' <<<"$name" || true)
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$file" "${lay[0]:--}" "${lay[1]:--}" "$name" "$k" >> "$OUT/index.tsv"
  count() { grep -cE "\\b$1" "$OUT/$file" || true; }
  printf "%-16s %6s %6s %6s %6s %6s  %s\n" "$id" \
    "$(count HMMA)" "$(count LDSM)" "$(count LDGSTS)" "$(count 'LDG\.')" "$(count 'STS')" "$name" \
    >> "$OUT/instruction-mix.txt"
done

(cd "$OUT" && sha256sum ./*.sass > SHA256SUMS)
echo "$(($(wc -l < "$OUT/index.tsv") - 1)) kernels, $(find "$OUT" -name '*.sass' | wc -l) listings; build $(cat "$OUT/BUILD")"
echo "SASS written to $OUT (index.tsv, instruction-mix.txt)"
