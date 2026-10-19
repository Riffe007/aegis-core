#!/usr/bin/env bash
# ctest capi_exports: libaegis_capi.so exports exactly the C ABI. Every defined
# dynamic symbol (nm -D --defined-only) is an aegis_* function declared with
# AEGIS_API in include/aegis/aegis.h, and every such function is exported.
#   tests/test_capi_exports.sh LIB HEADER
set -uo pipefail
LIB="${1:?libaegis_capi.so}" HEADER="${2:?aegis.h}"
exported="$(nm -D --defined-only "$LIB" | awk '{print $NF}' | sort -u)"
declared="$(sed -n 's/^AEGIS_API .*\b\(aegis_[a-z0-9_]*\) *(.*/\1/p' "$HEADER" | sort -u)"
fails=0
other="$(grep -v '^aegis_' <<<"$exported" || true)"
if [[ -n "$other" ]]; then
  echo "FAIL  $(wc -l <<<"$other") exported symbols outside the C ABI:"; sed 's/^/        /' <<<"$other" | head -20
  fails=1
fi
undeclared="$(comm -23 <(grep '^aegis_' <<<"$exported") <(echo "$declared"))"
missing="$(comm -13 <(grep '^aegis_' <<<"$exported") <(echo "$declared"))"
[[ -z "$undeclared" ]] || { echo "FAIL  exported but not declared in aegis.h: $(tr '\n' ' ' <<<"$undeclared")"; fails=1; }
[[ -z "$missing" ]] || { echo "FAIL  declared in aegis.h but not exported: $(tr '\n' ' ' <<<"$missing")"; fails=1; }
((fails)) || echo "ok    $(wc -l <<<"$exported") exported symbols, all aegis_* and all declared in aegis.h ($(wc -l <<<"$declared") declared)"
exit $fails
