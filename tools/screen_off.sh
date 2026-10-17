#!/usr/bin/env bash
# Run one command with the local display blanked, then turn it back on.
#   tools/screen_off.sh CMD [ARGS...]
# On GB10, while the screen is on, the compositor's per-process GPU samples
# rise with our bursty work (gnome-shell 4-13% during aegis_tune, 25% summed
# per process while the device read 2%) and the in-run idle watch marks the
# run disturbed; with the display off they stay at 0-1% (LOG 2026-10-17).
# Without a reachable X display (SSH, no desktop) the command just runs.
set -uo pipefail
[[ $# -ge 1 ]] || { echo "usage: tools/screen_off.sh CMD [ARGS...]" >&2; exit 2; }
export DISPLAY="${DISPLAY:-:1}"
[[ -n "${XAUTHORITY:-}" || ! -r "/run/user/$(id -u)/gdm/Xauthority" ]] || export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
if xset q >/dev/null 2>&1; then
  trap 'xset dpms force on 2>/dev/null' EXIT INT TERM
  xset dpms force off && sleep 2
  echo "screen_off: display $DISPLAY off for: $*" >&2
else
  echo "screen_off: no X display reachable; running without blanking" >&2
fi
"$@"
