#!/usr/bin/env bash
# Rename PNGs ending in "0##.png" to "##.png" (e.g. "Deck.001.png" -> "01.png").
# Usage: rename_slides.sh [-n] [dir]    (-n = dry run; dir defaults to cwd)
set -euo pipefail

dry=0
if [[ "${1:-}" == "-n" ]]; then dry=1; shift; fi
dir="${1:-.}"

re='0([0-9]{2})\.png$'
shopt -s nullglob
count=0
for f in "$dir"/*.png; do
  base="${f##*/}"
  [[ "$base" =~ $re ]] || continue
  new="$dir/${BASH_REMATCH[1]}.png"
  [[ "$f" == "$new" ]] && continue
  if [[ -e "$new" ]]; then
    echo "skip (exists): $base -> ${new##*/}" >&2
    continue
  fi
  echo "$base -> ${new##*/}"
  (( dry )) || mv -n -- "$f" "$new"
  count=$((count + 1))
done
echo "$count file(s) $( ((dry)) && echo 'would be ')renamed"
