#!/usr/bin/env bash
# usage: new_lecture.sh <number> <template.prproj>   
#        e.g. new_lecture.sh 16 "Lecture 05/Premiere/Visions of the End L05.prproj"
# Needs ffprobe and python3. A relative template path resolves against the current directory.
set -euo pipefail
n=$((10#${1:?lecture number}))
pad=$(printf '%02d' "$n")
root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/Lecture $pad"
raw="$root/_Raw/Lecture $n"
tpl="${2:?template .prproj}"
template="$(cd "$(dirname "$tpl")" && pwd)/$(basename "$tpl")"

mkdir -p "$dir/Premiere" "$dir/Slides" "$dir/Sources"
if [ -d "$raw" ]; then
  for f in "$raw"/*.mp4; do [ -e "$f" ] && mv -n "$f" "$dir/Sources/"; done
fi
[ "$(ls "$dir"/Sources/*.mp4 2>/dev/null | wc -l)" -eq 4 ] || { echo "expected 4 .mp4 files in $dir/Sources"; exit 1; }

python3 "$root/_Scripts/make_prproj.py" --config "$root/_Scripts/prproj.config.json" \
  --template "$template" --media-dir "$dir/Sources" --project-root "$dir" \
  "$dir/Premiere/Visions of the End L$pad.prproj"
