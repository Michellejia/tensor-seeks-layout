#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT_DIR/out/samples}"

mkdir -p "$OUT_ROOT"

for input in "$ROOT_DIR"/samples/*.txt; do
  stem="$(basename "$input" .txt)"
  work_dir="$OUT_ROOT/$stem"
  "$ROOT_DIR/run_layout_selection.sh" "$input" "$work_dir" "$work_dir/${stem}.solved.txt"
done
