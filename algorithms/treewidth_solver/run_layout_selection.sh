#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  cat <<'USAGE'
Usage:
  ./run_layout_selection.sh <input_dump.txt> [work_dir] [output_dump.txt]

Defaults:
  work_dir      = out/pipeline/<input_basename_without_ext>
  output_dump   = <work_dir>/<input_basename_without_ext>.solved.txt

Environment:
  LS_TIMEOUT_MS   per-stage timeout in milliseconds (default: 600000)
  LS_MEM_CAP_MB   per-stage address-space cap in MB (default: 8192, hard max: 8192)
  TW_EXACT        path to external tw-exact wrapper
  TW_SOLVER_DIR   path to external PACE2017-TrackA checkout
  TW_XMX/TW_XMS/TW_XSS  forwarded to the external Java treewidth solver
USAGE
  exit 1
fi

INPUT_DUMP="$1"
WORK_DIR="${2:-}"
OUTPUT_DUMP="${3:-}"
TIMEOUT_MS="${LS_TIMEOUT_MS:-600000}"
TIMEOUT_SEC=$(( (TIMEOUT_MS + 999) / 1000 ))
MAX_MEM_CAP_MB=8192
MEM_CAP_MB="${LS_MEM_CAP_MB:-8192}"

if [[ ! -f "$INPUT_DUMP" ]]; then
  echo "error: input file not found: $INPUT_DUMP" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LS_OPT="$ROOT_DIR/ls_opt"
DEFAULT_TW_SOLVER_DIR="$ROOT_DIR/external/PACE2017-TrackA"
TW_SOLVER_DIR="${TW_SOLVER_DIR:-$DEFAULT_TW_SOLVER_DIR}"
TW_EXACT="${TW_EXACT:-$TW_SOLVER_DIR/tw-exact}"
TW_EXACT_DIR="$(cd "$(dirname "$TW_EXACT")" 2>/dev/null && pwd || true)"
TW_MAIN_CLASS="${TW_EXACT_DIR}/tw/exact/MainDecomposer.class"

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: required command 'timeout' not found" >&2
  exit 1
fi

if [[ -n "$MEM_CAP_MB" ]]; then
  if ! [[ "$MEM_CAP_MB" =~ ^[0-9]+$ ]] || [[ "$MEM_CAP_MB" -le 0 ]]; then
    echo "error: LS_MEM_CAP_MB must be a positive integer (MB), got '$MEM_CAP_MB'" >&2
    exit 1
  fi
  if [[ "$MEM_CAP_MB" -gt "$MAX_MEM_CAP_MB" ]]; then
    echo "warning: LS_MEM_CAP_MB=$MEM_CAP_MB exceeds max $MAX_MEM_CAP_MB; clamping to $MAX_MEM_CAP_MB" >&2
    MEM_CAP_MB="$MAX_MEM_CAP_MB"
  fi
fi

if [[ ! -x "$LS_OPT" ]]; then
  echo "[build] ls_opt not found, running make"
  (cd "$ROOT_DIR" && make)
fi

if [[ ! -x "$TW_EXACT" ]]; then
  echo "error: external tw-exact not found or not executable at $TW_EXACT" >&2
  echo "hint: see README.md for how to clone and build TCS-Meiji/PACE2017-TrackA" >&2
  exit 1
fi

if [[ ! -f "$TW_MAIN_CLASS" ]]; then
  if [[ -f "${TW_EXACT_DIR}/Makefile" ]]; then
    echo "[build] tw-exact classes not found, running make exact in ${TW_EXACT_DIR}"
    (cd "$TW_EXACT_DIR" && make exact)
  fi
fi

if [[ ! -f "$TW_MAIN_CLASS" ]]; then
  echo "error: tw-exact exists, but compiled classes are missing under ${TW_EXACT_DIR}/tw/exact" >&2
  echo "hint: build the solver with 'make exact' in the PACE2017-TrackA checkout" >&2
  exit 1
fi

base_name="$(basename "$INPUT_DUMP")"
stem="${base_name%.*}"

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$ROOT_DIR/out/pipeline/$stem"
fi
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [[ -z "$OUTPUT_DUMP" ]]; then
  OUTPUT_DUMP="$WORK_DIR/${stem}.solved.txt"
fi
if [[ "$OUTPUT_DUMP" != /* ]]; then
  OUTPUT_DUMP="$WORK_DIR/$OUTPUT_DUMP"
fi

GR_PATH="$WORK_DIR/instance.gr"
MAP_PATH="$WORK_DIR/instance.map"
TD_PATH="$WORK_DIR/instance.td"
NICE_TD_PATH="$WORK_DIR/instance.nice.td"

log() {
  printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"
}

run_with_cap() {
  if [[ -n "$MEM_CAP_MB" ]]; then
    timeout "${TIMEOUT_SEC}s" bash -c 'ulimit -Sv "$1"; shift; exec "$@"' _ "$((MEM_CAP_MB * 1024))" "$@"
  else
    timeout "${TIMEOUT_SEC}s" "$@"
  fi
}

log "Input: $INPUT_DUMP"
log "Work dir: $WORK_DIR"
log "Output: $OUTPUT_DUMP"
log "Timeout: ${TIMEOUT_MS} ms (${TIMEOUT_SEC}s) per stage"
if [[ -n "$MEM_CAP_MB" ]]; then
  log "Memory cap: ${MEM_CAP_MB} MB via ulimit -Sv"
else
  log "Memory cap: disabled"
fi
log "External treewidth solver: $TW_EXACT"

log "Stage 1/4: Build interaction graph (.gr/.map)"
"$LS_OPT" --input "$INPUT_DUMP" --emit-gr "$GR_PATH" --emit-map "$MAP_PATH"

log "Stage 2/4: Compute tree decomposition (.td)"
set +e
if [[ -n "$MEM_CAP_MB" ]]; then
  (cd "$TW_EXACT_DIR" && timeout "${TIMEOUT_SEC}s" bash -c 'ulimit -Sv "$1"; exec "$2"' _ "$((MEM_CAP_MB * 1024))" "$TW_EXACT" < "$GR_PATH" > "$TD_PATH")
else
  (cd "$TW_EXACT_DIR" && timeout "${TIMEOUT_SEC}s" "$TW_EXACT" < "$GR_PATH" > "$TD_PATH")
fi
code=$?
set -e
if [[ $code -ne 0 ]]; then
  if [[ $code -eq 124 ]]; then
    echo "error: Stage 2 timeout after ${TIMEOUT_MS} ms while running tw-exact" >&2
  else
    echo "error: Stage 2 failed with exit code $code (tw-exact)" >&2
  fi
  exit $code
fi

log "Stage 3/4: Convert/check nice TD"
set +e
run_with_cap "$LS_OPT" --input-td "$TD_PATH" --to-nice "$NICE_TD_PATH" --check-nice
code=$?
set -e
if [[ $code -ne 0 ]]; then
  if [[ $code -eq 124 ]]; then
    echo "error: Stage 3 timeout after ${TIMEOUT_MS} ms while converting/checking nice TD" >&2
  else
    echo "error: Stage 3 failed with exit code $code (nice TD conversion/check)" >&2
  fi
  exit $code
fi

log "Stage 4/4: Exact solve + write layout_selection"
set +e
run_with_cap "$LS_OPT" --input "$INPUT_DUMP" --input-td "$NICE_TD_PATH" --timeout-ms "$TIMEOUT_MS" --solve-and-write "$OUTPUT_DUMP"
code=$?
set -e
if [[ $code -ne 0 ]]; then
  if [[ $code -eq 124 ]]; then
    echo "error: Stage 4 timeout after ${TIMEOUT_MS} ms while solving DP" >&2
  else
    echo "error: Stage 4 failed with exit code $code (DP solve)" >&2
  fi
  exit $code
fi

log "Done"
log "Artifacts:"
log "  $GR_PATH"
log "  $MAP_PATH"
log "  $TD_PATH"
log "  $NICE_TD_PATH"
log "  $OUTPUT_DUMP"
