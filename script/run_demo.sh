#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/inference/head.yaml}"
OUTPUT="${OUTPUT:-outputs/phctqa_report.json}"
INPUT="${INPUT:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
DEVICE="${DEVICE:-cpu}"

usage() {
  cat <<'EOF'
Usage: bash script/run_demo.sh [options]
  --config PATH       Inference config. Default: config/inference/head.yaml
  --input PATH        DICOM directory or NIfTI case.
  --output PATH       JSON report path. Default: outputs/phctqa_report.json
  --weights-dir DIR   Override weights_dir in config.
  --device DEVICE     cpu | cuda | npu. Default: cpu
  --install           pip install -e . first.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG="$2"; shift 2;;
    --input)    INPUT="$2"; shift 2;;
    --output)   OUTPUT="$2"; shift 2;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2;;
    --device)   DEVICE="$2"; shift 2;;
    --install)  "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"; shift;;
    -h|--help)  usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

mkdir -p "$(dirname "${OUTPUT}")"

CMD=(
  "${PYTHON_BIN}" -m phctqa.inference
  --config "${CONFIG}"
  --output "${OUTPUT}"
  --device "${DEVICE}"
)
[[ -n "${INPUT}" ]]     && CMD+=(--input "${INPUT}")
[[ -n "${WEIGHTS_DIR}" ]] && CMD+=(--weights-dir "${WEIGHTS_DIR}")

PYTHONPATH="${ROOT_DIR}" "${CMD[@]}"