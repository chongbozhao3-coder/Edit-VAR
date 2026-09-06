#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -u edit_video.py "$@"
