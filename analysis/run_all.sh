#!/usr/bin/env bash
# Run the complete ESM-MSR analysis suite (or pass CLI flags through to analysis.run, e.g. --fast, --only p01).
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ANALYSIS_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${ANALYSIS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON:-python}"

exec "${PYTHON_BIN}" -m analysis.run "$@"
