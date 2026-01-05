#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${ROOT_DIR}/test_project.py" --root "${ROOT_DIR}" --skip-tests --verbose "$@"
