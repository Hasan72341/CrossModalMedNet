#!/bin/bash
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DOCS_DIR/refactor_reports.py"
