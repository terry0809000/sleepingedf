#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-}"

if [ -z "$REPO_DIR" ]; then
  echo "Usage: bash replace_original_notebook.sh /path/to/local/sleepingedf"
  exit 1
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Error: $REPO_DIR is not a Git repository."
  exit 1
fi

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$REPO_DIR/notebooks" "$REPO_DIR/scripts" "$REPO_DIR/docs"

cp "$PACKAGE_DIR/notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb"    "$REPO_DIR/notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb"

cp "$PACKAGE_DIR/notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb"    "$REPO_DIR/notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb"

cp "$PACKAGE_DIR/scripts/sleep_edf_pipeline_record_boundary_fixed.py"    "$REPO_DIR/scripts/sleep_edf_pipeline_record_boundary_fixed.py"

cp "$PACKAGE_DIR/README.md" "$REPO_DIR/README.md"
cp "$PACKAGE_DIR/requirements.txt" "$REPO_DIR/requirements.txt"
cp "$PACKAGE_DIR/.gitignore" "$REPO_DIR/.gitignore"
cp "$PACKAGE_DIR/docs/SECTION_INDEX.md" "$REPO_DIR/docs/SECTION_INDEX.md"

cd "$REPO_DIR"

echo
echo "Replacement files copied. Review with:"
echo "  git status"
echo "  git diff --stat"
echo
echo "Then commit and push with:"
echo "  git add notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb scripts/sleep_edf_pipeline_record_boundary_fixed.py README.md requirements.txt .gitignore docs/SECTION_INDEX.md"
echo "  git commit -m 'Replace notebook with BSPC final strengthened Sleep-EDF pipeline'"
echo "  git push origin main"
