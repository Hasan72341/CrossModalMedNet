#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TECTONIC_BIN="${TECTONIC_BIN:-tectonic}"
"$TECTONIC_BIN" cyclegan_brain/cyclegan_brain_report.tex > compile.log 2>&1
"$TECTONIC_BIN" cyclegan_pelvis/cyclegan_pelvis_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" pix2pix_brain/pix2pix_brain_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" pix2pix_pelvis/pix2pix_pelvis_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" paired_diffusion_brain/paired_diffusion_brain_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" paired_diffusion_pelvis/paired_diffusion_pelvis_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" unpaired_diffusion_brain/unpaired_diffusion_brain_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" unpaired_diffusion_pelvis/unpaired_diffusion_pelvis_report.tex >> compile.log 2>&1
"$TECTONIC_BIN" all_models_comparison/all_models_comparison_report.tex >> compile.log 2>&1
echo "DONE" >> compile.log
