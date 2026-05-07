#!/usr/bin/env python3
"""
SynthRAD 2023 Preprocessing Pipeline — Research Grade
======================================================
Task 1 only (MRI → CT). CBCT / Task 2 is ignored entirely.

Pipeline per patient:
  1.  Load            .nii.gz  via SimpleITK
  2.  [Placeholder]   Rigid registration MRI→CT (data arrives pre-registered)
  3.  Reorient        → RAS
  4.  Resample        → 1.5 × 1.5 × 1.5 mm  (B-spline order-3 for images, NN for mask)
  5.  Crop            → mask bounding box + 5-vox margin
  6.  Normalise
        CT  : clip to soft-tissue window [-1000, 1000] HU → scale to [-1, 1]
        MRI : percentile clip [p1, p99] (mask-only) → z-score (mask-only)
  7.  Quality check   discard if volume too small / fully empty
  8.  Save            <id>_ct.pt  /  <id>_mr.pt  (float32, [1,D,H,W])
  9.  Save            <id>_stats.json  (per-patient intensity statistics)

After all patients:
  10. 80/20 patient-wise train/val split  (seed = 42, shuffled once globally)
  11. Save  manifest.csv   — one row per processed file; columns:
            patient_id, region, split, modality, pt_path

Usage:
    python preprocess.py \
        --data2023  /path/to/.ignore/data/2023 \
        --output    /path/to/dataset/processed \
        [--workers  N]   [--dry_run]   [--patient_id 1BA001]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch

# optional tqdm — gracefully degrade if not installed
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

SEED             = 42
TRAIN_RATIO      = 0.80
TARGET_SPACING_BR = (1.0, 1.0, 1.0)          # mm (Brain - high isotropic detail)
TARGET_SPACING_PE = (1.0, 1.0, 2.5)          # mm (Pelvis - high axial, lower slice res)
CROP_MARGIN_VOX  = 5
CT_CLIP_MIN      = -1000.0                   # soft-tissue window (HU)
CT_CLIP_MAX      = 1000.0
MRI_PLOW         = 1.0                       # percentile low
MRI_PHIGH        = 99.0                      # percentile high
MIN_VOXEL_COUNT  = 1_000                     # minimum foreground voxels

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def _find_file(folder: Path, stem: str) -> Optional[Path]:
    """Find <stem>.nii.gz or common variants inside folder."""
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def _load(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


# ────────────────────────────────────────────────────────────────────────────
# Step 3 — reorient to RAS
# ────────────────────────────────────────────────────────────────────────────

def reorient_ras(img: sitk.Image) -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


# ────────────────────────────────────────────────────────────────────────────
# Step 4 — resample
# ────────────────────────────────────────────────────────────────────────────

def resample(
    img: sitk.Image,
    new_spacing: tuple[float, float, float],
    is_mask: bool = False,
) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size    = img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    return sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )


# ────────────────────────────────────────────────────────────────────────────
# Step 5 — crop to mask bounding box
# ────────────────────────────────────────────────────────────────────────────

def crop_to_mask(
    images: dict[str, sitk.Image],
    mask: sitk.Image,
    margin: int = CROP_MARGIN_VOX,
) -> dict[str, sitk.Image]:
    """Crop a dict of images using the bounding box of the binary mask."""
    arr = sitk.GetArrayFromImage(mask)          # [D, H, W] (numpy z,y,x)
    fg  = np.argwhere(arr > 0)
    if fg.size == 0:
        return images

    lo  = fg.min(axis=0)                        # [z_min, y_min, x_min]
    hi  = fg.max(axis=0)                        # [z_max, y_max, x_max]
    sz  = mask.GetSize()                        # (X, Y, Z) in sitk

    lower = [
        max(0, int(lo[2]) - margin),            # x
        max(0, int(lo[1]) - margin),            # y
        max(0, int(lo[0]) - margin),            # z
    ]
    upper = [
        max(0, sz[0] - min(sz[0], int(hi[2]) + 1 + margin)),
        max(0, sz[1] - min(sz[1], int(hi[1]) + 1 + margin)),
        max(0, sz[2] - min(sz[2], int(hi[0]) + 1 + margin)),
    ]

    cf = sitk.CropImageFilter()
    cf.SetLowerBoundaryCropSize(lower)
    cf.SetUpperBoundaryCropSize(upper)
    return {k: cf.Execute(v) for k, v in images.items()}


# ────────────────────────────────────────────────────────────────────────────
# Step 6a — CT normalisation  →  [-1, 1]
# ────────────────────────────────────────────────────────────────────────────

def normalise_ct(img: sitk.Image) -> np.ndarray:
    """Clip soft-tissue HU window then scale to [-1, 1]."""
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    a = np.clip(a, CT_CLIP_MIN, CT_CLIP_MAX)
    a = (a - CT_CLIP_MIN) / (CT_CLIP_MAX - CT_CLIP_MIN) * 2.0 - 1.0   # → [-1,1]
    return a


# ────────────────────────────────────────────────────────────────────────────
# Step 6b — MRI normalisation  →  z-score (mask-aware, percentile-clipped)
# ────────────────────────────────────────────────────────────────────────────

def normalise_mri(
    img: sitk.Image,
    mask: Optional[sitk.Image] = None,
) -> tuple[np.ndarray, dict]:
    """
    1. Extract foreground values using mask (or intensity threshold fallback).
    2. Percentile-clip to [p1, p99] — removes scanner noise / clamps outliers.
    3. Z-score using foreground mean & std.
    Returns (normalised_array, stats_dict).
    """
    a = sitk.GetArrayFromImage(img).astype(np.float32)

    # foreground selection
    if mask is not None:
        m = sitk.GetArrayFromImage(mask).astype(bool)
    else:
        m = a > (np.percentile(a, 5))

    fg = a[m]
    if fg.size < MIN_VOXEL_COUNT:
        raise ValueError(f"MRI foreground too small ({fg.size} voxels)")

    p_lo  = float(np.percentile(fg, MRI_PLOW))
    p_hi  = float(np.percentile(fg, MRI_PHIGH))
    a     = np.clip(a, p_lo, p_hi)
    fg    = a[m]           # recompute fg after clip

    mu    = float(fg.mean())
    sigma = float(fg.std())
    if sigma < 1e-6:
        sigma = 1.0

    a = (a - mu) / sigma
    stats = {
        "mri_percentile_low":  p_lo,
        "mri_percentile_high": p_hi,
        "mri_fg_mean":         mu,
        "mri_fg_std":          sigma,
        "mri_foreground_vox":  int(fg.size),
    }
    return a, stats


# ────────────────────────────────────────────────────────────────────────────
# Dataset scanning (2023 only)
# ────────────────────────────────────────────────────────────────────────────

def scan_2023(root: Path) -> list[dict]:
    """
    Walk the SynthRAD 2023 directory tree and return all Task1 (MRI→CT) patients.
    Layout: <root>/SynthRAD2023_*/[sub/]Task1/<region>/<patient_id>/
    """
    patients: list[dict] = []

    def _walk(base: Path):
        task1 = base / "Task1"
        if task1.exists():
            for region_dir in sorted(task1.iterdir()):
                if not region_dir.is_dir() or region_dir.name in ("overview", "overviews"):
                    continue
                for pat_dir in sorted(region_dir.iterdir()):
                    if not pat_dir.is_dir():
                        continue
                    ct   = _find_file(pat_dir, "ct")
                    mr   = _find_file(pat_dir, "mr")
                    mask = _find_file(pat_dir, "mask")
                    if ct is None and mr is None:
                        continue
                    patients.append({
                        "id":        pat_dir.name,
                        "region":    region_dir.name,
                        "ct_path":   str(ct)   if ct   else None,
                        "mr_path":   str(mr)   if mr   else None,
                        "mask_path": str(mask) if mask else None,
                        "is_paired": ct is not None and mr is not None,
                    })
        else:
            for sub in sorted(base.iterdir()):
                if sub.is_dir():
                    _walk(sub)

    for part in sorted(root.iterdir()):
        if part.is_dir():
            _walk(part)

    # deduplicate
    seen: set[tuple] = set()
    out: list[dict] = []
    for p in patients:
        key = (p["id"], p["region"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# ────────────────────────────────────────────────────────────────────────────
# 80/20 Stratified Split
# ────────────────────────────────────────────────────────────────────────────

def make_splits(patients: list[dict], seed: int = SEED) -> dict[str, str]:
    """
    Stratified patient-wise split: ensure each region (brain/pelvis)
    is balanced 80/20 across train/val.
    """
    rng = random.Random(seed)
    region_to_ids = defaultdict(list)
    for p in patients:
        region_to_ids[p["region"]].append(p["id"])

    splits: dict[str, str] = {}
    for region, ids in region_to_ids.items():
        # Get unique IDs for this region
        unique_ids = sorted(list(set(ids)))
        rng.shuffle(unique_ids)

        n_train = int(round(len(unique_ids) * TRAIN_RATIO))
        for i, pid in enumerate(unique_ids):
            splits[pid] = "train" if i < n_train else "val"

    return splits


# ────────────────────────────────────────────────────────────────────────────
# Per-patient worker
# ────────────────────────────────────────────────────────────────────────────

def process_patient(args: tuple) -> Optional[dict]:
    """
    Process one patient.  Returns a record dict on success, None on failure.
    Intended to run in a multiprocessing pool.
    """
    pat, output_root, dry_run = args
    pid    = pat["id"]
    region = pat["region"]

    out_dir = Path(output_root) / region
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"{pid}_stats.json"

    # --- resume guard ---
    # We check for the presence of CT, MR, and Mask AND verify spacing matches target.
    target_spacing = TARGET_SPACING_BR if region == "brain" else TARGET_SPACING_PE
    if stats_path.exists() and (out_dir / f"{pid}_mask.pt").exists():
        try:
            stats = json.loads(stats_path.read_text())
            # If spacing doesn't match target, we must re-process
            if list(target_spacing) == stats.get("spacing_mm"):
                return {
                    "patient_id": pid, "region": region,
                    "ct_path": str(out_dir / f"{pid}_ct.pt"),
                    "mr_path": str(out_dir / f"{pid}_mr.pt"),
                    "mask_path": str(out_dir / f"{pid}_mask.pt"),
                    "is_paired":  pat["is_paired"],
                    "skipped": True,
                }
        except Exception:
            pass  # corrupt stats → re-process

    if dry_run:
        return {"patient_id": pid, "region": region, "skipped": True}

    try:
        # ── Step 1: load ──────────────────────────────────────────────────
        ct_img = mr_img = mask_img = None
        if pat["ct_path"]:
            ct_img = _load(Path(pat["ct_path"]))
        if pat["mr_path"]:
            mr_img = _load(Path(pat["mr_path"]))
        if pat["mask_path"]:
            mask_img = _load(Path(pat["mask_path"]))

        # ── Step 2: registration placeholder (data is pre-registered) ─────
        # mr_img = rigid_register(ct_img, mr_img)  # activate for raw DICOM

        # ── Step 3: reorient → RAS ────────────────────────────────────────
        if ct_img:   ct_img   = reorient_ras(ct_img)
        if mr_img:   mr_img   = reorient_ras(mr_img)
        if mask_img: mask_img = reorient_ras(mask_img)

        # ── Step 4: resample ───────────────────────────────────────────
        spacing = TARGET_SPACING_BR if region == "brain" else TARGET_SPACING_PE
        if ct_img:   ct_img   = resample(ct_img,   spacing, is_mask=False)
        if mr_img:   mr_img   = resample(mr_img,   spacing, is_mask=False)
        if mask_img: mask_img = resample(mask_img, spacing, is_mask=True)

        # build fallback mask from MRI intensity if missing
        if mask_img is None and mr_img is not None:
            a = sitk.GetArrayFromImage(mr_img)
            fb = (a > np.percentile(a, 5)).astype(np.uint8)
            mask_img = sitk.GetImageFromArray(fb)
            mask_img.CopyInformation(mr_img)

        # ── Step 5: crop ──────────────────────────────────────────────────
        to_crop: dict[str, sitk.Image] = {}
        if ct_img:   to_crop["ct"]   = ct_img
        if mr_img:   to_crop["mr"]   = mr_img
        if mask_img: to_crop["mask"] = mask_img

        if mask_img is not None:
            cropped  = crop_to_mask(to_crop, mask_img, CROP_MARGIN_VOX)
            ct_img   = cropped.get("ct")
            mr_img   = cropped.get("mr")
            mask_img = cropped.get("mask")

        # ── Steps 6 + 7: normalise & quality check ────────────────────────
        stats: dict = {
            "id": pid, "region": region,
            "ct_clip_min": CT_CLIP_MIN, "ct_clip_max": CT_CLIP_MAX,
        }

        ct_arr = mr_arr = None

        if ct_img is not None:
            ct_arr = normalise_ct(ct_img)
            fg_ct  = ct_arr[sitk.GetArrayFromImage(mask_img).astype(bool)] if mask_img else ct_arr.ravel()
            if fg_ct.size < MIN_VOXEL_COUNT:
                raise ValueError(f"CT foreground too small ({fg_ct.size} voxels)")
            stats.update({
                "ct_shape_DHW": list(ct_arr.shape),
                "ct_fg_mean":   float(fg_ct.mean()),
                "ct_fg_std":    float(fg_ct.std()),
                "ct_min":       float(ct_arr.min()),
                "ct_max":       float(ct_arr.max()),
                "spacing_mm":   list(TARGET_SPACING_BR if region == "brain" else TARGET_SPACING_PE),
            })

        if mr_img is not None:
            mr_arr, mri_stats = normalise_mri(mr_img, mask_img)
            stats.update(mri_stats)
            stats["mr_shape_DHW"] = list(mr_arr.shape)

        # ── Step 8: save .pt ──────────────────────────────────────────────
        ct_pt_path = mr_pt_path = ma_pt_path = None
        if ct_arr is not None:
            ct_pt_path = out_dir / f"{pid}_ct.pt"
            torch.save(torch.from_numpy(ct_arr).unsqueeze(0).contiguous(), ct_pt_path)
        if mr_arr is not None:
            mr_pt_path = out_dir / f"{pid}_mr.pt"
            torch.save(torch.from_numpy(mr_arr).unsqueeze(0).contiguous(), mr_pt_path)
        if mask_img is not None:
            ma_pt_path = out_dir / f"{pid}_mask.pt"
            ma_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
            torch.save(torch.from_numpy(ma_arr).unsqueeze(0).contiguous(), ma_pt_path)

        # ── Step 9: save stats ────────────────────────────────────────────
        stats_path.write_text(json.dumps(stats, indent=2))

        return {
            "patient_id": pid,
            "region":     region,
            "is_paired":  pat["is_paired"],
            "ct_path":    str(ct_pt_path)   if ct_pt_path   else None,
            "mr_path":    str(mr_pt_path)   if mr_pt_path   else None,
            "mask_path":  str(ma_pt_path)   if ma_pt_path   else None,
            "skipped":    False,
        }

    except Exception as exc:
        log.error(f"[FAIL] {pid}: {exc}", exc_info=False)
        return None


# ────────────────────────────────────────────────────────────────────────────
# Manifest CSV writer (Steps 10–11)
# ────────────────────────────────────────────────────────────────────────────

def write_manifest(
    records: list[dict],
    splits:  dict[str, str],
    out_dir: Path,
) -> Path:
    """Write manifest.csv with one row per (patient, modality)."""
    rows = []
    for rec in records:
        if rec is None:
            continue
        pid    = rec["patient_id"]
        split  = splits.get(pid, "unknown")
        region = rec["region"]

        for mod, key in [("mr", "mr_path"), ("ct", "ct_path")]:
            val = rec.get(key)
            if val:
                # Use relative path for portability
                rel_path = os.path.relpath(val, out_dir)
                mask_rel = os.path.relpath(rec["mask_path"], out_dir) if rec.get("mask_path") else None
                rows.append({
                    "patient_id": pid,
                    "region":     region,
                    "split":      split,
                    "modality":   mod,
                    "pt_path":    rel_path,
                    "mask_path":  mask_rel,
                    "is_paired":  rec.get("is_paired", False),
                })

    manifest_path = out_dir / "manifest.csv"
    fieldnames = ["patient_id", "region", "split", "modality", "pt_path", "mask_path", "is_paired"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── split summary ──────────────────────────────────────────────────────
    summary: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[r["split"]][r["region"]] += 1
    log.info("Dataset split summary:")
    for spl in ("train", "val"):
        total = sum(summary[spl].values()) // 2   # /2 because ct+mr rows
        log.info(f"  {spl}: {total} patients  "
                 + "  ".join(f"{reg}={c//2}" for reg, c in sorted(summary[spl].items())))
    return manifest_path


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data2023",   type=Path, required=True,
                   help="Root of the SynthRAD 2023 data directory")
    p.add_argument("--output",     type=Path, default=Path("/usershome/cs671_user4/SynthRAD2023_Dataset"),
                   help="Output root directory for the processed dataset")
    p.add_argument("--workers",    type=int,  default=1,
                   help="Parallel workers (0 = all CPUs)")
    p.add_argument("--dry_run",    action="store_true",
                   help="Scan only — no files written")
    p.add_argument("--patient_id", type=str,  default=None,
                   help="Process only this patient ID (debugging)")
    p.add_argument("--seed",       type=int,  default=SEED,
                   help=f"Random seed for split (default: {SEED})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── global seeds ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── discover ───────────────────────────────────────────────────────────
    if not args.data2023.exists():
        log.error(f"Data path not found: {args.data2023}")
        sys.exit(1)

    all_patients = scan_2023(args.data2023)
    log.info(f"Discovered {len(all_patients)} Task1 patients in SynthRAD 2023")

    # ── stratified split ───────────────────────────────────────────────────
    splits = make_splits(all_patients, seed=args.seed)

    # ── verification ───────────────────────────────────────────────────────
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for p in all_patients:
        s = splits[p["id"]]
        if s == "train": train_counts[p["region"]] += 1
        else:            val_counts[p["region"]] += 1

    log.info("Split Verification:")
    for reg in sorted(set(list(train_counts.keys()) + list(val_counts.keys()))):
        t = train_counts[reg]
        v = val_counts[reg]
        total = t + v
        log.info(f"  {reg:8s}: total={total:3d} | train={t:3d} ({t/total:.1%}) | val={v:3d} ({v/total:.1%})")

    # check for leakage
    train_ids = {p["id"] for p in all_patients if splits[p["id"]] == "train"}
    val_ids = {p["id"] for p in all_patients if splits[p["id"]] == "val"}
    leakage = train_ids & val_ids
    if leakage:
        log.error(f"FATAL: Patient leakage detected! {leakage}")
        sys.exit(1)
    else:
        log.info("Leakage check: PASSED (0 overlap)")

    if args.patient_id:
        all_patients = [p for p in all_patients if p["id"] == args.patient_id]
        if not all_patients:
            log.error(f"Patient '{args.patient_id}' not found.")
            sys.exit(1)

    if not all_patients:
        log.error("No patients found.")
        sys.exit(1)

    if args.dry_run:
        log.info("Dry-run — listing IDs only.")
        for p in all_patients:
            print(f"  {splits[p['id']]:5s}  {p['region']:8s}  {p['id']}")
        return

    # ── process ────────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    tasks   = [(p, args.output, False) for p in all_patients]
    n_work  = args.workers if args.workers > 0 else cpu_count()

    if n_work == 1:
        iterable = tqdm(tasks, desc="Processing", unit="pt") if _TQDM else tasks
        records  = [process_patient(t) for t in iterable]
    else:
        log.info(f"Using {n_work} parallel workers.")
        with Pool(processes=n_work) as pool:
            if _TQDM:
                records = list(tqdm(
                    pool.imap(process_patient, tasks),
                    total=len(tasks), desc="Processing", unit="pt",
                ))
            else:
                records = pool.map(process_patient, tasks)

    ok   = [r for r in records if r is not None]
    fail = len(records) - len(ok)
    log.info(f"Processed: {len(ok)}  failed: {fail}")

    # ── manifest ───────────────────────────────────────────────────────────
    manifest = write_manifest(ok, splits, args.output)
    log.info(f"Manifest saved → {manifest}")

    # ── save config snapshot ───────────────────────────────────────────────
    config = {
        "data2023":      str(args.data2023),
        "output":        str(args.output),
        "seed":          args.seed,
        "train_ratio":   TRAIN_RATIO,
        "target_spacing_brain": list(TARGET_SPACING_BR),
        "target_spacing_pelvis":list(TARGET_SPACING_PE),
        "ct_clip":       [CT_CLIP_MIN, CT_CLIP_MAX],
        "ct_scale":      "[-1, 1]",
        "mri_plow":      MRI_PLOW,
        "mri_phigh":     MRI_PHIGH,
        "interpolation": "BSpline-3 (images) / NearestNeighbour (mask)",
        "crop_margin_vox": CROP_MARGIN_VOX,
    }
    (args.output / "preprocessing_config.json").write_text(json.dumps(config, indent=2))
    log.info("Done.")


if __name__ == "__main__":
    main()
