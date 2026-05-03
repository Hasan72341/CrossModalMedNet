"""
pix2pix_standard/data.py
-----------------------
Regional Dataset for Pix2Pix CT-MRI Translation.
Supports on-the-fly slice extraction from 3D volumes.
"""
import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch.nn.functional as F

from .config import DataConfig, TrainingConfig


class SynthRADDataset(Dataset):
    """Dataset for extracting 2D slices from 3D CT-MRI volumes."""
    
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        region: str | None = None,
        image_size: int = 256,
        slice_axis: str = "axial",
        slice_strategy: Literal["random", "center", "fixed"] = "random",
        fixed_slice_index: int | None = None,
    ):
        self.root = os.path.dirname(os.path.abspath(manifest_path))
        df = pd.read_csv(manifest_path)

        # Filter by split and region
        df = df[df["split"] == split]
        if region:
            df = df[df["region"] == region]

        self.image_size = image_size
        self.slice_axis = slice_axis.lower()
        self.slice_strategy = slice_strategy
        self.fixed_slice_index = fixed_slice_index

        # Axis dimension in [C, D, H, W]
        self.axis_dim = {"axial": 1, "coronal": 2, "sagittal": 3}.get(self.slice_axis, 1)

        # Create pairs by patient_id and region
        mr_df = df[df["modality"] == "mr"]
        ct_df = df[df["modality"] == "ct"]
        paired_df = pd.merge(mr_df, ct_df, on=["patient_id", "region"], suffixes=("_mr", "_ct"))
        
        self.mri_paths = paired_df["pt_path_mr"].tolist()
        self.ct_paths = paired_df["pt_path_ct"].tolist()
        # Derive mask paths: e.g. brain/1BA001_ct.pt -> brain/1BA001_mask.pt
        self.mask_paths = [p.replace("_ct.pt", "_mask.pt").replace("_mr.pt", "_mask.pt") for p in self.ct_paths]
        self.ids = paired_df["patient_id"].tolist()
        self.regions = paired_df["region"].tolist()
        self.length = len(paired_df)

        if self.length == 0:
            print(f"Warning: No paired images found for split={split}, region={region}")

    def __len__(self) -> int:
        return self.length

    def _load_pt(self, rel_path: str) -> torch.Tensor:
        abs_path = os.path.join(self.root, rel_path)
        return torch.load(abs_path, map_location="cpu")

    def _get_slice_indices(self, max_dim: int) -> list[int]:
        if self.slice_strategy == "center":
            return [max_dim // 2]
        if self.slice_strategy == "fixed" and self.fixed_slice_index is not None:
            return [max(0, min(max_dim - 1, self.fixed_slice_index))]
        return [random.randint(0, max_dim - 1)]

    def _extract_slices(self, vol: torch.Tensor, indices: list[int]) -> torch.Tensor:
        # vol shape is [C, D, H, W]
        if self.slice_axis == "axial":
            slices = vol[:, indices, :, :]
        elif self.slice_axis == "coronal":
            # [C, D, H, W] -> [H, C, D, W] -> [C, D, W] ?
            slices = vol[:, :, indices, :].permute(2, 0, 1, 3) 
        else:
            slices = vol[:, :, :, indices].permute(3, 0, 1, 2)
            
        return slices.squeeze(0) # [C, H, W] or similar

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Implementation of on-the-fly slicing
        # We try a few times to find a slice with actual content (standard medical image practice)
        for _ in range(5):
            mri_vol = self._load_pt(self.mri_paths[index])
            ct_vol = self._load_pt(self.ct_paths[index])
            mask_vol = self._load_pt(self.mask_paths[index])

            # Get indices (support multiple slices if needed, but here 1)
            depth = mri_vol.shape[self.axis_dim]
            indices = self._get_slice_indices(depth)
            
            mri = self._extract_slices(mri_vol, indices)
            ct = self._extract_slices(ct_vol, indices)
            mask = self._extract_slices(mask_vol, indices)

            # Resize
            mri = F.interpolate(mri.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            ct = F.interpolate(ct.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.float().unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)

            # Intensity Normalization
            def normalize_volume(x, identity="mri"):
                if identity == "ct":
                    # CT: Clamp to HU range -1000 to 1000
                    x = torch.clamp(x, -1000.0, 1000.0)
                    return (x + 1000.0) / 2000.0 * 2.0 - 1.0
                else:
                    # MRI: Robust Min-Max
                    v_min, v_max = x.min(), x.max()
                    if v_max > v_min:
                        return 2.0 * (x - v_min) / (v_max - v_min + 1e-8) - 1.0
                    return x * 0 - 1.0

            mri = normalize_volume(mri, "mri")
            ct = normalize_volume(ct, "ct")

            # Check if slice is valid (not just background)
            if ct.mean() > -0.9 and ct.std() > 0.05:
                break
            
            # If not valid, try a different center/random index or just stop trying
            if self.slice_strategy != "fixed":
                continue
            else:
                break

        meta = {
            'patient_id': self.ids[index],
            'region': self.regions[index],
            'index': index,
            'ct_path': self.ct_paths[index],
            'mri_path': self.mri_paths[index],
            'mask_path': self.mask_paths[index]
        }
        
        return ct, mri, mask, meta


def create_dataloaders(
    config: DataConfig,
    training_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    
    manifest_path = config.manifest_path
    
    # Training Loader
    train_datasets = []
    for region in config.train_regions:
        train_datasets.append(SynthRADDataset(
            manifest_path=manifest_path,
            split="train",
            region=region,
            image_size=config.image_size,
            slice_axis=config.slice_axis,
            slice_strategy=config.slice_strategy
        ))
    
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Test/Val Loader
    test_datasets = []
    for region in config.eval_regions:
        test_datasets.append(SynthRADDataset(
            manifest_path=manifest_path,
            split="val", # Manifest uses "val"
            region=region,
            image_size=config.image_size,
            slice_axis=config.slice_axis,
            slice_strategy="center" # Center slice for evaluation consistency
        ))
        
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(training_config, "test_batch_size", 1),
        shuffle=False,
        num_workers=1,
        pin_memory=True
    ) if test_dataset else None

    return train_loader, None, test_loader
