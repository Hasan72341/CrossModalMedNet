import sys
import torch
from data.dataset import SynthRADDataset
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
manifest_path = os.environ.get(
    "MANIFEST_PATH",
    str(PROJECT_ROOT.parent / "SynthRAD2023_Dataset" / "manifest.csv"),
)
dataset = SynthRADDataset(manifest_path, split="train", unpaired=True, mode="2d")
item = dataset[0]
mri = item['mri']
ct = item['ct']
print("MRI stats:", mri.min(), mri.max(), mri.mean(), mri.shape)
print("CT stats:", ct.min(), ct.max(), ct.mean(), ct.shape)

dataset_paired = SynthRADDataset(manifest_path, split="train", unpaired=False, mode="2d")
item2 = dataset_paired[0]
mri2 = item2['mri']
ct2 = item2['ct']
print("MRI paired stats:", mri2.min(), mri2.max(), mri2.mean(), mri2.shape)
print("CT paired stats:", ct2.min(), ct2.max(), ct2.mean(), ct2.shape)
