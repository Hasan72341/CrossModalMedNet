import torch
import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT.parent / "SynthRAD2023_Dataset"))
df = pd.read_csv(DATA_ROOT / 'manifest.csv')
mr = df[df['modality']=='mr'].iloc[0]['pt_path']
ct = df[df['modality']=='ct'].iloc[0]['pt_path']
mr_vol = torch.load(DATA_ROOT / mr, map_location='cpu', weights_only=True)
ct_vol = torch.load(DATA_ROOT / ct, map_location='cpu', weights_only=True)
print(f"MR min: {mr_vol.min()}, max: {mr_vol.max()}, shape: {mr_vol.shape}")
print(f"CT min: {ct_vol.min()}, max: {ct_vol.max()}, shape: {ct_vol.shape}")
