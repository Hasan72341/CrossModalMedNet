import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class SynthRADDataset(Dataset):
    """Unified Dataset for SynthRAD 2023 CT-MRI Translation."""
    
    def __init__(
        self,
        ct_images: List[str],
        mri_images: List[str],
        image_size: int = 256,
        mode: str = "train",
        unpaired: bool = False,
        transform: Optional[transforms.Compose] = None,
    ):
        self.ct_images = sorted(ct_images)
        self.mri_images = sorted(mri_images)
        self.image_size = image_size
        self.mode = mode
        self.unpaired = unpaired
        self.transform = transform if transform else self._default_transform()
        
        if not unpaired and len(self.ct_images) != len(self.mri_images):
            raise ValueError(f"Paired mode requires equal CT and MRI images. Found {len(self.ct_images)} CT and {len(self.mri_images)} MRI.")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Map [0, 1] to [-1, 1]
        ])

    def __len__(self) -> int:
        return max(len(self.ct_images), len(self.mri_images))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        ct_path = Path(self.ct_images[index % len(self.ct_images)])
        
        if self.unpaired:
            mri_idx = random.randrange(len(self.mri_images))
        else:
            mri_idx = index % len(self.mri_images)
        mri_path = Path(self.mri_images[mri_idx])
        
        ct = self._load_image(ct_path)
        mri = self._load_image(mri_path)
        
        ct = self.transform(ct)
        mri = self.transform(mri)
        
        meta = {
            'ct_path': str(ct_path),
            'mri_path': str(mri_path),
            'ct_id': ct_path.stem,
            'mri_id': mri_path.stem
        }
        
        return ct, mri, meta

    def _load_image(self, path: Path) -> Image.Image:
        if path.suffix.lower() == '.pt':
            data = torch.load(path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            # Handle different shapes [C, H, W] or [H, W]
            if data.ndim == 3:
                data = data[0] if data.shape[0] < 10 else data[data.shape[0]//2]
            return Image.fromarray(data.astype(np.float32), mode='F')
        return Image.open(path).convert("L")

def get_patient_wise_splits(root_dir: str, region: str, split_ratio: float = 0.8, seed: int = 42):
    """Utility to get patient-wise train/test splits."""
    region_dir = Path(root_dir) / region
    ct_files = sorted(list(region_dir.glob("*_ct.*")))
    
    patients = sorted(list(set(f.name.split('_')[0] for f in ct_files)))
    random.Random(seed).shuffle(patients)
    
    split_idx = int(len(patients) * split_ratio)
    train_patients = set(patients[:split_idx])
    test_patients = set(patients[split_idx:])
    
    train_ct = [str(f) for f in ct_files if f.name.split('_')[0] in train_patients]
    test_ct = [str(f) for f in ct_files if f.name.split('_')[0] in test_patients]
    
    train_mri = [s.replace("_ct.", "_mr.") for s in train_ct]
    test_mri = [s.replace("_ct.", "_mr.") for s in test_ct]
    
    return (train_ct, train_mri), (test_ct, test_mri)
