"""
Dataset for CycleGAN CT-MRI Translation with patient-wise splitting.
"""
import os
import random
from pathlib import Path
from typing import Optional, Tuple, List
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .config import DataConfig, TrainingConfig


class CTMRIDataset(Dataset):
    """Dataset for CT-MRI image translation."""
    
    def __init__(
        self,
        config: DataConfig,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
        ct_images: Optional[List[str]] = None,
        mri_images: Optional[List[str]] = None,
        unpaired: Optional[bool] = None,
        ct_dir: Optional[Path | str] = None, # kept for compatibility but largely ignored if lists refer to absolute paths
        mri_dir: Optional[Path | str] = None,
    ):
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # Get image lists safely
        self.ct_images = list(ct_images) if ct_images is not None else []
        self.mri_images = list(mri_images) if mri_images is not None else []
        
        if len(self.ct_images) == 0:
            raise ValueError(f"No CT images provided/found")
        if len(self.mri_images) == 0:
            raise ValueError(f"No MRI images provided/found")
        
        # For 2.5D and 3D modes
        self.use_25d = config.use_25d
        self.use_3d = config.use_3d
        self.num_adjacent_slices = config.num_adjacent_slices

        # Unpaired translation: sample MRI independently during training
        self.unpaired = (mode == "train") if unpaired is None else unpaired
        
        self.mri_indices = list(range(len(self.mri_images)))
        print(f"[{mode}] Loaded {len(self.ct_images)} CT images, {len(self.mri_images)} MRI images")
    
    @staticmethod
    def _get_image_list(directory: Path, suffix: str = "") -> List[str]:
        """Safely get list of absolute image file paths"""
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            return []
        
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.nii', '.nii.gz', '.pt'}
        images = []
        
        for f in directory.iterdir():
            if not f.is_file():
                continue
            if suffix and not f.name.lower().endswith(f"{suffix}{f.suffix.lower()}"):
                continue

            if f.suffix.lower() in extensions:
                images.append(str(f.resolve()))
            elif str(f).endswith('.nii.gz'):
                images.append(str(f.resolve()))
        
        return sorted(images)
    
    def __len__(self) -> int:
        return max(len(self.ct_images), len(self.mri_images))
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        ct_idx = index % len(self.ct_images)
        ct_path = Path(self.ct_images[ct_idx])
        
        if self.unpaired:
            mri_idx = random.randrange(len(self.mri_images))
        else:
            mri_idx = self.mri_indices[index % len(self.mri_images)]
        mri_path = Path(self.mri_images[mri_idx])
        
        if self.use_3d:
            ct = self._load_volume(ct_path)
            mri = self._load_volume(mri_path)
            mask = self._load_volume(ct_path.parent / ct_path.name.replace("_ct", "_mask"))
        elif self.use_25d:
            ct = self._load_25d(ct_path, ct_idx)
            mri = self._load_25d(mri_path, mri_idx)
            mask = self._load_2d(ct_path.parent / ct_path.name.replace("_ct", "_mask"))
        else:
            ct = self._load_2d(ct_path)
            mri = self._load_2d(mri_path)
            mask = self._load_2d(ct_path.parent / ct_path.name.replace("_ct", "_mask"))
        
        if self.transform:
            ct = self.transform(ct)
            mri = self.transform(mri)
            if mask is not None:
                # Use same transform for mask (should be resized/padded identically)
                mask = self.transform(mask)

        # Data Normalization: enforce strictly [-1, 1] for both modalities
        def strict_norm(x):
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min + 1e-8)
                x = x * 2 - 1
            else:
                x = x * 0 - 1
            return x

        ct = strict_norm(ct)
        mri = strict_norm(mri)
        # Mask: strictly binary [0, 1]
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0.5).float()
            else:
                mask = (np.array(mask) > 127).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)

        meta = {
            'ct_path': str(ct_path),
            'mri_path': str(mri_path),
            'ct_idx': ct_idx,
            'mri_idx': mri_idx,
        }
        
        return ct, mri, mask, meta
    
    def _load_2d(self, path: Path) -> Image.Image:
        if path.suffix.lower() == '.pt':
            data = torch.load(path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if data.ndim == 4:
                data = data[0]
            if data.ndim == 3:
                mid_idx = data.shape[0] // 2
                data = data[mid_idx]
            return Image.fromarray(data.astype(np.float32), mode='F')
        return Image.open(path).convert("L")
    
    def _load_25d(self, path: Path, idx: int) -> Image.Image:
        return Image.open(path).convert("L")
    
    def _load_volume(self, path: Path) -> torch.Tensor:
        try:
            import nibabel as nib
            nii = nib.load(str(path))
            volume = nii.get_fdata()
            volume = torch.from_numpy(volume).float()
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            volume = volume * 2 - 1
            return volume.unsqueeze(0)
        except ImportError:
            raise ImportError("nibabel required")
    
    def shuffle_mri_indices(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.mri_indices)
    
    def set_mri_indices(self, indices: List[int]):
        self.mri_indices = indices


def get_transforms(config: DataConfig, mode: str = "train") -> transforms.Compose:
    def _pad_only(img: Image.Image) -> Image.Image:
        size = int(config.image_size)
        if size <= 0:
            return img
        w, h = img.size
        pad_w = max(0, size - w)
        pad_h = max(0, size - h)
        if pad_w or pad_h:
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = TF.pad(img, padding, fill=-1)
        return img

    transform_list = [
        transforms.Resize((int(config.image_size), int(config.image_size)), antialias=True),
        transforms.Lambda(_pad_only),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def _split_list(items: List[str], split: float, seed: int) -> Tuple[List[str], List[str]]:
    """Patient-wise exact train/test split."""
    items = list(items)
    if len(items) < 2:
        return items, []

    def get_patient(p: str) -> str:
        return Path(p).name.split('_')[0]

    patients = sorted(list(set(get_patient(item) for item in items)))
    
    rng = random.Random(seed)
    rng.shuffle(patients)
    split_idx = int(len(patients) * split)
    split_idx = max(1, min(len(patients) - 1, split_idx))
    
    train_patients = set(patients[:split_idx])
    test_patients = set(patients[split_idx:])
    
    train_items = [item for item in items if get_patient(item) in train_patients]
    test_items = [item for item in items if get_patient(item) in test_patients]
    
    return train_items, test_items


def create_dataloaders(
    config: DataConfig,
    training_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    
    ct_images = []
    mri_images = []
    
    # Allows multiple regions for cyclegan config (e.g. "brain,pelvis")
    for region in config.ct_train_dir.split(','):
        region_dir = Path(config.dataset_dir) / region.strip()
        if not region_dir.exists():
            raise ValueError(f"Dataset directory not found: {region_dir}")
        ct_images.extend(CTMRIDataset._get_image_list(region_dir, suffix="_ct"))
        mri_images.extend(CTMRIDataset._get_image_list(region_dir, suffix="_mr"))

    if len(ct_images) == 0 or len(mri_images) == 0:
        raise ValueError(f"No CT or MRI images found. CT: {len(ct_images)}, MRI: {len(mri_images)}")

    seed = int(getattr(training_config, "seed", getattr(config, "seed", 42)))
    train_ct, test_ct = _split_list(ct_images, split=0.8, seed=seed)
    train_mri, test_mri = _split_list(mri_images, split=0.8, seed=seed + 1)

    train_dataset = CTMRIDataset(
        config=config,
        mode="train",
        transform=get_transforms(config, "train"),
        ct_images=train_ct,
        mri_images=train_mri,
        unpaired=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None

    test_dataset = CTMRIDataset(
        config=config,
        mode="train", # train mode implies unpaired? Wait, keeping original parameter
        transform=get_transforms(config, "val"),
        ct_images=test_ct,
        mri_images=test_mri,
        unpaired=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(training_config, "test_batch_size", 1),
        shuffle=False,
        num_workers=1, # config.num_workers causing freezing during testing usually
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
