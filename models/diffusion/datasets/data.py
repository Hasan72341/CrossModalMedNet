from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, List, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from configs.config import Settings
from utils.utils import ensure_3ch, normalize_tensor


def medical_collate_fn(batch: List[dict]) -> dict:
    """Custom collate that handles None masks and string meta dicts."""
    elem = batch[0]
    result = {}
    for key in elem:
        values = [b[key] for b in batch]
        if key == "mask":
            # If all None → keep as None, else replace None with zeros
            if all(v is None for v in values):
                result[key] = None
            else:
                ref = next(v for v in values if v is not None)
                values = [v if v is not None else torch.zeros_like(ref) for v in values]
                result[key] = default_collate(values)
        elif key == "meta":
            # Collate list of dicts of strings into dict of lists
            result[key] = {k: [d[k] for d in values] for k in values[0]}
        else:
            result[key] = default_collate(values)
    return result

class CTMRIDataset(Dataset):
    """Dataset matched with cyclegan_brain/data.py"""
    
    def __init__(
        self,
        settings: Settings,
        mode: str = "train",
        ct_images: Optional[List[str]] = None,
        mri_images: Optional[List[str]] = None,
        unpaired: bool = True,
    ):
        self.settings = settings
        self.mode = mode
        
        self.ct_images = list(ct_images) if ct_images is not None else []
        self.mri_images = list(mri_images) if mri_images is not None else []
        
        if len(self.ct_images) == 0:
            raise ValueError("No CT images")
        if len(self.mri_images) == 0:
            raise ValueError("No MRI images")
            
        self.use_25d = (settings.data_mode == "2.5d")
        self.unpaired = unpaired
        self.mri_indices = list(range(len(self.mri_images)))
        
        # Transform logic from cyclegan_brain
        def _pad_only(img: Image.Image) -> Image.Image:
            size = int(settings.image_size)
            w, h = img.size
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            if pad_w or pad_h:
                padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
                img = TF.pad(img, padding, fill=-1)
            return img

        self.transform = transforms.Compose([
            transforms.Resize((int(settings.image_size), int(settings.image_size)), antialias=True),
            transforms.Lambda(_pad_only),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _get_image_list(directory: Path, suffix: str = "") -> List[str]:
        if not directory.exists():
            return []
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.nii', '.nii.gz', '.pt'}
        images = []
        for f in directory.iterdir():
            if not f.is_file():
                continue
            if suffix and not f.name.lower().endswith(f"{suffix}{f.suffix.lower()}"):
                continue
            if f.suffix.lower() in extensions or str(f).endswith('.nii.gz'):
                images.append(str(f.resolve()))
        return sorted(images)

    def __len__(self) -> int:
        if self.unpaired:
            return max(len(self.ct_images), len(self.mri_images))
        return min(len(self.ct_images), len(self.mri_images))

    def _load_2d(self, path: Path) -> Image.Image:
        if path.suffix.lower() == '.pt':
            data = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if data.ndim == 4:
                data = data[0]
            if data.ndim == 3:
                mid_idx = data.shape[0] // 2
                data = data[mid_idx]
            return Image.fromarray(data.astype(np.float32), mode='F')
        return Image.open(path).convert("L")

    def _load_25d(self, path: Path) -> Image.Image:
        if path.suffix.lower() == '.pt':
            return self._load_2d(path)
        return Image.open(path).convert("L")

    def __getitem__(self, index: int) -> dict[str, Any]:
        ct_idx = index % len(self.ct_images)
        ct_path = Path(self.ct_images[ct_idx])
        
        if self.unpaired:
            mri_idx = random.randrange(len(self.mri_images))
        else:
            mri_idx = self.mri_indices[index % len(self.mri_images)]
        mri_path = Path(self.mri_images[mri_idx])
        
        if self.use_25d:
            ct = self._load_25d(ct_path)
            mri = self._load_25d(mri_path)
        else:
            ct = self._load_2d(ct_path)
            mri = self._load_2d(mri_path)
            
        mask_path = ct_path.parent / ct_path.name.replace("_ct", "_mask")
        mask = None
        if mask_path.exists():
            mask = self._load_2d(mask_path)
            
        ct = self.transform(ct)
        mri = self.transform(mri)
        if mask is not None:
            mask = self.transform(mask)
            mask = (mask > 0.5).float()
            
        # strict norm [-1, 1]
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

        # Convert to 3ch for diffusion
        ct = ensure_3ch(ct)
        mri = ensure_3ch(mri)
        if mask is not None:
            mask = ensure_3ch(mask)

        if self.settings.source_modality == "mr":
            src, tgt = mri, ct
            src_path, tgt_path = mri_path, ct_path
        else:
            src, tgt = ct, mri
            src_path, tgt_path = ct_path, mri_path

        patient_id = ct_path.name.split('_')[0]

        return {
            "pixel_values_src": src,
            "pixel_values_tgt": tgt,
            "mask": mask,
            "meta": {
                "id": patient_id,
                "region": self.settings.region,
                "src_path": str(src_path),
                "tgt_path": str(tgt_path),
                "src_modality": self.settings.source_modality,
                "tgt_modality": self.settings.target_modality,
            },
        }

def _split_list(items: List[str], split: float, seed: int) -> Tuple[List[str], List[str]]:
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

def create_dataloader(
    settings: Settings,
    split: str | None = None,
    unpaired: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    augment: bool | None = None, # Added for paired compatibility
) -> DataLoader:
    # Allows multiple regions for cyclegan config (e.g. "brain,pelvis")
    ct_images = []
    mri_images = []
    
    region = settings.region or "brain"
    dataset_dir = "/usershome/cs671_user4/SynthRAD2023_Dataset" # Assume SLICED has the _ct.png files
    # Actually wait. The prompt said `ignore joint one`. In standard model it was reading from `config.dataset_dir` which is `~/SynthRAD2023_SLICED` in cyclegan.
    
    region_dir = Path(dataset_dir) / region
    ct_images.extend(CTMRIDataset._get_image_list(region_dir, suffix="_ct"))
    mri_images.extend(CTMRIDataset._get_image_list(region_dir, suffix="_mr"))
    
    seed = settings.seed
    train_ct, test_ct = _split_list(ct_images, split=0.8, seed=seed)
    train_mri, test_mri = _split_list(mri_images, split=0.8, seed=seed + 1)
    
    req_split = split or settings.split
    is_train = (req_split == "train")
    
    if is_train:
        ds_ct, ds_mri = train_ct, train_mri
    else:
        ds_ct, ds_mri = test_ct, test_mri
        shuffle = False
        drop_last = False

    dataset = CTMRIDataset(
        settings=settings,
        mode="train" if is_train else "val",
        ct_images=ds_ct,
        mri_images=ds_mri,
        unpaired=False,
    )

    return DataLoader(
        dataset,
        batch_size=settings.batch_size if is_train else 1,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        drop_last=drop_last,
        collate_fn=medical_collate_fn,
    )
