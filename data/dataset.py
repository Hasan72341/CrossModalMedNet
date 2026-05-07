import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class SynthRADDataset(Dataset):
    """
    Portable Multi-Modal Dataset loader for SynthRAD 2023.
    
    Supports:
    - Relative paths (portable across machines)
    - Split filtering (train/val)
    - Region filtering (brain/pelvis)
    - Unpaired/Paired modes
    - Dimensionality: 2D, 2.5D, 3D Patches
    """
    
    def __init__(
        self, 
        manifest_path, 
        split='train', 
        region=None, 
        unpaired=False, 
        mode='3d',
        patch_size=None,
        slice_axis='axial',
        num_slices=3,
        transform=None
    ):
        """
        Args:
            manifest_path (str): Path to manifest.csv. The directory of this file
                                  is used as the root for all relative paths.
        """
        self.root = os.path.dirname(os.path.abspath(manifest_path))
        df = pd.read_csv(manifest_path)
        
        # Filter
        df = df[df['split'] == split]
        if region:
            df = df[df['region'] == region]
            
        self.df = df
        self.unpaired = unpaired
        self.mode = mode
        self.patch_size = patch_size
        self.slice_axis = slice_axis.lower()
        self.num_slices = num_slices
        self.transform = transform
        
        self.axis_dim = {'axial': 1, 'coronal': 2, 'sagittal': 3}.get(self.slice_axis, 1)

        # Separate
        mr_df = df[df['modality'] == 'mr']
        ct_df = df[df['modality'] == 'ct']
        
        if unpaired:
            self.mri_paths = mr_df['pt_path'].tolist()
            self.ct_paths = ct_df['pt_path'].tolist()
            self.mri_masks = mr_df['mask_path'].tolist()
            self.ct_masks = ct_df['mask_path'].tolist()
            self.mri_ids = mr_df['patient_id'].tolist()
            self.ct_ids = ct_df['patient_id'].tolist()
            self.mri_regions = mr_df['region'].tolist()
            self.ct_regions = ct_df['region'].tolist()
            self.length = len(self.mri_paths)
        else:
            paired_df = pd.merge(mr_df, ct_df, on=['patient_id', 'region'], suffixes=('_mr', '_ct'))
            self.mri_paths = paired_df['pt_path_mr'].tolist()
            self.ct_paths = paired_df['pt_path_ct'].tolist()
            self.mask_paths = paired_df['mask_path_mr'].tolist()
            self.ids = paired_df['patient_id'].tolist()
            self.regions = paired_df['region'].tolist()
            self.length = len(paired_df)

    def __len__(self):
        return self.length

    def _load_pt(self, rel_path):
        if not rel_path or pd.isna(rel_path):
            return None
        return torch.load(os.path.join(self.root, rel_path), weights_only=True)

    def _apply_augmentation(self, mri, ct, mask):
        dims = list(range(1, mri.dim()))
        for axis in dims:
            if random.random() > 0.5:
                mri = torch.flip(mri, [axis])
                if not self.unpaired:
                    ct = torch.flip(ct, [axis])
                    mask = torch.flip(mask, [axis])
        return mri, ct, mask

    def _get_slice_indices(self, max_dim):
        if self.mode == '2d':
            return [random.randint(0, max_dim - 1)]
        elif self.mode == '2.5d':
            half = self.num_slices // 2
            idx = random.randint(half, max_dim - half - 1)
            return list(range(idx - half, idx + half + 1))
        return []

    def _extract_slices(self, vol, indices):
        if self.slice_axis == 'axial':
            slices = vol[:, indices, :, :] 
        elif self.slice_axis == 'coronal':
            slices = vol[:, :, indices, :].permute(2, 0, 1, 3) 
        else: # sagittal
            slices = vol[:, :, :, indices].permute(3, 0, 1, 2)
        if slices.shape[1] == 1:
            slices = slices.squeeze(1)
        return slices

    def __getitem__(self, index):
        mri = self._load_pt(self.mri_paths[index])
        if self.unpaired:
            mask = self._load_pt(self.mri_masks[index])
            res_id = self.mri_ids[index]
            res_region = self.mri_regions[index]
            ct_idx = random.randint(0, len(self.ct_paths)-1)
            ct = self._load_pt(self.ct_paths[ct_idx])
        else:
            mask = self._load_pt(self.mask_paths[index])
            res_id = self.ids[index]
            res_region = self.regions[index]
            ct = self._load_pt(self.ct_paths[index])
            
        if self.mode == '3d' and self.patch_size:
            _, d, h, w = mri.shape
            pd, ph, pw = self.patch_size
            d_s, h_s, w_s = random.randint(0, max(0, d-pd)), random.randint(0, max(0, h-ph)), random.randint(0, max(0, w-pw))
            mri = mri[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]
            mask = mask[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]
            if self.unpaired:
                _, d, h, w = ct.shape
                d_s, h_s, w_s = random.randint(0, max(0, d-pd)), random.randint(0, max(0, h-ph)), random.randint(0, max(0, w-pw))
            ct = ct[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]
            # Padding
            if mri.shape[1:] != self.patch_size:
                mri = torch.nn.functional.pad(mri, (0, pw-mri.shape[3], 0, ph-mri.shape[2], 0, pd-mri.shape[1]))
                mask = torch.nn.functional.pad(mask, (0, pw-mask.shape[3], 0, ph-mask.shape[2], 0, pd-mask.shape[1]))
            if ct.shape[1:] != self.patch_size:
                ct = torch.nn.functional.pad(ct, (0, pw-ct.shape[3], 0, ph-ct.shape[2], 0, pd-ct.shape[1]))
        elif self.mode in ['2d', '2.5d']:
            indices_A = self._get_slice_indices(mri.shape[self.axis_dim])
            mri = self._extract_slices(mri, indices_A)
            mask = self._extract_slices(mask, indices_A)
            indices_B = self._get_slice_indices(ct.shape[self.axis_dim]) if self.unpaired else indices_A
            ct = self._extract_slices(ct, indices_B)

        if self.transform:
            mri, ct, mask = self._apply_augmentation(mri, ct, mask)
        return {'mri': mri, 'ct': ct, 'mask': mask, 'id': res_id, 'region': res_region}
