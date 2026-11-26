import glob
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class HairStructureDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False):
        self.paths = sorted(glob.glob(os.path.join(root_dir, '*.npz')))
        if not self.paths:
            raise FileNotFoundError(f'No .npz files found in {root_dir}')
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # horizontal flip
        if np.random.rand() < 0.5:
            for k in sample:
                sample[k] = np.flip(sample[k], axis=1).copy()

            # orientation_gt: H,W,2
            # horizontal flip => x component changes sign
            sample['orientation_gt'][..., 0] *= -1

        # brightness jitter only on image
        if np.random.rand() < 0.2:
            gain = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-0.03, 0.03)
            sample['image'] = np.clip(sample['image'] * gain + bias, 0.0, 1.0)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]

        with np.load(path) as data:
            sample = {
                'image': np.clip(data['image'].astype(np.float32), 0.0, 1.0),            # H,W,3
                'mask': data['mask'].astype(np.float32),                                  # H,W
                'line_gt': data['line_gt'].astype(np.float32),                            # H,W
                'centerline_gt': data['centerline_gt'].astype(np.float32),                # H,W
                'orientation_gt': data['orientation_gt'].astype(np.float32),              # H,W,2
            }

            # optional field
            if 'skeleton_gt' in data:
                sample['skeleton_gt'] = data['skeleton_gt'].astype(np.float32)

        if self.augment:
            sample = self._augment(sample)

        image = torch.from_numpy(sample['image']).permute(2, 0, 1)              # 3,H,W
        mask = torch.from_numpy(sample['mask'])[None, ...]                       # 1,H,W
        line_gt = torch.from_numpy(sample['line_gt'])[None, ...]                 # 1,H,W
        center_gt = torch.from_numpy(sample['centerline_gt'])[None, ...]         # 1,H,W
        orient_gt = torch.from_numpy(sample['orientation_gt']).permute(2, 0, 1)  # 2,H,W

        x = torch.cat([image, mask], dim=0)                                      # 4,H,W

        out = {
            'input': x,
            'image': image,
            'mask': mask,
            'line_gt': line_gt,
            'centerline_gt': center_gt,
            'orientation_gt': orient_gt,
            'path': path,
        }

        if 'skeleton_gt' in sample:
            out['skeleton_gt'] = torch.from_numpy(sample['skeleton_gt'])[None, ...]

        return out