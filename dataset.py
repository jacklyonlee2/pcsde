import os
import numpy as np

import torch
import torch.utils.data

_synsetid_to_cate = {
    "02691156": "airplane",
    "02958343": "car",
    "03001627": "chair",
}
_cate_to_synsetid = {v: k for k, v in _synsetid_to_cate.items()}


class ShapeNet15k(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        cate,
        split,
        random_sample,
        sample_size,
        recenter=True,
    ):
        self.data = []
        cate_dir = os.path.join(root, _cate_to_synsetid[cate], split)
        for fname in os.listdir(cate_dir):
            if fname.endswith(".npy"):
                path = os.path.join(cate_dir, fname)
                sample = np.load(path)[np.newaxis, ...]
                self.data.append(sample)

        # Normalize data
        self.data = np.concatenate(self.data)
        B, N, C = self.data.shape
        if recenter:
            mx = np.amax(self.data, axis=1).reshape(B, 1, C)
            mn = np.amin(self.data, axis=1).reshape(B, 1, C)
            rn = np.amax(mx - mn, axis=-1).reshape(B, 1, 1)
            self.mu, self.std = (mx + mn) / 2, rn / 2
        else:
            self.mu = self.data.reshape(-1, C).mean(axis=0).reshape(1, 1, C)
            self.std = self.data.reshape(-1).std(axis=0).reshape(1, 1, 1)
        self.data = (self.data - self.mu) / self.std

        # Convert to Torch tensor and resize
        self.data = torch.from_numpy(self.data).float()
        self.mu = torch.from_numpy(self.mu).float().expand(B, 1, C)
        self.std = torch.from_numpy(self.std).float().expand(B, 1, C)

        # Following lines are purely for reproducing results of
        # the official SetVAE implementation: github.com/jw9730/setvae
        tr_data, te_data = self.data.split(10000, dim=1)
        self.data = tr_data if split == "train" else te_data

        self.random_sample = random_sample
        self.sample_size = sample_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, mu, std = self.data[idx], self.mu[idx], self.std[idx]
        return (
            x[
                torch.randperm(x.size(0))[: self.sample_size]
                if self.random_sample
                else torch.arange(self.sample_size)
            ],
            self.mu,
            self.std,
        )
