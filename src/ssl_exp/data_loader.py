from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch

NUM_CLASSES = 1000
FEATURE_DIM = 1536


class DinoFeatures(Dataset):
    def __init__(self, dir_: Path, subset_ratio: float):
        self.dir = dir_
        all_samples = list(self.dir.glob("*.npy"))
        num_total_samples = len(all_samples)
        self.num_samples = round(num_total_samples * subset_ratio)
        self.samples = np.random.choice(num_total_samples, self.num_samples)

    def __getitem__(self, idx):
        global_idx = self.samples[idx]
        sample_path = self.dir / (str(global_idx) + ".npy")
        sample = np.load(sample_path)
        label, feature = int(sample[0]), sample[1:]
        return (torch.tensor(feature), label)

    def __len__(self):
        return self.num_samples

    @staticmethod
    def num_classes():
        return NUM_CLASSES

    @staticmethod
    def feature_dim():
        return FEATURE_DIM
