from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch
from src.utils.label_noise import symmetric_label_noise

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


class NoisyLabels(Dataset):
    def __init__(self, annot_quality: float, dataset: Dataset, num_classes: int):
        self.annot_quality = annot_quality
        self.num_classes = num_classes
        self._dataset = dataset
        self._label_distr = partial(symmetric_label_noise, annot_quality=annot_quality, num_classes=num_classes)

    def __getitem__(self, idx):
        x, true_label = self._dataset.__getitem__(idx)
        noisy_label = np.random.choice(self.num_classes, p=self._label_distr(true_label))
        return x, (noisy_label, true_label)

    def __len__(self):
        return self._dataset.__len__()
