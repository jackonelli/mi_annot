from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch

NUM_CLASSES = 1000
FEATURE_DIM = 1536


class DinoFeatures(Dataset):
    def __init__(self, dir_: Path, subset_ratio: float):
        self.dir = dir_
        self.num_samples = len(list(self.dir.glob("*.npy")))

    def __getitem__(self, idx):
        sample_path = self.dir / (str(idx) + ".npy")
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
        return NUM_CLASSES


class NoisyLabels(Dataset):
    def __init__(self, annot_quality: float, dataset: Dataset, num_classes: int):
        self.annot_quality = annot_quality
        self.num_classes = num_classes
        self._dataset = dataset
        self._label_distr = partial(symmetric_label_noise, annot_quality=annot_quality, num_classes=num_classes)

    def __getitem__(self, idx):
        x, label = self._dataset.__getitem__(idx)
        noisy_label = np.random.choice(self.num_classes, p=self._label_distr(label))
        return x, noisy_label

    def __len__(self):
        return self._dataset.__len__()


def symmetric_label_noise(label: int, annot_quality: float, num_classes: int):
    baseline = (1.0 - annot_quality) / num_classes * torch.ones((num_classes), dtype=torch.float64)
    true_label = torch.zeros(baseline.size(), dtype=torch.float64)
    true_label[label] = annot_quality
    tmp = baseline + true_label
    # # Numerical hack to make the probability vec sum to 1.
    # tmp[-1] = 1.0 - tmp[:-1].sum()
    return tmp / tmp.sum()
