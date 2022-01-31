"""Label noise correction"""
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset


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


def symmetric_label_noise(label: int, annot_quality: float, num_classes: int):
    baseline = (1.0 - annot_quality) / num_classes * torch.ones((num_classes), dtype=torch.float64)
    true_label = torch.zeros(baseline.size(), dtype=torch.float64)
    true_label[label] = annot_quality
    tmp = baseline + true_label
    # # Numerical hack to make the probability vec sum to 1.
    return tmp / tmp.sum()


class LabelNoiseCorrector(ABC):
    @abstractmethod
    def label_noise_correction(self, prob_vec: torch.Tensor, label: int, input_: torch.Tensor):
        pass


class SymmetricNoise(LabelNoiseCorrector):
    def __init__(self, annot_quality, num_classes):
        self._annot_quality = annot_quality
        self._num_classes = num_classes
        self._baseline = (1 - self._annot_quality) / self._num_classes * torch.ones((self._num_classes,))

    def to(self, device):
        self._baseline = self._baseline.to(device)

    def label_noise_correction(self, prob_vec: torch.Tensor, _noisy_label: int, _input: torch.Tensor):
        return self._annot_quality * prob_vec + self._baseline
