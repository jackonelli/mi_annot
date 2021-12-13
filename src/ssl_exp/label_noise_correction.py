"""Label noise correction"""
from abc import ABC, abstractmethod
import torch


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
