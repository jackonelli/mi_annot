"""Metrics"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as torch_nn


class Metric(ABC):
    def __init__(self, id_):
        super().__init__()
        self.id = id_
        self._vals = list()

    def full_sample(self):
        return np.array(self._vals)

    def mean_and_std(self) -> Tuple[float, float]:
        full_sample = self.full_sample()
        mean = full_sample.mean()
        std = full_sample.std()
        return mean, std / np.sqrt(len(full_sample))

    def add_sample(self, y_true: torch.Tensor, y_hat: torch.Tensor, x: Optional[torch.Tensor]):
        val = self.compute_metric(y_true, y_hat, x)
        self._vals.append(val)

    def save(self, dir_: Path, prefix: str):
        vals = self.full_sample()
        filename = dir_ / f"{prefix}_{self.id}.csv"
        np.savetxt(filename, vals)

    def last(self):
        return self._vals[-1]

    @abstractmethod
    def compute_metric(self, y_true: torch.Tensor, y_hat: torch.Tensor, x: Optional[torch.Tensor]) -> float:
        pass


class Metrics:
    def __init__(self, metrics: List[Metric]):
        keys = set([m.id for m in metrics])
        if len(keys) == len(metrics):
            self._metrics = metrics
        else:
            raise KeyError("Metric id's are not unique")

    def add_sample(self, y_true: torch.Tensor, y_hat: torch.Tensor, x: Optional[torch.Tensor]):
        for metric in self._metrics:
            metric.add_sample(y_true, y_hat, x)

    def save(self, dir_: Path, prefix: str):
        for m in self._metrics:
            m.save(dir_, prefix)

    def summary(self):
        str_ = str()
        for met in self._metrics:
            mean, _ = met.mean_and_std()
            str_ += f"{met.id}: {mean:.2f}, "
        return str_[:-2]

    def last(self):
        str_ = str()
        for met in self._metrics:
            str_ += f"{met.id}: {met.last():.2f}, "
        return str_[:-2]

    def ids(self):
        return [m.id for m in self._metrics]

    def __iter__(self):
        return self._metrics.__iter__()

    def __getitem__(self, id_) -> Optional[Metric]:
        idx = self.ids().index(id_)
        if idx is not None:
            return self._metrics[idx]
        else:
            return None


class Nll(Metric):
    def __init__(self, id_):
        super().__init__(id_)
        self._nll = torch_nn.NLLLoss()

    @torch.no_grad()
    def compute_metric(self, y_true: torch.Tensor, y_hat: torch.Tensor, _x: Optional[torch.Tensor]) -> float:
        return self._nll(torch.log(y_hat), y_true).item()


class TopXAccuracy(Metric):
    def __init__(self, id_, rank):
        super().__init__(id_)
        self._rank = rank

    def compute_metric(self, y_true: torch.Tensor, y_hat: torch.Tensor, _x: Optional[torch.Tensor]) -> float:
        sorted_ = torch.argsort(y_hat, descending=True, dim=1)
        top_ranked = sorted_[:, : self._rank]
        correct_count = 0
        for y, top in zip(y_true, top_ranked):
            correct_count += int(y in top)

        return correct_count / len(y_true)


class Accuracy(Metric):
    def __init__(self, id_):
        super().__init__(id_)

    def compute_metric(self, y_true: torch.Tensor, y_hat: torch.Tensor, _x: Optional[torch.Tensor]) -> float:
        hard_preds = output_to_label(y_hat)
        return accuracy(hard_preds, y_true)


def output_to_label(prob_vec):
    """Map network output prob_vec to a hard label {0, 1}

    Args:
        prob_vec (Tensor): Probabilities for each sample in a batch.
    """
    if prob_vec.dim() == 1:
        label = torch.zeros(prob_vec.shape, dtype=torch.long, device=prob_vec.device)
        label[prob_vec >= 0.5] = 1
        label[prob_vec < 0.5] = 0
        return label
    else:
        return torch.argmax(prob_vec, dim=1)


def accuracy(hard_preds, labels):
    return (hard_preds == labels).float().mean().item()
