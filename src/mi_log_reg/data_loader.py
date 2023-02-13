from typing import Tuple, List
import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal


class MvnMultiClassData(Dataset):
    def __init__(self, num_samples: int, clusters: List[Tuple[torch.Tensor, torch.Tensor]], seed: int):
        self._num_samples = num_samples
        self._clusters = clusters
        self.samples = self._gen_samples()

    def __getitem__(self, idx):
        inputs, labels = self.samples
        return (inputs[idx], labels[idx])

    def __len__(self):
        return self._num_samples

    def _gen_samples(self):
        cluster_size, remainder = divmod(self._num_samples, self.num_classes())

        remaining_distr = MultivariateNormal(loc=self._clusters[0][0], covariance_matrix=self._clusters[0][1])
        input_ = remaining_distr.sample((remainder,))
        labels = torch.zeros((remainder, 1), dtype=torch.long)
        for class_idx, (mean, cov) in enumerate(self._clusters):
            sampling_distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
            x = sampling_distr.sample((cluster_size,))
            input_ = torch.row_stack((input_, x))
            labels = torch.row_stack((labels, class_idx * torch.ones((cluster_size, 1), dtype=torch.long)))
        return input_, labels.reshape((self._num_samples,))

    def num_classes(self):
        return len(self._clusters)

    def feature_dim(self):
        return self._clusters[0][0].size(0)
