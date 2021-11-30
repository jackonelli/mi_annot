from torch.utils.data import Dataset
import torch


class NoisyDataset(Dataset):
    def __init__(self, noise_level: float, dataset: Dataset, num_classes: int):
        self.noise_level = noise_level
        self._dataset = dataset
        self._baseline = self.noise_level / num_classes * torch.ones((num_classes))

    def __getitem__(self, idx):
        noisy_one_hot = torch.zeros(self._baseline.size())
        x, label = self._dataset.__getitem__(idx)
        noisy_one_hot[label] = 1 - self.noise_level
        return x, self._baseline + noisy_one_hot

    def __len__(self):
        return self._dataset.__len__()
