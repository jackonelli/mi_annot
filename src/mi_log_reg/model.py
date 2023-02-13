import sys
from pathlib import Path
import torch
import torch.nn as torch_nn

sys.path.append(str(Path.cwd()))
from src.label_noise import LabelNoiseCorrector


class LinearClassifier(torch_nn.Module):
    """Linear layer"""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.num_labels = num_classes
        self.linear = torch_nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self._device = torch.device("cpu")

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class Ensemble(torch_nn.Module):
    def __init__(
        self,
        constructor: LinearClassifier,
        noise_corr: LabelNoiseCorrector,
        feature_dim: int,
        num_classes: int,
        ens_size: int,
    ):
        self._num_classes = num_classes
        self._members = []
        for _ in range(ens_size):
            lin_class = LinearClassifier(feature_dim, num_classes)
            model = LabelNoiseCorrectedClassifier(lin_class, noise_corr)
            self._members.append(model)

    def to(self, device):
        for member in self._members:
            member.to(device)
        self._device = device

    def mi_approx(self, data_loader):
        mi = []
        for (input_, _label) in data_loader:
            input_ = input_.to(self._device)
            prob_vecs = self.ind_p_y_given_x(input_)
            mi.append(mi_approx(prob_vecs))
        mi = torch.Tensor(mi)
        return mi.mean()

    def noisy_mi_approx(self, data_loader):
        mi = []
        for (input_, _label) in data_loader:
            input_ = input_.to(self._device)
            prob_vecs = self.ind_p_y_tilde_given_x(input_)
            mi.append(mi_approx(prob_vecs))
        mi = torch.Tensor(mi)
        return mi.mean()

    def ind_p_y_tilde_given_x(self, input_):
        prob_vecs = torch.empty(
            (
                len(input_),
                self._num_classes,
                len(self._members),
            )
        )
        for mem_idx, mem in enumerate(self._members):
            p_vec = mem.p_y_given_x(input_)
            noisy_p_vec = mem.p_y_tilde_given_x(p_vec)
            prob_vecs[:, :, mem_idx] = noisy_p_vec
        return prob_vecs

    def ind_p_y_given_x(self, input_):
        prob_vecs = torch.empty(
            (
                len(input_),
                self._num_classes,
                len(self._members),
            )
        )
        for mem_idx, mem in enumerate(self._members):
            p_vec = mem.p_y_given_x(input_)
            prob_vecs[:, :, mem_idx] = p_vec
        return prob_vecs


def mi_approx(batch_prob_vecs):
    batch_mi = []
    for prob_vecs in batch_prob_vecs:
        avg_prob_vec = prob_vecs.mean(0)
        entropy_of_avg = entropy(avg_prob_vec)
        entropies = entropy(prob_vecs)
        avg_entropy = entropies.mean()
        batch_mi.append(entropy_of_avg - avg_entropy)
    batch_mi = torch.tensor(batch_mi)
    return batch_mi.mean()


def entropy(prob_vecs: torch.Tensor):
    log_prob_vecs = torch.log(prob_vecs)
    return -(prob_vecs * log_prob_vecs).sum(0)


class LabelNoiseCorrectedClassifier(torch_nn.Module):
    def __init__(self, model: torch_nn.Module, label_noise_corrector: LabelNoiseCorrector):
        super().__init__()
        self._model = model
        self._label_noise_corr = label_noise_corrector
        self._sm = torch_nn.Softmax(dim=1)

    def p_y_given_x(self, x):
        """Network output without

        Computes: (p(y | x),
        """
        return self._sm(self._model.forward(x))

    def p_y_tilde_given_x(self, p_y_given_x: torch.Tensor):
        """Network output with label noise correction

        Computes: p_tilde(y_tilde | x),
        where p_tilde(y_tilde | x) = \\sum_y p(y_tilde | y) p(y | x)
        and p(y | x) is the distribution of the uncorrupted labels.
        """
        return self._label_noise_corr.label_noise_correction(p_y_given_x, None, None)

    def to(self, device):
        self._model.to(device)
        self._label_noise_corr.to(device)
