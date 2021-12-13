import sys
from pathlib import Path
import torch
import torch.nn as torch_nn

sys.path.append(str(Path.cwd()))
from src.ssl_exp.label_noise_correction import LabelNoiseCorrector
from src.utils.torch_helpers import device as torch_device


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


class LinearClassifier(torch_nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.num_labels = num_classes
        self.linear = torch_nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == "__main__":
    from src.ssl_exp.label_noise_correction import SymmetricNoise

    feat_dim = 1536
    num_classes = 1000
    batch_size = 2
    model = LabelNoiseCorrectedClassifier(LinearClassifier(feat_dim, num_classes), SymmetricNoise(0.9, num_classes))
    model.to(torch_device())
    x = torch.randn((batch_size, feat_dim)).to(torch_device())
    p_hat = model(x)
    print(list(map(lambda x: x.size(), model.parameters())))
    assert p_hat.size() == (batch_size, num_classes)
