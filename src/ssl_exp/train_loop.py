import argparse
from pathlib import Path
import torch
import torch.nn as torch_nn
import torch.backends.cudnn as cudnn
import sys

sys.path.append(str(Path.cwd()))
from src.ssl_exp.data_loader import NoisyLabels, DinoFeatures


def main():
    """Main entry point."""
    args = _parse_args()
    clean_train_set = DinoFeatures(args.data_path / "train")
    train_set = NoisyLabels(0.0, clean_train_set, clean_train_set.num_classes())
    print(train_set[0])

    # cudnn.benchmark = True
    # lin_class = LinearClassifier(clean_train_set.feature_dim(), clean_train_set.num_classes())
    # lin_class.cuda()  # TODO: proper device


def _parse_args():
    parser = argparse.ArgumentParser("Training with noisy labels on DINO Imagenet features")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--data_path", default=Path("/path/to/imagenet/"), type=Path)
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument("--val_freq", default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument("--output_dir", default=".", help="Path to save logs and checkpoints")
    return parser.parse_args()


class LinearClassifier(torch_nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_classes):
        super(LinearClassifier, self).__init__()
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
    main()
