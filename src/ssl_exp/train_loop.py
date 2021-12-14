import argparse
from pathlib import Path
import sys
from time import time
import numpy as np
import torch
import torch.nn as torch_nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd()))
from src.ssl_exp.data_loader import NoisyLabels, DinoFeatures
from src.ssl_exp.model import LabelNoiseCorrectedClassifier, LinearClassifier
from src.ssl_exp.label_noise_correction import SymmetricNoise
from src.utils.torch_helpers import device
from src.utils.metrics import Metrics, Accuracy, TopXAccuracy, Nll
from src.utils.exp import ExperimentConfig

EXPERIMENT_NAME = "dino_ssl_linear_imagenet_1k"
DEVICE = device()


def main():
    """Main entry point."""
    args = _parse_args()
    cudnn.benchmark = True

    alphas = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    ratios = np.array([0.01, 0.1, 0.2, 0.5, 0.75])
    for alpha_idx, annot_quality in enumerate(alphas):
        for ratio_idx, ratio in enumerate(ratios):
            gen_metrics(annot_quality, ratio, args)


def gen_metrics(annot_quality, ratio, args):
    clean_train_set = DinoFeatures(args.data_path / "train", subset_ratio=ratio)
    num_classes = clean_train_set.num_classes()
    clean_val_set = DinoFeatures(args.data_path / "val", subset_ratio=args.val_set_ratio)
    print(
        f"Using {len(clean_train_set)} training samples with quality {annot_quality}, {len(clean_val_set)} val. samples with quality 1.0"
    )
    train_set = NoisyLabels(annot_quality, clean_train_set, num_classes)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_set = NoisyLabels(annot_quality, clean_val_set, num_classes)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    lin_class = LinearClassifier(clean_train_set.feature_dim(), num_classes)
    label_noise_corr = SymmetricNoise(annot_quality, num_classes)
    model = LabelNoiseCorrectedClassifier(lin_class, label_noise_corr)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001 * args.batch_size / 256,
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    loss_fn = torch_nn.NLLLoss()

    train_metrics = Metrics([Accuracy("acc"), TopXAccuracy("top-5-acc", 5), Nll("nll")])
    val_metrics = Metrics([Accuracy("acc"), TopXAccuracy("top-5-acc", 5), Nll("nll")])

    model, train_losses, train_metrics, val_metrics = train(
        model, train_loader, val_loader, optimizer, loss_fn, train_metrics, val_metrics, args.epochs
    )
    sub_exp = f"annot_quality_{annot_quality}_ratio_{ratio}"
    save_exp(args.output_dir / EXPERIMENT_NAME / sub_exp, model, train_losses, train_metrics, val_metrics, args)


def train(model, train_loader, val_loader, optimizer, loss_fn, train_metrics, val_metrics, num_epochs):
    print("Starting training")
    train_losses = []
    for epoch in np.arange(1, num_epochs + 1):
        epoch_start = time()
        batch_losses, train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, train_metrics)
        train_losses.append(np.array(batch_losses))
        val_metrics = validate_epoch(model, val_loader, val_metrics)
        epoch_end = time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch}/{num_epochs}, time: {epoch_time:.1f}s. Val: {val_metrics.summary()}")
    return model, train_losses, train_metrics, val_metrics


def train_epoch(model, train_loader, optimizer, loss_fn, metrics):
    model.train()
    batch_losses = []
    for batch_idx, (inp, (noisy_label, true_label)) in enumerate(train_loader, 1):
        inp, noisy_label, true_label = inp.to(DEVICE), noisy_label.to(DEVICE), true_label.to(DEVICE)
        optimizer.zero_grad()
        p_true = model.p_y_given_x(inp)
        p_tilde = model.p_y_tilde_given_x(p_true)
        loss = loss_fn(torch.log(p_tilde), noisy_label)
        loss.backward()
        batch_losses.append(loss.item())
        metrics.add_sample(true_label, p_true, inp)
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}: Loss {round(loss.item(), 2)}, {metrics.last()}")
    metrics.checkpoint()

    return np.array(batch_losses), metrics


@torch.no_grad()
def validate_epoch(model, val_loader, metrics):
    model.eval()
    for batch_idx, (inp, (_, true_label)) in enumerate(val_loader):
        inp, true_label = inp.to(DEVICE), true_label.to(DEVICE)
        p_true = model.p_y_given_x(inp)
        metrics.add_sample(true_label, p_true, inp)
    metrics.checkpoint()
    return metrics


def save_exp(exp_dir, model, train_losses, train_metrics, val_metrics, args):
    exp_dir.mkdir(exist_ok=True, parents=True)
    ExperimentConfig(EXPERIMENT_NAME, args).save(exp_dir)
    torch.save(model.state_dict(), exp_dir / "model")
    train_metrics.save(exp_dir, "train")
    train_checkpoints = [timestep for (timestep, _, _) in train_metrics["nll"].list_checkpoints()]
    val_metrics.save_checkpoints(exp_dir, "val", train_checkpoints)


def _parse_args():
    parser = argparse.ArgumentParser("Training with noisy labels on DINO Imagenet features")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument(
        "--val_set_ratio", default=1.0, type=float, help="The ratio of full Imagenet to use for validation"
    )
    parser.add_argument("--output_dir", default="stored", type=Path, help="Path to save logs and checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    main()
