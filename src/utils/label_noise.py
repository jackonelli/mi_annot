import torch


def symmetric_label_noise(label: int, annot_quality: float, num_classes: int):
    baseline = (1.0 - annot_quality) / num_classes * torch.ones((num_classes), dtype=torch.float64)
    true_label = torch.zeros(baseline.size(), dtype=torch.float64)
    true_label[label] = annot_quality
    tmp = baseline + true_label
    # # Numerical hack to make the probability vec sum to 1.
    return tmp / tmp.sum()
