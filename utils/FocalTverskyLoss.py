import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true = y_true.to(y_pred.device)  # 确保y_true和y_pred在同一设备

        # Flatten label and prediction tensors
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (y_true_flat * y_pred_flat).sum()
        FP = ((1 - y_true_flat) * y_pred_flat).sum()
        FN = (y_true_flat * (1 - y_pred_flat)).sum()

        # Tversky Index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal Tversky Loss
        focal_tversky_loss = (1 - tversky_index) ** self.gamma

        return focal_tversky_loss


def gradient_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = (y_true_f * y_pred_f).sum()
        return 1 - (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)
