import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, gamma):
        super().__init__(weight=None)
        self.gamma = gamma

    def forward(self, input_tensor, target):
        cross_entropy = super().forward(input_tensor, target)
        input_probability = F.softmax(input_tensor, dim=1)
        # Check paper: "Focal Loss for Dense Object Detection" by Lin, TY et al.
        loss = torch.pow(1-input_probability, self.gamma) * cross_entropy
        loss_mean = torch.mean(loss)
        return loss_mean