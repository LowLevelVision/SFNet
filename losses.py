import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss