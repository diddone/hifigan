import torch
from torch import nn

class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss(reduction='none')
    def forward(self, inputs, targets):
        loss = 0.

        for x, y in zip(inputs, targets):
            batch_size = x.size(0)
            n_features = x.numel() / batch_size
            coef = 1 / x.numel()
            loss += coef * self.l1_loss(x, y)

        return loss

class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, melspec_gen, melspec_real):

        return l1_loss(melspec_gen - melspec_real)

class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, disc_gen_output):
        return self.mse_loss(x, 1.)

class DiscriminatorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, disc_real_output, disc_gen_output):
        return self.mse_loss(disc_real_output, 1.) + self.mse_loss(disc_gen_output, 0.)
