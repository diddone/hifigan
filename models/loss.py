import torch
from torch import nn

class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, inputs_list, targets_list):
        loss = 0.

        for inputs, targets in zip(inputs_list, targets_list):
            for x, y in zip(inputs, targets):
                batch_size = x.size(0)
                n_features = x.numel() / batch_size
                coef = 1. / x.numel()
                print(x.shape, y.shape, coef)
                loss += coef * self.l1_loss(x, y)

        return loss

class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, melspec_gen, melspec_real):

        return self.l1_loss(melspec_gen, melspec_real)

class GeneratorAdvLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.one = torch.tensor([1.]).to(device)

    def forward(self, disc_gen_outputs):
        loss = 0.

        for x in disc_gen_outputs:
            loss += self.mse_loss(x, self.one)

        return loss

class DiscriminatorAdvLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse_loss = nn.MSELoss()

        self.zero = torch.tensor([0.]).to(device)
        self.one = torch.tensor([1.]).to(device)
    def forward(self, disc_real_outputs, disc_gen_outputs):
        disc_loss = 0.
        for y_real, y_gen in zip(disc_real_outputs, disc_gen_outputs):
            disc_loss += self.mse_loss(y_real, self.one) + self.mse_loss(y_gen, self.zero)

        return disc_loss
