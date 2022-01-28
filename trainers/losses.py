import torch
import torch.nn as nn
import numpy as np


def randomize_sigma(x, sigmas):
    B, N, _ = x.shape
    label = np.random.randint(0, len(sigmas), (B, 1, 1))
    sigma = torch.from_numpy(sigmas[label]).to(x)
    label = torch.from_numpy(label).to(x).expand(B, N, 1)
    return sigma, label


def compute_dsm_loss(x, prtbx, sigma, score, lm=1):
    target = -1 / (sigma ** 2) * (prtbx - x)
    loss = 0.5 * ((score - target) ** 2).sum(dim=-1)
    loss = (loss * lm).mean()
    return loss


class DSMLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, net, x):
        sigma = torch.full((x.size(0), 1, 1), self.sigma).to(x)
        prtbx = x + torch.randn_like(x) * sigma
        return compute_dsm_loss(
            x,
            prtbx,
            sigma,
            net(prtbx),
        )


class AnnealedDSMLoss(nn.Module):
    def __init__(self, sigma_start=1, sigma_end=0.1, n_sigmas=10):
        super().__init__()
        self.sigmas = np.geomspace(
            sigma_start,
            sigma_end,
            num=n_sigmas,
        )

    def forward(self, net, x):
        sigma, label = randomize_sigma(x, self.sigmas)
        prtbx = x + torch.randn_like(x) * sigma
        condx = torch.cat((prtbx, label), dim=-1)
        return compute_dsm_loss(
            x,
            prtbx,
            sigma,
            net(condx),
            lm=sigma.view(-1, 1) ** 2,
        )
