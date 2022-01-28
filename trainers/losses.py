import torch
import torch.nn as nn
import numpy as np


def randomize_sigma(x, sigmas):
    B, N, _ = x.shape
    label = torch.randint(0, len(sigmas), (B, 1, 1))
    sigma = sigmas[label.to(sigmas)]
    return sigma, label.expand(B, N, 1)


def perturb_sample(x, sigma):
    prtbx = x + torch.randn_like(x) * sigma
    targx = -1 / (sigma ** 2) * (prtbx - x)
    return prtbx, targx


def compute_dsm_loss(score, targx, lm=1):
    loss = 1 / 2 * ((score - targx) ** 2).sum(dim=-1)
    loss = (loss * lm).mean()
    return loss


class DSMLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, net, x):
        sigma = torch.full((x.size(0), 1, 1), self.sigma).to(x)
        prtbx, targx = perturb_sample(x, sigma)
        loss = compute_dsm_loss(net(prtbx), targx)
        return loss


class AnnealedDSMLoss(nn.Module):
    def __init__(self, sigma_start=1, sigma_end=0.01, n_sigmas=10):
        super().__init__()
        self.sigmas = torch.from_numpy(
            np.exp(
                np.linspace(
                    np.log(sigma_start),
                    np.log(sigma_end),
                    n_sigmas,
                )
            )
        )

    def forward(self, net, x):
        sigma, label = randomize_sigma(x, self.sigmas)
        prtbx, targx = perturb_sample(x, sigma)
        condx = torch.cat((pertbx, label), dim=-1)
        lmbda = sigma.view(-1, 1) ** 2
        loss = compute_dsm_loss(net(condx), targx, lm=lmbda)
        return loss
