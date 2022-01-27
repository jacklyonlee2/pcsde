import torch
import torch.nn as nn
import numpy as np


class DSMLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, net, x):
        B, N, _ = x.shape
        sigma = torch.full((B, 1, 1), self.sigma).to(x)
        pertbx = x + torch.randn_like(x) * sigma
        score = net(pertbx)
        # Modified the original DSM for numerical stability:
        # >>> score - (-1 / (sigma ** 2) * (pertbx - x))) ** 2
        # <<< (score + (pertbx - x)) ** 2 * (1 / sigma)
        loss = (score + (pertbx - x)) ** 2 * (1 / sigma)
        loss = 0.5 * loss.sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def sample(self, net, prior, n_steps=100):
        def sample_steps(x):
            for _ in range(n_steps):
                noise = torch.randn_like(x)
                x = x + noise * self.sigma
                score = net(x)
                x = x + score
                yield score.abs().mean().cpu().item(), x

        _, x = min(sample_steps(prior))
        return x


class AnnealedDSMLoss(nn.Module):
    def __init__(self, sigma_start=1, sigma_end=0.01, n_sigmas=10):
        super().__init__()
        self.sigmas = np.exp(
            np.linspace(
                np.log(sigma_start),
                np.log(sigma_end),
                n_sigmas,
            )
        )

    def forward(self, net, x):
        B, N, _ = x.shape
        sigma = np.random.choice(self.sigmas, (B, 1, 1))
        sigma = torch.from_numpy(sigma).to(x)
        pertbx = x + torch.randn_like(x) * sigma
        label = sigma.expand(B, N, 1)
        condx = torch.cat((pertbx, label), dim=-1)
        score = net(condx)
        # Modified the original DSM for numerical stability:
        # >>> (score - (-1 / (sigma ** 2) * (pertbx - x))) ** 2)) * sigma ** 2
        # <<< (score + (pertbx - x)) ** 2 * (1 / sigma)
        loss = (score + (pertbx - x)) ** 2 * (1 / sigma)
        loss = 0.5 * loss.sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def sample(self, net, prior, n_steps_per_sigma=10):
        def sample_steps(x):
            B, N, _ = x.shape
            for n, sigma in enumerate(self.sigmas):
                label = torch.full((B, N, 1), sigma).to(x)
                for t in range(n_steps_per_sigma):
                    noise = torch.randn_like(x)
                    x = x + noise * sigma
                    condx = torch.cat((x, label), dim=-1)
                    score = net(condx)
                    x = x + score
                    yield score.abs().mean().cpu().item(), x

        _, x = min(sample_steps(prior))
        return x
