import torch
import torch.nn as nn
import numpy as np


def langevin_dynamics(net, x, n_steps, step_lr, label=None):
    for t in range(n_steps):
        condx = x if label is None else torch.cat((x, label), dim=-1)
        noise = torch.randn_like(x) * np.sqrt(step_lr * 2)
        x = x + noise
        score = net(condx)
        x = x + step_lr * score
    return x


class LangevinSampler(nn.Module):
    def __init__(self, n_steps=50, step_lr=5e-4):
        super().__init__()
        self.n_steps = n_steps
        self.step_lr = step_lr

    @torch.no_grad()
    def forward(self, net, x):
        return langevin_dynamics(
            net,
            x,
            self.n_steps,
            self.step_lr,
        )


class AnnealedLangevinSampler(nn.Module):
    def __init__(
        self,
        sigmas,
        n_steps_each=5,
        step_lr=5e-4,
    ):
        super().__init__()
        self.sigmas = sigmas
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr

    @torch.no_grad()
    def forward(self, net, x):
        B, N, _ = x.shape
        for label, sigma in enumerate(self.sigmas):
            label = torch.full((B, N, 1), label).to(x)
            step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2
            x = langevin_dynamics(
                net,
                x,
                self.n_steps_each,
                step_size,
                label=label,
            )
        return x
