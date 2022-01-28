import os
import wandb
import torch
import numpy as np
from tqdm import tqdm

from .utils import plot_samples
from metrics import compute_cd, compute_metrics


class Trainer:
    def __init__(
        self,
        net,
        device,
        batch_size,
        loss=None,
        sample=None,
        opt=None,
        sch=None,
        max_epoch=None,
        log_every_n_step=None,
        val_every_n_epoch=None,
        ckpt_every_n_epoch=None,
        ckpt_dir=None,
    ):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.loss = loss and loss.to(device)
        self.sample = sample and sample.to(device)
        self.opt = opt
        self.sch = sch
        self.step = 0
        self.epoch = 0
        self.max_epoch = max_epoch
        self.log_every_n_step = log_every_n_step
        self.val_every_n_epoch = val_every_n_epoch
        self.ckpt_every_n_epoch = ckpt_every_n_epoch
        self.ckpt_dir = ckpt_dir

    def _state_dict(self):
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "sch": self.sch.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
        }

    def _load_state_dict(self, state_dict):
        for k, m in {
            "net": self.net,
            "opt": self.opt,
            "sch": self.sch,
        }.items():
            m and m.load_state_dict(state_dict[k])
        self.step, self.epoch, self.max_epoch, self.kl_warmup_epoch = map(
            state_dict.get,
            (
                "step",
                "epoch",
                "max_epoch",
            ),
        )

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.epoch}.pth")
        torch.save(self._state_dict(), ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        if not ckpt_path:  # Find last checkpoint in ckpt_dir
            ckpt_paths = [p for p in os.listdir(self.ckpt_dir) if p.endswith(".pth")]
            assert ckpt_paths, "No checkpoints found."
            ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
        self._load_state_dict(torch.load(ckpt_path))

    def _train_step(self, x, mu, std):
        return self.loss(self.net, x)

    def train(self, train_loader, val_loader):
        while self.epoch < self.max_epoch:

            if self.epoch % self.val_every_n_epoch == 0:
                results = self.test(val_loader, validate=True)
                wandb.log(
                    {
                        **results,
                        "Step": self.step,
                        "Epoch": self.epoch,
                    }
                )
            if self.epoch % self.ckpt_every_n_epoch == 0:
                self.save_checkpoint()

            self.net.train()
            with tqdm(train_loader) as t:
                for batch in t:
                    batch = [_.to(self.device) for _ in batch]
                    loss = self._train_step(*batch)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    t.set_description(f"Epoch:{self.epoch}|Loss:{loss.item():.4f}")
                    if self.step % self.log_every_n_step == 0:
                        wandb.log(
                            {
                                "Loss": loss.cpu(),
                                "Step": self.step,
                                "Epoch": self.epoch,
                            }
                        )

                    self.step += 1
            self.sch.step()
            self.epoch += 1

    def _test_step(self, x, mu, std):
        prior = torch.rand_like(x) * 2 - 1
        o = self.sample(self.net, prior)
        # x, o = x * std + mu, o * std + mu  # denormalize
        return o, x

    def _test_end(self, o, x, validate):
        return {
            **compute_metrics(
                o,
                x,
                batch_size=self.batch_size,
                exclude_knn=validate,
            ),
            "Samples": plot_samples(o),
        }

    @torch.no_grad()
    def test(self, test_loader, validate=False):
        results = []
        self.net.eval()
        for batch in tqdm(test_loader):
            batch = [_.to(self.device) for _ in batch]
            results.append(self._test_step(*batch))
        results = [torch.cat(_, dim=0) for _ in zip(*results)]
        return self._test_end(*results, validate)
