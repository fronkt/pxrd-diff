"""DDIM sampler for PXRD-Diff.

Generates crystal structures (fractional coords + lattice parameters)
from Gaussian noise, conditioned on a PXRD pattern and known atom types.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pxrd_diff.diffusion import cosine_alpha_bar


class DDIMSampler:
    def __init__(self, encoder: nn.Module, denoiser: nn.Module,
                 n_steps: int = 50, eta: float = 0.0,
                 lat_mean: torch.Tensor | None = None,
                 lat_std: torch.Tensor | None = None,
                 predict_x0: bool = False):
        self.encoder = encoder
        self.denoiser = denoiser
        self.n_steps = n_steps
        self.eta = eta
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.predict_x0 = predict_x0

    @torch.no_grad()
    def sample(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
               lattice_init: torch.Tensor, mask: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        device = pxrd.device
        B, N = atom_types.shape

        pxrd_global, pxrd_feats = self.encoder(pxrd)

        x_t = torch.randn(B, N, 3, device=device)
        l_t = torch.randn(B, 6, device=device)

        timesteps = torch.linspace(1.0 - 0.5 / self.n_steps, 1e-4,
                                    self.n_steps + 1, device=device)

        for i in range(self.n_steps):
            t_now, t_next = timesteps[i], timesteps[i + 1]
            ab_now = cosine_alpha_bar(t_now.unsqueeze(0))
            ab_next = cosine_alpha_bar(t_next.unsqueeze(0))
            t_batch = t_now.expand(B)

            lattice_for_dist = self._params_to_approx_lattice(l_t, lattice_init)

            pred_c, pred_l = self.denoiser(
                x_t % 1.0, atom_types, lattice_for_dist, t_batch,
                pxrd_global, pxrd_feats, mask, l_t,
            )

            if self.predict_x0:
                # Model output is residual: x0_pred = x_t + pred
                x0_c = (x_t % 1.0) + pred_c
                x0_l = l_t + pred_l
                # Convert to eps for DDIM step
                eps_c = (x_t - ab_now.view(-1, 1, 1).sqrt() * x0_c) / (1 - ab_now.view(-1, 1, 1)).sqrt().clamp(min=0.05)
                eps_l = (l_t - ab_now.view(-1, 1).sqrt() * x0_l) / (1 - ab_now.view(-1, 1)).sqrt().clamp(min=0.05)
            else:
                eps_c = pred_c
                eps_l = pred_l

            x_t = self._ddim_step(x_t, eps_c, ab_now, ab_next) % 1.0
            l_t = self._ddim_step(l_t, eps_l, ab_now, ab_next)

        if self.lat_mean is not None and self.lat_std is not None:
            lat_mean = self.lat_mean.to(device)
            lat_std = self.lat_std.to(device)
            lattice_params = l_t * lat_std + lat_mean
        else:
            lattice_params = l_t

        return x_t % 1.0, lattice_params

    def _ddim_step(self, x_t: torch.Tensor, eps: torch.Tensor,
                   ab_now: torch.Tensor, ab_next: torch.Tensor,
                   ) -> torch.Tensor:
        shape = [1] * (x_t.dim() - 1)
        ab_now = ab_now.view(-1, *shape)
        ab_next = ab_next.view(-1, *shape)
        x0_pred = (x_t - (1 - ab_now).sqrt() * eps) / ab_now.sqrt().clamp(min=0.05)
        sigma = self.eta * ((1 - ab_next) / (1 - ab_now) * (1 - ab_now / ab_next)).sqrt()
        dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
        x_next = ab_next.sqrt() * x0_pred + dir_xt
        if self.eta > 0:
            x_next = x_next + sigma * torch.randn_like(x_t)
        return x_next

    def _params_to_approx_lattice(self, l_params_norm: torch.Tensor,
                                  lattice_init: torch.Tensor) -> torch.Tensor:
        return lattice_init
