"""Predictor-corrector sampler for the periodic model (model_family "csp", JAC R2).

Coordinates: reverse wrapped-normal VE diffusion, a Langevin corrector step
followed by an ancestral predictor step at each noise level (as in DiffCSP).
Cell: ancestral DDPM on the normalised lattice parameters (cosine VP schedule),
or held at a supplied clean cell (oracle / indexer modes). Same call signature
as DDIMSampler.sample / sample_ensemble, so scripts/03_sample.py can use either.

In learned mode the network never sees lattice_init: it is not read, so a
reference cell cannot leak into generation (JAC R2 Q1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pxrd_diff.diffusion import cosine_alpha_bar
from pxrd_diff.sampler import lattice_matrix_to_params
from pxrd_diff.torus import TorusVE


class PCSampler:
    def __init__(self, encoder: nn.Module, denoiser: nn.Module, torus: TorusVE,
                 lat_mean: torch.Tensor, lat_std: torch.Tensor,
                 n_steps: int = 1000, step_lr: float = 1e-5,
                 null_pattern: bool = False):
        self.encoder = encoder
        self.denoiser = denoiser
        self.torus = torus
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.n_steps = n_steps
        self.step_lr = step_lr
        self.null_pattern = null_pattern

    def _denorm(self, l_norm: torch.Tensor) -> torch.Tensor:
        return l_norm * self.lat_std + self.lat_mean

    def _net(self, f, types, l_norm, t, g, feats, mask, given, drop):
        tt = torch.full((f.shape[0],), float(t), device=f.device)
        score_n, eps_l = self.denoiser(f, types, self._denorm(l_norm), l_norm, tt,
                                       g, feats, mask, given, drop)
        sigma = self.torus.sigma(tt)
        return score_n * self.torus.score_rms(sigma).view(-1, 1, 1), eps_l

    @torch.no_grad()
    def sample(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
               lattice_init: torch.Tensor, mask: torch.Tensor,
               wyckoff: torch.Tensor | None = None,
               fixed_lattice: bool = False, guide_scale: float = 0.0,
               **_unused) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fractional coords (B, N, 3) in [0, 1), cell parameters (B, 6))."""
        if guide_scale > 0:
            raise ValueError("Debye guidance is not implemented for the PC sampler")
        B, N = atom_types.shape
        dev = pxrd.device
        g, feats = self.encoder(pxrd)
        drop = torch.full((B,), self.null_pattern, dtype=torch.bool, device=dev)
        given = torch.full((B,), fixed_lattice, dtype=torch.bool, device=dev)
        m = mask.unsqueeze(-1).to(pxrd.dtype)

        f = torch.rand(B, N, 3, device=dev)
        if fixed_lattice:
            given_params = lattice_matrix_to_params(lattice_init)
            l_norm = (given_params - self.lat_mean) / self.lat_std
        else:
            l_norm = torch.randn(B, 6, device=dev)

        T = self.n_steps
        ts = torch.linspace(1.0, 0.0, T + 1).tolist()
        for i in range(T):
            t, t_next = ts[i], ts[i + 1]
            sigma = self.torus.sigma(torch.tensor(t)).item()
            sigma_next = self.torus.sigma(torch.tensor(t_next)).item() if i < T - 1 else 0.0

            # Corrector: one Langevin step at this noise level (coordinates only).
            score, _ = self._net(f, atom_types, l_norm, t, g, feats, mask, given, drop)
            step = self.step_lr * (sigma / self.torus.sigma_min) ** 2
            f = (f + (step * score + (2 * step) ** 0.5 * torch.randn_like(f)) * m) % 1.0

            # Predictor: ancestral step sigma -> sigma_next.
            score, eps_l = self._net(f, atom_types, l_norm, t, g, feats, mask, given, drop)
            dvar = sigma ** 2 - sigma_next ** 2
            noise_std = (sigma_next ** 2 * dvar / sigma ** 2) ** 0.5
            f = (f + (dvar * score + noise_std * torch.randn_like(f)) * m) % 1.0

            if not fixed_lattice:
                ab = cosine_alpha_bar(torch.tensor(t)).item()
                ab_next = cosine_alpha_bar(torch.tensor(t_next)).item()
                x0 = ((l_norm - (1 - ab) ** 0.5 * eps_l) / ab ** 0.5).clamp(-5.0, 5.0)
                if i < T - 1:
                    s_ts = ((1 - ab_next) / (1 - ab) * (1 - ab / ab_next)) ** 0.5
                    eps_dir = (l_norm - ab ** 0.5 * x0) / (1 - ab) ** 0.5
                    l_norm = (ab_next ** 0.5 * x0
                              + max(1 - ab_next - s_ts ** 2, 0.0) ** 0.5 * eps_dir
                              + s_ts * torch.randn_like(l_norm))
                else:
                    l_norm = x0

        f = f * m
        if fixed_lattice:
            return f, given_params
        return f, self._denorm(l_norm)

    @torch.no_grad()
    def sample_ensemble(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
                        lattice_init: torch.Tensor, mask: torch.Tensor,
                        n_samples: int = 10, fixed_lattice: bool = False,
                        **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """n_samples independent candidates per pattern:
        (coords (B, S, N, 3), cell parameters (B, S, 6))."""
        B, N = atom_types.shape
        S = n_samples
        rep = lambda x: x.unsqueeze(1).expand(B, S, *x.shape[1:]).reshape(B * S, *x.shape[1:])
        kwargs.pop("eta", None)
        kwargs.pop("wyckoff", None)
        f, lp = self.sample(rep(pxrd), rep(atom_types), rep(lattice_init), rep(mask),
                            fixed_lattice=fixed_lattice, **kwargs)
        return f.view(B, S, N, 3), lp.view(B, S, 6)
