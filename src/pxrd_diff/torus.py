"""Periodic (wrapped-normal) diffusion for fractional coordinates (JAC R2 redesign).

Fractional coordinates live on the torus [0, 1)^3. The forward process adds
Gaussian noise and wraps, F_t = wrap(F_0 + sigma_t * eps), with sigma_t
log-uniform in t between sigma_min and sigma_max (variance exploding, as in
DiffCSP, Jiao et al. 2023). The scale is set by the cell: sigma_max = 0.5 is
already close to uniform on a period-1 torus, sigma_min = 0.005 is ~0.03 A in a
6 A cell. The v10-v24 models instead applied the Euclidean VP schedule to the
torus, which leaves the coordinates as uniform noise for ~81 % of timesteps
(paper/submissions/JAC-R2/analysis/denoiser_collapse/vp_torus_check.txt).

The model learns the score of the wrapped normal, d/dF_t log q(F_t | F_0),
divided by its RMS at that sigma, so a zero prediction scores 1.0 on the loss.
The target depends only on wrap(F_t - F_0), so it is unchanged by a global
translation of the structure and by integer shifts of single coordinates.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_IMAGES = 10  # periodic images summed on each side in the wrapped-normal density


def wrap_signed(x: torch.Tensor) -> torch.Tensor:
    """Map to [-0.5, 0.5]."""
    return x - torch.round(x)


def wrapped_normal_score(d: torch.Tensor, sigma: torch.Tensor,
                         n_images: int = N_IMAGES) -> torch.Tensor:
    """d/dx log sum_k N(x + k; 0, sigma^2), evaluated at x = d.

    d: any shape; sigma: broadcastable to d.
    """
    k = torch.arange(-n_images, n_images + 1, device=d.device, dtype=d.dtype)
    x = wrap_signed(d).unsqueeze(-1) + k                       # (..., 2K+1)
    s = torch.as_tensor(sigma, dtype=d.dtype, device=d.device).broadcast_to(d.shape).unsqueeze(-1)
    w = torch.softmax(-x ** 2 / (2 * s ** 2), dim=-1)
    return (w * (-x / s ** 2)).sum(-1)


def _score_rms(sigma: float, n_grid: int = 20000) -> float:
    """sqrt(E[score^2]) under the wrapped normal of width sigma, by quadrature."""
    x = (torch.arange(n_grid, dtype=torch.float64) + 0.5) / n_grid - 0.5
    k = torch.arange(-N_IMAGES, N_IMAGES + 1, dtype=torch.float64)
    xx = x.unsqueeze(-1) + k
    logw = -xx ** 2 / (2 * sigma ** 2)
    p = torch.logsumexp(logw, dim=-1).exp()
    score = (torch.softmax(logw, dim=-1) * (-xx / sigma ** 2)).sum(-1)
    return float(((p * score ** 2).sum() / p.sum()).sqrt())


class TorusVE(nn.Module):
    """Wrapped-normal VE process on fractional coordinates."""

    def __init__(self, sigma_min: float = 0.005, sigma_max: float = 0.5,
                 n_table: int = 400):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        log_s = torch.linspace(math.log(sigma_min), math.log(sigma_max), n_table)
        rms = torch.tensor([_score_rms(math.exp(v)) for v in log_s.tolist()])
        self.register_buffer("log_s_table", log_s)
        self.register_buffer("log_rms_table", rms.log().float())

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """t in [0, 1] -> sigma, log-linear from sigma_min to sigma_max."""
        return self.sigma_min ** (1 - t) * self.sigma_max ** t

    def score_rms(self, sigma: torch.Tensor) -> torch.Tensor:
        """sqrt(E[score^2]) at sigma, interpolated in log-log space."""
        ls = sigma.log().clamp(self.log_s_table[0], self.log_s_table[-1])
        pos = (ls - self.log_s_table[0]) / (self.log_s_table[1] - self.log_s_table[0])
        i0 = pos.floor().long().clamp(0, len(self.log_s_table) - 2)
        frac = pos - i0
        lr = self.log_rms_table[i0] * (1 - frac) + self.log_rms_table[i0 + 1] * frac
        return lr.exp()

    def forward_q(self, f0: torch.Tensor, t: torch.Tensor):
        """Noise clean coordinates f0 (B, N, 3) to level t (B,).

        Returns (f_t in [0, 1), normalised score target, sigma (B,)).
        """
        sigma = self.sigma(t)
        s = sigma.view(-1, 1, 1)
        f_t = (f0 + s * torch.randn_like(f0)) % 1.0
        target = wrapped_normal_score(f_t - f0, s) / self.score_rms(sigma).view(-1, 1, 1)
        return f_t, target, sigma
