"""DDIM sampler for PXRD-Diff.

Generates crystal structures (fractional coords + lattice parameters)
from Gaussian noise, conditioned on a PXRD pattern and known atom types.

Phase 4 additions:
- sample_ensemble: generate N candidates per pattern using independent init noise
- refine_structure: Rietveld-style gradient refinement of frac_coords (and optionally
  lattice) by minimizing 1 - Pearson against the target PXRD via DiffPXRD
- pearson_score: per-sample Pearson for ensemble selection
- lattice_params_to_matrix: differentiable (a,b,c,α,β,γ) -> 3x3 matrix
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pxrd_diff.diffusion import cosine_alpha_bar


def lattice_params_to_matrix(lp: torch.Tensor) -> torch.Tensor:
    """Differentiably convert lattice params (a,b,c,α,β,γ) -> 3x3 lattice matrix.

    Uses pymatgen row-vector convention: a along x, b in xy-plane.

    Args:
        lp: (..., 6) tensor with [a, b, c, alpha_deg, beta_deg, gamma_deg]

    Returns:
        (..., 3, 3) lattice matrix with rows = [a_vec, b_vec, c_vec]
    """
    a = lp[..., 0]
    b = lp[..., 1]
    c = lp[..., 2]
    alpha = lp[..., 3] * (math.pi / 180.0)
    beta = lp[..., 4] * (math.pi / 180.0)
    gamma = lp[..., 5] * (math.pi / 180.0)

    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma).clamp(min=1e-8)

    zero = torch.zeros_like(a)
    a_vec = torch.stack([a, zero, zero], dim=-1)
    b_vec = torch.stack([b * cos_g, b * torch.sin(gamma), zero], dim=-1)
    c_x = c * cos_b
    c_y = c * (cos_a - cos_b * cos_g) / sin_g
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    c_z = torch.sqrt(c_z_sq.clamp(min=1e-8))
    c_vec = torch.stack([c_x, c_y, c_z], dim=-1)

    return torch.stack([a_vec, b_vec, c_vec], dim=-2)


def _interp_to_bins(pattern: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Linear-interpolate a (B, L) pattern to (B, n_bins)."""
    if pattern.shape[-1] == n_bins:
        return pattern
    return F.interpolate(
        pattern.unsqueeze(1), size=n_bins, mode="linear", align_corners=True,
    ).squeeze(1)


def pearson_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample Pearson correlation between two (B, L) patterns.

    Resamples target to pred's bin count if needed. Returns (B,) in [-1, 1].
    """
    targ = _interp_to_bins(target, pred.shape[-1])
    pred_z = pred - pred.mean(dim=-1, keepdim=True)
    targ_z = targ - targ.mean(dim=-1, keepdim=True)
    pred_n = pred_z / pred_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    targ_n = targ_z / targ_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (pred_n * targ_n).sum(dim=-1)


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
               wyckoff: torch.Tensor | None = None,
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
                wyckoff=wyckoff,
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

    @torch.no_grad()
    def sample_ensemble(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
                        lattice_init: torch.Tensor, mask: torch.Tensor,
                        n_samples: int = 10,
                        eta: float | None = None,
                        wyckoff: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate n_samples candidate structures per input pattern.

        Each candidate gets independent initial noise; with eta > 0, additional
        noise is injected at each DDIM step. Replicates the input batch internally
        so all n_samples * B candidates run through the model in one forward pass.

        Args:
            pxrd: (B, n_bins) PXRD patterns
            atom_types: (B, N)
            lattice_init: (B, 3, 3) lattice matrix used for periodic distances
            mask: (B, N)
            n_samples: candidates per input pattern
            eta: stochasticity per DDIM step. None = use self.eta
            wyckoff: (B, N) optional

        Returns:
            coords: (B, n_samples, N, 3) fractional coords (in [0, 1))
            lat_params: (B, n_samples, 6) lattice parameters (denormalized if
                        lat_mean/lat_std were set on the sampler)
        """
        B, N = atom_types.shape
        S = n_samples

        # Replicate inputs along a new "samples" axis, then flatten to (B*S, ...)
        pxrd_rep = pxrd.unsqueeze(1).expand(-1, S, *([-1] * (pxrd.dim() - 1))).reshape(B * S, *pxrd.shape[1:])
        at_rep = atom_types.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)
        lat_rep = lattice_init.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3)
        mask_rep = mask.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)
        if wyckoff is not None:
            wyck_rep = wyckoff.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)
        else:
            wyck_rep = None

        old_eta = self.eta
        if eta is not None:
            self.eta = eta
        try:
            coords_flat, lat_params_flat = self.sample(
                pxrd_rep, at_rep, lat_rep, mask_rep, wyck_rep,
            )
        finally:
            self.eta = old_eta

        coords = coords_flat.view(B, S, N, 3)
        lat_params = lat_params_flat.view(B, S, 6)
        return coords, lat_params

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


# ---------- Phase 4 helpers --------------------------------------------------------

def select_best_by_pearson(coords: torch.Tensor, lat_params: torch.Tensor,
                           atom_types: torch.Tensor, mask: torch.Tensor,
                           lattice_for_score: torch.Tensor,
                           target_pxrd: torch.Tensor,
                           debye: nn.Module,
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick best candidate per input pattern by DiffPXRD-Pearson against target.

    Args:
        coords: (B, S, N, 3) candidate fractional coordinates
        lat_params: (B, S, 6) candidate lattice parameters
        atom_types: (B, N)
        mask: (B, N)
        lattice_for_score: (B, 3, 3) lattice matrix used to compute DiffPXRD.
            For coord-only eval pass true lattice; otherwise build from predicted
            lat_params via lattice_params_to_matrix and pass per-candidate.
        target_pxrd: (B, n_bins_target) target patterns
        debye: a DiffPXRD instance (in eval mode, on the right device)

    Returns:
        best_coords: (B, N, 3)
        best_lat_params: (B, 6)
        best_scores: (B,) Pearson values for the picked candidates
    """
    B, S, N, _ = coords.shape

    coords_flat = coords.reshape(B * S, N, 3)
    at_rep = atom_types.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)
    mask_rep = mask.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N)

    if lattice_for_score.dim() == 3:
        # (B, 3, 3) -> (B, S, 3, 3) -> (B*S, 3, 3)
        lat_rep = lattice_for_score.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3)
    else:
        # already (B, S, 3, 3)
        lat_rep = lattice_for_score.reshape(B * S, 3, 3)

    with torch.no_grad():
        pred_pxrd = debye(coords_flat, at_rep, lat_rep, mask_rep)  # (B*S, n_bins_diff)

    targ_rep = target_pxrd.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
    scores = pearson_score(pred_pxrd, targ_rep).view(B, S)

    best_idx = scores.argmax(dim=1)  # (B,)
    arange_b = torch.arange(B, device=coords.device)

    best_coords = coords[arange_b, best_idx]
    best_lat_params = lat_params[arange_b, best_idx]
    best_scores = scores[arange_b, best_idx]
    return best_coords, best_lat_params, best_scores


def refine_structure(frac_coords: torch.Tensor, atom_types: torch.Tensor,
                     lattice: torch.Tensor, mask: torch.Tensor,
                     target_pxrd: torch.Tensor, debye: nn.Module,
                     steps: int = 200, lr: float = 1e-3,
                     refine_lattice: bool = False,
                     ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Rietveld-style refinement: gradient descent on coords (and optionally
    lattice) to minimize 1 - Pearson(DiffPXRD(structure), target).

    Coordinates are kept on the [0, 1) torus by wrapping at every step.
    Loss is mean across the batch but gradients are per-sample.

    Args:
        frac_coords: (B, N, 3) starting fractional coords
        atom_types: (B, N)
        lattice: (B, 3, 3) lattice matrix; cloned and made trainable iff
            refine_lattice=True
        mask: (B, N) bool
        target_pxrd: (B, n_bins_target) target PXRD patterns
        debye: DiffPXRD instance
        steps: optimizer iterations
        lr: Adam learning rate
        refine_lattice: if True, also optimizes the 3x3 lattice matrix

    Returns:
        coords_refined: (B, N, 3) wrapped to [0, 1)
        lattice_refined: (B, 3, 3)
        loss_history: per-step mean loss values
    """
    coords = frac_coords.detach().clone().requires_grad_(True)
    if refine_lattice:
        lat = lattice.detach().clone().requires_grad_(True)
        params = [coords, lat]
    else:
        lat = lattice.detach()
        params = [coords]

    # Atoms outside the mask should not contribute gradient to coords; we
    # multiply their grads by mask after backward so they don't drift.
    atoms_mask = mask.float().unsqueeze(-1)  # (B, N, 1)

    optimizer = torch.optim.Adam(params, lr=lr)
    history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()
        coords_wrap = coords - torch.floor(coords)  # differentiable wrap to [0,1)
        pred = debye(coords_wrap, atom_types, lat, mask)
        scores = pearson_score(pred, target_pxrd)        # (B,) in [-1, 1]
        loss = (1 - scores).mean()
        loss.backward()
        # Zero out grads on padding atoms
        if coords.grad is not None:
            coords.grad = coords.grad * atoms_mask
        optimizer.step()
        history.append(float(loss.item()))

    with torch.no_grad():
        coords_final = (coords - torch.floor(coords)).detach()
        lat_final = lat.detach()
    return coords_final, lat_final, history
