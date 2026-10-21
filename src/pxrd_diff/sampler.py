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


def lattice_matrix_to_params(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) lattice matrix (rows = lattice vectors) -> (..., 6) params
    [a, b, c, alpha_deg, beta_deg, gamma_deg]; inverse of lattice_params_to_matrix."""
    lengths = m.norm(dim=-1)

    def angle(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-8)
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    a_vec, b_vec, c_vec = m[..., 0, :], m[..., 1, :], m[..., 2, :]
    return torch.cat([lengths, torch.stack([angle(b_vec, c_vec), angle(a_vec, c_vec),
                                            angle(a_vec, b_vec)], dim=-1)], dim=-1)


def noisy_lattice_matrix(l_norm: torch.Tensor, lat_mean: torch.Tensor,
                         lat_std: torch.Tensor) -> torch.Tensor:
    """Normalised (possibly noisy) lattice params -> a finite lattice matrix.

    Used for the denoiser's distance features when they are taken from the
    lattice being denoised rather than from the reference cell (JAC R2 Q1).
    Lengths are clamped to [0.5, 100] Å and angles to [10, 170]°, as for the
    sampled cells elsewhere in this module.
    """
    lp = l_norm * lat_std.to(l_norm) + lat_mean.to(l_norm)
    lp = torch.cat([lp[..., :3].clamp(0.5, 100.0), lp[..., 3:].clamp(10.0, 170.0)],
                   dim=-1)
    return lattice_params_to_matrix(lp)


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
                 predict_x0: bool = False,
                 dist_lattice: str = "clean"):
        """dist_lattice selects the lattice behind the denoiser's distance
        features. "clean" (hat5032 v1/R1 behaviour, v10-v22) uses lattice_init
        throughout; in the learned-head mode the caller passes the reference
        cell there, so that mode is not a no-true-lattice measurement (JAC R2
        Q1). "noisy" (checkpoints trained with --dist-lattice noisy) uses the
        lattice being denoised, l_t; with fixed_lattice=True, l_t is replaced
        at every step by the forward-noised given cell, matching training."""
        if dist_lattice not in ("clean", "noisy"):
            raise ValueError(f"dist_lattice must be 'clean' or 'noisy', got {dist_lattice!r}")
        if dist_lattice == "noisy" and (lat_mean is None or lat_std is None):
            raise ValueError("dist_lattice='noisy' needs lat_mean and lat_std")
        self.encoder = encoder
        self.denoiser = denoiser
        self.n_steps = n_steps
        self.eta = eta
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.predict_x0 = predict_x0
        self.dist_lattice = dist_lattice

    def sample(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
               lattice_init: torch.Tensor, mask: torch.Tensor,
               wyckoff: torch.Tensor | None = None,
               guide_debye: "nn.Module | None" = None,
               guide_target: "torch.Tensor | None" = None,
               guide_scale: float = 0.0,
               guide_start_t: float = 0.5,
               guide_clip: float = 0.05,
               fixed_lattice: bool = False,
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDIM reverse sampling with optional Debye-gradient guidance (Phase 9.2).

        fixed_lattice: lattice_init is a given cell (oracle, indexer, aux head)
        rather than a placeholder. Only read when dist_lattice == "noisy"; with
        "clean" lattice_init is always used for the distances.

        Guidance: at each step where t <= guide_start_t (late, low-noise steps),
        compute x0_pred from the current (x_t, eps), render its PXRD via
        guide_debye, take ∇_{x0_pred}[1 - Pearson(pred, target)], nudge x0_pred
        opposite the gradient, and recompute eps. Step size is annealed by
        (1 - t_now) and norm-clipped per-sample to guide_clip to stay on the
        data manifold. Lattice is guided when not fixed by --true/--aux/--index;
        coordinates are always guided when active.
        """
        device = pxrd.device
        B, N = atom_types.shape

        do_guide = (guide_scale > 0.0 and guide_debye is not None
                    and guide_target is not None)

        with torch.no_grad():
            pxrd_global, pxrd_feats = self.encoder(pxrd)

            x_t = torch.randn(B, N, 3, device=device)
            l_t = torch.randn(B, 6, device=device)

            # "noisy" distance lattice with a given cell: l_t follows q(l_t | l_0)
            # of that cell instead of the model's own lattice channel.
            given_l0 = None
            if self.dist_lattice == "noisy" and fixed_lattice:
                given_l0 = ((lattice_matrix_to_params(lattice_init)
                             - self.lat_mean.to(device)) / self.lat_std.to(device))

        timesteps = torch.linspace(1.0 - 0.5 / self.n_steps, 1e-4,
                                    self.n_steps + 1, device=device)

        for i in range(self.n_steps):
            t_now, t_next = timesteps[i], timesteps[i + 1]
            ab_now = cosine_alpha_bar(t_now.unsqueeze(0))
            ab_next = cosine_alpha_bar(t_next.unsqueeze(0))
            t_batch = t_now.expand(B)

            with torch.no_grad():
                if given_l0 is not None:
                    l_t = (ab_now.sqrt() * given_l0
                           + (1 - ab_now).sqrt() * torch.randn_like(given_l0))
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
                    eps_c = (x_t - ab_now.view(-1, 1, 1).sqrt() * x0_c) / (1 - ab_now.view(-1, 1, 1)).sqrt().clamp(min=0.05)
                    eps_l = (l_t - ab_now.view(-1, 1).sqrt() * x0_l) / (1 - ab_now.view(-1, 1)).sqrt().clamp(min=0.05)
                else:
                    eps_c = pred_c
                    eps_l = pred_l

            # ---- Phase 9.2: Debye-gradient guidance ----------------------
            if do_guide and float(t_now.item()) <= guide_start_t:
                # Recompute x0_c from eps to keep this branch self-contained
                with torch.no_grad():
                    abc = ab_now.view(-1, 1, 1)
                    x0_c_recon = (x_t - (1 - abc).sqrt() * eps_c) / abc.sqrt().clamp(min=0.05)
                    abl = ab_now.view(-1, 1)
                    x0_l_recon = (l_t - (1 - abl).sqrt() * eps_l) / abl.sqrt().clamp(min=0.05)
                    # Convert l_t (normalised) to physical params for DiffPXRD
                    if self.lat_mean is not None and self.lat_std is not None:
                        lp_phys = (x0_l_recon * self.lat_std.to(device)
                                   + self.lat_mean.to(device))
                    else:
                        lp_phys = x0_l_recon
                    lp_phys = torch.cat([
                        lp_phys[..., :3].clamp(0.5, 100.0),
                        lp_phys[..., 3:].clamp(10.0, 170.0),
                    ], dim=-1)
                    lat_mat_phys = lattice_params_to_matrix(lp_phys)
                    if given_l0 is not None:
                        lat_mat_phys = lattice_init
                # Enable grad just for the Debye + Pearson path on x0_c
                x0_c_g = (x0_c_recon % 1.0).detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    pred_pxrd = guide_debye(x0_c_g, atom_types, lat_mat_phys, mask)
                    scores = pearson_score(pred_pxrd, guide_target)
                    loss = (1.0 - scores).sum()
                    grad = torch.autograd.grad(loss, x0_c_g)[0]
                with torch.no_grad():
                    # Anneal: stronger guidance near low noise (small t_now)
                    eff = float(guide_scale) * (1.0 - float(t_now.item()))
                    # Per-sample L2-norm clip on the nudge
                    flat_norm = grad.reshape(B, -1).norm(dim=1).clamp(min=1e-6)
                    scale_clip = (guide_clip / flat_norm).clamp(max=1.0).view(B, 1, 1)
                    nudge = grad * (eff * scale_clip)
                    # Zero on padding atoms
                    nudge = nudge * mask.float().unsqueeze(-1)
                    x0_c_nudged = x0_c_recon - nudge
                    # Reconstruct eps_c from nudged x0
                    eps_c = (x_t - abc.sqrt() * x0_c_nudged) / (1 - abc).sqrt().clamp(min=0.05)

            with torch.no_grad():
                x_t = self._ddim_step(x_t, eps_c, ab_now, ab_next) % 1.0
                l_t = self._ddim_step(l_t, eps_l, ab_now, ab_next)

        if given_l0 is not None:
            l_t = given_l0
        if self.lat_mean is not None and self.lat_std is not None:
            lat_mean = self.lat_mean.to(device)
            lat_std = self.lat_std.to(device)
            lattice_params = l_t * lat_std + lat_mean
        else:
            lattice_params = l_t

        return x_t % 1.0, lattice_params

    def sample_ensemble(self, pxrd: torch.Tensor, atom_types: torch.Tensor,
                        lattice_init: torch.Tensor, mask: torch.Tensor,
                        n_samples: int = 10,
                        eta: float | None = None,
                        wyckoff: torch.Tensor | None = None,
                        guide_debye: "nn.Module | None" = None,
                        guide_target: "torch.Tensor | None" = None,
                        guide_scale: float = 0.0,
                        guide_start_t: float = 0.5,
                        guide_clip: float = 0.05,
                        fixed_lattice: bool = False,
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

        # Replicate guide_target across the S-axis too (per-sample target)
        if guide_target is not None:
            guide_target_rep = guide_target.unsqueeze(1).expand(
                -1, S, *([-1] * (guide_target.dim() - 1))
            ).reshape(B * S, *guide_target.shape[1:])
        else:
            guide_target_rep = None

        old_eta = self.eta
        if eta is not None:
            self.eta = eta
        try:
            coords_flat, lat_params_flat = self.sample(
                pxrd_rep, at_rep, lat_rep, mask_rep, wyck_rep,
                guide_debye=guide_debye,
                guide_target=guide_target_rep,
                guide_scale=guide_scale,
                guide_start_t=guide_start_t,
                guide_clip=guide_clip,
                fixed_lattice=fixed_lattice,
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
        if self.dist_lattice == "noisy":
            return noisy_lattice_matrix(l_params_norm, self.lat_mean, self.lat_std)
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
