"""Gate G0 for the JAC R2 redesign (model_family "csp").

  - the wrapped-normal score is the gradient of the wrapped-normal log-density,
    and its normalisation makes the zero predictor score 1.0
  - the denoiser is invariant to a global periodic translation and to +1 on
    any single coordinate, equivariant to atom permutations, deaf to padding
    atoms, and does read the pattern and the cell_given flag
  - the PC sampler never reads lattice_init in learned mode and returns the
    supplied cell in fixed-cell mode
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.model.csp_denoiser import CSPDenoiser             # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder             # noqa: E402
from pxrd_diff.sampler import lattice_matrix_to_params, lattice_params_to_matrix  # noqa: E402
from pxrd_diff.sampler_pc import PCSampler                       # noqa: E402
from pxrd_diff.torus import TorusVE, wrapped_normal_score        # noqa: E402

torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("sigma", [0.01, 0.1, 0.3, 0.5])
def test_score_is_gradient_of_wrapped_normal_log_density(sigma):
    d = torch.linspace(-0.49, 0.49, 41, dtype=torch.float64, requires_grad=True)
    k = torch.arange(-10, 11, dtype=torch.float64)
    logp = torch.logsumexp(-(d.unsqueeze(-1) + k) ** 2 / (2 * sigma ** 2), dim=-1)
    (grad,) = torch.autograd.grad(logp.sum(), d)
    ours = wrapped_normal_score(d.detach(), torch.tensor(sigma, dtype=torch.float64))
    torch.testing.assert_close(ours, grad, rtol=1e-6, atol=1e-6)
    # periodic: d and d + 1 give the same score
    torch.testing.assert_close(wrapped_normal_score(d.detach() + 1.0, torch.tensor(sigma, dtype=torch.float64)),
                               ours, rtol=1e-6, atol=1e-6)


def test_normalised_target_has_unit_mean_square():
    torus = TorusVE()
    torch.manual_seed(0)
    for t_val in (0.0, 0.3, 0.6, 1.0):
        f0 = torch.rand(256, 20, 3)
        _, target, _ = torus.forward_q(f0, torch.full((256,), t_val))
        assert abs(target.pow(2).mean().item() - 1.0) < 0.05, (t_val, target.pow(2).mean().item())
    # small sigma: the wrapped normal is a plain normal, E[score^2] = 1 / sigma^2
    s = torch.tensor([0.005, 0.02])
    torch.testing.assert_close(torus.score_rms(s), 1 / s, rtol=1e-3, atol=0)


def _inputs(seed=0, B=2, N=6, d=32, L=12):
    g = torch.Generator().manual_seed(seed)
    types = torch.tensor([[14, 8, 8, 0, 0, 0], [26, 26, 8, 8, 8, 0]])
    mask = types > 0
    f = torch.rand(B, N, 3, generator=g)
    lp = torch.tensor([[4.0, 5.0, 6.0, 80.0, 100.0, 95.0], [7.0, 7.0, 7.0, 90.0, 90.0, 90.0]])
    ln = torch.randn(B, 6, generator=g)
    t = torch.tensor([0.3, 0.7])
    pg = torch.randn(B, d, generator=g) * 30
    pf = torch.randn(B, L, d, generator=g) * 300                 # large, unnormalised tokens
    given = torch.tensor([False, True])
    return f, types, lp, ln, t, pg, pf, mask, given


@pytest.fixture(scope="module")
def den():
    torch.manual_seed(0)
    m = CSPDenoiser(d_model=32, n_layers=2, n_heads=2, n_freq=4).eval()
    # make FiLM non-trivial so the test does not pass on a zero-initialised path
    for layer in m.layers:
        torch.nn.init.normal_(layer.film.weight, std=0.1)
    return m


def _run(den, f, types, lp, ln, t, pg, pf, mask, given, drop=None):
    with torch.no_grad():
        return den(f, types, lp, ln, t, pg, pf, mask, given, drop)


def test_global_translation_and_integer_shift_invariance(den):
    x = _inputs()
    s0, l0 = _run(den, *x)
    f = x[0]
    shifted = (f + torch.tensor([0.37, -0.81, 0.12])) % 1.0
    s1, l1 = _run(den, shifted, *x[1:])
    torch.testing.assert_close(s1, s0, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(l1, l0, atol=2e-5, rtol=1e-4)
    one = f.clone()
    one[1, 2, 0] += 1.0
    s2, _ = _run(den, one, *x[1:])
    torch.testing.assert_close(s2, s0, atol=2e-5, rtol=1e-4)


def test_permutation_equivariance(den):
    x = _inputs()
    f, types, *rest = x
    mask = x[7]
    perm = torch.tensor([2, 0, 1, 3, 4, 5])                    # permute the real atoms of structure 0
    f_p, types_p = f.clone(), types.clone()
    f_p[0], types_p[0] = f[0, perm], types[0, perm]
    s0, l0 = _run(den, *x)
    s1, l1 = _run(den, f_p, types_p, *rest)
    torch.testing.assert_close(s1[0], s0[0, perm], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(l1, l0, atol=2e-5, rtol=1e-4)


def test_padding_atoms_are_inert(den):
    x = _inputs()
    f, types, lp, ln, t, pg, pf, mask, given = x
    f2, types2 = f.clone(), types.clone()
    f2[~mask] = torch.rand(int((~mask).sum()), 3)
    s0, l0 = _run(den, *x)
    s1, l1 = _run(den, f2, types2, lp, ln, t, pg, pf, mask, given)
    torch.testing.assert_close(s1, s0, atol=1e-6, rtol=0)
    torch.testing.assert_close(l1, l0, atol=1e-6, rtol=0)
    assert torch.all(s0[~mask] == 0)


def test_output_reads_positions_species_pattern_and_cell_flag(den):
    x = _inputs()
    f, types, lp, ln, t, pg, pf, mask, given = x
    s0, _ = _run(den, *x)
    rel = lambda a: ((a - s0)[mask].norm() / s0[mask].norm()).item()
    # Untrained network: the bar is "a working path", orders of magnitude above the
    # collapsed v22 denoiser (~1e-7). Trained checkpoints face 1e-2 in gate G2.
    assert rel(_run(den, torch.rand_like(f), *x[1:])[0]) > 1e-3
    other = torch.where(mask, (types + 5) % 90 + 1, types)
    assert rel(_run(den, f, other, *x[2:])[0]) > 1e-3
    assert rel(_run(den, f, types, lp, ln, t, pg.roll(1, 0), pf.roll(1, 0), mask, given)[0]) > 1e-3
    assert rel(_run(den, f, types, lp, ln, t, pg, pf, mask, ~given)[0]) > 1e-3
    assert rel(_run(den, *x, drop=torch.tensor([True, True]))[0]) > 1e-3
    spread = (s0[mask] - s0[mask].mean(0)).norm() / s0[mask].norm()
    assert spread > 1e-2


def _sampler(null_pattern=False):
    torch.manual_seed(0)
    enc = PXRDEncoder(d_model=32).eval()
    den = CSPDenoiser(d_model=32, n_layers=1, n_heads=2, n_freq=4).eval()
    return PCSampler(enc, den, TorusVE(n_table=50), lat_mean=torch.tensor([5.0, 5, 5, 90, 90, 90]),
                     lat_std=torch.tensor([1.0, 1, 1, 10, 10, 10]), n_steps=6,
                     null_pattern=null_pattern)


_CELL_A = lattice_params_to_matrix(torch.tensor([[4.0, 5.0, 6.0, 80.0, 100.0, 95.0],
                                                 [7.0, 7.0, 7.0, 90.0, 90.0, 90.0]]))
_CELL_B = lattice_params_to_matrix(torch.tensor([[9.0, 3.0, 4.0, 60.0, 120.0, 70.0],
                                                 [3.5, 3.5, 11.0, 90.0, 90.0, 120.0]]))


def _go(s, cell, fixed):
    torch.manual_seed(1)
    pxrd = torch.rand(2, 4251)
    types = torch.tensor([[14, 8, 8, 0], [26, 26, 8, 8]])
    return s.sample(pxrd, types, cell, types > 0, fixed_lattice=fixed)


def test_learned_mode_never_reads_lattice_init():
    s = _sampler()
    c_a, l_a = _go(s, _CELL_A, False)
    c_b, l_b = _go(s, _CELL_B, False)
    c_n, l_n = _go(s, torch.full_like(_CELL_A, float("nan")), False)
    assert torch.equal(c_a, c_b) and torch.equal(l_a, l_b)
    assert torch.equal(c_a, c_n) and torch.equal(l_a, l_n)
    assert torch.isfinite(c_a).all() and (c_a >= 0).all() and (c_a < 1).all()


def test_fixed_mode_conditions_on_and_returns_the_given_cell():
    s = _sampler()
    c_a, l_a = _go(s, _CELL_A, True)
    c_b, l_b = _go(s, _CELL_B, True)
    assert not torch.allclose(c_a, c_b)
    torch.testing.assert_close(l_a, lattice_matrix_to_params(_CELL_A))
    torch.testing.assert_close(l_b, lattice_matrix_to_params(_CELL_B))


def test_ensemble_shapes_and_padding():
    s = _sampler()
    torch.manual_seed(2)
    types = torch.tensor([[14, 8, 8, 0], [26, 26, 8, 8]])
    c, lp = s.sample_ensemble(torch.rand(2, 4251), types, _CELL_A, types > 0, n_samples=3,
                              eta=0.0, wyckoff=None, guide_scale=0.0)
    assert c.shape == (2, 3, 4, 3) and lp.shape == (2, 3, 6)
    assert torch.all(c[0, :, 3] == 0)
