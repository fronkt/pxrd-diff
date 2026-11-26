"""Phase 19 amendment A3: micro-batched training in scripts/14_train_csp.py.

  - the accumulated micro-batch gradient equals the full-batch gradient of the pre-A3 loss
  - a CUDA out-of-memory error halves the micro-batch and the redone step matches a direct call
  - validation does not depend on its chunk size
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.model.aux_head import AuxLatHead                  # noqa: E402
from pxrd_diff.model.csp_denoiser import CSPDenoiser             # noqa: E402
from pxrd_diff.model.pxrd_encoder import PXRDEncoder             # noqa: E402
from pxrd_diff.torus import TorusVE                              # noqa: E402

_spec = importlib.util.spec_from_file_location("train_csp", ROOT / "scripts" / "14_train_csp.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

B, N, BINS = 7, 5, 200
LAT_MEAN = torch.tensor([5.0, 6.0, 7.0, 90.0, 90.0, 90.0])
LAT_STD = torch.tensor([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
WEIGHTS = (1.0, 0.7, 1.3)


def _nets():
    torch.manual_seed(0)
    enc = PXRDEncoder(n_bins=BINS, d_model=32)
    den = CSPDenoiser(d_model=32, n_layers=2, n_heads=2, n_freq=4)
    for layer in den.layers:                    # non-trivial FiLM, as in test_csp_model
        torch.nn.init.normal_(layer.film.weight, std=0.1)
    return enc, den, AuxLatHead(32)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    n_atoms = torch.tensor([1, 5, 3, 2, 4, 5, 1])
    types = torch.randint(1, 80, (B, N), generator=g) * (torch.arange(N) < n_atoms[:, None])
    f0 = torch.rand(B, N, 3, generator=g)
    return {"pxrd": torch.rand(B, BINS, generator=g), "f0": f0, "types": types, "mask": types > 0,
            "l0": torch.randn(B, 6, generator=g), "t": torch.rand(B, generator=g),
            "f_t": (f0 + 0.1 * torch.randn(B, N, 3, generator=g)) % 1.0,
            "target": torch.randn(B, N, 3, generator=g), "eps_l": torch.randn(B, 6, generator=g),
            "l_in": torch.randn(B, 6, generator=g),
            "given": torch.tensor([True, False, False, True, False, True, False]),
            "drop": torch.tensor([False, False, True, False, False, False, True])}


def _grads(nets):
    return torch.cat([p.grad.flatten() for net in nets for p in net.parameters() if p.grad is not None])


def _pre_a3_step(nets, x):
    """The pre-A3 training step: one forward on the whole batch."""
    enc, den, aux = nets
    for net in nets:
        net.zero_grad(set_to_none=True)
    g, feats = enc(x["pxrd"])
    pred, eps_l_pred = den(x["f_t"], x["types"], x["l_in"] * LAT_STD + LAT_MEAN, x["l_in"], x["t"],
                           g, feats, x["mask"], x["given"], x["drop"])
    lat_aux = aux(g)
    loss_c = tc.coord_loss(pred, x["target"], x["mask"])
    learned = (~x["given"]).float()
    loss_l = (((eps_l_pred - x["eps_l"]) ** 2).mean(-1) * learned).sum() / learned.sum().clamp(min=1)
    loss_a = ((lat_aux - x["l0"]) ** 2).mean()
    (WEIGHTS[0] * loss_c + WEIGHTS[1] * loss_l + WEIGHTS[2] * loss_a).backward()
    return torch.stack([loss_c, loss_l, loss_a]).detach(), _grads(nets)


@pytest.mark.parametrize("mb", [7, 3, 1])
def test_micro_batches_give_the_full_batch_gradient(mb):
    nets, x = _nets(), _batch()
    ref_loss, ref_grad = _pre_a3_step(nets, x)
    for net in nets:
        net.zero_grad(set_to_none=True)
    loss = tc.backward_in_chunks(nets, x, mb, WEIGHTS, LAT_MEAN, LAT_STD, False)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(_grads(nets), ref_grad, rtol=1e-4, atol=1e-6)


def test_oom_halves_the_micro_batch_and_redoes_the_step(monkeypatch):
    nets, x = _nets(), _batch()
    _, ref_grad = _pre_a3_step(nets, x)
    den = nets[1]
    real_forward = den.forward
    calls = []

    def forward(frac_t, *a, **k):
        calls.append(frac_t.shape[0])
        if frac_t.shape[0] > 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return real_forward(frac_t, *a, **k)

    monkeypatch.setattr(den, "forward", forward)
    loss, mb = tc.grads_with_oom_fallback(nets, x, 7, WEIGHTS, LAT_MEAN, LAT_STD, False)
    # the full batch is tried once; then micro-batches of 3, 3, 1 (the denoiser re-calls itself on
    # a micro-batch whose trailing atom columns are all padding)
    assert mb == 3 and calls[0] == 7 and max(calls[1:]) == 3 and sum(c == 3 for c in calls) == 2
    torch.testing.assert_close(_grads(nets), ref_grad, rtol=1e-4, atol=1e-6)


def test_validation_does_not_depend_on_its_chunk_size():
    enc, den, _ = _nets()
    x = _batch()
    val = {"pxrd_pattern": x["pxrd"], "frac_coords": x["f0"], "atom_types": x["types"],
           "mask": x["mask"], "lattice_params": x["l0"] * LAT_STD + LAT_MEAN}
    torus = TorusVE(0.005, 0.5)
    whole = tc.validate((enc, den), torus, val, LAT_MEAN, LAT_STD, torch.device("cpu"))
    for chunk in (3, 1):
        assert tc.validate((enc, den), torus, val, LAT_MEAN, LAT_STD, torch.device("cpu"), chunk) == whole
