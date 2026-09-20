"""Consistency checks added for the J. Appl. Cryst. HAT5032 revision (referee 2
point 1, referee 3 point 3a): the differentiable simulator must evaluate the
atomic scattering factor with the same convention as pymatgen.XRDCalculator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.debye import DiffPXRD  # noqa: E402
from pxrd_diff.eval import r_pearson  # noqa: E402
from pxrd_diff.simulator import PXRDSimulator  # noqa: E402


def _ff(module, z, s):
    types = torch.tensor([[z]])
    s_vals = torch.tensor([[s]], dtype=torch.float32)
    # bypass the Debye-Waller factor by setting b_iso = 0 for the test
    module.b_iso = 0.0
    return float(module._form_factor_at_s(types, s_vals)[0, 0, 0])


@pytest.mark.parametrize("z", [8, 14, 26, 55])
def test_form_factor_equals_z_at_zero_angle(z):
    m = DiffPXRD(n_bins=64, hkl_max=2, form_factor="pymatgen")
    assert abs(_ff(m, z, 0.0) - z) < 1e-4


@pytest.mark.parametrize("z,s", [(14, 0.2), (26, 0.4), (55, 0.6)])
def test_form_factor_matches_pymatgen_expression(z, s):
    from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
    from pymatgen.core.periodic_table import Element
    c = np.array(ATOMIC_SCATTERING_PARAMS[Element.from_Z(z).symbol])
    ref = z - 41.78214 * s * s * float((c[:, 0] * np.exp(-c[:, 1] * s * s)).sum())
    m = DiffPXRD(n_bins=64, hkl_max=2, form_factor="pymatgen")
    assert abs(_ff(m, z, s) - ref) < 1e-3


def test_legacy_convention_is_not_z_at_zero_angle():
    # documents the submitted-version defect so it cannot silently return
    m = DiffPXRD(n_bins=64, hkl_max=2, form_factor="legacy")
    assert _ff(m, 14, 0.0) < 7.0


def test_pattern_agrees_with_xrdcalculator_on_nacl():
    from pymatgen.core import Lattice, Structure
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"],
                                  [[0, 0, 0], [0.5, 0.5, 0.5]])
    target = PXRDSimulator().simulate(s)
    fc = torch.tensor(s.frac_coords, dtype=torch.float32).unsqueeze(0)
    zt = torch.tensor([sp.Z for sp in s.species]).unsqueeze(0)
    lat = torch.tensor(s.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(s), dtype=torch.bool)
    t = torch.nn.functional.interpolate(torch.tensor(target).view(1, 1, -1), size=256,
                                        mode="linear", align_corners=True)[0, 0].numpy()
    p = DiffPXRD(n_bins=256, hkl_max=5, form_factor="pymatgen")(fc, zt, lat, mask)[0].detach().numpy()
    assert r_pearson(p, t) > 0.99


def test_invalid_convention_rejected():
    with pytest.raises(ValueError):
        DiffPXRD(form_factor="cromer-mann")
