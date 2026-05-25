"""GSAS-II auto-indexing adapter for Phase 9.0.7.

The native Q-space indexer in `scripts/09_index_benchmark.py` works well on
high-symmetry systems (cubic, tetragonal, hexagonal, orthorhombic) but is
hypothesis-capped on monoclinic (4 free params) and triclinic (6 free params)
and mis-handles rhombohedral cells in hex setting on trigonal. GSAS-II's
`DoIndexPeaks` implements the full Visser successive-dichotomy search and
handles those systems competently.

This module wraps `GSASII.GSASIIindex.DoIndexPeaks` behind a function whose
output matches the calling convention of `index_pattern` in
09_index_benchmark.py, so the two can be dispatched on `crystal_system`.

GSAS-II Bravais code order (per the GSASIIindex.DoIndexPeaks bravaisNames
list, 18 entries):
    0  Cubic-F           1  Cubic-I            2  Cubic-P
    3  Trigonal-R        4  Trigonal/Hex-P
    5  Tetragonal-I      6  Tetragonal-P
    7  Ortho-F           8  Ortho-I            9  Ortho-A
   10  Ortho-B          11  Ortho-C           12  Ortho-P
   13  Mono-I           14  Mono-A            15  Mono-C           16  Mono-P
   17  Triclinic
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pymatgen.core import Lattice

# Which Bravais codes to enable for each pymatgen crystal system.
# Constraining to one system makes GSAS-II's findBestCell loop hang on some
# real patterns (it has no internal timeout for the inner search; the
# DoIndexPeaks timeout only fires between Bravais iterations). Empirically
# the all-bravais path is both more robust AND faster on the noise-free
# simulated MP-20 patterns: GSAS-II's M20 ranking puts the right-symmetry
# cell on top, and the search exits the early Bravais codes quickly when
# they're a bad fit. So `system` is now an unused hint, retained for
# API symmetry with the native indexer.
ALL_BRAVAIS = list(range(18))
_BRAVAIS_BY_SYSTEM = {s: ALL_BRAVAIS for s in
                      ("cubic", "trigonal", "hexagonal", "tetragonal",
                       "orthorhombic", "monoclinic", "triclinic")}


def _make_peak_row(two_theta_deg: float, d_spacing: float, intensity: float = 1.0):
    """One row of GSAS-II's "Index Peak List" data block.

    This is distinct from the "Peak List" 9-element format used during
    profile fitting. Per the format constructed in GSASIIpwdGUI.OnReload
    (line ~4188), the index peak schema is:
        [pos_2theta, intensity, use_flag, refine_flag, 0, 0, 0, d_spacing, 0]
    DoIndexPeaks reads d via `peaks[i][-2]` (= index 7), so the d-spacing
    field is what's load-bearing for the dichotomy search.
    """
    return [float(two_theta_deg),
            float(max(intensity, 1e-6)),
            True,                          # use this peak for indexing
            False,                         # don't refine position
            0, 0, 0,
            float(d_spacing),
            0.0]


def _default_controls(wavelength: float):
    """Minimal `controls` list that DoIndexPeaks expects.

    Mirrors the default constructed in GSASIIdataGUI.py for a new
    "Unit Cells List" data block:
        [zeroflag, zero, ncno, V1, _, SG_short,
         a, b, c, al, be, ga, Z, SG_full]

    Wavelength is set via `peaks` d-spacings, not controls — controls is
    cell-search bookkeeping. Defaults below match the GUI's "new" settings
    except for V1 (we start higher to be safe).
    """
    # ncno=10 (was GUI default 4) — be more permissive about Nc/Nobs ratio
    # since we know the patterns are noise-free simulations.
    # V1=10 (was GUI default 25) — start small so the dichotomy covers small
    # unit cells; the search expands V automatically.
    return [0, 0.0, 10, 10.0, 0, 'P1',
            1.0, 1.0, 1.0, 90.0, 90.0, 90.0,
            1.0, 'P 1']


def _row_to_lattice(cell_row) -> Lattice | None:
    """GSAS-II cell row -> pymatgen Lattice.

    Cell row schema (from DoIndexPeaks return): typically
    [M20, X20, ibrav, a, b, c, alpha, beta, gamma, Volume, sel, refl_list, ...]
    """
    try:
        a, b, c, al, be, ga = (float(x) for x in cell_row[3:9])
        if not all(2.0 < L < 30.0 for L in (a, b, c)):
            return None
        if not all(20.0 < ang < 160.0 for ang in (al, be, ga)):
            return None
        return Lattice.from_parameters(a, b, c, al, be, ga)
    except (ValueError, IndexError, FloatingPointError):
        return None


def index_pattern_gsas(
    peaks_two_theta: np.ndarray,
    intensities: np.ndarray | None,
    system: str,
    topk: int = 1,
    wavelength: float = 1.54184,
    m20_min: float = 2.0,
    timeout: float | None = 30.0,
) -> List[Tuple[Lattice, float]]:
    """Auto-index a powder pattern with GSAS-II.

    Args:
        peaks_two_theta: peak positions in 2theta degrees, sorted ascending.
        intensities: optional intensities aligned with peaks; uses 1.0 if None.
        system: pymatgen-style crystal system name. Determines which Bravais
            lattices GSAS-II will try.
        topk: number of candidate cells to return (sorted by M20 desc).
        wavelength: X-ray wavelength in Angstrom (default Cu Kalpha weighted).
            GSAS-II derives d from 2theta+wavelength; we set wavelength via
            peak d-spacings rather than controls, since DoIndexPeaks uses
            `getDmin(peaks)` and `getDmax(peaks)` not controls[wavelength].
        m20_min: minimum de-Wolff M20 figure of merit to keep a cell.
        timeout: per-call wall-clock cap (seconds); GSAS-II's triclinic search
            can run minutes in the worst case.

    Returns:
        List of (pymatgen.Lattice, M20_score) tuples, length <= topk.
        Empty list if GSAS-II finds no candidates above m20_min.

    Raises:
        ImportError if GSAS-II is not importable in the active env.
    """
    try:
        from GSASII import GSASIIindex as G2idx
    except ImportError as e:
        raise ImportError(
            "GSAS-II is required for monoclinic/triclinic indexing. "
            "Install via `pip install -e <GSAS-II checkout>` (needs gfortran, "
            "meson, ninja, cython)."
        ) from e

    if peaks_two_theta is None or len(peaks_two_theta) < 6:
        # Need at least 6 peaks for triclinic (6 free) plus a couple for
        # over-determination; monoclinic only needs 4 but we use one bar.
        return []

    if intensities is None:
        intensities = np.ones_like(peaks_two_theta)

    # Convert 2theta -> d via Bragg's law: d = lambda / (2 sin(theta)).
    # DoIndexPeaks reads d from peaks[i][7] (= peaks[i][-2]); it does NOT
    # re-derive d from 2theta + wavelength, so feeding correct d here is
    # what makes the wavelength choice matter.
    theta_rad = np.radians(np.asarray(peaks_two_theta, float)) / 2.0
    d_vals = wavelength / (2.0 * np.sin(theta_rad))

    peaks = [_make_peak_row(tt, d, I)
             for tt, d, I in zip(peaks_two_theta, d_vals, intensities)]

    # 18 Bravais flags; True for ones we want to try.
    bravais_flags = [False] * 18
    for code in _BRAVAIS_BY_SYSTEM.get(system, []):
        bravais_flags[code] = True

    controls = _default_controls(wavelength)
    # Inject the wavelength via the inst-param hook DoIndexPeaks uses
    # (the Index Peak List "use Inst" pathway): actually DoIndexPeaks reads
    # d from peak[7]/peak[8] via getDmin/getDmax — peak[0] is 2theta-pos.
    # We populated those above; GSAS-II will compute d internally.

    # dlg=None: DoIndexPeaks gates all wx.ProgressDialog calls behind
    # `if dlg:` checks. A stub-dialog class would need to implement Raise(),
    # Update() returning (cont, skip), and Pulse() — easier to just pass None.
    try:
        OK, dmin_out, cells_out = G2idx.DoIndexPeaks(
            peaks, controls, bravais_flags, None,
            ifX20=True, timeout=timeout, M20_min=m20_min,
        )
    except Exception:
        # Worst-case GSAS-II throws on degenerate inputs; treat as no cells.
        return []

    if not cells_out:
        return []

    scored = []
    for row in cells_out:
        try:
            m20 = float(row[0])
        except (TypeError, ValueError, IndexError):
            continue
        if m20 < m20_min:
            continue
        lat = _row_to_lattice(row)
        if lat is None:
            continue
        scored.append((lat, m20))

    scored.sort(key=lambda t: -t[1])
    return scored[:topk]
