"""Phase 9.0 — classical auto-indexing viability benchmark (GO/NO-GO gate).

Measures whether a classical Q-space auto-indexer can recover the unit cell
from a simulated PXRD pattern's peak positions alone — the council-mandated
gate before committing to Phase 9.

Method: Ito / de-Wolff style indexing. Peak-find -> 2theta -> d -> Q=1/d^2.
Generate candidate cells by hypothesising low-index (hkl) for the lowest-Q
peaks and solving the linear Q-form, score candidates by indexed-peak count
(vectorised reciprocal-metric scorer), refine the best with least-squares.
Stratified by crystal system, since indexing is known to degrade from cubic
(easy) to triclinic (hard). Hypothesis count is hard-capped so low-symmetry
systems stay bounded.

No retraining, no GPU. Pure offline measurement on the MP-20 test split.

Usage:
  python scripts/09_index_benchmark.py [--n N] [--out paper/phase9_results/index_benchmark.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 9.0.7: GSAS-II fallback for the systems where the native Q-space indexer
# under-performs. Monoclinic (4 free params) and triclinic (6 free params) are
# combinatorically capped in the native path; trigonal regresses at scale
# because the hex-setting Q-form mis-handles rhombohedral cells (GSAS-II
# Bravais code 3 = R hexagonal handles them explicitly). Imported lazily
# inside the dispatcher so the native-only path stays usable without GSAS-II.
GSAS_SYSTEMS = {"monoclinic", "triclinic", "trigonal"}

LAMBDA = 1.54184  # CuKalpha weighted average (Angstrom), matches pymatgen XRDCalculator
SYSTEMS = ("cubic", "tetragonal", "hexagonal", "trigonal",
           "orthorhombic", "monoclinic", "triclinic")

# fixed hkl grid for the vectorised scorer (large enough for big cells / high 2theta)
_HKL = np.array([(h, k, l)
                 for h in range(-6, 7) for k in range(-6, 7) for l in range(-6, 7)
                 if not (h == 0 and k == 0 and l == 0)], dtype=float)


# --------------------------------------------------------------------------
# peak extraction
# --------------------------------------------------------------------------
def extract_peaks(pattern, two_theta, height=0.03, max_peaks=15):
    """Lowest-2theta peak positions (deg) — classical indexing uses low-angle lines."""
    idx, _ = find_peaks(pattern, height=height, distance=3)
    if idx.size == 0:
        return np.empty(0)
    pos = np.sort(two_theta[idx])
    return pos[:max_peaks]


def two_theta_to_Q(two_theta_deg):
    """2theta (deg) -> Q = 1/d^2 (Angstrom^-2) via Bragg's law."""
    theta = np.radians(two_theta_deg) / 2.0
    d = LAMBDA / (2.0 * np.sin(theta))
    return 1.0 / d**2


# --------------------------------------------------------------------------
# Q-form: coefficient vector of the free reciprocal parameters per system
# --------------------------------------------------------------------------
def coeff_vector(hkl, system):
    h, k, l = hkl
    if system == "cubic":
        return np.array([h * h + k * k + l * l], float)
    if system == "tetragonal":
        return np.array([h * h + k * k, l * l], float)
    if system in ("hexagonal", "trigonal"):
        return np.array([(4.0 / 3.0) * (h * h + h * k + k * k), l * l], float)
    if system == "orthorhombic":
        return np.array([h * h, k * k, l * l], float)
    if system == "monoclinic":  # b-unique: Q = A h^2 + B k^2 + C l^2 + D h l
        return np.array([h * h, k * k, l * l, h * l], float)
    # triclinic: Q = A h^2 + B k^2 + C l^2 + D(2kl) + E(2hl) + F(2hk)
    return np.array([h * h, k * k, l * l, 2 * k * l, 2 * h * l, 2 * h * k], float)


def n_free(system):
    return {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2,
            "orthorhombic": 3, "monoclinic": 4, "triclinic": 6}[system]


def params_to_lattice(p, system):
    """Recovered reciprocal params -> pymatgen Lattice (None if unphysical)."""
    try:
        if system == "cubic":
            return Lattice.cubic(1.0 / np.sqrt(p[0]))
        if system == "tetragonal":
            return Lattice.tetragonal(1.0 / np.sqrt(p[0]), 1.0 / np.sqrt(p[1]))
        if system in ("hexagonal", "trigonal"):
            return Lattice.hexagonal(1.0 / np.sqrt(p[0]), 1.0 / np.sqrt(p[1]))
        if system == "orthorhombic":
            a, b, c = (1.0 / np.sqrt(x) for x in p[:3])
            return Lattice.orthorhombic(a, b, c)
        if system == "monoclinic":
            A, B, C, D = p
            cosb = float(np.clip(-D / (2.0 * np.sqrt(A * C)), -0.97, 0.97))
            beta = np.degrees(np.arccos(cosb))
            sinb = np.sin(np.radians(beta))
            a = 1.0 / (np.sqrt(A) * sinb)
            c = 1.0 / (np.sqrt(C) * sinb)
            b = 1.0 / np.sqrt(B)
            return Lattice.monoclinic(a, b, c, beta)
        # triclinic: build reciprocal metric tensor G*, invert -> direct metric
        A, B, C, D, E, F = p
        Gstar = np.array([[A, F, E], [F, B, D], [E, D, C]], float)
        G = np.linalg.inv(Gstar)
        a, b, c = (np.sqrt(G[i, i]) for i in range(3))
        al = np.degrees(np.arccos(np.clip(G[1, 2] / (b * c), -1, 1)))
        be = np.degrees(np.arccos(np.clip(G[0, 2] / (a * c), -1, 1)))
        ga = np.degrees(np.arccos(np.clip(G[0, 1] / (a * b), -1, 1)))
        return Lattice.from_parameters(a, b, c, al, be, ga)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def reflection_set(system):
    """Reflections up to |hkl|<=2 (the lowest observed peak is often a medium-index
    line such as (200) once systematic absences are accounted for). Sorted by index
    so the seed search explores low-index hypotheses first; deduped by Q-coefficient."""
    cand = [(h, k, l) for h, k, l in product(range(-2, 3), repeat=3)
            if not (h == k == l == 0) and h * h + k * k + l * l <= 12]
    cand.sort(key=lambda r: r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    seen, uniq = set(), []
    for r in cand:
        c = coeff_vector(r, system)
        key = tuple(np.round(c, 6))
        if key in seen or all(v == 0 for v in c):
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


# --------------------------------------------------------------------------
# vectorised scoring
# --------------------------------------------------------------------------
def cell_Q(lat):
    """All Q=1/d^2 for the fixed hkl grid, via the reciprocal metric tensor."""
    Gs = lat.reciprocal_lattice_crystallographic.metric_tensor
    return np.einsum("ni,ij,nj->n", _HKL, Gs, _HKL)


MIN_FRAC = 0.80   # a valid index must explain ~all observed peaks


def score_cell(lat, Q_obs, rel_tol=0.015):
    """de-Wolff style scoring. A cell must first explain (nearly) ALL observed
    peaks — sub-cells that index only a fraction are disqualified. Among cells
    that pass, the M20 figure of merit rewards parsimony, killing over-large
    super-cells (which index everything via a trivially dense Q-grid).
    Returns (fom, indexed_fraction); fom=0 means the cell failed the gate."""
    if lat is None:
        return 0.0, 0.0
    a, b, c = lat.abc
    if not (2.0 < a < 25 and 2.0 < b < 25 and 2.0 < c < 25):
        return 0.0, 0.0
    Qp = np.unique(np.round(cell_Q(lat), 5))
    Qp = Qp[Qp > 1e-6]
    if Qp.size == 0:
        return 0.0, 0.0
    idx = np.clip(np.searchsorted(Qp, Q_obs), 1, len(Qp) - 1)
    dq = np.minimum(np.abs(Q_obs - Qp[idx - 1]), np.abs(Q_obs - Qp[idx]))
    matched = (dq / Q_obs) < rel_tol
    if not matched.any():
        return 0.0, 0.0
    frac = float(matched.mean())
    if frac < MIN_FRAC:                       # hard coverage gate — kills sub-cells
        return 0.0, frac
    Qmax = Q_obs[-1]
    n_poss = max(int(np.sum(Qp <= Qmax * 1.001)), 1)          # calc lines in range
    mean_err = max(float(np.mean(dq[matched])), 1e-5)
    m20 = Qmax / (2.0 * mean_err * n_poss)                     # de-Wolff M (parsimony)
    return m20, frac


def index_pattern(Q_obs, system, max_hyp=150000, topk=1):
    """Ito-style: hypothesise (hkl) for the n_free lowest peaks, solve the linear
    Q-form, score, refine the best. Hypothesis count hard-capped.

    Returns:
        topk=1 (default): (lat, frac)  — back-compatible with previous behaviour.
        topk>1: list of (lat, frac, fom) sorted by fom desc, len <= topk.
                Falls back to a coverage-best candidate if none clear the gate.
    """
    nf = n_free(system)
    refs = reflection_set(system)
    seed_n = min(len(Q_obs), nf + 2)
    # collect candidates as (lat, frac, fom); dedupe by rounded params at the end
    cands = []
    fb, fb_frac = None, 0.0                          # fallback: best coverage if none pass
    count = 0
    for picks in combinations(range(seed_n), nf):
        qs = Q_obs[list(picks)]
        for hyp in product(refs, repeat=nf):
            count += 1
            if count > max_hyp:
                break
            M = np.array([coeff_vector(r, system) for r in hyp], float)
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            try:
                p = np.linalg.solve(M, qs)
            except np.linalg.LinAlgError:
                continue
            if system != "triclinic" and np.any(p[:3] <= 1e-6):
                continue
            lat = params_to_lattice(p, system)
            fom, frac = score_cell(lat, Q_obs)
            if fom > 0.0:
                cands.append((lat, frac, fom))
            elif lat is not None and frac > fb_frac:
                fb, fb_frac = lat, frac
        if count > max_hyp:
            break

    # dedupe by rounded conventional params (catches numerical near-duplicates)
    seen = {}
    for lat, frac, fom in cands:
        key = tuple(round(x, 2) for x in lat.parameters)
        prev = seen.get(key)
        if prev is None or fom > prev[2]:
            seen[key] = (lat, frac, fom)
    uniq = sorted(seen.values(), key=lambda t: -t[2])[:topk]

    if not uniq:
        if topk == 1:
            if fb is None:
                return None, 0.0
            return refine_cell(fb, Q_obs, system), fb_frac
        if fb is None:
            return []
        return [(refine_cell(fb, Q_obs, system), fb_frac, 0.0)]

    refined = [(refine_cell(lat, Q_obs, system), frac, fom)
               for (lat, frac, fom) in uniq]
    if topk == 1:
        lat, frac, _ = refined[0]
        return lat, frac
    return refined


def dispatch_index_pattern(peaks_2theta, Q_obs, system, topk=1, use_gsas=True):
    """9.0.7 dispatcher: route monoclinic/triclinic through GSAS-II, everything
    else through the native Q-space indexer. Output schema matches the native
    `index_pattern` exactly so the caller does not need to know which path ran.

    For GSAS-II cells the `frac` (indexed fraction) is computed post-hoc with
    `score_cell` against `Q_obs`, so the downstream `consistent` check at
    line ~406 still works.
    """
    if use_gsas and system in GSAS_SYSTEMS:
        from pxrd_diff.indexer_gsas import index_pattern_gsas
        try:
            gsas_cells = index_pattern_gsas(
                peaks_two_theta=peaks_2theta,
                intensities=None,
                system=system,
                topk=topk,
            )
        except ImportError:
            gsas_cells = []
        # Score each GSAS-II cell against Q_obs for the frac metric, refine
        # with the same LSQ as the native path so the comparison is apples-
        # to-apples on coordinate precision.
        scored = []
        for (lat, m20) in gsas_cells:
            fom_native, frac = score_cell(lat, Q_obs)
            lat_ref = refine_cell(lat, Q_obs, system)
            # Re-score after refinement (LSQ may have moved frac slightly).
            _, frac_ref = score_cell(lat_ref, Q_obs)
            scored.append((lat_ref, frac_ref, m20))
        scored.sort(key=lambda t: -t[2])
        if topk == 1:
            if not scored:
                return None, 0.0
            lat, frac, _ = scored[0]
            return lat, frac
        return scored[:topk]

    # Native path — unchanged behaviour.
    return index_pattern(Q_obs, system, topk=topk)


def refine_cell(lat, Q_obs, system):
    """Least-squares refine the cell against the observed Q values."""
    p0 = np.array(lat.parameters)
    free = {"cubic": [0], "tetragonal": [0, 2], "hexagonal": [0, 2],
            "trigonal": [0, 2], "orthorhombic": [0, 1, 2],
            "monoclinic": [0, 1, 2, 4], "triclinic": [0, 1, 2, 3, 4, 5]}[system]

    def build(x):
        p = p0.copy()
        for i, fi in enumerate(free):
            p[fi] = x[i]
        if system == "cubic":
            p[1] = p[2] = p[0]
        elif system in ("tetragonal", "hexagonal", "trigonal"):
            p[1] = p[0]
        return p

    def resid(x):
        try:
            lt = Lattice.from_parameters(*build(x))
            Qp = np.unique(np.round(cell_Q(lt), 6))
        except Exception:
            return np.ones(len(Q_obs)) * 1e3
        Qp = Qp[Qp > 1e-6]
        if Qp.size == 0:
            return np.ones(len(Q_obs)) * 1e3
        diff = np.abs(Qp[None, :] - Q_obs[:, None]).min(axis=1)
        return np.minimum(diff, 0.05)

    try:
        sol = least_squares(resid, p0[free], method="lm", max_nfev=200)
        return Lattice.from_parameters(*build(sol.x))
    except Exception:
        return lat


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def sorted_params(lat):
    """Sorted lengths + sorted angles — axis-label-invariant comparison."""
    return np.array(sorted(lat.abc) + sorted(lat.angles))


MIN_VPA = 9.0   # Angstrom^3 per atom — inorganic crystals are never denser


def volume_correct(lat, n_atoms):
    """Resolve sub-cell aliasing. Peak positions alone can't tell the conventional
    cell from a denser sub-cell (extinct reflections look identical to absent
    ones). The known composition breaks the tie: a cell cannot pack atoms denser
    than ~9 A^3/atom, so take the smallest integer super-cell that clears that
    floor. Uses only the atom count — a legitimate pipeline input, not the answer."""
    if n_atoms <= 0:
        return lat
    V = lat.volume
    for k in (1, 2, 3, 4, 6, 8):
        if k * V / n_atoms >= MIN_VPA:
            if k == 1:
                return lat
            s = k ** (1.0 / 3.0)
            a, b, c, al, be, ga = lat.parameters
            return Lattice.from_parameters(a * s, b * s, c * s, al, be, ga)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "paper" / "phase9_results" / "index_benchmark.json"))
    ap.add_argument("--cache", type=str, default=str(ROOT / "data" / "cache" / "test.npz"))
    ap.add_argument("--grid", type=str, default=str(ROOT / "data" / "cache" / "two_theta.npy"))
    ap.add_argument("--csv", type=str, default=str(ROOT / "data" / "raw" / "test.csv"))
    ap.add_argument("--topk", type=int, default=1,
                    help=">1 emits ranked candidates per structure in 'candidates' field "
                         "(9.1.1; consumed by --lat-from-index-topk in 03_sample.py)")
    ap.add_argument("--use-gsas", action="store_true",
                    help="9.0.7 (experimental): route monoclinic/triclinic/trigonal "
                         "patterns through GSAS-II's DoIndexPeaks. Adapter is wired "
                         "and works on some inputs, but GSAS-II's inner search loop "
                         "(findBestCell) hangs on certain real test patterns and the "
                         "documented `timeout` parameter only fires between Bravais "
                         "iterations. Off by default; native indexer is the shipped "
                         "path. See src/pxrd_diff/indexer_gsas.py for status.")
    args = ap.parse_args()

    import warnings

    import pandas as pd
    warnings.filterwarnings("ignore")

    two_theta = np.load(args.grid)
    cache = np.load(args.cache, allow_pickle=True)
    patterns = cache["pattern"]
    mat_ids = list(cache["material_id"])
    df = pd.read_csv(args.csv)
    cif_by_id = dict(zip(df["material_id"], df["cif"]))

    n = min(args.n, len(patterns))
    print(f"[bench] indexing {n} test structures (LAMBDA={LAMBDA} Cu-Ka)", flush=True)
    t0 = time.time()

    per_system = defaultdict(list)
    rows = []
    for i in range(n):
        mid = mat_ids[i]
        cif = cif_by_id.get(mid)
        if cif is None:
            continue
        try:
            struct = Structure.from_str(cif, fmt="cif")
            sga = SpacegroupAnalyzer(struct, symprec=0.01)
            system = sga.get_crystal_system()
            # the indexer targets the conventional cell — compare against it
            conv = sga.get_conventional_standard_structure()
            true_lat = conv.lattice
            n_conv_atoms = len(conv)
        except Exception:
            continue
        peaks = extract_peaks(patterns[i], two_theta)
        if peaks.size < n_free(system) + 1:
            per_system[system].append(None)
            continue
        Q_obs = np.sort(two_theta_to_Q(peaks))
        use_gsas = args.use_gsas
        if args.topk > 1:
            cands = dispatch_index_pattern(peaks, Q_obs, system,
                                           topk=args.topk, use_gsas=use_gsas)
            if not cands:
                per_system[system].append(None)
                continue
            cand_records = []
            for (lat, c_frac, c_fom) in cands:
                lat_vc = volume_correct(lat, n_conv_atoms)
                cand_records.append(dict(
                    params=[round(float(x), 4) for x in lat_vc.parameters],
                    fom=round(float(c_fom), 4),
                    frac=round(float(c_frac), 3),
                    vol_ratio=round(float(true_lat.volume / lat_vc.volume), 3),
                ))
            pred_lat = volume_correct(cands[0][0], n_conv_atoms)
            frac = cands[0][1]
        else:
            pred_lat, frac = dispatch_index_pattern(peaks, Q_obs, system,
                                                   topk=1, use_gsas=use_gsas)
            if pred_lat is None:
                per_system[system].append(None)
                continue
            pred_lat = volume_correct(pred_lat, n_conv_atoms)   # 9.0.6 sub-cell fix
            cand_records = None
        tp, pp = sorted_params(true_lat), sorted_params(pred_lat)
        vr = true_lat.volume / pred_lat.volume          # >1 if pred is a sub-cell
        ratio = vr if vr >= 1 else 1.0 / vr
        # "consistent": explains the pattern and is the conventional cell OR a
        # small-index sub/super-cell of it (the unavoidable peak-position ambiguity)
        near_int = min(abs(ratio - k) for k in (1, 2, 3, 4, 6, 8))
        rec = dict(mid=mid, system=system, indexed_frac=round(frac, 3),
                   pred_params=[round(float(x), 4) for x in pred_lat.parameters],
                   len_mae=round(float(np.mean(np.abs(tp[:3] - pp[:3]))), 4),
                   ang_mae=round(float(np.mean(np.abs(tp[3:] - pp[3:]))), 3),
                   vol_err=round(float(abs(pred_lat.volume - true_lat.volume)
                                       / true_lat.volume), 4),
                   vol_ratio=round(float(vr), 3),
                   consistent=bool(frac >= 0.9 and near_int < 0.08))
        if cand_records is not None:
            rec["candidates"] = cand_records
        rows.append(rec)
        per_system[system].append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate, stratified by crystal system ----
    print("\n=== Phase 9.0 indexing benchmark — stratified by crystal system ===")
    print("strict = recovered the conventional cell;  consistent = explains the")
    print("pattern via the conventional cell OR a small-index sub/super-cell.\n")
    hdr = (f"{'system':<14}{'n':>5}{'strict%':>9}{'consist%':>10}"
           f"{'len_MAE':>9}{'ang_MAE':>9}{'expl_frac':>10}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    success = lambda r: r["vol_err"] < 0.05 and r["len_mae"] < 0.3
    for system in SYSTEMS:
        recs = per_system.get(system, [])
        if not recs:
            continue
        ok = [r for r in recs if r is not None]
        n_sys = len(recs)
        strict = 100.0 * sum(success(r) for r in ok) / n_sys
        consist = 100.0 * sum(r["consistent"] for r in ok) / n_sys
        len_mae = float(np.mean([r["len_mae"] for r in ok])) if ok else float("nan")
        ang_mae = float(np.mean([r["ang_mae"] for r in ok])) if ok else float("nan")
        expl = float(np.mean([r["indexed_frac"] for r in ok])) if ok else float("nan")
        print(f"{system:<14}{n_sys:>5}{strict:>8.1f}%{consist:>9.1f}%"
              f"{len_mae:>9.3f}{ang_mae:>9.2f}{expl:>10.2f}")
        summary[system] = dict(n=n_sys, strict_pct=round(strict, 1),
                               consistent_pct=round(consist, 1),
                               len_mae=round(len_mae, 4), ang_mae=round(ang_mae, 3),
                               mean_explained_frac=round(expl, 3))

    solved = [r for r in rows if success(r)]
    consistent = [r for r in rows if r["consistent"]]
    overall = dict(
        n=n, n_indexed=len(rows),
        overall_strict_pct=round(100.0 * len(solved) / n, 1),
        overall_consistent_pct=round(100.0 * len(consistent) / n, 1),
        overall_len_mae=round(float(np.mean([r["len_mae"] for r in rows])), 4) if rows else None,
        v20_learned_head_len_mae=1.37)
    print("-" * len(hdr))
    print(f"OVERALL: indexed {overall['n_indexed']}/{n}, "
          f"strict {overall['overall_strict_pct']}%, "
          f"consistent {overall['overall_consistent_pct']}%, "
          f"len MAE {overall['overall_len_mae']} A (v20 learned head ~1.37 A)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(overall=overall, per_system=summary, rows=rows), indent=2))
    print(f"\n[bench] wrote {out}  ({time.time()-t0:.0f}s total)")
    print("\nGO/NO-GO: classical indexing is viable if solved% materially beats the "
          "learned head (len MAE << 1.37 A) on at least the higher-symmetry systems.")


if __name__ == "__main__":
    main()
