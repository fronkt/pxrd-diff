"""Simulate and cache PXRD patterns for every MP-20 structure.

Output: data/cache/{split}.npz with arrays:
    material_id: (N,) string
    pattern:     (N, B) float32   -- intensities on the fixed 2theta grid
    n_atoms:     (N,) int16
    spacegroup:  (N,) int16
    formula:     (N,) string
The 2theta grid itself is saved once at data/cache/two_theta.npy.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxrd_diff.simulator import PXRDConfig, PXRDSimulator  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

_GLOBAL_SIM: PXRDSimulator | None = None


def _init_worker(cfg_dict: dict) -> None:
    global _GLOBAL_SIM
    _GLOBAL_SIM = PXRDSimulator(PXRDConfig(**cfg_dict))


def _simulate_one(args: tuple[str, str]) -> tuple[str, np.ndarray, int]:
    mid, cif = args
    try:
        s = CifParser.from_str(cif).parse_structures(primitive=False)[0]
        pattern = _GLOBAL_SIM.simulate(s)
        return mid, pattern, len(s)
    except Exception as e:
        # Return zeros + n_atoms=-1 to mark failure; we will filter later.
        return mid, np.zeros(_GLOBAL_SIM.cfg.n_bins, dtype=np.float32), -1


def run_split(split: str, cfg: PXRDConfig, n_workers: int, limit: int | None) -> None:
    raw = ROOT / "data" / "raw" / f"{split}.csv"
    out = ROOT / "data" / "cache" / f"{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw)
    if limit is not None:
        df = df.iloc[:limit]
    print(f"[{split}] {len(df):,} structures, n_workers={n_workers}, n_bins={cfg.n_bins}")

    t0 = time.perf_counter()
    args = list(zip(df["material_id"].tolist(), df["cif"].tolist()))
    cfg_dict = cfg.__dict__.copy()

    if n_workers <= 1:
        _init_worker(cfg_dict)
        results = [_simulate_one(a) for a in tqdm(args, desc=split)]
    else:
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(cfg_dict,)) as pool:
            results = list(tqdm(pool.imap(_simulate_one, args, chunksize=32),
                                total=len(args), desc=split))

    mids = np.array([r[0] for r in results])
    patterns = np.stack([r[1] for r in results]).astype(np.float32)
    n_atoms = np.array([r[2] for r in results], dtype=np.int16)
    sgs = df["spacegroup.number"].astype(np.int16).to_numpy()
    formulas = df["pretty_formula"].astype(str).to_numpy()

    fail = int((n_atoms == -1).sum())
    print(f"[{split}] simulated in {time.perf_counter()-t0:.1f}s; fail={fail}")

    np.savez_compressed(
        out,
        material_id=mids,
        pattern=patterns,
        n_atoms=n_atoms,
        spacegroup=sgs,
        formula=formulas,
    )
    print(f"[{split}] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=None, help="Subsample for smoke testing")
    args = ap.parse_args()

    cfg = PXRDConfig()
    np.save(ROOT / "data" / "cache" / "two_theta.npy", cfg.two_theta_grid)

    for split in args.splits:
        run_split(split, cfg, args.workers, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())