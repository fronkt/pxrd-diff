"""Download canonical MP-20 splits from the CDVAE repository.

CDVAE (Xie et al. 2021) defines the canonical MP-20 splits used by all subsequent
crystal structure generation papers (FlowMM, DiffCSP, MatterGen, CrystalDiT,
DiffractGPT). Sticking to these splits is what makes our results comparable.

Source: https://github.com/txie-93/cdvae/tree/main/data/mp_20
Splits: train (27,136) / val (9,047) / test (9,046). All structures have <=20 atoms
per unit cell, sourced from Materials Project.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20"
EXPECTED_ROWS = {"train": 27136, "val": 9047, "test": 9046}


def download(name: str) -> Path:
    url = f"{BASE}/{name}.csv"
    out = RAW / f"{name}.csv"
    if out.exists():
        print(f"[skip] {out.name} already present ({out.stat().st_size:,} bytes)")
        return out
    print(f"[get ] {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        out.write_bytes(r.read())
    print(f"[ok  ] wrote {out.name} ({out.stat().st_size:,} bytes)")
    return out


def verify(path: Path, expected_rows: int) -> None:
    df = pd.read_csv(path)
    required = {"material_id", "cif", "spacegroup.number", "pretty_formula", "elements"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path.name}: missing columns {missing}")
    if len(df) != expected_rows:
        raise RuntimeError(f"{path.name}: got {len(df)} rows, expected {expected_rows}")
    sha = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    print(f"[verify] {path.name}: rows={len(df):,} sha1={sha} cols={list(df.columns)}")


def main() -> int:
    for split, n in EXPECTED_ROWS.items():
        path = download(split)
        verify(path, n)
    print("\nAll MP-20 splits downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())