"""R2-1 / R3-3a: re-validate DiffPXRD against pymatgen.XRDCalculator with the
legacy (hat5032 v1) and corrected (pymatgen-convention) form factors.
Same protocol as scripts/04_verify_debye.py: 256-bin coarse grid, hkl_max=5
(the training configuration), Pearson vs linearly-downsampled pymatgen pattern."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.debye import DiffPXRD
from pxrd_diff.eval import r_pearson
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ds = CrystalPXRDDataset(ROOT / "data", split="test", limit=N)
mods = {ff: DiffPXRD(n_bins=256, hkl_max=5, form_factor=ff).eval() for ff in ("legacy", "pymatgen")}
rows = []
t0 = time.time()
for i in range(min(N, len(ds))):
    s = ds[i]
    coords = s["frac_coords"].unsqueeze(0); types = s["atom_types"].unsqueeze(0)
    lat = s["lattice"].unsqueeze(0); mask = s["mask"].unsqueeze(0)
    tgt = torch.nn.functional.interpolate(s["pxrd_pattern"].view(1, 1, -1), size=256,
                                          mode="linear", align_corners=True)[0, 0].numpy()
    r = {"material_id": str(s["material_id"]), "n_atoms": int(mask.sum())}
    with torch.no_grad():
        for ff, m in mods.items():
            r[ff] = r_pearson(m(coords, types, lat, mask)[0].numpy(), tgt)
    rows.append(r)
out = {}
for ff in ("legacy", "pymatgen"):
    v = np.array([r[ff] for r in rows])
    for n in (50, len(rows)):
        w = v[:n]
        out[f"{ff}_n{n}"] = dict(n=int(n), mean=float(w.mean()), std=float(w.std()), min=float(w.min()),
                                 max=float(w.max()), median=float(np.median(w)),
                                 frac_above_0p7=float((w > 0.7).mean()), frac_above_0p9=float((w > 0.9).mean()))
d = np.array([r["pymatgen"] - r["legacy"] for r in rows])
out["delta_pymatgen_minus_legacy"] = dict(mean=float(d.mean()), median=float(np.median(d)),
                                          frac_improved=float((d > 0).mean()), frac_worse=float((d < 0).mean()))
json.dump(dict(summary=out, rows=rows), open(ROOT / "paper/submissions/JAC-R1/analysis/simulator_revalidation.json", "w"), indent=1)
for k, v in out.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print(f"{time.time()-t0:.0f}s")
