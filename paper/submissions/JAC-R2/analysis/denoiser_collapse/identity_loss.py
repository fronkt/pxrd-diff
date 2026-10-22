"""Loss of the identity coordinate prediction (x0_pred = x_t, zero residual) under
the training objective, averaged over the training timestep distribution."""
import sys, torch
sys.path.insert(0, "C:/Users/frank/pxrd-diff-pub/src")
from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.diffusion import DiffusionProcess
ds = CrystalPXRDDataset("C:/Users/frank/pxrd-diff-pub/data", split="test")
items = [ds[i] for i in range(512)]
coords = torch.stack([b["frac_coords"] for b in items]); mask = torch.stack([b["mask"] for b in items])
dp = DiffusionProcess()
torch.manual_seed(0)
vals = []
for _ in range(20):
    t = dp.sample_timesteps(coords.shape[0], "cpu")
    nc, _ = dp.forward_q(coords, t, wrap=True)
    vals.append(float(dp.loss(nc, coords, mask, periodic=True)))
vals_mean = sum(vals) / len(vals)
# and a uniform-random prediction for reference
ur = float(dp.loss(torch.rand_like(coords), coords, mask, periodic=True))
open(sys.argv[1], "w").write(f"identity prediction (x0_pred = x_t) loss: {vals_mean:.4f} (20 draws, n=512)\nuniform random prediction loss: {ur:.4f}\n")
