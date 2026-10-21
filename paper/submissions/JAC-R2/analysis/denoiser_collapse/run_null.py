"""Run scripts/03_sample.py with the denoiser's coordinate output replaced by zeros
(x0-residual mode: x0 estimate = x_t). argv[1] = 'null' or 'real'; rest go to 03_sample.py."""
import runpy, sys, torch
sys.path.insert(0, "C:/Users/frank/pxrd-diff-pub/src")
import pxrd_diff.model.denoiser as dn
mode = sys.argv[1]
if mode in ("null", "randshift"):
    _orig = dn.CrystalDenoiser.forward
    def forward(self, *a, **k):
        out = _orig(self, *a, **k)
        if mode == "null":
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        # same form as the trained output: one vector per structure, shared by all
        # atoms, same per-atom norm, but random (fresh each call, own RNG stream)
        c = out[0]
        g = torch.Generator().manual_seed(int(forward.calls) + 12345); forward.calls += 1
        v = torch.randn(c.shape[0], 1, 3, generator=g)
        v = v / v.norm(dim=-1, keepdim=True) * c.norm(dim=-1).mean()
        return (v.expand_as(c).clone(),) + tuple(out[1:])
    forward.calls = 0
    dn.CrystalDenoiser.forward = forward
sys.argv = ["03_sample.py"] + sys.argv[2:]
runpy.run_path("C:/Users/frank/pxrd-diff-pub/scripts/03_sample.py", run_name="__main__")
