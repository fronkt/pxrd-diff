import sys, torch
sys.path.insert(0, "C:/Users/frank/pxrd-diff-pub/src")
from pxrd_diff.model.denoiser import CrystalDenoiser, periodic_distances, rbf_expansion
from pxrd_diff.model.pxrd_encoder import PXRDEncoder
from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.diffusion import DiffusionProcess
out_lines = []
ck = torch.load("C:/Users/frank/pxrd-diff-pub/runs/gpu_v22_jac/ckpt_final.pt", map_location="cpu", weights_only=False)
enc = PXRDEncoder(d_model=256).eval(); den = CrystalDenoiser().eval()
enc.load_state_dict(ck["encoder"]); den.load_state_dict(ck["denoiser"])
ds = CrystalPXRDDataset("C:/Users/frank/pxrd-diff-pub/data", split="test")
items = [ds[i] for i in range(16)]
st = lambda k: torch.stack([b[k] for b in items])
pxrd, types, coords, lat, lat_p, mask = (st("pxrd_pattern"), st("atom_types"), st("frac_coords"), st("lattice"), st("lattice_params"), st("mask"))
t = torch.full((16,), 0.3)
nc, _ = DiffusionProcess().forward_q(coords, t, wrap=True)
with torch.no_grad():
    g, feats = enc(pxrd)
    res = {}
    for name, L in [("ref", lat), ("far", torch.eye(3).expand(16, 3, 3) * 1000.0)]:
        d = periodic_distances(nc, L, mask); r = rbf_expansion(d)
        h = den.atom_emb(types) + den.coord_proj(nc)
        tc = den.time_emb(t)
        layer = den.layers[0]
        ew = layer.edge_net(r)
        pm = mask.unsqueeze(2).unsqueeze(-1).float()
        agg = (ew * h.unsqueeze(1) * pm).sum(dim=2)
        out = layer.node_net(torch.cat([h, agg], dim=-1))
        sc, sh = layer.film(tc).unsqueeze(1).chunk(2, dim=-1)
        out2 = out * (1 + sc) + sh
        h1 = layer.norm(h + out2)
        h1x = layer.cross_attn(h1, feats, mask)
        res[name] = (agg, out2, h1, h1x)
        out_lines.append(f"{name}: rbf sum {r.sum():.2f} |agg| {agg.norm():.2f} |node out| {out.norm():.1f} |film out| {out2.norm():.1f} |h0| {h.norm():.2f}")
    for i, nm in enumerate(["agg", "film out", "after LayerNorm", "after cross-attn"]):
        a, b = res["ref"][i], res["far"][i]
        out_lines.append(f"rel diff ref vs far at {nm}: {(a-b).norm()/a.norm():.3e}")
open(sys.argv[1], "w").write("\n".join(out_lines) + "\n")
