"""Why does the cross-attention erase the atom state? Magnitudes and attention entropy, v22 layer 0."""
import sys, math, torch
sys.path.insert(0, "C:/Users/frank/pxrd-diff-pub/src")
from pxrd_diff.model.denoiser import CrystalDenoiser, periodic_distances, rbf_expansion
from pxrd_diff.model.pxrd_encoder import PXRDEncoder
from pxrd_diff.data import CrystalPXRDDataset
from pxrd_diff.diffusion import DiffusionProcess
L = []
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc = PXRDEncoder(d_model=256).eval(); den = CrystalDenoiser().eval()
enc.load_state_dict(ck["encoder"]); den.load_state_dict(ck["denoiser"])
ds = CrystalPXRDDataset("C:/Users/frank/pxrd-diff-pub/data", split="test")
items = [ds[i] for i in range(32)]
st = lambda k: torch.stack([b[k] for b in items])
pxrd, types, coords, lat, mask = st("pxrd_pattern"), st("atom_types"), st("frac_coords"), st("lattice"), st("mask")
t = torch.full((32,), 0.3)
nc, _ = DiffusionProcess().forward_q(coords, t, wrap=True)
with torch.no_grad():
    g, feats = enc(pxrd)
    L.append(f"PXRD feature map: shape {tuple(feats.shape)}, per-token norm mean {feats.norm(dim=-1).mean():.1f}, max {feats.norm(dim=-1).max():.1f}")
    r = rbf_expansion(periodic_distances(nc, lat, mask))
    h = den.atom_emb(types) + den.coord_proj(nc)
    layer = den.layers[0]
    ew = layer.edge_net(r)
    agg = (ew * h.unsqueeze(1) * mask.unsqueeze(2).unsqueeze(-1).float()).sum(dim=2)
    out = layer.node_net(torch.cat([h, agg], dim=-1))
    sc, sh = layer.film(den.time_emb(t)).unsqueeze(1).chunk(2, dim=-1)
    h1 = layer.norm(h + out * (1 + sc) + sh)
    ca = layer.cross_attn
    attn_out, w = ca.attn(h1, feats, feats, need_weights=True, average_attn_weights=True)
    ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)
    m = mask.float()
    L.append(f"atom state into cross-attn: per-atom norm {h1.norm(dim=-1)[mask].mean():.1f}")
    L.append(f"cross-attn output: per-atom norm {attn_out.norm(dim=-1)[mask].mean():.1f}  (ratio to atom state {attn_out.norm(dim=-1)[mask].mean() / h1.norm(dim=-1)[mask].mean():.1f}x)")
    L.append(f"attention entropy per query: mean {ent[mask].mean():.3f} nats (uniform over {feats.shape[1]} keys = {math.log(feats.shape[1]):.2f}); max weight mean {w.max(-1).values[mask].mean():.3f}")
    # do different atoms in one structure attend to the same key?
    am = w.argmax(-1)
    same = sum(int((am[b, :int(mask[b].sum())] == am[b, 0]).all()) for b in range(32))
    L.append(f"structures whose atoms all pick the same top key: {same}/32")
    after = ca.norm(h1 + attn_out * mask.unsqueeze(-1).float())
    hm = (after * m.unsqueeze(-1)).sum(1, keepdim=True) / m.sum(1, keepdim=True).unsqueeze(-1)
    L.append(f"after residual+LayerNorm: within-structure spread / norm = {(((after - hm) * m.unsqueeze(-1)).norm() / (after * m.unsqueeze(-1)).norm()):.2e}")
open(sys.argv[2], "w").write("\n".join(L) + "\n")
