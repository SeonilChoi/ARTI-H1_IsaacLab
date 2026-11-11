"""Warm-start converter: 45-D velocity policy checkpoint -> 81-D (obs_action_history=True).
Old obs: gyro 3 | gravity 3 | q 12 | qd 12 | a(t) 12 | cmd 3.  New: ... a(t) 12 | a(t-1) 12 | a(t-2) 12 | q_target-default 12 | cmd 3.
New first-layer columns start at zero (same behaviour as the old policy); normalizer stats for the new inputs are copied
from a(t) (actions) and q (target) because the running count is huge and they would barely adapt. Optimizer kept (first-layer moments expanded)."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
c = torch.load(src, map_location="cpu", weights_only=False)
old = list(range(42)) + [78, 79, 80]
def expand(v, fill_src):
    out = torch.zeros(81, dtype=v.dtype)
    out[old] = v
    out[42:54], out[54:66], out[66:78] = v[30:42], v[30:42], v[6:18]
    return out
w = c["policy"]["net_container.0.weight"]
nw = torch.zeros(w.shape[0], 81, dtype=w.dtype); nw[:, old] = w
c["policy"]["net_container.0.weight"] = nw
pre = c["observation_preprocessor"]
pre["running_mean"] = expand(pre["running_mean"], None); pre["running_variance"] = expand(pre["running_variance"], None)
# keep the optimizer (r11 lesson: dropping it reset Adam and the KL-adaptive lr 1.3e-4 -> 1e-3, the policy std went
# 0.08 -> 0.4 and the run collapsed into standing still). Expand the first-layer Adam moments the same way (zeros).
opt = c.get("optimizer")
if opt is not None:
    for st in opt["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            t = st.get(k)
            if t is not None and t.dim() == 2 and t.shape[1] == 45:
                n_ = torch.zeros(t.shape[0], 81, dtype=t.dtype); n_[:, old] = t; st[k] = n_
    print("optimizer kept, lr", [g["lr"] for g in opt["param_groups"]])
torch.save(c, dst); print("saved", dst, tuple(nw.shape))
