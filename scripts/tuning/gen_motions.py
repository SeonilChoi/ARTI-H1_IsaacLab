"""Generate reference clips for a motion_generator robot config with distinct random commands (no duplicate clips).
    python gen_motions.py <robot> <num> <jobs>   (run with the env_placo python)"""
import os, subprocess, sys, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
robot, num, jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root = os.environ.get("MOTION_GENERATOR", "C:/motion_generator")  # motion_generator checkout
b = json.load(open(f"{root}/config/{robot}/motion.json"))
# Commands are rounded to 0.01, so the distinct set is a grid; generate every grid point once (no duplicates).
def grid(bd):
    hi = int(round(bd[1] * 100))
    return [v / 100.0 for v in range(-hi, hi + 1)]
cmds = [(dx, dy, dth) for dx in grid(b["dx_bounds"]) for dy in grid(b["dy_bounds"]) for dth in grid(b["dth_bounds"])]
if num > len(cmds):
    # more clips than the 0.01 grid: add distinct points of the 0.005 grid (fixed seed, reproducible)
    def fine(bd):
        hi = int(round(bd[1] * 200))
        return [round(v / 200.0, 3) for v in range(-hi, hi + 1)]
    coarse = set(cmds)
    extra = [c for c in ((dx, dy, dth) for dx in fine(b["dx_bounds"]) for dy in fine(b["dy_bounds"])
                         for dth in fine(b["dth_bounds"])) if c not in coarse]
    rng = np.random.default_rng(0)
    cmds += [extra[i] for i in rng.choice(len(extra), num - len(cmds), replace=False)]
cmds = cmds[:num]
seen = set(cmds)
script = f"{root}/src/motion_generator/_gg_{robot}.py"
def run(i):
    dx, dy, dth = cmds[i]
    out = f"{root}/data/{robot}/{i}.json"
    if os.path.isfile(out):
        return
    subprocess.run([sys.executable, script, "--index", str(i), "--robot", robot, "--dx", str(dx), "--dy", str(dy),
                    "--dth", str(dth), "--stand", "False", "--visual", "False"], cwd=f"{root}/scripts",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with ThreadPoolExecutor(max_workers=jobs) as ex:
    list(ex.map(run, range(num)))
print("done", len(os.listdir(f"{root}/data/{robot}")), "clips", len(seen), "distinct commands")
