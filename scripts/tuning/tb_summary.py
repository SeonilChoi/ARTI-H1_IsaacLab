import sys, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run = sys.argv[1]
ea = EventAccumulator(run, size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]
want = sys.argv[2:] or tags
for t in tags:
    if not any(w in t for w in want): continue
    ev = ea.Scalars(t)
    if not ev: continue
    n = len(ev); idx = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    print(f"{t:60s} n={n:4d} " + "  ".join(f"{ev[i].step}:{ev[i].value:.4g}" for i in idx))
