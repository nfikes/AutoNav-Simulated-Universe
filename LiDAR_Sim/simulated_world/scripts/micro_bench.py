#!/usr/bin/env python3
"""Micro-benchmark for build_grade_costmap at full ray fidelity.

Runs do_scan + build_grade_costmap N times at a fixed set of agent
positions / headings and reports mean / min wall time per scan. Used
to compare pre / post optimization without doing a full benchmark.
"""
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

import lidar_sim_gui as sim  # noqa: E402
import trimesh  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=11520)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading mesh, generating {args.rays} rays...")
    mesh = trimesh.load(sim.STL_PATH)
    rays = sim.gen_rays(args.rays)

    positions = [
        (-2.0, -1.0, 0.0),
        (-0.5, -1.5, np.pi / 4),
        (0.75, -1.5, np.pi / 2),
        (2.25, -1.5, np.pi / 3),
        (3.75, -2.0, 0.5),
        (0.0, 1.5, np.pi),
        (1.0, 0.0, -np.pi / 6),
    ]

    # Warmup
    sim.do_scan(mesh, 0, 0, rays[:64], 0.0)

    scan_times = []
    pipeline_times = []
    for it in range(args.iters):
        for (px, py, heading) in positions:
            t0 = time.perf_counter()
            cloud, origin, sn = sim.do_scan(mesh, px, py, rays, heading)
            t1 = time.perf_counter()
            if cloud is None:
                continue
            obs, ms, R, seen = sim.build_grade_costmap(cloud, origin, sn)
            t2 = time.perf_counter()
            scan_times.append(t1 - t0)
            pipeline_times.append(t2 - t1)

    n = len(scan_times)
    if n == 0:
        print("no samples")
        return
    print(f"\n  Samples: {n}")
    print(f"  do_scan         mean={np.mean(scan_times)*1000:7.2f} ms  "
          f"min={np.min(scan_times)*1000:7.2f} ms  "
          f"median={np.median(scan_times)*1000:7.2f} ms")
    print(f"  pca pipeline    mean={np.mean(pipeline_times)*1000:7.2f} ms  "
          f"min={np.min(pipeline_times)*1000:7.2f} ms  "
          f"median={np.median(pipeline_times)*1000:7.2f} ms")
    print(f"  total/scan      mean={(np.mean(scan_times)+np.mean(pipeline_times))*1000:7.2f} ms")


if __name__ == "__main__":
    main()
