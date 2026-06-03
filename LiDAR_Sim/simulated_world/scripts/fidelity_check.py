#!/usr/bin/env python3
"""Equivalence check for build_grade_costmap optimizations.

Runs do_scan + build_grade_costmap at a fixed set of (px, py, heading)
positions on a fixed RNG seed and prints SHA-256 hashes for:
    * obs grid (bool)
    * ms slope grid (float; NaN normalized)
    * observed grid (bool)
    * obstacle count

The hashes must match between pre- and post-optimization runs.
"""
import sys
import hashlib
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

import lidar_sim_gui as sim  # noqa: E402
import trimesh  # noqa: E402


def _hash_array(arr):
    if arr.dtype.kind == "f":
        # Normalize NaN bit pattern for deterministic hashing
        a = arr.copy()
        a[np.isnan(a)] = np.float64(-9999.0)
        return hashlib.sha256(a.tobytes()).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def main():
    print("Loading mesh...")
    mesh = trimesh.load(sim.STL_PATH)
    rays = sim.gen_rays(11520)

    # Fixed test positions spanning the platform — ramps, flat, walls
    positions = [
        (-2.0, -1.0, 0.0),
        (-0.5, -1.5, np.pi / 4),
        (0.75, -1.5, np.pi / 2),
        (2.25, -1.5, np.pi / 3),
        (3.75, -2.0, 0.5),
        (0.0, 1.5, np.pi),
        (1.0, 0.0, -np.pi / 6),
    ]

    print(f"\n  {'pos':<24} {'obs_hash':<18} {'ms_hash':<18} {'seen_hash':<18} {'#obs'}")
    print(f"  {'-' * 24} {'-' * 18} {'-' * 18} {'-' * 18} {'-' * 6}")
    total_obs = 0
    for (px, py, heading) in positions:
        cloud, origin, sn = sim.do_scan(mesh, px, py, rays, heading)
        if cloud is None:
            print(f"  ({px:+.2f},{py:+.2f}) h={heading:+.2f}  NO HIT")
            continue
        obs, ms, R, seen = sim.build_grade_costmap(cloud, origin, sn)
        h_obs = _hash_array(obs)
        h_ms = _hash_array(ms)
        h_seen = _hash_array(seen)
        n_obs = int(obs.sum())
        total_obs += n_obs
        print(f"  ({px:+.2f},{py:+.2f}) h={heading:+.2f}  "
              f"{h_obs:<18} {h_ms:<18} {h_seen:<18} {n_obs}")
    print(f"\n  Total obs across positions: {total_obs}")


if __name__ == "__main__":
    main()
