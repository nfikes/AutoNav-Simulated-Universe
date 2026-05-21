"""Diagnose the failure modes of agents that don't arrive.
Runs a --crazy ensemble headless, then reports the final-distance
distribution of the misses, plus a few representative bad agents'
state (heading error, recovery counts, EKF σ_θ).

Usage:
  python3 scripts/diagnose_misses.py [--seed N] [--agents N] [--steps N]
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import gps_sim_gui as G


def make_args(seed: int, n_agents: int, mode: str,
              scatter: bool = False) -> SimpleNamespace:
    args = SimpleNamespace(
        seed=seed,
        obstacles=12, roofs=3, projectors=4,
        jammers=0, foliage=0, spoofers=0,
        random=False, crazy=False, real=False,
        scatter=scatter, agents=n_agents, single=False,
        heading_deg=None, goal_lat=None, goal_lon=None,
        headless=True, headless_steps=0,
    )
    if mode == "crazy":
        args.crazy = True
        args.random = True
        args.obstacles = 30
        args.roofs = 8
        args.projectors = 12
        args.jammers = 5
        args.foliage = 8
        args.spoofers = 3
        G._apply_crazy_overrides()
    elif mode == "real":
        args.real = True
        args.random = True
        args.obstacles = 6
        args.roofs = 1
        args.projectors = 1
        args.foliage = 3
        G._apply_real_overrides()
    return args


def run(seed, n_agents, max_steps, mode, scatter=False):
    args = make_args(seed, n_agents, mode, scatter=scatter)
    scenario = G.build_scenario(args)
    agents = G.build_agents(args, scenario, n_agents)

    for step_i in range(max_steps):
        any_run = False
        for s in agents:
            if s.step():
                any_run = True
        if not any_run:
            break

    gx, gy = agents[0].goal_world
    summaries = []
    for i, s in enumerate(agents):
        d = math.hypot(s.true_pos[0] - gx, s.true_pos[1] - gy)
        herr = math.degrees(
            (s.true_heading - s.heading_offset_est + math.pi) % (2 * math.pi) - math.pi)
        summaries.append({
            "i": i,
            "arrived": s.arrived,
            "d": d,
            "herr": herr,
            "ekf_sigma_th": math.degrees(s.ekf.theta_std) if s.ekf else float('nan'),
            "moving_away": s._moving_away_event_count,
            "stuck": s._stuck_event_count,
            "rejects": s._cand_reject_count,
            "bootstrap_done": s.bootstrap_done,
            "has_path": s.path_world is not None and len(s.path_world) >= 2,
        })
    return summaries, args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--agents", type=int, default=10000)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--mode", choices=["crazy", "real"], default="crazy")
    p.add_argument("--scatter", action="store_true",
                   help="Spawn each agent at a random valid start "
                        "(matches `gps_sim_gui.py --scatter`).")
    a = p.parse_args()
    suffix = " --scatter" if a.scatter else ""
    print(f"Running --{a.mode}{suffix} agents={a.agents} "
          f"steps={a.steps} seed={a.seed} …")
    summaries, args = run(a.seed, a.agents, a.steps, a.mode,
                          scatter=a.scatter)
    n = len(summaries)
    arrived = sum(1 for s in summaries if s["arrived"])
    misses = [s for s in summaries if not s["arrived"]]
    print(f"\narrived = {arrived}/{n}    misses = {len(misses)}\n")

    # Bucket misses by final distance.
    buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
    print("=== Final true→goal distance distribution of misses ===")
    for lo, hi in buckets:
        cnt = sum(1 for s in misses if lo <= s["d"] < hi)
        if cnt > 0:
            print(f"  {lo:>3}–{hi:<3} m : {cnt}")

    if not misses:
        return

    # Show 10 representative misses.
    misses_sorted = sorted(misses, key=lambda s: s["d"], reverse=True)
    print("\n=== Top 10 farthest misses ===")
    for s in misses_sorted[:10]:
        print(f"  agent {s['i']:>5}: d={s['d']:>6.2f} m  "
              f"|herr|={abs(s['herr']):>6.1f}°  σθ={s['ekf_sigma_th']:>4.1f}°  "
              f"moving_away={s['moving_away']:>2}  stuck={s['stuck']:>2}  "
              f"rejects={s['rejects']:>4}  "
              f"boot={s['bootstrap_done']}  path={s['has_path']}")

    # Stats
    import numpy as np
    miss_d = np.array([s["d"] for s in misses])
    miss_herr = np.array([abs(s["herr"]) for s in misses])
    print("\n=== Aggregate over misses ===")
    print(f"  median d = {np.median(miss_d):.2f} m   max d = {miss_d.max():.2f} m")
    print(f"  median |herr| = {np.median(miss_herr):.1f}°   "
          f"max |herr| = {miss_herr.max():.1f}°")
    print(f"  bootstrap-failed:    "
          f"{sum(1 for s in misses if not s['bootstrap_done'])}/{len(misses)}")
    print(f"  no-path:             "
          f"{sum(1 for s in misses if not s['has_path'])}/{len(misses)}")
    print(f"  moving_away events median = "
          f"{int(np.median([s['moving_away'] for s in misses]))}, "
          f"max = {max(s['moving_away'] for s in misses)}")


if __name__ == "__main__":
    main()
