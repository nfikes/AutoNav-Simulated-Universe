"""Per-tick motion-smoothness probe. Runs a single agent, records
forward velocity, angular velocity, body heading, true position, and
the candidate-goal projection at every tick, and plots them so any
discontinuities (jerky velocity steps, body-heading jumps, abrupt
trajectory kinks) caused by the moving-away suspension show up as
visible spikes.

Verification target: with the information-only suspension (no
smoother reset, no forced replan), velocity should be C1-smooth at
every detected moving-away event and the agent's path through the
detection should look indistinguishable from the unflagged regions.

Usage:
    python3 scripts/plot_motion_smoothness.py [--mode real|crazy]
                [--seed N] [--steps N]
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import gps_sim_gui as G


def make_args(seed: int, mode: str) -> SimpleNamespace:
    args = SimpleNamespace(
        seed=seed,
        obstacles=12, roofs=3, projectors=4,
        jammers=0, foliage=0, spoofers=0,
        random=False, crazy=False, real=False,
        scatter=False, agents=1, single=True,
        heading_deg=None, goal_lat=None, goal_lon=None,
        headless=True, headless_steps=0,
    )
    if mode == "real":
        args.real = True
        args.random = True
        args.obstacles = 6
        args.roofs = 1
        args.projectors = 1
        args.jammers = 0
        args.foliage = 3
        args.spoofers = 0
        G._apply_real_overrides()
    elif mode == "crazy":
        args.crazy = True
        args.random = True
        args.obstacles = 30
        args.roofs = 8
        args.projectors = 12
        args.jammers = 5
        args.foliage = 8
        args.spoofers = 3
        G._apply_crazy_overrides()
    return args


def run(seed: int, mode: str, max_steps: int):
    args = make_args(seed, mode)
    scenario = G.build_scenario(args)
    sim = G.build_agents(args, scenario, 1)[0]

    rec = {
        "t": [], "fv": [], "av": [], "bh": [],
        "tx": [], "ty": [], "cx": [], "cy": [],
        "d_goal": [],
        "moving_away_count": [],
        "stuck_count": [],
        "env_suspended": [],   # 1 if suspension active this tick
    }

    def snap(s):
        rec["t"].append(s.sim_time)
        rec["fv"].append(s.forward_vel)
        rec["av"].append(s.angular_vel)
        rec["bh"].append(s.body_heading)
        rec["tx"].append(s.true_pos[0])
        rec["ty"].append(s.true_pos[1])
        c = s.intermediate_goal_world()
        rec["cx"].append(c[0])
        rec["cy"].append(c[1])
        gx, gy = s.goal_world
        rec["d_goal"].append(
            math.hypot(s.true_pos[0] - gx, s.true_pos[1] - gy))
        rec["moving_away_count"].append(s._moving_away_event_count)
        rec["stuck_count"].append(s._stuck_event_count)
        rec["env_suspended"].append(
            1 if s.sim_time < s._envelope_suspended_until else 0)

    snap(sim)
    for _ in range(max_steps):
        if not sim.step():
            snap(sim)
            break
        snap(sim)

    return rec, sim


def plot(rec, sim, out_path: str, seed: int, mode: str):
    t = np.asarray(rec["t"])
    fv = np.asarray(rec["fv"])
    av = np.asarray(rec["av"])
    bh = np.asarray(rec["bh"])
    tx = np.asarray(rec["tx"])
    ty = np.asarray(rec["ty"])
    cx = np.asarray(rec["cx"])
    cy = np.asarray(rec["cy"])
    susp = np.asarray(rec["env_suspended"])
    ma = np.asarray(rec["moving_away_count"])
    sk = np.asarray(rec["stuck_count"])

    # First-difference of forward velocity = numerical jerk-proxy.
    # Sharp spikes here indicate non-smooth controller transitions.
    dt = np.maximum(np.diff(t, prepend=t[0]), 1e-6)
    fv_rate = np.gradient(fv, t)
    av_rate = np.gradient(av, t)

    # Detect rising edges in moving-away count → event timestamps.
    ma_events_t = t[1:][np.diff(ma) > 0]
    sk_events_t = t[1:][np.diff(sk) > 0]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 2, 1]})
    ax_v, ax_a, ax_h, ax_e = axes

    # Shade envelope-suspension intervals (red transparent).
    in_suspend = susp.astype(bool)
    if in_suspend.any():
        # find runs
        edges = np.diff(np.concatenate(
            [[False], in_suspend, [False]]).astype(int))
        starts = t[np.where(edges == 1)[0].clip(max=len(t) - 1)]
        ends = t[np.where(edges == -1)[0].clip(max=len(t) - 1)]
        for s_, e_ in zip(starts, ends):
            for ax in (ax_v, ax_a, ax_h):
                ax.axvspan(s_, e_, color="red", alpha=0.10,
                           label=("envelope suspended"
                                  if s_ == starts[0] else None))

    ax_v.plot(t, fv, lw=1.0, color="#1f77b4", label="forward velocity [m/s]")
    ax_v.plot(t, fv_rate, lw=0.7, color="#aaaaaa", alpha=0.7,
              label="dv/dt (acceleration) [m/s²]")
    for et in ma_events_t:
        ax_v.axvline(et, color="red", lw=0.7, ls="--", alpha=0.8)
    for et in sk_events_t:
        ax_v.axvline(et, color="orange", lw=0.7, ls=":", alpha=0.7)
    ax_v.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax_v.set_ylabel("v [m/s]")
    ax_v.legend(loc="upper right", fontsize=8)
    ax_v.grid(True, alpha=0.3)

    ax_a.plot(t, av, lw=1.0, color="#9467bd",
              label="angular velocity [rad/s]")
    ax_a.plot(t, av_rate, lw=0.7, color="#aaaaaa", alpha=0.7,
              label="dω/dt [rad/s²]")
    ax_a.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax_a.set_ylabel("ω [rad/s]")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(True, alpha=0.3)

    bh_deg = np.rad2deg(np.unwrap(bh))
    ax_h.plot(t, bh_deg, lw=1.0, color="#2ca02c", label="body heading [deg]")
    ax_h.set_ylabel("body heading [deg]")
    ax_h.legend(loc="upper right", fontsize=8)
    ax_h.grid(True, alpha=0.3)

    ax_e.plot(t, ma, lw=1.0, color="red",
              label=f"moving-away events (final={int(ma[-1])})")
    ax_e.plot(t, sk, lw=1.0, color="orange",
              label=f"stuck events (final={int(sk[-1])})")
    ax_e.set_xlabel("sim time [s]")
    ax_e.set_ylabel("count")
    ax_e.legend(loc="upper left", fontsize=8)
    ax_e.grid(True, alpha=0.3)

    arrived = sim.arrived
    final_d = math.hypot(sim.true_pos[0] - sim.goal_world[0],
                          sim.true_pos[1] - sim.goal_world[1])
    fig.suptitle(
        f"Motion smoothness — --{mode} --single seed={seed}\n"
        f"arrived={arrived}  final true→goal={final_d:.2f} m  "
        f"moving-away events={int(ma[-1])}  stuck events={int(sk[-1])}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    return ma[-1], sk[-1], arrived, final_d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["real", "crazy"], default="real")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(
        REPO_ROOT, f"motion_{a.mode}_seed{a.seed}.png")
    rec, sim = run(a.seed, a.mode, a.steps)
    plot(rec, sim, out, a.seed, a.mode)


if __name__ == "__main__":
    main()
