"""Run a single agent in --real mode and plot the distance between the
candidate goal (the rotated projection of the GPS goal under the current
heading estimate) and the true GPS goal over time.

Usage:
    python3 scripts/plot_candidate_distance.py [--seed N] [--steps N]
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


def make_args(seed: int, goal_xy_m=None) -> SimpleNamespace:
    args = SimpleNamespace(
        seed=seed,
        obstacles=12, roofs=3, projectors=4,
        jammers=0, foliage=0, spoofers=0,
        random=False, crazy=False, real=True,
        scatter=False, agents=1, single=True,
        heading_deg=None, goal_lat=None, goal_lon=None,
        headless=True, headless_steps=0,
    )
    args.random = True
    args.obstacles = 6
    args.roofs = 1
    args.projectors = 1
    args.jammers = 0
    args.foliage = 3
    args.spoofers = 0
    if goal_xy_m is not None:
        gx, gy = goal_xy_m
        args.goal_lat, args.goal_lon = G.meters_to_latlon(gx, gy)
    G._apply_real_overrides()
    return args


def set_max_speed_mph(mph: float):
    """Monkey-patch the module-level speed cap. step() reads
    MAX_SPEED_MPS as a global, so reassigning here propagates."""
    G.MAX_SPEED_MPH = mph
    G.MAX_SPEED_MPS = mph * 0.44704
    G.MAX_THRUST = G.MAX_SPEED_MPS * G.LINEAR_DAMPING


def run(seed: int, max_steps: int, goal_xy_m=None):
    args = make_args(seed, goal_xy_m=goal_xy_m)
    scenario = G.build_scenario(args)
    sim = G.build_agents(args, scenario, 1)[0]

    times = []
    cand_dist = []          # live candidate vs true goal
    pub_dist = []           # NAV2-published candidate vs true goal
    true_dist = []          # true robot pos vs true goal
    heading_err_deg = []    # current heading-offset error
    bootstrap_flag = []     # 1 while EKF still bootstrapping

    gx, gy = sim.goal_world
    for _ in range(max_steps):
        live = sim.intermediate_goal_world()
        pub = sim.published_goal_world
        times.append(sim.sim_time)
        cand_dist.append(math.hypot(live[0] - gx, live[1] - gy))
        pub_dist.append(math.hypot(pub[0] - gx, pub[1] - gy))
        true_dist.append(
            math.hypot(sim.true_pos[0] - gx, sim.true_pos[1] - gy))
        err = ((sim.true_heading - sim.heading_offset_est + math.pi)
               % (2 * math.pi)) - math.pi
        heading_err_deg.append(math.degrees(err))
        bootstrap_flag.append(
            1 if (sim.ekf is None or getattr(sim.ekf, "bootstrapping", False))
            else 0)
        if not sim.step():
            # capture final point
            live = sim.intermediate_goal_world()
            pub = sim.published_goal_world
            times.append(sim.sim_time)
            cand_dist.append(math.hypot(live[0] - gx, live[1] - gy))
            pub_dist.append(math.hypot(pub[0] - gx, pub[1] - gy))
            true_dist.append(
                math.hypot(sim.true_pos[0] - gx, sim.true_pos[1] - gy))
            err = ((sim.true_heading - sim.heading_offset_est + math.pi)
                   % (2 * math.pi)) - math.pi
            heading_err_deg.append(math.degrees(err))
            bootstrap_flag.append(0)
            break

    return {
        "t": np.asarray(times),
        "cand_dist": np.asarray(cand_dist),
        "pub_dist": np.asarray(pub_dist),
        "true_dist": np.asarray(true_dist),
        "heading_err_deg": np.asarray(heading_err_deg),
        "bootstrap": np.asarray(bootstrap_flag),
        "arrived": sim.arrived,
        "final_dist_to_goal": math.hypot(
            sim.true_pos[0] - gx, sim.true_pos[1] - gy),
        "true_heading_deg": math.degrees(sim.true_heading),
        "goal_world": sim.goal_world,
    }


def plot(data, out_path: str, seed: int):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    t = data["t"]
    ax1.plot(t, data["cand_dist"], lw=1.6, color="#1f77b4",
             label="‖candidate goal − true goal‖ (live, every tick)")
    ax1.plot(t, data["pub_dist"], lw=1.0, color="#ff7f0e", alpha=0.75,
             label="‖published goal − true goal‖ (NAV2, 1 Hz sampled)")
    ax1.plot(t, data["true_dist"], lw=1.0, color="#2ca02c", alpha=0.6,
             ls="--", label="‖robot − true goal‖ (for context)")

    # Zoomed inset on the post-resync window — the "second hump" lives
    # here and gets visually crushed by the 78 m bootstrap plateau on
    # the linear top-level axis.
    cd = data["cand_dist"]
    if len(cd) > 5:
        # Zoom window: start AFTER the bootstrap-→resync cliff so the
        # 78→0 m drop doesn't dominate the y-axis. Anchor at the first
        # tick where the candidate is already below 3 m, and walk out
        # ~8 s to capture any post-resync hump.
        # Both the live and the rate-limited published goal need to be
        # below the threshold — the NAV2 publisher lags by up to 1 s.
        both_below = np.where((cd < 3.0) & (data["pub_dist"] < 3.0))[0]
        if both_below.size:
            i0 = both_below[0]
            t0 = t[i0]
            t1 = min(t[-1], t0 + 8.0)
            mask = (t >= t0) & (t <= t1)
            if mask.any():
                axin = inset_axes(ax1, width="42%", height="42%",
                                  loc="center right",
                                  bbox_to_anchor=(0.0, -0.05, 1.0, 1.0),
                                  bbox_transform=ax1.transAxes,
                                  borderpad=2)
                axin.plot(t[mask], cd[mask], lw=1.6, color="#1f77b4")
                axin.plot(t[mask], data["pub_dist"][mask], lw=1.0,
                          color="#ff7f0e", alpha=0.75)
                axin.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
                axin.set_xlim(t0, t1)
                ymax = max(cd[mask].max(), data["pub_dist"][mask].max())
                axin.set_ylim(0, max(0.6, ymax * 1.20))
                axin.set_title("zoom: post-resync convergence",
                               fontsize=9)
                axin.grid(True, alpha=0.3)
                axin.tick_params(labelsize=8)

    # Shade the EKF bootstrap window if present.
    boot = data["bootstrap"]
    if boot.any():
        # Find first index where bootstrap turns off.
        idx = np.argmax(boot == 0) if (boot == 0).any() else len(boot)
        if idx > 0:
            ax1.axvspan(t[0], t[idx - 1], color="grey", alpha=0.10,
                        label="EKF bootstrap")

    ax1.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.5,
                label="1 m goal radius")
    ax1.set_ylabel("distance to true goal [m]")
    ax1.set_title(
        f"Candidate-goal convergence  (--real --single  seed={seed}  "
        f"max_v={data.get('max_speed_mph', 5.0):.1f} mph)\n"
        f"true heading = {data['true_heading_deg']:+.1f}°  ·  "
        f"arrived = {data['arrived']}  ·  "
        f"final true→goal = {data['final_dist_to_goal']:.2f} m")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, data["heading_err_deg"], lw=1.0, color="#9467bd")
    ax2.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax2.set_ylabel("heading-offset\nerror [deg]")
    ax2.set_xlabel("sim time [s]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")

    # Summary statistics — useful for confirming the "second hump" claim.
    cd = data["cand_dist"]
    if len(cd) > 5:
        # Find peaks: a sample is a local max if greater than its
        # neighbours within a small window. Cheap and dependency-free.
        peaks = []
        win = max(3, len(cd) // 80)
        for i in range(win, len(cd) - win):
            seg = cd[i - win:i + win + 1]
            if cd[i] == seg.max() and cd[i] > 0.5:
                peaks.append((t[i], cd[i]))
        # Deduplicate close peaks.
        dedup = []
        for tp, vp in peaks:
            if not dedup or tp - dedup[-1][0] > 2.0:
                dedup.append((tp, vp))
        print(f"local maxima (>0.5 m, dedup): {len(dedup)}")
        for tp, vp in dedup[:8]:
            print(f"  t={tp:6.2f}s   dist={vp:.2f} m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8000,
                   help="max physics ticks (0.1 s each)")
    p.add_argument("--mph", type=float, default=1.0,
                   help="terminal speed cap in mph (default 1.0 — "
                        "much slower than the stock 5 mph so the "
                        "candidate-goal transient plays out over "
                        "more sim time)")
    p.add_argument("--goal-x", type=float, default=60.0,
                   help="goal east-coordinate in metres from map "
                        "centre (default 60 m east — agent starts at "
                        "centre, goal lives on the side; map is "
                        "152 m, half-width 76.2 m).")
    p.add_argument("--goal-y", type=float, default=0.0,
                   help="goal north-coordinate in metres from map "
                        "centre (default 0).")
    p.add_argument("--out", default=os.path.join(
        REPO_ROOT, "candidate_goal_distance.png"))
    a = p.parse_args()
    set_max_speed_mph(a.mph)
    data = run(a.seed, a.steps, goal_xy_m=(a.goal_x, a.goal_y))
    data["max_speed_mph"] = a.mph
    plot(data, a.out, a.seed)


if __name__ == "__main__":
    main()
