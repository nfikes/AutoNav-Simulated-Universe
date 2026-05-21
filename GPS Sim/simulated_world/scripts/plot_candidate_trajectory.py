"""Trace the candidate goal's XY position over time alongside the
agent's true trajectory. Renders a map view with:
  - Static scenery (obstacles / projectors / roofs / goal)
  - Agent's true path (white-grey trail)
  - Candidate goal's path, colour-coded by time (turbo colormap)
  - Real goal as a green star

Lets you see exactly how the candidate goal moves: does it converge
straight onto the real goal, oscillate, jump after a heading-resync,
park at a wrong spot, etc.

Usage:
    python3 scripts/plot_candidate_trajectory.py [--mode crazy|real]
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
from matplotlib.patches import Circle, Rectangle, Polygon
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
        args.foliage = 3
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


def run(seed, mode, max_steps):
    args = make_args(seed, mode)
    scenario = G.build_scenario(args)
    sim = G.build_agents(args, scenario, 1)[0]

    rec = {"t": [], "tx": [], "ty": [], "cx": [], "cy": [],
           "moving_away": [], "env_suspended": []}

    def snap():
        c = sim.intermediate_goal_world()
        rec["t"].append(sim.sim_time)
        rec["tx"].append(sim.true_pos[0])
        rec["ty"].append(sim.true_pos[1])
        rec["cx"].append(c[0])
        rec["cy"].append(c[1])
        rec["moving_away"].append(sim._moving_away_event_count)
        rec["env_suspended"].append(
            1 if sim.sim_time < sim._envelope_suspended_until else 0)

    snap()
    for _ in range(max_steps):
        if not sim.step():
            snap()
            break
        snap()

    return rec, sim, scenario


def plot(rec, sim, scenario, out_path, seed, mode):
    (cm, start, goal, true_heading, obstacles, roofs, projectors,
     jammers, foliage, spoofers) = scenario

    t = np.asarray(rec["t"])
    tx = np.asarray(rec["tx"])
    ty = np.asarray(rec["ty"])
    cx = np.asarray(rec["cx"])
    cy = np.asarray(rec["cy"])
    ma = np.asarray(rec["moving_away"])

    fig, (ax_map, ax_t) = plt.subplots(
        1, 2, figsize=(15, 7),
        gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ── MAP PANEL ────────────────────────────────────────────────────
    ax_map.set_xlim(-G.MAP_HALF, G.MAP_HALF)
    ax_map.set_ylim(-G.MAP_HALF, G.MAP_HALF)
    ax_map.set_aspect("equal")
    ax_map.set_facecolor("#0d1f12")
    ax_map.grid(True, color="#163020", linestyle=":", linewidth=0.4)
    ax_map.set_title(f"--{mode} --single seed={seed}   "
                     f"map view (candidate trajectory time-coded)",
                     fontsize=10)

    for cx_o, cy_o, r in obstacles:
        ax_map.add_patch(Circle((cx_o, cy_o), r,
                                facecolor="#2a2a2a",
                                edgecolor="#555", linewidth=0.4, zorder=2))
    for x_min, y_min, x_max, y_max in roofs:
        ax_map.add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor=(0.45, 0.55, 0.95, 0.13),
            edgecolor="#7099dd", linewidth=0.5,
            linestyle="--", zorder=2.4))
    for verts, _bias in projectors:
        ax_map.add_patch(Polygon(list(verts), closed=True,
                                 facecolor="#3a2f1f",
                                 edgecolor="#c8a360",
                                 linewidth=0.5, zorder=2.5))
    for cx_f, cy_f, r in foliage:
        ax_map.add_patch(Circle(
            (cx_f, cy_f), r,
            facecolor=(0.35, 0.75, 0.35, 0.14),
            edgecolor="#5fbb5f", linewidth=0.4,
            linestyle=":", zorder=2.3))
    for cx_j, cy_j, r in jammers:
        verts = G.hex_vertices(cx_j, cy_j, r)
        ax_map.add_patch(Polygon(verts, closed=True,
                                 facecolor=(0.85, 0.15, 0.45, 0.10),
                                 edgecolor="#ff4080", linewidth=0.5,
                                 linestyle="--", zorder=2.35))
    for (sx_, sy_), _ in spoofers:
        ax_map.add_patch(Circle(
            (sx_, sy_), G.SPOOFER_INFLUENCE_RADIUS_M, fill=False,
            edgecolor="#cc33ff", linestyle=":", linewidth=0.5,
            alpha=0.6, zorder=2.45))

    # Real goal & success ring
    ax_map.add_patch(Circle(goal, G.GOAL_RADIUS,
                            facecolor=(0.2, 0.9, 0.3, 0.18),
                            edgecolor="#33ff66", linewidth=0.8, zorder=3))
    ax_map.plot([goal[0]], [goal[1]], "*", color="#33ff66",
                markersize=18, markeredgecolor="white",
                markeredgewidth=0.8, zorder=11, label="real goal")

    # Agent trail
    ax_map.plot(tx, ty, color="#dddddd", lw=0.8, alpha=0.7,
                zorder=4, label="agent path")
    ax_map.plot(tx[0], ty[0], "o", color="white", markersize=6,
                zorder=12, label="agent start")
    ax_map.plot(tx[-1], ty[-1], "s", color="white", markersize=6,
                zorder=12, label="agent end")

    # Candidate trajectory — colour by time (turbo).
    cmap = matplotlib.colormaps["turbo"]
    n = len(cx)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=t[-1] if n else 1)
    seg_colors = [cmap(norm(ti)) for ti in t]
    ax_map.scatter(cx, cy, s=4, c=seg_colors, edgecolors="none",
                   alpha=0.85, zorder=10)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_map, fraction=0.04, pad=0.02)
    cb.set_label("sim time [s]", fontsize=9)

    ax_map.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax_map.tick_params(labelsize=7, colors="#444")

    # ── TIME-SERIES PANEL ───────────────────────────────────────────
    ax_t.plot(t, cx - goal[0], lw=1.0, color="#1f77b4",
              label="candidate.x − goal.x")
    ax_t.plot(t, cy - goal[1], lw=1.0, color="#ff7f0e",
              label="candidate.y − goal.y")

    susp = np.asarray(rec["env_suspended"]).astype(bool)
    if susp.any():
        edges = np.diff(np.concatenate(
            [[False], susp, [False]]).astype(int))
        starts_idx = np.where(edges == 1)[0]
        ends_idx = np.where(edges == -1)[0]
        for si, ei in zip(starts_idx, ends_idx):
            si = min(si, len(t) - 1)
            ei = min(ei, len(t) - 1)
            ax_t.axvspan(t[si], t[ei], color="red", alpha=0.10,
                         label=("env. suspended"
                                if si == starts_idx[0] else None))

    ma_events_t = t[1:][np.diff(ma) > 0]
    for et in ma_events_t:
        ax_t.axvline(et, color="red", lw=0.6, ls="--", alpha=0.6)

    ax_t.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_t.set_xlabel("sim time [s]")
    ax_t.set_ylabel("candidate − real goal [m]")
    ax_t.set_title("candidate-goal offset from real goal (per axis)",
                   fontsize=10)
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(True, alpha=0.3)

    arrived = sim.arrived
    final_d = math.hypot(sim.true_pos[0] - goal[0],
                         sim.true_pos[1] - goal[1])
    fig.suptitle(
        f"Candidate-goal trajectory — --{mode} --single seed={seed}\n"
        f"arrived={arrived}  final true→goal={final_d:.2f} m  "
        f"moving-away events={int(ma[-1])}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["crazy", "real"], default="real")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(
        REPO_ROOT, f"candidate_traj_{a.mode}_seed{a.seed}.png")
    rec, sim, scenario = run(a.seed, a.mode, a.steps)
    plot(rec, sim, scenario, out, a.seed, a.mode)


if __name__ == "__main__":
    main()
