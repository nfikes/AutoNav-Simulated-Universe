#!/usr/bin/env python3
"""Headless GIF baker for the GPS waypoint sim.

Renders an agent ensemble running --crazy (or another mode) to a GIF
without needing a display. Every FRAME_EVERY ticks the agents'
ground-truth positions are captured as a single matplotlib frame and
appended to the output GIF.

Usage (defaults to 10 000 agents in --crazy):
  python scripts/bake_gif.py
  python scripts/bake_gif.py --agents 5000 --steps 1000 --out /tmp/x.gif
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import math
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from PIL import Image

# Make `import src.gps_sim_gui` work from the project root no matter
# where this script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.gps_sim_gui as m


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agents", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=1500,
                        help="Sim ticks to advance (each = 0.1 s).")
    parser.add_argument("--frame-every", type=int, default=10,
                        help="Capture a frame every N sim ticks "
                             "(10 = 1 frame per second of sim).")
    parser.add_argument("--fps", type=int, default=20,
                        help="GIF playback frame rate.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("crazy", "real", "random"),
                        default="crazy")
    parser.add_argument("--scatter-starts", action="store_true",
                        help="Force agents to scatter (default for "
                             "--crazy; useful with --real or --random).")
    parser.add_argument("--out", default="/tmp/gps_sim.gif")
    parser.add_argument("--size", type=int, default=720,
                        help="Square output resolution in pixels.")
    parser.add_argument("--colors", type=int, default=128,
                        help="GIF palette size (lower = smaller file).")
    parser.add_argument("--dot-size", type=float, default=1.5,
                        help="Scatter point size (matplotlib `s`).")
    parser.add_argument("--no-ekf", action="store_true",
                        help="Disable the GPS-heading θ EKF and the "
                             "LIDAR/IMU yaw fusion. Models the May 9 "
                             "field config — encoder yaw bias still "
                             "on, but no closed-form θ refit and no "
                             "GPS position update.")
    args = parser.parse_args()

    # Build a scenario dispatch matching gps_sim_gui's main()
    sim_args = argparse.Namespace(
        seed=args.seed,
        obstacles=12, roofs=3, projectors=4,
        jammers=0, foliage=0, spoofers=0,
        random=False, crazy=False, real=False,
        scatter=args.scatter_starts,
        single=False, heading_deg=None,
        goal_lat=None, goal_lon=None,
        headless=False, headless_steps=args.steps,
    )
    if args.mode == "crazy":
        sim_args.crazy = True
        sim_args.random = True
        sim_args.obstacles = 30
        sim_args.roofs = 8
        sim_args.projectors = 12
        sim_args.jammers = 5
        sim_args.foliage = 8
        sim_args.spoofers = 3
        m._apply_crazy_overrides()
    elif args.mode == "real":
        sim_args.real = True
        sim_args.random = True
        sim_args.obstacles = 6
        sim_args.roofs = 1
        sim_args.projectors = 1
        sim_args.foliage = 3
        m._apply_real_overrides()
    elif args.mode == "random":
        sim_args.random = True

    if args.no_ekf:
        m.GPS_HEADING_EKF_ENABLE  = False
        m.LIDAR_IMU_FUSION_ENABLE = False
        m.ODOM_YAW_BIAS_ENABLE    = True

    print(f"Building --{args.mode} scenario, seed={args.seed}…")
    scenario = m.build_scenario(sim_args)
    (cm, start, goal, true_heading, obstacles, roofs, projectors,
     jammers, foliage, spoofers) = scenario

    print(f"Building {args.agents} agents…")
    t_build0 = time.perf_counter()
    agents = m.build_agents(sim_args, scenario, args.agents)
    print(f"  built in {time.perf_counter()-t_build0:.1f}s")

    # ── Figure ─────────────────────────────────────────────────────
    dpi = 80
    inches = args.size / dpi
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi,
                            facecolor="#1a1a1a")
    ax.set_facecolor("#0d1f12")
    ax.set_xlim(-m.MAP_HALF, m.MAP_HALF)
    ax.set_ylim(-m.MAP_HALF, m.MAP_HALF)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.grid(True, color="#163020", linestyle=":", linewidth=0.4)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06)

    # ── Static scenery ─────────────────────────────────────────────
    for cx, cy, r in obstacles:
        ax.add_patch(Circle((cx, cy), r, facecolor="#2a2a2a",
                             edgecolor="#555", linewidth=0.3, zorder=2))
    for x_min, y_min, x_max, y_max in roofs:
        ax.add_patch(Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            facecolor=(0.45, 0.55, 0.95, 0.13),
            edgecolor="#7099dd", linewidth=0.5,
            linestyle="--", zorder=2.4))
    for verts, _bias in projectors:
        ax.add_patch(Polygon(list(verts), closed=True,
                              facecolor="#3a2f1f",
                              edgecolor="#c8a360",
                              linewidth=0.5, zorder=2.5))
    for cx, cy, r in foliage:
        ax.add_patch(Circle(
            (cx, cy), r,
            facecolor=(0.35, 0.75, 0.35, 0.16),
            edgecolor="#5fbb5f", linewidth=0.4,
            linestyle=":", zorder=2.3))
    for cx, cy, r in jammers:
        verts = m.hex_vertices(cx, cy, r)
        ax.add_patch(Polygon(verts, closed=True,
                              facecolor=(0.85, 0.15, 0.45, 0.10),
                              edgecolor="#ff4080", linewidth=0.5,
                              linestyle="--", zorder=2.35))
    for (cx, cy), (fx, fy) in spoofers:
        ax.add_patch(Circle((cx, cy), m.SPOOFER_INFLUENCE_RADIUS_M,
                             fill=False, edgecolor="#cc33ff",
                             linestyle=":", linewidth=0.5, alpha=0.6,
                             zorder=2.45))
        ax.plot([cx], [cy], marker="D", markerfacecolor="#cc33ff",
                 markeredgecolor="white", markersize=4,
                 linestyle="None", zorder=2.6)

    # Goal
    ax.add_patch(Circle(goal, m.GOAL_RADIUS,
                         facecolor=(0.2, 0.9, 0.3, 0.18),
                         edgecolor="#33ff66", linewidth=0.8, zorder=3))
    ax.plot([goal[0]], [goal[1]], "*", color="#33ff66",
             markersize=12, markeredgecolor="white",
             markeredgewidth=0.4, zorder=11)

    # Initial agent scatter
    xy = np.empty((len(agents), 2), dtype=float)
    for i, s in enumerate(agents):
        xy[i, 0] = s.true_pos[0]
        xy[i, 1] = s.true_pos[1]
    agent_scatter = ax.scatter(
        xy[:, 0], xy[:, 1], s=args.dot_size, c="#ff7777",
        edgecolors="none", alpha=0.65, zorder=11)

    # Candidate-goal scatter — one small white pixel per agent at
    # `intermediate_goal_world()`. Lets the viewer see whether each
    # agent's belief of where the goal lives is converging on the
    # true goal (the green star) or oscillating / parked off-target.
    cxy = np.empty((len(agents), 2), dtype=float)
    for i, s in enumerate(agents):
        c = s.intermediate_goal_world()
        cxy[i, 0] = c[0]
        cxy[i, 1] = c[1]
    candidate_scatter = ax.scatter(
        cxy[:, 0], cxy[:, 1], s=0.6, c="#ffffff",
        edgecolors="none", alpha=0.55, zorder=10.5)

    title_text = ax.set_title(
        f"--{args.mode} --agents {args.agents}   t=0.0s   arrived=0",
        color="#e0e0e0", fontsize=10)

    # ── Capture loop ───────────────────────────────────────────────
    print(f"Stepping {args.steps} ticks, capturing every "
          f"{args.frame_every} → up to {args.steps // args.frame_every} "
          f"frames at {args.fps} fps")
    frames = []
    t0 = time.perf_counter()
    last_print = t0
    for step in range(args.steps):
        any_running = False
        for s in agents:
            if s.step():
                any_running = True
        if step % args.frame_every == 0:
            for i, s in enumerate(agents):
                xy[i, 0] = s.true_pos[0]
                xy[i, 1] = s.true_pos[1]
                c = s.intermediate_goal_world()
                cxy[i, 0] = c[0]
                cxy[i, 1] = c[1]
            agent_scatter.set_offsets(xy)
            candidate_scatter.set_offsets(cxy)
            arrived = sum(1 for s in agents if s.arrived)
            title_text.set_text(
                f"--{args.mode} --agents {args.agents}   "
                f"t={step*0.1:5.1f}s   arrived={arrived}/{args.agents}")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.renderer.buffer_rgba())
            frames.append(Image.fromarray(img).convert(
                "P", palette=Image.ADAPTIVE, colors=args.colors))
        # progress every ~10s wall
        now = time.perf_counter()
        if now - last_print > 10.0:
            arrived = sum(1 for s in agents if s.arrived)
            print(f"  step {step:5d}  t={step*0.1:5.1f}s  "
                  f"arrived={arrived}/{args.agents}  "
                  f"frames={len(frames)}  "
                  f"wall={now-t0:.1f}s")
            last_print = now
        if not any_running and step > 50:
            print(f"  all agents idle at step {step}; stopping early.")
            break

    print(f"Captured {len(frames)} frames in {time.perf_counter()-t0:.1f}s")
    print(f"Saving GIF → {args.out}")
    t_save = time.perf_counter()
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True)
    sz = os.path.getsize(args.out)
    print(f"Saved {sz/1e6:.2f} MB in {time.perf_counter()-t_save:.1f}s")


if __name__ == "__main__":
    main()
