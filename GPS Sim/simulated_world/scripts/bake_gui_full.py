#!/usr/bin/env python3
"""Headless bake of the FULL GPSWaypointGUI (not the stripped scatter).

Drives ``GPSWaypointGUI`` under the Agg backend — no Qt window pops —
and saves the 16:9 figure (main map + mini cams + status panel) as a
GIF. The output looks exactly like the live ``Run_GPS_SIM`` window.

Usage (single agent, --real, default seed):
  python scripts/bake_gui_full.py
  python scripts/bake_gui_full.py --mode crazy --agents 1 --steps 1200
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import argparse
import time
import numpy as np

import matplotlib
# Neutralize gps_sim_gui's top-level ``matplotlib.use("Qt5Agg")`` so the
# import doesn't switch us off Agg (no display required, no Qt window).
_orig_use = matplotlib.use
matplotlib.use = lambda *a, **kw: None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gps_sim_gui as m  # noqa: E402

matplotlib.use = _orig_use
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("real", "crazy", "random", "scripted"),
                   default="real")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=1200,
                   help="Max sim ticks (each = 0.1s).")
    p.add_argument("--frame-every", type=int, default=3,
                   help="Capture every Nth tick (3 ≈ 10fps sim).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="bakes/single_agent_full_gui.gif")
    p.add_argument("--colors", type=int, default=128)
    args = p.parse_args()

    sim_args = argparse.Namespace(
        seed=args.seed,
        obstacles=6, roofs=1, projectors=1,
        jammers=0, foliage=3, spoofers=0,
        random=False, crazy=False, real=False,
        scatter=False, single=(args.agents == 1),
        heading_deg=None, goal_lat=None, goal_lon=None,
        headless=False, headless_steps=args.steps,
        agents=args.agents,
    )
    if args.mode == "real":
        sim_args.real = True
        sim_args.random = True
        m._apply_real_overrides()
    elif args.mode == "crazy":
        sim_args.crazy = True
        sim_args.random = True
        sim_args.obstacles = 30
        sim_args.roofs = 8
        sim_args.projectors = 12
        sim_args.jammers = 5
        sim_args.foliage = 8
        sim_args.spoofers = 3
        m._apply_crazy_overrides()
    elif args.mode == "random":
        sim_args.random = True

    print(f"Building --{args.mode} scenario seed={args.seed}…")
    scenario = m.build_scenario(sim_args)
    (cm, start, goal, true_heading, obstacles, roofs, projectors,
     jammers, foliage, spoofers) = scenario

    print(f"Building {args.agents} agent(s)…")
    agents = m.build_agents(sim_args, scenario, max(1, args.agents))
    sim = agents[0]
    peers = agents[1:]

    print("Constructing full GUI (headless)…")
    gui = m.GPSWaypointGUI(
        sim, obstacles, sim_args,
        roofs=roofs, projectors=projectors,
        jammers=jammers, foliage=foliage,
        spoofers=spoofers, peers=peers)
    # Force the full-redraw path. Blit needs a live event loop to
    # snapshot backgrounds correctly; for offline rendering we just do
    # a full canvas.draw() per captured frame.
    gui._blit_supported = False

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Stepping ≤{args.steps} ticks, capturing every "
          f"{args.frame_every} → up to "
          f"{args.steps // args.frame_every} frames @ {args.fps} fps")
    frames = []
    t0 = time.perf_counter()
    last_print = t0
    for step in range(args.steps):
        any_running = False
        for s in gui.all_sims:
            if s.step():
                any_running = True
        if step % args.frame_every == 0:
            gui._render_dynamic_main()
            gui._render_dynamic_panels()
            gui.fig.canvas.draw()
            img = np.asarray(gui.fig.canvas.renderer.buffer_rgba())
            frames.append(Image.fromarray(img).convert(
                "P", palette=Image.ADAPTIVE, colors=args.colors))
        now = time.perf_counter()
        if now - last_print > 10.0:
            print(f"  step {step:5d}  t={step*0.1:5.1f}s  "
                  f"frames={len(frames)}  wall={now-t0:.1f}s")
            last_print = now
        # Stop early once primary agent is classified and idle.
        if not any_running and step > 50:
            print(f"  all agents idle at step {step}; stopping.")
            break

    print(f"Captured {len(frames)} frames in "
          f"{time.perf_counter()-t0:.1f}s")
    print(f"Saving GIF → {out_path}")
    t_save = time.perf_counter()
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True)
    sz = os.path.getsize(out_path)
    print(f"Saved {sz/1e6:.2f} MB in {time.perf_counter()-t_save:.1f}s")


if __name__ == "__main__":
    main()
