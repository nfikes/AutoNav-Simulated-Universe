#!/usr/bin/env python3
"""Headless cProfile harness for lidar_sim_gui.

Forces matplotlib Agg, instantiates DualNavGUI *partially* (skips
plt.show), simulates a click and a Shift+Click, then runs a bounded
number of _animate() iterations. Used to surface non-trimesh/embreex
hotspots remaining after embreex was installed.
"""
import sys
import os
import cProfile
import pstats
import io
import time
import argparse
from pathlib import Path

# Force Agg before any pyplot import inside lidar_sim_gui
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Ensure we can import the GUI module
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import lidar_sim_gui as sim  # noqa: E402


class HeadlessGUI(sim.DualNavGUI):
    """Subclass that skips plt.show() so the harness keeps control."""
    def _setup_gui(self):
        # Re-implement just enough of _setup_gui to populate the
        # attributes that _do_scan / _plan_paths / _animate touch,
        # without entering the Qt event loop.
        #
        # The single-agent GUI now uses a 2-panel mpl figure (map + BEV)
        # plus a PyQtGraph GLViewWidget — but the headless harness can't
        # construct an OpenGL context under MPLBACKEND=Agg with no
        # display, so it substitutes ``sim_view.MockThreeDView`` (a
        # no-op stand-in with the same setData API). The rest of the
        # animate loop runs unchanged.
        self.fig = plt.figure(figsize=(15, 7), facecolor="#1e1e1e")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
        self.ax_map = self.fig.add_subplot(gs[0])
        import numpy as np
        self.ax_map.imshow(
            self.terrain_img, origin="lower", extent=self.terrain_extent,
            cmap="gray", aspect="equal", vmin=0, vmax=1)
        self.global_img = self.ax_map.imshow(
            np.zeros((10, 10, 4), dtype=np.uint8),
            origin="lower", extent=[-8, 8, -8, 8],
            aspect="equal", alpha=0.6, zorder=3)
        self.new_mk, = self.ax_map.plot([], [], "ro", markersize=10)
        self.old_mk, = self.ax_map.plot([], [], "bs", markersize=10)
        self.old_mk.set_visible(False)
        self.goal_mk, = self.ax_map.plot([], [], "g*", markersize=18)
        self.new_path_line, = self.ax_map.plot([], [], "r-", linewidth=2)
        self.old_path_line, = self.ax_map.plot([], [], "b--", linewidth=2)
        self.old_path_line.set_visible(False)

        self.ax_bev = self.fig.add_subplot(gs[1])
        self.bev_handle = self.ax_bev.imshow(
            np.full((sim.BEV_SIZE, sim.BEV_SIZE, 3), 240, dtype=np.uint8),
            aspect="equal")

        # 3D panel: mock view (no-op setData) for headless runs.
        import sim_view
        self.three_d = sim_view.MockThreeDView()
        self.ax3d = None  # legacy alias — no mpl axis on the GL path
        self._3d_state = sim_view.ThreeDState()

        # ``self.canvas`` aliases ``self.fig.canvas`` so the _animate
        # loop's blit code (which now reads ``self.canvas``) works
        # under the headless harness too.
        self.canvas = self.fig.canvas

        self.status = self.fig.text(0.5, 0.01, "")

        # Auto-place + initial scan + plan, identical to the real GUI.
        sx, sy = self.default_start
        gx, gy = self.default_goal
        self.robot_pos = (sx, sy)
        self.goal_pos = (gx, gy)
        self.new_mk.set_data([sx], [sy])
        self.old_mk.set_data([sx], [sy])
        self.goal_mk.set_data([gx], [gy])
        self._do_scan(sx, sy)
        self._plan_paths()
        # NOTE: deliberately no plt.show()


class _FakeEvent:
    def __init__(self, ax, x, y, key=None):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y
        self.key = key


def run_workload(gui, max_anim_seconds=20.0, max_frames=400):
    """Realistic-ish interaction sequence."""
    # Click at (0, 0)
    gui._on_click(_FakeEvent(gui.ax_map, 0.0, 0.0, key=None))
    # Shift+Click at (5, 0)
    gui._on_click(_FakeEvent(gui.ax_map, 5.0, 0.0, key="shift"))
    # Bounded animation by monkey-patching the loop bound + deadline.
    deadline = time.perf_counter() + max_anim_seconds
    real_perf = time.perf_counter

    # Monkey-patch range(5000) used inside _animate by overriding it
    # via a wrapper that bails out when time/frame budget exceeded.
    import lidar_sim_gui as _sim
    original_animate = gui._animate

    def bounded_animate():
        # Patch in a tiny "frame" cap by monkey-patching the module's
        # builtins. Simpler: just call the real animate, but stop early
        # by setting goal_pos to a far reachable place, then breaking
        # via an exception-driven deadline. Easiest: temporarily replace
        # plt.pause to count frames and raise once budget hit.
        frame_count = {"n": 0}
        real_pause = plt.pause

        def counting_pause(t):
            frame_count["n"] += 1
            if frame_count["n"] >= max_frames or real_perf() >= deadline:
                raise StopIteration
            return real_pause(t)
        plt.pause = counting_pause
        try:
            original_animate()
        except StopIteration:
            pass
        finally:
            plt.pause = real_pause
        return frame_count["n"]

    return bounded_animate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=1000)
    ap.add_argument("--threshold", type=int, default=15,
                    choices=[15, 20, 30])
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--max-anim-seconds", type=float, default=25.0)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default=None,
                    help="Optional path to dump a sortable .prof file")
    ap.add_argument("--filter", default=None,
                    help="If set, only show stats matching this substring "
                         "(e.g. 'lidar_sim_gui').")
    args = ap.parse_args()

    cfg = {15: (6.0, 8.5), 20: (8.0, 11.3), 30: (12.0, 16.7)}
    free_max, deg = cfg[args.threshold]
    sim.GRADE_FREE_MAX = free_max
    sim.GRADE_MODERATE_MAX = deg
    sim.GRADE_STEEP_MAX = deg
    sim.TRAVERSABLE_MAX_DEG = deg
    sim.N_RAYS = args.rays

    print(f"Building GUI (rays={args.rays}, threshold={args.threshold})...")
    t0 = time.perf_counter()
    gui = HeadlessGUI(sim.STL_PATH)
    t_setup = time.perf_counter() - t0
    print(f"  setup took {t_setup:.2f}s")

    pr = cProfile.Profile()
    print("Profiling workload...")
    t0 = time.perf_counter()
    pr.enable()
    try:
        frames = run_workload(gui,
                              max_anim_seconds=args.max_anim_seconds,
                              max_frames=args.max_frames)
    finally:
        pr.disable()
    t_work = time.perf_counter() - t0
    print(f"  workload took {t_work:.2f}s ({frames} animation frames)")

    if args.out:
        pr.dump_stats(args.out)
        print(f"  wrote raw profile to {args.out}")

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats("cumulative")
    if args.filter:
        stats.print_stats(args.filter, args.top)
    else:
        stats.print_stats(args.top)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
