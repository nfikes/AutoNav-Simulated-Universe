"""Run N agents in --real mode (same map, independent GPS noise +
random initial heading) and plot every agent's candidate-goal-vs-real-
goal distance over time as one rainbow-coloured line per agent.

Usage:
    python3 scripts/plot_ensemble_candidate_distance.py [--n 100]
                [--seed N] [--mph N] [--steps N]
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


def make_args(seed: int, n_agents: int) -> SimpleNamespace:
    args = SimpleNamespace(
        seed=seed,
        obstacles=12, roofs=3, projectors=4,
        jammers=0, foliage=0, spoofers=0,
        random=True, crazy=False, real=True,
        scatter=False, agents=n_agents, single=False,
        heading_deg=None, goal_lat=None, goal_lon=None,
        headless=True, headless_steps=0,
    )
    # Mirror the --real branch of main(): override counts, then apply
    # the realistic-outdoor noise overrides.
    args.obstacles = 6
    args.roofs = 1
    args.projectors = 1
    args.jammers = 0
    args.foliage = 3
    args.spoofers = 0
    G._apply_real_overrides()
    return args


def set_max_speed_mph(mph: float):
    G.MAX_SPEED_MPH = mph
    G.MAX_SPEED_MPS = mph * 0.44704
    G.MAX_THRUST = G.MAX_SPEED_MPS * G.LINEAR_DAMPING


def run(seed: int, n_agents: int, max_steps: int):
    args = make_args(seed, n_agents)
    scenario = G.build_scenario(args)
    agents = G.build_agents(args, scenario, n_agents)

    gx, gy = agents[0].goal_world

    # Per-agent traces. Capture t, candidate-distance, and the
    # geometric envelope inputs (r = odom travel, L = robot→goal).
    traces = [{"t": [], "d": [], "r": [], "L": []} for _ in agents]

    def record(idx, sim):
        c = sim.intermediate_goal_world()
        traces[idx]["t"].append(sim.sim_time)
        traces[idx]["d"].append(math.hypot(c[0] - gx, c[1] - gy))
        traces[idx]["r"].append(math.hypot(sim.odom[0], sim.odom[1]))
        if sim.ekf is not None:
            ex, ey = sim.ekf.pos_xy
        else:
            ex, ey = sim.latest_gps()
        traces[idx]["L"].append(math.hypot(ex - gx, ey - gy))

    for i, s in enumerate(agents):
        record(i, s)

    alive = [True] * len(agents)
    for _ in range(max_steps):
        any_running = False
        for i, s in enumerate(agents):
            if not alive[i]:
                continue
            if s.step():
                record(i, s)
                any_running = True
            else:
                record(i, s)
                alive[i] = False
        if not any_running:
            break

    arrivals = [s.arrived for s in agents]
    final_dists = [
        math.hypot(s.true_pos[0] - gx, s.true_pos[1] - gy) for s in agents
    ]
    rejects = [s._cand_reject_count for s in agents]
    return {
        "traces": traces,
        "arrived": arrivals,
        "final_dists": final_dists,
        "rejects": rejects,
        "goal_world": (gx, gy),
        "n_agents": len(agents),
    }


def plot(data, out_path: str, seed: int, mph: float, log_y: bool,
         filter_on: bool):
    traces = data["traces"]
    n = data["n_agents"]
    arrived = sum(data["arrived"])
    final = np.asarray(data["final_dists"])
    rejects = np.asarray(data["rejects"])

    cmap = matplotlib.colormaps["turbo"]
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, tr in enumerate(traces):
        if not tr["t"]:
            continue
        ax.plot(tr["t"], tr["d"], color=colors[i], lw=0.75, alpha=0.85)

    # Theoretical envelope: d_env(r, L) = max(floor, GAIN · L / r),
    # rendered against time using the ensemble's median r(t) and L(t)
    # so the curve tracks the typical agent geometry rather than any
    # single one. Only plot where we have ≥ half the agents alive.
    max_len = max(len(tr["t"]) for tr in traces)
    rs = np.full((n, max_len), np.nan)
    ls = np.full((n, max_len), np.nan)
    ts = np.full((n, max_len), np.nan)
    for i, tr in enumerate(traces):
        m = len(tr["t"])
        rs[i, :m] = tr["r"]
        ls[i, :m] = tr["L"]
        ts[i, :m] = tr["t"]
    median_r = np.nanmedian(rs, axis=0)
    median_L = np.nanmedian(ls, axis=0)
    median_t = np.nanmedian(ts, axis=0)
    valid = (~np.isnan(median_t)
             & ~np.isnan(median_r)
             & ~np.isnan(median_L)
             & (median_r > 0))
    env = np.where(
        median_r[valid] > G.CANDIDATE_ENV_MIN_R_M,
        np.maximum(G.CANDIDATE_ENV_FLOOR_M,
                   G.CANDIDATE_ENV_GAIN_M * median_L[valid]
                   / np.maximum(median_r[valid], 1e-9)),
        np.nan)
    rej_thr = G.CANDIDATE_ENV_REJECT_K * env
    ax.plot(median_t[valid], env, color="black", lw=2.0, alpha=0.85,
            label=f"envelope d_env = max({G.CANDIDATE_ENV_FLOOR_M:.2g}, "
                  f"{G.CANDIDATE_ENV_GAIN_M:.2g}·L/r)  "
                  f"(median r,L)")
    ax.plot(median_t[valid], rej_thr, color="black", lw=1.0, ls="--",
            alpha=0.7,
            label=f"reject threshold ({G.CANDIDATE_ENV_REJECT_K:.0f}× envelope)")

    ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.6,
               label="1 m goal radius")
    ax.set_xlabel("sim time [s]")
    ax.set_ylabel("‖candidate goal − true goal‖ [m]")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax.set_title(
        f"Candidate-goal convergence — {n} agents, --real "
        f"(seed={seed}, max_v={mph:.1f} mph, filter={'ON' if filter_on else 'OFF'})\n"
        f"arrived = {arrived}/{n}  ·  "
        f"true→goal:  median={np.median(final):.2f} m, "
        f"max={final.max():.2f} m  ·  "
        f"rejects: median={int(np.median(rejects))}, max={int(rejects.max())}")

    # Sidecar colourbar mapping line index → colour, so the rainbow
    # is interpretable as "agent ID".
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=max(1, n - 1)),
        cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.04)
    cb.set_label("agent index", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100,
                   help="number of agents (default 100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mph", type=float, default=5.0,
                   help="terminal speed cap (default 5.0 — stock --real)")
    p.add_argument("--steps", type=int, default=6000,
                   help="max physics ticks (0.1 s each)")
    p.add_argument("--log-y", action="store_true",
                   help="plot distance on a log y-axis")
    p.add_argument("--filter", choices=["on", "off"], default="on",
                   help="enable/disable the 1/r-envelope outlier "
                        "filter (default on). With it off the agent "
                        "uses only the EWMA + snap smoother.")
    p.add_argument("--out", default=os.path.join(
        REPO_ROOT, "ensemble_candidate_distance.png"))
    a = p.parse_args()
    set_max_speed_mph(a.mph)
    G.CANDIDATE_ENV_ENABLE = (a.filter == "on")
    data = run(a.seed, a.n, a.steps)
    plot(data, a.out, a.seed, a.mph, a.log_y, filter_on=(a.filter == "on"))


if __name__ == "__main__":
    main()
