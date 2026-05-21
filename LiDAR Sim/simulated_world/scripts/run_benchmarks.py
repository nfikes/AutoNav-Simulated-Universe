#!/usr/bin/env python3
"""Run the ramp test at 15%, 20%, and 30% grade thresholds.
Results saved to benchmark_results/ folder.

Usage:
  python run_benchmarks.py                # run all three (15/20/30)
  python run_benchmarks.py --only 15      # run only the 15% threshold
  python run_benchmarks.py --only 20,30   # run a comma-separated subset
"""
import argparse
import shutil
from pathlib import Path

import lidar_sim_gui as sim
import test_ramp_crossing as trc

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
OUT = PROJECT_ROOT / "benchmark_results"

ALL_THRESHOLDS = [
    {"pct": 15, "deg": 8.5,  "free": 6.0, "pass_up_to": 5},
    {"pct": 20, "deg": 11.3, "free": 8.0, "pass_up_to": 10},
    {"pct": 30, "deg": 16.7, "free": 12.0, "pass_up_to": 15},
]


def _select_thresholds(only):
    if not only:
        return ALL_THRESHOLDS
    wanted = {int(s.strip()) for s in only.split(",") if s.strip()}
    sel = [t for t in ALL_THRESHOLDS if t["pct"] in wanted]
    if not sel:
        raise SystemExit(
            f"--only {only!r} matched no thresholds in "
            f"{[t['pct'] for t in ALL_THRESHOLDS]}")
    return sel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated subset of thresholds to run (e.g. '15' or '20,30').")
    parser.add_argument(
        "--agents", type=int, default=None,
        help="Run only the first N agents (smoke test). Default: all 15.")
    parser.add_argument(
        "--suffix", default="",
        help="Suffix appended to output filenames (e.g. '_smoke').")
    args = parser.parse_args()
    thresholds = _select_thresholds(args.only)

    OUT.mkdir(exist_ok=True)

    # Optionally trim to first N agents (smoke test before full 15-agent run).
    if args.agents is not None:
        if args.agents <= 0 or args.agents > len(trc.RAMPS):
            raise SystemExit(f"--agents must be 1..{len(trc.RAMPS)}")
        trc.RAMPS = trc.RAMPS[:args.agents]
        print(f"[smoke] running first {args.agents} agents only")

    for t in thresholds:
        pct, deg = t["pct"], t["deg"]
        print("\n" + "=" * 60)
        print(f"  RUNNING {pct}% GRADE THRESHOLD ({deg} deg)")
        print("=" * 60)
        sim.GRADE_FREE_MAX = t["free"]
        sim.GRADE_MODERATE_MAX = deg
        sim.GRADE_STEEP_MAX = deg
        sim.TRAVERSABLE_MAX_DEG = deg

        for r in trc.RAMPS:
            r["expected"] = "PASS" if r["slope"] <= t["pass_up_to"] else "NO-PASS"

        trc.main()
        suffix = args.suffix
        p1_path = OUT / f"results_{pct}pct_phase1{suffix}.png"
        p2_path = OUT / f"results_{pct}pct_phase2{suffix}.png"
        shutil.move(str(BASE / "test_ramp_results.png"), str(p2_path))
        shutil.move(str(BASE / "test_ramp_results_phase1.png"), str(p1_path))
        shutil.move(str(BASE / "test_ramp_crossing.gif"),
                    str(OUT / f"ramp_test_{pct}pct{suffix}.gif"))
        # B&W obstacle masks (ground-truth comparable)
        bw1 = BASE / "test_ramp_costmap_phase1_bw.png"
        bw2 = BASE / "test_ramp_costmap_phase2_bw.png"
        if bw1.exists():
            shutil.move(str(bw1), str(OUT / f"costmap_{pct}pct_phase1_bw{suffix}.png"))
        if bw2.exists():
            shutil.move(str(bw2), str(OUT / f"costmap_{pct}pct_phase2_bw{suffix}.png"))
        # Combined side-by-side image for fast visual comparison.
        # Crop the LiDAR (left) panel of each phase, keep only the
        # costmap panel (right half), then paste side-by-side.
        try:
            from PIL import Image
            p1 = Image.open(str(p1_path))
            p2 = Image.open(str(p2_path))
            p1_cm = p1.crop((p1.width // 2, 0, p1.width, p1.height))
            p2_cm = p2.crop((p2.width // 2, 0, p2.width, p2.height))
            h = max(p1_cm.height, p2_cm.height)
            combo = Image.new("RGB", (p1_cm.width + p2_cm.width, h), (30, 30, 30))
            combo.paste(p1_cm, (0, 0))
            combo.paste(p2_cm, (p1_cm.width, 0))
            combo_path = OUT / f"results_{pct}pct_combined{suffix}.png"
            combo.save(str(combo_path))
            print(f"\n  Saved to benchmark_results/: "
                  f"results_{pct}pct_phase1{suffix}.png, "
                  f"results_{pct}pct_phase2{suffix}.png, "
                  f"results_{pct}pct_combined{suffix}.png, "
                  f"ramp_test_{pct}pct{suffix}.gif")
        except Exception as e:
            print(f"\n  (combined-image step failed: {e})")

    print("\n" + "=" * 60)
    print("  ALL BENCHMARKS COMPLETE")
    print(f"  Results in: {OUT}")
    print("=" * 60)


# IMPORTANT: must be guarded so multiprocessing.spawn workers do not
# re-execute the benchmark loop when they reimport this script.
if __name__ == "__main__":
    main()
