#!/usr/bin/env python3
"""cProfile harness for the multi-agent ramp benchmark.

Headless, deterministic. Sums per-call costs into the main process for
the parallel scan path. The pool workers run on their own — we profile
only the main thread (orchestration + viz cost), which is what users
actually feel.
"""
import sys
import os
import time
import cProfile
import pstats
import io
import argparse
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=1000)
    ap.add_argument("--threshold", type=int, default=30, choices=[15, 20, 30])
    ap.add_argument("--agents", type=int, default=5, choices=[5, 10, 15])
    ap.add_argument("--phase", choices=["1", "both"], default="1")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    # Mimic test_ramp_crossing.main() argv plumbing
    sys.argv = [
        "test_ramp_crossing.py",
        "--threshold", str(args.threshold),
        "--rays", str(args.rays),
        "--agents", str(args.agents),
        "--phase", args.phase,
        "--no-video",
    ]

    import test_ramp_crossing as bench  # noqa: E402

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    try:
        bench.main()
    except SystemExit:
        pass
    finally:
        pr.disable()
    elapsed = time.perf_counter() - t0
    print(f"\n  Benchmark wall-clock: {elapsed:.2f}s")

    if args.out:
        pr.dump_stats(args.out)
        print(f"  wrote raw profile to {args.out}")

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    if args.filter:
        stats.print_stats(args.filter, args.top)
    else:
        stats.print_stats(args.top)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
