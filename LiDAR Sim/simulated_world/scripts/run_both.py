#!/usr/bin/env python3
"""Run the ramp test at 20% and 30% grade thresholds, saving separate GIFs."""
import lidar_sim_gui as sim
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent

# ── Run 20% threshold ──
print("\n" + "="*60)
print("  RUNNING 20% GRADE THRESHOLD (11.3 deg)")
print("="*60)
sim.GRADE_FREE_MAX = 8.0
sim.GRADE_MODERATE_MAX = 11.3
sim.GRADE_STEEP_MAX = 11.3
sim.TRAVERSABLE_MAX_DEG = 11.3

import test_ramp_crossing as trc
# Patch the title
import matplotlib.pyplot as plt

# Save original
orig_setup = trc._setup_figure.__code__

trc.main()
shutil.move(str(BASE / "test_ramp_results.png"), str(BASE / "results_20pct.png"))
shutil.move(str(BASE / "test_ramp_crossing.gif"), str(BASE / "ramp_test_20pct.gif"))
print(f"\n  Saved: results_20pct.png, ramp_test_20pct.gif")

# ── Run 30% threshold ──
print("\n" + "="*60)
print("  RUNNING 30% GRADE THRESHOLD (16.7 deg)")
print("="*60)
sim.GRADE_FREE_MAX = 12.0
sim.GRADE_MODERATE_MAX = 16.7
sim.GRADE_STEEP_MAX = 16.7
sim.TRAVERSABLE_MAX_DEG = 16.7

# Update expected results for 30%
for r in trc.RAMPS:
    if r["slope"] <= 15:
        r["expected"] = "PASS"
    else:
        r["expected"] = "NO-PASS"

trc.main()
shutil.move(str(BASE / "test_ramp_results.png"), str(BASE / "results_30pct.png"))
shutil.move(str(BASE / "test_ramp_crossing.gif"), str(BASE / "ramp_test_30pct.gif"))
print(f"\n  Saved: results_30pct.png, ramp_test_30pct.gif")

print("\n" + "="*60)
print("  BOTH RUNS COMPLETE")
print("="*60)
