"""Render the May 9 field datapoints as 2 x 2 trajectory plots.

  Row 1 = DP1, Row 2 = DP2
  Col 1 = EKF OFF (raw odom — field config)
  Col 2 = EKF ON  (the corrected GPS-heading EKF + LIDAR/IMU fusion)

Each panel plots, in world frame:
  * Truth path   — red  : where the body physically is.
  * Virtual path — green: where the EKF says the body is
                          (``sim.ekf.pos_xy`` sampled per step).
Plus the start (yellow X), the GPS goal (green star + 1 m circle),
and the recorded field-end position (pink X) for reference.

Output: render_field_datapoints.png
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gps_sim_gui as g
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

EARTH_R = 6378137.0
def latlon_to_m(lat, lon, lat0, lon0):
    de = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    dn = math.radians(lat - lat0) * EARTH_R
    return de, dn

DATAPOINTS = [
    dict(name="DP1",
         start=(37.23000, -80.42498),
         goal =(37.23027, -80.42504),
         end  =(37.22984, -80.42516),
         th_deg=170),
    dict(name="DP2",
         start=(37.23038, -80.42492),
         goal =(37.23027, -80.42504),
         end  =(37.23051, -80.42493),
         th_deg=328),
]

def run_sim(th_deg, gx, gy, max_steps=5000):
    rng = np.random.default_rng(42)
    cm = g.Costmap(obstacles=[], projectors=[])
    sim = g.GPSWaypointSim(cm, (0.0, 0.0), math.radians(th_deg),
                           (gx, gy), rng,
                           roofs=[], projectors=[],
                           jammers=[], foliage=[], spoofers=[])
    truth_xy  = [tuple(sim.true_pos)]
    virtual_xy = [
        tuple(sim.ekf.pos_xy) if sim.ekf is not None else (0.0, 0.0)
    ]
    n = 0
    while n < max_steps:
        if not sim.step():
            break
        n += 1
        truth_xy.append(tuple(sim.true_pos))
        virtual_xy.append(
            tuple(sim.ekf.pos_xy) if sim.ekf is not None else (0.0, 0.0))
    return sim, n, np.array(truth_xy), np.array(virtual_xy)


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.patch.set_facecolor("#101010")

for row, dp in enumerate(DATAPOINTS):
    s_lat, s_lon = dp["start"]
    g_lat, g_lon = dp["goal"]
    e_lat, e_lon = dp["end"]
    gx, gy = latlon_to_m(g_lat, g_lon, s_lat, s_lon)
    ex, ey = latlon_to_m(e_lat, e_lon, s_lat, s_lon)

    for col, label, ekf_on in (
            (0, "EKF OFF (field config)", False),
            (1, "EKF ON  (corrected)",    True)):

        g._apply_real_overrides()
        g.GPS_HEADING_EKF_ENABLE  = ekf_on
        g.LIDAR_IMU_FUSION_ENABLE = ekf_on
        g.ODOM_YAW_BIAS_ENABLE    = True

        sim, steps, truth, virtual = run_sim(dp["th_deg"], gx, gy)

        ax = axes[row, col]
        ax.set_facecolor("#0d1f12")
        ax.set_aspect("equal")
        ax.set_title(f"{dp['name']} — {label}\n"
                     f"true_heading={dp['th_deg']}°, steps={steps}, "
                     f"arrived={sim.arrived}",
                     color="#e0e0e0", fontsize=10)
        ax.tick_params(colors="#a0a0a0")
        for spine in ax.spines.values():
            spine.set_color("#444")

        # Truth path (red) — where the body physically is.
        ax.plot(truth[:, 0], truth[:, 1],
                "-", color="#ff5050", lw=1.5, alpha=0.9,
                label="Truth path")
        ax.plot([truth[-1, 0]], [truth[-1, 1]],
                "o", color="#ff5050", markersize=8,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Truth end")
        # Virtual / EKF path (green) — where the robot thinks it is.
        ax.plot(virtual[:, 0], virtual[:, 1],
                "-", color="#33ff66", lw=1.2, alpha=0.85,
                label="Virtual (EKF) path")
        ax.plot([virtual[-1, 0]], [virtual[-1, 1]],
                "o", color="#33ff66", markersize=7,
                markeredgecolor="black", markeredgewidth=0.5,
                label="Virtual end")
        # GPS goal
        ax.plot([gx], [gy], "*", color="#33ff66", markersize=18,
                markeredgecolor="white", markeredgewidth=0.5,
                label="GPS goal")
        ax.add_patch(Circle((gx, gy), 1.0,
                            facecolor=(0.2, 0.9, 0.3, 0.18),
                            edgecolor="#33ff66", lw=1.2))
        # Start
        ax.plot([0], [0], "x", color="#ffcc00", markersize=12,
                markeredgewidth=2, label="Start")
        # Field end (recorded — single position, no path data)
        ax.plot([ex], [ey], "X", color="#ff8888", markersize=14,
                markeredgecolor="white", markeredgewidth=0.6,
                label="Field end (recorded)")

        # Distances summary
        d_truth_goal  = math.hypot(truth[-1, 0]   - gx, truth[-1, 1]   - gy)
        d_virt_goal   = math.hypot(virtual[-1, 0] - gx, virtual[-1, 1] - gy)
        d_truth_field = math.hypot(truth[-1, 0]   - ex, truth[-1, 1]   - ey)
        ax.text(0.02, 0.98,
                f"Truth → goal:       {d_truth_goal:6.2f} m\n"
                f"Virtual → goal:     {d_virt_goal:6.2f} m\n"
                f"Truth → field_end:  {d_truth_field:6.2f} m",
                transform=ax.transAxes,
                color="#e0e0e0", fontsize=8,
                fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(facecolor="#141414",
                          edgecolor="#333", alpha=0.85))

        # Auto-zoom: include all key points and full trails.
        xs = list(truth[:, 0]) + list(virtual[:, 0]) + [0, gx, ex]
        ys = list(truth[:, 1]) + list(virtual[:, 1]) + [0, gy, ey]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad = max(2.0, 0.08 * max(x_max - x_min, y_max - y_min))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.legend(loc="lower right", fontsize=7,
                  facecolor="#101010", labelcolor="#d0d0d0",
                  edgecolor="#333")
        ax.set_xlabel("east (m)", color="#a0a0a0", fontsize=8)
        ax.set_ylabel("north (m)", color="#a0a0a0", fontsize=8)

fig.suptitle("May 9 Field Datapoints — Truth (red) vs Virtual/EKF (green)",
             color="#e0e0e0", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(os.path.dirname(__file__), "..",
                   "render_field_datapoints.png")
fig.savefig(out, dpi=130, facecolor="#101010")
print(f"wrote {out}")
