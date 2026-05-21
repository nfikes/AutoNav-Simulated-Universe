#!/usr/bin/env python3
"""Plot the GPS path of the three-waypoint success run against the goals.

Renders with a satellite basemap (Esri World Imagery via contextily).
GPS coords are projected to Web Mercator (EPSG:3857) so the path
overlays the basemap correctly.
"""
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as cx


CSV_PATH = "/Users/nathanfikes/Projects/Claude-Sandbox/GPS-Waypoint-Simulation/empirical_results/run_2026-05-17_1643_three-goals-success.csv"
OUT_PATH = "/Users/nathanfikes/Projects/Claude-Sandbox/GPS-Waypoint-Simulation/empirical_results/run_2026-05-17_1643_three-goals-success.png"

GOALS = [
    ("Goal 1", 37.23027, -80.42504),
    ("Goal 2", 37.23013, -80.42524),
    ("Goal 3", 37.22999, -80.42507),
]

WEB_MERCATOR_R = 6_378_137.0


def lonlat_to_webmercator(lon, lat):
    """Standard EPSG:3857 forward projection."""
    x = math.radians(lon) * WEB_MERCATOR_R
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * WEB_MERCATOR_R
    return x, y


def main():
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["gps_lat"])
                lon = float(row["gps_lon"])
            except (KeyError, ValueError):
                continue
            if math.isnan(lat) or math.isnan(lon):
                continue
            # Reject the lon sign-flip glitch and any 0,0 outliers.
            if lon > 0 or lat <= 0:
                continue
            x, y = lonlat_to_webmercator(lon, lat)
            rows.append({
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y,
                "mode": row.get("mode", "???"),
                "goal_status": row.get("goal_status", ""),
                "goal_id": row.get("goal_id", ""),
            })

    if not rows:
        print("no GPS rows", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(11, 11))

    # Path segments colored by mode.
    mode_colors = {"AUTO": "#33ff33", "MAN": "#ff3333", "???": "#cccccc"}
    mode_label_seen = set()
    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        c = mode_colors.get(curr["mode"], "#cccccc")
        label = None
        if curr["mode"] not in mode_label_seen:
            label = curr["mode"]
            mode_label_seen.add(curr["mode"])
        ax.plot(
            [prev["x"], curr["x"]],
            [prev["y"], curr["y"]],
            "-",
            color=c,
            linewidth=2.2,
            alpha=0.95,
            solid_capstyle="round",
            label=label,
        )

    # Start marker.
    start_x, start_y = rows[0]["x"], rows[0]["y"]
    ax.plot(start_x, start_y, marker="o", markersize=14, color="#22ff66",
            markeredgecolor="black", markeredgewidth=1.5, label="Start", zorder=5)

    # Goals.
    for i, (name, glat, glon) in enumerate(GOALS, start=1):
        gx, gy = lonlat_to_webmercator(glon, glat)
        ax.plot(gx, gy, marker="*", markersize=28, color="#ffcc00",
                markeredgecolor="black", markeredgewidth=1.5,
                label="GPS Goals" if i == 1 else None, zorder=5)
        ax.annotate(
            name, (gx, gy),
            xytext=(10, 10), textcoords="offset points",
            fontsize=12, fontweight="bold",
            color="white",
            path_effects=[],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
            zorder=6,
        )

    # Goal success markers.
    success_label_used = False
    for r in rows:
        if r["goal_status"] == "SUCCEEDED":
            ax.plot(
                r["x"], r["y"], marker="X", markersize=20,
                color="#22ff66", markeredgecolor="black", markeredgewidth=1.5,
                label=None if success_label_used else "SUCCEEDED",
                zorder=6,
            )
            success_label_used = True

    ax.set_title(
        "AutoNav 3-waypoint GPS mission — 2026-05-17 16:43-16:52\n"
        f"3/3 goals reached, ~72 m total path, 8 min 27 s"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    # Esri World Imagery basemap. Web Mercator (EPSG:3857) matches the
    # path coordinates above. zoom is auto-picked by contextily but
    # we hint a higher level for the ~70 m mission footprint.
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=cx.providers.Esri.WorldImagery,
            zoom=20,
        )
    except Exception as e:
        print(f"warning: basemap fetch failed ({e}); falling back to plain plot")

    ax.set_xlabel("Easting (m, Web Mercator)")
    ax.set_ylabel("Northing (m, Web Mercator)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f"wrote {OUT_PATH}")
    print(f"  {len(rows)} GPS samples, first fix lat={rows[0]['lat']:.6f} lon={rows[0]['lon']:.6f}")


if __name__ == "__main__":
    main()
