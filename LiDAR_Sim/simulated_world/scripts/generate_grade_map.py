#!/usr/bin/env python3
"""
Lidar-Simulation Grade Map Generator

Loads a terrain STL mesh and produces a color-coded gradient (slope) map
following the algorithm from terrain-grade-layer-plan.md.

Uses triangle face normals for accurate slope computation, then rasterizes
each triangle into a 2D grid weighted by area.

Grade thresholds (from the plan):
  0 -  8 deg  -> FREE        (green)
  8 - 12 deg  -> MODERATE    (yellow)
 12 - 17 deg  -> STEEP       (orange-red)
 >  17 deg    -> LETHAL      (dark red)
 z_spread > threshold -> OBSTACLE (black)
 No data     -> UNKNOWN      (gray)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from stl import mesh


# ── Grade thresholds (degrees) ──────────────────────────────────────
FREE_MAX_DEG = 8.0
MODERATE_MAX_DEG = 12.0
STEEP_MAX_DEG = 17.0
Z_SPREAD_THRESHOLD = 0.3  # meters – vertical spread that marks an obstacle


def load_stl(stl_path: str) -> mesh.Mesh:
    print(f"Loading STL: {stl_path}")
    m = mesh.Mesh.from_file(stl_path)
    print(f"  Triangles: {len(m.vectors):,}")
    return m


def compute_face_slopes(m: mesh.Mesh) -> np.ndarray:
    """Compute slope angle (degrees) for each triangle from its face normal."""
    normals = m.normals  # (N, 3)
    norms_mag = np.linalg.norm(normals, axis=1)
    norms_mag[norms_mag == 0] = 1
    cos_angle = np.abs(normals[:, 2]) / norms_mag
    return np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))


def compute_face_areas(m: mesh.Mesh) -> np.ndarray:
    """Compute the area of each triangle."""
    v = m.vectors  # (N, 3, 3)
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    cross = np.cross(e1, e2)
    return 0.5 * np.linalg.norm(cross, axis=1)


def rasterize_triangles(
    m: mesh.Mesh,
    face_slopes: np.ndarray,
    cell_res: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    """
    Rasterize triangle face slopes onto a 2D grid.

    Each triangle is placed into the grid cell of its centroid.
    Cells get the area-weighted average slope of all triangles whose
    centroid falls in that cell.

    Obstacle detection: cells containing near-vertical faces (slope > 70°)
    are flagged via the obstacle_grid (replaces z_spread).
    """
    verts = m.vectors  # (N, 3, 3)
    areas = compute_face_areas(m)
    centroids = verts.mean(axis=1)  # (N, 3)

    x_min_all = verts[:, :, 0].min()
    y_min_all = verts[:, :, 1].min()
    x_max_all = verts[:, :, 0].max()
    y_max_all = verts[:, :, 1].max()

    grid_w = int(np.ceil((x_max_all - x_min_all) / cell_res)) + 1
    grid_h = int(np.ceil((y_max_all - y_min_all) / cell_res)) + 1
    n_cells = grid_w * grid_h
    print(f"  Grid size: {grid_w} x {grid_h} ({n_cells:,} cells) @ {cell_res}m resolution")

    # Bin each triangle centroid into its grid cell
    cx = np.clip(((centroids[:, 0] - x_min_all) / cell_res).astype(int), 0, grid_w - 1)
    cy = np.clip(((centroids[:, 1] - y_min_all) / cell_res).astype(int), 0, grid_h - 1)
    cell_idx = cy * grid_w + cx

    # Area-weighted slope accumulation
    weighted_slope = face_slopes * areas
    slope_sum = np.zeros(n_cells, dtype=np.float64)
    weight_sum = np.zeros(n_cells, dtype=np.float64)
    np.add.at(slope_sum, cell_idx, weighted_slope)
    np.add.at(weight_sum, cell_idx, areas)

    # Obstacle detection: accumulate area of near-vertical faces (>70°) per cell
    vertical_mask = face_slopes > 70.0
    vert_area = np.zeros(n_cells, dtype=np.float64)
    np.add.at(vert_area, cell_idx[vertical_mask], areas[vertical_mask])

    valid = weight_sum > 0
    slope_grid = np.full(n_cells, np.nan)
    slope_grid[valid] = slope_sum[valid] / weight_sum[valid]

    # A cell is an obstacle if vertical faces make up significant area in it
    obstacle_grid = np.zeros(n_cells, dtype=np.float64)
    obstacle_grid[valid] = vert_area[valid] / weight_sum[valid]

    slope_grid = slope_grid.reshape(grid_h, grid_w)
    obstacle_grid = obstacle_grid.reshape(grid_h, grid_w)

    print(f"  Rasterized {len(verts):,} triangles by centroid")

    return slope_grid, obstacle_grid, valid.reshape(grid_h, grid_w), x_min_all, y_min_all, grid_w, grid_h


def grade_to_color(slope: np.ndarray, obstacle_frac: np.ndarray) -> np.ndarray:
    """
    Map slope (degrees) to RGBA color image.

    Colors follow the plan's classification:
      FREE       (0-8°)   : green
      MODERATE   (8-12°)  : yellow
      STEEP      (12-17°) : orange-red
      LETHAL     (>17°)   : dark red
      OBSTACLE   (vertical faces) : black
      UNKNOWN    (no data): gray
    """
    h, w = slope.shape
    img = np.zeros((h, w, 4), dtype=np.float32)

    no_data = np.isnan(slope)
    # Cell is obstacle if >20% of its area is near-vertical faces
    obstacle = (obstacle_frac > 0.2) & ~no_data
    s = np.nan_to_num(slope, nan=0.0)

    # FREE: green gradient
    free = (~no_data) & (~obstacle) & (s <= FREE_MAX_DEG)
    t = np.clip(s / FREE_MAX_DEG, 0, 1)
    img[free, 0] = 0.2 + 0.4 * t[free]
    img[free, 1] = 0.8 + 0.1 * t[free]
    img[free, 2] = 0.2
    img[free, 3] = 1.0

    # MODERATE: yellow-orange gradient
    moderate = (~no_data) & (~obstacle) & (s > FREE_MAX_DEG) & (s <= MODERATE_MAX_DEG)
    t = np.clip((s - FREE_MAX_DEG) / (MODERATE_MAX_DEG - FREE_MAX_DEG), 0, 1)
    img[moderate, 0] = 0.9 + 0.1 * t[moderate]
    img[moderate, 1] = 0.9 - 0.3 * t[moderate]
    img[moderate, 2] = 0.1
    img[moderate, 3] = 1.0

    # STEEP: orange to red
    steep = (~no_data) & (~obstacle) & (s > MODERATE_MAX_DEG) & (s <= STEEP_MAX_DEG)
    t = np.clip((s - MODERATE_MAX_DEG) / (STEEP_MAX_DEG - MODERATE_MAX_DEG), 0, 1)
    img[steep, 0] = 1.0 - 0.1 * t[steep]
    img[steep, 1] = 0.6 - 0.45 * t[steep]
    img[steep, 2] = 0.1 - 0.05 * t[steep]
    img[steep, 3] = 1.0

    # LETHAL: dark red
    lethal = (~no_data) & (~obstacle) & (s > STEEP_MAX_DEG)
    img[lethal, 0] = 0.6
    img[lethal, 1] = 0.0
    img[lethal, 2] = 0.0
    img[lethal, 3] = 1.0

    # OBSTACLE: black
    img[obstacle, 0:3] = 0.0
    img[obstacle, 3] = 1.0

    # UNKNOWN: gray
    img[no_data, 0] = 0.5
    img[no_data, 1] = 0.5
    img[no_data, 2] = 0.5
    img[no_data, 3] = 0.4

    return img


def generate_grade_map(
    stl_path: str,
    output_path: str,
    cell_res: float = 1.0,
    dpi: int = 200,
):
    """Full pipeline: STL -> gradient map image."""

    # 1. Load mesh
    m = load_stl(stl_path)

    # 2. Compute per-face slope from normals
    print("Computing face slopes from normals...")
    face_slopes = compute_face_slopes(m)
    print(f"  Face slope stats: min={face_slopes.min():.1f}°  "
          f"median={np.median(face_slopes):.1f}°  max={face_slopes.max():.1f}°")

    # 3. Rasterize onto grid (area-weighted)
    slope_grid, obstacle_frac, has_data, x_min, y_min, grid_w, grid_h = rasterize_triangles(
        m, face_slopes, cell_res
    )

    valid_slope = slope_grid[~np.isnan(slope_grid)]
    print(f"  Grid slope stats: min={valid_slope.min():.1f}°  "
          f"median={np.median(valid_slope):.1f}°  max={valid_slope.max():.1f}°")

    # 4. Classify and colorize
    print("Mapping grade -> color...")
    img = grade_to_color(slope_grid, obstacle_frac)

    # 5. Count classifications
    no_data = np.isnan(slope_grid)
    obstacle = (obstacle_frac > 0.2) & ~no_data
    s = np.nan_to_num(slope_grid, nan=0.0)
    valid = ~no_data & ~obstacle
    n_free = np.sum(valid & (s <= FREE_MAX_DEG))
    n_mod = np.sum(valid & (s > FREE_MAX_DEG) & (s <= MODERATE_MAX_DEG))
    n_steep = np.sum(valid & (s > MODERATE_MAX_DEG) & (s <= STEEP_MAX_DEG))
    n_lethal = np.sum(valid & (s > STEEP_MAX_DEG))
    n_obs = np.sum(obstacle)
    n_unk = np.sum(no_data)
    total = grid_w * grid_h
    print(f"\n  Classification breakdown:")
    print(f"    FREE       (< {FREE_MAX_DEG}°):   {n_free:>8,}  ({100*n_free/total:5.1f}%)")
    print(f"    MODERATE   ({FREE_MAX_DEG}-{MODERATE_MAX_DEG}°): {n_mod:>8,}  ({100*n_mod/total:5.1f}%)")
    print(f"    STEEP      ({MODERATE_MAX_DEG}-{STEEP_MAX_DEG}°): {n_steep:>8,}  ({100*n_steep/total:5.1f}%)")
    print(f"    LETHAL     (> {STEEP_MAX_DEG}°):  {n_lethal:>8,}  ({100*n_lethal/total:5.1f}%)")
    print(f"    OBSTACLE   (z_spread):   {n_obs:>8,}  ({100*n_obs/total:5.1f}%)")
    print(f"    UNKNOWN    (no data):    {n_unk:>8,}  ({100*n_unk/total:5.1f}%)")

    # 6. Render figure
    print(f"\nRendering image to {output_path} ...")
    x_max = x_min + grid_w * cell_res
    y_max = y_min + grid_h * cell_res
    extent = [x_min, x_max, y_min, y_max]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [3, 1]})

    # Main gradient map
    ax = axes[0]
    ax.imshow(img, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Terrain Grade Map — Slope Classification", fontsize=14, fontweight="bold")

    # Legend panel
    ax2 = axes[1]
    ax2.axis("off")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.3, 0.85, 0.2), label=f"FREE (0-{FREE_MAX_DEG}°)"),
        Patch(facecolor=(0.95, 0.75, 0.1), label=f"MODERATE ({FREE_MAX_DEG}-{MODERATE_MAX_DEG}°)"),
        Patch(facecolor=(1.0, 0.4, 0.08), label=f"STEEP ({MODERATE_MAX_DEG}-{STEEP_MAX_DEG}°)"),
        Patch(facecolor=(0.6, 0.0, 0.0), label=f"LETHAL (>{STEEP_MAX_DEG}°)"),
        Patch(facecolor=(0.0, 0.0, 0.0), label="OBSTACLE (vertical)"),
        Patch(facecolor=(0.5, 0.5, 0.5, 0.4), label="UNKNOWN (no data)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=11,
               title="Grade Classification", title_fontsize=13,
               frameon=True, fancybox=True, shadow=True)

    stats_text = (
        f"Surface: {Path(stl_path).name}\n"
        f"Resolution: {cell_res} m/cell\n"
        f"Grid: {grid_w} x {grid_h}\n"
        f"Slope range: {valid_slope.min():.1f} - {valid_slope.max():.1f} deg\n"
        f"Median slope: {np.median(valid_slope):.1f} deg\n\n"
        f"Thresholds (from plan):\n"
        f"  Free:     < {FREE_MAX_DEG} deg\n"
        f"  Moderate: {FREE_MAX_DEG}-{MODERATE_MAX_DEG} deg\n"
        f"  Steep:    {MODERATE_MAX_DEG}-{STEEP_MAX_DEG} deg\n"
        f"  Lethal:   > {STEEP_MAX_DEG} deg\n"
        f"  Obstacle: vertical faces > 20%"
    )
    ax2.text(0.05, 0.35, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Done! Saved to: {output_path}")

    # Raw slope heatmap
    slope_img_path = str(Path(output_path).with_name("slope_raw.png"))
    fig2, ax3 = plt.subplots(1, 1, figsize=(10, 10))
    masked_slope = np.ma.masked_invalid(slope_grid)
    im = ax3.imshow(masked_slope, origin="lower", extent=extent, aspect="equal",
                    cmap="RdYlGn_r", vmin=0, vmax=30, interpolation="nearest")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Raw Slope Angle (degrees)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax3, label="Slope (deg)", shrink=0.8)
    fig2.savefig(slope_img_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Raw slope map saved to: {slope_img_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a terrain grade map from an STL surface mesh."
    )
    parser.add_argument(
        "--stl",
        default=str(Path(__file__).resolve().parent.parent / "3d_assets" / "Lidar_Surface.stl"),
        help="Path to the input STL file (default: ../3d_assets/Lidar_Surface.stl)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "grade_map.png"),
        help="Output image path (default: grade_map.png in this directory)",
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Grid cell resolution in meters (default: 1.0)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output image DPI (default: 200)",
    )
    args = parser.parse_args()
    generate_grade_map(stl_path=args.stl, output_path=args.output,
                       cell_res=args.resolution, dpi=args.dpi)


if __name__ == "__main__":
    main()
