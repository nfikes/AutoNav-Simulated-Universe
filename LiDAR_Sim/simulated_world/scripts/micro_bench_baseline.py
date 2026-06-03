#!/usr/bin/env python3
"""Micro-benchmark with the *pre-optimization* algorithm reconstructed.

Calls the original raw-point DBSCAN path + per-cell np.median spike
detection + duplicate R = rot_from_normal at the end. This is what
build_grade_costmap looked like before today's edits. Used to quantify
the speedup against today's vectorized version.
"""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.cluster import DBSCAN

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)

import lidar_sim_gui as sim  # noqa: E402
import trimesh  # noqa: E402


def build_grade_costmap_baseline(cloud, origin, sn):
    """Pre-optimization build_grade_costmap reconstructed.
    Same up through Step-4 PCA (already batched in the baseline) but
    uses the original raw-point DBSCAN + per-cell np.median spike loop.
    """
    res = sim.COSTMAP_RES; half = sim.COSTMAP_HALF
    gw = gh = int(2 * half / res); nc = gw * gh

    obs = np.zeros((gh, gw), dtype=bool)
    ms = np.full((gh, gw), np.nan)
    R = sim.rot_from_normal(sn)

    if len(cloud) > sim.MIN_PCA_POINTS:
        dx = cloud[:, 0] - origin[0]; dy = cloud[:, 1] - origin[1]
        px = ((dx + half) / res).astype(int); py = ((dy + half) / res).astype(int)
        valid = (px >= 0) & (px < gw) & (py >= 0) & (py < gh)
        vpts = cloud[valid]; vcx = px[valid]; vcy = py[valid]
        cell_idx_arr = vcy * gw + vcx
        sort_order = np.argsort(cell_idx_arr)
        sorted_idx = cell_idx_arr[sort_order]
        sorted_pts = vpts[sort_order]
        changes = np.where(np.diff(sorted_idx) != 0)[0] + 1
        splits = np.split(sorted_pts, changes)
        cell_keys = sorted_idx[np.concatenate([[0], changes])]
        cell_dict = {int(k): s for k, s in zip(cell_keys, splits)}

        ground_cell_dict = {}
        non_ground_list = []
        wall_detected = np.zeros((gh, gw), dtype=bool)
        own_cell_keys = list(cell_dict.keys())
        for k0 in own_cell_keys:
            cy = k0 // gw; cx = k0 % gw
            neighborhood = []
            for ddy in range(-1, 2):
                for ddx in range(-1, 2):
                    ny, nx = cy + ddy, cx + ddx
                    if 0 <= nx < gw and 0 <= ny < gh:
                        k = ny * gw + nx
                        arr = cell_dict.get(k)
                        if arr is not None:
                            neighborhood.append(arr)
            if not neighborhood:
                continue
            pts = neighborhood[0] if len(neighborhood) == 1 else np.concatenate(neighborhood)
            if len(pts) < sim.MIN_PCA_POINTS:
                continue
            zs = np.sort(pts[:, 2])
            z_cut = None
            if len(zs) > 1:
                for gi in range(len(zs) - 1):
                    if zs[gi + 1] - zs[gi] <= sim.Z_GROUND_BAND:
                        continue
                    candidate_cut = (zs[gi] + zs[gi + 1]) / 2
                    if zs[-1] - candidate_cut <= sim.WALL_MIN_HEIGHT:
                        continue
                    z_cut = candidate_cut
                    non_ground_list.append(pts[pts[:, 2] > z_cut])
                    own_pts_for_wall = cell_dict.get(cy * gw + cx, np.empty((0, 3)))
                    if len(own_pts_for_wall) > 0 and np.any(own_pts_for_wall[:, 2] > z_cut):
                        wall_detected[cy, cx] = True
                    pts = pts[pts[:, 2] <= z_cut]
                    break
            cell_idx = cy * gw + cx
            if cell_idx in cell_dict:
                own_arr = cell_dict[cell_idx]
                own_ground = own_arr[own_arr[:, 2] <= z_cut] if z_cut is not None else own_arr
                if len(own_ground) > 0:
                    ground_cell_dict[cell_idx] = own_ground

        if ground_cell_dict:
            keys = list(ground_cell_dict.keys())
            cell_yx = [(k // gw, k % gw) for k in keys]
            nbr_pts = []; nbr_yx = []
            for k, (cy, cx) in zip(keys, cell_yx):
                neighborhood = []
                for ddy in range(-1, 2):
                    for ddx in range(-1, 2):
                        ny, nx = cy + ddy, cx + ddx
                        if 0 <= nx < gw and 0 <= ny < gh:
                            nk = ny * gw + nx
                            arr = ground_cell_dict.get(nk)
                            if arr is not None:
                                neighborhood.append(arr)
                if not neighborhood:
                    continue
                pts = neighborhood[0] if len(neighborhood) == 1 else np.concatenate(neighborhood)
                if len(pts) < sim.MIN_PCA_POINTS:
                    continue
                nbr_pts.append(pts); nbr_yx.append((cy, cx))
            if nbr_pts:
                n_cells = len(nbr_pts)
                covs = np.empty((n_cells, 3, 3))
                pass_mask = np.zeros(n_cells, dtype=bool)
                for i, pts in enumerate(nbr_pts):
                    n = len(pts)
                    spreads = pts.max(axis=0) - pts.min(axis=0)
                    sorted_spreads = np.sort(spreads)
                    if sorted_spreads[1] < 0.10:
                        continue
                    centroid = pts.mean(axis=0)
                    centered = pts - centroid
                    covs[i] = (centered.T @ centered) / (n - 1)
                    pass_mask[i] = True
                if pass_mask.any():
                    valid_idx = np.where(pass_mask)[0]
                    eigvals_b, eigvecs_b = np.linalg.eigh(covs[valid_idx])
                    for vi, ci in enumerate(valid_idx):
                        eigvals = eigvals_b[vi]
                        eigvecs = eigvecs_b[vi]
                        if eigvals[1] < eigvals[2] * 0.01:
                            continue
                        if eigvals[2] > 1e-12 and (eigvals[0] / eigvals[2]) > sim.PCA_PLANARITY_MAX:
                            continue
                        normal = eigvecs[:, 0]
                        cos_angle = abs(normal[2]) / np.linalg.norm(normal)
                        cy, cx = nbr_yx[ci]
                        ms[cy, cx] = np.degrees(np.arccos(min(1.0, max(0.0, cos_angle))))

        # Pre-opt spike detection (per-cell np.median)
        vertical_obs = np.zeros((gh, gw), dtype=bool)
        for k, pts_arr in cell_dict.items():
            if len(pts_arr) < sim.SPIKE_MIN_ELEVATED:
                continue
            cy_v, cx_v = k // gw, k % gw
            if not np.isnan(ms[cy_v, cx_v]) and ms[cy_v, cx_v] <= sim.TRAVERSABLE_MAX_DEG + sim.PCA_NOISE_MARGIN_DEG:
                continue
            zs = np.sort(pts_arr[:, 2])
            n_ground = max(1, len(zs) // 3)
            ground_z = np.median(zs[:n_ground])
            elevated = pts_arr[pts_arr[:, 2] > ground_z + sim.SPIKE_HEIGHT]
            if len(elevated) >= sim.SPIKE_MIN_ELEVATED:
                vertical_obs[cy_v, cx_v] = True
                non_ground_list.append(elevated)

        obstacle_adjacent = binary_dilation(wall_detected | vertical_obs, iterations=2)

        # PRE-OPT: raw-point DBSCAN
        steep_thresh = sim.TRAVERSABLE_MAX_DEG + sim.PCA_NOISE_MARGIN_DEG
        steep_cells = (~np.isnan(ms) & (ms > steep_thresh)
                       & (ms < sim.PCA_MAX_VALID_DEG) & ~obstacle_adjacent)
        obstacle_candidates = list(non_ground_list)
        steep_ys, steep_xs = np.where(steep_cells)
        for sy, sx in zip(steep_ys, steep_xs):
            k = sy * gw + sx
            if k in ground_cell_dict:
                obstacle_candidates.append(ground_cell_dict[k])
        if obstacle_candidates:
            obs_arr = np.concatenate(obstacle_candidates)
        else:
            obs_arr = np.empty((0, 3))
        if len(obs_arr) >= sim.DBSCAN_MIN_SAMPLES:
            labels = DBSCAN(eps=sim.DBSCAN_EPS, min_samples=sim.DBSCAN_MIN_SAMPLES).fit_predict(obs_arr)
            unique_labels = set(labels)
            unique_labels.discard(-1)
            for lbl in unique_labels:
                cluster_mask = labels == lbl
                if cluster_mask.sum() < sim.MIN_CLUSTER_SIZE:
                    continue
                cluster_pts = obs_arr[cluster_mask]
                cdx = cluster_pts[:, 0] - origin[0]; cdy = cluster_pts[:, 1] - origin[1]
                cpx = ((cdx + half) / res).astype(int); cpy = ((cdy + half) / res).astype(int)
                cv = (cpx >= 0) & (cpx < gw) & (cpy >= 0) & (cpy < gh)
                obs[cpy[cv], cpx[cv]] = True
        obs |= wall_detected | vertical_obs

        bleed_zone = obstacle_adjacent & ~wall_detected & ~vertical_obs
        traversable = (~np.isnan(ms) & (ms <= steep_thresh) & ~vertical_obs) | bleed_zone
        obs = obs & ~traversable
        obs = obs | vertical_obs

    observed = np.zeros((gh, gw), dtype=bool)
    all_dx = cloud[:, 0] - origin[0]; all_dy = cloud[:, 1] - origin[1]
    all_px = ((all_dx + half) / res).astype(int); all_py = ((all_dy + half) / res).astype(int)
    all_v = (all_px >= 0) & (all_px < gw) & (all_py >= 0) & (all_py < gh)
    observed[all_py[all_v], all_px[all_v]] = True
    R = sim.rot_from_normal(sn)  # duplicate (matches pre-opt)
    return obs, ms, R, observed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=11520)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading mesh, generating {args.rays} rays...")
    mesh = trimesh.load(sim.STL_PATH)
    rays = sim.gen_rays(args.rays)

    positions = [
        (-2.0, -1.0, 0.0),
        (-0.5, -1.5, np.pi / 4),
        (0.75, -1.5, np.pi / 2),
        (2.25, -1.5, np.pi / 3),
        (3.75, -2.0, 0.5),
        (0.0, 1.5, np.pi),
        (1.0, 0.0, -np.pi / 6),
    ]
    sim.do_scan(mesh, 0, 0, rays[:64], 0.0)

    scan_times = []
    pipeline_times = []
    for it in range(args.iters):
        for (px, py, heading) in positions:
            t0 = time.perf_counter()
            cloud, origin, sn = sim.do_scan(mesh, px, py, rays, heading)
            t1 = time.perf_counter()
            if cloud is None:
                continue
            obs, ms, R, seen = build_grade_costmap_baseline(cloud, origin, sn)
            t2 = time.perf_counter()
            scan_times.append(t1 - t0)
            pipeline_times.append(t2 - t1)

    n = len(scan_times)
    print(f"\n  Samples: {n}")
    print(f"  do_scan         mean={np.mean(scan_times)*1000:7.2f} ms  "
          f"min={np.min(scan_times)*1000:7.2f} ms  "
          f"median={np.median(scan_times)*1000:7.2f} ms")
    print(f"  pca pipeline (BASELINE)  mean={np.mean(pipeline_times)*1000:7.2f} ms  "
          f"min={np.min(pipeline_times)*1000:7.2f} ms  "
          f"median={np.median(pipeline_times)*1000:7.2f} ms")


if __name__ == "__main__":
    main()
