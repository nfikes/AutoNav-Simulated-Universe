#!/usr/bin/env python3
"""
Dual-Robot Navigation: Grade-Based vs Height-Band

RED robot:  New grade-based slope detection
BLUE robot: Old height-band filter (min_obstacle_height from AutoNav repo)

Left:   Terrain + persistent obstacle overlay (red/blue/purple) + paths
Center: Both costmaps overlaid (red=grade, blue=height, purple=both)
Right:  3D point cloud from new system lidar

Click = place both | Shift+Click = goal | G = go
"""

import sys, heapq, time
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass  # Headless (e.g. multiprocessing worker): keep whatever backend is active
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, binary_dilation, median_filter
from sklearn.cluster import DBSCAN


# ── Config ──────────────────────────────────────────────────────────
STL_PATH = str(Path(__file__).resolve().parent.parent / "3d_assets" / "Lidar_Surface_fast.stl")
LIDAR_HEIGHT = 0.3
N_RAYS = 11520           # 720 pts/layer × 16 layers = 0.5° horizontal
                         # angular resolution per the SICK multiScan165
                         # datasheet (360° / 720 = 0.5°)
LIDAR_RANGE = 10.0       # configured operational range; hardware capable
                         # of 0.05-62 m but robot driver caps at 10 m
ROBOT_RADIUS = 0.15         # restored to the original working value
INFLATION_RADIUS = 0.15     # minimal inflation (decay zone width)
COST_SCALING = 5.0          # real cost_scaling_factor

COSTMAP_RES = 0.1           # local costmap cell size (matches real robot)
COSTMAP_HALF = 8.0          # local costmap extent
GLOBAL_RES = 0.05           # persistent global costmap resolution (400x400 grid)
GLOBAL_HALF = 10.0          # persistent global costmap extent

BEV_SIZE = 500
NAV_STEP = 0.03             # movement step size (m / animation frame). 0.03 × 15 FPS = 0.45 m/s.
SCAN_EVERY = 0.1            # scan trigger distance. 0.1 m means a scan fires every ~3 frames at NAV_STEP=0.03, giving ~5 Hz BEV/3D refresh while leaving most frames as cheap blits so the agent visibly glides at corner-to-corner distance. Setting this equal to NAV_STEP (scan every frame) is too aggressive on Windows matplotlib + PyQtGraph: heavy frames pile up at 80-100 ms each and the agent appears stuck.

# Thresholds for RELATIVE grade (angle between PCA normal and LiDAR normal).
# Measured in the sensor's own frame — no gravity reference needed while moving.
# From flat ground a ramp reads its true slope (first encounter marks the
# global costmap permanently).  Once on the ramp it reads ~0°, but the
# global costmap already has the obstacle from the first scan.
#
# 20% grade = arctan(0.20) = 11.3° — competition standard
GRADE_FREE_MAX     = 8.0    # 0-8°:   flat/gentle terrain (FREE)
GRADE_MODERATE_MAX = 11.3   # 8-11.3°: noticeable slope (penalized)
GRADE_STEEP_MAX    = 11.3   # 11.3° — obstacle boundary
TRAVERSABLE_MAX_DEG = 16.7  # matches traversable_max_deg in deployed grade_detector.yaml on AutoNav_25-26 (30% grade ≈ 2× competition ramp).
PCA_NOISE_MARGIN_DEG = 1.5  # Small noise headroom for genuine PCA jitter.
PCA_MAX_VALID_DEG    = 89.0 # matches pca_max_valid_deg in deployed
                            # grade_detector.yaml on AutoNav_25-26. The
                            # plan's 60.0 opened a dead zone (60–90°) where
                            # steep ramps fell through both the slope and
                            # wall detectors — the deployed code widens
                            # the valid PCA window to 89°. (Was 60.0:
                            # a terrain reading. Real ramps in this
                            # competition top out at ~25°. Cells whose PCA
                            # returns a near-vertical normal (~90°) are
                            # almost always contaminated by ramp edges or
                            # wall faces sneaking into the neighborhood.
                            # We measured 50% of the 5° ramp cells reading
                            # ~83-90° from a single scan — those are
                            # vertical-plane fits, not slope.  Drop them
                            # from the steep-cell set; let the wall/spike
                            # detectors handle real vertical features.
Z_GROUND_BAND = 0.1     # matches z_ground_band in deployed grade_detector.yaml on AutoNav_25-26 (NOT the plan doc — plan doc is stale, deployed YAML overrides it).
                        # If the first z-gap between sorted points in a PCA
                        # neighborhood exceeds this, split and keep only the
                        # lower cluster.  Inspired by vertical-neighbor
                        # segmentation (arxiv 2404.14281).
WALL_MIN_HEIGHT = 0.5   # matches wall_min_height in deployed grade_detector.yaml on AutoNav_25-26.
                        # split point to count as a wall/obstacle.  Ramp peaks
                        # (0.1-0.4m) are terrain; walls (2.17m) are obstacles.
DBSCAN_EPS = 0.3           # meters — max distance between points in a cluster
DBSCAN_MIN_SAMPLES = 3     # min points to form a core point
MIN_CLUSTER_SIZE = 10      # matches min_cluster_size in deployed grade_detector.yaml on AutoNav_25-26 (was 15 in the sim's synthetic-density era).

# Spike detection (cones, posts, poles — even tilted)
SPIKE_HEIGHT = 0.15        # min height above ground to flag as spike (m)
SPIKE_MIN_ELEVATED = 2     # min elevated points in cell to trigger

OLD_MIN_H = 0.4; OLD_MAX_H = 2.0


def _qt_key_to_mpl(event):
    """Translate a PyQt QKeyEvent into the lowercase string matplotlib
    uses for its ``key_press_event`` callbacks.

    Returns ``None`` for keys that don't have an obvious mapping; the
    GUI only cares about a couple of letters (G, R, P), so a thin
    translation is enough."""
    try:
        text = event.text()
    except Exception:
        return None
    if text and text.strip():
        return text.lower()
    return None


# ── Lidar ───────────────────────────────────────────────────────────

# Matches `front_arc_only` in the deployed grade_detector.yaml.
# When True, gen_rays generates rays only in the agent's forward 180°
# (azimuth in [-π/2, +π/2] relative to agent forward) and do_scan
# rotates them by the agent's heading. Halves the ray-cast workload
# AND matches what the production ROS node filters down to before
# feeding pca_pipeline.cpp.
FRONT_ARC_ONLY = True


def gen_rays(n, full_360=None):
    """Generate LiDAR ray directions for the SICK multiScan165 model.

    Real hardware: SICK multiScan165 (MULS1AA-114322), 16 layers, 360°.
    Datasheet native FOV: -7.5° (down) to +35° (up). Mounted upside-down
    on our robot, so in the robot/world frame the hardware sees:
      -35° (toward ground) to +7.5° (slightly above horizon)
    We trim the bottom of the FOV to -20° here: anything below that
    lands inside the robot's 0.8 m near-zone, which is already covered
    by the dedicated 15-ray surface_normal PCA estimator. Layers
    spread instead across the useful 0.8-10 m planning range.

    When ``full_360`` is None (the default), the module's
    ``FRONT_ARC_ONLY`` flag controls the azimuth range. With
    ``FRONT_ARC_ONLY=True`` the rays span only ``[-π/2, +π/2]`` in
    the agent's local frame; ``do_scan`` then rotates by the agent's
    heading so "forward" matches the real chassis orientation.
    """
    if full_360 is None:
        full_360 = not FRONT_ARC_ONLY
    layers = 16
    pts = n // layers
    dirs = []
    if full_360:
        az = np.linspace(0, 2 * np.pi, pts, endpoint=False)
    else:
        # Front 180° centered on +x (agent forward) in agent-local frame.
        az = np.linspace(-np.pi / 2, np.pi / 2, pts, endpoint=False)
    for ed in np.linspace(-20, 7.5, layers):
        e = np.radians(ed)
        dirs.append(np.column_stack([
            np.cos(e) * np.cos(az),
            np.cos(e) * np.sin(az),
            np.full(pts, np.sin(e)),
        ]))
    return np.vstack(dirs)

def surface_z(mesh,x,y):
    l,_,_=mesh.ray.intersects_location([[x,y,500]],[[0,0,-1]],multiple_hits=True)
    return float(l[:,2].max()) if len(l) else None

def surface_normal(mesh, x, y):
    """Estimate the LiDAR's chassis tilt by scattering 15 ground samples
    in a disk under the robot, then PCA-fitting the resulting points.

    Why PCA: a 3-point tripod cross-product is wildly sensitive to a
    single foot landing on a wall edge — that one outlier flips the
    normal by 30°+ and corrupts the whole scan. PCA on 15+ samples is
    naturally robust: one outlier barely moves the fitted plane.

    Why 15: enough to swamp a couple of wall-edge outliers, few enough
    that even Windows-native trimesh-without-embree finishes in <50ms.
    On a real robot this would come from the LiDAR's own near-ground
    rings; in sim we cast 15 vertical rays around the chassis."""
    sz = surface_z(mesh, x, y)
    if sz is None:
        return np.array([0, 0, 1.0])
    N = 15
    R = max(ROBOT_RADIUS, 0.5)  # disk radius (≥ 0.5m smooths small bumps)
    rng = np.random.default_rng(42)
    sub_r = R * np.sqrt(rng.random(N))     # uniform on disk
    sub_t = 2 * np.pi * rng.random(N)
    sxs = x + sub_r * np.cos(sub_t)
    sys_ = y + sub_r * np.sin(sub_t)
    origins = np.column_stack([sxs, sys_, np.full(N, 500.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (N, 1))
    locs, idx_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=True)
    pts = []
    for i in range(N):
        hits = locs[idx_ray == i]
        if not len(hits):
            continue
        # Only keep ground-level hits (within 0.15m of center) — drops
        # wall tops and platform undersides.
        cand = hits[(hits[:, 2] >= sz - 0.15) & (hits[:, 2] <= sz + 0.15)]
        if not len(cand):
            continue
        z_best = cand[np.argmin(np.abs(cand[:, 2] - sz)), 2]
        pts.append([sxs[i], sys_[i], z_best])
    if len(pts) < 3:
        return np.array([0, 0, 1.0])
    pts = np.asarray(pts)
    # PCA: smallest eigenvector = surface normal
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normal = eigvecs[:, 0]                   # smallest eigenvalue
    if normal[2] < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _surface_normal_legacy(mesh,x,y):
    """Old single-ray-per-foot tripod, kept for reference."""
    r = max(ROBOT_RADIUS, 0.5)
    center_z = surface_z(mesh, x, y)
    if center_z is None:
        return np.array([0,0,1.0])
    pts = []
    for ang in [0, 120, 240]:
        rad = np.radians(ang)
        px, py = x + r * np.cos(rad), y + r * np.sin(rad)
        locs,_,_ = mesh.ray.intersects_location(
            [[px, py, 500]], [[0,0,-1]], multiple_hits=True)
        if not len(locs):
            pts.append(np.array([px, py, center_z]))
            continue
        z_hits = locs[:, 2]
        candidates = z_hits[z_hits >= center_z - 0.1]
        if len(candidates) == 0:
            best = center_z
        else:
            best = candidates[np.argmin(np.abs(candidates - center_z))]
        # Reject if still >0.5m off from center
        if abs(best - center_z) > 0.5:
            best = center_z
        pts.append(np.array([px, py, best]))
    e1 = pts[1] - pts[0]
    e2 = pts[2] - pts[0]
    n = np.cross(e1, e2)
    if np.linalg.norm(n) < 1e-10:
        return np.array([0,0,1.0])
    if n[2] < 0: n = -n
    return n / np.linalg.norm(n)

def rot_from_normal(sn):
    up=np.array([0,0,1.0]); n=sn/np.linalg.norm(sn); d=up@n
    if abs(d-1)<1e-6: return np.eye(3)
    if abs(d+1)<1e-6: return -np.eye(3)
    ax=np.cross(up,n); ax/=np.linalg.norm(ax); a=np.arccos(np.clip(d,-1,1))
    # Clamp tilt to 45° max — prevents wild rotations from surface
    # imperfections or LiDAR landing on a wall face
    a = min(a, np.radians(45))
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)

def do_scan(mesh, px, py, rays, heading=0.0):
    """Simulate a LiDAR scan.  Returns points in the SENSOR (tilted) frame,
    just like a real LiDAR would.  The sensor frame is tilted to match the
    terrain surface normal at the robot's position.

    ``heading`` is the agent's yaw in radians (atan2 convention: 0 = +x,
    +π/2 = +y). When ``FRONT_ARC_ONLY`` is True the rays are pre-built in
    the agent's local frame and must be rotated by ``Rz(heading)`` so the
    forward arc aligns with the agent's actual orientation. Pass 0.0 when
    no heading is known (e.g. the warm-up scan at startup).

    Returns (cloud_sensor_frame, origin_world, surface_normal).
    The caller must apply gravity compensation (R transform) before analysis.
    """
    sz = surface_z(mesh, px, py)
    if sz is None: return None, None, None
    origin = np.array([px, py, sz + LIDAR_HEIGHT])
    sn = surface_normal(mesh, px, py)
    # Yaw rotation: aligns the front-arc rays with the agent's forward
    # direction before the surface-tilt R is applied. When heading == 0
    # this is just an identity and we skip the matmul.
    if heading != 0.0:
        ch, sh = np.cos(heading), np.sin(heading)
        Rz = np.array([[ch, -sh, 0.0],
                       [sh,  ch, 0.0],
                       [0.0, 0.0, 1.0]])
        rays = (Rz @ rays.T).T
    R = rot_from_normal(sn); d = (R @ rays.T).T
    locs,_,_=mesh.ray.intersects_location(np.tile(origin,(len(d),1)),d,multiple_hits=False)
    if not len(locs): return None,origin,sn
    hits_world = locs[np.linalg.norm(locs-origin,axis=1)<=LIDAR_RANGE]
    # Transform hits from world frame into the sensor's tilted frame.
    # This is what the real SICK multiScan would output in lidar_footprint frame.
    # R maps world->sensor-aligned, R.T maps world->sensor-tilted.
    hits_sensor = (R.T @ (hits_world - origin).T).T + origin
    return hits_sensor, origin, sn


# ── Costmap builders ────────────────────────────────────────────────

MIN_PCA_POINTS = 6  # minimum points in neighborhood for reliable PCA plane fit

PCA_PLANARITY_MAX = 0.02   # matches pca_planarity_max in deployed
                           # grade_detector.yaml on AutoNav_25-26. The
                           # deployed value is 4× more permissive than the
                           # 0.005 in the plan / older sim because real
                           # lidar noise on moderate-slope plates (~1.5 cm)
                           # was pushing the ratio over 0.005 and dropping
                           # cells out of the slope detector. 0.02 still
                           # rejects ramp+wall mixed cells (ratios ≈ 0.1)
                           # but admits tire-class curved obstacles, which
                           # land in the 0.005–0.02 band.


def _pca_slope_deg(points, ref_normal=np.array([0,0,1.0])):
    """Compute surface grade (degrees) from a set of 3D points via PCA.
    Grade is the angle between the fitted surface normal and ref_normal.
    Returns NaN if too few points or single-ring (one axis has no spread)."""
    n = len(points)
    if n < MIN_PCA_POINTS:
        return np.nan
    # Reject single-ring points: need spread in all 3 axes
    # for a well-conditioned plane fit
    spreads = points.max(axis=0) - points.min(axis=0)
    sorted_spreads = np.sort(spreads)
    # Need at least 10cm in the 2 smallest axes (not just 1 big axis)
    if sorted_spreads[1] < 0.10:
        return np.nan
    # Covariance matrix of the point set
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / (n - 1)
    # Eigendecompose — smallest eigenvalue's eigenvector is the surface normal
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigh: ascending order, symmetric
    # Reject 1D point clouds (line-like)
    if eigvals[1] < eigvals[2] * 0.01:
        return np.nan
    # Reject non-planar point clouds (multiple surfaces in the neighborhood
    # — typically a ramp surface + an adjacent side wall or floor edge).
    # Without this check those cells get a near-vertical PCA normal that
    # paints the ramp surface as a steep obstacle.
    if eigvals[2] > 1e-12 and (eigvals[0] / eigvals[2]) > PCA_PLANARITY_MAX:
        return np.nan
    normal = eigvecs[:, 0]  # column 0 = smallest eigenvalue = surface normal
    # Grade = angle between PCA normal and the reference normal
    cos_angle = abs(np.dot(normal, ref_normal)) / (np.linalg.norm(normal) * np.linalg.norm(ref_normal))
    return np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))


PCA_RES = 2.0  # radius for PCA point gathering (m)

def build_grade_costmap(cloud, origin, sn):
    """Processing pipeline for LiDAR-based terrain classification.

    Uses dual resolution: coarse grid (0.5m) for PCA to capture
    multiple LiDAR rings, fine grid (0.1m) for obstacle projection.

    NOTE on back-slope smear: while climbing a ramp at angle α, the
    back-slope reads as ~2α in the sensor's tilted frame. RULES.md #1
    forbids correcting this with a world-frame reference. The chosen
    mitigation is operational, not algorithmic: set the deployment
    grade threshold (TRAVERSABLE_MAX_DEG) to ~2× the steepest ramp the
    robot is expected to climb. With that headroom, the doubled
    back-slope reading stays below threshold and is classified as
    traversable. No ray dropout or per-scan filtering needed.
    """
    res=COSTMAP_RES; half=COSTMAP_HALF
    gw=gh=int(2*half/res); nc=gw*gh

    obs = np.zeros((gh,gw), dtype=bool)
    ms = np.full((gh,gw), np.nan)
    R = rot_from_normal(sn)

    if len(cloud) > MIN_PCA_POINTS:
        # ── Fine grid for point binning and ground segmentation ──
        dx = cloud[:,0]-origin[0]; dy = cloud[:,1]-origin[1]
        px = ((dx+half)/res).astype(int); py = ((dy+half)/res).astype(int)
        valid = (px>=0)&(px<gw)&(py>=0)&(py<gh)
        vpts = cloud[valid]; vcx = px[valid]; vcy = py[valid]
        cell_idx_arr = vcy * gw + vcx

        # Vectorized cell binning using numpy sorting
        sort_order = np.argsort(cell_idx_arr)
        sorted_idx = cell_idx_arr[sort_order]
        sorted_pts = vpts[sort_order]
        # Find boundaries between cells
        changes = np.where(np.diff(sorted_idx) != 0)[0] + 1
        splits = np.split(sorted_pts, changes)
        cell_keys = sorted_idx[np.concatenate([[0], changes])]
        # Build dict: cell_id -> points array
        cell_dict = {}
        for k, s in zip(cell_keys, splits):
            cell_dict[int(k)] = s

        has_points = np.zeros((gh, gw), dtype=bool)
        for k in cell_dict:
            has_points[k // gw, k % gw] = True
        has_neighbor = binary_dilation(has_points, iterations=1)

        # ── Step 3: Ground Segmentation ──────────────────────────
        ground_cell_dict = {}
        non_ground_list = []
        wall_detected = np.zeros((gh, gw), dtype=bool)

        # Iterate over cells that actually have own points. The original
        # version walked every active_cells (binary_dilation of
        # has_points), which includes cells that only have a NEIGHBOR
        # with points but no points of their own — those cells can never
        # land in ground_cell_dict (see the `if cell_idx in cell_dict`
        # check below) and so can never contribute to the ms grid. The
        # 3x3 neighborhood gather inside this loop already picks up
        # surrounding points; we just don't need to start the loop from
        # an empty center cell. (Range gate removed long ago — walls
        # between ramps were being missed.)
        # active_cells preserved for reference: np.argwhere(has_neighbor)
        own_cell_keys = list(cell_dict.keys())
        for k0 in own_cell_keys:
            cy = k0 // gw; cx = k0 % gw
            # 3x3 neighborhood (0.3 m). Matches the deployed C++ Step-2
            # gather in pca_pipeline.cpp on AutoNav_25-26 (the plan doc's
            # 5x5 mention is stale; deployed uses 3x3).
            neighborhood = []
            for ddy in range(-1, 2):
                for ddx in range(-1, 2):
                    ny, nx = cy+ddy, cx+ddx
                    if 0 <= nx < gw and 0 <= ny < gh:
                        k = ny*gw+nx
                        arr = cell_dict.get(k)
                        if arr is not None:
                            neighborhood.append(arr)
            if not neighborhood:
                continue
            pts = (neighborhood[0] if len(neighborhood) == 1
                   else np.concatenate(neighborhood))
            if len(pts) < MIN_PCA_POINTS:
                continue

            # Step-2 split — walk gaps from low z up. Real lidar layers
            # create discrete elevation rings; on a tilted surface those
            # rings show as z-gaps even though the surface is continuous.
            # Splitting on a ring-gap artifact shreds tilted-plate point
            # sets into 1D slices that PCA can't classify. So skip gaps
            # whose upper cluster span is < WALL_MIN_HEIGHT (ring-gap)
            # and only split at the first one that looks like a real wall.
            # Mirrors pca_pipeline.cpp Step 2 in AutoNav_25-26.
            zs = np.sort(pts[:, 2])
            z_cut = None
            if len(zs) > 1:
                for gi in range(len(zs) - 1):
                    if zs[gi + 1] - zs[gi] <= Z_GROUND_BAND:
                        continue
                    candidate_cut = (zs[gi] + zs[gi + 1]) / 2
                    if zs[-1] - candidate_cut <= WALL_MIN_HEIGHT:
                        # Ring-gap artifact, not a wall — keep walking.
                        continue
                    z_cut = candidate_cut
                    non_ground_list.append(pts[pts[:, 2] > z_cut])
                    # Only flag THIS cell as wall when its OWN points
                    # contribute to the upper cluster.
                    own_pts_for_wall = cell_dict.get(cy * gw + cx,
                                                      np.empty((0, 3)))
                    if len(own_pts_for_wall) > 0 and \
                            np.any(own_pts_for_wall[:, 2] > z_cut):
                        wall_detected[cy, cx] = True
                    pts = pts[pts[:, 2] <= z_cut]
                    break

            # Store ground points for this cell
            cell_idx = cy*gw + cx
            if cell_idx in cell_dict:
                own_arr = cell_dict[cell_idx]
                if z_cut is not None:
                    own_ground = own_arr[own_arr[:, 2] <= z_cut]
                else:
                    own_ground = own_arr
                if len(own_ground) > 0:
                    ground_cell_dict[cell_idx] = own_ground

        # ── Step 4: PCA on ground points (3x3 neighborhood) ────
        # Batched: collect every cell's 3x3-neighborhood points first,
        # then compute centroid + covariance + eigh in a single batched
        # numpy call instead of one per cell. Same semantics as the
        # per-cell _pca_slope_deg, just vectorized across cells.
        if ground_cell_dict:
            keys = list(ground_cell_dict.keys())
            cell_yx = [(k // gw, k % gw) for k in keys]
            # Build neighborhood point sets (small Python loop — 9 dict
            # lookups per cell, then one concatenate). Same logic as
            # before; only the per-cell PCA is batched.
            nbr_pts = []
            nbr_keys = []
            nbr_yx = []
            for k, (cy, cx) in zip(keys, cell_yx):
                # 3x3 neighborhood (0.3 m). Matches deployed C++ Step-3
                # gather in pca_pipeline.cpp on AutoNav_25-26.
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
                pts = (neighborhood[0] if len(neighborhood) == 1
                       else np.concatenate(neighborhood))
                if len(pts) < MIN_PCA_POINTS:
                    continue
                nbr_pts.append(pts)
                nbr_keys.append(k)
                nbr_yx.append((cy, cx))

            # Cheap pre-filter (matches _pca_slope_deg lines 257-260):
            # need >=10cm spread in the 2 smallest axes; otherwise NaN.
            if nbr_pts:
                # Pre-compute centroid + covariance per neighborhood
                # cell. We keep this as a Python loop over neighborhoods
                # (each tiny, <100 points), but the eigh step is batched.
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
                    # Batched eigh — single C call for every cell that
                    # passed the cheap pre-filter.
                    valid_idx = np.where(pass_mask)[0]
                    eigvals_b, eigvecs_b = np.linalg.eigh(covs[valid_idx])
                    # Per-cell validation: line / non-planar rejection.
                    for vi, ci in enumerate(valid_idx):
                        eigvals = eigvals_b[vi]
                        eigvecs = eigvecs_b[vi]
                        if eigvals[1] < eigvals[2] * 0.01:
                            continue
                        if (eigvals[2] > 1e-12
                                and (eigvals[0] / eigvals[2]) > PCA_PLANARITY_MAX):
                            continue
                        normal = eigvecs[:, 0]
                        # ref_normal = [0, 0, 1] so cos = |normal[2]| (norm == 1).
                        cos_angle = abs(normal[2]) / np.linalg.norm(normal)
                        cy, cx = nbr_yx[ci]
                        ms[cy, cx] = np.degrees(np.arccos(min(1.0, max(0.0, cos_angle))))

        # No median filter — PCA grades are used directly. The
        # margin-based steep/traversable split below absorbs the noise.

        # ── Step 4b: Spike Detection (cones, posts, poles) ──────
        # AGGRESSIVE FILTER: only fire when (1) PCA cannot classify the
        # cell as traversable, (2) the cell is NOT next to an existing
        # wall (those flagged points are leakage, not spikes), AND
        # (3) the elevated points genuinely dominate the cell — most
        # of the cell's points are elevated AND the cell's ground floor
        # is itself elevated (the cell IS sitting on the obstacle, not
        # collecting stray side-rays from a tall feature elsewhere).
        #
        # Vectorized: we used to call np.median per cell (~10k calls/scan
        # at 11520 rays). The same scalar result drops out of the
        # per-cell sorted z-array's index ``(n_ground // 3) // 2`` —
        # see pca_pipeline.cpp:321 in AutoNav_25-26
        # (`ground_z = zs[n_ground / 2]`). Note the C++ uses integer
        # division which is what np.median of an odd-count slice does;
        # for even counts numpy's median averages two values while the
        # C++ picks the upper one. To stay byte-for-byte compatible with
        # the previous Python the simulator preserves np.median exactly.
        vertical_obs = np.zeros((gh, gw), dtype=bool)
        traverse_thr = TRAVERSABLE_MAX_DEG + PCA_NOISE_MARGIN_DEG
        for k, pts_arr in cell_dict.items():
            n_pts = len(pts_arr)
            if n_pts < SPIKE_MIN_ELEVATED:
                continue
            cy_v, cx_v = k // gw, k % gw
            # Skip cells where PCA found a (noise-margin-aware) traversable surface
            ms_v = ms[cy_v, cx_v]
            if not np.isnan(ms_v) and ms_v <= traverse_thr:
                continue
            # Ground z = median of bottom 30% (robust to outliers).
            # Hand-rolled median is much faster than np.median for tiny
            # arrays (no _ureduce dispatch overhead — np.median is ~14μs
            # per call but the underlying selection is sub-μs). The
            # value returned is identical to ``np.median(np.sort(zs)[:n_ground])``
            # — bit-exact to the previous implementation.
            zs_col = pts_arr[:, 2]
            n_ground = max(1, n_pts // 3)
            if n_ground >= n_pts:
                zs_low = np.sort(zs_col)
            else:
                # partition pivots the array so indices < n_ground are
                # the smallest n_ground values (unsorted). Sort that
                # sub-slice so we can index its median directly.
                part = np.partition(zs_col, n_ground - 1)
                zs_low = np.sort(part[:n_ground])
            mid = n_ground // 2
            if n_ground % 2 == 1:
                ground_z = zs_low[mid]
            else:
                ground_z = 0.5 * (zs_low[mid - 1] + zs_low[mid])
            # Count points spiking above ground
            elevated_mask = zs_col > ground_z + SPIKE_HEIGHT
            if int(elevated_mask.sum()) >= SPIKE_MIN_ELEVATED:
                vertical_obs[cy_v, cx_v] = True
                non_ground_list.append(pts_arr[elevated_mask])

        # Wall-adjacent and spike-adjacent masks: PCA is unreliable
        # near walls and spikes due to point bleeding
        obstacle_adjacent = binary_dilation(wall_detected | vertical_obs, iterations=2)

        # ── Step 5: voxel-downsampled DBSCAN on non-traversable points ─
        # Matches the deployed C++ in pca_pipeline.cpp:373+ (AutoNav_25-26).
        # The raw candidate set can be 1-2k points in a populated room
        # and DBSCAN is O(n²). Snap each candidate to its cell on the
        # algorithm's existing grid (COSTMAP_RES m), then represent the
        # cell by one centroid. DBSCAN runs on the centroids — typically
        # O(hundreds) unique cells — for a large speedup with no fidelity
        # loss vs the C++ reference. Each centroid carries a "mass" (raw
        # point count) so the original min_cluster_size semantics
        # (raw point count) are preserved.
        #
        # NOTE: this is *closer* to the deployed C++ than the previous
        # raw-point DBSCAN. The hashes change; the on-robot behavior does
        # not. See task notes.
        steep_thresh = TRAVERSABLE_MAX_DEG + PCA_NOISE_MARGIN_DEG
        steep_cells = (~np.isnan(ms) & (ms > steep_thresh)
                       & (ms < PCA_MAX_VALID_DEG) & ~obstacle_adjacent)
        obstacle_candidates = list(non_ground_list)
        steep_ys, steep_xs = np.where(steep_cells)
        for sy, sx in zip(steep_ys, steep_xs):
            k = sy*gw+sx
            if k in ground_cell_dict:
                obstacle_candidates.append(ground_cell_dict[k])

        if obstacle_candidates:
            obs_arr = np.concatenate(obstacle_candidates)
        else:
            obs_arr = np.empty((0, 3))

        if len(obs_arr) >= DBSCAN_MIN_SAMPLES:
            # Voxel bin candidate points into cells on the algorithm's
            # own grid. Mirrors CellAccum struct in pca_pipeline.cpp.
            cand_dx = obs_arr[:, 0] - origin[0]
            cand_dy = obs_arr[:, 1] - origin[1]
            cand_cx = ((cand_dx + half) / res).astype(np.int64)
            cand_cy = ((cand_dy + half) / res).astype(np.int64)
            in_bounds = (
                (cand_cx >= 0) & (cand_cx < gw)
                & (cand_cy >= 0) & (cand_cy < gh))
            obs_arr_v = obs_arr[in_bounds]
            cand_keys = (cand_cy[in_bounds] * gw + cand_cx[in_bounds])
            if len(cand_keys) >= DBSCAN_MIN_SAMPLES:
                # Sort by cell key, then split — equivalent to the
                # unordered_map<int, CellAccum> sum/count in C++.
                ord2 = np.argsort(cand_keys, kind="stable")
                k_sorted = cand_keys[ord2]
                p_sorted = obs_arr_v[ord2]
                boundaries = np.where(np.diff(k_sorted) != 0)[0] + 1
                cell_starts = np.concatenate(
                    [[0], boundaries, [len(k_sorted)]])
                n_cells_ds = len(cell_starts) - 1
                ds_centroids = np.empty((n_cells_ds, 3), dtype=np.float64)
                ds_mass = np.empty(n_cells_ds, dtype=np.int64)
                ds_keys = np.empty(n_cells_ds, dtype=np.int64)
                for i in range(n_cells_ds):
                    a, b = cell_starts[i], cell_starts[i + 1]
                    seg = p_sorted[a:b]
                    ds_centroids[i] = seg.mean(axis=0)
                    ds_mass[i] = b - a
                    ds_keys[i] = k_sorted[a]

                if n_cells_ds >= DBSCAN_MIN_SAMPLES:
                    labels = DBSCAN(
                        eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MIN_SAMPLES,
                    ).fit_predict(ds_centroids)
                    # Cluster mass = sum of voxel raw counts (C++ line 489).
                    valid = labels >= 0
                    if valid.any():
                        # Accumulate mass per label, decide pass/fail, mark cells.
                        unique = np.unique(labels[valid])
                        for lbl in unique:
                            mem = labels == lbl
                            if int(ds_mass[mem].sum()) < MIN_CLUSTER_SIZE:
                                continue
                            keep_keys = ds_keys[mem]
                            cy_arr = keep_keys // gw
                            cx_arr = keep_keys % gw
                            obs[cy_arr, cx_arr] = True
        # Always mark spike and wall cells regardless of DBSCAN
        obs |= wall_detected | vertical_obs

        # ── PCA override: protect traversable surfaces ───────────
        # Clear obstacle flag where PCA confirms traversable.
        # Also clear "bleed zone" — cells adjacent to obstacles whose
        # PCA was contaminated by nearby wall/spike points.
        bleed_zone = obstacle_adjacent & ~wall_detected & ~vertical_obs
        # Use the same hi-threshold here so cells in the noise band
        # (between TRAVERSABLE_MAX_DEG and steep_thresh) get cleared,
        # not left ambiguous.
        traversable = (~np.isnan(ms) & (ms <= steep_thresh)
                       & ~vertical_obs) | bleed_zone
        obs = obs & ~traversable
        # Re-mark spike cells (must stay as obstacles)
        obs = obs | vertical_obs

    # Observed = any cell that had at least one point
    observed = np.zeros((gh,gw), dtype=bool)
    all_dx = cloud[:,0]-origin[0]; all_dy = cloud[:,1]-origin[1]
    all_px = ((all_dx+half)/res).astype(int); all_py = ((all_dy+half)/res).astype(int)
    all_v = (all_px>=0)&(all_px<gw)&(all_py>=0)&(all_py<gh)
    observed[all_py[all_v], all_px[all_v]] = True
    # R was already computed at the top of the function (line ~358);
    # the trailing duplicate rot_from_normal(sn) call was a leftover.
    return obs, ms, R, observed

def build_height_costmap(cloud,origin,sn):
    """Old height-band obstacle detection.  Cloud arrives in sensor (tilted)
    frame — the height band applies directly in this frame, which is how
    the real ObstacleLayer works (no gravity compensation)."""
    R=rot_from_normal(sn); res=COSTMAP_RES; half=COSTMAP_HALF
    gw=gh=int(2*half/res)
    # Cloud is already in sensor frame (do_scan returns sensor coords).
    # Height band applies in the robot's local frame = sensor frame.
    local=cloud-origin
    z_robot=local[:,2]; is_obs=(z_robot>-LIDAR_HEIGHT+OLD_MIN_H)&(z_robot<-LIDAR_HEIGHT+OLD_MAX_H)
    px=((local[:,0]+half)/res).astype(int); py=((local[:,1]+half)/res).astype(int)
    v=(px>=0)&(px<gw)&(py>=0)&(py<gh)
    obs=np.zeros((gh,gw),dtype=bool)
    for i in np.where(v&is_obs)[0]: obs[py[i],px[i]]=True
    # Observed = any cell with a point
    observed = np.zeros((gh,gw), dtype=bool)
    observed[py[v], px[v]] = True
    return obs, R, observed


# ── Persistent global costmap ──────────────────────────────────────

class GlobalCostmap:
    # CONFIRM_THRESHOLD = 1 → single-scan hits become obstacles
    # immediately (matches the reference behavior). Streak filtering
    # happens at visualization time via multi-agent consensus.
    CONFIRM_THRESHOLD = 1

    def __init__(self):
        self.res=GLOBAL_RES; self.half=GLOBAL_HALF
        self.w=int(2*self.half/self.res); self.h=self.w
        self.obstacles=np.zeros((self.h,self.w),dtype=bool)
        self.obs_hits=np.zeros((self.h,self.w),dtype=np.int16)
        self.inflated=np.zeros((self.h,self.w),dtype=np.uint8) # 0=free, 254=lethal, 1-253=cost
        self.robot_cells=int(ROBOT_RADIUS/self.res)+1
        self.infl_cells=int((ROBOT_RADIUS+INFLATION_RADIUS)/self.res)+1
        # Cached obstacle-count snapshot for _inflate. distance_transform_edt
        # over a 400x400 grid is ~3ms but called every scan; skip when the
        # obstacle mask hasn't changed since the last inflation.
        self._inflated_obs_count = -1

    def add_platform_bounds(self):
        """No-op: Rule 3 — no map boundary walls. Agents detect the map
        edge via no LiDAR returns (platform drop-off) instead of
        artificial costmap walls."""
        pass

    def w2c(self,wx,wy):
        return (np.clip(int((wx+self.half)/self.res),0,self.w-1),
                np.clip(int((wy+self.half)/self.res),0,self.h-1))

    def c2w(self,cx,cy):
        return (cx+0.5)*self.res-self.half, (cy+0.5)*self.res-self.half

    def merge_local(self, local_obs, origin, R, observed=None, slope_grid=None):
        """Merge local obstacles into global costmap.
        Clears nearby cells first so current scan replaces stale data
        (prevents ramp-tilt artifacts from persisting on flat ground)."""
        gh,gw=local_obs.shape; half_l=COSTMAP_HALF; res_l=COSTMAP_RES

        # Obstacles persist — no clearing. PCA override in
        # build_grade_costmap handles traversable cell protection.

        oys, oxs = np.where(local_obs)
        if len(oys) == 0:
            self._inflate()
            return
        # Original always-2x2-fill projection: each 0.1m local cell
        # becomes a 2x2 block in the 0.05m global grid (closes gaps
        # between adjacent obstacle cells without leaving streaks).
        lxs = (oxs + 0.5) * res_l - half_l
        lys = (oys + 0.5) * res_l - half_l
        local_pts = np.column_stack([lxs, lys, np.zeros(len(lxs))])
        world_pts = (R @ local_pts.T).T + origin
        cxs = np.clip(((world_pts[:,0]+self.half)/self.res).astype(int), 0, self.w-1)
        cys = np.clip(((world_pts[:,1]+self.half)/self.res).astype(int), 0, self.w-1)
        self.obstacles[cys, cxs] = True
        for dx, dy in [(1,0),(0,1),(1,1)]:
            cxf = np.clip(cxs+dx, 0, self.w-1)
            cyf = np.clip(cys+dy, 0, self.h-1)
            self.obstacles[cyf, cxf] = True
        self._inflate()

    def _inflate(self):
        """NAV2-style inflation with cost_scaling_factor.

        Skip recomputation when the obstacle mask is unchanged since the
        last inflation. Obstacles only ever grow (no clearing), so a
        simple count comparison is a sound staleness check and avoids
        running distance_transform_edt on a 400x400 grid every scan.
        """
        obs_count = int(self.obstacles.sum())
        if obs_count == self._inflated_obs_count:
            return
        self.inflated[:]=0
        self.inflated[self.obstacles]=254
        if obs_count == 0:
            self._inflated_obs_count = 0
            return
        dist=distance_transform_edt(~self.obstacles)*self.res  # in meters
        inscribed=(dist<=ROBOT_RADIUS)&(dist>0)
        self.inflated[inscribed]=253
        decay_zone=(dist>ROBOT_RADIUS)&(dist<ROBOT_RADIUS+INFLATION_RADIUS)
        if decay_zone.any():
            costs=(252*np.exp(-COST_SCALING*(dist[decay_zone]-ROBOT_RADIUS))).astype(np.uint8)
            self.inflated[decay_zone]=np.maximum(self.inflated[decay_zone],costs)
        self._inflated_obs_count = obs_count

    def is_free(self,cx,cy):
        if cx<0 or cx>=self.w or cy<0 or cy>=self.h: return False
        return self.inflated[cy,cx]<253

    def render_overlay(self, size=None, view_half=None, other=None):
        """RGBA overlay with a smooth inflation-halo fade.

        ``size`` / ``view_half`` are ignored — the output is always
        ``(self.h, self.w)`` pixels covering ``±self.half`` meters
        (one pixel per global costmap cell). Callers should
        ``set_extent([-self.half, self.half, -self.half, self.half])``
        on the matplotlib ``imshow`` that consumes this.

        Each cell within ``ROBOT_RADIUS + INFLATION_RADIUS`` of a real
        obstacle gets a red tint whose alpha fades linearly from full
        at the inscribed band to 0 at the inflation edge — the same
        envelope ``GlobalCostmap._inflate`` uses for path planning,
        and the same look the BEV panel used in robot-frame.
        """
        h, w = self.h, self.w
        img = np.zeros((h, w, 4), dtype=np.uint8)
        s_obs = self.obstacles
        o_obs = other.obstacles if other else None
        any_obs = s_obs if o_obs is None else (s_obs | o_obs)
        if not any_obs.any():
            return img
        # Distance from each global cell to its nearest obstacle (in cells).
        cell_dist = distance_transform_edt(~any_obs)
        infl_cells_outer = (ROBOT_RADIUS + INFLATION_RADIUS) / self.res
        infl_cells_inner = ROBOT_RADIUS / self.res
        in_halo = cell_dist <= infl_cells_outer
        denom = max(infl_cells_outer - infl_cells_inner, 1e-3)
        weight = np.clip(
            (infl_cells_outer - cell_dist) / denom, 0.0, 1.0
        ).astype(np.float32)
        alpha = (weight * in_halo).astype(np.float32)
        # Per-cell tint: take the colour of the *nearest* obstacle when
        # ``other`` is set so combined sims still get red/blue/purple.
        if o_obs is None:
            tint_r = np.full((h, w), 220, dtype=np.uint8)
            tint_g = np.full((h, w), 30, dtype=np.uint8)
            tint_b = np.full((h, w), 30, dtype=np.uint8)
        else:
            _, (iy, ix) = distance_transform_edt(~any_obs, return_indices=True)
            near_s = s_obs[iy, ix]
            near_o = o_obs[iy, ix]
            near_both = near_s & near_o
            tint_r = np.zeros((h, w), dtype=np.uint8)
            tint_g = np.zeros((h, w), dtype=np.uint8)
            tint_b = np.zeros((h, w), dtype=np.uint8)
            tint_r[near_s & ~near_o] = 220
            tint_g[near_s & ~near_o] = 30
            tint_b[near_s & ~near_o] = 30
            tint_r[near_o & ~near_s] = 30
            tint_g[near_o & ~near_s] = 80
            tint_b[near_o & ~near_s] = 220
            tint_r[near_both] = 160
            tint_g[near_both] = 0
            tint_b[near_both] = 200
        img[..., 0] = tint_r
        img[..., 1] = tint_g
        img[..., 2] = tint_b
        img[..., 3] = (alpha * 220).astype(np.uint8)
        return img


# ── A* on global costmap ───────────────────────────────────────────

def _closest_path_index_after(pts, start_i, x, y, window):
    """Find the index of the closest point in ``pts`` to (x, y), searching
    only ``[start_i, start_i + window]``. Path indices advance monotonically
    once the agent has passed a point, so a forward-only search beats an
    O(N) global scan and avoids "lock-onto-the-loop-back" failures when a
    replanned path crosses itself."""
    if not pts or start_i >= len(pts):
        return max(0, len(pts) - 1)
    end = min(len(pts), start_i + window)
    best_i = start_i
    best_d2 = float('inf')
    for i in range(start_i, end):
        dx = pts[i][0] - x
        dy = pts[i][1] - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _look_ahead_point(pts, start_i, x, y, distance, goal_fallback):
    """Walk ``pts`` forward from ``start_i`` until cumulative path
    distance ≥ ``distance``, return that point as the pure-pursuit
    carrot. Falls back to the last path point (or the goal) if the
    path runs out before the look-ahead is reached."""
    if not pts:
        return goal_fallback if goal_fallback is not None else (x, y)
    if start_i >= len(pts) - 1:
        return pts[-1]
    cum = 0.0
    j = start_i
    while j + 1 < len(pts) and cum < distance:
        cum += float(np.hypot(pts[j + 1][0] - pts[j][0],
                              pts[j + 1][1] - pts[j][1]))
        j += 1
    return pts[j]


def _astar_raw(costmap, sx, sy, gx, gy):
    """Raw A* on the global costmap. Returns unsmoothed grid path.

    Traversal rule: only LETHAL (254) is hard-blocked. INSCRIBED cells
    (253) are traversable but carry a HEAVY cost penalty (≈50 per
    inscribed step vs ≈1 for a free step), so A* will only thread
    through inscribed cells when there is no free alternative — most
    importantly, when the start itself is inside a wall's inflation
    halo (otherwise A* refuses to expand and the agent permanently
    STUCKs at the wall edge). With this scheme, normal navigation paths
    stay clear of walls, but stuck agents can plan an escape."""
    scx,scy=costmap.w2c(sx,sy); gcx,gcy=costmap.w2c(gx,gy)
    def is_lethal(cx,cy):
        if cx<0 or cx>=costmap.w or cy<0 or cy>=costmap.h: return True
        return costmap.inflated[cy,cx] >= 254
    if is_lethal(scx,scy) or is_lethal(gcx,gcy): return None
    open_set=[(0,scy,scx)]; came={}; gs={(scy,scx):0}; closed=set()
    nbrs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    bc=[1.414,1,1.414,1,1,1.414,1,1.414]
    while open_set:
        _,cy,cx=heapq.heappop(open_set)
        if (cy,cx) in closed: continue
        closed.add((cy,cx))
        if cy==gcy and cx==gcx:
            path=[];p=(cy,cx)
            while p in came: path.append(costmap.c2w(p[1],p[0])); p=came[p]
            path.append(costmap.c2w(p[1],p[0])); return path[::-1]
        for (dy,dx),c in zip(nbrs,bc):
            ny,nx=cy+dy,cx+dx
            if (ny,nx) in closed: continue
            if is_lethal(nx,ny): continue
            # Diagonal corner check: don't cut a corner across two
            # lethal neighbors (would mean shaving a wall).
            if dy != 0 and dx != 0:
                if is_lethal(cx+dx, cy) or is_lethal(cx, cy+dy):
                    continue
            # Cost penalty: free=0, decay-zone proportional, inscribed=50.
            # 50 vs base step cost 1 means A* will only path through
            # inscribed when there is literally no free-cell route.
            extra = 0
            if 0<=ny<costmap.h and 0<=nx<costmap.w:
                v = costmap.inflated[ny,nx]
                if v >= 253:
                    extra = 50.0
                elif v > 0:
                    extra = v / 50.0
            ng=gs[(cy,cx)]+c+extra
            if (ny,nx) not in gs or ng<gs[(ny,nx)]:
                gs[(ny,nx)]=ng
                # Weight heuristic 1.5x to favor shorter paths
                h=1.5*np.sqrt((ny-gcy)**2+(nx-gcx)**2)
                heapq.heappush(open_set,(ng+h,ny,nx))
                came[(ny,nx)]=(cy,cx)
    return None


def plan_global(costmap, sx, sy, gx, gy):
    """A* + line-of-sight smoothing."""
    raw = _astar_raw(costmap, sx, sy, gx, gy)
    if raw is None:
        return None
    return smooth_path(costmap, raw)


def _line_of_sight(costmap, x0, y0, x1, y1):
    """Check if the straight line between two world points is free of LETHAL cells.
    Uses Bresenham-style ray marching on the costmap grid."""
    cx0, cy0 = costmap.w2c(x0, y0)
    cx1, cy1 = costmap.w2c(x1, y1)
    dx = abs(cx1 - cx0); dy = abs(cy1 - cy0)
    sx = 1 if cx0 < cx1 else -1
    sy = 1 if cy0 < cy1 else -1
    err = dx - dy
    while True:
        if cx0 < 0 or cx0 >= costmap.w or cy0 < 0 or cy0 >= costmap.h:
            return False
        if costmap.inflated[cy0, cx0] >= 254:
            return False
        if cx0 == cx1 and cy0 == cy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; cx0 += sx
        if e2 < dx:
            err += dx; cy0 += sy
    return True


def smooth_path(costmap, path):
    """Line-of-sight path pruning — removes grid-aligned zigzags by
    skipping intermediate waypoints when a direct line is obstacle-free.
    Uses bounded search (max 50 waypoints ahead) for performance."""
    if not path or len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        # Search up to 50 waypoints ahead (furthest first)
        end = min(len(path) - 1, i + 50)
        for j in range(end, i + 1, -1):
            if _line_of_sight(costmap, path[i][0], path[i][1],
                              path[j][0], path[j][1]):
                best = j
                break
        smoothed.append(path[best])
        i = best
    return smoothed


# ── BEV rendering ──────────────────────────────────────────────────

def _global_obs_in_robot_frame(global_costmap, robot_xy):
    """Project the persistent global obstacle grid into a robot-centered
    (gh, gw) bool array at ``COSTMAP_RES`` resolution, ±``COSTMAP_HALF``
    meters. Used to feed ``render_bev`` so the middle panel shows every
    obstacle ever detected — not just the latest local scan.

    Cell-index math mirrors ``GlobalCostmap.w2c``:
        ``cell = int((world + half) / res)``
    Each output cell samples one global cell via nearest-neighbor lookup.
    """
    gw = gh = int(2 * COSTMAP_HALF / COSTMAP_RES)
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    # Lower-left world corner of the robot-frame window
    wx0 = rx - COSTMAP_HALF
    wy0 = ry - COSTMAP_HALF
    i = np.arange(gw, dtype=np.float32)
    j = np.arange(gh, dtype=np.float32)
    wx_row = wx0 + i * COSTMAP_RES   # x of cell i's lower-left, shape (gw,)
    wy_col = wy0 + j * COSTMAP_RES   # y of cell j's lower-left, shape (gh,)
    g_cx = ((wx_row + global_costmap.half) / global_costmap.res).astype(np.int32)
    g_cy = ((wy_col + global_costmap.half) / global_costmap.res).astype(np.int32)
    g_cy_g, g_cx_g = np.meshgrid(g_cy, g_cx, indexing="ij")
    valid = ((g_cx_g >= 0) & (g_cx_g < global_costmap.w) &
             (g_cy_g >= 0) & (g_cy_g < global_costmap.h))
    out = np.zeros((gh, gw), dtype=bool)
    out[valid] = global_costmap.obstacles[g_cy_g[valid], g_cx_g[valid]]
    return out


def render_bev(new_obs, old_obs):
    """Both costmaps overlaid with an inflation halo that fades from
    full red at the obstacle cell out to white at the inflation edge.

    The halo width is ``ROBOT_RADIUS + INFLATION_RADIUS`` (meters)
    measured in cell units — the same envelope ``GlobalCostmap._inflate``
    uses for path planning. Cells inside the inscribed (robot-radius)
    band stay at full red; the decay band from inscribed → outer edge
    fades linearly to the white background. Same idea applies to the
    blue / purple channels.
    """
    gh, gw = new_obs.shape
    size = BEV_SIZE
    half = COSTMAP_HALF
    bg = np.array([240, 240, 240], dtype=np.float32)
    img = np.tile(bg, (size, size, 1))

    # ── Inflation halo ──
    # Distance from every grid cell to the nearest obstacle (any colour),
    # in cell units. scipy returns 0 for cells that ARE obstacles.
    any_obs = new_obs | old_obs
    if any_obs.any():
        cell_dist = distance_transform_edt(~any_obs)  # cells
        infl_cells_outer = (ROBOT_RADIUS + INFLATION_RADIUS) / COSTMAP_RES
        infl_cells_inner = ROBOT_RADIUS / COSTMAP_RES
        # Linear fade: 1.0 at cells inside the inscribed band,
        # → 0.0 at the outer edge of the inflation halo.
        in_halo = cell_dist <= infl_cells_outer
        denom = max(infl_cells_outer - infl_cells_inner, 1e-3)
        weight = np.clip(
            (infl_cells_outer - cell_dist) / denom, 0.0, 1.0
        ).astype(np.float32)
        # Tint by the dominant nearest-obstacle channel. Cheap proxy:
        # take the new/old/both classification of the nearest cell via
        # a separate distance-transform-with-indices lookup.
        _, (iy, ix) = distance_transform_edt(~any_obs, return_indices=True)
        nearest_new = new_obs[iy, ix]
        nearest_old = old_obs[iy, ix]
        nearest_both = nearest_new & nearest_old
        # Per-cell tint at GRID resolution, then upscale to BEV pixels.
        tint_cells = np.zeros((gh, gw, 3), dtype=np.float32)
        tint_cells[nearest_new & ~nearest_old] = (220, 30, 30)
        tint_cells[nearest_old & ~nearest_new] = (30, 80, 220)
        tint_cells[nearest_both]               = (160, 0, 200)
        alpha_cells = (weight * in_halo).astype(np.float32)
        # Upscale via Kronecker product (nearest-neighbor).
        rep_y = max(1, size // gh)
        rep_x = max(1, size // gw)
        alpha = np.kron(alpha_cells, np.ones((rep_y, rep_x), dtype=np.float32))
        tint_up = np.kron(tint_cells, np.ones((rep_y, rep_x, 1), dtype=np.float32))
        # Trim / pad to exactly (size, size).
        alpha = alpha[:size, :size]
        tint_up = tint_up[:size, :size]
        if alpha.shape[0] < size or alpha.shape[1] < size:
            pad_y = size - alpha.shape[0]
            pad_x = size - alpha.shape[1]
            alpha = np.pad(alpha, ((0, pad_y), (0, pad_x)))
            tint_up = np.pad(tint_up, ((0, pad_y), (0, pad_x), (0, 0)))
        # Image-y axis is flipped vs costmap-y; mirror to match.
        alpha = alpha[::-1]
        tint_up = tint_up[::-1]
        a3 = alpha[..., None]
        img = img * (1.0 - a3) + tint_up * a3

    # ── Origin marker (yellow square) ──
    cx, cy = size // 2, size // 2
    img[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = (255, 255, 0)
    return img.astype(np.uint8)


# ── GUI ────────────────────────────────────────────────────────────

class DualNavGUI:
    def __init__(self, stl_path, bake_mp4=None, bake_fps=60, bake_step=None):
        print("Loading mesh...")
        self.mesh=trimesh.load(stl_path)
        print(f"  {len(self.mesh.faces):,} faces")
        self.rays=gen_rays(N_RAYS)
        print("Building terrain..."); self._build_terrain()
        print("Init costmaps...")
        self.new_global=GlobalCostmap()
        self.new_global.add_platform_bounds()
        self.old_global=GlobalCostmap()
        self.old_global.add_platform_bounds()
        print("Warming up..."); _=do_scan(self.mesh,0,0,self.rays[:10])
        print("Ready!")
        self.robot_pos=None; self.goal_pos=None; self.animating=False
        self.new_path=None; self.old_path=None
        self.new_obs=None; self.old_obs=None
        # Default start behind walls, goal past ramp 2
        self.default_start=(-1, -3.5)
        self.default_goal=(-0.5, 4.5)
        # Offscreen-bake parameters. When ``bake_mp4`` is set the GUI skips
        # ``app.exec_()`` after the initial scan and instead runs
        # ``_bake_to_mp4`` to write a high-FPS MP4 of the agent traversing
        # the planned path.
        self.bake_mp4 = bake_mp4
        self.bake_fps = bake_fps
        self.bake_step = bake_step
        self._setup_gui()

    def _build_terrain(self):
        vh=8; self.terrain_extent=[-vh,vh,-vh,vh]
        cache_path=Path(STL_PATH).with_suffix('.terrain.png')
        stl_mtime=Path(STL_PATH).stat().st_mtime
        # Load cached terrain if STL hasn't changed
        if cache_path.exists() and cache_path.stat().st_mtime >= stl_mtime:
            print("  Loading cached terrain...")
            from PIL import Image
            img=np.array(Image.open(cache_path)).astype(np.float64)/65535.0
            self.terrain_img=img
            return
        print("  Ray-casting elevation (0.025m, caching as PNG)...")
        res=0.025; x=np.arange(-vh,vh,res); y=np.arange(-vh,vh,res)
        gw,gh=len(x),len(y)
        origins=[]; dirs=[]
        for iy in range(gh):
            for ix in range(gw):
                origins.append([x[ix],y[iy],500]); dirs.append([0,0,-1])
        origins=np.array(origins); dirs=np.array(dirs)
        locs,idx_ray,_=self.mesh.ray.intersects_location(origins,dirs,multiple_hits=True)
        elev=np.full((gh,gw),np.nan)
        for i in range(len(origins)):
            hits=locs[idx_ray==i]
            if len(hits): elev[i//gw,i%gw]=hits[:,2].max()
        m=np.isnan(elev)
        if m.any(): _,ni=distance_transform_edt(m,return_distances=True,return_indices=True); elev=elev[tuple(ni)]
        dzdx=np.gradient(elev,res,axis=1); dzdy=np.gradient(elev,res,axis=0)
        nm=np.sqrt(dzdx**2+dzdy**2+1)
        la,le=np.radians(315),np.radians(25)
        shade=np.clip(-dzdx/nm*np.cos(le)*np.cos(la)-dzdy/nm*np.cos(le)*np.sin(la)+1/nm*np.sin(le),0,1)
        self.terrain_img=0.15+0.85*shade
        # Cache as 16-bit PNG for full precision
        from PIL import Image
        img16=(self.terrain_img*65535).astype(np.uint16)
        Image.fromarray(img16).save(str(cache_path))
        print(f"  Cached terrain to {cache_path.name}")

    def _setup_gui(self):
        # The single-agent GUI uses a Qt window that embeds:
        #   * a matplotlib FigureCanvasQTAgg with ax_map + ax_bev (left + middle)
        #   * a PyQtGraph GLViewWidget for the 3D point cloud (right)
        # The matplotlib 3D panel was the dominant cost on heavy "scan-tick"
        # frames (Path3DCollection rebuild ~70 ms); PyQtGraph drops that to
        # a single VBO upload (single-digit ms).
        #
        # The multi-agent ramp benchmark still uses the mplot3d code path
        # via sim_view.build_three_panel_figure, so it's unaffected.
        import sim_view
        from PyQt5 import QtCore, QtWidgets

        handles = sim_view.build_three_panel_qt_widget(
            title="Grade vs Height-Band Navigation",
            terrain_img=self.terrain_img,
            terrain_extent=self.terrain_extent,
            # Translucent STL ghost in the 3D panel so the user can see
            # the terrain the agent is traversing. Alpha = 0.25.
            ghost_mesh=self.mesh,
        )
        self.container = handles["widget"]
        self.fig = handles["fig"]
        self.canvas = handles["canvas"]
        self.ax_map = handles["ax_map"]
        self.ax_bev = handles["ax_bev"]
        self.three_d = handles["three_d"]
        # ``ax3d`` kept as an alias for any legacy code path that still
        # references it — None because the 3D panel is no longer mpl.
        self.ax3d = None
        self.global_img = handles["global_img"]
        self.bev_handle = handles["bev_handle"]
        self.status = handles["status"]
        self.status.set_text("Click to place robots")

        # GUI-specific overlay on the shared map axes.
        self.ax_map.set_title("Click=Place | Shift+Click=Goal | G=Go",
                              color="white", fontsize=9)
        self.new_mk, = self.ax_map.plot(
            [], [], "ro", markersize=10, label="New (grade)", zorder=10)
        self.old_mk, = self.ax_map.plot(
            [], [], "bs", markersize=10, label="Old (height)", zorder=10)
        # Hide the blue (height-band) agent and path so only the red
        # PCA-based grade agent is visible. The internal old_global
        # costmap still updates in the background but isn't drawn.
        self.old_mk.set_visible(False)
        self.goal_mk, = self.ax_map.plot(
            [], [], "g*", markersize=18, label="Goal", zorder=10)
        self.new_path_line, = self.ax_map.plot(
            [], [], "r-", linewidth=2, alpha=0.8, zorder=8)
        self.old_path_line, = self.ax_map.plot(
            [], [], "b--", linewidth=2, alpha=0.8, zorder=8)
        self.old_path_line.set_visible(False)
        # Show only the visible markers (red agent + green goal) in the legend.
        self.ax_map.legend(handles=[self.new_mk, self.goal_mk],
                            loc="upper left", fontsize=7,
                            facecolor="#333", labelcolor="white",
                            edgecolor="#555")

        # 3D state — kept for the (now-legacy) mplot3d path. Unused on
        # the GL fast path but harmless to leave in place.
        self._3d_state = sim_view.ThreeDState()

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("key_press_event", self._on_key)
        # Click + drag inside the map re-positions the agent and
        # re-scans on the fly so the user can sweep the cursor and
        # watch the BEV / 3D update live.
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)

        # Forward Qt key presses on the GL view to the matplotlib
        # canvas's key_press_event handler so the G hotkey fires
        # regardless of which child widget has focus.
        def _gl_key_press(event, _self=self):
            key = _qt_key_to_mpl(event)
            if key is None:
                return
            class _E:
                pass
            e = _E()
            e.key = key
            _self._on_key(e)
        self.three_d.widget.keyPressEvent = _gl_key_press

        # Auto-place behind walls and set goal past ramp 2
        sx, sy = self.default_start
        gx, gy = self.default_goal
        self.robot_pos = (sx, sy)
        self.goal_pos = (gx, gy)
        self.new_mk.set_data([sx], [sy]); self.old_mk.set_data([sx], [sy])
        self.goal_mk.set_data([gx], [gy])
        self._do_scan(sx, sy)
        self._plan_paths()
        # In bake mode, append a brief "(bake)" tag to the status that
        # _plan_paths just set — the user already knows they're in bake
        # mode (they clicked the button); G fires the offscreen bake the
        # moment they're satisfied with the agent / goal placement.
        if self.bake_mp4:
            self.status.set_text(
                self.status.get_text() + f"  ·  G = bake @ {self.bake_fps} fps")
        self.canvas.draw_idle()

        # Show the container and enter the Qt event loop. ``plt.show``
        # would only show the mpl canvas's stand-alone manager window,
        # which would hide the GL view entirely.
        self.container.resize(1600, 700)
        self.container.show()
        QtWidgets.QApplication.instance().exec_()

    def _update_3d(self, cloud, origin, sn, slope_grid):
        """Update 3D point cloud panel via the shared sim_view helper.

        The single-agent GUI now routes through the PyQtGraph
        ``ThreeDView`` (single VBO upload per scan tick); the benchmark
        still hands ``update_3d_panel`` a matplotlib Axes3D and gets the
        legacy mplot3d path. Both branches share the per-point coloring
        logic via ``_classify_cloud_colors``.
        """
        import sim_view
        sim_view.update_3d_panel(
            self.three_d, self._3d_state,
            cloud, origin, sn, slope_grid, self.new_obs)

    def _heading_to_goal(self, px, py):
        """Direction (radians) from (px,py) toward the goal, 0 if unset."""
        if self.goal_pos is None:
            return 0.0
        return float(np.arctan2(self.goal_pos[1] - py,
                                self.goal_pos[0] - px))

    def _heading_along(self, px, py, pts, i, look_ahead_m=0.3):
        """Direction the agent is heading at this scan instant. Looks
        ``look_ahead_m`` metres ahead along the planned path so the
        heading averages over A* grid staircases — a true diagonal in
        the costmap shows up as alternating east / north single-cell
        steps, and aiming at the very next waypoint would flip the
        heading every cell. With FRONT_ARC_ONLY=True the lidar rays
        are yaw-rotated by this heading; a flipping heading would
        rotate the front arc 90° every cell-crossing and read as the
        cloud "flipping back and forth" on the 3D panel.
        """
        if pts and i + 1 < len(pts):
            cum = 0.0
            j = i
            while j + 1 < len(pts) and cum < look_ahead_m:
                cum += float(np.hypot(pts[j + 1][0] - pts[j][0],
                                       pts[j + 1][1] - pts[j][1]))
                j += 1
            tx, ty = pts[j]
        elif self.goal_pos is not None:
            tx, ty = self.goal_pos
        else:
            return 0.0
        dx, dy = tx - px, ty - py
        if dx * dx + dy * dy < 1e-6:
            return 0.0
        return float(np.arctan2(dy, dx))

    def _do_scan(self, px, py):
        heading = self._heading_to_goal(px, py)
        cloud_sensor,origin,sn=do_scan(self.mesh,px,py,self.rays,heading)
        if cloud_sensor is None or len(cloud_sensor)<3: return

        # Build local PCA grade costmap. Old height-band costmap is
        # skipped — we only show the red one.
        self.new_obs,new_slope,R_new,new_seen=build_grade_costmap(cloud_sensor,origin,sn)
        self.old_obs = np.zeros_like(self.new_obs)
        self.last_R=R_new; self.last_origin=origin
        cloud=cloud_sensor

        # Merge into persistent global costmap (only the new one)
        self.new_global.merge_local(self.new_obs,origin,R_new,new_seen,)

        # Update map overlay (now also draws the inflation halo —
        # render_overlay was upgraded to absorb the old BEV's job).
        overlay = self.new_global.render_overlay()
        self.global_img.set_data(overlay)
        self.global_img.set_extent(
            [-GLOBAL_HALF, GLOBAL_HALF, -GLOBAL_HALF, GLOBAL_HALF])

        self._update_3d(cloud, origin, sn, new_slope)

        n_new=self.new_obs.sum(); n_old=self.old_obs.sum()
        self.status.set_text(f"({px:.0f},{py:.0f}) | New: {n_new} obs | Old: {n_old} obs | Global: {self.new_global.obstacles.sum()}")

    def _plan_paths(self):
        if not self.robot_pos or not self.goal_pos: return
        rx,ry=self.robot_pos; gx,gy=self.goal_pos
        # Plan on PERSISTENT global costmaps — raw A* (no LOS smoothing)
        self.new_path=_astar_raw(self.new_global,rx,ry,gx,gy)
        self.old_path=_astar_raw(self.old_global,rx,ry,gx,gy)
        if self.new_path:
            self.new_path_line.set_data([p[0] for p in self.new_path],[p[1] for p in self.new_path])
        else: self.new_path_line.set_data([],[])
        if self.old_path:
            self.old_path_line.set_data([p[0] for p in self.old_path],[p[1] for p in self.old_path])
        else: self.old_path_line.set_data([],[])
        ns=f"{len(self.new_path)} steps" if self.new_path else "NO PATH"
        os=f"{len(self.old_path)} steps" if self.old_path else "NO PATH"
        self.status.set_text(f"New: {ns} | Old: {os} — press G")

    def _on_click(self,event):
        if event.inaxes!=self.ax_map or event.xdata is None: return
        mx,my=event.xdata,event.ydata
        if hasattr(event,'key') and event.key=='shift':
            self.goal_pos=(mx,my); self.goal_mk.set_data([mx],[my])
            self._plan_paths(); self.canvas.draw_idle(); return
        self.robot_pos=(mx,my)
        self.new_mk.set_data([mx],[my]); self.old_mk.set_data([mx],[my])
        self._do_scan(mx,my)
        if self.goal_pos: self._plan_paths()
        self.canvas.draw_idle()

    def _on_drag(self, event):
        """Left-mouse drag inside the map = move the agent + re-scan
        live, so the user can sweep the cursor around and see the
        BEV + 3D point cloud update to whatever the agent would see
        from that position. Throttled so we don't fire a 11 520-ray
        scan on every pixel of mouse motion.

        Skipped when an animation is running (would fight the
        ``_animate`` loop) or when Shift is held (the click handler
        treats Shift as a goal placement)."""
        if self.animating:
            return
        if event.inaxes != self.ax_map or event.xdata is None:
            return
        if event.button != 1:
            return
        if event.key == 'shift':
            return
        # Time throttle: at most ~8 scans per second of dragging.
        now = time.perf_counter()
        if now - getattr(self, '_last_drag_t', 0.0) < 0.12:
            return
        # Distance throttle: re-scan only after the cursor moved
        # noticeably in world coords (avoid burning a scan on sub-cell
        # jitter from mouse tracking).
        if self.robot_pos is not None:
            dx = event.xdata - self.robot_pos[0]
            dy = event.ydata - self.robot_pos[1]
            if dx * dx + dy * dy < 0.04:   # < 0.2 m moved
                return
        self._last_drag_t = now
        mx, my = event.xdata, event.ydata
        self.robot_pos = (mx, my)
        self.new_mk.set_data([mx], [my])
        self.old_mk.set_data([mx], [my])
        self._do_scan(mx, my)
        if self.goal_pos:
            self._plan_paths()
        self.canvas.draw_idle()

    def _on_key(self,event):
        if event.key=='g' and not self.animating:
            if not self.robot_pos:
                self.status.set_text("Click to place the robot first.")
                self.canvas.draw_idle()
                return
            if not self.goal_pos:
                self.status.set_text(
                    "Shift+Click a goal first, then press G to animate.")
                self.canvas.draw_idle()
                return
            if not self.new_path and not self.old_path:
                # Goal exists but A* couldn't find a route from the
                # current robot position. Re-plan once before bailing
                # — sometimes the costmap changed since last click.
                self._plan_paths()
                if not self.new_path and not self.old_path:
                    self.status.set_text(
                        "No path to goal — A* failed. Click a clearer "
                        "robot start or move the goal.")
                    self.canvas.draw_idle()
                    return
            if self.bake_mp4:
                # Bake-mode dispatch: capture the agent traversing the
                # current plan to MP4 instead of running the live
                # animation, then quit so the launcher shell returns.
                self.animating = True
                try:
                    self._bake_to_mp4(self.bake_mp4, self.bake_fps,
                                      step=self.bake_step)
                finally:
                    self.animating = False
                from PyQt5 import QtWidgets
                QtWidgets.QApplication.instance().quit()
                return
            self._animate()

    def _resample(self,path,step):
        if not path or len(path)<2: return path or []
        pts=[path[0]]
        for i in range(1,len(path)):
            dx=path[i][0]-path[i-1][0]; dy=path[i][1]-path[i-1][1]; d=np.sqrt(dx**2+dy**2)
            if d<0.01: continue
            n=max(1,int(d/step))
            for j in range(1,n+1): t=j/n; pts.append((path[i-1][0]+t*dx,path[i-1][1]+t*dy))
        return pts

    def _animate(self):
        """Both robots move and re-scan/re-plan during navigation.

        Rendering strategy (30 FPS target):
          * Map panel uses matplotlib blit. The terrain imshow + global
            costmap overlay form the static background; markers + path
            lines are flagged ``animated=True`` and drawn on top via
            ``draw_artist`` + ``canvas.blit(ax_map.bbox)``.
          * Scan-tick work is split into two tiers:
              - Tier A (algorithm-critical): ``do_scan`` + costmap merge
                + A* replan. Runs on EVERY scan tick (every SCAN_EVERY
                meters) so the agent's plan adapts to fresh obstacles.
              - Tier B (visual-only): BEV panel ``set_data`` + 3D scatter
                refresh + full canvas redraw + background re-snapshot.
                These cost ~50-80 ms together. Run them only every Nth
                scan tick so motion stays smooth between heavy frames.
                ``VISUAL_EVERY_N_SCANS`` controls the ratio.
          * Between heavy frames every Tier-A scan tick still updates the
            global costmap (merged into ``new_global``); the visual
            overlay simply lags by up to ``N-1`` scan ticks (~0.4 m of
            travel at SCAN_EVERY=0.2). The agent's planned path is
            re-computed every tick and reflected immediately because the
            path line is an animated artist drawn via blit.
          * Frames are paced to ~15 FPS via ``time.sleep`` so idle frames
            don't burn CPU.

        Algorithm output (do_scan / build_grade_costmap / merging /
        path planning) is untouched.
        """
        # Visual refresh cadence: BEV + 3D + global overlay redraw fires
        # every Nth scan tick. N=1 = every scan tick = BEV + 3D refresh
        # at SCAN_EVERY = 0.2 m of travel. The PyQtGraph GL 3D path keeps
        # heavy frames under the 15 FPS / 66.7 ms budget so there's no
        # need to throttle — every scan can paint a fresh view.
        VISUAL_EVERY_N_SCANS = 1
        if not self.new_path and not self.old_path: return
        self.animating = True

        # Keep BEV + 3D updated on scan ticks (not every frame — full
        # 3D scatter redraw is the dominant cost). _do_scan handles the
        # heavy panels; we just call it on scan-tick instead of inlining
        # the cheaper-only path.

        new_pos = list(self.robot_pos) if self.robot_pos else None
        old_pos = list(self.robot_pos) if self.robot_pos else None
        new_pts = self._resample(self.new_path, NAV_STEP) if self.new_path else []
        old_pts = self._resample(self.old_path, NAV_STEP) if self.old_path else []
        new_i = 0; old_i = 0
        new_done = not new_pts; old_done = not old_pts
        new_dist = SCAN_EVERY; old_dist = SCAN_EVERY  # force scan on first move
        # Count scan ticks (Tier A events) so we can gate the heavy visual
        # refresh (Tier B) to every Nth tick. The first scan tick always
        # produces a visual refresh so the user sees the initial BEV/3D
        # state right away.
        scan_tick_count = 0

        # ---- Blit setup -------------------------------------------------
        # Mark the dynamic artists as animated so a full canvas draw
        # excludes them from the cached background. ``flush_events``
        # below still picks up any visibility/data changes on these
        # artists via the per-frame draw_artist calls.
        animated_artists = [self.new_mk, self.old_mk, self.goal_mk,
                            self.new_path_line, self.old_path_line]
        for art in animated_artists:
            art.set_animated(True)

        # The global obstacle overlay is *also* animated now — it used
        # to live in the static ax_map_bg snapshot, which forced
        # ``_heavy_render`` to do a full ``canvas.draw()`` whenever the
        # overlay changed (~107 ms / heavy frame). Making it animated
        # lets us stamp the latest overlay on top of the cached
        # background via ``draw_artist`` on every frame (idle + heavy)
        # — no full canvas redraw required.
        self.global_img.set_animated(True)
        # BEV was merged into the map; if a stale ``bev_handle`` exists
        # from a legacy build, still mark it animated so a stray full
        # canvas draw doesn't dirty the bg cache.
        if self.bev_handle is not None:
            self.bev_handle.set_animated(True)

        canvas = self.canvas
        # 15 FPS budget (66.7 ms / frame). Heavy frames are now ~30 ms
        # (two blits + minor set_data + the ~30 ms algorithm), well
        # under the 66.7 ms budget — every scan tick can refresh all
        # three panels without overshooting. VISUAL_EVERY_N_SCANS is
        # set to 1 below to match.
        target_dt = 1.0 / 15.0
        frame_times = []
        anim_t0 = time.perf_counter()
        # Track ``done`` transitions so we only fire a heavy render on
        # the frame a robot first arrives — not on every frame thereafter
        # while the other robot is still moving.
        prev_new_done = new_done
        prev_old_done = old_done

        def _full_draw_and_snapshot():
            """Force a full canvas re-render, then snapshot the static
            map background. Only called for the up-front setup —
            heavy frames blit instead of redrawing."""
            self.ax_map.grid(False, which="both")
            canvas.draw()
            return canvas.copy_from_bbox(self.ax_map.bbox)

        # Snapshot once up-front so even the very first idle frame has
        # something to restore. If blit isn't supported on this backend
        # for any reason (older Agg builds, weird managers, etc.) fall
        # back to the legacy draw_idle path.
        use_blit = True
        try:
            ax_map_bg = _full_draw_and_snapshot()
        except Exception:
            use_blit = False
            for art in animated_artists:
                art.set_animated(False)
            self.global_img.set_animated(False)
            if self.bev_handle is not None:
                self.bev_handle.set_animated(False)
            ax_map_bg = None

        def _blit_idle_frame():
            """Restore the saved map background and stamp the animated
            artists on top. Cheap — no BEV / 3D / overlay work, but
            does paint the persistent ``global_img`` overlay so newly
            merged obstacle cells appear even on Tier-A-only frames."""
            canvas.restore_region(ax_map_bg)
            if self.global_img.get_visible():
                self.ax_map.draw_artist(self.global_img)
            for art in animated_artists:
                if art.get_visible():
                    self.ax_map.draw_artist(art)
            canvas.blit(self.ax_map.bbox)
            canvas.flush_events()

        def _heavy_render():
            """Blit-only heavy refresh.

            Repaints both the map (global overlay + markers + paths)
            and the BEV axes via ``draw_artist`` + ``canvas.blit``.
            No full ``canvas.draw()`` — the static backgrounds were
            captured once at startup and stay valid for the run.
            ``global_img`` is animated, so its data is *not* baked
            into the cached background; the latest ``set_data(...)``
            is stamped on top each frame."""
            self.ax_map.grid(False, which="both")
            overlay = self.new_global.render_overlay()
            self.global_img.set_data(overlay)

            ns = "ARRIVED" if new_done else (
                f"({new_pts[min(new_i, len(new_pts) - 1)][0]:.0f},"
                f"{new_pts[min(new_i, len(new_pts) - 1)][1]:.0f})"
                if new_pts else "STUCK")
            os_ = "ARRIVED" if old_done else (
                f"({old_pts[min(old_i, len(old_pts) - 1)][0]:.0f},"
                f"{old_pts[min(old_i, len(old_pts) - 1)][1]:.0f})"
                if old_pts else "STUCK")
            self.status.set_text(f"New: {ns} | Old: {os_}")

            if use_blit:
                try:
                    # Map axes blit (overlay + markers + paths). The BEV
                    # panel was merged into the map; the inflation halo
                    # now lives inside ``global_img`` and is drawn here.
                    canvas.restore_region(ax_map_bg)
                    if self.global_img.get_visible():
                        self.ax_map.draw_artist(self.global_img)
                    for art in animated_artists:
                        if art.get_visible():
                            self.ax_map.draw_artist(art)
                    canvas.blit(self.ax_map.bbox)
                except Exception:
                    canvas.draw_idle()
            else:
                canvas.draw_idle()

        for frame in range(5000):  # generous upper bound
            t_frame_start = time.perf_counter()
            scan_happened = False
            # Advance new robot
            if not new_done and new_i < len(new_pts):
                nx, ny = new_pts[new_i]
                # Refuse to step into INSCRIBED or LETHAL cells — force re-scan
                ncx, ncy = self.new_global.w2c(nx, ny)
                if self.new_global.inflated[ncy, ncx] >= 253:
                    new_dist = SCAN_EVERY  # force re-scan
                    new_i = len(new_pts)   # invalidate path
                    continue
                if new_i > 0:
                    new_dist += np.sqrt((new_pts[new_i][0]-new_pts[new_i-1][0])**2 +
                                        (new_pts[new_i][1]-new_pts[new_i-1][1])**2)
                self.new_mk.set_data([nx], [ny])
                self.new_path_line.set_data([p[0] for p in new_pts[new_i:]],
                                            [p[1] for p in new_pts[new_i:]])

                # Re-scan and re-plan with path hysteresis
                if new_dist >= SCAN_EVERY:
                    new_dist = 0
                    cs, origin, sn = do_scan(self.mesh, nx, ny, self.rays,
                                              self._heading_along(nx, ny, new_pts, new_i))
                    if cs is not None and len(cs) > 3:
                        # ---- Tier A (always-on, algorithm-critical) ----
                        no, new_slope, Rn, ns = build_grade_costmap(cs, origin, sn)
                        # _update_3d colors points by self.new_obs — keep
                        # it in sync with the latest local scan so the 3D
                        # cloud's red/green matches the cells the planner
                        # just merged into the global costmap.
                        self.new_obs = no
                        self.new_global.merge_local(no, origin, Rn, ns)
                        self.last_R = Rn; self.last_origin = origin
                        scan_tick_count += 1
                        # Tier B (visual refresh) only fires every Nth
                        # scan tick — every other Tier-A tick still merges
                        # the costmap and re-plans the path, but skips the
                        # ~50-80 ms BEV / 3D / canvas redraw so the agent
                        # keeps moving smoothly. The first scan tick after
                        # animation start always fires Tier B so the panels
                        # update immediately.
                        visual_refresh_due = (
                            scan_tick_count == 1 or
                            (scan_tick_count % VISUAL_EVERY_N_SCANS) == 0)
                        if visual_refresh_due:
                            # BEV was merged into the map — only the 3D
                            # panel needs explicit refresh now. The map's
                            # global_img overlay (set via render_overlay
                            # in _heavy_render) carries the inflation halo.
                            self._update_3d(cs, origin, sn, new_slope)
                        # ``scan_happened`` triggers a heavy render at the
                        # bottom of the frame ONLY when Tier B is due;
                        # otherwise the frame falls through to a cheap
                        # blit-idle render so motion stays smooth.
                        scan_happened = visual_refresh_due
                        # Always adopt new path — fast response to new obstacles
                        new_path = _astar_raw(self.new_global, nx, ny,
                                               self.goal_pos[0], self.goal_pos[1])
                        if new_path and len(new_path) > 1:
                            new_pts = self._resample(new_path, NAV_STEP)
                            new_i = 0
                            self.new_path_line.set_data([p[0] for p in new_pts],
                                                        [p[1] for p in new_pts])
                            # Inline-render only when Tier B is due —
                            # fuses scan-tick work with the heavy redraw
                            # into ONE expensive frame. On Tier-A-only
                            # ticks we fall through to the bottom of the
                            # frame for a cheap blit-only render so the
                            # agent's marker + new path line are stamped
                            # on top of the cached background without the
                            # 80 ms canvas redraw. Original ``continue``
                            # semantics (skip old-robot advance, arrival
                            # check, path-exhausted tail) are preserved
                            # for the heavy-frame branch.
                            if visual_refresh_due:
                                _heavy_render()
                                # Fall through (no ``continue``) so that
                                # ``new_i += 1`` at the bottom of the
                                # frame still advances the agent. With a
                                # ``continue`` here, the next frame would
                                # start with ``new_i = 0`` and accumulate
                                # zero distance, so the scan check would
                                # fail and the BEV / 3D would only refresh
                                # every other frame instead of every frame.
                                # The bottom-of-loop blit also re-stamps
                                # the marker on the latest position.

                # Check arrival
                if self.goal_pos:
                    d = np.sqrt((nx-self.goal_pos[0])**2 + (ny-self.goal_pos[1])**2)
                    if d < NAV_STEP * 2: new_done = True
                new_i += 1

            # Advance old robot
            if not old_done and old_i < len(old_pts):
                ox, oy = old_pts[old_i]
                if old_i > 0:
                    old_dist += np.sqrt((old_pts[old_i][0]-old_pts[old_i-1][0])**2 +
                                        (old_pts[old_i][1]-old_pts[old_i-1][1])**2)
                self.old_mk.set_data([ox], [oy])
                self.old_path_line.set_data([p[0] for p in old_pts[old_i:]],
                                            [p[1] for p in old_pts[old_i:]])

                # Re-scan and re-plan for old method
                if old_dist >= SCAN_EVERY:
                    old_dist = 0
                    cloud, origin, sn = do_scan(self.mesh, ox, oy, self.rays,
                                                 self._heading_along(ox, oy, old_pts, old_i))
                    if cloud is not None and len(cloud) > 3:
                        oo, Ro, os = build_height_costmap(cloud, origin, sn)
                        self.old_global.merge_local(oo, origin, Ro, os)
                        old_path = _astar_raw(self.old_global, ox, oy,
                                               self.goal_pos[0], self.goal_pos[1])
                        if old_path and len(old_path) > 1:
                            old_pts = self._resample(old_path, NAV_STEP)
                            old_i = 0
                            self.old_path_line.set_data([p[0] for p in old_pts],
                                                        [p[1] for p in old_pts])
                            # Old (height-band) robot is hidden in this
                            # build (set_visible(False) on the marker and
                            # path line). Skip the heavy render — there's
                            # nothing visible to redraw for the old robot
                            # in either the BEV or the map overlay (which
                            # only render the new/grade costmap).
                            continue

                if self.goal_pos:
                    d = np.sqrt((ox-self.goal_pos[0])**2 + (oy-self.goal_pos[1])**2)
                    if d < NAV_STEP * 2: old_done = True
                old_i += 1

            # ---- Per-frame render ---------------------------------------
            # Heavy redraw only on scan ticks (or arrival): refresh the
            # global costmap overlay + BEV + 3D + status text, do a full
            # canvas draw, and re-snapshot the map background.
            # Every other frame is a blit-only idle frame that just stamps
            # the markers + path lines on top of the cached background.
            done_edge = (new_done and not prev_new_done) or \
                        (old_done and not prev_old_done)
            # ``scan_happened`` is set when the scan branch ran but the
            # A* replan failed (no ``continue``); in that case we still
            # want to land the BEV / 3D updates this frame.
            heavy = scan_happened or done_edge
            prev_new_done = new_done
            prev_old_done = old_done
            if heavy:
                _heavy_render()
            else:
                if use_blit:
                    try:
                        _blit_idle_frame()
                    except Exception:
                        canvas.draw_idle()
                else:
                    canvas.draw_idle()

            # Event pumping. We call ``canvas.flush_events()`` directly
            # (Qt processEvents) instead of ``plt.pause`` — the latter
            # calls ``canvas.start_event_loop(0.001)`` which under the
            # Agg backend used by the headless profile harness sleeps
            # for the full 10 ms ``timestep`` regardless of the requested
            # timeout (see matplotlib/backend_bases.py:start_event_loop).
            # That would put a 10 ms / frame floor on the loop and cap
            # us below 30 FPS on the harness. Under Qt5Agg flush_events
            # is the native ``QApplication.processEvents()`` call.
            # Clear figure.stale first so any deferred ``draw_idle``
            # request inside Qt doesn't re-render BEV/3D.
            self.fig.stale = False
            canvas.flush_events()

            # 15 FPS pacing — sleep the remainder of the frame budget so
            # we don't burn 100% CPU on cheap idle frames.
            elapsed = time.perf_counter() - t_frame_start
            frame_times.append(elapsed)
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

            if new_done and old_done: break
            # If a robot ran out of path points, re-scan and re-plan from current pos
            if new_i >= len(new_pts) and not new_done and new_pts:
                nx, ny = new_pts[-1]
                # Path exhausted — aim at the goal directly.
                cloud, origin, sn = do_scan(self.mesh, nx, ny, self.rays,
                                             self._heading_to_goal(nx, ny))
                if cloud is not None and len(cloud) > 3:
                    no, new_slope, Rn, ns = build_grade_costmap(cloud, origin, sn)
                    self.new_global.merge_local(no, origin, Rn, ns)
                    new_path = _astar_raw(self.new_global, nx, ny, self.goal_pos[0], self.goal_pos[1])
                    if new_path and len(new_path) > 1:
                        new_pts = self._resample(new_path, NAV_STEP)
                        new_i = 0; new_dist = 0
                    else:
                        new_done = True
                else:
                    new_done = True
            if old_i >= len(old_pts) and not old_done and old_pts:
                ox, oy = old_pts[-1]
                # Path exhausted — aim at the goal directly.
                cloud, origin, sn = do_scan(self.mesh, ox, oy, self.rays,
                                             self._heading_to_goal(ox, oy))
                if cloud is not None and len(cloud) > 3:
                    oo, Ro, os = build_height_costmap(cloud, origin, sn)
                    self.old_global.merge_local(oo, origin, Ro, os)
                    old_path = _astar_raw(self.old_global, ox, oy, self.goal_pos[0], self.goal_pos[1])
                    if old_path and len(old_path) > 1:
                        old_pts = self._resample(old_path, NAV_STEP)
                        old_i = 0; old_dist = 0
                    else:
                        old_done = True
                else:
                    old_done = True

        # Restore non-animated state so static interactions (clicks,
        # full ``draw_idle`` after the animation ends) render the
        # markers + path lines + overlay normally.
        for art in animated_artists:
            art.set_animated(False)
        self.global_img.set_animated(False)
        if self.bev_handle is not None:
            self.bev_handle.set_animated(False)

        # Report frame-time distribution over the run — useful both as a
        # regression gauge and as the success signal for the profile
        # harness. ``p95`` and the heavy-frame count expose stutter that
        # the median hides (idle frames are < 1 ms so the median is
        # dominated by them even when every 4th frame is 80 ms heavy).
        if frame_times:
            dts = sorted(frame_times)
            n = len(dts)
            median_dt = dts[n // 2]
            p95_dt = dts[min(n - 1, int(n * 0.95))]
            heavy_count = sum(1 for d in frame_times if d > 0.050)
            wall_elapsed = time.perf_counter() - anim_t0
            heavy_per_sec = heavy_count / wall_elapsed if wall_elapsed > 0 else 0.0
            if median_dt > 0:
                print(f"[_animate] {n} frames, "
                      f"median {median_dt * 1000:.1f} ms/frame "
                      f"({1.0 / median_dt:.1f} FPS), "
                      f"p95 {p95_dt * 1000:.1f} ms, "
                      f"heavy (>50 ms) {heavy_count} in {wall_elapsed:.1f}s wall "
                      f"({heavy_per_sec:.2f}/s)")

        self.status.set_text(f"Done! New: {'ARRIVED' if new_done else 'STUCK'} | Old: {'ARRIVED' if old_done else 'STUCK'}")
        self.animating = False
        self.canvas.draw_idle()

    def _bake_to_mp4(self, out_path, fps, step=None):
        """Offscreen-render the agent traversing its planned path to MP4.

        Mirrors the scan + replan cadence of ``_animate`` (SCAN_EVERY-meter
        ticks) but skips Qt event pacing and idle-frame blits — every frame
        is a full ``canvas.draw()`` so the encoded video stays crisp at the
        requested fps. Grabs the full container (map + costmap overlay +
        agent + planned path + 3D point cloud) at device-pixel resolution
        so HiDPI displays bake at 2x.

        ``step`` is the per-frame travel distance in metres. ``None``
        auto-derives it so the on-screen speed matches the live animation
        (which is paced for 15 fps at NAV_STEP=0.03 m/frame = 0.45 m/s):
        ``step = NAV_STEP * 15 / fps``. At 60 fps that's 0.0075 m/frame,
        giving 4x more frames than the live cadence and visibly smoother
        sub-cell motion.
        """
        import imageio.v2 as imageio
        from PyQt5 import QtGui, QtWidgets
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if not self.new_path:
            print(f"[bake] No path planned — nothing to bake. "
                  f"Default start/goal may be unreachable on this STL.")
            return

        # Per-frame travel: default keeps the live-mode on-screen speed.
        if step is None or step <= 0:
            step = NAV_STEP * 15.0 / float(fps)
        bake_step = float(step)
        # Arrival tolerance scales with the per-frame step so the agent
        # always reaches the goal within a frame or two regardless of fps.
        arrival_tol = max(NAV_STEP * 2.0, bake_step * 3.0)
        print(f"[bake] step={bake_step*1000:.1f} mm/frame "
              f"({bake_step * fps:.2f} m/s on screen)")

        new_pts = self._resample(self.new_path, bake_step)
        new_done = not new_pts

        # ── Pure-pursuit kinematics ──
        # The agent doesn't snap to ``new_pts[i]`` every frame; instead a
        # unicycle model integrates position from velocity and heading
        # from a rate-limited angular velocity. The planned path is the
        # target — each frame we find the closest point on it, look
        # ahead a small distance, and steer toward that point. This is
        # what avoids the "instant 180° rotation" glitch: when a replan
        # changes the path direction sharply, the agent rotates over
        # multiple frames at ``omega_max`` instead of snapping.
        v = bake_step * fps                # m/s (matches the previous
                                            # bake's on-screen speed)
        omega_max = 2.5                    # rad/s (~143°/s) — sharp
                                            # enough to follow corners,
                                            # smooth enough that no
                                            # single frame snaps > a
                                            # few degrees
        look_ahead_m = 0.45                # pure-pursuit carrot distance
        dt = 1.0 / float(fps)
        # Search window for closest-point lookup on the path. With
        # ``bake_step`` ~7.5 mm and the agent moving forward ≤ bake_step
        # per frame, a 80-step window (~60 cm) is plenty.
        closest_window = 80

        phys_x, phys_y = self.robot_pos
        # Seed heading at the look-ahead direction from the start so the
        # first frame doesn't have to slew from 0.
        _seed = _look_ahead_point(new_pts, 0, phys_x, phys_y,
                                  look_ahead_m, self.goal_pos)
        phys_heading = float(np.arctan2(_seed[1] - phys_y,
                                         _seed[0] - phys_x))
        last_closest_i = 0
        # Distance travelled since the last scan tick (SCAN_EVERY gate).
        phys_dist = SCAN_EVERY

        # ── Headless once the bake starts ──
        # Hiding the container removes it from macOS's compositor so the
        # window server stops drawing it to the screen. The QOpenGLWidget
        # keeps its GL context (it was initialized on first show), and
        # ``container.grab()`` still composites the widget tree into the
        # pixmap — we just skip painting to the visible display. Cuts
        # ~15–25 ms / frame on retina.
        self.container.hide()

        def _grab_container_rgb():
            """Composite the whole container (mpl canvas + GL panel) into
            an RGB ndarray at device-pixel resolution.

            Synchronous: ``canvas.draw()`` rasterizes the matplotlib
            figure on this thread, and ``widget.repaint()`` forces the
            QOpenGLWidget to paint immediately (no event-loop turn
            needed). ``container.grab()`` then composites everything
            into the pixmap. ``Format_RGBA8888`` is packed (no per-row
            padding) so the numpy reshape works directly.
            """
            self.fig.canvas.draw()
            self.three_d.widget.repaint()
            pixmap = self.container.grab()
            qimg = pixmap.toImage().convertToFormat(
                QtGui.QImage.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            try:
                nbytes = qimg.sizeInBytes()
            except AttributeError:
                nbytes = qimg.byteCount()
            ptr = qimg.constBits()
            ptr.setsize(nbytes)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
            return arr[:, :, :3].copy()


        # Probe one frame to lock in dimensions for the writer. libx264
        # wants even width/height; trim to the nearest even pixel.
        probe = _grab_container_rgb()
        h, w = probe.shape[:2]
        even_w = w - (w % 2)
        even_h = h - (h % 2)

        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,           # imageio scale 0..10; 8 = visually crisp
            macro_block_size=1,  # don't force a second dim trim
        )
        print(f"[bake] -> {out_path}  ({even_w}x{even_h} @ {fps} fps)")
        t0 = time.perf_counter()
        max_frames = 60 * 60 * fps  # 60-minute safety cap
        frame_count = 0
        try:
            for _ in range(max_frames):
                if new_done:
                    break

                # ── Pure-pursuit step ──
                # Find closest point on the planned path, look ahead from
                # there, slew heading toward the carrot at omega_max, and
                # integrate position forward at v.
                last_closest_i = _closest_path_index_after(
                    new_pts, last_closest_i, phys_x, phys_y, closest_window)
                carrot = _look_ahead_point(
                    new_pts, last_closest_i, phys_x, phys_y,
                    look_ahead_m, self.goal_pos)
                target_heading = float(np.arctan2(
                    carrot[1] - phys_y, carrot[0] - phys_x))
                # Shortest signed angular difference in [-pi, pi].
                err = ((target_heading - phys_heading + np.pi)
                       % (2.0 * np.pi)) - np.pi
                max_dtheta = omega_max * dt
                if err > max_dtheta:
                    err = max_dtheta
                elif err < -max_dtheta:
                    err = -max_dtheta
                phys_heading += err
                phys_x += v * dt * float(np.cos(phys_heading))
                phys_y += v * dt * float(np.sin(phys_heading))
                phys_dist += v * dt

                nx, ny = phys_x, phys_y
                self.new_mk.set_data([nx], [ny])
                if last_closest_i < len(new_pts):
                    self.new_path_line.set_data(
                        [p[0] for p in new_pts[last_closest_i:]],
                        [p[1] for p in new_pts[last_closest_i:]])

                # Scan + classify every frame so the cloud, point colors,
                # and 3D camera all track the moving agent at 60 fps.
                # do_scan is ~3 ms with embreex; build_grade_costmap is
                # ~15 ms. The heavier global-costmap merge + A* replan
                # still fires only at the deployed SCAN_EVERY cadence so
                # navigation behavior matches the real robot.
                cs, origin, sn = do_scan(
                    self.mesh, nx, ny, self.rays, phys_heading)
                if cs is not None and len(cs) > 3:
                    no, slope, R, seen = build_grade_costmap(cs, origin, sn)
                    self.new_obs = no
                    self.last_R = R; self.last_origin = origin
                    self._update_3d(cs, origin, sn, slope)
                    # Merge + replan on the deployed SCAN_EVERY cadence.
                    if phys_dist >= SCAN_EVERY:
                        phys_dist = 0.0
                        self.new_global.merge_local(no, origin, R, seen)
                        overlay = self.new_global.render_overlay()
                        self.global_img.set_data(overlay)
                        replanned = _astar_raw(
                            self.new_global, nx, ny,
                            self.goal_pos[0], self.goal_pos[1])
                        if replanned and len(replanned) > 1:
                            # Anchor at the agent's true position so the
                            # transition between OLD and NEW path doesn't
                            # introduce a cell-snap hop.
                            anchored = [(nx, ny)] + list(replanned)
                            new_pts = self._resample(anchored, bake_step)
                            last_closest_i = 0
                            self.new_path_line.set_data(
                                [p[0] for p in new_pts],
                                [p[1] for p in new_pts])

                # Arrival check (distance from physical position to goal).
                if self.goal_pos is not None:
                    d = float(np.hypot(nx - self.goal_pos[0],
                                       ny - self.goal_pos[1]))
                    if d < arrival_tol:
                        new_done = True

                self.status.set_text(
                    f"Bake: frame {frame_count + 1} | "
                    f"pos ({nx:.1f},{ny:.1f}) "
                    f"h {np.degrees(phys_heading):+.0f}°")

                frame = _grab_container_rgb()
                writer.append_data(frame[:even_h, :even_w, :])
                frame_count += 1
                if frame_count % fps == 0:
                    secs = frame_count / fps
                    elapsed = time.perf_counter() - t0
                    print(f"[bake]   {frame_count:5d} frames "
                          f"({secs:5.1f}s video, {elapsed:5.1f}s wall)",
                          flush=True)
        finally:
            writer.close()
        wall = time.perf_counter() - t0
        secs = frame_count / fps if fps else 0.0
        print(f"[bake] Done: {frame_count} frames "
              f"({secs:.1f}s video) in {wall:.1f}s wall -> {out_path}")

def main():
    import argparse
    global GRADE_FREE_MAX, GRADE_MODERATE_MAX, GRADE_STEEP_MAX, TRAVERSABLE_MAX_DEG, N_RAYS
    parser = argparse.ArgumentParser(
        description="Interactive PCA grade-classification GUI. "
                    "Click to place an agent and see how PCA classifies the "
                    "ground around it. Shift+Click sets a goal. Press G to "
                    "animate.")
    parser.add_argument(
        "--threshold", "-t", type=int, default=15,
        choices=[15, 20, 30],
        help="Grade tolerance: 15 (8.5 deg), 20 (11.3 deg), or 30 (16.7 deg). "
             "Default 15 (= 5 deg passes, 10 deg+ blocks).")
    parser.add_argument(
        "--rays", type=int, default=N_RAYS,
        help=f"Number of LiDAR rays per scan (default {N_RAYS}). "
             f"Use a smaller value (~2000) on Windows-native Python "
             f"where trimesh has no embree and 10000 rays is glacial.")
    parser.add_argument(
        "--bake-mp4", type=str, default=None,
        help="Render the agent traversing the planned path to this MP4 path "
             "instead of opening an interactive window. Captures the map + "
             "costmap overlay + path at --bake-fps frames/sec.")
    parser.add_argument(
        "--bake-fps", type=int, default=60,
        help="Frames per second for --bake-mp4 output (default 60).")
    parser.add_argument(
        "--bake-step", type=float, default=None,
        help="Per-frame agent travel in metres for --bake-mp4. "
             "Default auto-derives so on-screen speed matches the live "
             "animation: NAV_STEP * 15 / fps (= 0.0075 m at 60 fps).")
    parser.add_argument(
        "stl", nargs="?", default=STL_PATH,
        help="Optional path to a different STL.")
    args = parser.parse_args()
    N_RAYS = args.rays

    # Apply the chosen threshold globally (these are the same constants
    # run_benchmarks.py overrides per threshold). Picked the matching
    # GRADE_FREE_MAX value from run_benchmarks.py.
    cfg = {15: (6.0,  8.5), 20: (8.0, 11.3), 30: (12.0, 16.7)}
    free_max, deg = cfg[args.threshold]
    GRADE_FREE_MAX = free_max
    GRADE_MODERATE_MAX = deg
    GRADE_STEEP_MAX = deg
    TRAVERSABLE_MAX_DEG = deg
    print(f"== PCA grade tolerance: {args.threshold}% (TRAVERSABLE_MAX_DEG={deg} deg) ==")
    print(f"   pass slopes <= {deg:.1f} deg, block slopes > {deg:.1f} deg + {PCA_NOISE_MARGIN_DEG} deg margin")
    _warn_if_slow_raycaster(args.rays)
    DualNavGUI(args.stl, bake_mp4=args.bake_mp4, bake_fps=args.bake_fps,
               bake_step=args.bake_step)


def _warn_if_slow_raycaster(n_rays):
    """Detect whether trimesh picked up an Embree-backed ray intersector.

    Without Embree, a single 11 520-ray scan on a 300k-face mesh takes ~2 min
    on Apple Silicon — looks like the sim is frozen. Print a clear, OS-aware
    install hint so the user doesn't think the GUI hung."""
    import platform
    try:
        import trimesh as _tm
        _probe = _tm.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
        backend = type(_probe.ray).__module__
    except Exception:
        backend = "unknown"
    if "pyembree" in backend or "embree" in backend:
        print(f"   raycaster: {backend} (fast)")
        return
    sysname = platform.system()
    hint = {
        "Darwin":  "pip install embreex   # Apple Silicon + Intel macOS",
        "Linux":   "pip install embreex",
        "Windows": "pip install embreex",
    }.get(sysname, "pip install embreex")
    print(f"[WARN] trimesh is using the pure-Python ray intersector ({backend}).")
    print(f"       A {n_rays}-ray scan on this mesh will take ~{max(1, n_rays // 250)} s each;")
    print(f"       the GUI will look frozen during the first scan.")
    print(f"       Install the Embree backend for ~1000x speedup:")
    print(f"         {hint}")

if __name__=="__main__": main()
