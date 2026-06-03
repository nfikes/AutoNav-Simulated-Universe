#!/usr/bin/env python3
"""
Ramp Crossing Test — Grade Classification Validation

Tests if grade classification correctly passes or blocks LiDAR agents
at ramps of increasing steepness.

Terrain: 5 ramps side-by-side at different x positions, all ~y=14-38:
  Ramp 1 (x=-30):  5 deg  — Expected: PASS
  Ramp 2 (x=-15): 10 deg  — Expected: PASS
  Ramp 3 (x=  0): 15 deg  — Expected: NO-PASS
  Ramp 4 (x= 14): 20 deg  — Expected: NO-PASS
  Ramp 5 (x= 28): 25 deg  — Expected: NO-PASS

Phase 1: Each agent navigates (ramp_x, 0) -> (ramp_x, 42).
         Tests grade classification on each ramp.

Phase 2: Agents teleported to (ramp_x, 42) -> (ramp_x, -40).
         Routing behind/around walls at y~=-26.  Tests that flat
         terrain is NOT falsely graded, and that walls ARE detected.
         Agents stay on the platform and route around walls.

Uses ABSOLUTE grade: PCA normals compared to world-up (gravity vector),
not the sensor's tilted frame.  This means every cell on a 15-deg ramp
reads 15 deg regardless of the robot's orientation on the ramp.

Walls:
  Wall A: x=-2.9..-0.9, y=-2.6..-2.4  (z=2.17, tall)
  Wall B: x= 0.1.. 2.1, y=-2.6..-2.4  (z=2.17, tall)
  Wall C: x= 3.7.. 3.95, y=-3.55..-1.45 (z=0.82, short)
  Wall D: x= 4.7.. 4.95, y=-3.55..-1.45 (z=0.82, short)
"""

import os
import sys
import numpy as np
import trimesh
import matplotlib
# Default to Qt5Agg so the benchmark opens a live window like the GUI.
# Worker processes force Agg in _worker_init below (they don't render).
# Pass MPLBACKEND=Agg in the environment for headless / CI use.
try:
    matplotlib.use("Qt5Agg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
from multiprocessing import get_context
from scipy.ndimage import distance_transform_edt, binary_dilation

# When invoked from scripts/ directly, the parent simulated_world/ dir
# isn't on sys.path. Add it so `import lidar_sim_gui` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import simulation core
import lidar_sim_gui as sim


# ── Multiprocessing worker for per-agent scans ─────────────────────
# Each worker holds its own copy of the mesh + rays so the heavy
# do_scan + build_grade_costmap step can run in parallel across agents.
_W_MESH = None
_W_RAYS = None


def _worker_init(stl_path, n_rays, threshold_deg, free_max, mod_max, steep_max):
    """Initializer run once per worker process."""
    global _W_MESH, _W_RAYS
    import matplotlib as _mpl
    _mpl.use("Agg", force=True)
    import trimesh as _trimesh
    import lidar_sim_gui as wsim
    _W_MESH = _trimesh.load(stl_path)
    _W_RAYS = wsim.gen_rays(n_rays)
    # Sync thresholds with the parent (run_benchmarks mutates these
    # before each threshold; the spawn pool inherits them once at start).
    wsim.TRAVERSABLE_MAX_DEG = threshold_deg
    wsim.GRADE_FREE_MAX = free_max
    wsim.GRADE_MODERATE_MAX = mod_max
    wsim.GRADE_STEEP_MAX = steep_max


def _worker_scan(args):
    """Scan + build local grade costmap. Returns a picklable tuple or None.
    Tuple shape: (obs, slope, R, seen, origin, sn, cloud_or_none).
    `heading` is the agent's yaw (radians) so the front-arc rays line
    up with its forward direction."""
    px, py, heading, want_cloud = args
    import lidar_sim_gui as wsim
    c, o, sn = wsim.do_scan(_W_MESH, px, py, _W_RAYS, heading)
    if c is None or len(c) < 3:
        return None
    obs, slope, R, seen = wsim.build_grade_costmap(c, o, sn)
    return (obs, slope, R, seen, o, sn, c if want_cloud else None)


def _worker_scan_indexed(args):
    """Same as _worker_scan but tagged with the caller's index so
    imap_unordered results can be matched back to their agents."""
    idx = args[0]
    return idx, _worker_scan(args[1:])

# ── Test configuration ─────────────────────────────────────────────
STL_PATH = sim.STL_PATH  # share the GUI's resolved STL path (sim is in the parent dir; assets live at parent.parent / 3d_assets)

# Grade threshold is in sim module: TRAVERSABLE_MAX_DEG = 11.3 deg (20% grade)

RAMPS = [
    {"id": 1,  "x": -2.25, "slope": 5,  "expected": "PASS"},
    {"id": 2,  "x": -0.50, "slope": 10, "expected": "PASS"},
    {"id": 3,  "x":  0.75, "slope": 15, "expected": "PASS"},
    {"id": 4,  "x":  2.25, "slope": 20, "expected": "NO-PASS"},
    {"id": 5,  "x":  3.75, "slope": 25, "expected": "NO-PASS"},
    # Second row
    {"id": 6,  "x": -2.25, "slope": 5,  "expected": "PASS"},
    {"id": 7,  "x": -0.50, "slope": 10, "expected": "PASS"},
    {"id": 8,  "x":  0.75, "slope": 15, "expected": "PASS"},
    {"id": 9,  "x":  2.25, "slope": 20, "expected": "NO-PASS"},
    {"id": 10, "x":  2.75, "slope": 25, "expected": "NO-PASS"},
    # Third row
    {"id": 11, "x": -2.25, "slope": 5,  "expected": "PASS"},
    {"id": 12, "x": -0.50, "slope": 10, "expected": "PASS"},
    {"id": 13, "x":  0.75, "slope": 15, "expected": "PASS"},
    {"id": 14, "x":  2.25, "slope": 20, "expected": "NO-PASS"},
    {"id": 15, "x":  2.75, "slope": 25, "expected": "NO-PASS"},
]

COLORS = ["#33cc33", "#3388ee", "#ff9900", "#dd2222", "#9933cc",
          "#66ee66", "#66aaff", "#ffcc44", "#ff6666", "#cc66ff",
          "#99ff99", "#99ccff", "#ffe066", "#ff9999", "#dd99ff"]

ARRIVAL_THRESH = 0.3     # meters — close enough to goal
VIEW_HALF = 6            # visualization extent (m)
SCAN_PERIOD = 0.05       # seconds per scan — 20 Hz to match real
                         # SICK multiScan165 baseline scanning frequency.
                         # Was distance-based (0.1 m/scan) which under-
                         # sampled at slow speeds (only ~3 Hz at 0.29 m/s).
MAX_STEPS = 3000         # max animation steps per phase

# ── Cached spawn-validity Z-slice ─────────────────────────────────
# Pre-rendered "what cells are flat ground" map.  Built once per STL
# (or loaded from disk if the cache is fresh).  Replaces the per-spawn
# LiDAR scan/build_costmap path, which was the dominant pre-phase cost.
SPAWN_MASK_RES  = 0.05   # m per cell
SPAWN_MASK_HALF = 8.0    # m — covers the entire navigable platform
SPAWN_GROUND_MAX = 0.10  # m — max surface height to be counted as flat ground
# Physics model: 10 kg mass pushed by 8 N with viscous friction
MASS = 10.0              # kg
F_DRIVE = 8.0            # Newtons — force toward next waypoint.
                         # 12 N moved agents too fast (1.2 m/s) and they
                         # overshot waypoints near walls. 8 N gives 0.8
                         # m/s terminal — still 60% faster than the
                         # original 5 N tune so narrow lanes traverse,
                         # but the agent stays close enough to the path
                         # that the spring keeps it on rails.
FRICTION = 28.0          # N·s/m — viscous drag coefficient.
                         # Doubled from 14 → 28 to halve terminal speed
                         # (0.57 → 0.29 m/s) so agents have more scan
                         # cycles per metre traveled and less momentum
                         # carrying them into inflation halos.
DT = 0.02                # seconds per simulation step
# Terminal velocity = F_DRIVE / FRICTION = 0.29 m/s, ~0.006 m/step
PATH_SPRING_K        = 22.0  # N/m lateral pull onto the planned path
PATH_SPRING_K_NEAR_O = 14.0  # softer near walls, but still > F_DRIVE
                             # so the agent can't drift sideways into
                             # an inflation halo while heading forward


# ── Test costmap builder (uses sim's build_grade_costmap directly) ──
# The test now uses the same sensor-frame PCA as lidar_sim_gui.py.
# No custom builder needed — sim.build_grade_costmap has the 11.3 deg
# threshold, wall dilation, y-only slope dilation, and full-cell merge.


# ── Agent ──────────────────────────────────────────────────────────

class Agent:
    """LiDAR navigation agent with physics-based movement.
    10 kg mass pushed by 5 N toward the next waypoint, with viscous
    friction limiting terminal velocity to 0.5 m/s.  Wall collisions
    slide along the boundary instead of stopping dead."""

    def __init__(self, ramp, mesh, rays, color):
        self.id = ramp["id"]
        self.ramp_x = ramp["x"]
        self.slope = ramp["slope"]
        self.expected = ramp["expected"]
        self.color = color
        self.mesh = mesh
        self.rays = rays

        self.costmap = sim.GlobalCostmap()
        self.pos = self.goal = None
        self.vx = self.vy = 0.0
        self.path = []
        self.wp_idx = 0          # current target waypoint index
        self.traveled = []
        self.done = self.stuck = False
        self.time_since_scan = 999.0  # seconds; >= SCAN_PERIOD triggers a scan
        self.stall_count = 0
        self.last_scan_world = None
        self.last_scan_is_obs = None
        # Sensor-frame data needed by the shared 3D panel (sim_view.update_3d_panel).
        # Populated for Agent 1 only — the 3D panel mirrors what the GUI
        # shows, and that's a single agent's view.
        self.last_scan_cloud = None       # sensor-frame point cloud
        self.last_scan_origin = None      # LiDAR origin in world coords
        self.last_scan_sn = None          # surface normal at origin
        self.last_scan_slope_grid = None  # per-cell PCA slope (deg)
        self.last_scan_obs = None         # local obstacle grid
        self.result = None
        self.phase1_result = None

    def reset(self, sx, sy, gx, gy, clear_costmap=True):
        if clear_costmap:
            self.costmap = sim.GlobalCostmap()
            _add_platform_bounds(self.costmap)
        self.pos = (sx, sy)
        self.goal = (gx, gy)
        self.vx = self.vy = 0.0
        self.path = []
        self.wp_idx = 0
        self.traveled = [(sx, sy)]
        self.done = self.stuck = False
        self.time_since_scan = 999.0  # seconds; >= SCAN_PERIOD triggers a scan
        self.stall_count = 0
        self.result = None

    def _is_lethal(self, x, y):
        """Hard-block movement into INSCRIBED + LETHAL cells.
        Keeps the agent ≥ ROBOT_RADIUS away from any obstacle so it
        never bumps a wall, which previously corrupted the next scan
        with a 30°+ sensor tilt and produced FP streaks."""
        cx, cy = self.costmap.w2c(x, y)
        if cx < 0 or cx >= self.costmap.w or cy < 0 or cy >= self.costmap.h:
            return True
        return self.costmap.inflated[cy, cx] >= 253

    def _path_blocked(self):
        """Trigger a replan if a remaining waypoint became INSCRIBED or
        LETHAL (a new obstacle was detected since planning). A* won't
        cross inscribed cells in the new traversal rule, so any inscribed
        waypoint means the existing path is no longer safe."""
        for i in range(self.wp_idx, len(self.path)):
            cx, cy = self.costmap.w2c(self.path[i][0], self.path[i][1])
            if 0 <= cx < self.costmap.w and 0 <= cy < self.costmap.h:
                if self.costmap.inflated[cy, cx] >= 253:
                    return True
        return False

    def _path_cost(self, path):
        if not path or len(path) < 2:
            return float('inf')
        return sum(np.sqrt((path[i][0]-path[i-1][0])**2 +
                           (path[i][1]-path[i-1][1])**2)
                   for i in range(1, len(path)))

    def heading_along_path(self):
        """Direction (radians) the agent is currently heading. Follows
        the planned path: aims at the next waypoint along ``self.path``.
        Falls back to the goal bearing if no usable waypoint exists."""
        if self.path and self.wp_idx < len(self.path):
            tx, ty = self.path[self.wp_idx]
            dx, dy = tx - self.pos[0], ty - self.pos[1]
            if dx * dx + dy * dy > 1e-6:
                return float(np.arctan2(dy, dx))
        if self.goal is not None:
            return float(np.arctan2(self.goal[1] - self.pos[1],
                                     self.goal[0] - self.pos[0]))
        return 0.0

    def _serial_scan(self):
        """Run scan + build_grade_costmap inline (no worker pool).
        Returns the same tuple shape as _worker_scan, or None on failure."""
        c, o, sn = sim.do_scan(self.mesh, self.pos[0], self.pos[1],
                                self.rays, self.heading_along_path())
        if c is None or len(c) < 3:
            return None
        obs, slope, R, seen = sim.build_grade_costmap(c, o, sn)
        return (obs, slope, R, seen, o, sn, c if self.id == 1 else None)

    def scan_and_plan(self):
        """LiDAR scan, build grade costmap, replan with path hysteresis.
        Single-process path (used at startup before the pool is online)."""
        return self.process_scan_result(self._serial_scan())

    def process_scan_result(self, scan_result):
        """Common post-scan work: stash viz data (agent 1), merge into the
        persistent costmap, replan A*. Accepts either a serial-scan result
        or one produced by a worker process."""
        if scan_result is None:
            self.stall_count += 1
            return False
        obs, slope, R, seen, o, sn, c = scan_result
        # Track LiDAR tilt over the run for diagnostics. The sensor's
        # surface_normal sn[2] is the cosine of tilt-from-world-up.
        # Anything >5 deg means the agent is on a slope (ramp / edge);
        # huge tilts (>30 deg) indicate the tripod estimate hit a wall
        # face and the rest of the scan should be regarded with care.
        tilt_deg = float(np.degrees(np.arccos(np.clip(sn[2], -1.0, 1.0))))
        if not hasattr(self, "tilt_log"):
            self.tilt_log = []
        self.tilt_log.append((self.pos[0], self.pos[1], tilt_deg))
        if tilt_deg > 30.0:
            print(f"  [tilt-warn] R{self.id} at ({self.pos[0]:+.2f},"
                  f"{self.pos[1]:+.2f}): sensor tilt = {tilt_deg:.1f} deg")
        # Store scan data for LiDAR visualization (Agent 1 only). The
        # shared 3D panel (sim_view.update_3d_panel) needs the sensor-
        # frame cloud + slope grid + obstacle grid — the same inputs the
        # GUI's _update_3d uses. The cell lookup inside update_3d_panel
        # subtracts origin and bins at COSTMAP_RES, so the cloud must be
        # in the same frame the slope grid was built in (sensor frame).
        if self.id == 1 and c is not None:
            self.last_scan_cloud = c
            self.last_scan_origin = o
            self.last_scan_sn = sn
            self.last_scan_slope_grid = slope
            self.last_scan_obs = obs
            # Legacy fields kept around in case any other code reads
            # them — the new viz path uses last_scan_* above.
            res = sim.COSTMAP_RES; half = sim.COSTMAP_HALF
            dx = c[:, 0] - o[0]; dy = c[:, 1] - o[1]
            px = ((dx + half) / res).astype(int)
            py = ((dy + half) / res).astype(int)
            gw = int(2 * half / res)
            is_obs = np.zeros(len(c), dtype=bool)
            in_grid = (px >= 0) & (px < gw) & (py >= 0) & (py < gw)
            is_obs[in_grid] = obs[py[in_grid], px[in_grid]]
            cloud_world = (R @ (c - o).T).T + o
            self.last_scan_world = cloud_world
            self.last_scan_is_obs = is_obs
        self.costmap.merge_local(obs, o, R, seen)
        self.time_since_scan = 0.0

        has_path = len(self.path) > 0 and self.wp_idx < len(self.path)
        blocked = has_path and self._path_blocked()

        # Use raw A* path (no LOS smoothing) — many segments that stay
        # in safe corridors.  The physics model smooths the trajectory.
        new_path = sim._astar_raw(
            self.costmap, self.pos[0], self.pos[1],
            self.goal[0], self.goal[1])

        if new_path and len(new_path) > 1:
            self.stall_count = 0  # path exists → not stuck
            if not has_path or blocked:
                # Current path is blocked or missing — always adopt new
                self.path = new_path
                self.wp_idx = 0
                return True
            # Always adopt new path — ensures fast response to new obstacles
            self.path = new_path
            self.wp_idx = 0
            return True

        # Only count stall when A* returns None (truly no path)
        self.stall_count += 1
        return False

    def advance(self):
        """Physics step: apply drive force toward target, friction, integrate.
        Wall collisions decompose into axis-aligned sliding."""
        if self.done or self.stuck:
            return False
        if not self.path:
            return False

        # Advance waypoint index: skip waypoints we've passed or are close to
        while self.wp_idx < len(self.path) - 1:
            wx, wy = self.path[self.wp_idx]
            dist_cur = np.sqrt((wx - self.pos[0])**2 + (wy - self.pos[1])**2)
            # Advance if within 0.3m, or if we're closer to the NEXT waypoint
            if dist_cur < 0.3:
                self.wp_idx += 1
            else:
                nx, ny = self.path[self.wp_idx + 1]
                dist_next = np.sqrt((nx - self.pos[0])**2 + (ny - self.pos[1])**2)
                seg_len = np.sqrt((nx - wx)**2 + (ny - wy)**2)
                if seg_len > 0.01 and dist_next < seg_len:
                    # We're closer to next waypoint than the segment is long
                    # — we've likely passed the current waypoint
                    self.wp_idx += 1
                else:
                    break

        # Target: current waypoint, or goal if we've exhausted the path
        if self.wp_idx < len(self.path):
            tx, ty = self.path[self.wp_idx]
        else:
            tx, ty = self.goal

        # ── Drive force toward target waypoint ──
        dx = tx - self.pos[0]
        dy = ty - self.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            fx = F_DRIVE * dx / dist
            fy = F_DRIVE * dy / dist
        else:
            fx = fy = 0.0

        # ── Obstacle repulsion: push away from nearby lethal cells ──
        # Final safety net only — A*'s inscribed-cell clearance already
        # keeps planned paths off walls. The previous probe set fired
        # heavy 15 N pushes at 0.20 m, which is INSIDE A*'s legal
        # clearance band, so legal paths got fought instead of followed.
        # New probes only fire genuinely close (<= 0.12 m) and the cap
        # is below F_DRIVE so net force stays toward the waypoint.
        repel_fx = repel_fy = 0.0
        for probe_r, force_n in [(0.12, 4.0), (0.06, 10.0)]:
            for angle in range(0, 360, 45):
                rad = np.radians(angle)
                px = self.pos[0] + probe_r * np.cos(rad)
                py = self.pos[1] + probe_r * np.sin(rad)
                if self._is_lethal(px, py):
                    repel_fx += force_n * (self.pos[0] - px) / probe_r
                    repel_fy += force_n * (self.pos[1] - py) / probe_r
        repel_mag = np.sqrt(repel_fx**2 + repel_fy**2)
        if repel_mag > 8.0:
            repel_fx *= 8.0 / repel_mag
            repel_fy *= 8.0 / repel_mag
        fx += repel_fx
        fy += repel_fy
        near_obstacle = repel_mag > 0.1

        # ── Path-following force: pull agent back onto the path segment ──
        # Reduced when near obstacles to avoid force deadlocks.
        if self.wp_idx < len(self.path) and self.wp_idx > 0:
            ax_s, ay_s = self.path[self.wp_idx - 1]
            bx_s, by_s = self.path[self.wp_idx]
            abx, aby = bx_s - ax_s, by_s - ay_s
            seg_len2 = abx**2 + aby**2
            if seg_len2 > 0.001:
                t = ((self.pos[0] - ax_s) * abx +
                     (self.pos[1] - ay_s) * aby) / seg_len2
                t = max(0.0, min(1.0, t))
                proj_x = ax_s + t * abx
                proj_y = ay_s + t * aby
                lat_dx = proj_x - self.pos[0]
                lat_dy = proj_y - self.pos[1]
                lat_dist = np.sqrt(lat_dx**2 + lat_dy**2)
                if lat_dist > 0.01:
                    # Reduce spring near obstacles to prevent force deadlock,
                    # but keep it stronger than F_DRIVE so the agent can't
                    # drift off the path into an inflation halo.
                    stiffness = PATH_SPRING_K_NEAR_O if near_obstacle else PATH_SPRING_K
                    spring = stiffness * lat_dist
                    fx += spring * lat_dx / lat_dist
                    fy += spring * lat_dy / lat_dist

        # Viscous friction
        fx -= FRICTION * self.vx
        fy -= FRICTION * self.vy

        # Integrate (semi-implicit Euler)
        ax_phys = fx / MASS
        ay_phys = fy / MASS
        self.vx += ax_phys * DT
        self.vy += ay_phys * DT
        new_x = self.pos[0] + self.vx * DT
        new_y = self.pos[1] + self.vy * DT

        # Hard clamp: if new position is lethal, slide and force replan
        if self._is_lethal(new_x, new_y):
            self.time_since_scan = SCAN_PERIOD  # force rescan on ANY collision
            if not self._is_lethal(new_x, self.pos[1]):
                new_y = self.pos[1]
                self.vy *= 0.5
            elif not self._is_lethal(self.pos[0], new_y):
                new_x = self.pos[0]
                self.vx *= 0.5
            else:
                new_x, new_y = self.pos
                self.vx *= -0.3
                self.vy *= -0.3

        self.time_since_scan += DT
        self.pos = (new_x, new_y)
        self.traveled.append((new_x, new_y))

        # Check arrival
        if np.sqrt((self.pos[0] - self.goal[0])**2 +
                   (self.pos[1] - self.goal[1])**2) < ARRIVAL_THRESH:
            self.done = True
        return True

    def needs_scan(self):
        return self.time_since_scan >= SCAN_PERIOD

    def crossed_ramp(self):
        """Did the traveled path go through the ramp zone?
        Checks if the agent passed within 0.8m of its ramp center
        at the ramp zone (y=1.5-3.8).  Zone is generous because
        inflation from nearby terrain features can offset the path
        slightly while still crossing the ramp surface."""
        for x, y in self.traveled:
            if 1.5 <= y <= 3.8 and abs(x - self.ramp_x) <= 0.8:
                return True
        return False

    def lethal_violations(self):
        """Count path points that pass through LETHAL (254) cells.
        Any violation means the agent went through a wall or steep ramp."""
        count = 0
        for x, y in self.traveled:
            cx, cy = self.costmap.w2c(x, y)
            if 0 <= cx < self.costmap.w and 0 <= cy < self.costmap.h:
                if self.costmap.inflated[cy, cx] >= 254:
                    count += 1
        return count


# ── Helpers ────────────────────────────────────────────────────────

def _save_bw_costmap(agents, out_path, size=640):
    """Save a clean B&W obstacle map at `size` pixels — UNION of every
    agent's costmap obstacles (the amalgamation a real multi-robot
    fleet would share). No inflation halos, no agent colors. Direct
    comparison target against the hand-drawn ground-truth costmap."""
    from PIL import Image
    cm0 = agents[0].costmap
    combined = np.zeros((cm0.h, cm0.w), dtype=bool)
    for a in agents:
        combined |= a.costmap.obstacles
    img_arr = np.where(combined, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode='L')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # y increases up
    img = img.resize((size, size), resample=Image.NEAREST)
    img.save(out_path)


def _add_platform_bounds(costmap):
    """Mark platform edges — delegates to shared GlobalCostmap method."""
    costmap.add_platform_bounds()


def build_spawn_mask(mesh, force_rebuild=False):
    """Build (or load cached) a 2D Z-slice mask of flat-ground cells.

    For each (x, y) cell in the test domain we ray-cast straight down
    and take the *highest* surface hit. Cells whose top surface sits at
    or below SPAWN_GROUND_MAX are flat platform — safe to drop an agent
    on. Anything else (walls, the rising face of a ramp, cones) is
    blocked.

    The mask is cached next to the STL with a `.spawn_mask.npy` suffix
    and reused if the STL's mtime hasn't changed."""
    cache_path = Path(STL_PATH).with_suffix('.spawn_mask.npy')
    stl_mtime = Path(STL_PATH).stat().st_mtime
    if not force_rebuild and cache_path.exists() and cache_path.stat().st_mtime >= stl_mtime:
        print(f"Loading cached spawn mask: {cache_path.name}")
        return np.load(str(cache_path))

    print(f"Building spawn mask (Z-slice at z<{SPAWN_GROUND_MAX:.2f}m, "
          f"res={SPAWN_MASK_RES}m, half={SPAWN_MASK_HALF}m)...")
    res = SPAWN_MASK_RES
    half = SPAWN_MASK_HALF
    xs = np.arange(-half, half, res)
    ys = np.arange(-half, half, res)
    gw, gh = len(xs), len(ys)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    origins = np.column_stack([XX.ravel(), YY.ravel(), np.full(gw * gh, 500.0)])
    dirs = np.tile([0.0, 0.0, -1.0], (gw * gh, 1))
    locs, idx_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=True)
    # Track per-ray max z. Use -inf as the identity (NOT NaN — np.maximum
    # propagates NaN and would zero out every hit cell).
    elev = np.full(gw * gh, -np.inf)
    has_hit = np.zeros(gw * gh, dtype=bool)
    if len(locs):
        np.maximum.at(elev, idx_ray, locs[:, 2])
        has_hit[np.unique(idx_ray)] = True
    elev_2d = elev.reshape(gh, gw)
    has_hit_2d = has_hit.reshape(gh, gw)
    flat_2d = has_hit_2d & (elev_2d <= SPAWN_GROUND_MAX)
    # Erode by ROBOT_RADIUS + INFLATION_RADIUS (both 0.15 m → 6 cells at 0.05 m).
    # Without this, a goal can sit on a flat cell that is still inside the
    # inflation halo of an adjacent cone/wall — agent can never enter the
    # goal cell once it's marked inscribed.
    from scipy.ndimage import binary_erosion
    clearance_cells = int(np.ceil((sim.ROBOT_RADIUS + sim.INFLATION_RADIUS) / SPAWN_MASK_RES))
    mask_2d = binary_erosion(flat_2d, iterations=clearance_cells)
    np.save(str(cache_path), mask_2d)
    print(f"  Cached spawn mask: {mask_2d.shape}, "
          f"{mask_2d.sum():,}/{mask_2d.size:,} safe "
          f"({100 * mask_2d.mean():.1f}%, eroded by {clearance_cells} cells)")
    return mask_2d


def is_spawn_safe(mask, x, y):
    """O(1) check against the cached spawn mask."""
    cx = int((x + SPAWN_MASK_HALF) / SPAWN_MASK_RES)
    cy = int((y + SPAWN_MASK_HALF) / SPAWN_MASK_RES)
    if cx < 0 or cx >= mask.shape[1] or cy < 0 or cy >= mask.shape[0]:
        return False
    return bool(mask[cy, cx])


def find_safe_spawn(mask, x, y, label=""):
    """Mask-based spawn validation. If (x, y) lands on a wall/ramp/object,
    spiral outward at 0.2m steps until a flat-ground cell is found."""
    if is_spawn_safe(mask, x, y):
        return x, y
    for radius in np.arange(0.2, 3.0, 0.2):
        for angle in range(0, 360, 30):
            nx = x + radius * np.cos(np.radians(angle))
            ny = y + radius * np.sin(np.radians(angle))
            if is_spawn_safe(mask, nx, ny):
                if label:
                    print(f"    {label}: moved from "
                          f"({x:.1f},{y:.1f}) to ({nx:.1f},{ny:.1f})")
                return nx, ny
    return x, y  # fallback — caller's hardcoded position will be used


def _resample(path, step):
    if not path or len(path) < 2:
        return list(path) if path else []
    pts = [path[0]]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        if d < 0.01:
            continue
        n = max(1, int(d / step))
        for j in range(1, n + 1):
            t = j / n
            pts.append((path[i - 1][0] + t * dx, path[i - 1][1] + t * dy))
    return pts


def _build_terrain(mesh):
    from PIL import Image
    vh = VIEW_HALF
    cache_path = Path(STL_PATH).with_suffix('.test_terrain.png')
    stl_mtime = Path(STL_PATH).stat().st_mtime
    if cache_path.exists() and cache_path.stat().st_mtime >= stl_mtime:
        print("Loading cached test terrain...")
        img = np.array(Image.open(cache_path)).astype(np.float64) / 65535.0
        return img, [-vh, vh, -vh, vh]
    print("Building terrain hillshade (0.025m resolution, caching as PNG)...")
    res = 0.025
    x = np.arange(-vh, vh, res)
    y = np.arange(-vh, vh, res)
    gw, gh = len(x), len(y)
    origins, dirs = [], []
    for iy in range(gh):
        for ix in range(gw):
            origins.append([x[ix], y[iy], 500])
            dirs.append([0, 0, -1])
    origins = np.array(origins)
    dirs = np.array(dirs)
    locs, idx_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=True)
    elev = np.full((gh, gw), np.nan)
    for i in range(len(origins)):
        hits = locs[idx_ray == i]
        if len(hits):
            elev[i // gw, i % gw] = hits[:, 2].max()
    m = np.isnan(elev)
    if m.any():
        _, ni = distance_transform_edt(
            m, return_distances=True, return_indices=True)
        elev = elev[tuple(ni)]
    dzdx = np.gradient(elev, res, axis=1)
    dzdy = np.gradient(elev, res, axis=0)
    nm = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    la, le = np.radians(315), np.radians(25)
    shade = np.clip(
        -dzdx / nm * np.cos(le) * np.cos(la)
        - dzdy / nm * np.cos(le) * np.sin(la)
        + 1 / nm * np.sin(le), 0, 1)
    terrain = 0.15 + 0.85 * shade
    img16 = (terrain * 65535).astype(np.uint16)
    Image.fromarray(img16).save(str(cache_path))
    print(f"  Cached to {cache_path.name}")
    return terrain, [-vh, vh, -vh, vh]


# ── Visualization ──────────────────────────────────────────────────

def _is_headless_backend():
    """True when matplotlib is on Agg (or any non-interactive backend)
    so we cannot construct a real Qt5/OpenGL widget. The benchmark falls
    back to a matplotlib-only single-axes figure in that case — same
    halo overlay, no embedded 3D panel."""
    if os.environ.get("MPLBACKEND", "").lower().startswith("agg"):
        return True
    return matplotlib.get_backend().lower().startswith("agg")


def _decorate_map_axes(ax, agents, fig):
    """Add the benchmark's per-agent markers / paths / goals / trails,
    ramp zone rectangles, agent-color legend, and figure suptitle on
    top of a freshly-built ``ax_map``. Shared between the legacy
    matplotlib-only ``_setup_figure`` path and the new Qt path so both
    layouts look identical."""
    cm_view = VIEW_HALF
    ax.set_xlim(-cm_view, cm_view)
    ax.set_ylim(-cm_view, cm_view)

    # Per-agent map markers / goals / paths / traveled trails.
    # Marker SHAPES + sizes mirror the GUI conventions:
    #   filled circle s=10 (agent), star s=18 alpha=0.5 (goal),
    #   dashed lw=1.5 alpha=0.4 (planned path),
    #   solid  lw=2.5 alpha=0.8 (traveled trail).
    markers, goal_mks, path_lines, trav_lines = [], [], [], []
    for a in agents:
        mk, = ax.plot([], [], "o", color=a.color, markersize=10, zorder=10)
        gk, = ax.plot([], [], "*", color=a.color, markersize=18,
                       alpha=0.5, zorder=9)
        pl, = ax.plot([], [], "--", color=a.color, lw=1.5,
                       alpha=0.4, zorder=7)
        tl, = ax.plot([], [], "-", color=a.color, lw=2.5,
                       alpha=0.8, zorder=8)
        markers.append(mk)
        goal_mks.append(gk)
        path_lines.append(pl)
        trav_lines.append(tl)

    # Ramp zone rectangles (only the 5 unique ramp x positions).
    for r in RAMPS[:5]:
        ax.add_patch(Rectangle(
            (r["x"] - 0.6, 1.5), 1.2, 2.5, lw=1,
            edgecolor=COLORS[r["id"] - 1], facecolor="none",
            linestyle=":", alpha=0.3, zorder=5))
        ax.text(r["x"], 1.3,
                f"{r['slope']}°",
                ha="center", va="top", fontsize=9,
                color=COLORS[r["id"] - 1], alpha=0.7)

    legend_handles = [
        Patch(facecolor=c, label=f"R{r['id']} ({r['slope']}d)")
        for c, r in zip(COLORS, RAMPS)]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=3,
              facecolor="#333", labelcolor="white", edgecolor="#555")

    pct = int(round(np.tan(np.radians(sim.TRAVERSABLE_MAX_DEG)) * 100))
    fig.suptitle(f"PCA Technique with {pct}% Relative Grade Threshold",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    ax.set_title("", color="white", fontsize=14, fontweight="bold")

    return markers, goal_mks, path_lines, trav_lines


def _setup_figure_qt(agents, mesh, terrain_img=None, terrain_extent=None,
                     headless=False):
    """Build the benchmark visualization, mirroring the single-agent GUI.

    Layout (interactive Qt5Agg):
      Left  — matplotlib FigureCanvasQTAgg (terrain + per-agent markers/
               paths/trails/goals + UNION-of-all-agents persistent
               obstacle halo via ``GlobalCostmap.render_overlay``-style
               smooth-alpha red fade)
      Right — PyQtGraph GLViewWidget (Agent 1's 3D point cloud + ghost STL)

    Layout (headless Agg, ``headless=True``):
      Single matplotlib figure with the same map axes; the GL panel is
      replaced with a ``MockThreeDView`` so ``update_3d_panel`` calls
      remain no-ops. The gif export captures just the map (no 3D panel
      embedded) — the halo overlay carries the obstacle picture.

    Returns the same 9-tuple shape ``_setup_figure`` did so the rest of
    the benchmark plumbing stays unchanged.
    """
    import sim_view

    if headless:
        # Matplotlib-only fallback. Single ax_map with a halo overlay
        # imshow, sized to the persistent GlobalCostmap extent so the
        # overlay's natural extent matches.
        fig = plt.figure(figsize=(11, 11), facecolor="#1e1e1e")
        try:
            fig.canvas.manager.set_window_title("Ramp Crossing Test")
        except Exception:
            pass
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#222")
        if terrain_img is not None and terrain_extent is not None:
            ax.imshow(terrain_img, origin="lower", extent=terrain_extent,
                      cmap="gray", aspect="equal", vmin=0, vmax=1)
        global_img = ax.imshow(
            np.zeros((10, 10, 4), dtype=np.uint8),
            origin="lower",
            extent=[-sim.GLOBAL_HALF, sim.GLOBAL_HALF,
                    -sim.GLOBAL_HALF, sim.GLOBAL_HALF],
            aspect="equal", alpha=0.85, zorder=3,
            interpolation="none")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        ax.grid(False, which="both")
        ax.minorticks_off()
        for s in ax.spines.values():
            s.set_edgecolor("#555")
        status = fig.text(0.5, 0.01, "", ha="center", fontsize=10,
                          color="white", style="italic")

        three_d_view = sim_view.MockThreeDView()
        container = None
    else:
        handles = sim_view.build_three_panel_qt_widget(
            title="Ramp Crossing Test",
            terrain_img=terrain_img,
            terrain_extent=terrain_extent,
            figsize=(11, 11),
            ghost_mesh=mesh,
        )
        fig = handles["fig"]
        ax = handles["ax_map"]
        global_img = handles["global_img"]
        status = handles["status"]
        three_d_view = handles["three_d"]
        container = handles["widget"]

    # The persistent GlobalCostmap covers ±GLOBAL_HALF (=10 m) so the
    # halo overlay's extent stays at its native size; the map's xlim/
    # ylim of ±VIEW_HALF crops the visible region to the interesting bit.
    global_img.set_extent([-sim.GLOBAL_HALF, sim.GLOBAL_HALF,
                           -sim.GLOBAL_HALF, sim.GLOBAL_HALF])

    markers, goal_mks, path_lines, trav_lines = _decorate_map_axes(
        ax, agents, fig)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Bundle every artist + 3D handle the update loop needs. ``ax_bev``
    # / ``ax3d`` / ``bev_handle`` are kept as None for back-compat with
    # callers that probe them; the new pipeline doesn't touch them.
    viz = {
        "ax_map": ax,
        "ax_bev": None,
        "ax3d": None,
        "global_img": global_img,
        "bev_handle": None,
        "three_d": three_d_view,
        "3d_state": None,           # GL path doesn't need ThreeDState
        "markers": markers,
        "goal_mks": goal_mks,
        "path_lines": path_lines,
        "trav_lines": trav_lines,
        "status": status,
        # Blit state — populated lazily on first ``_update_viz`` call so
        # the static terrain bg is snapshotted *after* the figure has
        # been shown / sized.
        "blit_bg": None,
        "container": container,
        "headless": headless,
    }
    return (fig, ax, markers, goal_mks, path_lines, trav_lines,
            status, global_img, viz)


def _setup_figure(agents, terrain_img=None, terrain_extent=None):
    """Legacy 3-panel matplotlib figure (mplot3d + BEV + map).

    Retained as a fallback for any caller that explicitly wants the
    pre-port look; the active main() uses ``_setup_figure_qt`` for both
    Qt5Agg and Agg paths so the GUI + benchmark share the same render
    system.
    """
    import sim_view

    handles = sim_view.build_three_panel_figure(
        title="Ramp Crossing Test",
        terrain_img=terrain_img,
        terrain_extent=terrain_extent,
        figsize=(22, 11),
    )
    fig = handles["fig"]
    ax = handles["ax_map"]
    ax_bev = handles["ax_bev"]
    ax3d = handles["ax3d"]
    bev_handle = handles["bev_handle"]
    status = handles["status"]

    cm_view = VIEW_HALF
    handles["global_img"].set_extent([-cm_view, cm_view, -cm_view, cm_view])

    markers, goal_mks, path_lines, trav_lines = _decorate_map_axes(
        ax, agents, fig)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    viz = {
        "ax_map": ax,
        "ax_bev": ax_bev,
        "ax3d": ax3d,
        "global_img": handles["global_img"],
        "bev_handle": bev_handle,
        "three_d": None,
        "3d_state": sim_view.ThreeDState(),
        "markers": markers,
        "goal_mks": goal_mks,
        "path_lines": path_lines,
        "trav_lines": trav_lines,
        "status": status,
        "blit_bg": None,
        "container": None,
        "headless": True,
        "legacy_mpl3d": True,
    }
    return (fig, ax, markers, goal_mks, path_lines, trav_lines,
            status, handles["global_img"], viz)


def _render_costmap_overlay(agents, size=400):
    """Per-agent dot overlay (LEGACY — kept for `_setup_figure` fallback).

    The active rendering path is ``_render_halo_overlay`` below, which
    matches the single-agent GUI's smooth-fade red halo. This dot-color
    overlay is only used by the legacy 3-panel mpl3d setup, which the
    main flow no longer takes."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    view = float(VIEW_HALF)
    for a in agents:
        cm = a.costmap
        if not cm.obstacles.any():
            continue
        hex_c = a.color.lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        oys, oxs = np.where(cm.obstacles)
        wxs = (oxs + 0.5) * cm.res - cm.half
        wys = (oys + 0.5) * cm.res - cm.half
        pxs = ((wxs + view) / (2 * view) * size).astype(int)
        pys = ((wys + view) / (2 * view) * size).astype(int)
        v = (pxs >= 0) & (pxs < size) & (pys >= 0) & (pys < size)
        img[pys[v], pxs[v]] = [r, g, b, 220]
    return img


def _render_halo_overlay(agents):
    """Smooth-alpha red halo over the UNION of all agents' obstacles.

    Mirrors ``GlobalCostmap.render_overlay`` in the single-agent GUI:
    one ``distance_transform_edt`` on the merged 400×400 obstacle grid,
    then a per-cell weight from 1.0 at the inscribed band down to 0.0
    at the inflation edge. Output is an RGBA uint8 image at the same
    cell pitch as ``GlobalCostmap`` so the imshow's native extent
    [-GLOBAL_HALF, GLOBAL_HALF] aligns 1:1 with the costmap cells.

    Cheap: a single distance transform + a few vectorized array ops.
    """
    # Use the first agent's costmap as the geometry reference. All
    # agents share GLOBAL_RES / GLOBAL_HALF so their grids are
    # identical-shape and OR'ing them is a per-cell bitwise-or.
    if not agents:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    cm0 = agents[0].costmap
    h, w = cm0.h, cm0.w
    union = np.zeros((h, w), dtype=bool)
    for a in agents:
        union |= a.costmap.obstacles
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if not union.any():
        return img
    cell_dist = distance_transform_edt(~union)
    infl_cells_outer = (sim.ROBOT_RADIUS + sim.INFLATION_RADIUS) / cm0.res
    infl_cells_inner = sim.ROBOT_RADIUS / cm0.res
    in_halo = cell_dist <= infl_cells_outer
    denom = max(infl_cells_outer - infl_cells_inner, 1e-3)
    weight = np.clip(
        (infl_cells_outer - cell_dist) / denom, 0.0, 1.0
    ).astype(np.float32)
    alpha = (weight * in_halo).astype(np.float32)
    img[..., 0] = 220
    img[..., 1] = 30
    img[..., 2] = 30
    img[..., 3] = (alpha * 220).astype(np.uint8)
    return img


def _update_viz(agents, markers, goal_mks, path_lines, trav_lines,
                status, phase, costmap_overlay=None, viz=None):
    """Refresh every artist in the benchmark figure.

    Uses matplotlib blit on the map axes (mirrors the GUI's _animate):
    snapshot the static terrain bg once, then per-frame restore the bg
    and ``draw_artist`` the halo overlay + 15 markers + 15 path lines +
    15 trails + 15 goal stars + status text + canvas.blit(ax_map.bbox).
    The 3D panel is updated via ``sim_view.update_3d_panel`` which
    dispatches on the view type (real GL widget or mock no-op).
    """
    import sim_view
    parts = []
    for i, a in enumerate(agents):
        if a.pos:
            markers[i].set_data([a.pos[0]], [a.pos[1]])
        if a.goal:
            goal_mks[i].set_data([a.goal[0]], [a.goal[1]])
        if a.path and a.wp_idx < len(a.path):
            remaining = [(a.pos[0], a.pos[1])] + a.path[a.wp_idx:]
            path_lines[i].set_data(
                [p[0] for p in remaining], [p[1] for p in remaining])
        else:
            path_lines[i].set_data([], [])
        if a.traveled:
            trav_lines[i].set_data(
                [p[0] for p in a.traveled],
                [p[1] for p in a.traveled])
        tag = ("DONE" if a.done
               else "STUCK" if a.stuck
               else f"({a.pos[0]:.0f},{a.pos[1]:.0f})" if a.pos
               else "?")
        parts.append(f"R{a.id}:{tag}")
    status.set_text(f"{phase}  |  " + "  |  ".join(parts))

    if viz is None:
        # Pure legacy path — no halo, no 3D, just the old dot overlay.
        if costmap_overlay is not None:
            costmap_overlay.set_data(_render_costmap_overlay(agents))
        return

    # ── Halo overlay (union of every agent's persistent obstacles) ──
    # Mirrors the single-agent GUI's GlobalCostmap.render_overlay output
    # so both front-ends share the same look.
    halo = _render_halo_overlay(agents)
    viz["global_img"].set_data(halo)

    # Legacy mpl3d path still writes a BEV panel + matplotlib 3D scatter.
    if viz.get("legacy_mpl3d"):
        if costmap_overlay is not None:
            costmap_overlay.set_data(_render_costmap_overlay(agents))
        if viz.get("bev_handle") is not None:
            agents_obs = [(a.last_scan_obs, a.color) for a in agents]
            viz["bev_handle"].set_data(
                sim_view.render_bev_multi(agents_obs))
        a1 = agents[0]
        if (viz.get("ax3d") is not None
                and a1.last_scan_cloud is not None
                and a1.last_scan_origin is not None
                and a1.last_scan_slope_grid is not None
                and a1.last_scan_obs is not None):
            sim_view.update_3d_panel(
                viz["ax3d"], viz["3d_state"],
                a1.last_scan_cloud, a1.last_scan_origin,
                a1.last_scan_sn, a1.last_scan_slope_grid,
                a1.last_scan_obs)
        return

    # ── 3D panel: Agent 1's latest scan, GL fast-path or mock no-op ──
    three_d_view = viz.get("three_d")
    a1 = agents[0]
    if (three_d_view is not None
            and a1.last_scan_cloud is not None
            and a1.last_scan_origin is not None
            and a1.last_scan_slope_grid is not None
            and a1.last_scan_obs is not None):
        sim_view.update_3d_panel(
            three_d_view, None,
            a1.last_scan_cloud, a1.last_scan_origin,
            a1.last_scan_sn, a1.last_scan_slope_grid,
            a1.last_scan_obs)


def _blit_map_axes(viz):
    """Blit the map axes: restore the cached terrain bg, draw_artist
    every dynamic artist (halo + per-agent markers/paths/trails/goals
    + status), and canvas.blit(ax_map.bbox). Replaces the per-frame
    ``fig.canvas.draw()`` that was ~50-100 ms with a ~3-8 ms blit.

    Falls back to ``canvas.draw_idle()`` if the bg snapshot is missing
    (first call) or the blit machinery isn't available on this backend.
    """
    if viz is None:
        return False
    ax = viz["ax_map"]
    fig = ax.figure
    canvas = fig.canvas
    bg = viz.get("blit_bg")

    # Lazily snapshot the static bg the first time we're called. The
    # halo + markers + paths + trails + goals + status are all flagged
    # animated so they aren't baked into the snapshot.
    if bg is None:
        # Mark every per-frame artist as animated *before* we snapshot
        # the bg so the snapshot only contains the static terrain.
        viz["global_img"].set_animated(True)
        for m in viz["markers"]:
            m.set_animated(True)
        for m in viz["goal_mks"]:
            m.set_animated(True)
        for m in viz["path_lines"]:
            m.set_animated(True)
        for m in viz["trav_lines"]:
            m.set_animated(True)
        if viz.get("status") is not None:
            viz["status"].set_animated(True)
        ax.grid(False, which="both")
        try:
            canvas.draw()
            bg = canvas.copy_from_bbox(ax.bbox)
            viz["blit_bg"] = bg
        except Exception:
            return False

    try:
        canvas.restore_region(bg)
        if viz["global_img"].get_visible():
            ax.draw_artist(viz["global_img"])
        for m in viz["trav_lines"]:
            if m.get_visible():
                ax.draw_artist(m)
        for m in viz["path_lines"]:
            if m.get_visible():
                ax.draw_artist(m)
        for m in viz["goal_mks"]:
            if m.get_visible():
                ax.draw_artist(m)
        for m in viz["markers"]:
            if m.get_visible():
                ax.draw_artist(m)
        # Status text lives on the figure, not the map axes — drawing
        # it via figure-level blit would require a separate bg snapshot;
        # cheap-out is to skip it from the blit path and let the next
        # full draw (e.g. _grab_frame) repaint it.
        canvas.blit(ax.bbox)
        return True
    except Exception:
        return False


# ── Phase runner ───────────────────────────────────────────────────

def _grab_frame(fig, viz=None):
    """Rasterize current figure to an RGB numpy array.

    ``fig.canvas.draw()`` is the dominant cost on the benchmark's old
    visualization path (~50-100 ms / frame on a 22x11 figure). We still
    need a full draw before grabbing the gif frame so all the artists
    are present in the rasterized output — there's no cheap way to
    blit a buffer directly to memory while keeping the static terrain.
    The blit fast-path in ``_blit_map_axes`` is reserved for the live
    on-screen frames; only the every-10th gif-capture frame pays the
    full-draw cost.

    Calling ``canvas.draw()`` invalidates the cached blit bg (the
    underlying renderer state changes), so we drop ``viz["blit_bg"]``
    here and let ``_blit_map_axes`` re-snapshot on the next call.
    """
    # Temporarily un-animate so the full draw paints every artist into
    # the canvas buffer (animated artists are otherwise skipped by the
    # draw pass and would only appear after a separate draw_artist +
    # blit). We restore the animated flags after grabbing the frame.
    restored = []
    if viz is not None:
        for key in ("global_img",):
            art = viz.get(key)
            if art is not None and art.get_animated():
                art.set_animated(False)
                restored.append(art)
        for key in ("markers", "goal_mks", "path_lines", "trav_lines"):
            for art in viz.get(key, []) or []:
                if art.get_animated():
                    art.set_animated(False)
                    restored.append(art)
        status = viz.get("status")
        if status is not None and status.get_animated():
            status.set_animated(False)
            restored.append(status)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3].copy()
    # Re-animate for the next blit cycle, and invalidate the cached bg
    # because canvas.draw() repopulated the renderer with the animated
    # artists baked into the buffer.
    for art in restored:
        art.set_animated(True)
    if viz is not None:
        viz["blit_bg"] = None
    return frame


def run_phase(agents, fig, ax, markers, goal_mks, path_lines,
              trav_lines, status, phase_name, costmap_overlay=None,
              frames=None, viz=None, pool=None):
    if frames is None:
        frames = []
    hdr = f"\n{'=' * 60}\n  {phase_name}\n{'=' * 60}"
    print(hdr)
    ax.set_title(phase_name, color="white", fontsize=12, fontweight="bold")

    # Initial scan for every agent — batch through the pool so all 15
    # ray casts run in parallel instead of one at a time on the main thread.
    init_args = [(a.pos[0], a.pos[1], a.heading_along_path(), a.id == 1)
                 for a in agents]
    if pool is not None:
        init_results = pool.map(_worker_scan, init_args)
    else:
        init_results = [a._serial_scan() for a in agents]
    for a, r in zip(agents, init_results):
        print(f"  Agent {a.id} ({a.slope:2d}deg): "
              f"({a.pos[0]:.0f},{a.pos[1]:.0f}) -> "
              f"({a.goal[0]:.0f},{a.goal[1]:.0f})  ...", end="", flush=True)
        ok = a.process_scan_result(r)
        print(" path found" if ok else " NO PATH")
        if not ok:
            a.stall_count = 0
            ok = a.scan_and_plan()  # one serial retry
            if not ok:
                a.stuck = True

    _update_viz(agents, markers, goal_mks, path_lines, trav_lines,
                status, phase_name, costmap_overlay, viz)
    fig.canvas.draw()
    frames.append(_grab_frame(fig, viz))

    for step in range(MAX_STEPS):
        all_done = True
        # ── Phase A: collect agents that need a fresh scan ──
        to_scan = []
        for a in agents:
            if a.done or a.stuck:
                continue
            all_done = False
            if a.needs_scan() or a.wp_idx >= len(a.path):
                to_scan.append(a)

        # ── Phase B: dispatch scans in parallel (or serial fallback) ──
        # Use imap_unordered so the main thread starts merging the first
        # result while the remaining workers are still scanning. Smooths
        # the saw-tooth CPU pattern that pool.map produces.
        scan_failed = set()
        if to_scan:
            if pool is not None:
                args_list = [(i, a.pos[0], a.pos[1], a.heading_along_path(),
                              a.id == 1)
                             for i, a in enumerate(to_scan)]
                # chunksize=1 — every result becomes available as soon
                # as a worker finishes, no batching delay.
                for idx, r in pool.imap_unordered(_worker_scan_indexed,
                                                   args_list, chunksize=1):
                    a = to_scan[idx]
                    ok = a.process_scan_result(r)
                    if not ok:
                        scan_failed.add(id(a))
                        if a.stall_count >= 30:
                            a.stuck = True
                            print(f"  Agent {a.id}: STUCK at "
                                  f"({a.pos[0]:.1f}, {a.pos[1]:.1f})")
            else:
                for a in to_scan:
                    r = a._serial_scan()
                    ok = a.process_scan_result(r)
                    if not ok:
                        scan_failed.add(id(a))
                        if a.stall_count >= 30:
                            a.stuck = True
                            print(f"  Agent {a.id}: STUCK at "
                                  f"({a.pos[0]:.1f}, {a.pos[1]:.1f})")

        # ── Phase C: physics advance for agents whose scans succeeded
        # (or didn't need one this step) ──
        for a in agents:
            if a.done or a.stuck:
                continue
            if id(a) in scan_failed:
                continue  # match original "skip advance after failed scan"
            a.advance()
            if a.done:
                print(f"  Agent {a.id}: ARRIVED at "
                      f"({a.pos[0]:.1f}, {a.pos[1]:.1f})")

        # Every step: blit-only refresh of the live window (cheap —
        # restores the cached terrain bg + draw_artist on the animated
        # artists). Headless gif export captures every 10th step via
        # a full ``canvas.draw()`` + ``_grab_frame``; in LIVE Qt mode
        # that full draw causes a visible flash on screen, so we skip
        # capture there. The gif is the benchmark's batch-mode output,
        # not the interactive view, so it's safe to gate.
        _update_viz(agents, markers, goal_mks, path_lines, trav_lines,
                    status, phase_name, costmap_overlay, viz)
        capture = viz.get("headless", False) and ((step % 10 == 0) or all_done)
        if capture:
            fig.canvas.draw()
            frames.append(_grab_frame(fig, viz))
        else:
            # Blit fast-path on the map axes. Falls back to draw_idle()
            # internally if the backend can't blit.
            ok = _blit_map_axes(viz)
            if not ok:
                fig.canvas.draw_idle()
        # Pump the GUI event loop so the live Qt window stays responsive.
        # No-op cost under Agg.
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        if all_done:
            break

    print(f"\n  {'Agent':>5}  {'Slope':>5}  {'Expected':>8}  "
          f"{'Result':>8}  {'Crossed':>7}  {'Lethal':>6}  {'Match':>5}")
    print(f"  {'-----':>5}  {'-----':>5}  {'--------':>8}  "
          f"{'--------':>8}  {'-------':>7}  {'------':>6}  {'-----':>5}")
    for a in agents:
        a.result = "ARRIVED" if a.done else "STUCK"
        crossed = a.crossed_ramp()
        violations = a.lethal_violations()
        # Allow small violation count from start position / edge artifacts
        lethal_ok = violations <= 30
        if "Cross" in phase_name:
            if a.expected == "PASS":
                ok = a.done and crossed and lethal_ok
            else:
                ok = not crossed and lethal_ok
        else:
            ok = a.done and lethal_ok
        match = "OK" if ok else "FAIL"
        print(f"  R{a.id:>4}  {a.slope:>4}d  {a.expected:>8}  "
              f"{a.result:>8}  {'YES' if crossed else 'NO':>7}  {violations:>6}  {match:>5}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="15-agent ramp crossing benchmark. "
                    "Use --agents 5 / 10 / 15 to pick a subset (one, two, "
                    "or all three rows), --phase 1 to skip the route-around-"
                    "walls phase, --no-video to skip the gif export.")
    parser.add_argument("--threshold", "-t", type=int, default=30,
                        choices=[15, 20, 30],
                        help="Grade tolerance: 15 (8.5°), 20 (11.3°), or "
                             "30 (16.7°). Defaults to 30 to match the "
                             "deployed grade_detector.yaml.")
    parser.add_argument("--rays", type=int, default=sim.N_RAYS,
                        help=f"LiDAR rays per scan (default {sim.N_RAYS}).")
    parser.add_argument("--agents", type=int, default=15, choices=[5, 10, 15],
                        help="Agent count: 5 = first row, 10 = first two "
                             "rows, 15 = all (default).")
    parser.add_argument("--phase", choices=["1", "both"], default="both",
                        help="'1' = stop after Phase 1 (ramp crossings); "
                             "'both' (default) = also run Phase 2 (route "
                             "around walls).")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip the gif export at the end. Faster.")
    args = parser.parse_args()

    # Apply threshold globally so workers + main process agree. Mirrors
    # the lidar_sim_gui.main() logic.
    cfg = {15: (6.0, 8.5), 20: (8.0, 11.3), 30: (12.0, 16.7)}
    free_max, deg = cfg[args.threshold]
    sim.GRADE_FREE_MAX = free_max
    sim.GRADE_MODERATE_MAX = deg
    sim.GRADE_STEEP_MAX = deg
    sim.TRAVERSABLE_MAX_DEG = deg
    sim.N_RAYS = args.rays

    # Slice the RAMPS roster to the requested count.
    global RAMPS
    RAMPS = RAMPS[:args.agents]

    print("=" * 60)
    print("  Ramp Crossing Test — Grade Classification Validation")
    print("=" * 60)
    print(f"  Grade threshold: {sim.TRAVERSABLE_MAX_DEG} deg ({args.threshold}% grade)")
    print(f"  Agents: {len(RAMPS)}  ·  Rays/scan: {sim.N_RAYS}  ·  Phase: {args.phase}")
    print()

    print("Loading mesh...")
    mesh = trimesh.load(STL_PATH)
    print(f"  {len(mesh.faces):,} faces")

    # One-time (then cached) precompute: which cells are flat ground.
    # Replaces per-spawn LiDAR-based validation.
    spawn_mask = build_spawn_mask(mesh)

    rays = sim.gen_rays(sim.N_RAYS)

    print("Warming up ray caster...")
    sim.do_scan(mesh, 0, 0, rays[:10], 0.0)

    agents = [Agent(r, mesh, rays, COLORS[i]) for i, r in enumerate(RAMPS)]

    # ── Spin up worker pool: one process per agent (capped at 16) ──
    n_workers = max(1, min(len(agents), (os.cpu_count() or 1), 16))
    pool = None
    if n_workers > 1:
        ctx = get_context("spawn")
        pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(STL_PATH, sim.N_RAYS, sim.TRAVERSABLE_MAX_DEG,
                      sim.GRADE_FREE_MAX, sim.GRADE_MODERATE_MAX,
                      sim.GRADE_STEEP_MAX),
        )
        print(f"  Parallel scan pool: {n_workers} workers")

    # Build (or load cached) terrain hillshade for the left map panel.
    # Mirrors what DualNavGUI does at startup so both front-ends look identical.
    terrain_img, terrain_extent = _build_terrain(mesh)

    # Choose between the new Qt5Agg path (PyQtGraph 3D + single-axes
    # matplotlib map with halo overlay, same render stack as the GUI)
    # and the headless matplotlib-only fallback (MockThreeDView; gif
    # captures the map axes only — no embedded 3D panel).
    headless = _is_headless_backend()
    fig, ax, markers, goal_mks, plines, tlines, status, cm_overlay, viz = \
        _setup_figure_qt(agents, mesh,
                         terrain_img=terrain_img,
                         terrain_extent=terrain_extent,
                         headless=headless)
    plt.ion()
    if headless:
        # Under Agg, show() is a no-op but harmless. The capture comes
        # from canvas.buffer_rgba() inside _grab_frame.
        try:
            plt.show(block=False)
        except Exception:
            pass
    else:
        # Qt path: pop the embedded matplotlib+GL container window.
        # plt.show() would only surface the mpl canvas's stand-alone
        # manager (without the GL view), so we drive the container
        # directly. The Qt event loop is pumped via canvas.flush_events()
        # inside run_phase — no app.exec_() here.
        container = viz.get("container")
        if container is not None:
            container.resize(1400, 800)
            container.show()

    frames = []

    # ── Phase 1: Cross the ramps ───────────────────────────────────
    rng = np.random.RandomState(42)  # deterministic randomization
    for a in agents:
        # Keep starts away from walls (y=-2.6) and platform edges
        sx = min(a.ramp_x, 3.0)
        if a.id in (4, 9, 14):
            sx = a.ramp_x - 1.0
        if a.id <= 5:
            sy = -1.0
        elif a.id <= 10:
            sy = -1.5
        else:
            sy = -2.0
        # Goals: aligned with agent's ramp so direct path crosses it
        # Small random offset so agents don't stack exactly
        gx = a.ramp_x + rng.uniform(-0.3, 0.3)
        gy = 5.5 + rng.uniform(-0.3, 0.3)
        a.reset(sx, sy, gx, gy)

    # Resolve spawn/goal positions via the cached Z-slice mask (no LiDAR).
    print("  Picking spawn positions from cached mask...")
    for a in agents:
        sx, sy = find_safe_spawn(spawn_mask, a.pos[0], a.pos[1],
                                 f"R{a.id} start")
        gx, gy = find_safe_spawn(spawn_mask, a.goal[0], a.goal[1],
                                 f"R{a.id} goal")
        a.pos = (sx, sy)
        a.goal = (gx, gy)
        a.traveled = [(sx, sy)]

    run_phase(agents, fig, ax, markers, goal_mks, plines, tlines,
              status, "Phase 1: Cross Ramps (1P 2P 3N 4N 5N)",
              cm_overlay, frames, viz, pool=pool)

    for a in agents:
        a.phase1_result = a.result
    phase1_crossed = {a.id: a.crossed_ramp() for a in agents}

    # ── Verify ramp obstacle coverage ──────────────────────────────
    # Only count obstacles inside the platform (exclude platform bounds)
    print("\n  Ramp obstacle check (excluding platform bounds):")
    for a in agents:
        cm = a.costmap
        oys, oxs = np.where(cm.obstacles)
        wxs = (oxs + 0.5) * cm.res - cm.half
        wys = (oys + 0.5) * cm.res - cm.half
        # Only count obstacles well inside platform, near the ramp
        in_platform = (wxs > -3.5) & (wxs < 9.5) & (wys > -4.0) & (wys < 5.5)
        in_ramp = (in_platform &
                   (np.abs(wxs - a.ramp_x) < 0.3) &
                   (wys > 1.5) & (wys < 3.8))
        n_ramp_obs = in_ramp.sum()
        expect_obs = "should have obs" if a.expected == "NO-PASS" else "should be CLEAR"
        # Allow up to 25 aliasing artifacts on traversable ramps
        status_str = "OK" if ((a.expected == "PASS" and n_ramp_obs < 25) or
                              (a.expected == "NO-PASS" and n_ramp_obs > 0)) else "FAIL"
        print(f"    R{a.id} ({a.slope:2d}deg): {n_ramp_obs:4d} obs in ramp zone — {expect_obs} [{status_str}]")

    # Hold last frame of phase 1
    for _ in range(10):
        frames.append(frames[-1])

    # Save a Phase-1-end snapshot of the costmap state (before Phase 2
    # spawns reset). This shows the costmap built up purely from
    # crossing the ramps, so we can compare to the Phase-2 final.
    p1_out = str(Path(__file__).resolve().parent / "test_ramp_results_phase1.png")
    fig.savefig(p1_out, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    print(f"  Phase 1 snapshot saved to: {p1_out}")

    # Save a clean B&W costmap (union of all agents' obstacles, no
    # inflation, no colors) for unambiguous review.
    p1_bw = str(Path(__file__).resolve().parent / "test_ramp_costmap_phase1_bw.png")
    _save_bw_costmap(agents, p1_bw)
    print(f"  Phase 1 B&W costmap saved to: {p1_bw}")

    # ── Phase 2: Route behind walls ────────────────────────────────
    if args.phase == "both":
        for tl in tlines:
            tl.set_data([], [])
        for pl in plines:
            pl.set_data([], [])

        for a in agents:
            # Start above the new wall, staggered rows
            if a.id <= 5:
                sy = 6.0
            elif a.id <= 10:
                sy = 5.5
            else:
                sy = 5.0
            sx = min(a.ramp_x + 0.2, 3.0)
            # Goals: safe zone below walls, away from edges
            gx = rng.uniform(-2.0, 2.5)
            gy = -4.5 + rng.uniform(0, 0.5)
            a.reset(sx, sy, gx, gy, clear_costmap=True)

        print("  Picking spawn positions from cached mask...")
        for a in agents:
            sx, sy = find_safe_spawn(spawn_mask, a.pos[0], a.pos[1],
                                     f"R{a.id} start")
            gx, gy = find_safe_spawn(spawn_mask, a.goal[0], a.goal[1],
                                     f"R{a.id} goal")
            a.pos = (sx, sy)
            a.goal = (gx, gy)
            a.traveled = [(sx, sy)]

        run_phase(agents, fig, ax, markers, goal_mks, plines, tlines,
                  status, "Phase 2: Route Behind Walls",
                  cm_overlay, frames, viz, pool=pool)

        # Hold last frame
        for _ in range(10):
            frames.append(frames[-1])
    else:
        # Phase 1 only — surface Phase 1 results in the summary's
        # "Ph2" column so the table is still readable.
        for a in agents:
            a.result = a.phase1_result

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Grade threshold: {sim.TRAVERSABLE_MAX_DEG} deg (20% grade)\n")
    print(f"  {'Ramp':>4}  {'Slope':>5}  {'Expected':>8}  "
          f"{'Ph1':>8}  {'Crossed':>7}  {'Ph2':>8}")
    print(f"  {'----':>4}  {'-----':>5}  {'--------':>8}  "
          f"{'--------':>8}  {'-------':>7}  {'--------':>8}")
    for a in agents:
        print(f"  R{a.id:>3}  {a.slope:>4}d  {a.expected:>8}  "
              f"{a.phase1_result:>8}  "
              f"{'YES' if phase1_crossed[a.id] else 'NO':>7}  "
              f"{a.result:>8}")

    out = str(Path(__file__).resolve().parent / "test_ramp_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    print(f"\n  Results figure saved to: {out}")

    if args.phase == "both":
        p2_bw = str(Path(__file__).resolve().parent / "test_ramp_costmap_phase2_bw.png")
        _save_bw_costmap(agents, p2_bw)
        print(f"  Phase 2 B&W costmap saved to: {p2_bw}")

    # ── Save video ─────────────────────────────────────────────────
    if not args.no_video:
        from PIL import Image
        vid_path = str(Path(__file__).resolve().parent / "test_ramp_crossing.gif")
        print(f"  Saving {len(frames)} frames to {vid_path} ...")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(vid_path, save_all=True, append_images=imgs[1:],
                     duration=8, loop=0)  # ~120fps
        print(f"  Video saved to: {vid_path}")
    plt.close("all")
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
