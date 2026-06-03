#!/usr/bin/env python3
"""BEHAVIOR TREE Sim — Chaplygin-sleigh robot in a corridor maze.

Physics: two rear knife-edge wheels + frictionless front caster.
Controller: per-wheel force allocation (analog of nav2 controller_server / DWB).
Planner: Dijkstra over the live discovered occupancy grid (analog of planner_server).
Decision tree: 8-state recovery automaton ported from AutoNav_25-26 `path_following`
branch (`isaac_ros-dev/src/slam/behavior_trees/bt_nav.xml`):

    NORMAL_FOLLOWING
    FORWARD_BLOCKED_BREADCRUMB_REVERSE
    FORWARD_BLOCKED_WAIT_FOR_REPLAN
    GOAL_BEND
    BACKUP_RECOVERY
    GRADIENT_ESCAPE
    CLEAR_COSTMAP_RECOVERY
    WAIT_TRANSIENT_RECOVERY

Render: matplotlib GridSpec — main tracking camera, top-right minimap of the
maze + discovered %, bottom-right text panel with the live BT state and
controller telemetry.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge


# ── Robot constants (real-robot values from AutoNav_25-26 URDF + user spec) ──
ROBOT_MASS_KG       = 35.0          # 77 lb
COM_OFFSET_M        = 0.25          # rear axle → COM (user-specified)
WHEELBASE_M         = 0.39          # rear axle → front caster (URDF)
TRACK_WIDTH_M       = 0.54          # left wheel ↔ right wheel (URDF)
WHEEL_RADIUS_M      = 0.20          # rear driven wheels (URDF)
CASTER_RADIUS_M     = 0.09          # front caster (URDF)
FOOTPRINT_HALF_W    = 0.21          # half-width for collision (URDF footprint)
FOOTPRINT_LEN_BACK  = 0.10          # body extends this far behind rear axle
FOOTPRINT_LEN_FWD   = WHEELBASE_M + 0.05  # body extends a bit past caster

# Inertia about COM ≈ rectangle (l = footprint length, w = footprint width).
_L_FP = FOOTPRINT_LEN_BACK + FOOTPRINT_LEN_FWD
_W_FP = 2 * FOOTPRINT_HALF_W
INERTIA_COM         = ROBOT_MASS_KG * (_L_FP * _L_FP + _W_FP * _W_FP) / 12.0
# Inertia about the rear-axle midpoint (parallel-axis).
INERTIA_REAR        = INERTIA_COM + ROBOT_MASS_KG * COM_OFFSET_M * COM_OFFSET_M

# Per-wheel force cap. URDF allows 200 N·m at 0.2 m radius = 1000 N which is
# wildly over-spec for a 35 kg sleigh; pick a number the controller can
# actually saturate without launching the body off the floor.
F_WHEEL_MAX_N       = 200.0
F_WHEEL_MIN_N       = -120.0        # reverse capability per wheel

# Damping — air drag + drivetrain friction.
LIN_DAMP            = 6.0           # N per (m/s)
ANG_DAMP            = 2.0           # N·m per (rad/s)


# ── Sensor ──
SENSOR_RANGE_M      = 5.0           # 5 m hemisphere — short range forces
                                    # the robot to commit to corridors
                                    # before seeing what's around the
                                    # next turn, driving BT firings.
SENSOR_FOV_RAD      = math.pi       # full hemisphere in front
# Per-ray probability that a wall voxel along the ray is actually
# detected. < 1.0 simulates noisy lidar / occlusion / glass — missed
# hits mark the cell as FREE_KNOWN and the ray continues. Multiple rays
# usually rediscover the wall on a later sweep.
WALL_DETECTION_PROB = 0.85
SENSOR_NUM_RAYS     = 81            # ~2.25° resolution — enough for a 2D
                                    # maze, halves the per-frame aaline
                                    # cost vs. the original 121 rays.


# ── Maze ──
CELL_SIZE_M         = 5.0           # corridor spacing (user-specified)
WALL_THICKNESS_M    = 0.25
DEFAULT_MAZE_CELLS  = 7             # 7×7 cells → 35 m × 35 m world
                                    # More corridors = more dead-ends so
                                    # the BT recovery branches fire more
                                    # often during a single run.


# ── Costmap (planner discovery grid) ──
GRID_RES_M          = 0.40          # discovered-occupancy cell size
INFLATION_CELLS     = 2             # warning halo thickness in grid cells
                                    # (TRIGGERS gradient_escape, does NOT
                                    # block planner)
# Map-padder corridor: half-side of the per-tick "local window" around the
# robot, in metres. Matches `local_window_radius_m` in
# AutoNav_25-26/isaac_ros-dev/src/map_padder/map_padder/map_padder_node.py
# (default 3.0 m).
PADDER_LOCAL_WINDOW_M = 3.0


# ── Controller (analog of nav2 controller_server) ──
LOOKAHEAD_M         = 1.20
DESIRED_SPEED_MPS   = 0.75          # user-specified cruise speed
# DWB obstacle-aware critic: penalises candidate trajectories that pass
# within DWB_CRITIC_RADIUS_CELLS of a discovered WALL_KNOWN cell. Cell 0
# (the wall itself) is treated as a collision and the candidate is
# rejected outright. The critic samples N candidates around the pure-
# pursuit baseline, simulates each forward DWB_HORIZON_S seconds at
# DWB_HORIZON_DT_S resolution, and picks the lowest combined score
# (path-alignment + obstacle penalty).
DWB_CRITIC_RADIUS_CELLS = 3
DWB_CRITIC_WEIGHT       = 0.25      # per-cell proximity weight. Penalty
                                    # values are 1–4 per step × ~4 steps
                                    # so total raw penalty maxes ~16;
                                    # weight 0.25 → up to 4 m of added
                                    # cost, comparable to path_err range.
DWB_HORIZON_S           = 0.8
DWB_HORIZON_DT_S        = 0.2
DWB_V_DELTAS            = (0.35, 0.6, 1.0, 1.25)     # multipliers on baseline v
DWB_W_DELTAS            = (-0.6, -0.25, 0.0, 0.25, 0.6)  # additive ω offsets
APPROACH_SLOW_M     = 1.5
GOAL_TOLERANCE_M    = 0.45
KP_LIN, KD_LIN      = 35.0, 8.0
KP_ANG, KD_ANG      = 22.0, 4.0


# ── Behavior-tree thresholds (mirror nav2_paramsv2.yaml progress_checker etc.) ──
PROGRESS_STALL_SEC  = 4.0           # no 0.10 m progress in 4 s → BACKUP
PROGRESS_DIST_M     = 0.10
PATH_BEHIND_PERSIST_SEC = 0.4       # "path behind body" must hold this long
                                    # before tripping FORWARD_BLOCKED. Short
                                    # window — long enough that brief mid-turn
                                    # planner wobbles don't trip it, but tight
                                    # enough that the BT fires before the
                                    # robot has time to spin most of the way
                                    # toward the carrot under controller alone.
BACKUP_DIST_M       = 1.00          # how far BACKUP_RECOVERY drives
BACKUP_SPEED        = 0.55          # backwards cruise speed (recoveries)
BREADCRUMB_SPEED    = 0.65          # bcrumb reverse — fastest of the three;
                                    # robot is retracing a known-clear path
                                    # so it can move with confidence.
GRADIENT_ESC_SPEED  = 0.65          # speed during gradient escape (close
                                    # to cruise — user-requested less
                                    # dramatic slowdown).
GRADIENT_POST_ESCAPE_M = 0.7        # how much further to drive after the
                                    # robot exits the halo, before starting
                                    # the alignment-to-path phase.
GRADIENT_ESC_SEC    = 8.0
WAIT_SEC            = 1.0
BREADCRUMB_DROP_M   = 0.40
BREADCRUMB_MAX      = 200           # ~80 m of recorded history at 0.40 m
                                    # drop — long enough that a robot
                                    # trapped past a waypoint can reverse
                                    # all the way to earlier waypoints
                                    # along its driven path.
BREADCRUMB_CONSUME_LIMIT = 15       # max breadcrumbs eaten per single
                                    # BREADCRUMB_REVERSE session before
                                    # the BT gives up and falls through
                                    # to GOAL_BEND-away-from-costmap.
GOAL_BEND_DURATION_SEC = 1.5        # transient hold time on a bent goal
TRAIL_DROP_M        = 0.15          # solid-line trail sample distance
GOAL_BEND_RAD       = 1.05          # ≈60° intermediate offset
REPLAN_PERIOD_S     = 1.0
BT_TICK_PERIOD_S    = 0.10


# ── Sim integration ──
PHYS_DT             = 1.0 / 240.0   # ~4 ms physics steps
RENDER_FPS          = 30            # locked render rate (user-specified)
SUBSTEPS_PER_FRAME  = int(round((1.0 / RENDER_FPS) / PHYS_DT))


# ──────────────────────────────────────────────────────────────────────────────
#                                   Maze
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Maze:
    """World geometry container. Two generators feed it:
      • `generate(n_cells, seed)`  — DFS perfect-maze on an n×n grid.
      • `generate_track(seed)`     — wavy annular track with dead-ends.
    Both produce the same downstream contract: `walls` (segment list),
    `wall_voxel_mask` (rasterised), `size_m` (discovery-grid extent),
    and `default_start_xy` / `default_goal_xy` (where Sim drops the
    robot + goal).
    """
    walls: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Discovery-grid extent in metres (square world).
    size_m: float = 35.0
    # Where Sim.build spawns the robot and places the goal. The
    # generators set these; Sim never has to know about cells or angles.
    default_start_xy: tuple[float, float] = (2.5, 2.5)
    default_goal_xy: tuple[float, float] = (32.5, 32.5)
    # Optional list of consecutive goals (for the track layout's
    # 4-waypoint loop mission). If non-empty, Sim treats these as a
    # mission and advances through them; otherwise it just uses
    # default_goal_xy as the single goal.
    default_goal_waypoints: list[tuple[float, float]] = field(default_factory=list)
    # Only meaningful for the grid generator; kept for HUD / legacy.
    n_cells: int = 0
    cell: float = CELL_SIZE_M
    # Voxelised wall presence — same shape as DiscoveryGrid.cells.
    wall_voxel_mask: Optional[np.ndarray] = None

    def add_obstacles(self, rng, n: int):
        """Drop `n` rectangular obstacles into random corridor cells.

        Each obstacle is a 4-segment rectangle of side 1.2–2.8 m, placed
        with a 0.6 m margin from the cell's bounding walls. We skip the
        start cell (0, 0) and the goal cell (n_cells−1, n_cells−1) so
        the robot can always set off and the goal stays open.

        These obstacles create *surprise dead-ends* — the robot only
        sees them once the 5 m lidar bumps into them — which is exactly
        what fires the BT recovery branches (BACKUP_RECOVERY,
        CLEAR_AROUND_ROBOT, GOAL_BEND, …). Some cells may end up fully
        blocked; that's intentional, the planner falls through to
        CLEAR_GLOBAL / GOAL_BEND / round-robin as designed.
        """
        c = self.cell
        margin = 0.6
        placed = 0
        attempts = 0
        while placed < n and attempts < n * 50:
            attempts += 1
            i = rng.randrange(self.n_cells)
            j = rng.randrange(self.n_cells)
            if (i, j) in ((0, 0), (self.n_cells - 1, self.n_cells - 1)):
                continue
            # Smaller obstacles (1.0-2.0 m) leave enough corridor for
            # the robot to pass on one side most of the time. Some
            # combinations with the maze walls still fully block —
            # exactly the surprise dead-end scenario we want.
            w = rng.uniform(1.0, 2.0)
            h = rng.uniform(1.0, 2.0)
            max_off_x = c - w - 2 * margin
            max_off_y = c - h - 2 * margin
            if max_off_x <= 0 or max_off_y <= 0:
                continue
            x0 = i * c + margin + rng.uniform(0, max_off_x)
            y0 = j * c + margin + rng.uniform(0, max_off_y)
            x1, y1 = x0 + w, y0 + h
            self.walls.append((x0, y0, x1, y0))
            self.walls.append((x1, y0, x1, y1))
            self.walls.append((x1, y1, x0, y1))
            self.walls.append((x0, y1, x0, y0))
            placed += 1
        # Re-rasterise the wall voxel mask so the sensor + renderer pick
        # the new obstacles up.
        self._voxelize_walls()

    def _voxelize_walls(self):
        """Rasterise every wall segment into the discovery-grid cells.
        Each segment is walked in half-cell steps and every cell it
        passes through gets marked. This produces a 1-cell-thick wall
        skeleton at GRID_RES_M resolution."""
        n = int(math.ceil(self.size_m / GRID_RES_M))
        mask = np.zeros((n, n), dtype=np.bool_)
        step = GRID_RES_M * 0.5
        for (x0, y0, x1, y1) in self.walls:
            L = math.hypot(x1 - x0, y1 - y0)
            steps = max(2, int(L / step) + 1)
            for k in range(steps + 1):
                t = k / steps
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                i = int(x / GRID_RES_M)
                j = int(y / GRID_RES_M)
                if 0 <= i < n and 0 <= j < n:
                    mask[i, j] = True
        self.wall_voxel_mask = mask

    @classmethod
    def generate(cls, n_cells: int, seed: int) -> "Maze":
        """Recursive-backtracker maze. Walls between cells are line segments;
        the outer border is always present.

        Coordinate convention: cell (i, j) → lower-left corner at
        (i*cell, j*cell). i = column (east), j = row (north).
        """
        rng = random.Random(seed)
        # Each cell has 4 walls: N, E, S, W. True = wall present.
        N, E, S, W = 0, 1, 2, 3
        walls = [[[True] * 4 for _ in range(n_cells)] for _ in range(n_cells)]
        visited = [[False] * n_cells for _ in range(n_cells)]
        stack = [(0, 0)]
        visited[0][0] = True
        while stack:
            i, j = stack[-1]
            neighbors = []
            if j + 1 < n_cells and not visited[i][j + 1]:
                neighbors.append((i, j + 1, N, S))
            if i + 1 < n_cells and not visited[i + 1][j]:
                neighbors.append((i + 1, j, E, W))
            if j - 1 >= 0 and not visited[i][j - 1]:
                neighbors.append((i, j - 1, S, N))
            if i - 1 >= 0 and not visited[i - 1][j]:
                neighbors.append((i - 1, j, W, E))
            if not neighbors:
                stack.pop()
                continue
            ni, nj, w_here, w_there = rng.choice(neighbors)
            walls[i][j][w_here] = False
            walls[ni][nj][w_there] = False
            visited[ni][nj] = True
            stack.append((ni, nj))

        # Convert wall flags → segment list. Each segment given as (x0,y0,x1,y1).
        c = CELL_SIZE_M
        seg = []
        # Border (outer): always full.
        size = n_cells * c
        seg.append((0, 0, size, 0))
        seg.append((0, size, size, size))
        seg.append((0, 0, 0, size))
        seg.append((size, 0, size, size))
        for i in range(n_cells):
            for j in range(n_cells):
                x0, y0 = i * c, j * c
                # We only emit interior walls; border was added above.
                if walls[i][j][N] and j + 1 < n_cells:
                    seg.append((x0, y0 + c, x0 + c, y0 + c))
                if walls[i][j][E] and i + 1 < n_cells:
                    seg.append((x0 + c, y0, x0 + c, y0 + c))
        # Dedup (each interior wall is shared by two cells but we emitted only
        # from the south/west side).
        c = CELL_SIZE_M
        size = n_cells * c
        m = cls(
            walls=seg,
            size_m=size,
            n_cells=n_cells,
            cell=c,
            default_start_xy=(c * 0.5, c * 0.5),
            default_goal_xy=(c * (n_cells - 0.5), c * (n_cells - 0.5)),
        )
        m._voxelize_walls()
        return m

    @classmethod
    def generate_track(cls, seed: int) -> "Maze":
        """Wavy 10 m-wide annular track with multiple ~10 m dead-end
        pockets and a 4-waypoint mission around the loop.

        Geometry:
          • Outer ring at R_OUTER = 20 m, inner ring at R_INNER = 10 m
            (track 10 m wide). Sinusoidal radial wobble gives the track
            twists and turns so the controller can never just go straight.
          • N_DEAD_ENDS U-shaped pockets sprinkled around the loop. Each
            pocket is a roughly-10 m-long arc-aligned alcove with one
            closed end — the robot can drive in, hit the back wall, and
            must reverse out (firing FORWARD_BLOCKED_BREADCRUMB_REVERSE).

        The mission waypoints (default 4) are placed at evenly-spaced
        angles around the loop. Sim.build picks them up via
        `default_goal_waypoints`.
        """
        rng = random.Random(seed)
        SIZE = 50.0
        cx, cy = SIZE / 2.0, SIZE / 2.0
        R_OUTER = 20.0
        R_INNER = 10.0
        R_MID = (R_OUTER + R_INNER) / 2.0
        wave_amp = 1.5
        wave_n = 4
        num_seg = 128

        walls: list[tuple[float, float, float, float]] = []

        def ring_pt(r_base, theta, amp_sign):
            r = r_base + amp_sign * wave_amp * math.sin(wave_n * theta)
            return (cx + r * math.cos(theta), cy + r * math.sin(theta))

        # Outer + inner rings (sinusoidal radius).
        for k in range(num_seg):
            a0 = 2 * math.pi * k / num_seg
            a1 = 2 * math.pi * (k + 1) / num_seg
            walls.append((*ring_pt(R_OUTER, a0, +1), *ring_pt(R_OUTER, a1, +1)))
            walls.append((*ring_pt(R_INNER, a0, -1), *ring_pt(R_INNER, a1, -1)))

        # 4 mission waypoints, evenly spaced around the loop. Placed in
        # the OUTER lane (between R_MID and R_OUTER) since R_MID itself
        # is a wall after the bisect. The robot spawns at the angle
        # midway between waypoint 4 and waypoint 1, also in the outer
        # lane.
        n_waypoints = 4
        r_lane_outer = (R_MID + R_OUTER) / 2.0
        wp_phase = rng.uniform(0.0, 2.0 * math.pi)
        waypoint_angles = [(wp_phase + 2 * math.pi * k / n_waypoints)
                           % (2 * math.pi) for k in range(n_waypoints)]
        waypoints = [(cx + r_lane_outer * math.cos(a),
                      cy + r_lane_outer * math.sin(a))
                     for a in waypoint_angles]

        # ── Dead-end traps via a chopped middle ring ──
        # The annulus is bisected by a third ring at R_MID, splitting it
        # into an outer lane (R_MID..R_OUTER) and an inner lane
        # (R_INNER..R_MID). The middle ring is sliced into N arcs with
        # small gaps between them — those gaps are where the robot can
        # switch lanes. Each arc's CW end gets a radial wall connecting
        # it out to either the outer or inner ring, sealing the lane on
        # that side at that angle. A robot travelling CCW must be in the
        # OPPOSITE lane from the attachment when it arrives at each arc,
        # or it dead-ends and has to back out.
        # ~5 m of arc per dead-end pocket, with ~5 m of clear space (arc)
        # between adjacent pockets. arc_len_m / R_MID = radians.
        target_arc_len_m = 5.0
        target_gap_len_m = 5.0
        arc_per_seg = target_arc_len_m / R_MID
        gap_per_seg = target_gap_len_m / R_MID
        total_per_seg = arc_per_seg + gap_per_seg
        N_ARCS = max(4, int(round(2 * math.pi / total_per_seg)))
        attach_phase = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(N_ARCS):
            a_lo = (attach_phase + k * 2 * math.pi / N_ARCS) \
                % (2.0 * math.pi)                            # low-angle end
            seg_span = 2 * math.pi / N_ARCS - gap_per_seg
            a_hi = a_lo + seg_span                           # high-angle end
            # Draw the middle arc (discretised into chords).
            sub_seg = 24
            for j in range(sub_seg):
                af0 = a_lo + (j / sub_seg) * seg_span
                af1 = a_lo + ((j + 1) / sub_seg) * seg_span
                walls.append((cx + R_MID * math.cos(af0),
                              cy + R_MID * math.sin(af0),
                              cx + R_MID * math.cos(af1),
                              cy + R_MID * math.sin(af1)))
            # Radial connector on the CW-facing (high-angle) end → outer
            # OR inner ring. A CW-bound robot reaches `a_hi` first, so a
            # wall here is what it'll see; a CCW-bound robot reaches the
            # gap at `a_hi` and crosses lanes freely. The trap is in
            # the CW direction.
            attach_outer = rng.random() < 0.5
            r_attach = R_OUTER if attach_outer else R_INNER
            walls.append((cx + R_MID * math.cos(a_hi),
                          cy + R_MID * math.sin(a_hi),
                          cx + r_attach * math.cos(a_hi),
                          cy + r_attach * math.sin(a_hi)))

        # Spawn: midway between waypoint 4 and waypoint 1, in the OUTER
        # lane. Heading is set tangentially by Sim.build.
        a_spawn = (waypoint_angles[-1] + 2 * math.pi / n_waypoints / 2.0) \
            % (2.0 * math.pi)
        start_xy = (cx + r_lane_outer * math.cos(a_spawn),
                    cy + r_lane_outer * math.sin(a_spawn))

        m = cls(
            walls=walls,
            size_m=SIZE,
            default_start_xy=start_xy,
            default_goal_xy=waypoints[0],
        )
        m.default_goal_waypoints = waypoints
        m._voxelize_walls()
        return m

    def segs_near(self, x: float, y: float, r: float):
        """Cheap broad-phase: yield wall segments whose AABB intersects
        the (x, y) ± r box."""
        for (x0, y0, x1, y1) in self.walls:
            if max(x0, x1) < x - r or min(x0, x1) > x + r:
                continue
            if max(y0, y1) < y - r or min(y0, y1) > y + r:
                continue
            yield (x0, y0, x1, y1)


def _point_segment_dist(px, py, x0, y0, x1, y1):
    """Return (distance, closest_pt_x, closest_pt_y, t∈[0,1]) for a point
    vs. a segment."""
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        d = math.hypot(px - x0, py - y0)
        return d, x0, y0, 0.0
    t = ((px - x0) * dx + (py - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = x0 + t * dx, y0 + t * dy
    return math.hypot(px - cx, py - cy), cx, cy, t


def _ray_segment_hit(rx, ry, dx, dy, x0, y0, x1, y1, max_t):
    """Ray (origin rx,ry; unit dir dx,dy) vs. segment. Returns t (distance
    along ray) of nearest hit ≤ max_t, or None."""
    sx, sy = x1 - x0, y1 - y0
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None
    ex, ey = x0 - rx, y0 - ry
    t = (ex * sy - ey * sx) / denom
    u = (ex * dy - ey * dx) / denom
    if t < 0 or t > max_t:
        return None
    if u < 0 or u > 1:
        return None
    return t


# ──────────────────────────────────────────────────────────────────────────────
#                              Discovery grid
# ──────────────────────────────────────────────────────────────────────────────
UNKNOWN, FREE_KNOWN, WALL_KNOWN = 0, 1, 2


@dataclass
class DiscoveryGrid:
    size_m: float
    res: float = GRID_RES_M
    cells: np.ndarray = field(init=False)

    def __post_init__(self):
        n = int(math.ceil(self.size_m / self.res))
        self.n = n
        self.cells = np.full((n, n), UNKNOWN, dtype=np.uint8)

    def world_to_cell(self, x: float, y: float):
        i = int(x / self.res)
        j = int(y / self.res)
        if 0 <= i < self.n and 0 <= j < self.n:
            return i, j
        return None

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * self.res, (j + 0.5) * self.res

    def mark_ray(self, ox: float, oy: float, dx: float, dy: float,
                 max_range: float, wall_voxel_mask: np.ndarray,
                 rng: Optional[random.Random] = None) -> float:
        """Walk the ray cell-by-cell. Per-cell behaviour:
          * Wall voxel — with probability WALL_DETECTION_PROB, mark
            WALL_KNOWN and stop. Otherwise (false negative) treat the
            cell as free and keep going; the wall stays UNKNOWN this
            sweep and gets a chance to be redetected by neighbouring
            rays or the next BT tick.
          * UNKNOWN free cell — mark FREE_KNOWN, keep going.
        Returns distance at which the ray actually terminated.
        """
        if rng is None:
            rng = random
        step = self.res * 0.5
        max_steps = int(max_range / step) + 1
        last_c = None
        for s in range(max_steps):
            t = s * step
            if t > max_range:
                return max_range
            c = self.world_to_cell(ox + t * dx, oy + t * dy)
            if c is None:
                return t
            if c == last_c:
                continue
            last_c = c
            if wall_voxel_mask[c]:
                if rng.random() < WALL_DETECTION_PROB:
                    self.cells[c] = WALL_KNOWN
                    return t
                # Missed — pretend the cell was free this sweep.
                if self.cells[c] == UNKNOWN:
                    self.cells[c] = FREE_KNOWN
                continue
            if self.cells[c] == UNKNOWN:
                self.cells[c] = FREE_KNOWN
        return max_range

    def wall_distance_grid(self, max_cells: int = 3) -> np.ndarray:
        """Chebyshev distance (in cells) from each cell to the nearest
        WALL_KNOWN cell, capped at `max_cells + 1`. Cell value 0 = the
        wall itself, 1 = direct neighbour, …, max_cells+1 = beyond the
        critic radius (no penalty).

        Used by the DWB obstacle-aware critic: a candidate trajectory's
        score includes a penalty proportional to `max_cells + 1 - d` for
        each sampled point's wall distance d ≤ max_cells.
        """
        wall = (self.cells == WALL_KNOWN)
        n = wall.shape[0]
        dist = np.full((n, n), max_cells + 1, dtype=np.uint8)
        dist[wall] = 0
        cur = wall.copy()
        for d in range(1, max_cells + 1):
            nxt = cur.copy()
            nxt[:-1, :] |= cur[1:, :]
            nxt[1:,  :] |= cur[:-1, :]
            nxt[:, :-1] |= cur[:, 1:]
            nxt[:, 1:]  |= cur[:, :-1]
            nxt[:-1, :-1] |= cur[1:, 1:]
            nxt[1:,  1:]  |= cur[:-1, :-1]
            nxt[:-1, 1:]  |= cur[1:,  :-1]
            nxt[1:, :-1]  |= cur[:-1, 1:]
            new = nxt & ~cur
            dist[new] = d
            cur = nxt
        return dist

    def inflation_mask(self) -> np.ndarray:
        """Cells within INFLATION_CELLS of any discovered wall, EXCLUDING
        the wall cells themselves. Used by the BT as a "warning halo" —
        if the robot's centre enters one of these cells, GRADIENT_ESCAPE
        fires. The planner is allowed to plan THROUGH these cells (per
        user spec: inflation triggers escape, never blocks)."""
        wall = (self.cells == WALL_KNOWN)
        rad = INFLATION_CELLS
        if rad <= 0:
            return np.zeros_like(wall)
        infl = wall.copy()
        from itertools import product
        n = self.n
        for di, dj in product(range(-rad, rad + 1), repeat=2):
            if di * di + dj * dj > rad * rad:
                continue
            sl_i = slice(max(0, di), n + min(0, di))
            sl_j = slice(max(0, dj), n + min(0, dj))
            src_i = slice(max(0, -di), n + min(0, -di))
            src_j = slice(max(0, -dj), n + min(0, -dj))
            infl[sl_i, sl_j] |= wall[src_i, src_j]
        return infl & ~wall

    def discovered_fraction(self) -> float:
        return float((self.cells != UNKNOWN).sum()) / self.cells.size

    def inflated_obstacle_mask(self) -> np.ndarray:
        """Planner-blocking mask: discovered wall voxels PLUS their
        INFLATION_CELLS halo. The planner respects the inflation so it
        routes around walls with margin. The BT still keeps GRADIENT_ESCAPE
        as a safety net for when the robot drifts into the halo from
        unrelated motion (controller is allowed to drive through halo
        cells when commanded — only the planner avoids them)."""
        wall = (self.cells == WALL_KNOWN)
        return wall | self.inflation_mask()


# ──────────────────────────────────────────────────────────────────────────────
#         Map padder — corridor + wall geometry (port of map_padder_node.py)
# ──────────────────────────────────────────────────────────────────────────────
class MapPadder:
    """Port of AutoNav_25-26 isaac_ros-dev/src/map_padder/map_padder/
    map_padder_node.py.

    Builds a cumulative "corridor" of allowed planner cells around the
    robot footprint, the goal, the straight line robot→goal, and the
    cells of the current plan. Each seed group gets a 1-ring 8-neighbour
    buffer. Cells inside the corridor are FREE for planning; cells
    outside it are forced LETHAL so Dijkstra's search collapses to the
    corridor (fast). Discovered red wall cells inside the corridor stay
    lethal — they're the only thing that ever forces a detour.

    Corridor membership is monotonic: once a cell joins the corridor, it
    never leaves. That matches the upstream design ("no eating-away of
    the map behind the robot").
    """

    def __init__(self, grid: DiscoveryGrid):
        self.grid = grid
        self.cumulative = np.zeros((grid.n, grid.n), dtype=np.bool_)
        # Local window radius in cells (≈ 3 m on real robot).
        self.window_cells = max(1, int(round(PADDER_LOCAL_WINDOW_M / grid.res)))

    @staticmethod
    def _ring(mask: np.ndarray) -> np.ndarray:
        """8-neighbour dilation by one cell (the upstream `_ring` op)."""
        n = mask.shape[0]
        out = mask.copy()
        out[:-1, :] |= mask[1:, :]
        out[1:,  :] |= mask[:-1, :]
        out[:, :-1] |= mask[:, 1:]
        out[:, 1:]  |= mask[:, :-1]
        out[:-1, :-1] |= mask[1:,  1:]
        out[1:,  1:]  |= mask[:-1, :-1]
        out[:-1, 1:]  |= mask[1:,  :-1]
        out[1:,  :-1] |= mask[:-1, 1:]
        return out

    @staticmethod
    def _line_cells_into(mask: np.ndarray, c0, c1):
        """Bresenham-style: stamp every cell on the segment c0→c1 into
        `mask`. Matches upstream `_line_tiles` (which raster-walks at
        half-tile steps)."""
        x0, y0 = c0
        x1, y1 = c1
        n = max(abs(x1 - x0), abs(y1 - y0))
        if n == 0:
            if 0 <= x0 < mask.shape[0] and 0 <= y0 < mask.shape[1]:
                mask[x0, y0] = True
            return
        for k in range(n + 1):
            f = k / n
            i = int(round(x0 + f * (x1 - x0)))
            j = int(round(y0 + f * (y1 - y0)))
            if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
                mask[i, j] = True

    def update(self,
               robot_cell: tuple[int, int],
               goal_cell:  tuple[int, int],
               plan_cells: list[tuple[int, int]] | None,
               extra_seed_cells: list[tuple[int, int]] | None = None):
        n = self.grid.n
        seeds = np.zeros((n, n), dtype=np.bool_)
        ri, rj = robot_cell
        w = self.window_cells
        i0, i1 = max(0, ri - w), min(n, ri + w + 1)
        j0, j1 = max(0, rj - w), min(n, rj + w + 1)
        seeds[i0:i1, j0:j1] = True
        # Goal tile + straight line robot→goal.
        if 0 <= goal_cell[0] < n and 0 <= goal_cell[1] < n:
            seeds[goal_cell[0], goal_cell[1]] = True
            self._line_cells_into(seeds, robot_cell, goal_cell)
        # Plan tiles — feedback from the previous successful plan keeps the
        # corridor following the planner's chosen route through walls.
        if plan_cells:
            for (ci, cj) in plan_cells:
                if 0 <= ci < n and 0 <= cj < n:
                    seeds[ci, cj] = True
        # Extra seeds (e.g. bent goal + line to it during GOAL_BEND).
        if extra_seed_cells:
            for (ci, cj) in extra_seed_cells:
                if 0 <= ci < n and 0 <= cj < n:
                    seeds[ci, cj] = True
        # Corridor = seeds ⊕ 3-iteration 8-ring buffer. Diagnostic trace
        # showed the single-ring (1.2 m wide @ 0.4 m grid res) was so
        # narrow that any wall hit produced NO_PATH for 25+ seconds while
        # the robot oscillated in GOAL_BEND. Three rings ≈ 3.6 m wide,
        # enough headroom to detour around a single discovered wall
        # within a 5 m maze corridor without escaping the padder.
        corridor = self._ring(self._ring(self._ring(seeds)))
        self.cumulative |= corridor

    def blocked_mask(self) -> np.ndarray:
        """LETHAL outside the cumulative corridor + discovered wall voxels
        AND their inflation halo inside it. The inflation is now a
        planner-block (so the global plan steers clear of walls); the
        controller is still allowed to drive into the halo, and the BT
        treats halo-entry as a GRADIENT_ESCAPE trigger."""
        wall = (self.grid.cells == WALL_KNOWN)
        return wall | self.grid.inflation_mask() | (~self.cumulative)


# ──────────────────────────────────────────────────────────────────────────────
#                  A* planner (analog: planner_server, "Dijkstra")
# ──────────────────────────────────────────────────────────────────────────────
# Octile distance — exact 8-connected metric (1 step orthogonal, √2
# diagonal) and admissible for A*. Hand-rolled instead of math.hypot so
# the inner loop has no function-call overhead.
_SQRT2 = math.sqrt(2.0)
_OCTILE_DIAG = _SQRT2 - 1.0


def _octile(i1, j1, i2, j2):
    dx = abs(i1 - i2)
    dy = abs(j1 - j2)
    if dx < dy:
        dx, dy = dy, dx
    return dx + _OCTILE_DIAG * dy


def _nudge_to_free(grid_blocked: np.ndarray, ci: int, cj: int,
                   radius: int = 4) -> Optional[tuple[int, int]]:
    """Return the closest free cell to (ci, cj) within `radius`, or None."""
    n = grid_blocked.shape[0]
    if 0 <= ci < n and 0 <= cj < n and not grid_blocked[ci, cj]:
        return (ci, cj)
    best = None
    best_d2 = 1e18
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ii, jj = ci + di, cj + dj
            if not (0 <= ii < n and 0 <= jj < n):
                continue
            if grid_blocked[ii, jj]:
                continue
            d2 = di * di + dj * dj
            if d2 < best_d2:
                best_d2 = d2
                best = (ii, jj)
    return best


def _bresenham_line_clear(grid_blocked: np.ndarray,
                          i0: int, j0: int, i1: int, j1: int) -> bool:
    """Bresenham line walk from (i0, j0) to (i1, j1) over a uint8/bool
    blocked mask. Returns True if every cell along the line (including
    endpoints) is NOT blocked. Cheap O(max(|dx|, |dy|)) — single LOS
    check is roughly a hundred lookups, faster than running A*.
    """
    n = grid_blocked.shape[0]
    if not (0 <= i0 < n and 0 <= j0 < n and 0 <= i1 < n and 0 <= j1 < n):
        return False
    dx = abs(i1 - i0)
    dy = abs(j1 - j0)
    sx = 1 if i0 < i1 else -1
    sy = 1 if j0 < j1 else -1
    err = dx - dy
    i, j = i0, j0
    while True:
        if grid_blocked[i, j]:
            return False
        if i == i1 and j == j1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            i += sx
        if e2 < dx:
            err += dx
            j += sy


def _subgoal_mask_from_goal(grid_blocked: np.ndarray,
                             gi: int, gj: int,
                             num_rays: int = 72) -> np.ndarray:
    """Sweep `num_rays` rays outward from (gi, gj) and mark every cell
    each ray passes through (until it hits a blocked cell) as a subgoal.

    This is a coarse visibility polygon from the goal: any cell in the
    mask has *direct line of sight* to the goal. A* can terminate the
    instant it pops a subgoal cell, then we append a straight-line path
    from there to the goal — skipping the rest of the Dijkstra search.

    72 rays at 5° spacing is enough to fill a 25 m maze corridor at
    GRID_RES_M = 0.4 m without gaps. Cost: ~72 × 60 cell lookups per
    replan = ~4 k ops, basically free.
    """
    n = grid_blocked.shape[0]
    mask = np.zeros((n, n), dtype=np.bool_)
    if not (0 <= gi < n and 0 <= gj < n):
        return mask
    if not grid_blocked[gi, gj]:
        mask[gi, gj] = True
    max_t = float(n) * math.sqrt(2.0)
    for k in range(num_rays):
        a = 2 * math.pi * k / num_rays
        dx = math.cos(a)
        dy = math.sin(a)
        # Step in half-cell increments along the ray.
        step = 0.5
        t = step
        while t < max_t:
            i = gi + int(round(t * dx))
            j = gj + int(round(t * dy))
            if not (0 <= i < n and 0 <= j < n):
                break
            if grid_blocked[i, j]:
                break
            mask[i, j] = True
            t += step
    return mask


def astar(grid_blocked: np.ndarray,
          start_ij: tuple[int, int],
          goal_ij:  tuple[int, int],
          subgoal_mask: Optional[np.ndarray] = None,
          ) -> Optional[list[tuple[int, int]]]:
    """8-connected A* with the octile heuristic. Same call signature and
    return type as the old Dijkstra so the planner glue doesn't change.

    If `subgoal_mask` is provided, the search terminates the instant it
    pops ANY cell in that mask (not just the goal). The caller is
    responsible for appending the straight-line segment from the
    early-terminated cell to the real goal. This skips most of the
    Dijkstra search when wide-open areas exist between the robot and
    the goal — exactly the "maze opens and goal is in line of sight"
    case the user called out.

    The octile heuristic is admissible (never over-estimates the true
    8-connected distance) and tight enough to keep frontier expansions
    proportional to the optimal-path length.
    """
    n = grid_blocked.shape[0]
    si, sj = start_ij
    gi, gj = goal_ij
    if not (0 <= si < n and 0 <= sj < n and 0 <= gi < n and 0 <= gj < n):
        return None
    # Don't nudge the START — if the robot is inside the costmap halo
    # the planner SHOULD return NO_PATH so the BT can fire GRADIENT_ESCAPE.
    # The goal is still nudged to the nearest free cell (so a goal placed
    # right against a wall still plans).
    if grid_blocked[si, sj]:
        return None
    goal = _nudge_to_free(grid_blocked, gi, gj, 4)
    if goal is None:
        return None
    gi, gj = goal
    INF = math.inf
    g = np.full((n, n), INF, dtype=np.float32)
    g[si, sj] = 0.0
    parent: dict[int, int] = {}                 # encoded (i * n + j) → parent
    pq: list[tuple[float, int, int]] = [(_octile(si, sj, gi, gj), si, sj)]
    nbrs = ((-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
            ( 0, -1, 1.0),                   ( 0, 1, 1.0),
            ( 1, -1, _SQRT2), ( 1, 0, 1.0), ( 1, 1, _SQRT2))
    g_view = g
    blocked = grid_blocked
    heappush = heapq.heappush
    heappop = heapq.heappop
    final_i, final_j = gi, gj          # where we actually terminated
    while pq:
        _f, i, j = heappop(pq)
        if (i, j) == (gi, gj):
            final_i, final_j = i, j
            break
        if subgoal_mask is not None and subgoal_mask[i, j]:
            final_i, final_j = i, j
            break
        ci = g_view[i, j]
        for di, dj, w in nbrs:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= n or nj < 0 or nj >= n:
                continue
            if blocked[ni, nj]:
                continue
            nc = ci + w
            if nc < g_view[ni, nj]:
                g_view[ni, nj] = nc
                parent[ni * n + nj] = i * n + j
                heappush(pq, (nc + _octile(ni, nj, gi, gj), ni, nj))
    if g_view[final_i, final_j] == INF:
        return None
    path = [(final_i, final_j)]
    cur = final_i * n + final_j
    start_key = si * n + sj
    while cur != start_key:
        cur = parent[cur]
        path.append((cur // n, cur % n))
    path.reverse()
    return path


# Backwards-compat alias: the planner code calls `dijkstra(...)` because
# that's the spec ("Dijkstra algorithm to get the robot to a goal"). A*
# with an admissible heuristic returns the same optimal path as Dijkstra
# on uniform-cost grids, so we keep the name and route through A*.
dijkstra = astar


# ──────────────────────────────────────────────────────────────────────────────
#                            Robot (Chaplygin sleigh)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Robot:
    """State is the rear-axle midpoint (x, y), heading θ, forward speed u
    (along body x), and yaw rate ω. COM lies at body offset (+COM_OFFSET_M, 0).
    Rear axle is constrained by the two knife edges to move along ±body-x
    only — that nonholonomic constraint *is* the Chaplygin sleigh.
    """
    x: float
    y: float
    theta: float
    u: float = 0.0
    omega: float = 0.0
    # Most-recent applied wheel forces (for HUD only).
    F_left: float = 0.0
    F_right: float = 0.0

    # ── Frame helpers ──
    def rear_axle(self) -> tuple[float, float]:
        return self.x, self.y

    def com(self) -> tuple[float, float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + COM_OFFSET_M * c, self.y + COM_OFFSET_M * s)

    def front_caster(self) -> tuple[float, float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + WHEELBASE_M * c, self.y + WHEELBASE_M * s)

    def left_knife(self) -> tuple[float, float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        # +body-y = left (ROS REP-103: x fwd, y left)
        return (self.x + (-s) * TRACK_WIDTH_M / 2,
                self.y + (c) * TRACK_WIDTH_M / 2)

    def right_knife(self) -> tuple[float, float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x - (-s) * TRACK_WIDTH_M / 2,
                self.y - (c) * TRACK_WIDTH_M / 2)

    # ── Dynamics ──
    def step(self, F_left: float, F_right: float, dt: float, maze: Maze):
        # Saturate per-wheel forces (controller-server limit).
        F_left = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_left))
        F_right = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_right))
        self.F_left, self.F_right = F_left, F_right

        F_total = F_left + F_right
        torque_rear = (F_right - F_left) * TRACK_WIDTH_M / 2.0
        # Standard Chaplygin-sleigh accelerations w/ centripetal coupling from
        # COM offset, plus drivetrain damping.
        du = (F_total + ROBOT_MASS_KG * COM_OFFSET_M * self.omega ** 2
              - LIN_DAMP * self.u) / ROBOT_MASS_KG
        dw = (torque_rear
              - ROBOT_MASS_KG * COM_OFFSET_M * self.u * self.omega
              - ANG_DAMP * self.omega) / INERTIA_REAR
        self.u += du * dt
        self.omega += dw * dt
        # Kinematic update (rear axle slides only along body x).
        self.x += self.u * math.cos(self.theta) * dt
        self.y += self.u * math.sin(self.theta) * dt
        self.theta += self.omega * dt
        # Wrap heading to (-π, π].
        if self.theta > math.pi:
            self.theta -= 2 * math.pi
        elif self.theta <= -math.pi:
            self.theta += 2 * math.pi

        # Collision: keep the entire body footprint clear of walls. Cheap
        # rep: check the 5 reference points (rear axle, both knife edges,
        # front caster, COM) against any nearby wall, push out + kill the
        # velocity component into the wall.
        self._resolve_collisions(maze)

    def _resolve_collisions(self, maze: Maze):
        rad = FOOTPRINT_HALF_W + WALL_THICKNESS_M / 2 + 0.02
        pts = [self.rear_axle(), self.left_knife(), self.right_knife(),
               self.front_caster(), self.com()]
        for (px, py) in pts:
            for seg in maze.segs_near(px, py, rad + 0.1):
                d, cx, cy, _ = _point_segment_dist(px, py, *seg)
                if d < rad and d > 1e-6:
                    nx, ny = (px - cx) / d, (py - cy) / d
                    push = (rad - d) + 1e-3
                    # Push the rear axle along the contact normal — keeps
                    # the integration stable without modelling the impulse.
                    self.x += nx * push
                    self.y += ny * push
                    # Kill velocity into the wall (projected along normal).
                    vx = self.u * math.cos(self.theta)
                    vy = self.u * math.sin(self.theta)
                    vn = vx * nx + vy * ny
                    if vn < 0:
                        # bleed body-frame forward speed (whatever's into
                        # the wall) plus a chunk of yaw rate from the
                        # impact.
                        self.u *= max(0.0, 1.0 - abs(vn) * 0.5)
                        self.omega *= 0.8


# ──────────────────────────────────────────────────────────────────────────────
#                        Sensor (15 m hemispheric raycaster)
# ──────────────────────────────────────────────────────────────────────────────
def cast_sensor(robot: Robot, maze: Maze, grid: "DiscoveryGrid",
                num_rays: int = SENSOR_NUM_RAYS,
                max_range: float = SENSOR_RANGE_M,
                rng: Optional[random.Random] = None):
    """Cast `num_rays` from the front caster, fanned over ±SENSOR_FOV_RAD/2.
    Each ray walks the discovery grid cell-by-cell against the maze's
    voxelised wall mask — the same cell that holds the wall geometry is
    the cell that gets marked WALL_KNOWN, so adjacent rays always agree
    on which voxel they hit. Free cells along the ray flip from UNKNOWN
    to FREE_KNOWN as a side effect.

    Returns (origin_xy, list_of_(angle_world, hit_range)).
    """
    ox, oy = robot.front_caster()
    half = SENSOR_FOV_RAD / 2
    results = []
    mask = maze.wall_voxel_mask
    for k in range(num_rays):
        local_a = -half + (k / max(1, num_rays - 1)) * SENSOR_FOV_RAD
        a = robot.theta + local_a
        dx, dy = math.cos(a), math.sin(a)
        t_hit = grid.mark_ray(ox, oy, dx, dy, max_range, mask, rng)
        results.append((a, t_hit))
    return (ox, oy), results


# ──────────────────────────────────────────────────────────────────────────────
#               Controller server (per-wheel force allocation)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ControllerOutput:
    F_left: float
    F_right: float
    v_des: float
    omega_des: float
    backwards_request: bool   # planner asked for body-x < 0; DWB would refuse


def pure_pursuit_to_wheel_forces(
    robot: Robot,
    path_xy: list[tuple[float, float]],
    *,
    allow_reverse: bool = False,
    target_speed: float = DESIRED_SPEED_MPS,
    lookahead: float = LOOKAHEAD_M,
) -> ControllerOutput:
    """Pure-pursuit + PD on (v, ω) → per-wheel forces.

    The split into per-wheel forces is the analog of nav2's controller_server
    publishing /cmd_vel: we go one level lower and emit the underlying
    actuator commands the BT can shape (e.g. cap reverse, cap rotation in
    place, etc.).
    """
    if not path_xy:
        return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
    rx, ry = robot.rear_axle()
    # Find closest path index, then a carrot lookahead m ahead of it.
    best_k = 0
    best_d2 = 1e18
    for k, (px, py) in enumerate(path_xy):
        d2 = (px - rx) ** 2 + (py - ry) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_k = k
    acc = 0.0
    carrot = path_xy[-1]
    for k in range(best_k, len(path_xy) - 1):
        x0, y0 = path_xy[k]
        x1, y1 = path_xy[k + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= lookahead:
            t = (lookahead - acc) / max(seg, 1e-6)
            carrot = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            break
        acc += seg
    # Slow down near goal (last path point).
    goal = path_xy[-1]
    dist_to_goal = math.hypot(goal[0] - rx, goal[1] - ry)
    v_des = target_speed * min(1.0, dist_to_goal / APPROACH_SLOW_M)
    if dist_to_goal < GOAL_TOLERANCE_M:
        v_des = 0.0
    # Bearing to carrot (world).
    bearing = math.atan2(carrot[1] - ry, carrot[0] - rx)
    err = bearing - robot.theta
    # Wrap to (-π, π].
    while err > math.pi:
        err -= 2 * math.pi
    while err <= -math.pi:
        err += 2 * math.pi

    backwards = abs(err) > math.pi / 2  # carrot is behind body
    if backwards and not allow_reverse:
        # Forward motion is refused, but we still rotate IN PLACE toward
        # the carrot — this is what real DWB does and it avoids tripping
        # the BT's FORWARD_BLOCKED recovery on every brief mis-alignment
        # mid-turn. The backwards_request flag still surfaces so the BT
        # can react if persistent.
        v_des = 0.0
        # Scale ω higher when stationary so we actually turn briskly.
        omega_des = max(-1.5, min(1.5, 2.4 * err))
        a_yaw = KP_ANG * (omega_des - robot.omega)
        a_long = KP_LIN * (v_des - robot.u)
        F_total = ROBOT_MASS_KG * a_long / 10.0
        tau_total = INERTIA_REAR * a_yaw / 6.0
        F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
        F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
        return ControllerOutput(F_left, F_right, v_des, omega_des, True)
    if allow_reverse and backwards:
        # Drive backwards: flip the bearing, drive negative v_des.
        err = err + math.pi if err < 0 else err - math.pi
        v_des = -abs(v_des)

    # PD on linear (forward speed) and angular (yaw err → ω).
    omega_des = max(-1.0, min(1.0, 1.8 * err))
    a_long = KP_LIN * (v_des - robot.u) - KD_LIN * 0.0
    a_yaw = KP_ANG * (omega_des - robot.omega) - KD_ANG * 0.0
    F_total = ROBOT_MASS_KG * a_long / 10.0     # /10 = stiffness scaling
    tau_total = INERTIA_REAR * a_yaw / 6.0
    F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
    F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
    return ControllerOutput(F_left, F_right, v_des, omega_des, False)


def dwb_with_obstacle_critic(
    robot: Robot,
    path_xy: list[tuple[float, float]],
    wall_dist_grid: Optional[np.ndarray],
    grid_res: float,
    *,
    allow_reverse: bool = False,
    target_speed: float = DESIRED_SPEED_MPS,
    lookahead: float = LOOKAHEAD_M,
) -> ControllerOutput:
    """DWB-style controller: take pure-pursuit's (v, ω) suggestion as a
    baseline, then sample nearby (v, ω) candidates, simulate each forward
    over DWB_HORIZON_S seconds, and pick the trajectory with the lowest
    combined cost:

        cost = path_alignment_err + DWB_CRITIC_WEIGHT * obstacle_penalty

    where:
      * path_alignment_err = distance from the trajectory's final point
        to the pure-pursuit carrot (smaller = better path following).
      * obstacle_penalty = Σ (DWB_CRITIC_RADIUS_CELLS + 1 − d) for every
        sampled trajectory point whose nearest wall is at d ≤ critic
        radius cells. A candidate that hits a wall (d = 0) is rejected.

    If the pure-pursuit baseline refuses to move (carrot behind, no
    reverse allowed), or if the wall-distance grid isn't ready yet, we
    just return the pure-pursuit output unchanged — the sampler has no
    useful signal without obstacle data.
    """
    baseline = pure_pursuit_to_wheel_forces(
        robot, path_xy,
        allow_reverse=allow_reverse,
        target_speed=target_speed,
        lookahead=lookahead,
    )
    if wall_dist_grid is None or not path_xy or baseline.v_des == 0.0:
        return baseline

    # Recompute the carrot (same logic as pure_pursuit) so we can score
    # path alignment without threading it back through the baseline.
    rx, ry = robot.rear_axle()
    best_k = 0
    best_d2 = 1e18
    for k, (px, py) in enumerate(path_xy):
        d2 = (px - rx) ** 2 + (py - ry) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_k = k
    acc = 0.0
    carrot = path_xy[-1]
    for k in range(best_k, len(path_xy) - 1):
        x0, y0 = path_xy[k]
        x1, y1 = path_xy[k + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= lookahead:
            t = (lookahead - acc) / max(seg, 1e-6)
            carrot = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            break
        acc += seg

    n = wall_dist_grid.shape[0]
    max_d = DWB_CRITIC_RADIUS_CELLS
    weight = DWB_CRITIC_WEIGHT
    steps = max(1, int(round(DWB_HORIZON_S / DWB_HORIZON_DT_S)))
    dt = DWB_HORIZON_DT_S

    best_score = math.inf
    best_v = baseline.v_des
    best_w = baseline.omega_des

    for vf in DWB_V_DELTAS:
        v_cand = baseline.v_des * vf
        for wd in DWB_W_DELTAS:
            w_cand = baseline.omega_des + wd
            # Clamp |ω| to the same envelope pure-pursuit uses.
            if w_cand > 1.5:
                w_cand = 1.5
            elif w_cand < -1.5:
                w_cand = -1.5
            x, y, th = rx, ry, robot.theta
            collided = False
            obstacle_penalty = 0.0
            for _ in range(steps):
                x += v_cand * math.cos(th) * dt
                y += v_cand * math.sin(th) * dt
                th += w_cand * dt
                i = int(x / grid_res)
                j = int(y / grid_res)
                if 0 <= i < n and 0 <= j < n:
                    d = int(wall_dist_grid[i, j])
                    if d == 0:
                        collided = True       # candidate hits a hard wall
                        break
                    if d <= max_d:
                        # Proximity penalty: closer = higher cost. The
                        # critic prefers (v, ω) combos whose trajectory
                        # stays farther from the costmap halo, without
                        # changing the v_des envelope — at the same
                        # speed it just picks a more open path.
                        obstacle_penalty += (max_d + 1 - d)
            if collided:
                continue
            path_err = math.hypot(x - carrot[0], y - carrot[1])
            score = path_err + weight * obstacle_penalty
            if score < best_score:
                best_score = score
                best_v = v_cand
                best_w = w_cand

    # Translate the winning (v, ω) to wheel forces via the same PD shaping
    # pure_pursuit uses, so the dynamics envelope is identical.
    a_long = KP_LIN * (best_v - robot.u)
    a_yaw  = KP_ANG * (best_w - robot.omega)
    F_total   = ROBOT_MASS_KG * a_long / 10.0
    tau_total = INERTIA_REAR * a_yaw / 6.0
    F_left  = 0.5 * F_total - tau_total / TRACK_WIDTH_M
    F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
    return ControllerOutput(F_left, F_right, best_v, best_w,
                            baseline.backwards_request)


# ──────────────────────────────────────────────────────────────────────────────
#                              Behavior tree
# ──────────────────────────────────────────────────────────────────────────────
# States during which the sim should halt sensor-driven discovery
# updates — mirrors the real robot, where during a recovery action the
# perception pipeline (LiDAR + camera) is temporarily paused so it
# doesn't overwrite the costmap with motion-blurred / out-of-date hits
# as the chassis is rapidly reversing. Breadcrumb dropping is paused
# too so the consumption ledger doesn't get re-padded by reverse
# motion.
SENSING_HALTED_STATES = frozenset((
    "FORWARD_BLOCKED_BREADCRUMB_REVERSE",
    "BACKUP_RECOVERY",
    "GRADIENT_ESCAPE",
))


BT_STATES = (
    "NORMAL_FOLLOWING",
    "FORWARD_BLOCKED_BREADCRUMB_REVERSE",
    "FORWARD_BLOCKED_WAIT_FOR_REPLAN",
    "GOAL_BEND",
    "BACKUP_RECOVERY",
    "GRADIENT_ESCAPE",
    # Local costmap clear only — per user spec, the global costmap is
    # never wiped (wiping it makes the planner think the world is clear
    # and route backwards through invisible walls).
    "CLEAR_AROUND_ROBOT",
    "WAIT_TRANSIENT_RECOVERY",
)


@dataclass
class BTState:
    """Tracks the active behavior-tree node + a few transition signals."""
    name: str = "NORMAL_FOLLOWING"
    entered_at: float = 0.0
    # Counters surfaced on the HUD.
    n_planner_failures: int = 0
    n_backup_fires: int = 0
    n_gradient_fires: int = 0
    n_clear_around_robot: int = 0   # local 1 m wipe (FollowPathRecovery)
    # (CLEAR_GLOBAL / CLEAR_BOTH removed — global costmap is never wiped.)
    n_goal_bends: int = 0
    n_breadcrumb_reverses: int = 0
    # FollowPathRecovery: 2 around-robot clears before escalating to the
    # round-robin RecoveryFallback (matches `RecoveryNode number_of_retries=2`
    # in bt_nav.xml:120).
    around_robot_attempts: int = 0
    # RecoveryFallback round-robin cursor: 0=BACKUP, 1=GRADIENT, 2=CLEAR_BOTH.
    # Advances on each successive stuck event while in the recovery dance.
    recovery_round_step: int = 0
    in_recovery_round_robin: bool = False
    # Cooldown: after exiting GRADIENT_ESCAPE, suppress the inflation
    # trigger for this many seconds so the BT can escalate to BACKUP /
    # CLEAR_* via the stuck timer instead of looping GRADIENT_ESCAPE
    # forever when the escape can't actually leave the halo.
    last_gradient_exit_time: Optional[float] = None
    # GRADIENT_ESCAPE has three sub-phases:
    #   1. Reactive escape — drive out of the costmap halo.
    #   1.5. Post-escape margin — keep driving the same direction for
    #        another ~0.7 m so the robot is comfortably clear of the
    #        halo before it starts rotating.
    #   2. Alignment — rotate in place to face the planned path.
    gradient_aligning: bool = False
    # Position at which the robot first exited the inflation halo
    # (None if still inside it).
    gradient_exit_pos: Optional[tuple[float, float]] = None
    # Counts how many breadcrumbs the current BREADCRUMB_REVERSE
    # session has popped. Capped at BREADCRUMB_CONSUME_LIMIT — beyond
    # that, the BT gives up on reversing and bends the goal away from
    # the costmap instead.
    breadcrumbs_consumed: int = 0
    # Set the first time BREADCRUMB_REVERSE notices a solidly-forward
    # plan. The BT then waits to consume ONE MORE breadcrumb before
    # exiting to NORMAL — the extra margin keeps the robot from
    # slamming into the costmap right after taking control back.
    breadcrumb_exit_pending: bool = False
    # Progress watchdog (PROGRESS_STALL_SEC of no PROGRESS_DIST_M motion).
    last_progress_pos: tuple[float, float] = (0.0, 0.0)
    last_progress_time: float = 0.0
    # Per-state scratch.
    backup_start: tuple[float, float] = (0.0, 0.0)
    gradient_start_time: float = 0.0
    wait_until: float = 0.0
    # Sub-goal for GOAL_BEND.
    bent_goal: Optional[tuple[float, float]] = None
    # Persistent: number of consecutive replans that returned no path.
    consec_planner_fails: int = 0
    # "Path is geometrically behind body" is a noisy per-tick check —
    # require it to hold for `PATH_BEHIND_PERSIST_SEC` before firing the
    # FORWARD_BLOCKED recoveries. Otherwise every brief mid-turn or
    # carrot-snap fires BREADCRUMB_REVERSE even when the robot was
    # actively making progress.
    path_behind_since: Optional[float] = None


class BehaviorTree:
    def __init__(self):
        self.state = BTState()

    def enter(self, name: str, t: float, **scratch):
        if name == self.state.name:
            return
        self.state.name = name
        self.state.entered_at = t
        if name == "BACKUP_RECOVERY":
            self.state.n_backup_fires += 1
        elif name == "GRADIENT_ESCAPE":
            self.state.n_gradient_fires += 1
            self.state.gradient_start_time = t
            self.state.gradient_aligning = False
            self.state.gradient_exit_pos = None
        elif name == "CLEAR_AROUND_ROBOT":
            self.state.n_clear_around_robot += 1
        elif name == "GOAL_BEND":
            self.state.n_goal_bends += 1
        elif name == "FORWARD_BLOCKED_BREADCRUMB_REVERSE":
            self.state.n_breadcrumb_reverses += 1
            self.state.breadcrumbs_consumed = 0
            self.state.breadcrumb_exit_pending = False
        elif name == "WAIT_TRANSIENT_RECOVERY":
            self.state.wait_until = t + WAIT_SEC
        for k, v in scratch.items():
            setattr(self.state, k, v)


# ──────────────────────────────────────────────────────────────────────────────
#                                 Simulation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Sim:
    maze: Maze
    robot: Robot
    goal: tuple[float, float]
    grid: DiscoveryGrid
    bt: BehaviorTree
    padder: "MapPadder" = field(init=False)
    t: float = 0.0
    # Mission: an ordered list of goal waypoints. `goal` is always the
    # *current* one. When the robot reaches it, `mission_idx` advances
    # and `goal` becomes the next waypoint. `done` flips only when the
    # mission has run out of waypoints to visit.
    mission: list[tuple[float, float]] = field(default_factory=list)
    mission_idx: int = 0
    # Planner output (cached between replans).
    path_world: list[tuple[float, float]] = field(default_factory=list)
    # Separate intermediate path emitted by GOAL_BEND (drawn in a different
    # colour). The main `path_world` always targets the real goal.
    bend_path: list[tuple[float, float]] = field(default_factory=list)
    last_replan_time: float = -1e9
    # Trail / breadcrumbs.
    trail: list[tuple[float, float]] = field(default_factory=list)
    breadcrumbs: list[tuple[float, float]] = field(default_factory=list)
    # Live BT-tick clock.
    last_bt_tick: float = -1e9
    last_controller: ControllerOutput = field(
        default_factory=lambda: ControllerOutput(0.0, 0.0, 0.0, 0.0, False))
    # Win condition.
    done: bool = False
    done_time: Optional[float] = None
    # For HUD.
    last_progress_dist: float = 0.0
    last_plan_status: str = "OK"
    # Cached inflation warning mask (refreshed each BT tick by _sense).
    # Used by the BT's GRADIENT_ESCAPE trigger and by the renderer.
    _cached_inflation_mask: Optional[np.ndarray] = None
    # Cached wall-distance grid (Chebyshev cells → nearest WALL_KNOWN,
    # capped at DWB_CRITIC_RADIUS+1). Refreshed every BT tick. Used by
    # the DWB obstacle critic for O(1) trajectory scoring.
    _cached_wall_dist: Optional[np.ndarray] = None

    @classmethod
    def build(cls, *, n_cells: int, seed: int,
              n_obstacles: int = 0, layout: str = "track") -> "Sim":
        """Build a Sim around either:
          • layout="track" — wavy bisected circular track (default)
          • layout="maze"  — DFS perfect maze on an n_cells×n_cells grid
        Obstacles only apply to maze layout.
        """
        if layout == "track":
            maze = Maze.generate_track(seed)
        else:
            maze = Maze.generate(n_cells, seed)
            if n_obstacles > 0:
                obs_rng = random.Random(seed * 1000003 + 17 + n_obstacles)
                maze.add_obstacles(obs_rng, n_obstacles)
        start_xy = maze.default_start_xy
        goal_xy = maze.default_goal_xy
        # Heading: rough tangent — for tracks point roughly along the
        # track centreline at the spawn point; for mazes use the previous
        # 45° default.
        if layout == "track":
            cx, cy = maze.size_m / 2.0, maze.size_m / 2.0
            theta0 = math.atan2(start_xy[1] - cy, start_xy[0] - cx) + math.pi / 2
        else:
            theta0 = math.pi / 4
        robot = Robot(x=start_xy[0], y=start_xy[1], theta=theta0)
        grid = DiscoveryGrid(size_m=maze.size_m)
        bt = BehaviorTree()
        bt.state.last_progress_pos = (robot.x, robot.y)
        # Mission: use waypoints if the layout provides them, otherwise
        # fall back to the single goal point.
        mission = list(maze.default_goal_waypoints) or [goal_xy]
        sim = cls(maze=maze, robot=robot, goal=mission[0],
                  grid=grid, bt=bt, mission=mission, mission_idx=0)
        sim.padder = MapPadder(grid)
        # Per-sim RNG (seeded off the maze seed) drives the lidar
        # miss-detection rolls — keeps each scatter robot deterministic.
        sim._sensor_rng = random.Random(seed * 999983 + 13)
        return sim

    # ─── Sensor + discovery ───
    def _sense(self):
        # cast_sensor now handles the mark_ray side-effect itself (one
        # voxel-walk per ray against the maze's wall_voxel_mask) — no
        # second pass needed. Per-ray miss-detection rolls use the
        # sim's deterministic RNG.
        origin, hits = cast_sensor(self.robot, self.maze, self.grid,
                                   rng=getattr(self, '_sensor_rng', None))
        self._last_sensor_origin = origin
        self._last_sensor_hits = hits
        # Refresh the BT's inflation-warning cache (cheap; same cadence as
        # the BT tick because _sense IS only called from the BT tick).
        self._cached_inflation_mask = self.grid.inflation_mask()
        # Wall-distance grid for the DWB obstacle critic.
        self._cached_wall_dist = self.grid.wall_distance_grid(
            max_cells=DWB_CRITIC_RADIUS_CELLS)

    # ─── Planner ───
    def _plan(self) -> str:
        """Plan from robot to goal with two shortcut stages before
        falling back to a full A* search:

        0. **LOS fast-path** — if a Bresenham line from robot to goal
           crosses no blocked cells, emit a 2-point path [robot, goal]
           and skip A* entirely. Most common in late-exploration.
        1. **Subgoal A*** — precompute cells visible from the goal via
           a ray sweep, then A* terminates the moment it pops any
           visible cell. From there, straight line to the goal.
        2. **Fallback** — if the padder-corridor search returns no path,
           retry on the full inflated-wall grid (corridor restriction
           dropped). Same shortcut logic still applies.

        Returns: 'OK', 'OK_LOS', 'OK_SUBGOAL', 'OK_FALLBACK', or 'NO_PATH'.
        """
        sc = self.grid.world_to_cell(*self.robot.rear_axle())
        gc = self.grid.world_to_cell(*self.goal)
        if sc is None or gc is None:
            self.path_world = []
            self.bend_path = []
            return "NO_PATH"
        bg = self.bt.state.bent_goal
        extras: list[tuple[int, int]] = []
        if bg is not None:
            bgc = self.grid.world_to_cell(*bg)
            if bgc is not None:
                extras.append(bgc)
        prev_plan_cells = [
            self.grid.world_to_cell(x, y)
            for (x, y) in self.path_world
        ]
        prev_plan_cells = [c for c in prev_plan_cells if c is not None]
        self.padder.update(sc, gc, prev_plan_cells, extras)
        blocked = self.padder.blocked_mask()

        # Stage 0 — LOS fast-path. If the line from robot cell to goal
        # cell crosses no blocked cells, the planner has nothing useful
        # to do; just point straight at the goal.
        if _bresenham_line_clear(blocked, sc[0], sc[1], gc[0], gc[1]):
            self.path_world = [self.robot.rear_axle(), self.goal]
            self.bend_path = []
            return "OK_LOS"

        # Stage 1 — Subgoal A*. Cells visible from the goal serve as
        # early-termination targets; A* stops at the first one it pops
        # and we splice a straight line to the goal.
        subgoals = _subgoal_mask_from_goal(blocked, gc[0], gc[1])
        cell_path = astar(blocked, sc, gc, subgoal_mask=subgoals)
        status = "OK_SUBGOAL" if cell_path is not None else "NO_PATH"
        used_blocked = blocked
        if cell_path is None:
            # Stage 2 — Fallback: drop the corridor restriction, keep
            # only the inflated walls.
            wall_only = self.grid.inflated_obstacle_mask()
            if _bresenham_line_clear(wall_only, sc[0], sc[1], gc[0], gc[1]):
                self.path_world = [self.robot.rear_axle(), self.goal]
                self.bend_path = []
                return "OK_LOS"
            sub_fb = _subgoal_mask_from_goal(wall_only, gc[0], gc[1])
            cell_path = astar(wall_only, sc, gc, subgoal_mask=sub_fb)
            used_blocked = wall_only
            status = "OK_FALLBACK" if cell_path is not None else "NO_PATH"
            # No "last-resort plan through inflation" stage — when the
            # planner can't find a path that respects the costmap, the
            # BT's NO_PATH handler decides what to do (GRADIENT_ESCAPE
            # if the robot is in the halo, GOAL_BEND otherwise).
        if cell_path is None:
            self.path_world = []
            self.bend_path = []
            return "NO_PATH"
        # Convert the cell path to world coords, then splice in the goal
        # if the A* terminated at a subgoal (not at the real goal cell).
        self.path_world = [self.grid.cell_to_world(i, j) for (i, j) in cell_path]
        last_i, last_j = cell_path[-1]
        if (last_i, last_j) != gc:
            self.path_world.append(self.goal)

        if bg is not None:
            bgc = self.grid.world_to_cell(*bg)
            if bgc is not None:
                bp = astar(used_blocked, sc, bgc)
                self.bend_path = (
                    [self.grid.cell_to_world(i, j) for (i, j) in bp]
                    if bp is not None else [self.robot.rear_axle(), bg]
                )
            else:
                self.bend_path = [self.robot.rear_axle(), bg]
        else:
            self.bend_path = []
        return status

    # ─── Breadcrumbs ───
    def _maybe_drop_breadcrumb(self):
        if not self.breadcrumbs:
            self.breadcrumbs.append(self.robot.rear_axle())
            return
        bx, by = self.breadcrumbs[-1]
        if math.hypot(self.robot.x - bx, self.robot.y - by) > BREADCRUMB_DROP_M:
            self.breadcrumbs.append(self.robot.rear_axle())
            if len(self.breadcrumbs) > BREADCRUMB_MAX:
                self.breadcrumbs.pop(0)

    # ─── BT logic ───
    def _path_first_seg_bearing_err_signed(self) -> Optional[float]:
        """Signed bearing error (rad, in (-π, π]) of the first useful
        path waypoint ahead of the robot. Returns None if no waypoint.
        Same pure-pursuit-style lookahead as the absolute-value version.
        """
        if len(self.path_world) < 2:
            return None
        rx, ry = self.robot.rear_axle()
        best_k = 0
        best_d2 = 1e18
        for k, (px, py) in enumerate(self.path_world):
            d2 = (px - rx) ** 2 + (py - ry) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_k = k
        for k in range(best_k + 1, len(self.path_world)):
            px, py = self.path_world[k]
            if math.hypot(px - rx, py - ry) > 0.6:
                bearing = math.atan2(py - ry, px - rx)
                err = bearing - self.robot.theta
                while err > math.pi:
                    err -= 2 * math.pi
                while err <= -math.pi:
                    err += 2 * math.pi
                return err
        return None

    def _path_first_seg_bearing_err(self) -> Optional[float]:
        """Return absolute bearing error (rad) from robot heading to the
        first useful path waypoint AHEAD of the robot's projection onto
        the plan. Returns None if no such waypoint exists.

        Pure-pursuit-style lookahead: find the closest waypoint, then
        scan FORWARD in path order. The old version scanned from the
        beginning of the path, which picked early-plan waypoints that
        the robot had reversed past — those appeared "forward" once the
        robot was behind them, even when the plan as a whole still bent
        backward.
        """
        if len(self.path_world) < 2:
            return None
        rx, ry = self.robot.rear_axle()
        best_k = 0
        best_d2 = 1e18
        for k, (px, py) in enumerate(self.path_world):
            d2 = (px - rx) ** 2 + (py - ry) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_k = k
        for k in range(best_k + 1, len(self.path_world)):
            px, py = self.path_world[k]
            if math.hypot(px - rx, py - ry) > 0.6:
                bearing = math.atan2(py - ry, px - rx)
                err = bearing - self.robot.theta
                while err > math.pi:
                    err -= 2 * math.pi
                while err <= -math.pi:
                    err += 2 * math.pi
                return abs(err)
        return None

    def _path_first_seg_behind(self) -> bool:
        """True if the planned path's first useful segment requires the
        robot to drive backwards (carrot bearing > 90° + hysteresis)."""
        err = self._path_first_seg_bearing_err()
        return err is not None and err > math.pi / 2 + 0.15

    def _path_solidly_forward(self) -> bool:
        """Stricter than `not _path_first_seg_behind()`. True only when
        the first path segment is clearly aligned with the body's +x
        (err < 60°). Used by BREADCRUMB_REVERSE to decide whether to bail
        on a pop — the previous `not behind` test let near-perpendicular
        plans (err ≈ 95°) trigger an exit prematurely."""
        err = self._path_first_seg_bearing_err()
        return err is not None and err < math.pi / 3

    def _goal_behind(self) -> bool:
        rx, ry = self.robot.rear_axle()
        gx, gy = self.bt.state.bent_goal or self.goal
        bearing = math.atan2(gy - ry, gx - rx)
        err = bearing - self.robot.theta
        while err > math.pi:
            err -= 2 * math.pi
        while err <= -math.pi:
            err += 2 * math.pi
        return abs(err) > math.pi / 2 + 0.15

    def _tick_bt(self):
        """Mirror of bt_nav.xml — checks transitions in the same priority
        order as the BT.

        Source: AutoNav_25-26 (path_following branch),
        isaac_ros-dev/src/slam/behavior_trees/bt_nav.xml.
        """
        s = self.bt.state
        # Progress watchdog (mirrors progress_checker_v2).
        dx = self.robot.x - s.last_progress_pos[0]
        dy = self.robot.y - s.last_progress_pos[1]
        d = math.hypot(dx, dy)
        self.last_progress_dist = d
        if d > PROGRESS_DIST_M:
            s.last_progress_pos = self.robot.rear_axle()
            s.last_progress_time = self.t
        stuck = (self.t - s.last_progress_time) > PROGRESS_STALL_SEC

        # 1. Recovery escalation when already in a recovery state.
        if s.name == "BACKUP_RECOVERY":
            travelled = math.hypot(self.robot.x - s.backup_start[0],
                                   self.robot.y - s.backup_start[1])
            if travelled >= BACKUP_DIST_M or self.t - s.entered_at > 4.0:
                # Backup done. Advance the round-robin cursor and hand
                # off — next stuck event will fire the next step (GRADIENT).
                if s.in_recovery_round_robin:
                    s.recovery_round_step = (s.recovery_round_step + 1) % 3
                s.last_progress_time = self.t
                s.last_progress_pos = self.robot.rear_axle()
                self.bt.enter("WAIT_TRANSIENT_RECOVERY", self.t)
            return
        if s.name == "GRADIENT_ESCAPE":
            # Two-phase state machine:
            #   Phase 1 (reactive escape): drive out of the costmap halo.
            #   Phase 2 (alignment): once out, replan and rotate in place
            #     to face the new path's first segment, THEN hand control
            #     back to NORMAL_FOLLOWING.
            rc = self.grid.world_to_cell(*self.robot.rear_axle())
            in_halo = (
                rc is not None
                and self._cached_inflation_mask is not None
                and self._cached_inflation_mask[rc]
            )
            timed_out = self.t - s.gradient_start_time > GRADIENT_ESC_SEC
            if timed_out:
                if s.in_recovery_round_robin:
                    s.recovery_round_step = (s.recovery_round_step + 1) % 3
                s.last_progress_time = self.t
                s.last_progress_pos = self.robot.rear_axle()
                s.last_gradient_exit_time = self.t
                s.gradient_aligning = False
                self.bt.enter("WAIT_TRANSIENT_RECOVERY", self.t)
                return
            if not s.gradient_aligning:
                if in_halo:
                    # Still escaping; if we'd previously left and drifted
                    # back, clear the exit marker so the margin restarts.
                    s.gradient_exit_pos = None
                    return
                # Out of halo. Record the exit position the first time,
                # then keep driving in escape direction until we've gone
                # GRADIENT_POST_ESCAPE_M past that point.
                if s.gradient_exit_pos is None:
                    s.gradient_exit_pos = self.robot.rear_axle()
                    return
                ex, ey = s.gradient_exit_pos
                if math.hypot(self.robot.x - ex, self.robot.y - ey) \
                        >= GRADIENT_POST_ESCAPE_M:
                    # Margin met — replan and switch to alignment.
                    self.last_plan_status = self._plan()
                    self.last_replan_time = self.t
                    s.gradient_aligning = True
                return
            # Phase 2 — aligning to the planned path heading. Drifted
            # back into the halo? Bounce back to phase 1.
            if in_halo:
                s.gradient_aligning = False
                return
            err = self._path_first_seg_bearing_err()
            if err is None or err < math.radians(20):
                if s.in_recovery_round_robin:
                    s.recovery_round_step = (s.recovery_round_step + 1) % 3
                s.last_progress_time = self.t
                s.last_progress_pos = self.robot.rear_axle()
                s.last_gradient_exit_time = self.t
                s.gradient_aligning = False
                self.bt.enter("WAIT_TRANSIENT_RECOVERY", self.t)
            return
        if s.name == "CLEAR_AROUND_ROBOT":
            # One-shot 1 m local wipe of WALL_KNOWN cells around the
            # robot's rear axle (matches bt_nav.xml:136
            # ClearCostmapAroundRobot reset_distance="1.0"). Free/unknown
            # cells stay as-is — only the discovered walls clear so the
            # planner can retry the same route after a transient
            # mis-detection.
            self._wipe_walls_around(self.robot.rear_axle(), 1.0)
            self.bt.enter("WAIT_TRANSIENT_RECOVERY", self.t)
            return
        if s.name == "WAIT_TRANSIENT_RECOVERY":
            if self.t > s.wait_until:
                s.last_progress_time = self.t
                s.last_progress_pos = self.robot.rear_axle()
                self.bt.enter("NORMAL_FOLLOWING", self.t)
            return
        if s.name == "FORWARD_BLOCKED_BREADCRUMB_REVERSE":
            # On each breadcrumb pop:
            #  1. Force a fresh replan against the robot's new pose.
            #  2. If the new plan is SOLIDLY FORWARD (err < 60°), exit
            #     straight to NORMAL_FOLLOWING — no goal bend needed,
            #     the planner already has a usable forward path.
            # Other exits, which DO use GOAL_BEND-away-from-costmap:
            #  • BREADCRUMB_CONSUME_LIMIT crumbs eaten and path still bent.
            #  • Buffer empty or session timeout.
            popped_now = False
            if self.breadcrumbs:
                bx, by = self.breadcrumbs[-1]
                if math.hypot(self.robot.x - bx, self.robot.y - by) < 0.35:
                    self.breadcrumbs.pop()
                    s.breadcrumbs_consumed += 1
                    popped_now = True
            if popped_now:
                # Fresh plan from the new pose.
                self.last_plan_status = self._plan()
                self.last_replan_time = self.t
                if s.breadcrumb_exit_pending:
                    # This pop is the bonus margin crumb — now exit.
                    s.path_behind_since = None
                    s.last_progress_pos = self.robot.rear_axle()
                    s.last_progress_time = self.t
                    s.breadcrumb_exit_pending = False
                    self.bt.enter("NORMAL_FOLLOWING", self.t)
                    return
                if self._path_solidly_forward():
                    # Path is good, but consume ONE MORE crumb before
                    # handing back — gives a buffer so NORMAL_FOLLOWING
                    # doesn't immediately slam into the costmap.
                    s.breadcrumb_exit_pending = True
            def _bend_away():
                self.bt.enter("GOAL_BEND", self.t,
                              bent_goal=self._compute_bent_goal_away_from_costmap())
            if s.breadcrumbs_consumed >= BREADCRUMB_CONSUME_LIMIT:
                _bend_away()
                return
            if not self.breadcrumbs or self.t - s.entered_at > 12.0:
                _bend_away()
                return
            return
        if s.name == "FORWARD_BLOCKED_WAIT_FOR_REPLAN":
            # Wait 1.1 s for the next replan to (hopefully) be forward.
            if self.t - s.entered_at > 1.1:
                if self._path_first_seg_behind():
                    # Still behind → goal bend.
                    self.bt.enter("GOAL_BEND", self.t,
                                  bent_goal=self._compute_bent_goal())
                else:
                    self.bt.enter("NORMAL_FOLLOWING", self.t)
            return
        if s.name == "GOAL_BEND":
            # Transient steering nudge: hold the bent goal for at most
            # GOAL_BEND_DURATION_SEC, then drop it and let NORMAL_FOLLOWING
            # re-plan against the real goal. Also bail early if we
            # somehow reach the bent waypoint before the timeout.
            if s.bent_goal is None \
                    or self.t - s.entered_at > GOAL_BEND_DURATION_SEC:
                s.bent_goal = None
                s.last_progress_pos = self.robot.rear_axle()
                s.last_progress_time = self.t
                self.bt.enter("NORMAL_FOLLOWING", self.t)
                return
            bx, by = s.bent_goal
            if math.hypot(self.robot.x - bx, self.robot.y - by) < 1.0:
                s.bent_goal = None
                self.bt.enter("NORMAL_FOLLOWING", self.t)
            return

        # 2. From NORMAL_FOLLOWING: pick a transition.
        # If the robot makes real distance forward, it has clearly escaped
        # whatever recovery loop we were in — reset the recovery cursors
        # so the NEXT stuck event starts fresh from CLEAR_AROUND_ROBOT
        # again rather than the round-robin mid-cycle.
        if (self.t - s.last_progress_time) < 0.1 and \
                self.last_progress_dist > 0.5:
            s.around_robot_attempts = 0
            s.in_recovery_round_robin = False
            s.recovery_round_step = 0

        # Stuck check FIRST — "stuck" (no PROGRESS_DIST_M of motion in
        # PROGRESS_STALL_SEC) is the strongest signal something is wrong,
        # and it must preempt the inflation trigger so the BT can escalate
        # to BACKUP / CLEAR_AROUND / round-robin when GRADIENT_ESCAPE
        # genuinely can't get the robot out of a halo wedge.
        if stuck:
            # FollowPathRecovery: first 2 stalls get a targeted 1 m local
            # clear before we escalate to the round-robin RecoveryFallback.
            if not s.in_recovery_round_robin and s.around_robot_attempts < 2:
                s.around_robot_attempts += 1
                self.bt.enter("CLEAR_AROUND_ROBOT", self.t)
                return
            # RecoveryFallback (round-robin BACKUP → GRADIENT_ESCAPE →
            # CLEAR_AROUND_ROBOT). Global costmap clears are NOT used.
            s.in_recovery_round_robin = True
            step = s.recovery_round_step % 3
            if step == 0:
                self.bt.enter("BACKUP_RECOVERY", self.t,
                              backup_start=self.robot.rear_axle())
            elif step == 1:
                self.bt.enter("GRADIENT_ESCAPE", self.t)
            else:
                self.bt.enter("CLEAR_AROUND_ROBOT", self.t)
            return

        # No auto-trigger on "robot in inflation halo" — the BT operates
        # in discrete steps like the real robot. GRADIENT_ESCAPE is only
        # entered from the round-robin RecoveryFallback (step 1 after
        # BACKUP). Once entered, GRADIENT_ESCAPE exits as soon as the
        # robot's centre is clear of the inflation halo (see the
        # GRADIENT_ESCAPE state handler above).

        if self.last_plan_status == "NO_PATH":
            # ComputePathRecovery: wipe the global costmap and retry, up
            # to 8 attempts total (matches bt_nav.xml:84 RecoveryNode
            # number_of_retries=8). After that, escalate to GOAL_BEND.
            s.consec_planner_fails += 1
            s.n_planner_failures += 1
            # Per user spec: NO_PATH typically means the robot is sitting
            # inside the costmap halo and the planner can't reach the
            # goal from there. Fire GRADIENT_ESCAPE if the robot really
            # is in the halo; otherwise GOAL_BEND (no path even from a
            # clear cell → the goal itself is unreachable, bend it).
            rc = self.grid.world_to_cell(*self.robot.rear_axle())
            in_inflation = (
                rc is not None
                and self._cached_inflation_mask is not None
                and self._cached_inflation_mask[rc]
            )
            if in_inflation:
                self.bt.enter("GRADIENT_ESCAPE", self.t)
            else:
                self.bt.enter("GOAL_BEND", self.t,
                              bent_goal=self._compute_bent_goal())
            s.consec_planner_fails = 0
            return
        else:
            s.consec_planner_fails = 0
        if self._path_first_seg_behind():
            if s.path_behind_since is None:
                s.path_behind_since = self.t
            elif (self.t - s.path_behind_since) >= PATH_BEHIND_PERSIST_SEC:
                # The geometry has been consistently behind us long enough
                # that this isn't just a mid-turn artefact — fire the
                # appropriate FORWARD_BLOCKED recovery.
                s.path_behind_since = None
                if self._goal_behind():
                    self.bt.enter("GOAL_BEND", self.t,
                                  bent_goal=self._compute_bent_goal())
                elif self.breadcrumbs:
                    self.bt.enter("FORWARD_BLOCKED_BREADCRUMB_REVERSE", self.t)
                else:
                    self.bt.enter("FORWARD_BLOCKED_WAIT_FOR_REPLAN", self.t)
                return
        else:
            s.path_behind_since = None

    def _wipe_walls_around(self, centre: tuple[float, float],
                            radius_m: float):
        """Clear WALL_KNOWN cells within `radius_m` back to UNKNOWN."""
        self._wipe_around(centre, radius_m, wipe_free=False)

    def _wipe_around(self, centre: tuple[float, float], radius_m: float,
                      *, wipe_free: bool):
        """Local discovery wipe inside a disc of `radius_m`. Always
        clears WALL_KNOWN cells; if `wipe_free`, also resets FREE_KNOWN
        cells back to UNKNOWN. Keeping wipes LOCAL is critical — wiping
        the entire grid causes the planner to draw a straight line to
        the goal through invisible walls and the robot drives backwards
        across the maze."""
        cx, cy = centre
        c = self.grid.world_to_cell(cx, cy)
        if c is None:
            return
        ci, cj = c
        rad = int(math.ceil(radius_m / self.grid.res))
        n = self.grid.n
        r2 = rad * rad
        for ii in range(max(0, ci - rad), min(n, ci + rad + 1)):
            for jj in range(max(0, cj - rad), min(n, cj + rad + 1)):
                if (ii - ci) ** 2 + (jj - cj) ** 2 > r2:
                    continue
                v = self.grid.cells[ii, jj]
                if v == WALL_KNOWN:
                    self.grid.cells[ii, jj] = UNKNOWN
                elif wipe_free and v == FREE_KNOWN:
                    self.grid.cells[ii, jj] = UNKNOWN

    def _compute_bent_goal(self) -> tuple[float, float]:
        """Emit an intermediate waypoint offset by GOAL_BEND_RAD from the
        true goal bearing, on whichever side has more free space."""
        gx, gy = self.goal
        rx, ry = self.robot.rear_axle()
        bearing = math.atan2(gy - ry, gx - rx)
        # Try both sides; pick the one whose first 4 m of ray distance has
        # more free space.
        best_a = bearing
        best_score = -1e18
        for sign in (-1, +1):
            a = self.robot.theta + sign * GOAL_BEND_RAD * 0.6
            dx, dy = math.cos(a), math.sin(a)
            best_t = SENSOR_RANGE_M
            for seg in self.maze.segs_near(
                    rx + dx * 4, ry + dy * 4, 4.5):
                t = _ray_segment_hit(rx, ry, dx, dy, *seg, best_t)
                if t is not None and t < best_t:
                    best_t = t
            if best_t > best_score:
                best_score = best_t
                best_a = a
        dist = min(3.0, best_score - 0.5)
        return (rx + dist * math.cos(best_a), ry + dist * math.sin(best_a))

    def _compute_bent_goal_away_from_costmap(self) -> tuple[float, float]:
        """Pick an intermediate waypoint that maximises distance from
        the costmap walls. Used after BREADCRUMB_REVERSE exhausts its
        budget and the path is still bent — biases the bent goal toward
        open space so the next replan won't immediately re-bend.

        Samples 24 candidate headings around the robot, walks ~4 m down
        each, and scores the resulting cell by the cached wall-distance
        grid. Highest wall distance wins.
        """
        wd = self._cached_wall_dist
        rx, ry = self.robot.rear_axle()
        if wd is None:
            return self._compute_bent_goal()
        max_d = DWB_CRITIC_RADIUS_CELLS + 1
        sample_dist = 4.0
        best_a = self.robot.theta
        best_score = -1
        best_dist = 2.5
        for k in range(24):
            a = 2 * math.pi * k / 24
            dx, dy = math.cos(a), math.sin(a)
            # Walk along the ray; stop before hitting a wall so the bent
            # goal is reachable.
            t = sample_dist
            for seg in self.maze.segs_near(
                    rx + dx * sample_dist / 2, ry + dy * sample_dist / 2,
                    sample_dist / 2 + 0.5):
                th = _ray_segment_hit(rx, ry, dx, dy, *seg, t)
                if th is not None and th < t:
                    t = th
            target_t = max(1.0, t - 0.5)
            tx = rx + target_t * dx
            ty = ry + target_t * dy
            tc = self.grid.world_to_cell(tx, ty)
            if tc is None:
                continue
            d = int(wd[tc])
            # Prefer cells that are AT or BEYOND the critic radius (i.e.
            # outside the inflation halo), and break ties by clear range.
            score = d * 10 + int(t * 10)
            if d >= max_d:
                score += 1000
            if score > best_score:
                best_score = score
                best_a = a
                best_dist = target_t
        return (rx + best_dist * math.cos(best_a),
                ry + best_dist * math.sin(best_a))

    # ─── Per-state controller wiring ───
    def _controller_for_state(self) -> ControllerOutput:
        s = self.bt.state
        if s.name == "NORMAL_FOLLOWING":
            # DWB-style sampler with the obstacle-aware critic. Falls back
            # to pure pursuit internally if no wall-distance grid exists
            # yet (e.g. tick 0 before first sensor sweep).
            return dwb_with_obstacle_critic(
                self.robot, self.path_world,
                self._cached_wall_dist, self.grid.res)
        if s.name == "GOAL_BEND":
            if s.bent_goal:
                # Follow the intermediate bend-path if Dijkstra produced one,
                # otherwise fall back to a straight 2-point segment to the
                # bent waypoint. Critic also applied here so the bent route
                # still avoids walls.
                pp = self.bend_path or [self.robot.rear_axle(), s.bent_goal]
                return dwb_with_obstacle_critic(
                    self.robot, pp,
                    self._cached_wall_dist, self.grid.res)
            return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
        if s.name == "BACKUP_RECOVERY":
            # Drive backwards at BACKUP_SPEED for BACKUP_DIST_M.
            tgt = -BACKUP_SPEED
            a_long = KP_LIN * (tgt - self.robot.u)
            a_yaw = -KP_ANG * self.robot.omega
            F_total = ROBOT_MASS_KG * a_long / 10.0
            tau = INERTIA_REAR * a_yaw / 6.0
            return ControllerOutput(0.5 * F_total - tau / TRACK_WIDTH_M,
                                    0.5 * F_total + tau / TRACK_WIDTH_M,
                                    tgt, 0.0, False)
        if s.name == "FORWARD_BLOCKED_BREADCRUMB_REVERSE":
            # Dedicated backwards-drive controller. pure_pursuit zeros
            # v_des when the carrot is within GOAL_TOLERANCE_M; the
            # breadcrumb-pop threshold (0.35 m) is INSIDE that zone, so
            # routing through pure_pursuit stalled the robot exactly at
            # the goal-tolerance ring without ever popping. Drive at a
            # constant negative v with steering aimed so the REAR of the
            # body tracks the breadcrumb.
            if not self.breadcrumbs:
                return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
            bx, by = self.breadcrumbs[-1]
            rx, ry = self.robot.rear_axle()
            bearing = math.atan2(by - ry, bx - rx)
            err = bearing - self.robot.theta
            while err > math.pi:
                err -= 2 * math.pi
            while err <= -math.pi:
                err += 2 * math.pi
            # Going backwards: flip the error onto the body's -x side so
            # the controller steers to keep the breadcrumb behind us.
            err_back = err - math.pi if err > 0 else err + math.pi
            v_des = -BREADCRUMB_SPEED
            omega_des = max(-1.0, min(1.0, 1.5 * err_back))
            a_long = KP_LIN * (v_des - self.robot.u)
            a_yaw = KP_ANG * (omega_des - self.robot.omega)
            F_total = ROBOT_MASS_KG * a_long / 10.0
            tau_total = INERTIA_REAR * a_yaw / 6.0
            F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
            F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
            return ControllerOutput(F_left, F_right, v_des, omega_des, False)
        if s.name == "GRADIENT_ESCAPE":
            # Phase 2 — out of the halo, rotate in place until aligned
            # with the planned path. The BT exit fires when err < 20°.
            if s.gradient_aligning:
                err = self._path_first_seg_bearing_err_signed()
                if err is None:
                    return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
                v_des = 0.0
                omega_des = max(-1.4, min(1.4, 2.4 * err))
                a_long = KP_LIN * (v_des - self.robot.u)
                a_yaw  = KP_ANG * (omega_des - self.robot.omega)
                F_total   = ROBOT_MASS_KG * a_long / 10.0
                tau_total = INERTIA_REAR * a_yaw / 6.0
                F_left  = 0.5 * F_total - tau_total / TRACK_WIDTH_M
                F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
                return ControllerOutput(F_left, F_right, v_des, omega_des, False)
            # Phase 1 — PURELY reactive escape. The robot cannot follow
            # a goal when it is inside the costmap halo (the planner
            # returns NO_PATH, which is what kicked off this state).
            # Don't even go through pure_pursuit; compute wheel forces
            # directly from the local wall-distance gradient.
            #
            # Method: sample wall-distance at 16 points 2.5 m around the
            # robot, take the vector sum weighted by distance value.
            # That points toward the direction with the most "open
            # space"; turn and drive that way.
            wd = self._cached_wall_dist
            rx, ry = self.robot.rear_axle()
            if wd is None:
                return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
            sample_dist = 2.5
            sum_dx, sum_dy = 0.0, 0.0
            for k in range(16):
                a = 2 * math.pi * k / 16
                tx = rx + sample_dist * math.cos(a)
                ty = ry + sample_dist * math.sin(a)
                tc = self.grid.world_to_cell(tx, ty)
                if tc is None:
                    continue
                d = int(wd[tc])
                sum_dx += d * math.cos(a)
                sum_dy += d * math.sin(a)
            if math.hypot(sum_dx, sum_dy) < 1e-3:
                return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
            bearing = math.atan2(sum_dy, sum_dx)
            err = bearing - self.robot.theta
            while err > math.pi:
                err -= 2 * math.pi
            while err <= -math.pi:
                err += 2 * math.pi
            # Smart fwd/back decision: sample the wall-distance grid 1.2 m
            # ahead AND 1.2 m behind the robot in its OWN body-x axis.
            # Whichever side has the larger wall-distance is the side
            # that escapes the halo faster — drive that way. Falls back
            # to the bearing-based half-plane test if the samples tie.
            probe_dist = 1.2
            cos_t = math.cos(self.robot.theta)
            sin_t = math.sin(self.robot.theta)
            fc = self.grid.world_to_cell(
                rx + probe_dist * cos_t, ry + probe_dist * sin_t)
            bc = self.grid.world_to_cell(
                rx - probe_dist * cos_t, ry - probe_dist * sin_t)
            f_d = int(wd[fc]) if fc is not None else 0
            b_d = int(wd[bc]) if bc is not None else 0
            if f_d == b_d:
                backwards = abs(err) > math.pi / 2
            else:
                backwards = b_d > f_d
            if backwards:
                err_steer = err - math.pi if err > 0 else err + math.pi
                v_des = -GRADIENT_ESC_SPEED
            else:
                err_steer = err
                v_des = GRADIENT_ESC_SPEED
            omega_des = max(-1.4, min(1.4, 2.0 * err_steer))
            a_long = KP_LIN * (v_des - self.robot.u)
            a_yaw  = KP_ANG * (omega_des - self.robot.omega)
            F_total   = ROBOT_MASS_KG * a_long / 10.0
            tau_total = INERTIA_REAR * a_yaw / 6.0
            F_left  = 0.5 * F_total - tau_total / TRACK_WIDTH_M
            F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
            return ControllerOutput(F_left, F_right, v_des, omega_des, False)
        # WAIT / CLEAR_COSTMAP / WAIT_TRANSIENT_RECOVERY / etc.: hold still.
        return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)

    # ─── Top-level step ───
    def step(self, dt: float):
        self.t += dt
        # Cheap sensing every physics tick is too expensive — only refresh
        # the rays + discovery grid at the BT-tick cadence.
        if self.t - self.last_bt_tick >= BT_TICK_PERIOD_S:
            # Real-robot parity: during recovery actions (reversing,
            # gradient escape) the LiDAR+camera pipeline is paused so
            # motion-blurred sweeps don't poison the costmap. Breadcrumb
            # drops are paused for the same reason — the ledger shouldn't
            # re-pad as the robot retraces its own path.
            sensing_halted = self.bt.state.name in SENSING_HALTED_STATES
            if not sensing_halted:
                self._sense()
                self._maybe_drop_breadcrumb()
            # Replanning DOES still fire during recovery actions — the
            # BT needs a fresh plan against the robot's new pose so the
            # at-pop exit check can fire when reversing has cleared the
            # bend. (Sensing stays halted, so the planner operates on
            # the frozen discovery grid + the updated robot position.)
            if self.t - self.last_replan_time > REPLAN_PERIOD_S:
                self.last_plan_status = self._plan()
                self.last_replan_time = self.t
            self._tick_bt()
            self.last_controller = self._controller_for_state()
            self.last_bt_tick = self.t

        self.robot.step(self.last_controller.F_left,
                        self.last_controller.F_right,
                        dt, self.maze)

        # Mission progress: advance to the next waypoint when we hit the
        # current one; flag `done` only when the whole mission is done.
        gx, gy = self.goal
        if not self.done and math.hypot(self.robot.x - gx, self.robot.y - gy) \
                < GOAL_TOLERANCE_M:
            if self.mission_idx + 1 < len(self.mission):
                self.mission_idx += 1
                self.goal = self.mission[self.mission_idx]
                # Force a replan on the next BT tick by invalidating the
                # cached plan; clear bent-goal scratch so the controller
                # starts fresh.
                self.path_world = []
                self.bend_path = []
                self.bt.state.bent_goal = None
                # Reset the stuck timer so the recovery state machine
                # doesn't fire on the goal-switch moment.
                self.bt.state.last_progress_pos = self.robot.rear_axle()
                self.bt.state.last_progress_time = self.t
            else:
                self.done = True
                self.done_time = self.t

        # Trail policy: keep the ENTIRE driven path, but only sample a
        # new point every TRAIL_DROP_M of motion. Drawing 240 points/sec
        # (one per physics tick) blew the renderer past the budget; drops
        # at ~0.15 m give a visually continuous solid line at any zoom
        # while keeping the per-frame `draw.lines` cost bounded — a 100 m
        # run is ~700 points, not 24 000.
        if not self.trail:
            self.trail.append(self.robot.rear_axle())
        else:
            tx, ty = self.trail[-1]
            if math.hypot(self.robot.x - tx, self.robot.y - ty) >= TRAIL_DROP_M:
                self.trail.append(self.robot.rear_axle())


# ──────────────────────────────────────────────────────────────────────────────
#                          Robot sprite (top-down)
# ──────────────────────────────────────────────────────────────────────────────
# Marker color the user paints into the source artwork to register the sprite
# against the simulator's body frame. Four pixels are expected: the upper two
# straddle the rear knife-edge wheels, the middle pixel sits at the COM, and
# the bottom pixel sits at the front caster (frictionless contact). In the
# sprite's native pixel orientation, the rear knife edges are at the top and
# the front caster is at the bottom — so the rear→caster pixel vector defines
# +body-x (the robot's forward direction), and the upper-pair separation
# defines the track width.
SPRITE_MARKER_RGB     = (0xE4 / 255, 0x87 / 255, 0x87 / 255)
SPRITE_MARKER_TOL     = 0.15     # per-channel tolerance (some art tools shift
                                 # painted #E48787 toward #FF6682 on PNG export
                                 # — 0.15 catches both without grabbing the
                                 # red emergency-stop button etc.).
# Cap on the sprite's long side after registration. The native asset is
# ~1.9 MP; the affine-rotated copy is resampled to screen pixels EVERY
# frame, costing ~25 ms at the native size. 256 px long-side resamples in
# under a millisecond and is still sharper than the screen-space footprint
# of a ~1.5 m wide robot inside an 8 m-radius camera window.
SPRITE_MAX_LONG_SIDE  = 256


@dataclass
class RobotSprite:
    """Holds the raw RGBA image plus a body-frame transform: place the sprite
    so its embedded markers line up with rear-axle midpoint / COM / caster
    at the robot's current pose."""
    image: np.ndarray                       # H × W × 4, float[0,1]
    px_per_m: float                         # pixels per world metre
    # All offsets are in the sprite's NATIVE pixel frame (origin = upper-left,
    # +x right, +y down), but expressed in METRES already (divided by
    # px_per_m), so we can use them directly with matplotlib's data-coord
    # transform after rotation.
    rear_axle_px: tuple[float, float]       # midpoint of upper marker pair
    caster_px:    tuple[float, float]
    com_px:       tuple[float, float]
    sprite_forward_deg: float               # angle of (rear→caster) in pixel space

    @classmethod
    def try_load(cls, path: Path) -> Optional["RobotSprite"]:
        if not path.exists():
            return None
        img = mpimg.imread(str(path))
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        # Ensure 4 channels.
        if img.ndim == 2:
            img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
        elif img.shape[2] == 3:
            alpha = np.ones(img.shape[:2] + (1,), dtype=img.dtype)
            img = np.concatenate([img, alpha], axis=-1)
        # Marker scan runs on the float copy (channel tolerance is per-unit).
        # We'll convert to uint8 AFTER the downsample so the matplotlib draw
        # path doesn't have to re-quantise on every frame.

        r, g, b = SPRITE_MARKER_RGB
        tol = SPRITE_MARKER_TOL
        mask = (
            (np.abs(img[..., 0] - r) < tol)
            & (np.abs(img[..., 1] - g) < tol)
            & (np.abs(img[..., 2] - b) < tol)
            & (img[..., 3] > 0.5)
        )
        ys, xs = np.where(mask)
        if len(xs) == 0:
            print(f"[sprite] {path.name}: no #E48787 reference pixels found; "
                  "falling back to stick figure.", file=sys.stderr)
            return None

        # Cluster connected (or near-by) marker pixels — there might be a few
        # adjacent ones for each painted blob.
        markers = _cluster_pixels(xs, ys)
        if len(markers) < 4:
            print(f"[sprite] {path.name}: found only {len(markers)} marker "
                  "blobs (expected 4: 2 knife edges + COM + caster); falling "
                  "back to stick figure.", file=sys.stderr)
            return None
        # Sort blobs by y (pixel-down), then by x.
        markers.sort(key=lambda c: (c[1], c[0]))
        # The top 2 (smallest y) are the knife-edge pair. The bottom 1
        # (largest y) is the caster. The remaining one in the middle is COM.
        if len(markers) > 4:
            # Keep the 4 most extreme: top 2 by y, bottom 1 by y, and the one
            # in the middle band closest to the vertical centerline.
            top2 = markers[:2]
            bot1 = markers[-1]
            mid_candidates = markers[2:-1]
            center_x = 0.5 * (top2[0][0] + top2[1][0])
            mid_candidates.sort(key=lambda c: abs(c[0] - center_x))
            mid = mid_candidates[0]
            chosen = [*top2, mid, bot1]
        else:
            chosen = markers
        chosen.sort(key=lambda c: c[1])
        knife_l, knife_r = sorted(chosen[:2], key=lambda c: c[0])
        com = chosen[2]
        caster = chosen[3]

        rear_axle = (0.5 * (knife_l[0] + knife_r[0]),
                     0.5 * (knife_l[1] + knife_r[1]))
        # Downsample the source if it's much larger than the on-screen
        # footprint will ever need (default 256 px long side). The marker
        # coordinates and px-per-m scale shrink by the same factor, so
        # registration is exact. Big perf win: the per-frame affine resample
        # is ~O(src_pixels) — a 5× shrink is ~25× faster to redraw.
        h_px, w_px = img.shape[:2]
        long_side = max(h_px, w_px)
        if long_side > SPRITE_MAX_LONG_SIDE:
            scale = SPRITE_MAX_LONG_SIDE / long_side
            try:
                from PIL import Image
                new_w = max(1, int(round(w_px * scale)))
                new_h = max(1, int(round(h_px * scale)))
                pil = Image.fromarray((img * 255.0).astype(np.uint8), mode="RGBA")
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                img = np.asarray(pil, dtype=np.float32) / 255.0
                knife_l = (knife_l[0] * scale, knife_l[1] * scale)
                knife_r = (knife_r[0] * scale, knife_r[1] * scale)
                com     = (com[0]     * scale, com[1]     * scale)
                caster  = (caster[0]  * scale, caster[1]  * scale)
                rear_axle = (rear_axle[0] * scale, rear_axle[1] * scale)
                print(f"[sprite] downsampled {w_px}×{h_px} → "
                      f"{new_w}×{new_h} (scale {scale:.3f})")
            except Exception as exc:
                print(f"[sprite] downsample skipped ({exc}); native size kept.",
                      file=sys.stderr)
        # Pre-quantise the (possibly downsampled) image to uint8 RGBA so
        # matplotlib's draw path doesn't have to convert float→uint8 on
        # every frame. Profiled as the dominant single cost (`_rgb_to_rgba`
        # 1.95s/120 frames in the native float pipeline).
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Pixel scale: the rear-axle → caster distance in the sprite must
        # equal the robot's wheelbase (real-world metres).
        dx_px = caster[0] - rear_axle[0]
        dy_px = caster[1] - rear_axle[1]
        wheelbase_px = math.hypot(dx_px, dy_px)
        if wheelbase_px < 1.0:
            print(f"[sprite] {path.name}: rear-axle and caster markers "
                  "coincide; falling back to stick figure.", file=sys.stderr)
            return None
        px_per_m = wheelbase_px / WHEELBASE_M
        # Pixel angle of body-forward: +x_pixel = right, +y_pixel = down.
        # math.atan2 gives an angle measured from +x_pixel toward +y_pixel.
        sprite_forward_deg = math.degrees(math.atan2(dy_px, dx_px))
        # Verify track width loosely (don't fail on it — sprite artwork
        # rarely matches CAD to the millimetre — just log a warning).
        track_px = math.hypot(knife_r[0] - knife_l[0],
                              knife_r[1] - knife_l[1])
        track_m = track_px / px_per_m
        print(f"[sprite] {path.name}: loaded {img.shape[1]}×{img.shape[0]}, "
              f"{px_per_m:.1f} px/m, sprite forward {sprite_forward_deg:+.1f}°, "
              f"track {track_m:.3f} m (URDF {TRACK_WIDTH_M:.3f} m)")
        return cls(
            image=img,
            px_per_m=px_per_m,
            rear_axle_px=rear_axle,
            caster_px=caster,
            com_px=com,
            sprite_forward_deg=sprite_forward_deg,
        )


def _cluster_pixels(xs, ys, link_dist: int = 3) -> list[tuple[float, float]]:
    """Tiny single-link cluster over (x, y) pixel coords. Returns a list of
    cluster centroids (x_mean, y_mean). Adequate for ≤ a few hundred marker
    pixels."""
    pts = list(zip(xs.tolist(), ys.tolist()))
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][0] - pts[j][0]) <= link_dist and \
               abs(pts[i][1] - pts[j][1]) <= link_dist:
                union(i, j)

    clusters: dict[int, list[tuple[int, int]]] = {}
    for i, p in enumerate(pts):
        clusters.setdefault(find(i), []).append(p)
    return [
        (sum(p[0] for p in members) / len(members),
         sum(p[1] for p in members) / len(members))
        for members in clusters.values()
    ]


# ──────────────────────────────────────────────────────────────────────────────
#                                  GUI
# ──────────────────────────────────────────────────────────────────────────────
def _draw_robot_skeleton(ax, robot: Robot):
    """Stick-figure render of the chaplygin sleigh: rear axle bar, body
    spar from rear axle to front caster, COM dot, knife edges as little
    parallel tick marks, caster as a hollow circle."""
    rx, ry = robot.rear_axle()
    fcx, fcy = robot.front_caster()
    lx, ly = robot.left_knife()
    rkx, rky = robot.right_knife()
    cx, cy = robot.com()

    # Rear axle bar (between knife edges).
    ax.plot([lx, rkx], [ly, rky], color="#f5d142", lw=3, zorder=6)
    # Body spar (rear axle midpoint → front caster).
    ax.plot([rx, fcx], [ry, fcy], color="#cfd8dc", lw=3, zorder=6)
    # Knife edges: small ticks along body x (forward direction) to denote
    # the rolling-edge orientation.
    c, s = math.cos(robot.theta), math.sin(robot.theta)
    for (wx, wy) in ((lx, ly), (rkx, rky)):
        ax.plot([wx - 0.10 * c, wx + 0.10 * c],
                [wy - 0.10 * s, wy + 0.10 * s],
                color="#ff6e40", lw=3, zorder=7)
    # COM marker.
    ax.plot([cx], [cy], marker="o", color="#80ff80",
            markersize=6, zorder=7)
    # Caster: hollow circle.
    ax.add_patch(Circle((fcx, fcy), CASTER_RADIUS_M,
                        edgecolor="#80c4ff", facecolor="none", lw=1.5,
                        zorder=7))


class SimRenderer:
    def __init__(self, sim: Sim, sprite: Optional[RobotSprite] = None):
        self.sim = sim
        self.sprite = sprite
        plt.rcParams["toolbar"] = "None"
        self.fig = plt.figure(figsize=(15.0, 8.5),
                              facecolor="#0a0e0a")
        self.fig.canvas.manager.set_window_title(
            "BEHAVIOR TREE Sim — Chaplygin maze")
        gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1],
                      wspace=0.12, hspace=0.14,
                      left=0.04, right=0.985, top=0.965, bottom=0.04)
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_mini = self.fig.add_subplot(gs[0, 1])
        self.ax_stats = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_main, self.ax_mini):
            ax.set_facecolor("#0d1f12")
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_color("#345434")
        # Tick labels are expensive to lay out and rasterise every frame
        # (main camera follows the robot, so values change continuously;
        # minimap re-rasterises the same labels per frame). Strip both —
        # the grid stays for visual reference.
        for ax in (self.ax_main, self.ax_mini):
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_main.grid(True, color="#163020", lw=0.5, alpha=0.6)
        self.ax_mini.grid(True, color="#163020", lw=0.5, alpha=0.6)

        self.ax_stats.set_facecolor("#0a0e0a")
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        for spine in self.ax_stats.spines.values():
            spine.set_color("#345434")

        self._build_static()

    def _build_static(self):
        s = self.sim
        m = s.maze
        # ── Minimap shows full maze + discovered cells ──
        self.ax_mini.set_xlim(0, m.size_m)
        self.ax_mini.set_ylim(0, m.size_m)
        # Full wall list as a LineCollection (static).
        wall_lines = [((x0, y0), (x1, y1))
                      for (x0, y0, x1, y1) in m.walls]
        self.ax_mini.add_collection(LineCollection(
            wall_lines, colors="#5fb361", linewidths=1.2, zorder=4))
        # Start + goal markers.
        c = CELL_SIZE_M
        self.ax_mini.plot([c * 0.5], [c * 0.5], marker="s",
                          color="#80c4ff", markersize=8, zorder=5)
        gx, gy = s.goal
        self.ax_mini.plot([gx], [gy], marker="*",
                          color="#ffe066", markersize=14, zorder=5)
        # Imshow layer for discovered cells (updated dynamically).
        self._mini_img = self.ax_mini.imshow(
            np.zeros((s.grid.n, s.grid.n, 4), dtype=np.float32),
            origin="lower",
            extent=[0, m.size_m, 0, m.size_m],
            zorder=2, interpolation="nearest")

        # ── Main camera prep ──
        self._main_wall_lc = LineCollection(
            wall_lines, colors="#7ab07a", linewidths=2.2, zorder=4)
        self.ax_main.add_collection(self._main_wall_lc)
        # Single combined discovery + corridor overlay (corridor as faint
        # blue underlay, discovered cells composited on top). Two imshows
        # were a perf hit; one RGBA blend in numpy is cheaper than two draws.
        self._main_disc_img = self.ax_main.imshow(
            np.zeros((s.grid.n, s.grid.n, 4), dtype=np.float32),
            origin="lower",
            extent=[0, m.size_m, 0, m.size_m],
            zorder=1, interpolation="nearest")

        # ── Dynamic artists (updated each frame) ──
        # NAV2 Dijkstra plan to the real goal — runs straight at the goal
        # through unknown space, only routing around discovered red cells.
        self._path_line, = self.ax_main.plot(
            [], [], color="#80ff80", lw=2.0, alpha=0.95, zorder=5,
            label="NAV2 Dijkstra")
        # GOAL_BEND intermediate path — dashed orange so it's obviously
        # transient and separate from the NAV2 plan.
        self._bend_path_line, = self.ax_main.plot(
            [], [], color="#ff9f40", lw=1.8, alpha=0.95, ls="--", zorder=5,
            label="GOAL_BEND intermediate")
        self._trail_line, = self.ax_main.plot(
            [], [], color="#34c2eb", lw=1.0, alpha=0.55, zorder=5)
        self._breadcrumb_scatter = self.ax_main.scatter(
            [], [], s=12, c="#ff8b34", marker="o", zorder=6,
            edgecolors="none")
        self._sensor_lines = LineCollection(
            [], colors="#ffe066", linewidths=0.45, alpha=0.4, zorder=3)
        self.ax_main.add_collection(self._sensor_lines)
        # Persistent sensor-cone wedge. Live updates use set_center /
        # set_theta1 / set_theta2 each frame — far cheaper than the
        # add_patch/remove dance the old code did.
        self._sensor_wedge = Wedge(
            (0, 0), SENSOR_RANGE_M, -90, 90,
            facecolor=(1.0, 0.92, 0.4, 0.05),
            edgecolor=(1.0, 0.92, 0.4, 0.18),
            lw=0.6, zorder=2)
        self.ax_main.add_patch(self._sensor_wedge)
        self._goal_marker, = self.ax_main.plot(
            [], [], marker="*", color="#ffe066", markersize=18, zorder=8)
        self._bent_goal_marker, = self.ax_main.plot(
            [], [], marker="X", color="#ff80c0", markersize=11, zorder=8)
        # Mini-map dynamic.
        self._mini_robot_dot, = self.ax_mini.plot(
            [], [], marker="o", color="#ff8b34", markersize=8, zorder=6)
        self._mini_robot_arrow = self.ax_mini.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->",
                            color="#ff8b34", lw=1.5), zorder=6)
        # Stats text.
        self._stats_text = self.ax_stats.text(
            0.02, 0.98, "", color="#c8d8c8", fontsize=10,
            family="monospace", va="top", ha="left",
            transform=self.ax_stats.transAxes)
        self._state_banner = self.ax_stats.text(
            0.5, 0.94, "", color="#ffe066", fontsize=13,
            family="monospace", fontweight="bold",
            va="top", ha="center",
            transform=self.ax_stats.transAxes)

        # FPS counter — top-left of the main camera, axes coords so it
        # doesn't drift when the camera follows.
        self._fps_text = self.ax_main.text(
            0.012, 0.985, "", color="#a8ffa8", fontsize=10,
            family="monospace", fontweight="bold",
            va="top", ha="left",
            transform=self.ax_main.transAxes, zorder=10,
            bbox=dict(facecolor=(0, 0, 0, 0.55), edgecolor="none",
                      pad=3.0))
        # Rolling FPS over the last N frame intervals.
        self._frame_times: list[float] = []
        self._fps_window = 30
        # Stats panel + FPS text only need to refresh at ~10 Hz; rendering
        # the 20-line monospace block + tick labels each frame was the
        # second-largest per-frame cost in the profile. Throttle here.
        self._last_text_refresh = 0.0
        self._text_refresh_period_s = 0.10

        # Persistent sprite imshow (created once if sprite is loaded; only
        # its transform is touched each frame). Faster than recreating an
        # AxesImage per frame.
        self._sprite_im = None
        if self.sprite is not None:
            sp = self.sprite
            h_px, w_px = sp.image.shape[:2]
            w_m = w_px / sp.px_per_m
            h_m = h_px / sp.px_per_m
            self._sprite_im = self.ax_main.imshow(
                sp.image, extent=[0, w_m, -h_m, 0],
                origin="upper", interpolation="bilinear", zorder=6)
            self._sprite_rax_m = (sp.rear_axle_px[0] / sp.px_per_m,
                                  -sp.rear_axle_px[1] / sp.px_per_m)
            self._sprite_native_angle_deg = -sp.sprite_forward_deg

    # Color mapping for the discovery imshow layers.
    @staticmethod
    def _discovery_rgba(cells: np.ndarray) -> np.ndarray:
        h, w = cells.shape
        out = np.zeros((w, h, 4), dtype=np.float32)
        # Note transpose: imshow expects (rows, cols) = (y, x); our grid is
        # indexed (i, j) = (x, y).
        c = cells.T
        # Unknown: fully transparent.
        # Free: dim green tint.
        free = (c == FREE_KNOWN)
        out[free] = (0.15, 0.55, 0.25, 0.20)
        wall = (c == WALL_KNOWN)
        out[wall] = (0.9, 0.35, 0.20, 0.55)
        return out

    def update(self, _frame=None):
        s = self.sim
        # Sample wall-clock right at frame start so the rolling FPS measures
        # the actual paint interval (animation timer + physics + draw).
        now = time.perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) > self._fps_window + 1:
            self._frame_times.pop(0)
        for _ in range(SUBSTEPS_PER_FRAME):
            s.step(PHYS_DT)

        # Camera follow.
        cam_half = 8.0
        self.ax_main.set_xlim(s.robot.x - cam_half, s.robot.x + cam_half)
        self.ax_main.set_ylim(s.robot.y - cam_half, s.robot.y + cam_half)

        # Clear last-frame robot overlay (sprite or stick figure).
        if not hasattr(self, "_skel_artists"):
            self._skel_artists = []
        for a in self._skel_artists:
            a.remove()
        self._skel_artists = []
        rb = s.robot
        rx, ry = rb.rear_axle()
        fcx, fcy = rb.front_caster()
        lx, ly = rb.left_knife()
        rkx, rky = rb.right_knife()
        cx, cy = rb.com()
        c_th, s_th = math.cos(rb.theta), math.sin(rb.theta)

        if self._sprite_im is not None:
            # Persistent AxesImage — just refresh the transform every frame.
            rax_x, rax_y = self._sprite_rax_m
            rot_deg = math.degrees(rb.theta) - self._sprite_native_angle_deg
            tform = (
                mtransforms.Affine2D()
                .translate(-rax_x, -rax_y)
                .rotate_deg(rot_deg)
                .translate(rx, ry)
                + self.ax_main.transData
            )
            self._sprite_im.set_transform(tform)
        else:
            # Fallback: stick figure (rear axle bar + body spar + knife edges
            # + COM dot + caster ring).
            l1, = self.ax_main.plot(
                [lx, rkx], [ly, rky], color="#f5d142", lw=3, zorder=6)
            l2, = self.ax_main.plot(
                [rx, fcx], [ry, fcy], color="#cfd8dc", lw=3, zorder=6)
            self._skel_artists.extend([l1, l2])
            for (wx, wy) in ((lx, ly), (rkx, rky)):
                ln, = self.ax_main.plot(
                    [wx - 0.10 * c_th, wx + 0.10 * c_th],
                    [wy - 0.10 * s_th, wy + 0.10 * s_th],
                    color="#ff6e40", lw=3, zorder=7)
                self._skel_artists.append(ln)
            com_dot, = self.ax_main.plot(
                [cx], [cy], marker="o", color="#80ff80",
                markersize=6, zorder=7)
            self._skel_artists.append(com_dot)
            caster_patch = Circle(
                (fcx, fcy), CASTER_RADIUS_M,
                edgecolor="#80c4ff", facecolor="none", lw=1.5, zorder=7)
            self.ax_main.add_patch(caster_patch)
            self._skel_artists.append(caster_patch)

        # Sensor wedge silhouette behind the rays — update the persistent
        # patch in-place instead of add_patch/remove every frame.
        self._sensor_wedge.set_center((fcx, fcy))
        self._sensor_wedge.set_theta1(math.degrees(rb.theta - SENSOR_FOV_RAD / 2))
        self._sensor_wedge.set_theta2(math.degrees(rb.theta + SENSOR_FOV_RAD / 2))

        # Sensor rays (from last sense, may be one tick old).
        ray_segs = []
        if hasattr(s, "_last_sensor_origin"):
            ox, oy = s._last_sensor_origin
            for (a, t_hit) in s._last_sensor_hits:
                ray_segs.append((
                    (ox, oy),
                    (ox + math.cos(a) * t_hit,
                     oy + math.sin(a) * t_hit)))
        self._sensor_lines.set_segments(ray_segs)

        # Path + trail + breadcrumbs + goals.
        if s.path_world:
            xs, ys = zip(*s.path_world)
            self._path_line.set_data(xs, ys)
        else:
            self._path_line.set_data([], [])
        if s.bend_path and s.bt.state.name == "GOAL_BEND":
            bxs, bys = zip(*s.bend_path)
            self._bend_path_line.set_data(bxs, bys)
        else:
            self._bend_path_line.set_data([], [])
        if s.trail:
            xs, ys = zip(*s.trail)
            self._trail_line.set_data(xs, ys)
        if s.breadcrumbs:
            bx, by = zip(*s.breadcrumbs)
            self._breadcrumb_scatter.set_offsets(np.column_stack([bx, by]))
        else:
            self._breadcrumb_scatter.set_offsets(np.zeros((0, 2)))
        self._goal_marker.set_data([s.goal[0]], [s.goal[1]])
        if s.bt.state.bent_goal:
            bx, by = s.bt.state.bent_goal
            self._bent_goal_marker.set_data([bx], [by])
        else:
            self._bent_goal_marker.set_data([], [])

        # Combined discovery + corridor RGBA, rebuilt only on text-tick
        # boundaries (we already early-returned for non-tick frames above).
        rgba = self._discovery_rgba(s.grid.cells)
        corr = s.padder.cumulative.T
        empty = (rgba[..., 3] == 0) & corr
        rgba[empty] = (0.30, 0.55, 0.95, 0.10)
        self._main_disc_img.set_data(rgba)
        rgba_mini = rgba.copy()
        rgba_mini[..., 3] = np.clip(rgba_mini[..., 3] * 2.5, 0, 0.9)
        self._mini_img.set_data(rgba_mini)
        self._mini_robot_dot.set_data([rb.x], [rb.y])
        dx_arrow = 1.6 * math.cos(rb.theta)
        dy_arrow = 1.6 * math.sin(rb.theta)
        self._mini_robot_arrow.xy = (rb.x + dx_arrow, rb.y + dy_arrow)
        self._mini_robot_arrow.set_position((rb.x, rb.y))

        # Stats panel + FPS counter throttled to ~10 Hz; the same numbers
        # appear in 4–6 consecutive frames otherwise. We also use the same
        # gate to skip the (relatively expensive) discovery + corridor
        # RGBA blend — it only changes at BT-tick rate (10 Hz) anyway.
        if (now - self._last_text_refresh) < self._text_refresh_period_s:
            return ()
        self._last_text_refresh = now

        bt = s.bt.state
        co = s.last_controller
        disc = s.grid.discovered_fraction() * 100
        v_mps = s.robot.u
        w_dps = math.degrees(s.robot.omega)
        dist_goal = math.hypot(s.goal[0] - rb.x, s.goal[1] - rb.y)
        bearing_to_goal = math.degrees(
            math.atan2(s.goal[1] - rb.y, s.goal[0] - rb.x)) - math.degrees(rb.theta)
        bearing_to_goal = ((bearing_to_goal + 180) % 360) - 180
        time_in_state = s.t - bt.entered_at
        plan_age = s.t - s.last_replan_time
        progress_age = s.t - bt.last_progress_time

        self._state_banner.set_text(f"BT: {bt.name}")
        lines = [
            f"  t in state:    {time_in_state:6.2f} s",
            f"  plan age:      {plan_age:6.2f} s   ({s.last_plan_status})",
            f"  progress age:  {progress_age:6.2f} s  (stall@{PROGRESS_STALL_SEC:.0f}s)",
            "",
            f"  v (body fwd):  {v_mps:+6.3f} m/s",
            f"  omega:         {w_dps:+7.2f} °/s",
            f"  F_left:        {co.F_left:+7.1f} N",
            f"  F_right:       {co.F_right:+7.1f} N",
            f"  controller   : v* {co.v_des:+5.2f}   ω* {co.omega_des:+5.2f}",
            f"  backwards?   : {'YES' if co.backwards_request else 'no'}",
            "",
            f"  dist to goal:  {dist_goal:6.2f} m",
            f"  bearing:       {bearing_to_goal:+6.1f}°",
            f"  discovered:    {disc:5.1f} %",
            f"  breadcrumbs:   {len(s.breadcrumbs):3d}",
            "",
            "  ─ BT firings ─────────────────",
            f"  planner_fail : {bt.n_planner_failures:3d}",
            f"  goal_bend    : {bt.n_goal_bends:3d}",
            f"  bcrumb_rev   : {bt.n_breadcrumb_reverses:3d}",
            f"  backup_recov : {bt.n_backup_fires:3d}",
            f"  gradient_esc : {bt.n_gradient_fires:3d}",
            f"  clear_around : {bt.n_clear_around_robot:3d}",
        ]
        if s.done:
            lines.append("")
            lines.append(f"  *** GOAL REACHED @ t={s.done_time:.2f} s ***")
        self._stats_text.set_text("\n".join(lines))

        # FPS counter (top-left of the tracking camera).
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            n = len(self._frame_times) - 1
            fps = n / elapsed if elapsed > 1e-6 else 0.0
            colour = "#a8ffa8" if fps >= RENDER_FPS * 0.9 else \
                     "#ffe066" if fps >= RENDER_FPS * 0.6 else "#ff8080"
            self._fps_text.set_color(colour)
            self._fps_text.set_text(
                f"FPS {fps:5.1f} / target {RENDER_FPS}")
        return ()


# ──────────────────────────────────────────────────────────────────────────────
#               Pygame renderer (default — matplotlib is fallback)
# ──────────────────────────────────────────────────────────────────────────────
# Pygame is `pygame-ce` (community edition) in requirements.txt — same
# import name, API-compatible, builds against Python 3.13/3.14 which the
# legacy `pygame` doesn't yet.
# ──────────────────────────────────────────────────────────────────────────────
#                      Behaviour-tree diagram model
# ──────────────────────────────────────────────────────────────────────────────
# Mirror of the high-level BT structure the user described:
#
#                                     [ Root ]
#                  ┌────────────┬─────────────┴─────────────┬────────────┐
#       [Controller Server]  [Cond: goal in front       [Cond: goal    [Cond: robot
#       (fire goal, repath,   AND path commands         behind →        in costmap →
#        wait 0.5 s, drive)   backward → BCRUMB         GOAL_BEND]      GRADIENT_ESCAPE]
#                             CONSUMPTION]
#
# Controller Server is the "default" branch — it's also where the
# various WAIT / CLEAR* / BACKUP sub-states live (all part of "repath +
# wait" cycles before handing back to NORMAL_FOLLOWING). Each of the
# three CONDITIONAL branches has exactly one BT state under it.
BT_DIAGRAM_NODES: list[tuple[str, str, int, float, Optional[str]]] = [
    ("root",        "Root",                       0, 0.50, None),
    # Level 1 — four top-level branches.
    ("ctrl",        "Controller Server",          1, 0.20, "root"),
    ("bcrum",       "Cond: goal front + path back→Breadcrumb", 1, 0.50, "root"),
    ("bend",        "Cond: goal behind→Goal Bend",          1, 0.72, "root"),
    ("grad",        "Cond: in costmap→Gradient Escape",     1, 0.90, "root"),
    # Level 2 — Controller Server sub-states (fire-goal + repath +
    # local clear + wait + drive). GLOBAL costmap clears are NOT an
    # option per user spec — only the local around-robot clear is here.
    ("normal",      "NORMAL_FOLLOW",              2, 0.06, "ctrl"),
    ("wait_trans",  "WAIT_TRANS",                 2, 0.14, "ctrl"),
    ("wait_repl",   "WAIT_REPLAN",                2, 0.22, "ctrl"),
    ("clr_arnd",    "CLR_AROUND",                 2, 0.30, "ctrl"),
    ("backup",      "BACKUP",                     2, 0.38, "ctrl"),
    # Level 2 — single conditional-action leaves.
    ("bcrev_leaf",  "BCRUMB_REV",                 2, 0.50, "bcrum"),
    ("bend_leaf",   "GOAL_BEND",                  2, 0.72, "bend"),
    ("grad_leaf",   "GRAD_ESCAPE",                2, 0.90, "grad"),
]

# Map BT state name → leaf node id in the diagram.
BT_STATE_TO_DIAGRAM_LEAF = {
    "NORMAL_FOLLOWING":                    "normal",
    "WAIT_TRANSIENT_RECOVERY":             "wait_trans",
    "FORWARD_BLOCKED_WAIT_FOR_REPLAN":     "wait_repl",
    "CLEAR_AROUND_ROBOT":                  "clr_arnd",
    "BACKUP_RECOVERY":                     "backup",
    "FORWARD_BLOCKED_BREADCRUMB_REVERSE":  "bcrev_leaf",
    "GOAL_BEND":                           "bend_leaf",
    "GRADIENT_ESCAPE":                     "grad_leaf",
}


class PygameRenderer:
    """Pygame port of `SimRenderer`. Same visual content (tracking camera,
    discovered overlay, NAV2 + bend paths, sensor wedge + rays, sprite,
    minimap, stats panel, FPS counter) at ~60 FPS — matplotlib couldn't
    hit it for this scene.

    The render loop is a normal pygame event loop; physics is stepped at
    the same `SUBSTEPS_PER_FRAME` rate as the matplotlib path, gated by
    `clock.tick(RENDER_FPS)` so wall time matches sim time.
    """

    # ── colours ─────────────────────────────────────────────────────
    C_BG          = (10, 14, 10)
    C_PANEL       = (13, 31, 18)
    C_PANEL_DARK  = (10, 14, 10)
    C_BORDER      = (52, 84, 52)
    # Wall voxel colours. WALL_PRESENT (undiscovered, drawn from
    # maze.wall_voxel_mask) reads as dim slate so the maze structure is
    # visible. WALL_DISCOVERED (the sensor has hit this cell) reads as
    # bright red. Inflation halo is translucent amber — a warning zone
    # that fires GRADIENT_ESCAPE if the robot enters it.
    C_WALL_PRESENT_FILL    = (90, 110, 120, 180)
    C_WALL_DISCOVERED_FILL = (230, 70, 50, 230)
    C_INFLATION_FILL       = (255, 200, 60, 60)
    C_FREE_FILL   = (38, 140, 64, 50)
    C_CORR_FILL   = (76, 140, 240, 25)
    C_PATH        = (128, 255, 128)
    C_BEND        = (255, 159, 64)
    C_TRAIL       = (52, 194, 235)
    C_BCRUMB      = (255, 139, 52)
    C_GOAL        = (255, 224, 102)
    C_START       = (128, 196, 255)
    C_BENT_GOAL   = (255, 128, 192)
    C_SENSOR      = (255, 224, 102, 90)
    C_WEDGE_FACE  = (255, 235, 102, 14)
    C_WEDGE_EDGE  = (255, 235, 102, 45)
    C_FPS_GOOD    = (168, 255, 168)
    C_FPS_OK      = (255, 224, 102)
    C_FPS_BAD     = (255, 128, 128)
    C_STATS_TEXT  = (200, 216, 200)
    C_STATE_BANN  = (255, 224, 102)
    # BT diagram colours
    C_BT_NODE_BG          = (28, 42, 30)
    C_BT_NODE_EDGE        = (60, 90, 60)
    C_BT_NODE_TEXT        = (180, 195, 180)
    C_BT_ACTIVE_BG        = (40, 130, 60)
    C_BT_ACTIVE_EDGE      = (130, 230, 150)
    C_BT_ACTIVE_TEXT      = (240, 255, 240)
    C_BT_LINE             = (60, 90, 60)
    C_BT_LINE_ACTIVE      = (130, 230, 150)

    def __init__(self, sim: "Sim", sprite: Optional["RobotSprite"] = None,
                 window_size: tuple[int, int] = (1500, 850),
                 cam_half_w_m: float = 8.0):
        import pygame
        self.pygame = pygame
        self.sim = sim
        self.sprite = sprite
        self.cam_half_w = cam_half_w_m
        pygame.init()
        pygame.display.set_caption("BEHAVIOR TREE Sim — Chaplygin maze")
        # vsync=0 lets `clock.tick(RENDER_FPS)` be the sole frame-rate cap.
        # macOS SDL otherwise inserts a ~8 ms vblank wait inside
        # `display.flip()` even when the GPU finished early — that's the
        # delta between our 17 ms offscreen draw and the 25 ms (40 FPS)
        # interactive measurement.
        try:
            self.screen = pygame.display.set_mode(window_size, vsync=0)
        except TypeError:
            # Very old pygame builds lack the vsync kwarg.
            self.screen = pygame.display.set_mode(window_size)
        self.W, self.H = window_size

        # Layout
        right_w = 380
        bt_h = 150   # bottom strip for the behaviour-tree diagram
        top_h = self.H - bt_h
        self.main_rect  = pygame.Rect(0, 0, self.W - right_w, top_h)
        self.mini_rect  = pygame.Rect(self.W - right_w, 0,
                                      right_w, top_h // 2)
        self.stats_rect = pygame.Rect(self.W - right_w, top_h // 2,
                                      right_w, top_h - top_h // 2)
        self.bt_rect    = pygame.Rect(0, top_h, self.W, bt_h)
        # World→main scale: fit 2 * cam_half_w metres across the main rect.
        self.main_scale = self.main_rect.w / (2 * self.cam_half_w)
        # World→mini scale: fit the entire maze into the minimap with 4 px
        # margin on each side.
        self.mini_pad_px = 8
        self.mini_scale = (
            min(self.mini_rect.w, self.mini_rect.h) - 2 * self.mini_pad_px
        ) / sim.maze.size_m

        # Fonts.
        # pygame.font.SysFont accepts a comma-list — picks the first one
        # the OS has.
        self.font_stats  = pygame.font.SysFont(
            "Consolas,Menlo,Courier New,monospace", 11)
        self.font_banner = pygame.font.SysFont(
            "Consolas,Menlo,Courier New,monospace", 15, bold=True)
        self.font_fps    = pygame.font.SysFont(
            "Consolas,Menlo,Courier New,monospace", 14, bold=True)
        self.font_bt     = pygame.font.SysFont(
            "Consolas,Menlo,Courier New,monospace", 11, bold=True)

        # Pre-compute BT diagram node screen positions inside bt_rect.
        # Each entry: id -> {label, level, parent, cx, cy, rect, surf,
        # surf_active}. The label surfs are pre-rendered (text never
        # changes) and only the highlight state differs per frame.
        self._bt_nodes: dict[str, dict] = {}
        row_h = self.bt_rect.h // 4   # 4 vertical rows: top + 3 levels
        for nid, label, level, x_frac, parent in BT_DIAGRAM_NODES:
            cx = self.bt_rect.left + int(x_frac * self.bt_rect.w)
            cy = self.bt_rect.top + row_h // 2 + level * row_h
            label_surf = self.font_bt.render(
                label, True, self.C_BT_NODE_TEXT)
            label_active = self.font_bt.render(
                label, True, self.C_BT_ACTIVE_TEXT)
            pad_x, pad_y = 8, 4
            box = pygame.Rect(0, 0,
                              label_surf.get_width() + 2 * pad_x,
                              label_surf.get_height() + 2 * pad_y)
            box.center = (cx, cy)
            self._bt_nodes[nid] = {
                "label": label, "level": level, "parent": parent,
                "cx": cx, "cy": cy, "rect": box,
                "surf": label_surf, "surf_active": label_active,
            }

        # Pre-render the static parts of the minimap (walls, start, goal).
        self._mini_static = pygame.Surface(
            (self.mini_rect.w, self.mini_rect.h), pygame.SRCALPHA)
        self._render_mini_static()

        # Sprite handling: pad the source so the rear-axle marker sits at
        # the exact pixel centre, then scale to world units. After that,
        # rotating the surface around its centre keeps the rear-axle pivot
        # at the surface centre — no offset math at blit time.
        self._sprite_surf_padded = None
        if sprite is not None:
            self._sprite_surf_padded = self._build_padded_sprite(sprite)

        # Persistent dynamic-overlay surfaces — re-fill with transparent
        # each frame instead of re-allocating 1+ MB of SRCALPHA storage.
        # Same pattern for the per-frame ray + wedge layers below.
        self._main_overlay = pygame.Surface(
            (self.main_rect.w, self.main_rect.h), pygame.SRCALPHA)
        self._mini_overlay = pygame.Surface(
            (self.mini_rect.w, self.mini_rect.h), pygame.SRCALPHA)
        self._ray_surf = pygame.Surface(
            (self.main_rect.w, self.main_rect.h), pygame.SRCALPHA)
        self._wedge_surf = pygame.Surface(
            (self.main_rect.w, self.main_rect.h), pygame.SRCALPHA)
        # Combined alpha-compose layer — the corridor overlay, wedge, and
        # sensor rays all get drawn onto this single buffer, then blitted
        # to the screen with ONE alpha blit instead of three. Each large
        # SRCALPHA-to-screen blit is ~0.5 ms on macOS, so this is ~1 ms
        # saved per frame.
        self._compose_main = pygame.Surface(
            (self.main_rect.w, self.main_rect.h), pygame.SRCALPHA)
        # Cached stats panel surface — built when stats refresh, blitted
        # once per frame (was 22 individual text blits before).
        self._stats_panel_surf: Optional["pygame.Surface"] = None
        # Scaled discovery+corridor textures cached per viewport; only
        # rebuilt when the underlying cell grid changed (10 Hz). The
        # "underlay" texture has corridor + free cells (rendered BELOW
        # the sensor wedge + rays); the "walls" texture has only the red
        # WALL_KNOWN cells (rendered ABOVE the wedge + rays so the
        # discovered walls always sit on top of the lidar viz).
        self._scaled_underlay_main: Optional["pygame.Surface"] = None
        self._scaled_underlay_mini: Optional["pygame.Surface"] = None
        self._scaled_walls_main: Optional["pygame.Surface"] = None
        self._scaled_walls_mini: Optional["pygame.Surface"] = None

        # FPS rolling window.
        self._frame_times: list[float] = []
        self._fps_window = 30
        # Stats / overlay caches refreshed at 10 Hz only.
        self._last_text_refresh = 0.0
        self._text_refresh_period_s = 0.10
        self._stats_surfs: list = []
        self._banner_surf = None
        self._cached_overlay_main: Optional["pygame.Surface"] = None
        self._cached_overlay_mini: Optional["pygame.Surface"] = None

        self.clock = pygame.time.Clock()
        self.done = False

    # ─── Coordinate transforms ───
    def _w2m(self, x: float, y: float) -> tuple[int, int]:
        """World → main-camera screen coords (with y-flip)."""
        rb = self.sim.robot
        sx = self.main_rect.centerx + (x - rb.x) * self.main_scale
        sy = self.main_rect.centery - (y - rb.y) * self.main_scale
        return int(sx), int(sy)

    def _w2mini(self, x: float, y: float) -> tuple[int, int]:
        """World → minimap screen coords (with y-flip). Returned in
        GLOBAL screen coords (not mini-local)."""
        ox = self.mini_rect.left + self.mini_pad_px
        oy = self.mini_rect.bottom - self.mini_pad_px
        return (int(ox + x * self.mini_scale),
                int(oy - y * self.mini_scale))

    def _w2mini_local(self, x: float, y: float) -> tuple[int, int]:
        """World → mini-surface-LOCAL coords (used during static prerender)."""
        ox = self.mini_pad_px
        oy = self.mini_rect.h - self.mini_pad_px
        return (int(ox + x * self.mini_scale),
                int(oy - y * self.mini_scale))

    # ─── Static minimap prerender ───
    def _render_mini_static(self):
        """Only the start marker goes on the static layer. Mission
        waypoints / current goal are painted per-frame because the
        active waypoint advances as the robot reaches each one."""
        pg = self.pygame
        s = self.sim
        sx, sy = s.maze.default_start_xy
        sp = self._w2mini_local(sx, sy)
        pg.draw.rect(self._mini_static, self.C_START,
                     (sp[0] - 5, sp[1] - 5, 10, 10))

    # ─── Sprite helpers ───
    def _build_padded_sprite(self, sprite: "RobotSprite") -> "pygame.Surface":
        """Pad the sprite so the rear-axle marker sits exactly at the
        padded image's centre, then scale to the main camera's px/m.

        Centred padding means `pygame.transform.rotate(surf, angle)` keeps
        the rear-axle pivot at the surface centre, so we can blit with
        `rect = rotated.get_rect(center=rb_screen)` — no offset math.
        """
        pg = self.pygame
        h_px, w_px = sprite.image.shape[:2]
        rax_x, rax_y = sprite.rear_axle_px
        # Distance from rax to each edge:
        dl = rax_x
        dr = w_px - rax_x
        dt = rax_y
        db = h_px - rax_y
        half = int(math.ceil(max(dl, dr, dt, db)))
        pad_l = int(round(half - dl))
        pad_t = int(round(half - dt))
        padded = np.zeros((2 * half, 2 * half, 4), dtype=np.uint8)
        padded[pad_t:pad_t + h_px, pad_l:pad_l + w_px] = sprite.image
        # pygame.image.frombuffer wants (width, height) and the bytes laid
        # out in row-major RGBA (which numpy with shape (H, W, 4) and
        # C-contiguous gives us automatically).
        surf = pg.image.frombuffer(
            padded.tobytes(), (2 * half, 2 * half), "RGBA")
        # Scale into world units (px on screen / px in sprite).
        scale_factor = self.main_scale / sprite.px_per_m
        new_size = max(8, int(round(2 * half * scale_factor)))
        return pg.transform.smoothscale(
            surf, (new_size, new_size)).convert_alpha()

    # ─── Frame loop ───
    def run(self):
        pg = self.pygame
        running = True
        last_t = time.perf_counter()
        # Maximum sim seconds to advance in a single frame — if the
        # window stalls (e.g. dragged, debugger pause), don't try to
        # catch up forever.
        MAX_DT_S = 0.25
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key in (pg.K_ESCAPE, pg.K_q):
                        running = False
            now = time.perf_counter()
            dt_s = min(MAX_DT_S, now - last_t)
            # Advance the sim by exactly `dt_s` of wall time, in PHYS_DT
            # chunks. Real-time is preserved regardless of render FPS:
            # 30 FPS → 8 substeps, 60 FPS → 4 substeps, etc.
            substeps = max(1, int(dt_s / PHYS_DT))
            for _ in range(substeps):
                self.sim.step(PHYS_DT)
            self._frame_times.append(now)
            if len(self._frame_times) > self._fps_window + 1:
                self._frame_times.pop(0)
            self._draw_frame(now)
            pg.display.flip()
            self.clock.tick(RENDER_FPS)
            last_t = now
        pg.quit()

    # ─── Drawing ───
    def _draw_frame(self, now: float):
        pg = self.pygame
        s = self.sim
        rb = s.robot
        # Background.
        self.screen.fill(self.C_BG)
        pg.draw.rect(self.screen, self.C_PANEL, self.main_rect)
        pg.draw.rect(self.screen, self.C_PANEL_DARK, self.stats_rect)

        # ── Main camera ──
        # All translucent layers (corridor underlay, sensor wedge, sensor
        # rays, costmap = inflation + walls) get composed onto a single
        # alpha buffer first, then ONE alpha blit to the screen — far
        # cheaper than 2+ separate SRCALPHA blits to the framebuffer.
        # Layer order inside the buffer (bottom → top):
        #   1. corridor + free underlay
        #   2. sensor wedge silhouette
        #   3. sensor rays
        #   4. costmap (inflation halo + wall voxels) — over the lidar
        #      viz per user spec
        self._compose_overlays(refresh=(now - self._last_text_refresh
                                        >= self._text_refresh_period_s))
        self._compose_main.fill((0, 0, 0, 0))
        self._compose_main.blit(self._main_overlay, (0, 0))
        self._draw_sensor_wedge_into(self._compose_main)
        if hasattr(s, "_last_sensor_origin") and s._last_sensor_hits:
            ox, oy = s._last_sensor_origin
            o_px = self._w2m(ox, oy)
            ol = (o_px[0] - self.main_rect.left,
                  o_px[1] - self.main_rect.top)
            for (a, t) in s._last_sensor_hits:
                ex, ey = ox + t * math.cos(a), oy + t * math.sin(a)
                p1 = self._w2m(ex, ey)
                pg.draw.aaline(
                    self._compose_main, self.C_SENSOR, ol,
                    (p1[0] - self.main_rect.left,
                     p1[1] - self.main_rect.top))
        # Costmap layer on top of the lidar viz (inflation + walls).
        if self._scaled_walls_main is not None:
            wox, woy = self._underlay_offset_main
            self._compose_main.blit(self._scaled_walls_main, (wox, woy))
        # Single alpha blit to the screen.
        self.screen.blit(self._compose_main, self.main_rect)

        # Trail (cyan, full history). Solid line, 2 px wide.
        if len(s.trail) > 1:
            pts = [self._w2m(x, y) for (x, y) in s.trail]
            pg.draw.lines(self.screen, self.C_TRAIL, False, pts, 2)
        # NAV2 path (green).
        if len(s.path_world) > 1:
            pts = [self._w2m(x, y) for (x, y) in s.path_world]
            pg.draw.lines(self.screen, self.C_PATH, False, pts, 2)
        # Bend path (orange, dashed).
        if s.bend_path and s.bt.state.name == "GOAL_BEND" and len(s.bend_path) > 1:
            pts = [self._w2m(x, y) for (x, y) in s.bend_path]
            for k in range(0, len(pts) - 1, 2):
                pg.draw.line(self.screen, self.C_BEND, pts[k], pts[k + 1], 2)
        # Breadcrumbs.
        for (bx, by) in s.breadcrumbs:
            pg.draw.circle(self.screen, self.C_BCRUMB,
                           self._w2m(bx, by), 4)
        # Mission waypoints on the main camera: visited ones small +
        # dim, the current goal big + bright, future ones medium-size +
        # hollow.
        for k, (wx, wy) in enumerate(s.mission):
            wp = self._w2m(wx, wy)
            if k < s.mission_idx:
                pg.draw.circle(self.screen, (80, 110, 80), wp, 6)
            elif k == s.mission_idx:
                pg.draw.circle(self.screen, self.C_GOAL, wp, 12)
                pg.draw.circle(self.screen, (0, 0, 0), wp, 12, 2)
            else:
                pg.draw.circle(self.screen, self.C_GOAL, wp, 8, 2)
        # Bent goal marker.
        if s.bt.state.bent_goal:
            bgp = self._w2m(*s.bt.state.bent_goal)
            pg.draw.line(self.screen, self.C_BENT_GOAL,
                         (bgp[0] - 7, bgp[1] - 7), (bgp[0] + 7, bgp[1] + 7), 2)
            pg.draw.line(self.screen, self.C_BENT_GOAL,
                         (bgp[0] - 7, bgp[1] + 7), (bgp[0] + 7, bgp[1] - 7), 2)
        # Robot sprite or stick figure.
        if self._sprite_surf_padded is not None:
            self._blit_robot_sprite()
        else:
            self._draw_stick_robot()

        # Clip mask: anything outside the main rect (path lines, trail,
        # rays) should not bleed onto the right column. Easiest fix is to
        # re-paint the right column area on top.
        right_x = self.main_rect.right
        pg.draw.rect(self.screen, self.C_BG,
                     (right_x, 0, self.W - right_x, self.H))

        # ── Minimap ──
        pg.draw.rect(self.screen, self.C_PANEL, self.mini_rect)
        # Composite order: corridor+free underlay, then static maze walls
        # (green lines + start/goal markers), then red discovered-wall
        # cells ON TOP so the costmap squares stay visible.
        self.screen.blit(self._mini_overlay, self.mini_rect)
        self.screen.blit(self._mini_static, self.mini_rect)
        if self._scaled_walls_mini is not None:
            mox, moy = self._underlay_offset_mini
            self.screen.blit(self._scaled_walls_mini,
                             (self.mini_rect.left + mox,
                              self.mini_rect.top + moy))
        # Trail — the actual path the robot has driven (cyan, same colour
        # as the main camera). Drawn under the live NAV2 plan so the
        # historical track + the current plan are both visible.
        if len(s.trail) > 1:
            pts = [self._w2mini(x, y) for (x, y) in s.trail]
            pg.draw.lines(self.screen, self.C_TRAIL, False, pts, 2)
        # NAV2 Dijkstra plan (green).
        if len(s.path_world) > 1:
            pts = [self._w2mini(x, y) for (x, y) in s.path_world]
            pg.draw.lines(self.screen, self.C_PATH, False, pts, 2)
        # GOAL_BEND intermediate path (orange dashed) when active.
        if s.bend_path and s.bt.state.name == "GOAL_BEND" \
                and len(s.bend_path) > 1:
            pts = [self._w2mini(x, y) for (x, y) in s.bend_path]
            for k in range(0, len(pts) - 1, 2):
                pg.draw.line(self.screen, self.C_BEND, pts[k], pts[k + 1], 2)
        # Mission waypoints on the minimap.
        for k, (wx, wy) in enumerate(s.mission):
            wp = self._w2mini(wx, wy)
            if k < s.mission_idx:
                pg.draw.circle(self.screen, (80, 110, 80), wp, 4)
            elif k == s.mission_idx:
                pg.draw.circle(self.screen, self.C_GOAL, wp, 7)
                pg.draw.circle(self.screen, (0, 0, 0), wp, 7, 1)
            else:
                pg.draw.circle(self.screen, self.C_GOAL, wp, 5, 2)
        # Robot dot + heading arrow.
        rpos = self._w2mini(rb.x, rb.y)
        pg.draw.circle(self.screen, self.C_BCRUMB, rpos, 5)
        ax = rb.x + 2.0 * math.cos(rb.theta)
        ay = rb.y + 2.0 * math.sin(rb.theta)
        apos = self._w2mini(ax, ay)
        pg.draw.line(self.screen, self.C_BCRUMB, rpos, apos, 2)

        # ── Stats panel ──
        self._draw_stats_panel(now)
        # ── BT diagram strip ──
        self._draw_bt_diagram()

        # Borders.
        pg.draw.rect(self.screen, self.C_BORDER, self.main_rect, 1)
        pg.draw.rect(self.screen, self.C_BORDER, self.mini_rect, 1)
        pg.draw.rect(self.screen, self.C_BORDER, self.stats_rect, 1)

        # FPS counter on top of the main camera.
        self._draw_fps_text()

    def _compose_overlays(self, refresh: bool):
        """If `refresh`, rebuild the discovery textures (split into
        underlay and walls) and rescale them once per viewport. Otherwise
        reuse the cached scaled textures — the camera offset is the only
        thing that changes per frame, and re-blitting a cached surface is
        cheap.
        """
        pg = self.pygame
        s = self.sim
        if refresh or self._scaled_underlay_main is None:
            cells = s.grid.cells           # (n, n) uint8
            corr  = s.padder.cumulative    # (n, n) bool
            wall_voxels = s.maze.wall_voxel_mask     # (n, n) bool
            discovered = (cells == WALL_KNOWN)        # (n, n) bool
            inflated = (s._cached_inflation_mask
                        if s._cached_inflation_mask is not None
                        else np.zeros_like(discovered))
            n = cells.shape[0]
            # Underlay (BELOW lidar viz): corridor + free cells ONLY.
            # The lidar wedge + rays read against this. Costmap-style
            # layers (inflation halo + walls) moved to the top layer per
            # user request — the costmap should render OVER the lidar so
            # red/amber cells stay legible behind the yellow rays.
            under = np.zeros((n, n, 4), dtype=np.uint8)
            under[corr.T] = self.C_CORR_FILL
            under[(cells == FREE_KNOWN).T] = self.C_FREE_FILL
            under_surf = pg.transform.flip(
                pg.image.frombuffer(under.tobytes(), (n, n), "RGBA"),
                False, True)
            # Costmap layer (ABOVE lidar viz). Order within the texture:
            # amber inflation halo first, then slate (undiscovered) wall
            # voxels, then red (discovered) wall voxels on top.
            walls = np.zeros((n, n, 4), dtype=np.uint8)
            walls[inflated.T] = self.C_INFLATION_FILL
            present_undisc = wall_voxels & ~discovered
            walls[present_undisc.T] = self.C_WALL_PRESENT_FILL
            walls[discovered.T] = self.C_WALL_DISCOVERED_FILL
            walls_surf = pg.transform.flip(
                pg.image.frombuffer(walls.tobytes(), (n, n), "RGBA"),
                False, True)
            # Pre-scale for each viewport once.
            world_w_main = int(round(s.maze.size_m * self.main_scale))
            mini_w_avail = self._mini_overlay.get_width() - 2 * self.mini_pad_px
            mini_h_avail = self._mini_overlay.get_height() - 2 * self.mini_pad_px
            mini_size = min(mini_w_avail, mini_h_avail)
            # .convert_alpha() rebinds the surface into the display's
            # optimal RGBA pixel layout. Per-pixel alpha blits then take
            # the SDL fast path instead of the generic blit path; saves
            # ~half the per-blit cost on macOS.
            self._scaled_underlay_main = pg.transform.scale(
                under_surf, (world_w_main, world_w_main)).convert_alpha()
            self._scaled_underlay_mini = pg.transform.scale(
                under_surf, (mini_size, mini_size)).convert_alpha()
            self._scaled_walls_main = pg.transform.scale(
                walls_surf, (world_w_main, world_w_main)).convert_alpha()
            self._scaled_walls_mini = pg.transform.scale(
                walls_surf, (mini_size, mini_size)).convert_alpha()

        # Per-frame: clear overlays + blit cached scaled textures at the
        # current camera offset.
        self._main_overlay.fill((0, 0, 0, 0))
        self._mini_overlay.fill((0, 0, 0, 0))
        # Main camera placement: world (cam_x, cam_y) lands at the centre
        # of the main rect; we flipped the surface vertically so its top
        # row corresponds to world_size_m. Compute the screen offset of
        # world (0, 0) accordingly.
        scale = self.main_scale
        cam_x = s.robot.x
        cam_y = s.robot.y
        cx_px = self._main_overlay.get_width() / 2
        cy_px = self._main_overlay.get_height() / 2
        ox = cx_px - cam_x * scale
        oy = cy_px - (s.maze.size_m - cam_y) * scale
        self._main_overlay.blit(self._scaled_underlay_main, (ox, oy))
        # Stash the offset so _draw_frame can place the walls layer at
        # the matching world-coord position (over the lidar viz).
        self._underlay_offset_main = (ox, oy)
        # Minimap placement: anchor to the SAME (pad_px, mini_h - pad_px)
        # corner the static walls use, so the red costmap squares line up
        # with the green maze walls. The old "centred in rect" placement
        # was off by ~22 px vertically because mini_rect.h > mini_rect.w.
        mini_size_px = self._scaled_underlay_mini.get_width()
        ox_m = self.mini_pad_px
        oy_m = self._mini_overlay.get_height() - self.mini_pad_px - mini_size_px
        self._mini_overlay.blit(self._scaled_underlay_mini, (ox_m, oy_m))
        self._underlay_offset_mini = (ox_m, oy_m)

    def _draw_sensor_wedge_into(self, target):
        """Draw the translucent sensor cone polygon directly onto `target`
        (expected to be a main-rect-sized SRCALPHA surface). Avoids an
        extra intermediate surface + alpha blit."""
        pg = self.pygame
        s = self.sim
        rb = s.robot
        fcx, fcy = rb.front_caster()
        centre = self._w2m(fcx, fcy)
        steps = 32
        half = SENSOR_FOV_RAD / 2
        pts = [centre]
        for k in range(steps + 1):
            a = rb.theta - half + (k / steps) * SENSOR_FOV_RAD
            wx = fcx + SENSOR_RANGE_M * math.cos(a)
            wy = fcy + SENSOR_RANGE_M * math.sin(a)
            pts.append(self._w2m(wx, wy))
        local_pts = [(p[0] - self.main_rect.left, p[1] - self.main_rect.top)
                     for p in pts]
        pg.draw.polygon(target, self.C_WEDGE_FACE, local_pts)
        pg.draw.polygon(target, self.C_WEDGE_EDGE, local_pts, 1)

    def _blit_robot_sprite(self):
        pg = self.pygame
        s = self.sim
        rb = s.robot
        # Sprite native forward = +y_pixel (rear→caster vector points down
        # in the pixel frame). pygame.transform.rotate is CCW visually for
        # positive angles, which matches world CCW (since our world →
        # screen transform also flips y). When θ=0, robot forward = world
        # +x = screen right, so we need sprite-down → screen-right, which
        # is a +90° rotation. For arbitrary θ:  angle = θ + 90°.
        angle_deg = math.degrees(rb.theta) + 90
        rotated = pg.transform.rotate(self._sprite_surf_padded, angle_deg)
        rect = rotated.get_rect(center=self._w2m(rb.x, rb.y))
        self.screen.blit(rotated, rect)

    def _draw_stick_robot(self):
        pg = self.pygame
        rb = self.sim.robot
        rax = self._w2m(*rb.rear_axle())
        fc  = self._w2m(*rb.front_caster())
        lk  = self._w2m(*rb.left_knife())
        rk  = self._w2m(*rb.right_knife())
        com = self._w2m(*rb.com())
        pg.draw.line(self.screen, (245, 209, 66), lk, rk, 3)
        pg.draw.line(self.screen, (207, 216, 220), rax, fc, 3)
        pg.draw.circle(self.screen, (128, 196, 255), fc, 5, 2)
        pg.draw.circle(self.screen, (128, 255, 128), com, 4)
        # Tiny knife-edge ticks.
        for (kx, ky) in (rb.left_knife(), rb.right_knife()):
            tx = kx + 0.10 * math.cos(rb.theta)
            ty = ky + 0.10 * math.sin(rb.theta)
            tx2 = kx - 0.10 * math.cos(rb.theta)
            ty2 = ky - 0.10 * math.sin(rb.theta)
            pg.draw.line(self.screen, (255, 110, 64),
                         self._w2m(tx, ty), self._w2m(tx2, ty2), 3)

    def _draw_fps_text(self):
        pg = self.pygame
        if len(self._frame_times) < 2:
            return
        elapsed = self._frame_times[-1] - self._frame_times[0]
        n = len(self._frame_times) - 1
        fps = n / elapsed if elapsed > 1e-6 else 0.0
        if fps >= RENDER_FPS * 0.9:
            color = self.C_FPS_GOOD
        elif fps >= RENDER_FPS * 0.6:
            color = self.C_FPS_OK
        else:
            color = self.C_FPS_BAD
        text = f"FPS {fps:5.1f} / target {RENDER_FPS}"
        surf = self.font_fps.render(text, True, color)
        # Translucent black backdrop.
        pad = 4
        bg = pg.Surface((surf.get_width() + 2 * pad,
                          surf.get_height() + 2 * pad), pg.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        self.screen.blit(bg, (self.main_rect.left + 10,
                               self.main_rect.top + 10))
        self.screen.blit(surf, (self.main_rect.left + 10 + pad,
                                 self.main_rect.top + 10 + pad))

    def _draw_bt_diagram(self):
        """Render the BT diagram in the bottom strip. Highlight the
        path from the root to the currently-active leaf in green."""
        pg = self.pygame
        s = self.sim
        pg.draw.rect(self.screen, self.C_PANEL, self.bt_rect)
        pg.draw.rect(self.screen, self.C_BORDER, self.bt_rect, 1)

        # Active path: leaf → parent → … → root.
        leaf = BT_STATE_TO_DIAGRAM_LEAF.get(s.bt.state.name)
        active: set[str] = set()
        cur = leaf
        while cur is not None:
            active.add(cur)
            cur = self._bt_nodes[cur]["parent"] if cur in self._bt_nodes else None

        # 1. Draw connector lines first so node rects sit on top.
        for nid, node in self._bt_nodes.items():
            pid = node["parent"]
            if pid is None:
                continue
            parent = self._bt_nodes[pid]
            # Bottom-centre of parent → top-centre of child.
            p_btm = (parent["cx"], parent["rect"].bottom)
            c_top = (node["cx"], node["rect"].top)
            line_active = nid in active and pid in active
            colour = self.C_BT_LINE_ACTIVE if line_active else self.C_BT_LINE
            width = 2 if line_active else 1
            pg.draw.line(self.screen, colour, p_btm, c_top, width)

        # 2. Draw boxes + labels.
        for nid, node in self._bt_nodes.items():
            is_active = nid in active
            bg = self.C_BT_ACTIVE_BG if is_active else self.C_BT_NODE_BG
            edge = self.C_BT_ACTIVE_EDGE if is_active else self.C_BT_NODE_EDGE
            surf = node["surf_active"] if is_active else node["surf"]
            pg.draw.rect(self.screen, bg, node["rect"])
            pg.draw.rect(self.screen, edge, node["rect"], 1)
            self.screen.blit(
                surf,
                (node["rect"].x + (node["rect"].w - surf.get_width()) // 2,
                 node["rect"].y + (node["rect"].h - surf.get_height()) // 2))

    def _draw_stats_panel(self, now: float):
        pg = self.pygame
        s = self.sim
        rb = s.robot
        # Refresh text only at 10 Hz — and bake the entire panel into ONE
        # cached surface so the per-frame cost is a single blit instead of
        # 22+ individual line blits.
        if (now - self._last_text_refresh) >= self._text_refresh_period_s \
                or self._stats_panel_surf is None:
            self._last_text_refresh = now
            bt = s.bt.state
            co = s.last_controller
            disc = s.grid.discovered_fraction() * 100
            v_mps = rb.u
            w_dps = math.degrees(rb.omega)
            dist_goal = math.hypot(s.goal[0] - rb.x, s.goal[1] - rb.y)
            bearing = math.degrees(
                math.atan2(s.goal[1] - rb.y, s.goal[0] - rb.x)) \
                - math.degrees(rb.theta)
            bearing = ((bearing + 180) % 360) - 180
            time_in_state = s.t - bt.entered_at
            plan_age = s.t - s.last_replan_time
            progress_age = s.t - bt.last_progress_time
            lines = [
                f"  t in state:    {time_in_state:6.2f} s",
                f"  plan age:      {plan_age:6.2f} s   ({s.last_plan_status})",
                f"  progress age:  {progress_age:6.2f} s  (stall@{PROGRESS_STALL_SEC:.0f}s)",
                "",
                f"  v (body fwd):  {v_mps:+6.3f} m/s",
                f"  omega:         {w_dps:+7.2f} °/s",
                f"  F_left:        {co.F_left:+7.1f} N",
                f"  F_right:       {co.F_right:+7.1f} N",
                f"  controller:    v* {co.v_des:+5.2f}   w* {co.omega_des:+5.2f}",
                f"  backwards?     {'YES' if co.backwards_request else 'no'}",
                "",
                f"  dist to goal:  {dist_goal:6.2f} m",
                f"  bearing:       {bearing:+6.1f} deg",
                f"  discovered:    {disc:5.1f} %",
                f"  breadcrumbs:   {len(s.breadcrumbs):3d}",
                "",
                "  -- BT firings ---------------",
                f"  planner_fail : {bt.n_planner_failures:3d}",
                f"  goal_bend    : {bt.n_goal_bends:3d}",
                f"  bcrumb_rev   : {bt.n_breadcrumb_reverses:3d}",
                f"  backup_recov : {bt.n_backup_fires:3d}",
                f"  gradient_esc : {bt.n_gradient_fires:3d}",
                f"  clear_around : {bt.n_clear_around_robot:3d}",
            ]
            if s.done:
                lines.append("")
                lines.append(f"  *** GOAL REACHED @ t={s.done_time:.2f} s ***")
            banner = self.font_banner.render(
                f"BT: {bt.name}", True, self.C_STATE_BANN)
            line_surfs = [
                self.font_stats.render(line, True, self.C_STATS_TEXT)
                for line in lines
            ]
            # Compose into the panel surface.
            panel = pg.Surface((self.stats_rect.w, self.stats_rect.h),
                                pg.SRCALPHA)
            bx = (self.stats_rect.w - banner.get_width()) // 2
            by = 10
            panel.blit(banner, (bx, by))
            y = by + banner.get_height() + 8
            for surf in line_surfs:
                panel.blit(surf, (8, y))
                y += surf.get_height() + 1
            self._stats_panel_surf = panel
        self.screen.blit(self._stats_panel_surf, self.stats_rect)


# ──────────────────────────────────────────────────────────────────────────────
def _resolve_sprite_path(arg_path: Optional[str]) -> Optional[Path]:
    """Resolve the robot sprite path. Priority:
      1. Explicit --sprite argument (whatever the caller specified).
      2. <sim>/../data/robot_top.png if it exists.
      3. The first *.png (alphabetical) found inside <sim>/../data/.
      4. The fallback default path (will print "no sprite" and use sticks).
    """
    if arg_path:
        return Path(arg_path).expanduser().resolve()
    here = Path(__file__).resolve().parent
    data_dir = (here.parent / "data").resolve()
    default = data_dir / "robot_top.png"
    if default.exists():
        return default
    if data_dir.exists():
        pngs = sorted(data_dir.glob("*.png"))
        if pngs:
            return pngs[0]
    return default


def run_gui(args):
    sim = Sim.build(n_cells=args.maze_cells, seed=args.seed,
                    n_obstacles=args.obstacles, layout=args.layout)
    if args.heading_deg is not None:
        sim.robot.theta = math.radians(args.heading_deg)
    # Prime the sensor cache so the renderer can read it on frame 0.
    sim._last_sensor_origin = sim.robot.front_caster()
    sim._last_sensor_hits = []

    sprite = None
    if not args.no_sprite:
        sprite_path = _resolve_sprite_path(args.sprite)
        sprite = RobotSprite.try_load(sprite_path)
        if sprite is None:
            print(f"[sprite] no sprite at {sprite_path} — using stick figure.")

    if args.renderer == "matplotlib":
        renderer = SimRenderer(sim, sprite=sprite)
        _anim = FuncAnimation(
            renderer.fig, renderer.update,
            interval=int(1000 / RENDER_FPS),
            blit=False, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
    else:
        # Default: pygame at 60 FPS. matplotlib couldn't hit it.
        renderer = PygameRenderer(sim, sprite=sprite)
        try:
            renderer.run()
        except KeyboardInterrupt:
            pass


def run_headless(args):
    sim = Sim.build(n_cells=args.maze_cells, seed=args.seed,
                    n_obstacles=args.obstacles, layout=args.layout)
    if args.heading_deg is not None:
        sim.robot.theta = math.radians(args.heading_deg)
    steps = args.headless_steps
    print(f"[headless] running {steps} ticks ({steps * PHYS_DT:.1f} s sim time)")
    t0 = time.time()
    for i in range(steps):
        sim.step(PHYS_DT)
        if i % 1000 == 0:
            print(f"  t={sim.t:6.2f}s  BT={sim.bt.state.name:<36}  "
                  f"goal_d={math.hypot(sim.goal[0]-sim.robot.x, sim.goal[1]-sim.robot.y):5.2f}m  "
                  f"disc={sim.grid.discovered_fraction()*100:4.1f}%")
        if sim.done:
            print(f"  GOAL @ t={sim.done_time:.2f}s "
                  f"(wall {time.time()-t0:.2f}s)")
            break
    else:
        print(f"[headless] ran out of steps without reaching goal "
              f"(t={sim.t:.1f}s, dist={math.hypot(sim.goal[0]-sim.robot.x, sim.goal[1]-sim.robot.y):.2f}m)")


def run_bake(args):
    """Render the live pygame sim offscreen, writing each frame to an
    MP4 via imageio-ffmpeg. Stops on `bake_secs` of sim time or when
    the robot completes the mission, whichever comes first.
    """
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    out_path = Path(args.bake_mp4).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bake] target: {out_path}")
    sim = Sim.build(n_cells=args.maze_cells, seed=args.seed,
                    n_obstacles=args.obstacles, layout=args.layout)
    if args.heading_deg is not None:
        sim.robot.theta = math.radians(args.heading_deg)
    sim._last_sensor_origin = sim.robot.front_caster()
    sim._last_sensor_hits = []
    sprite = None
    if not args.no_sprite:
        sprite = RobotSprite.try_load(_resolve_sprite_path(args.sprite))
    renderer = PygameRenderer(sim, sprite=sprite)
    pg = renderer.pygame
    fps = max(1, int(args.bake_fps))
    sub_per_frame = max(1, int(round((1.0 / fps) / PHYS_DT)))
    max_frames = int(args.bake_secs * fps)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=8, macro_block_size=1, format="FFMPEG")
    t_wall0 = time.time()
    frames_written = 0
    try:
        for f in range(max_frames):
            for _ in range(sub_per_frame):
                sim.step(PHYS_DT)
            renderer._draw_frame(time.perf_counter())
            arr = pg.surfarray.array3d(renderer.screen)
            # pygame surfarray gives (W, H, 3); MP4 wants (H, W, 3).
            arr = np.transpose(arr, (1, 0, 2))
            writer.append_data(arr)
            frames_written += 1
            if f % fps == 0:
                print(f"  bake t={sim.t:6.2f}s  frame {f+1}/{max_frames}  "
                      f"BT={sim.bt.state.name:<35}  "
                      f"wp={sim.mission_idx}/{len(sim.mission)}")
            if sim.done:
                # Hold the final frame for 1 second so the video ends
                # on the goal-reached banner.
                for _ in range(fps):
                    writer.append_data(arr)
                    frames_written += 1
                print(f"  GOAL @ t={sim.done_time:.2f}s")
                break
    finally:
        writer.close()
        pg.quit()
    print(f"[bake] wrote {frames_written} frames "
          f"in {time.time()-t_wall0:.1f}s wall time → {out_path}")


def run_scatter_bake(args):
    """Multi-robot bake — all N robots run on the SAME track layout
    (same maze seed → same waypoints) but get independent sensor RNGs
    so their explorations differ. Rendered in a fixed overhead view
    with each robot a different colour, written to MP4.
    """
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    n = max(2, args.scatter)
    out_path = Path(args.bake_mp4).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[scatter-bake] {n} robots, layout={args.layout}, → {out_path}")

    # Build N sims sharing the same maze layout (same seed). Override
    # each sim's sensor RNG so detection noise differs per robot.
    # Also scatter the spawn positions across the TOP half of the world
    # (track: outer lane between ~60°–120°; maze: top third of the grid)
    # so the robots fan out instead of all stacking on the default start.
    sims: list[Sim] = []
    rng_spawn = random.Random(args.seed * 31337 + 5)
    for i in range(n):
        s = Sim.build(n_cells=args.maze_cells, seed=args.seed,
                      n_obstacles=args.obstacles, layout=args.layout)
        s._sensor_rng = random.Random(
            args.seed * 999983 + 13 + i * 7919)
        # Scatter spawn across the top of the world.
        if args.layout == "track":
            # Outer lane radius (between R_MID and R_OUTER from the track
            # generator), top-of-loop angle band 60°–120° in world-radians
            # (CCW from +x, so π/3 to 2π/3).
            cx = cy = s.maze.size_m / 2.0
            r_lane = 17.0   # outer-lane mid-radius (matches generator)
            theta_w = rng_spawn.uniform(math.pi / 3, 2 * math.pi / 3)
            s.robot.x = cx + r_lane * math.cos(theta_w)
            s.robot.y = cy + r_lane * math.sin(theta_w)
            # Tangent CCW heading at the spawn point.
            s.robot.theta = theta_w + math.pi / 2
        else:
            # Maze: drop into a random cell in the top third whose
            # voxel mask is empty.
            wall = s.maze.wall_voxel_mask
            n_grid = wall.shape[0]
            res = s.grid.res
            j_min = int(n_grid * 2 / 3)
            for _ in range(200):
                ii = rng_spawn.randrange(n_grid)
                jj = rng_spawn.randrange(j_min, n_grid)
                if not wall[ii, jj]:
                    s.robot.x = (ii + 0.5) * res
                    s.robot.y = (jj + 0.5) * res
                    s.robot.theta = rng_spawn.uniform(0, 2 * math.pi)
                    break
        # Reset BT's progress tracker to the new pose.
        s.bt.state.last_progress_pos = s.robot.rear_axle()
        sims.append(s)

    pg.init()
    # Higher-res bake — 1920×1700 is big enough to read individual
    # robot colours and the BT-state legend without scaling artefacts.
    W, H = 1920, 1700
    try:
        screen = pg.display.set_mode((W, H), vsync=0)
    except TypeError:
        screen = pg.display.set_mode((W, H))
    pg.display.set_caption("BT Sim — scatter bake")

    maze = sims[0].maze
    margin_px = 30
    bar_strip_h = 280    # bottom area: stats + bar chart of BT firings
    scale = min((W - 2 * margin_px) / maze.size_m,
                (H - bar_strip_h - 2 * margin_px) / maze.size_m)
    offset_x = (W - maze.size_m * scale) / 2
    offset_y = H - bar_strip_h - margin_px

    def w2s(x: float, y: float) -> tuple[int, int]:
        return (int(offset_x + x * scale),
                int(offset_y - y * scale))

    # Per-robot colours via evenly-spaced HSV hues.
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        c = pg.Color(0)
        c.hsva = ((360 * i / n) % 360, 75, 95, 100)
        colors.append((c.r, c.g, c.b))

    # Pre-render maze walls as voxel boxes.
    wall_mask = maze.wall_voxel_mask
    n_cells = wall_mask.shape[0]
    res = sims[0].grid.res
    cell_screen = max(1, int(round(res * scale)))
    bg_surf = pg.Surface((W, H))
    bg_surf.fill((10, 14, 10))
    pg.draw.rect(bg_surf, (13, 31, 18),
                 (margin_px, margin_px,
                  W - 2 * margin_px,
                  H - bar_strip_h - 2 * margin_px))
    for ci in range(n_cells):
        for cj in range(n_cells):
            if wall_mask[ci, cj]:
                cx, cy = w2s((ci + 0.5) * res, (cj + 0.5) * res)
                pg.draw.rect(bg_surf, (95, 115, 125),
                             (cx - cell_screen // 2,
                              cy - cell_screen // 2,
                              cell_screen, cell_screen))
    # Waypoints (shared across all sims).
    for k, (wx, wy) in enumerate(sims[0].mission):
        wp = w2s(wx, wy)
        pg.draw.circle(bg_surf, (255, 224, 102), wp, 9, 2)
        lbl_font = pg.font.SysFont(
            "Consolas,Menlo,Courier New,monospace", 12, bold=True)
        lbl = lbl_font.render(str(k + 1), True, (255, 224, 102))
        bg_surf.blit(lbl, (wp[0] + 10, wp[1] - 7))

    font_hud = pg.font.SysFont(
        "Consolas,Menlo,Courier New,monospace", 16, bold=True)
    font_legend = pg.font.SysFont(
        "Consolas,Menlo,Courier New,monospace", 11)

    fps = max(1, int(args.bake_fps))
    sub_per_frame = max(1, int(round((1.0 / fps) / PHYS_DT)))
    max_frames = int(args.bake_secs * fps)

    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=8, macro_block_size=1, format="FFMPEG")
    t_wall0 = time.time()
    frames_written = 0
    try:
        for f in range(max_frames):
            for _ in range(sub_per_frame):
                for s in sims:
                    if not s.done:
                        s.step(PHYS_DT)
            screen.blit(bg_surf, (0, 0))
            # Trails + robots.
            for i, s in enumerate(sims):
                color = colors[i]
                if len(s.trail) > 1:
                    pts = [w2s(x, y) for (x, y) in s.trail]
                    pg.draw.lines(screen, color, False, pts, 1)
                rx, ry = s.robot.rear_axle()
                rp = w2s(rx, ry)
                pg.draw.circle(screen, color, rp, 6)
                ax = rx + 0.9 * math.cos(s.robot.theta)
                ay = ry + 0.9 * math.sin(s.robot.theta)
                pg.draw.line(screen, color, rp, w2s(ax, ay), 2)
            # ── Bottom HUD strip ──
            strip_top = H - bar_strip_h
            pg.draw.rect(screen, (13, 22, 16),
                         (0, strip_top, W, bar_strip_h))
            pg.draw.line(screen, (52, 84, 52),
                         (0, strip_top), (W, strip_top), 1)
            # Status line.
            n_done = sum(1 for s in sims if s.done)
            hud = font_hud.render(
                f"t = {sims[0].t:5.1f} s    completed: {n_done} / {n}",
                True, (220, 240, 220))
            screen.blit(hud, (margin_px, strip_top + 12))
            # Per-robot state legend — small swatch + state name. Two
            # columns so we can fit up to ~16 robots without overflow.
            legend_y0 = strip_top + 45
            col_w = W // 2 - margin_px
            for i, s in enumerate(sims):
                col = i // 8
                row = i % 8
                lx = margin_px + col * col_w
                ly = legend_y0 + row * 14
                pg.draw.rect(screen, colors[i], (lx, ly + 2, 10, 10))
                lbl = font_legend.render(
                    f" {i+1:>2}: {s.bt.state.name[:34]}",
                    True, (200, 220, 200))
                screen.blit(lbl, (lx + 14, ly))
            # ── Bar graph of total BT firings across all robots ──
            bar_x0 = W // 2 + 60
            bar_top = strip_top + 45
            bar_h = 18
            bar_gap = 6
            label_w = 130
            bar_max_w = W - bar_x0 - label_w - margin_px - 80
            algos = [
                ("planner_fail",  "n_planner_failures",     (140, 160, 200)),
                ("goal_bend",     "n_goal_bends",           (255, 159, 64)),
                ("breadcrumb",    "n_breadcrumb_reverses",  (255, 139, 52)),
                ("backup",        "n_backup_fires",         (130, 230, 150)),
                ("gradient",      "n_gradient_fires",       (255, 200, 60)),
                ("clear_around",  "n_clear_around_robot",   (90, 160, 240)),
            ]
            totals = [(label, sum(getattr(s.bt.state, attr) for s in sims),
                       colour) for (label, attr, colour) in algos]
            max_total = max((t for (_, t, _) in totals), default=0) or 1
            for k, (label, total, colour) in enumerate(totals):
                by = bar_top + k * (bar_h + bar_gap)
                # Label
                lbl = font_legend.render(label, True, (180, 200, 180))
                screen.blit(lbl, (bar_x0, by + 2))
                # Bar background + filled portion
                pg.draw.rect(screen, (24, 36, 28),
                             (bar_x0 + label_w, by, bar_max_w, bar_h))
                w_fill = int(bar_max_w * total / max_total)
                if w_fill > 0:
                    pg.draw.rect(screen, colour,
                                 (bar_x0 + label_w, by, w_fill, bar_h))
                pg.draw.rect(screen, (60, 90, 60),
                             (bar_x0 + label_w, by, bar_max_w, bar_h), 1)
                # Value count
                val = font_legend.render(
                    str(total), True, (220, 240, 220))
                screen.blit(val,
                            (bar_x0 + label_w + bar_max_w + 8, by + 2))
            # Capture frame.
            arr = pg.surfarray.array3d(screen)
            arr = np.transpose(arr, (1, 0, 2))
            writer.append_data(arr)
            frames_written += 1
            if f % fps == 0:
                print(f"  bake t={sims[0].t:6.2f}s frame {f+1}/{max_frames} "
                      f"done={n_done}/{n}")
            if n_done == n:
                for _ in range(fps):
                    writer.append_data(arr)
                    frames_written += 1
                print(f"  ALL DONE @ t={sims[0].t:.2f}s")
                break
    finally:
        writer.close()
        pg.quit()
    print(f"[scatter-bake] wrote {frames_written} frames in "
          f"{time.time()-t_wall0:.1f}s wall time → {out_path}")


def run_scatter(args):
    """Run N robots headless on different seeds, aggregate per-algorithm
    BT firings + completion stats. Surfaces which recovery branches
    actually do work in practice."""
    n = max(1, args.scatter)
    max_sim_secs = args.scatter_max_secs
    max_ticks = int(max_sim_secs / PHYS_DT)
    print(f"[scatter] running {n} robots, layout={args.layout}, "
          f"max {max_sim_secs:.0f}s sim time per robot")
    print(f"          (wall-detection prob: {WALL_DETECTION_PROB:.2f})")
    print()
    results = []
    t_wall0 = time.time()
    for i in range(n):
        seed = args.seed + i * 7919          # spread layouts deterministically
        sim = Sim.build(n_cells=args.maze_cells, seed=seed,
                        n_obstacles=args.obstacles, layout=args.layout)
        for _ in range(max_ticks):
            sim.step(PHYS_DT)
            if sim.done:
                break
        s = sim.bt.state
        outcome = "GOAL" if sim.done else "TMOT"
        results.append({
            "seed": seed, "done": sim.done, "t": sim.t,
            "mission_idx": sim.mission_idx,
            "planner_fail": s.n_planner_failures,
            "goal_bend":   s.n_goal_bends,
            "bcrumb_rev":  s.n_breadcrumb_reverses,
            "backup":      s.n_backup_fires,
            "gradient":    s.n_gradient_fires,
            "clear_around": s.n_clear_around_robot,
        })
        print(f"  robot {i+1:>3d}/{n}  seed={seed:<10d}  {outcome}  "
              f"t={sim.t:6.1f}s  wp={sim.mission_idx}/{len(sim.mission)}  "
              f"fires: bend={s.n_goal_bends} bc={s.n_breadcrumb_reverses} "
              f"bkup={s.n_backup_fires} grad={s.n_gradient_fires} "
              f"clr={s.n_clear_around_robot}")
    wall_dt = time.time() - t_wall0
    # Aggregate
    n_done = sum(1 for r in results if r["done"])
    print()
    print("=" * 64)
    print(f"  SCATTER SUMMARY  ({n} robots, {wall_dt:.1f}s wall time)")
    print("=" * 64)
    print(f"  completed:        {n_done}/{n}  ({100*n_done/n:.0f}%)")
    if n_done > 0:
        done_times = [r["t"] for r in results if r["done"]]
        print(f"  avg completion:   {sum(done_times)/len(done_times):.1f}s")
        print(f"  fastest / slowest:{min(done_times):.1f}s / {max(done_times):.1f}s")
    fields = ["planner_fail", "goal_bend", "bcrumb_rev",
              "backup", "gradient", "clear_around"]
    total_fires = sum(sum(r[f] for f in fields) for r in results)
    print(f"  total BT fires:   {total_fires}")
    print()
    print(f"  {'algorithm':<14s} {'total':>6s}  {'mean':>6s}  {'max':>4s}  {'% of fires':>10s}")
    print("  " + "-" * 50)
    for f in fields:
        vals = [r[f] for r in results]
        tot = sum(vals)
        mn = tot / max(1, len(vals))
        mx = max(vals) if vals else 0
        pct = 100 * tot / max(1, total_fires)
        print(f"  {f:<14s} {tot:>6d}  {mn:>6.2f}  {mx:>4d}  {pct:>9.1f}%")
    print()


def main():
    p = argparse.ArgumentParser(
        description="BEHAVIOR TREE Sim — Chaplygin sleigh in a corridor maze")
    p.add_argument("--maze-cells", type=int, default=DEFAULT_MAZE_CELLS,
                   help=f"N: maze is N×N cells of {CELL_SIZE_M:g} m each "
                        f"(default {DEFAULT_MAZE_CELLS}).")
    p.add_argument("--seed", type=int, default=7,
                   help="Maze seed (default 7).")
    p.add_argument("--obstacles", type=int, default=6,
                   help="Number of random rectangular obstacles to drop "
                        "inside corridor cells (default 6). Surprise "
                        "dead-ends — visible to the lidar only at close "
                        "range — that exercise the BT recovery branches. "
                        "Only applies to --layout=maze.")
    p.add_argument("--layout", choices=("track", "maze"), default="track",
                   help="World geometry: 'track' = wavy bisected annular "
                        "track with a one-way dead-end (default); 'maze' "
                        "= DFS perfect maze + obstacles.")
    p.add_argument("--heading-deg", type=float, default=None,
                   help="Override initial heading (degrees CCW from +x).")
    p.add_argument("--headless", action="store_true",
                   help="Run without a window; print BT state every 1000 ticks.")
    p.add_argument("--headless-steps", type=int, default=20000,
                   help="Physics ticks to run in --headless mode (default 20000).")
    p.add_argument("--sprite", default=None,
                   help="Path to robot top-down PNG (with #E48787 marker "
                        "pixels at knife edges / COM / caster). Defaults to "
                        "<sim>/data/robot_top.png if it exists.")
    p.add_argument("--no-sprite", action="store_true",
                   help="Force stick-figure rendering even if sprite exists.")
    p.add_argument("--renderer", choices=("pygame", "matplotlib"),
                   default="pygame",
                   help="Render backend: pygame (default, 60 FPS) or "
                        "matplotlib (legacy, ~15-25 FPS).")
    p.add_argument("--scatter", type=int, default=0,
                   help="Headless batch mode: run N robots on different "
                        "seeds (offset by 7919 per robot), aggregate "
                        "per-algorithm BT firing counts. Useful for "
                        "spotting which recovery branches do the work.")
    p.add_argument("--scatter-max-secs", type=float, default=400.0,
                   help="Per-robot sim-time cap in --scatter mode (default 400 s).")
    global WALL_DETECTION_PROB
    p.add_argument("--detection-prob", type=float, default=WALL_DETECTION_PROB,
                   help="Per-ray probability that a wall voxel hit is "
                        "actually detected (default 0.85 — misses leak "
                        "as false negatives until the next sweep).")
    p.add_argument("--bake-mp4", default=None,
                   help="Render the live sim offscreen to this MP4 "
                        "path and exit. Uses pygame + imageio-ffmpeg.")
    p.add_argument("--bake-fps", type=int, default=30,
                   help="MP4 frame rate (default 30).")
    p.add_argument("--bake-secs", type=float, default=180.0,
                   help="Max sim time to bake (default 180 s).")
    args = p.parse_args()
    # CLI override for the global detection probability.
    WALL_DETECTION_PROB = float(args.detection_prob)
    if args.bake_mp4 and args.scatter > 0:
        run_scatter_bake(args)
    elif args.bake_mp4:
        run_bake(args)
    elif args.scatter > 0:
        run_scatter(args)
    elif args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
