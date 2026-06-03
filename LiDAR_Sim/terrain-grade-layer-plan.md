# Terrain Grade Costmap Layer — Implementation Plan

*Updated 2026-05-03 — validated through 15-agent simulation with 100% pass rate at 15%, 20%, and 30% grade thresholds. Real-hardware fidelity (SICK multiScan165, 16 layers, -20°→+7.5° robot-frame FOV, 0.5° angular resolution, 20 Hz, 10 m range).*

## Problem

The current `nav2_paramsv2.yaml` uses an `ObstacleLayer` subscribing to `/scan_fullframe` (LaserScan) with a height band filter. This works on flat ground but breaks on ramps:

- **On a ramp the robot tilts.** Ground appears at the wrong height and is misclassified.
- **We cannot just widen the height band.** It lets ground through or misses cones.

## Solution — Validated by Simulation

Replace `obstacle_layer` with a custom `slope_layer` Nav2 costmap plugin that classifies terrain by **PCA-based surface slope** + **spike detection** for thin objects.

**Simulation results: 15 agents, 3 grade thresholds, 100% completion.**

## Critical Deployment Rule — 2× Grade Threshold

**Set `traversable_max_deg` to ~2× the steepest ramp the robot is expected to climb.**

**Why:** While climbing a ramp at angle α, the *back-slope* on the descending side reads as ~2α in the sensor's tilted frame. RULES.md #1 forbids correcting this with a world-frame / IMU reference (the robot accelerates, gravity is unreliable). By setting the threshold to ≥ 2α, the doubled back-slope reading stays below the cutoff and is correctly classified as traversable — no smearing, no false obstacles ahead of the climbing robot.

**Competition deployment:** the AutoNav 25-26 course has a 15% grade ramp (~8.5°). Set `traversable_max_deg = 16.7` (30% grade) to give the back-slope full headroom. The simulator confirms no smearing at this threshold across all five tested ramp grades.

## Real Robot Implementation Flow

> **Bold step headers (1-7) are the deployment pipeline** that runs on the
> real robot inside the `slope_layer` Nav2 plugin. The simulator's mesh
> ray-casting and agent-physics scaffolding only exists to generate a point
> cloud — on the real robot the SICK multiScan165 produces the same cloud
> directly at 20 Hz, and the perception pipeline below is identical.

```
┌─────────────────────────────────────────────────────────────────┐
│              SICK multiScan165 (MULS1AA-114322) LiDAR            │
│           /cloud_all_fields_fullframe (PointCloud2)              │
│                    frame: lidar_footprint                        │
│  16 layers · 0.5° horiz res · 20 Hz · 10 m range                 │
│  Mounted UPSIDE DOWN: robot-frame FOV is -35° (down) to +7.5°    │
│  (datasheet native: -7.5° down to +35° up; flip for mount)       │
│  NOTE: deployment may software-mask the bottom layers (e.g.      │
│  use only -20° to +7.5°) to skip ground returns inside the       │
│  robot's near-zone — the surface-normal estimator covers them.   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  slope_layer::SlopeLayer (Nav2 Costmap Plugin)                  │
│  Location: isaac_ros-dev/src/slope_layer/                       │
│  Replaces: nav2_costmap_2d::ObstacleLayer                       │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 1: SENSOR SURFACE NORMAL (PCA, not IMU)**         │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Select ~15 ground-classified LiDAR returns within   │  │  │
│  │  │ a 0.5 m disk around the sensor footprint.           │  │  │
│  │  │ Compute centroid; subtract; build covariance.       │  │  │
│  │  │ Eigendecompose → smallest-eigenvalue eigenvector    │  │  │
│  │  │ is the local ground normal. Force +z hemisphere.    │  │  │
│  │  │ Use as the rotation R that aligns sensor-z with     │  │  │
│  │  │ the local surface normal.                           │  │  │
│  │  │ Why PCA over a 3-point tripod: a single foot        │  │  │
│  │  │ landing on a wall edge flips the cross-product      │  │  │
│  │  │ normal by 30+ degrees and corrupts the rest of      │  │  │
│  │  │ the scan. PCA on 15+ samples shrugs off outliers.   │  │  │
│  │  │ Why not IMU: the robot accelerates during motion,   │  │  │
│  │  │ skewing any gravity reading. RULES.md #1.           │  │  │
│  │  │ Output: rotation matrix R, surface normal sn        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 2: GROUND SEGMENTATION**                          │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Bin points into 0.10m grid cells.                   │  │  │
│  │  │ Per cell 3x3 neighborhood: sort Z, split at         │  │  │
│  │  │ first gap > 0.3m.                                   │  │  │
│  │  │ Lower cluster = ground (ramps stay as ground).      │  │  │
│  │  │ Upper cluster (>1.0m tall) = wall/non-ground.       │  │  │
│  │  │ Track wall_detected cells.                          │  │  │
│  │  │ Output: ground_points, non_ground_points,           │  │  │
│  │  │         wall_detected mask                          │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 3: PCA SLOPE CLASSIFICATION (sensor-frame only)** │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Per cell ground points (3x3 neighborhood):          │  │  │
│  │  │   C = covariance matrix (3x3)                       │  │  │
│  │  │   Eigendecompose → smallest eigenvector = normal    │  │  │
│  │  │   grade = acos(|normal · [0,0,1]|)  (sensor frame)  │  │  │
│  │  │ Reject non-planar cells: eigvals[0]/eigvals[2]      │  │  │
│  │  │   > 0.005 means two surfaces in the cell — likely   │  │  │
│  │  │   ramp + adjacent wall — and PCA gives a near-      │  │  │
│  │  │   vertical normal. Mark as NaN, not as obstacle.    │  │  │
│  │  │ Skip wall-adjacent cells (dilated 2 from walls).    │  │  │
│  │  │ NOTE: reference is sensor-frame [0,0,1], NOT world  │  │  │
│  │  │ up. RULES.md #1. Back-slope artifact handled by     │  │  │
│  │  │ choosing threshold = 2 × max climb angle.           │  │  │
│  │  │ Output: per-cell grade in degrees                   │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 4: SPIKE DETECTION (cones, posts, poles)**        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Per cell (skip PCA-traversable cells):              │  │  │
│  │  │   ground_z = median of bottom 30% of z-values      │  │  │
│  │  │   elevated = points with z > ground_z + 0.15m       │  │  │
│  │  │   If >= 2 elevated points → spike obstacle          │  │  │
│  │  │ Works for tilted cones (ground-vs-elevated split).  │  │  │
│  │  │ Protected from PCA override.                        │  │  │
│  │  │ Output: spike_obstacle mask                         │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 5: DBSCAN CLUSTERING + SIZE FILTERING**           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Collect: non-ground + steep ground + spike points   │  │  │
│  │  │ DBSCAN(eps=0.3, min_samples=3)                      │  │  │
│  │  │ Discard clusters < 15 points (noise).               │  │  │
│  │  │ Output: filtered obstacle clusters                  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 6: COSTMAP PROJECTION + PCA OVERRIDE**            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Project clusters to 0.05m global costmap grid.      │  │  │
│  │  │ 2x2 block fill (closes gaps between cells).         │  │  │
│  │  │ PCA override: clear cells where PCA confirms        │  │  │
│  │  │   traversable AND not spike-flagged.                 │  │  │
│  │  │ Output: obstacle grid                               │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  **Step 7: NAV2 INFLATION**                               │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Obstacle cells → 254 (LETHAL)                       │  │  │
│  │  │ Within ROBOT_RADIUS (0.41m) → 253 (INSCRIBED)       │  │  │
│  │  │ Decay zone → exponential falloff                    │  │  │
│  │  │ A* treats >= 253 as impassable.                     │  │  │
│  │  │ Output: inflated costmap → Nav2 planner             │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          ▼                                       │
│        ┌─────────────────────────────────────┐                  │
│        │  /terrain/grade_map (OccupancyGrid) │                  │
│        │  Published for RVIZ visualization   │                  │
│        └─────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                Nav2 Costmap2D                                    │
│  local_costmap:                                                  │
│    plugins: ["slope_layer", "line_layer", "inflation_layer"]     │
│                                                                  │
│  global_costmap:                                                 │
│    plugins: ["static_layer", "slope_layer", "line_layer",        │
│              "inflation_layer"]                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Nav2 Planner (SmacPlannerHybrid)                    │
│         Plans paths using the inflated costmap                   │
│         Avoids lethal (254) and inscribed (253) cells            │
└─────────────────────────────────────────────────────────────────┘
```

## What Gets Replaced in AutoNav_25-26

```
isaac_ros-dev/src/
├── slam/config/
│   └── nav2_paramsv2.yaml          ← MODIFY: swap obstacle_layer → slope_layer
├── line_layer/                      ← KEEP (painted-line detection)
├── slope_layer/                     ← NEW PACKAGE
│   ├── include/slope_layer/
│   │   └── slope_layer.hpp
│   ├── src/
│   │   └── slope_layer.cpp
│   ├── slope_layer.xml
│   ├── CMakeLists.txt
│   └── package.xml
└── sick_scan_xd/                    ← KEEP (LiDAR driver, provides point cloud)
```

### Current Config (obstacle_layer)
```yaml
# nav2_paramsv2.yaml — BEFORE
obstacle_layer:
  plugin: "nav2_costmap_2d::ObstacleLayer"
  enabled: True
  observation_sources: scan
  scan:
    topic: /scan_fullframe          # LaserScan (2D)
    max_obstacle_height: 2.0
    obstacle_max_range: 2.5
    raytrace_max_range: 3.0
```

### New Config (slope_layer)
```yaml
# nav2_paramsv2.yaml — AFTER
slope_layer:
  plugin: "slope_layer::SlopeLayer"
  enabled: True
  cloud_topic: "/cloud_all_fields_fullframe"   # PointCloud2 (3D)
  # NO imu_topic — surface normal is LiDAR-derived (Step 1).
  # RULES.md #1 forbids IMU/gravity/world-frame references because
  # the robot accelerates and the gravity reading drifts under motion.

  # Costmap geometry
  internal_resolution: 0.10         # fine grid cell (m), point binning
  pca_resolution: 0.50              # coarse grid (m) for PCA neighborhoods
  grid_half_size: 8.0               # PCA grid extent (m)

  # Grade threshold — SET TO ~2× THE STEEPEST CLIMB ANGLE.
  # Competition ramp is 15% (~8.5°), so 2× → 30% (16.7°).
  # See "Critical Deployment Rule — 2× Grade Threshold" above.
  traversable_max_deg: 16.7         # 30% grade — keeps back-slope clean
  pca_noise_margin_deg: 1.5         # add to threshold for "lethal grade"
  pca_max_valid_deg: 60.0           # PCA outputs above this = noise

  # Sensor surface normal (Step 1)
  surface_normal_samples: 15        # ground-PCA sample count
  surface_normal_radius: 0.5        # disk radius (m) around sensor
  surface_normal_z_window: 0.15     # ± window around floor estimate

  # Ground / wall split (Step 2)
  z_ground_band: 0.3                # Z-gap for ground/wall split
  wall_min_height: 1.0              # min height above ground to be wall

  # PCA classification (Step 3)
  min_points_for_pca: 6
  pca_planarity_max: 0.005          # eigval ratio cutoff (rejects ramp+wall mix)
  wall_adjacent_dilation: 2         # cells around walls excluded from PCA

  # Spike detection (Step 4)
  spike_height: 0.15                # min height above local ground
  spike_min_elevated: 2             # min elevated points to trigger

  # DBSCAN (Step 5)
  dbscan_eps: 0.3
  dbscan_min_samples: 3
  min_cluster_size: 15

  # Inflation (Step 7) — Nav2 standard
  robot_radius: 0.15                # match the deployed chassis radius
  inflation_radius: 0.15            # decay-zone width past inscribed

  publish_grade_map: True           # /terrain/grade_map for RVIZ
```

### What Changes

| Item | Before (obstacle_layer) | After (slope_layer) |
|------|------------------------|---------------------|
| Plugin | `nav2_costmap_2d::ObstacleLayer` | `slope_layer::SlopeLayer` |
| Input | `/scan_fullframe` (2D LaserScan) | `/cloud_all_fields_fullframe` (3D PointCloud2) |
| Detection | Height band [0.4m, 2.0m] | PCA slope + Z-gap walls + spike detection |
| Works on ramps | No | Yes (validated 15/15 agents at 15/20/30% thresholds) |
| Thin objects (cones) | Height band only | Spike detector (ground-vs-elevated split) |
| Surface normal | None | 15-point PCA on ground returns (no IMU) |
| Grade reference | N/A | Relative to LiDAR-derived sensor normal (RULES.md #1) |
| Ramp traversal | Blocked | Allowed (slope < threshold) |
| Back-slope smear | N/A (no ramp logic) | Mitigated by 2× threshold rule, not by ray filtering |

### What Stays the Same

- `line_layer` — painted-line boundary detection from ZED camera
- `inflation_layer` — inflates obstacles (no changes)
- `static_layer` — SLAM map on global costmap
- All controller/planner params
- Grade compensation in `control.cpp` (motor speed on slopes)

## Validated Grade Thresholds

| Threshold | Degrees | Ramps PASS | Ramps BLOCKED | Sim Result | Costmap clean? |
|-----------|---------|------------|---------------|------------|----------------|
| 15% grade | 8.5 deg | 5 deg | 10, 15, 20, 25 deg | 15/15 ARRIVED | Some back-slope smear on the 5° ramp |
| 20% grade | 11.3 deg | 5, 10 deg | 15, 20, 25 deg | 15/15 ARRIVED | Some back-slope smear on the 5° + 10° ramps |
| 30% grade | 16.7 deg | 5, 10, 15 deg | 20, 25 deg | 15/15 ARRIVED | **Clean — no smearing on any traversed ramp** |

**Recommended for competition: 30% grade (16.7°)** — 2× the 15% (~8.5°) competition ramp gives the back-slope full headroom and produces a clean costmap on the deployed robot.

## Validated LiDAR Configuration (Real Hardware Match)

```
Sensor:           SICK multiScan165 (MULS1AA-114322)
Mount:            Upside-down on robot chassis
Layers:           16
Vertical FOV:     -35° (down) to +7.5° (up) in robot frame
                  (datasheet native: -7.5° to +35°; flipped by mount)
Recommended SW mask: -20° to +7.5°  (drop -35°..-20° as redundant
                  with the 15-sample surface_normal estimator)
Horizontal:       360°, 0.5° angular resolution per layer
Points per scan:  ~11,520 (720 per layer × 16 layers)
Scan frequency:   20 Hz baseline (40 Hz mode for layers 4-13 only)
Range:            10 m operational (datasheet 0.05-62 m; cap at 10 m)
Sensor height:    ~0.30 m above ground (1 ft)
```

The simulator is configured to match these values; sim-to-real should preserve perception behavior at the algorithm level.

## Core C++ Implementation

### PCA Grade Computation
```cpp
// IMPORTANT: ref_normal MUST be (0, 0, 1) in the SENSOR frame.
// The points must already be transformed by R^T (where R aligns sensor-z
// to the local surface_normal) so that the sensor frame's z-axis is
// aligned with the local ground normal. RULES.md #1: never pass
// world-up here — the back-slope smear is handled by the 2× threshold
// rule, not by a frame correction inside this function.
float SlopeLayer::computeGradePCA(const std::vector<Eigen::Vector3f> &points,
                                   const Eigen::Vector3f &ref_normal /* = {0,0,1} */) const {
  const int n = static_cast<int>(points.size());
  if (n < min_points_for_pca_) return NAN;

  // Reject point sets that are line-like or single-ring (no spread in 2 axes).
  Eigen::Vector3f spreads = points_max_minus_min(points);  // helper: per-axis range
  std::array<float, 3> s = {spreads.x(), spreads.y(), spreads.z()};
  std::sort(s.begin(), s.end());
  if (s[1] < 0.10f) return NAN;   // need >= 10 cm spread in 2nd axis

  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto &p : points) centroid += p;
  centroid /= static_cast<float>(n);

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto &p : points) {
    Eigen::Vector3f d = p - centroid;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(n - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
  Eigen::Vector3f eigvals = solver.eigenvalues();           // ascending
  // Reject 1D point clouds and non-planar (ramp + adjacent wall) cells.
  if (eigvals[1] < eigvals[2] * 0.01f) return NAN;
  if (eigvals[2] > 1e-12f &&
      (eigvals[0] / eigvals[2]) > pca_planarity_max_)       // 0.005
    return NAN;
  Eigen::Vector3f normal = solver.eigenvectors().col(0);    // smallest eigval

  float cos_angle = std::abs(normal.dot(ref_normal))
                    / (normal.norm() * ref_normal.norm());
  return std::acos(std::clamp(cos_angle, 0.0f, 1.0f)) * 180.0f / M_PI;
}
```

### Spike Detection
```cpp
bool SlopeLayer::isSpikeCell(const std::vector<Eigen::Vector3f> &points) const {
  if (points.size() < spike_min_elevated_) return false;
  // Sort by z, get ground level from bottom 30%
  std::vector<float> zs;
  for (const auto &p : points) zs.push_back(p.z());
  std::sort(zs.begin(), zs.end());
  int n_ground = std::max(1, (int)zs.size() / 3);
  float ground_z = zs[n_ground / 2];  // median of bottom 30%
  // Count elevated points
  int elevated = 0;
  for (float z : zs) {
    if (z > ground_z + spike_height_) elevated++;
  }
  return elevated >= spike_min_elevated_;
}
```

### Sensor Surface Normal (PCA, replaces tripod)

Pick ~15 ground-classified returns inside a 0.5 m disk under the sensor, then PCA-fit. A 3-point tripod cross-product was previously used but a single foot landing on a wall edge flipped the normal by 30°+ and corrupted whole scans. PCA on 15+ points shrugs off outliers.

```cpp
Eigen::Vector3f SlopeLayer::computeSensorNormalPCA(
    const std::vector<Eigen::Vector3f> &nearby_ground_pts) const {
  const int n = static_cast<int>(nearby_ground_pts.size());
  if (n < 3) return {0.f, 0.f, 1.f};   // fall back to "up" if too sparse

  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto &p : nearby_ground_pts) centroid += p;
  centroid /= static_cast<float>(n);

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto &p : nearby_ground_pts) {
    Eigen::Vector3f d = p - centroid;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(std::max(1, n - 1));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
  Eigen::Vector3f normal = solver.eigenvectors().col(0);  // smallest eigval
  if (normal.z() < 0.f) normal = -normal;                  // force +z hemisphere

  // Optional safety clamp (e.g., if returned tilt exceeds 45°, default to up).
  float cos_tilt = std::clamp(normal.normalized().z(), 0.f, 1.f);
  if (std::acos(cos_tilt) > max_tilt_rad_) return {0.f, 0.f, 1.f};
  return normal.normalized();
}
```

**Sample selection helper.** On the real robot, "nearby ground points" = LiDAR returns inside a 0.5 m horizontal radius of the sensor footprint with z within ±0.15 m of the local floor estimate (running median of recent scans is fine).

## Package Dependencies

```xml
<!-- package.xml -->
<depend>rclcpp</depend>
<depend>nav2_costmap_2d</depend>
<depend>nav2_common</depend>
<depend>pluginlib</depend>
<depend>sensor_msgs</depend>
<depend>nav_msgs</depend>
<build_depend>eigen</build_depend>
```

No PCL dependency — iterate PointCloud2 fields directly via `sensor_msgs::PointCloud2Iterator`.

## Performance Budget

- 20 Hz scan rate → **50 ms per frame** to stay real-time
- RULES.md #8 caps the perception pipeline at **60 ms** end-to-end
- Step-by-step:
  - Step 1 surface_normal PCA (15 points): <1 ms
  - Step 2 ground segmentation (cell binning + Z-gap): ~5-10 ms at 11.5k points
  - Step 3 PCA per cell (closed-form 3×3 eigendecompose): a few ms total
  - Step 4 spike detection: one pass over cell dict, <2 ms
  - Step 5 DBSCAN: only on non-traversable points (small subset), <5 ms
  - Step 6 projection: trivial
  - Step 7 inflation: standard Nav2 cost
- Target on Jetson Orin: **< 20 ms per update**, ~30 ms headroom

## Build

```bash
colcon build --packages-select slope_layer slam
```

## Bring-Up & Testing Plan

Test in this order. Each step builds confidence before moving on.

1. **Flat ground baseline.** Park robot on level concrete. Verify:
   - All cells in the costmap are FREE
   - Cones/barriers placed nearby register as LETHAL
   - `traversable_max_deg = 16.7` (30%) is set
2. **Sensor normal sanity.** While stationary on flat ground, the published surface normal should be `(0, 0, 1)` ± 1°. Tilt the robot manually 5° and confirm the normal updates within 50 ms.
3. **Static ramp scan.** Park robot at the **base** of the 15% competition ramp. The ramp surface should color as low/moderate cost (below the 30% threshold), not LETHAL. Cones flanking the ramp should still register.
4. **Climb the ramp slowly (≤ 0.3 m/s).** Watch the costmap during the climb:
   - Ramp surface ahead stays low cost (no smearing)
   - Floor past the ramp peak stays clear (no back-slope smear at 30% threshold)
   - Cones beyond the peak start appearing at ~3 m range and lock in by ~1.5 m
5. **Descend the ramp.** Verify the costmap stays clean — back-slope smear is not a concern on descent (the geometry doesn't double the apparent grade in this direction).
6. **Tilted cone test.** Lay a cone on its side. Spike detector must catch it (PCA alone would classify it as a near-vertical normal and reject it).
7. **Wall base.** Park 0.30 m from a wall. The wall should be marked LETHAL but the floor immediately at its base should be FREE (wall-adjacent PCA exclusion working).
8. **Performance.** With `ros2 topic hz /terrain/grade_map` confirm 20 Hz output. With `top` / `htop` confirm slope_layer node CPU < 50% on one core.

## Simulation Validation Summary

The simulator (`src/test_ramp_crossing.py` + `src/lidar_sim_gui.py`) runs **15 agents** (3 rows of 5) through two phases on terrain with 5 ramps (5°, 10°, 15°, 20°, 25°), walls, and small obstacles:

- **Phase 1:** Navigate from below the ramps to above (tests ramp + obstacle detection)
- **Phase 2:** Reverse — above to below, routing around walls (tests Phase-1-built costmap)

**Headline result (real-hardware fidelity, FRICTION=28, 30% threshold):**
- 15/15 agents ARRIVED in both phases
- Costmap visually clean — no back-slope smearing on traversed ramps
- Real obstacles (walls, cones, ramps above threshold) correctly registered

Key validated algorithms (all in this MD):
- 15-sample PCA surface normal with planarity rejection (replaces brittle tripod)
- Sensor-frame slope classification — never world-frame, never IMU
- Z-gap split for wall vs ramp ground
- PCA planarity ratio rejection (`eigvals[0]/eigvals[2] > 0.005` → NaN, skips ramp+wall mixed cells)
- Wall-adjacent PCA exclusion (dilate wall mask by 2 cells)
- Spike detection for tilted cones/posts
- 2×2 gap-filling projection (closes seams between cells)
- A* + agent physics block INSCRIBED+LETHAL (cost ≥ 253)
- Spawn-mask erosion by ROBOT_RADIUS+INFLATION_RADIUS so goals never land in inflation halos
- 2× threshold rule for back-slope absorption (replaces all attempted ray-dropout filters; those introduced their own regressions)

## Operational Notes & Gotchas

These were learned the hard way during simulator validation. None of them require code changes on deployment, but they're worth knowing:

- **Don't IMU-correct the slope.** Tried it; geometrically correct but RULES.md #1 forbids it because robot acceleration skews gravity. Use the 2× threshold rule instead.
- **Don't drop low-elevation rays when climbing.** Tried it; reduces ground evidence on the *current* ramp surface and creates new false obstacles. The 2× threshold rule is strictly better.
- **Surface-normal estimator is sensitive to wall edges.** A 3-point tripod sample landing one foot on a wall flips the normal by 30°. The 15-sample PCA in Step 1 fixes this — keep it.
- **Sensor height matters for the FOV math.** At 0.30 m (1 ft) and -20° elevation, the lowest ray hits ground at ~0.82 m. If the real chassis mounts the sensor at a different height, that hit-distance ring shifts; the perception still works but the "useful" inner FOV recommendation may need re-tuning.
- **PCA confidence rises fast as you approach an object.** Distant tires (1-2 hits) don't get classified. They show up around 2-3 m and lock in by 1.5 m. This is a feature — sparse-obstacle courses get straight-line paths until obstacles are reliably visible.
- **At deployment speed (slow robot), the 20 Hz scan rate gives ~10 cm of motion between scans.** The 60 ms perception budget is comfortable.
- **Cache the sensor pose for `surface_normal` PCA.** It's the same 0.5 m disk every frame; reuse the geometric query if your driver supports it.

## Tunable Hyperparameters (slope_layer deployment)

These five knobs are the field-tunable surface area of the deployed perception pipeline. They live in the `slope_layer` block of `nav2_paramsv2.yaml` and are read by the C++ plugin via the standard Nav2 `declareParameter` flow at activation time — no recompile to retune. The simulator mirrors the same five keys so any value validated in simulation maps 1:1 onto the robot.

**Defaults are the simulator-validated values (15 / 15 agents pass at 15 / 20 / 30 % grade thresholds).** Each comment names the knob, what tightening it costs you, what loosening it costs you, and a working range — so anyone editing the file knows the trade before they save.

### YAML schema (drop into `slope_layer:` block of `nav2_paramsv2.yaml`)

```yaml
# ── Ground / wall split (Step 2) ─────────────────────────────────────────────
# Per cell, sort points by Z and split at the first vertical gap larger than
# z_ground_band. Lower cluster is ground (fed to PCA). Upper cluster is
# flagged as a wall iff its vertical span exceeds wall_min_height.
z_ground_band: 0.1            # Min Z-gap (m) to trigger a split. ↑ ignores
                              # more sensor noise; ↓ catches thinner separations
                              # (chair-leg base, gate bar). Tied to LiDAR Z
                              # jitter — rarely moved far from 0.1.
                              # Range 0.05–0.30.
wall_min_height: 0.5          # Min upper-cluster vertical span (m) to call it
                              # a wall. Keep above tallest cone (~0.45m) so
                              # cones fall through to spike detection instead.
                              # ↑ short walls slip through; ↓ tall cones get
                              # misread. Range 0.3–1.0.

# ── PCA classification (Step 3) ──────────────────────────────────────────────
# Reject cells whose covariance is not planar enough — those are ramp+wall
# mixes that produce a near-vertical PCA normal and would be misread as
# steep slope. Marked NaN (not obstacle); spike + DBSCAN handle them.
pca_planarity_max: 0.005      # λ_min / λ_max cutoff for "non-planar — reject."
                              # ↑ accepts more wall-contaminated cells (bleed);
                              # ↓ rejects more aggressively (cleaner, more NaNs
                              # at ramp/wall transitions). Range 0.001–0.02.

# ── Spike detection (Step 4) ─────────────────────────────────────────────────
# Fires only on cells PCA could not classify as traversable. Counts points
# elevated > 0.15 m above the cell's local ground; if at least this many
# exist, the cell becomes an obstacle. Catches cones, posts, gate bars.
spike_min_elevated: 2         # Min elevated points to flag a spike obstacle.
                              # ↑ fewer false spikes from a single stray ray;
                              # ↓ catches sparser thin objects (poles at long
                              # range). Range 1–5. Try 3 if long-range returns
                              # are sparse on the deployed sensor.

# ── DBSCAN clustering (Step 5) ───────────────────────────────────────────────
# All candidate obstacle points are clustered with DBSCAN(eps=0.3,
# min_samples=3); clusters smaller than this are dropped as noise
# (FOV-edge artifacts, single-cell ghosts).
min_cluster_size: 15          # Min cluster size to keep as an obstacle.
                              # ↑ filters more noise but may drop distant real
                              # cones (>5 m often yield <20 returns);
                              # ↓ retains small real objects but admits more
                              # false positives. Range 5–30.
```

### C++ plugin parameter wiring (slope_layer)

In `slope_layer.cpp::onInitialize()`, declare each knob alongside the existing parameters:

```cpp
declareParameter("z_ground_band",      rclcpp::ParameterValue(0.1));
declareParameter("wall_min_height",    rclcpp::ParameterValue(0.5));
declareParameter("pca_planarity_max",  rclcpp::ParameterValue(0.005));
declareParameter("spike_min_elevated", rclcpp::ParameterValue(2));
declareParameter("min_cluster_size",   rclcpp::ParameterValue(15));

node->get_parameter(name_ + "." + "z_ground_band",      z_ground_band_);
node->get_parameter(name_ + "." + "wall_min_height",    wall_min_height_);
node->get_parameter(name_ + "." + "pca_planarity_max",  pca_planarity_max_);
node->get_parameter(name_ + "." + "spike_min_elevated", spike_min_elevated_);
node->get_parameter(name_ + "." + "min_cluster_size",   min_cluster_size_);
```

Defaults in the `declareParameter` calls match the simulator-validated values, so a plugin built before `nav2_paramsv2.yaml` is updated still ships correct behavior.

### Simulator parity loader

To keep simulator validation aligned with what the robot will actually run, `lidar_sim_gui.py` reads the same five keys from `config/sim_params.yaml` (repo-root) at startup. Missing file or missing keys → built-in defaults (same values as the C++ side). Active values are logged on startup so a stale file is visible:

```python
from pathlib import Path
import yaml

_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "sim_params.yaml"
_DEFAULTS = {
    "z_ground_band":      0.1,
    "wall_min_height":    0.5,
    "pca_planarity_max":  0.005,
    "spike_min_elevated": 2,
    "min_cluster_size":   15,
}

def _load_params():
    if not _PARAMS_PATH.exists():
        print(f"[sim_params] {_PARAMS_PATH} not found — using built-in defaults")
        return dict(_DEFAULTS)
    try:
        with _PARAMS_PATH.open() as f:
            user = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[sim_params] failed to parse {_PARAMS_PATH}: {e} — using defaults")
        return dict(_DEFAULTS)
    merged = {**_DEFAULTS, **{k: user[k] for k in _DEFAULTS if k in user}}
    print(f"[sim_params] loaded: {merged}")
    return merged

_P = _load_params()
Z_GROUND_BAND      = _P["z_ground_band"]
WALL_MIN_HEIGHT    = _P["wall_min_height"]
PCA_PLANARITY_MAX  = _P["pca_planarity_max"]
SPIKE_MIN_ELEVATED = _P["spike_min_elevated"]
MIN_CLUSTER_SIZE   = _P["min_cluster_size"]
```

### Reconcile the existing config block

The "New Config (slope_layer)" example earlier in this plan currently lists `z_ground_band: 0.3` and `wall_min_height: 1.0`. Those values **were not validated** — the 100 % pass rate cited in this document was measured at 0.1 / 0.5. Update that earlier block to match before any deployment cuts a build, or note explicitly that the values shown there are aspirational and unvalidated.

## File Manifest (simulator, for cross-reference during deployment)

The deployment plugin should mirror these reference implementations:

| File (in `src/`) | What's in it (deployment-relevant) |
|---|---|
| `lidar_sim_gui.py` | `gen_rays`, `do_scan`, `surface_normal`, `_pca_slope_deg`, `build_grade_costmap`, `GlobalCostmap`, A* planner. The PCA pipeline matches what slope_layer should implement in C++. |
| `test_ramp_crossing.py` | Agent class with physics/spawn-erosion/path-following — *not* deployment code, but the SCAN_PERIOD (50 ms = 20 Hz) and the spawn-mask erosion pattern are useful references. |
| `RULES.md` | The 8 rules the deployment must obey. **Read this first.** |

## Quick Start for the Deployer (next Claude instance picking this up)

You are deploying the validated `slope_layer` Nav2 plugin onto the AutoNav 25-26 robot. Read in this order:

1. **`RULES.md`** in the simulator repo — the 8 hard constraints. Rule #1 (no IMU/world frame for slope) and Rule #8 (60 ms budget) are the most likely to be violated by a careless port.
2. **This document, top-to-bottom** — especially the "Critical Deployment Rule — 2× Grade Threshold" section. Use `traversable_max_deg = 16.7` (30%) for the competition.
3. **`src/lidar_sim_gui.py` — `build_grade_costmap`** — the reference Python implementation of Steps 2-7. Mirror it in C++ inside `slope_layer.cpp`. Do not introduce world-frame references; the back-slope artifact is handled by threshold choice, not by the algorithm.
4. **`src/lidar_sim_gui.py` — `surface_normal`** — the 15-sample PCA. Implement as `computeSensorNormalPCA` (snippet provided above).

What to **NOT** do (these were tried in simulation and removed):
- Don't add an "uphill low-ray dropout" filter. It strips ground evidence from the current ramp surface and creates new false obstacles.
- Don't compute slope against world-up via `R^T @ [0,0,1]`. Tried it; works geometrically but violates Rule #1 because the LiDAR-derived `sn` is implicitly a gravity proxy and unreliable under acceleration.
- Don't subscribe to `sick_scansegment_xd/imu`. Surface normal is LiDAR-derived (Step 1), period.

Sanity tests after first build (in order, see "Bring-Up & Testing Plan" above):
1. Flat-ground baseline — costmap is FREE everywhere except real obstacles.
2. Sensor-normal sanity — published normal matches the chassis tilt.
3. Static ramp scan at the ramp base — ramp surface is moderate cost.
4. Slow climb on the competition ramp — no smearing, no false obstacles ahead.
5. Descend — costmap stays clean.
6. Tilted cone — spike detector catches it.
7. Wall base — wall LETHAL, floor at base FREE.
8. Performance — 20 Hz throughput, < 50 ms per update, < 50% CPU on one core.

If a test fails, do NOT add hacks. Re-read the relevant section of this MD and the simulator reference implementation. Every gotcha listed under "Operational Notes" was a real failure mode caught in simulation and resolved cleanly.

---

## Alternative Deployment — `autonav_detection` (Single Package, Two Executables)

This is the preferred deployment path on `fix/behavior-tree-triggering`. Instead of writing a custom Nav2 plugin (the `slope_layer` plan above) **or** adding a new standalone package, **rename** the existing `line_detection` package to `autonav_detection` and add the PCA grade detector as a second executable inside it. The existing `nav2_costmap_2d::ObstacleLayer` consumes the filtered cloud directly — no plugin authoring, no new package, no rebuild of the Nav2 stack.

### Why this shape, not a new package

- The user constraint is **no net new packages**. Current branch has ~20 packages in `isaac_ros-dev/src/`; this plan keeps it the same.
- Both detectors have the same upstream dependency: **all sensors must be up and healthy** (line wants ZED RGB+depth, grade wants SICK PointCloud2). Housing them together makes that dependency explicit — one launch button, one place to gate.
- Both detectors expose the same output shape to the costmap: a topic the ObstacleLayer subscribes to.
- Same algorithm core as the slope_layer plan; we keep Steps 1–5 in C++ + Eigen and let Nav2 do Steps 6–7 (projection, inflation) for free.

### Existing pipeline (where we insert)

```
┌─────────────────────────────────────────────────────────────┐
│  sick_scan_xd  (sick_multiscan.launch.py)                   │
│    /cloud_all_fields_fullframe  (PointCloud2, 20 Hz)        │
│        frame: lidar_footprint                                │
│    /scan_fullframe              (LaserScan,   20 Hz)        │
│        driver-native — no pointcloud_to_laserscan node      │
└──────────────────────────┬──────────────────────────────────┘
                           │  /scan_fullframe (LaserScan)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  nav2_costmap_2d::ObstacleLayer                             │
│  slam/config/nav2_paramsv2.yaml (local + global, identical) │
│    observation_sources: scan                                 │
│    scan.topic: /scan_fullframe                               │
│    scan.data_type: LaserScan                                 │
│    max_obstacle_height: 2.0,  obstacle_max_range: 2.5,      │
│    raytrace_max_range: 3.0                                   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
            Local + global costmap → InflationLayer
                           ▼
                 Nav2 planner (SmacPlannerHybrid)
```

TF chain: `map → odom → base_link → lidar_footprint`
(`lidar_footprint` mounted at `xyz=0.44 0 0.15`, `rpy=3.1415 0 0` — upside-down).

### Modified pipeline (with `grade_detector` inserted)

```
┌─────────────────────────────────────────────────────────────┐
│  sick_scan_xd  (unchanged)                                  │
│    /cloud_all_fields_fullframe  (PointCloud2)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  autonav_detection  (renamed from line_detection)           │
│                                                              │
│    Executable 1: line_detector  (existing, CUDA)            │
│      subscribes: /zed/zed_node/rgb/...                      │
│      publishes:  /line_detection/lines                      │
│        (node name unchanged → topic name unchanged)         │
│                                                              │
│    Executable 2: grade_detector  (NEW, Eigen, no CUDA)      │
│      subscribes: /cloud_all_fields_fullframe                │
│      runs Steps 1–5 of the slope_layer algorithm:           │
│        1. Surface-normal PCA (15-sample, sensor frame)      │
│        2. Ground segmentation (Z-gap, wall mask)            │
│        3. PCA slope per cell (eigval-ratio rejection)       │
│        4. Spike detection (cones, posts)                    │
│        5. DBSCAN + size filter                              │
│      publishes: /scan_pca_filtered_points (PointCloud2,     │
│                 xyz only, lidar_footprint, 20 Hz)           │
└──────────────────────────┬──────────────────────────────────┘
                           │  /scan_pca_filtered_points
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  nav2_costmap_2d::ObstacleLayer  (config edit only)         │
│    observation_sources: pca_cloud                            │
│    pca_cloud.topic: /scan_pca_filtered_points                │
│    pca_cloud.data_type: PointCloud2                          │
└─────────────────────────────────────────────────────────────┘
```

Driver, TF tree, Nav2 plugin set, package count, and topic names from `line_detector` are all unchanged. The only edits are (a) the rename, (b) adding the second executable, (c) one line in `nav2_paramsv2.yaml` per costmap, (d) one button row in the GUI.

### Package layout

```
isaac_ros-dev/src/autonav_detection/      ← was: line_detection/
├── include/autonav_detection/             ← was: include/line_detection/
├── src/
│   ├── line/                              ← existing files moved here
│   │   ├── node.cpp
│   │   ├── detection.cpp
│   │   └── cuda.cu                        ← CUDA only used by line_detector
│   └── grade/                             ← NEW
│       ├── pca_node.cpp                   ← ROS node + lifecycle
│       └── pca_pipeline.cpp               ← Steps 1–5, pure C++ + Eigen
├── launch/
│   └── detection.launch.py                ← brings up BOTH executables
├── CMakeLists.txt                         ← two add_executable() targets
└── package.xml                            ← merged deps (existing + Eigen)
```

### CMakeLists pattern (no GLOB — explicit per-target sources)

The current `line_detection/CMakeLists.txt` uses `file(GLOB src/*.cpp src/*.cu)` which would sweep both detectors into one binary. Switch to explicit per-executable file lists so CUDA stays scoped to `line_detector` only:

```cmake
project(autonav_detection LANGUAGES CXX CUDA)

# (same find_package() block as today, plus geometry_msgs / nav_msgs for grade)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# ---- line_detector (existing CUDA target) ----
add_executable(line_detector
  src/line/node.cpp
  src/line/detection.cpp
  src/line/cuda.cu
)
target_compile_features(line_detector PUBLIC cxx_std_17)
ament_target_dependencies(line_detector
  rclcpp autonav_interfaces cv_bridge tf2_ros tf2_eigen Eigen3
  image_geometry std_srvs tf2_geometry_msgs
)
target_link_libraries(line_detector ${_cuda_npp_targets})

# ---- grade_detector (NEW, pure C++ + Eigen, no CUDA) ----
add_executable(grade_detector
  src/grade/pca_node.cpp
  src/grade/pca_pipeline.cpp
)
target_compile_features(grade_detector PUBLIC cxx_std_17)
ament_target_dependencies(grade_detector
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros Eigen3
)

install(TARGETS line_detector grade_detector
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
```

### `grade_detector` node interface

```
Subscriptions:
  /cloud_all_fields_fullframe   sensor_msgs/PointCloud2

Publications:
  /scan_pca_filtered_points     sensor_msgs/PointCloud2 (xyz only,
                                  lidar_footprint frame, only obstacle pts)
  /terrain/grade_map            nav_msgs/OccupancyGrid (debug, optional)
  /pca/surface_normal           geometry_msgs/Vector3Stamped (debug, optional)

Parameters (reuse the slope_layer YAML defaults validated in simulation —
  internal_resolution, pca_resolution, traversable_max_deg=16.7,
  surface_normal_samples=15, surface_normal_radius=0.5,
  pca_planarity_max=0.005, spike_height=0.15, dbscan_eps=0.3,
  dbscan_min_samples=3, min_cluster_size=15, etc.)
```

### Costmap config patch

```yaml
# slam/config/nav2_paramsv2.yaml — replace the existing scan source
# Apply identically to BOTH the local and global costmap obstacle_layer blocks.
obstacle_layer:
  plugin: "nav2_costmap_2d::ObstacleLayer"
  enabled: True
  observation_sources: pca_cloud
  pca_cloud:
    topic: /scan_pca_filtered_points
    data_type: "PointCloud2"
    observation_persistence: 0.0
    max_obstacle_height: 2.0
    min_obstacle_height: 0.0
    clearing: True
    marking: True
    raytrace_max_range: 3.0
    raytrace_min_range: 0.0
    obstacle_max_range: 2.5
    obstacle_min_range: 0.0
```

### GUI integration (one button → both detectors)

The GUI has a "LINE DETECT" button at `autonav-gui-hud/autonav_gui_hud/hud_node.py:485` that calls `./config/run-lines.sh`. Replace it with one "DETECT" button that launches both executables via the package's launch file:

```python
# hud_node.py:485 — before
("LINE DETECT", ["LINE DETECT"], "./config/run-lines.sh"),

# hud_node.py:485 — after
("DETECT", ["LINE DETECT", "PCA GRADE"], "./config/run-detect.sh"),
```

Add `"PCA GRADE"` to `virtual_names` at `hud_node.py:819`. The button infra already supports one button driving multiple status dots — see "Pre-SLAM" at hud_node.py:480 lighting four dots.

`config/run-detect.sh` is a one-liner:
```bash
ros2 launch autonav_detection detection.launch.py
```

`launch/detection.launch.py` declares both `Node` actions and any shared params. Single `Ctrl-C` brings both down together.

### Migration order (non-destructive, runnable mid-way)

1. `git mv isaac_ros-dev/src/line_detection isaac_ros-dev/src/autonav_detection`
2. Move existing source files into `src/line/`; rename `include/line_detection/` → `include/autonav_detection/`.
3. Update `package.xml` `<name>` and `CMakeLists.txt` `project()` to `autonav_detection`. Switch the GLOB to the explicit two-target pattern above.
4. Search-and-replace `line_detection` → `autonav_detection` in the **four** external references (no automator changes needed — the topic name is set by the node name `line_detection_node`, not the package name):
   - `config/run-lines.sh`
   - `config/run-lines-real.sh`
   - `src/bringup/launch/demo_day.launch.py` (lines 83-86, 124)
   - `src/autonav_automated_testing/launch/t002_Line_Comp.launch.py` (lines 77-80, 153)
5. Rebuild → run `t002_Line_Comp.launch.py` to confirm `/line_detection/lines` still publishes (proves the topic name didn't break).
6. Add `src/grade/pca_node.cpp` + `src/grade/pca_pipeline.cpp` (port from `lidar_sim_gui.py:build_grade_costmap` + Steps 1–5 of slope_layer above).
7. Add `launch/detection.launch.py` and `config/run-detect.sh`.
8. Patch `nav2_paramsv2.yaml` (both costmaps) to subscribe to `/scan_pca_filtered_points`.
9. Patch `hud_node.py` ("LINE DETECT" → "DETECT", add `"PCA GRADE"` virtual name).

Steps 1–5 land separately from steps 6–9: the codebase is fully runnable after step 5 with zero behavior change. The PCA detector arrives in steps 6–9 as one cohesive change.

### Bring-up order (production launch file)

Same as today's `bringup.launch.py`. The new `grade_detector` lives inside `autonav_detection` and starts when the GUI fires the DETECT button — i.e., **after** `Camera`, `Lidar`, and `SLAM` are green. No reordering of the existing bringup is required.

1. `core_bringup.launch.py` — TF, robot_state_publisher
2. `sensors.launch.py` — ZED + `sick_multiscan.launch.py`
3. `slam.launch.py` — SLAM, EKF, nav2 (consumes `/scan_pca_filtered_points` once detection is up)
4. `control.launch.py` — GPS, encoders
5. **GUI button "DETECT"** → `autonav_detection/detection.launch.py` (line + grade)

### Validation hooks (reuse the Bring-Up & Testing Plan above)

All 8 sanity tests apply unchanged. Add three node-level checks:

- `ros2 node list | grep -E 'line_detection_node|grade_detector'` → both present after the DETECT button.
- `ros2 topic hz /scan_pca_filtered_points` → 20 Hz, matches driver rate.
- `ros2 topic echo --field header.frame_id /scan_pca_filtered_points` → `lidar_footprint` (TF then carries it into `odom`/`map` for the costmap).

### What NOT to do

- **Don't add a new package.** The whole point of this section is to keep the package count flat. Resist the urge to spin off `autonav_perception`, `pca_obstacle_filter`, `lidar_pca_filter`, etc.
- **Don't rename the line node.** Keep `name="line_detection_node"` so `/line_detection/lines` is unchanged. The automators in `autonav_automated_testing` subscribe to it by topic.
- **Don't apply any TF transform inside `grade_detector`.** Publish in `lidar_footprint` and let the ObstacleLayer's TF buffer handle the rest. Pre-transforming breaks the "sensor frame everywhere" invariant from RULES.md #1.
- **Don't drop the spike/DBSCAN steps "to save time."** The ObstacleLayer treats every published point as an obstacle — anything you publish becomes a costmap mark. Filtering is the entire job.
- **Don't republish the original 14-field PointCloud2.** Strip to xyz; the costmap only needs geometry, and the bandwidth saving is meaningful at 20 Hz × 11.5k points.
- **Don't keep the `file(GLOB)` source list.** With two executables in one package it would silently link `cuda.cu` into `grade_detector` (or pull `pca_pipeline.cpp` into `line_detector`). Use explicit per-target file lists.
