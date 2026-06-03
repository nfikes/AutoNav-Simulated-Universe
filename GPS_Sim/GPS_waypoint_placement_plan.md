# GPS Waypoint Placement — ROS 2 Migration Plan

This document captures how the simulator in this repo (`src/gps_sim_gui.py`)
maps onto the AutoNav robot's ROS 2 stack so the existing
`send_GPS_waypoint.sh` becomes a thin shell wrapper around **one new
GPS handler node** that owns the magnetometer-less heading-offset
estimation and exposes lat/lon ↔ local-xy as ROS services. The shell
script reads command-line lat/lon, asks the node for the local goal,
and publishes it to NAV2 on a 1 Hz tick.

Source repo for the migration:
`/Users/nathanfikes/Projects/GitHub/AutoNav_25-26`,
branch `fix/behavior-tree-triggering`,
package `isaac_ros-dev/src/gps_waypoint_handler`.

> **Distinction to keep clear throughout this document:**
>
> | Marker | Meaning |
> |---|---|
> | **[REAL]** | Ships to / runs on the AutoNav robot |
> | **[SIM]** | Lives only in `gps_sim_gui.py` (stress-test, visualizer, ensemble tooling) |
> | **[PORTED]** | Originated in the simulator, must also live on the robot |
> | **[REUSE]** | Already exists in `gps_waypoint_handler/`, keep |
>
> The Lidar-Simulation pattern at
> `/Users/nathanfikes/Projects/Claude-Sandbox/Lidar-Simulation` ↔
> branch `feature/lidar-pca-grade-detection` follows the same
> convention; if a section here is ambiguous, fall back to the markers
> above.

---

## 1. Goal

Replace the one-shot conversion in
`isaac_ros-dev/config/send_GPS_waypoint.sh` (currently calls
`navsat_transform_node`'s `/fromLL` once and fires a `NavigateToPose`
goal) with a **GPS handler node + thin shell wrapper** that:

1. **[REAL]** Accepts a GPS waypoint (lat/lon, optional yaw) on the
   command line.
2. **[REAL] [PORTED]** Estimates the rotation between the robot's
   odom/SLAM frame and the geographic frame **without a magnetometer**,
   by fusing GPS with continuous odometry — using the simulator's
   3-state EKF + closed-form bootstrap + heading resync (§3, §10).
3. **[REAL]** Re-projects the GPS goal into the SLAM `map` frame and
   re-publishes it to NAV2 at **1 Hz** (NAV2's global planner can't
   meaningfully digest more — see §10.3).
4. **[REAL]** **Stops re-emitting** once the goal estimate has
   converged — `‖ekf_pos − goal_world‖ < k · σ_GPS` (default `k=2` →
   ≈ 0.6 m at σ = 0.3 m).
5. **[REAL]** **Self-corrects when convergence locks on the wrong
   spot** — a multipath-poisoned bootstrap fit can produce an EKF
   heading that is internally consistent but biased; the candidate-
   goal then converges to a wrong place and the agent drives there.
   A pre-EKF trip wire (*is the GPS distance to the goal
   increasing?*) lifts the candidate-goal envelope filter and fires
   a wider-window closed-form heading-resync, letting the algorithm
   reconverge from there. Information-only — no step-change for the
   controller. See §10.6.
6. **[REAL]** Reports through ROS topics when the robot reaches the
   1 m success ring around the true GPS waypoint.

The simulator (gps_sim_gui.py) exercises this exact loop end-to-end
with a calibrated GPS noise model (10 Hz, σ = 0.30 m, slow bias
drift, occasional outlier hops, rare dropouts) — calibrated against
the F9P log noted in §9.3.

---

## 2. Architecture — shell wrapper + GPS handler node

### 2.1 The shell wrapper (thin, **[REAL]**)

```bash
# isaac_ros-dev/config/send_GPS_waypoint.sh
#!/usr/bin/env bash
# Usage: ./send_GPS_waypoint.sh <lat> <lon> [radius_m]
LAT=$1; LON=$2; RADIUS=${3:-1.0}

# Single action call. --cancel-on-disconnect ensures Ctrl+C in the
# terminal cancels the action server-side (not just the CLI), so
# the handler stops re-publishing /goal_pose and NAV2 stops driving.
# --feedback streams the structured Feedback message at 2 Hz.
ros2 action send_goal /navigate_to_waypoint \
  gps_waypoint_handler/action/NavigateToWaypoint \
  "{goal_type: 0,
    target: {header: {frame_id: 'wgs84'},
             pose: {position: {x: $LON, y: $LAT, z: 0.0},
                    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}},
    success_radius_m: $RADIUS}" \
  --feedback \
  --cancel-on-disconnect
```

The shell script:
- Parses lat/lon (DMS or decimal — preserve the existing parser).
- Sends the action goal once.
- `--feedback` streams the structured feedback at the action's 1 Hz
  rate (distance to goal, EKF heading, GPS state, refinement-locked
  flag) directly to the operator's terminal.
- Action result is parsed automatically; the shell exits with code 0
  on `succeeded: true`, non-zero otherwise.

That's the entire shell side. Everything else lives in the node.

### 2.2 The GPS handler node (new, **[REAL] [PORTED]**)

Located at `isaac_ros-dev/src/gps_waypoint_handler/gps_waypoint_handler/gps_handler_node.py`.

**Why a node, not just a shell loop calling `/fromLL`?** Because
`navsat_transform_node`'s `/fromLL` requires either a magnetometer or
known starting orientation (currently `magnetic_declination_radians:
0.0`, `use_odometry_yaw: true` — accurate **only** when the robot is
manually started facing north). The simulator's value is precisely
that it estimates `θ_offset` live from observed motion. That estimator
needs a long-running node to maintain state.

**Hosts the `/navigate_to_waypoint` action** — one unified interface
for both GPS and local-coords goals, so a higher-level mission node
or BT can chain heterogeneous waypoint lists through a single client.
The action message is defined in §4 (`NavigateToWaypoint.action`).

When the action receives a goal, the handler routes by `goal_type`:

- `GOAL_TYPE_GPS` (`target.header.frame_id == "wgs84"`): runs the
  θ_offset estimation, converts lat/lon → local XY using
  `R(θ_offset)`, publishes `/goal_pose` at 1 Hz, monitors arrival.
- `GOAL_TYPE_LOCAL` (`target.header.frame_id == "map"` or `"odom"`):
  skips the rotation, publishes the local pose directly to
  `/goal_pose` at 1 Hz (still throttled for NAV2's planner),
  monitors arrival.

Same Feedback / Result message in both cases.

**Two-timer architecture.** Inside the handler, two ROS timers run:

- **EKF heartbeat (always-on)** — fires every `/odometry/filtered`
  callback at ~30 Hz. Runs predict + update + heading-resync. Keeps
  `θ_offset` warm regardless of whether an action goal is active.
  This is what guarantees the *first* action goal — even right
  after node startup — converges quickly.
- **`/goal_pose` republisher (gated)** — only ticks at 1 Hz **while
  an action goal is active**. Stops on `on_succeed` / `on_cancel` /
  `on_abort` so cancelled / completed goals don't keep driving NAV2.

**Concurrent-goal handling: preempt-with-cancel.** When a new
`send_goal` request arrives while another is active, the handler:
(1) calls `cancel()` on the prior `ServerGoalHandle` and returns
the prior action with `terminal_status = STATUS_PREEMPTED`,
(2) waits for the prior to terminate (so the 1 Hz publisher stops),
(3) accepts the new goal and restarts the publisher. Default
`rclpy_action` parallel execution is **wrong** for navigation — two
goals racing on `/goal_pose` would yo-yo NAV2's planner.

**Threading.** EKF state must be protected by a `threading.Lock`
because the action server runs in a `ReentrantCallbackGroup` (so
`cancel` can fire mid-execution) while the EKF callback + services
+ publish timer share a `MutuallyExclusiveCallbackGroup`. Without
the lock, the conversion services could read partial EKF state
mid-update.

**Subscribes (consumes):**

- `/gps_fix` (`sensor_msgs/NavSatFix`) — from
  `isaac_ros-dev/src/gps_handler/src/gps_publisher.cpp` **[REAL]**.
- `/odometry/filtered` (`nav_msgs/Odometry`) — from
  `robot_localization`'s `ekf_filter_node_odom` **[REAL]**.
- TF: `map → odom` from the SLAM stack (only if the converted goal
  needs to be in `map` frame for NAV2; otherwise odom-frame is
  enough).

**Publishes (produces):**

- `/goal_pose` (`geometry_msgs/PoseStamped`) — the live `map`-frame
  goal, refreshed at **1 Hz** (`NAV2_GOAL_HZ`). NAV2's existing
  behavior tree (`bt_nav.xml`, see §12) already has a `GoalUpdated()`
  hook that triggers replanning — no BT changes needed.
- `/gps_waypoint/heading_offset` (`std_msgs/Float64`) — live
  `θ_offset` in radians.
- `/gps_waypoint/heading_offset_std_deg` (`std_msgs/Float64`) — its
  1-σ uncertainty in degrees, for diagnostics.
- `/gps_waypoint/debug` (`std_msgs/String` JSON) — verbose debug
  stream mirroring the simulator's `agent.debug` (recovery counters,
  rejection counts, etc.). Separate from the action's structured
  feedback because it carries fields the action message doesn't need.

**Action server (offered):**

- `/navigate_to_waypoint` (`gps_waypoint_handler/action/NavigateToWaypoint`)
  — see §4 for the message definition. Standard ROS 2 action: goal /
  feedback / result, supports `--cancel-on-disconnect`.

**Services (offered):**

- `/gps_waypoint/gps_to_local` (custom srv) — `(lat, lon)` → `(x, y)`
  in the **odom** frame using the current `θ_offset` estimate. Useful
  for any other node that wants the live conversion.
- `/gps_waypoint/local_to_gps` (custom srv) — inverse.

### 2.3 Topic naming convention

`robot_localization`'s default names are kept as-is — they're
standard across the ROS ecosystem, and renaming them would require
editing `dual_ekf_navsat_params.yaml` plus every other consumer.
Topics the handler owns get descriptive names. The table below is
the single source of truth for what each topic carries.

**Topics:**

| Topic | Frame | Source | Trust level | Who owns it |
|---|---|---|---|---|
| `/gps_fix` | (lat/lon) | `gps_publisher` (NMEA reader) | **raw** sensor stream | already on robot |
| `/imu/data` | sensor | per-sensor IMU filter (Madgwick or chipset) | per-sensor filtered, fed to EKF | already on robot |
| `/odom` | `odom` | wheel encoders | **raw** odometry | already on robot |
| `/odometry/filtered` | `odom` | `ekf_filter_node_odom` (wheel + IMU) | **post-EKF**, GPS-free | already on robot |
| `/odometry/filtered_map` | `map` | `ekf_filter_node_map` (wheel + IMU + GPS) | **post-EKF**, GPS-fused | already on robot |
| `/odometry/gps` | `map` | `navsat_transform_node` (lat/lon → map) | post-conversion, **rotation may be wrong without our handler** | already on robot |
| `/goal_pose` | `map` | handler at 1 Hz | **output** to NAV2 / BT | **handler owns** |
| `/gps_waypoint/heading_offset` | radians | handler | **handler-side derived**, the live θ_offset estimate | **handler owns** |
| `/gps_waypoint/heading_offset_std_deg` | degrees | handler | uncertainty of the above (from windowed-fit spread) | **handler owns** |
| `/gps_waypoint/debug` | JSON | handler | verbose debug stream — mirrors `agent.debug` from the sim, supplements the action's structured feedback | **handler owns** |

**Action endpoints** (the goal interface; preferred over topics for
goal-driven flow because it gives built-in cancellation, structured
feedback, structured result, and is the standard way to chain
goals):

| Endpoint | Type | Owner | Purpose |
|---|---|---|---|
| `/navigate_to_waypoint` | `gps_waypoint_handler/action/NavigateToWaypoint` | handler | the goal interface — accepts both GPS (lat/lon) and local (x/y) goals via a discriminator field; chained sequentially by mission scripts and BTs |

**Service endpoints** (live lat/lon ↔ local-xy conversions for any
other node that needs them):

| Endpoint | Type | Owner | Purpose |
|---|---|---|---|
| `/gps_waypoint/gps_to_local` | custom srv `(lat, lon) → (x, y)` | handler | apply current `θ_offset` to a lat/lon, return odom-frame coords |
| `/gps_waypoint/local_to_gps` | custom srv `(x, y) → (lat, lon)` | handler | inverse of the above |

**Notes on what is *not* in the table** — these slots don't exist by
design, even if naming might suggest they should:

- `/gps/ekf_fixed` or similar — the EKF doesn't *output* a filtered
  GPS reading; it only *consumes* `/gps_fix`. The closest thing is
  `/odometry/filtered_map`, which is "where the EKF thinks the
  robot is after digesting GPS," but that's a pose, not a GPS
  position.
- `/imu/ekf_fixed` — the IMU describes orientation + angular rate,
  not XY. The EKF uses it to constrain the orientation component of
  the fused pose, but doesn't output a "filtered IMU" stream. If
  you need filtered orientation, that comes from a per-sensor IMU
  filter (Madgwick / Mahony) upstream of the EKF, on `/imu/data`.

**Topic naming in code/launch files**: prefer fully-qualified names
in the handler node (`/odometry/filtered`, not `~/odometry/filtered`)
so launch-time remappings are explicit and traceable. The handler's
own outputs are namespaced under `/gps_waypoint/...` so they're
trivially discoverable with `ros2 topic list | grep gps_waypoint`.

---

## 3. Algorithm (lifted directly from the sim)

### State

3-state EKF on `[x_world, y_world, θ_offset]` where `θ_offset` is the
unknown rotation from odom frame to geographic frame.

### Predict

For every odom message (or every control tick when there's no odom),
take the odom-frame delta `(Δx_o, Δy_o)` and roll the world position
forward:

```
x_w' = x_w + cos(θ) Δx_o − sin(θ) Δy_o
y_w' = y_w + sin(θ) Δx_o + cos(θ) Δy_o
θ'   = θ
```

Linearize around the current θ to update covariance. Process noise on
position is small (we trust odom); process noise on θ is small but
nonzero so the filter stays responsive to late corrections.

### Update

For every `/gps_fix`:

1. Convert lat/lon → local tangent meters around a fixed datum (the
   first GPS fix, or the SLAM `navsat_transform_node` datum).
2. Compute innovation `y = z_gps − [x_w, y_w]`.
3. Compute Mahalanobis² `y' S⁻¹ y` and reject if above gate
   (default χ² = 50 — calibrated in the sim against bias drift +
   outliers; tighter gates rejected too many normal samples).
4. Apply standard Kalman update on the 3-state.

### Bootstrap

The pure EKF can't escape a 180°-wrong cold start because the
linearization at the wrong θ misses the global flip. So while the robot
has traveled less than 5 m, run a closed-form weighted circular mean
of `atan2(Δgps) − atan2(Δodom)` over all sample pairs and forcibly
reseed the EKF's θ on every fix. After 5 m of travel, hand off — the
EKF refines from a near-truth linearization point.

This is the fix for seed-3-style failures we saw in the simulator's
smoke tests.

### Goal projection

The robot's current best estimate of where the GPS waypoint sits in
the SLAM `map` frame is:

```
goal_in_map = T_map_odom · R(−θ_offset) · (goal_world − ekf_pos_world)
```

The action server publishes this to NAV2 as a `PoseStamped` whenever
either:

- A new GPS fix arrives, **or**
- The EKF's θ has shifted by more than ~1° since the last publish.

### Stop refining (the "drop re-placing as we get closer" rule)

Two stop conditions, OR'd together:

- **Distance gate.** Once `‖ekf_pos − goal_in_world‖ < k · σ_GPS`
  (default `k = 2`, ≈ 0.6 m at σ = 0.3 m), GPS noise dominates the
  re-projection and re-emitting just makes NAV2's local controller
  chatter. Hold the last published goal.
- **Convergence gate.** Once consecutive replan goals move by less
  than `σ_GPS / √n_samples` (the EKF's predicted innovation
  magnitude), the cloud has converged.

When either condition trips, set `refinement_locked = true` in
feedback and let NAV2's local planner finish the approach.

### Self-correction (the "lift the envelope when going wrong way" rule)

The envelope-filtered candidate goal converges directly onto the real
goal in the dominant case (see §10.6 numbers). But ~0.3 % of
`--crazy` runs hit a failure mode where the EKF heading lock is
internally consistent but biased (a multipath-poisoned closed-form
fit), and the candidate parks at the *wrong* place. The standard
heading-resync watchdog can't catch this because the closed-form
fit on the agent's recent motion gives the same wrong answer as the
EKF.

Two information-only signals — both computed from quantities the
agent already has (its own GPS, its own odom, the GPS goal) — break
the lock without any controller-side effect:

1. **Moving-away detector.** Track `‖GPS_pos − GPS_goal‖` over a
   3 s sliding window. If the agent is *farther* from the goal at
   the end of the window than at the start by more than 1 m, the
   heading lock the candidate depends on is wrong (the candidate is
   pointing the agent the wrong way). Action: suspend the 1/r
   envelope filter for 4 s. With the envelope lifted, raw candidate
   updates flow into the smoother through its EWMA + 5 m snap, so
   the next EKF correction propagates without being gated as an
   outlier.

2. **Force-resync.** Same trigger fires a wider-window closed-form
   heading fit (500 samples ≈ 50 s, baseline ≥ 3 m, diff > 20°). If
   the fit yields a θ that disagrees with `EKF.θ` by more than 20°,
   snap `EKF.θ` to the new value with `σ_θ = 10°`. This succeeds
   precisely in the cases the standard 100-sample / 2 m / 10° resync
   misses — the wider window accumulates real baseline even for
   limit-cycling agents, and the higher diff threshold keeps the
   noisy small-baseline fits from injecting wrong corrections.

Pre-EKF trip wire wording matters: the moving-away signal does not
consult the EKF state to decide whether the EKF is wrong. It uses
raw GPS only, so it can flag a biased EKF without circularly
trusting it. Estimator-side, ships per Rule 7.

The success criterion for the action result is the same as the
simulator's: the robot's true position (read off SLAM) is within the
configured `success_radius_m` (default 1 m) of the goal.

---

## 4. File layout

Audit of the actual branch state on `fix/behavior-tree-triggering`:

```
isaac_ros-dev/src/gps_waypoint_handler/
├── action/                            # [NEW] action interface generation
│   └── NavigateToWaypoint.action      #   unified GPS / local-coords goal
├── srv/                               # [NEW] live conversion endpoints
│   ├── GpsToLocal.srv                 #   (lat, lon) → (x, y) in odom
│   └── LocalToGps.srv                 #   inverse
├── gps_waypoint_handler/
│   ├── gps_conversions.py             # [REUSE] calculate_distance() and
│   │                                  #   apply_heading_offset() are kept
│   │                                  #   verbatim for the new node's
│   │                                  #   local-tangent math
│   ├── get_gps_positioning.py         # [REUSE] stationary lat/lon
│   │                                  #   averager — useful for datum
│   │                                  #   bootstrap, leave it
│   ├── gps_waypoint_bringup.py        # [REPLACE] hardcoded reference
│   │                                  #   coords + heading=-134.8°,
│   │                                  #   one-shot autonomous-mode
│   │                                  #   listener — superseded by the
│   │                                  #   shell + handler-node design
│   ├── waypoint_commander.py          # [REPLACE] file-based
│   │                                  #   BasicNavigator.followWaypoints,
│   │                                  #   superseded by the action server
│   │                                  #   inside the new node
│   └── gps_handler_node.py            # [NEW] [REAL] [PORTED] — the
│                                      #   continuous-refinement node:
│                                      #   /navigate_to_waypoint action
│                                      #   server, 3-state EKF, recoveries
│                                      #   from simulator §10.2, 1 Hz
│                                      #   /goal_pose publish loop, the
│                                      #   two conversion services
├── package.xml                        # [MODIFY] deps to add:
│                                      #   action_msgs, rclpy_action,
│                                      #   tf2_ros, nav2_msgs,
│                                      #   rosidl_default_generators,
│                                      #   rosidl_default_runtime
├── setup.py                           # [MODIFY] add console_script
│                                      #   entry for gps_handler_node;
│                                      #   drop the dead `tester_publisher`
└── CMakeLists.txt                     # [NEW] needed for action + srv
                                       #   interface generation
```

### `NavigateToWaypoint.action` definition

```
# isaac_ros-dev/src/gps_waypoint_handler/action/NavigateToWaypoint.action
#
# Unified action — handles both GPS goals (lat/lon) and local goals
# (x, y in map or odom frame) via the goal_type enum + a single
# PoseStamped target field. Idiomatic ROS 2: no NaN sentinels, the
# discriminator is explicit, and `header.frame_id` carries the
# coordinate system the same way every other NAV2 action does.
# Mission scripts and BTs chain heterogeneous waypoint lists through
# this single action.

# ─── Goal ───

# Discriminator
uint8 GOAL_TYPE_GPS    = 0
uint8 GOAL_TYPE_LOCAL  = 1
uint8 goal_type

# Single target. Interpretation depends on goal_type:
#   GOAL_TYPE_GPS   — header.frame_id = "wgs84"
#                     pose.position.x  = longitude (°)
#                     pose.position.y  = latitude  (°)
#                     pose.position.z  = altitude  (m, optional, ignored if 0)
#                     pose.orientation = desired final yaw (or identity = auto)
#   GOAL_TYPE_LOCAL — header.frame_id = "map" or "odom"
#                     pose.position    = target XY
#                     pose.orientation = desired final yaw (or identity = auto)
geometry_msgs/PoseStamped target

float64 success_radius_m         # default 1.0; ≥ 50 % footprint inside

---

# ─── Result ───

# Terminal status — controlled vocabulary, machine-readable. Use
# `failure_reason` for free-form addenda (NAV2-passed-through errors,
# specific TF frame names, etc.) but `terminal_status` is the
# single-source-of-truth for what happened.
uint8 STATUS_SUCCESS            = 0
uint8 STATUS_CANCELED           = 1
uint8 STATUS_PREEMPTED          = 2   # superseded by a newer goal
uint8 STATUS_ABORTED            = 3   # generic abort
uint8 STATUS_TIMEOUT            = 4
uint8 STATUS_GPS_LOST           = 5   # /gps_fix stale > N seconds
uint8 STATUS_TF_STALE           = 6   # map→odom missing
uint8 STATUS_EKF_NOT_CONVERGED  = 7   # action arrived before bootstrap
uint8 STATUS_NAV2_REJECTED      = 8   # NAV2 itself rejected the goal
uint8 STATUS_INVALID_GOAL       = 9   # malformed goal_type / frame_id
uint8 STATUS_GOAL_OUTSIDE_COSTMAP = 10
uint8 terminal_status

bool    succeeded                # convenience: terminal_status == STATUS_SUCCESS
float64 final_distance_m
float64 final_heading_err_deg
float64 final_latitude           # robot's lat at terminal moment (computed)
float64 final_longitude
float64 peak_theta_offset_std_deg  # worst-case heading uncertainty during run
float64 distance_traveled_m
uint32  ekf_updates
uint32  gps_outlier_rejects
uint32  heading_resyncs_fired
float64 elapsed_s
string  failure_reason           # free-form, optional addendum

---

# ─── Feedback (streamed at 2 Hz — faster than goal_pose's 1 Hz so
# the operator's terminal feels live; goal_pose stays at 1 Hz to
# spare NAV2's planner) ───

float64 distance_to_goal_m
float64 ekf_theta_deg
float64 ekf_theta_std_deg
geometry_msgs/PoseStamped current_goal_in_map
bool    gps_connected
bool    refinement_locked
```

### `send_GPS_waypoint.sh` becomes:

```bash
#!/usr/bin/env bash
# Continuously-refining GPS waypoint goal.
# Usage: ./send_GPS_waypoint.sh <lat> <lon> [yaw_deg] [radius_m]
LAT=$1; LON=$2; YAW=${3:-NaN}; RADIUS=${4:-1.0}

# Single action call — blocks until succeeded / aborted, with live
# feedback streamed to the operator's terminal. Ctrl+C cleanly
# cancels the action.
ros2 action send_goal /navigate_to_waypoint \
  gps_waypoint_handler/action/NavigateToWaypoint \
  "{latitude: $LAT, longitude: $LON, yaw_deg: $YAW, success_radius_m: $RADIUS}" \
  --feedback
```

The DMS / decimal parser stays in the shell for ergonomics; otherwise
the wrapper is genuinely thin — every algorithmic decision lives in
the handler node.

### What gets deleted

- `tester_publisher` console_scripts entry — points at no module on
  the branch, dead code.
- `current_declination` parameter in `gps_conversions.py::apply_heading_offset` —
  accepted but never subtracted, dead code.
- Hardcoded `-134.8°` heading in `gps_waypoint_bringup.py` — replaced
  by the live EKF estimate.
- `stored_waypoints.txt` file I/O dance — replaced by in-memory state.

---

## 5. Mapping: simulator concept → real-robot concept

| Simulator concept | Real-robot location | Status |
|---|---|---|
| `GPSEKF` class (`gps_sim_gui.py:643`) | `gps_handler_node.py:GpsEkf` (verbatim port, numpy only) | **[PORTED]** |
| `GPSWaypointSim._tick_gps()` | `/gps_fix` ROS subscriber callback | **[PORTED]** |
| `_bootstrap_theta()` (closed-form fit) | `gps_handler_node.py::bootstrap_theta()` | **[PORTED]** |
| `_closed_form_theta_window()` (sliding-window resync) | `gps_handler_node.py::closed_form_theta_window()` | **[PORTED]** |
| Mahalanobis gate (`EKF_GATE_CHI2 = 50`) | Same constant; tune against on-vehicle data | **[PORTED]** |
| EKF lock-in recovery (§10.2) | Same constant; same logic | **[PORTED]** |
| EKF position-variance floor (§10.2) | Same constant; same logic | **[PORTED]** |
| Heading resync (§10.2) | Same threshold (10°), cooldown (3 s), magnitude-ratio filter | **[PORTED]** |
| Candidate-goal smoother (§10.6.1: EWMA α = 0.15 + 5 m snap) | Same constants; ports verbatim into the handler's `intermediate_goal_world()` | **[PORTED]** |
| 1/r envelope filter (§10.6.2: `d_env = max(0.4, 0.5·L/r)`, K = 4) | Same constants; rejects raw candidates whose distance to the GPS goal escapes the envelope | **[PORTED]** |
| Moving-away detector (§10.6.3: 3 s window / 1 m threshold / 4 s suspension) | Same constants; pre-EKF trip wire on raw GPS-vs-goal distance | **[PORTED]** |
| Force-resync from moving-away (§10.6.4: 500-sample window / 3 m baseline / 20° diff) | Same constants; aggressive heading-resync that succeeds where the §10.2 standard one can't | **[PORTED]** |
| Live candidate-goal `intermediate_goal_world()` | Computed every `/gps_fix`; published as `visualization_msgs/Marker` for RViz | **[PORTED]** |
| `published_goal_world` (1 Hz throttle) | `/goal_pose` publish timer at `NAV2_GOAL_HZ = 1.0` | **[PORTED]** |
| Convergence lock (`‖ekf_pos − goal‖ < k·σ`) | Boolean stored in node; gates re-publish | **[PORTED]** |
| `gps_conversions.py::calculate_distance()` | (already in package) | **[REUSE]** |
| `gps_conversions.py::apply_heading_offset()` | (already in package) | **[REUSE]** |
| `get_gps_positioning.py` (stationary averager) | (already in package) | **[REUSE]** |
| `windowed_astar` / LOS shortcut / corridor mask | NAV2's planner server (Smac, NavFn) | **[REAL] [N/A in node]** |
| Chaplygin sleigh dynamics | NAV2's local controller (DWB / RPP / TEB) | **[REAL] [N/A in node]** |
| `GPSWaypointGUI` (matplotlib) | RViz with the markers above | **[N/A]** |
| `agent.debug` dataclass | `/gps_waypoint/debug` JSON topic (verbose) AND the action's structured `Feedback` message at 1 Hz (`distance_to_goal_m`, `ekf_theta_deg`, `ekf_theta_std_deg`, `gps_connected`, `refinement_locked`) | **[PORTED]** |
| Final stats line printed at end of a `--single` run | The action's `Result` message: `final_distance_m`, `final_heading_err_deg`, `ekf_updates`, `heading_resyncs_fired`, `elapsed_s`, `failure_reason` | **[PORTED]** |
| Operator launches `gps_sim_gui.py --single` and waits for arrival | Operator runs `send_GPS_waypoint.sh` which calls `/navigate_to_waypoint` action with `--feedback` and waits for `succeeded` | **[PORTED]** |
| Mission script chains multiple `set_goal()` calls | Mission node sends a list of `NavigateToWaypoint` action goals; the action's discriminator field handles GPS / local without two clients | **[NEW] [REAL]** |
| **The 8 GPS hazards** (jammers, spoofers, foliage, cycle slips, noise bursts, transient on/off rolls, projector multipath, roof blackouts) | **None — the real robot doesn't have these** | **[SIM]** |
| `--crazy` mode + 10 000-agent ensemble + `bake_gif.py` | **None — sandbox stress-test tooling** | **[SIM]** |
| 3×3 m goal-cam follow-window | RViz can be configured with an equivalent zoom on the published yellow X marker, but not required | **[SIM]** |
| `--scatter` random-start positions | **None — only one robot** | **[SIM]** |
| `random_goal()` placement constraints | None on robot — operator picks the lat/lon | **[SIM]** |
| `RULES.md` | Carries through as a behavioral contract; the node enforces Rule 1 by reading only `/gps_fix` and `/odometry/filtered` | **[PORTED]** |

---

## 6. Risks / open questions

- **Datum coherence.** Two layers compute lat/lon ↔ local-meters:
  `navsat_transform_node` (for the SLAM EKF) and our handler node
  (for goal projection). Both should anchor on the same datum — the
  first valid `/gps_fix` after node start — so the goal lines up
  with NAV2's costmap. If our node and `navsat_transform_node`
  disagree, the goal will drift by the datum offset.
- **TF latency.** Publishing `/goal_pose` in `map` frame requires a
  fresh `map → odom` transform; SLAM hiccups make the goal jump.
  Use a TF buffer with ≥ 0.5 s timeout and skip the publish on
  transform failure rather than emit a stale pose.
- **GPS rate vs publish rate.** Per repo audit (§9.2), the recorded
  log at the path the original plan cited shows median Δt ≈ 2.3 s
  on the field robot, but the chipset can do 10 Hz and the URDF
  advertises 10 Hz. The handler node must treat **any** GPS rate
  from 0.5 to 10 Hz as supported — its 1 Hz `/goal_pose` publish is
  an output-side throttle, decoupled from input rate.
- **`navsat_transform_node` coexistence.** The dual-EKF SLAM stack
  already runs `navsat_transform_node`. The handler node does not
  replace it — `navsat_transform_node` continues to provide
  `/odometry/gps` to the SLAM EKF. Our node reads the same `/gps_fix`
  and produces a different output (the live `θ_offset` plus the
  rate-limited `/goal_pose`).
- **Multi-waypoint missions.** The handler takes one target at a
  time. Chaining is the BT's job — `bt_nav.xml` can sequence
  multiple `/gps_waypoint/target` publishes via its existing
  recovery / mission nodes. **[REAL]**
- **CSV log path.** The original plan cited
  `AutoNav-GUI-Standalone/example-playback-csv/t000_20260427_185211/`
  as the calibration source. That path **does not exist on
  `fix/behavior-tree-triggering`** — it must have lived on a
  different branch or in a sibling repo. The simulator's `--real`
  calibration values are still anchored to the ZED-F9P
  characteristics from §9; if a fresh log is recorded, retune
  `GPS_NOISE_STD` and `GPS_BIAS_AMPL_M` in `_apply_real_overrides()`
  accordingly.

- **`bt_nav.xml` location uncertain.** §12 claims that the active
  production tree at `bt_nav.xml` already has the `GoalUpdated()` /
  `ReactiveFallback` hooks the handler relies on. Two repo audits
  disagreed: an earlier scan reported `bt_nav.xml` present; a later
  scan said only `gradient_escape.xml` (a plugin def, not a tree)
  was findable on `fix/behavior-tree-triggering`. **Verify in
  person before deployment**: locate the active BT XML being
  loaded by `bt_navigator`, confirm `GoalUpdated()` is in it, and
  if not, either (a) author a small replacement tree that has the
  hook, or (b) drop the topic-driven approach and have the handler
  send goals via NAV2's `/navigate_to_pose` action client instead.
  Plan B is a bigger code change but safer if the BT layer can't
  be modified in time.

---

## 7. Test plan

- Unit-test `gps_ekf.py` with synthetic odom + GPS streams matching
  the simulator (deterministic seeds), verify σ_θ converges below 1°
  inside 5 m of travel.
- Replay the `t000_20260427_185211` CSV against the action server in a
  ros2 bag context, verify the refined goal pose tracks the actual
  robot path.
- Field test (preferably the same site at 37.23027 N, 80.42504 W)
  with a known surveyed waypoint.
- Disable the magnetometer path entirely on the robot during testing
  to confirm the action server is the only source of GPS heading
  knowledge.

---

## 8. First commit checklist

- [ ] Create `action/NavigateToWaypoint.action` with the goal /
      result / feedback layout from §4
- [ ] Create `srv/GpsToLocal.srv` and `srv/LocalToGps.srv`
- [ ] Add `CMakeLists.txt` with `rosidl_generate_interfaces` for the
      action and the two services
- [ ] Update `package.xml` deps:
      `action_msgs`, `rclpy_action`, `tf2_ros`, `nav2_msgs`,
      `rosidl_default_generators` (build), `rosidl_default_runtime`
      (exec), and the `<member_of_group>rosidl_interface_packages</member_of_group>`
      entry
- [ ] Write `gps_handler_node.py`:
  - [ ] 3-state `GpsEkf` class (predict + update + Mahalanobis gate)
  - [ ] `bootstrap_theta()` (closed-form, weighted circular mean)
  - [ ] `closed_form_theta_window()` (sliding window for resync)
  - Recoveries from §10.2 (one checkbox per mechanism):
    - [ ] EKF lock-in recovery (`EKF_REJ_STREAK_RESET = 25`
          consecutive rejections → force-accept + re-inflate
          `P[0,0]/P[1,1]`)
    - [ ] EKF position-variance floor (`EKF_POS_VAR_FLOOR = 1.0²`)
    - [ ] Continuous heading resync (`HEADING_RESYNC_THRESHOLD_DEG = 10°`,
          3 s cooldown, sliding window of last 100 GPS samples)
    - [ ] Magnitude-ratio filter on heading-fit pairs (rejects
          spoofer-pinned and jam-degraded samples; ratio outside
          `[1/3, 3]` is dropped)
    - [ ] A* failure fallback (lethal start/goal cell snap; only
          relevant if the handler ever does its own grid search,
          otherwise N/A — NAV2 owns A*)
    - [ ] Blackout reconnect (sim concept of "open a 5 s window of
          fresh GPS on roof entry"; on the real robot this maps to
          handling an expected-but-recovered GPS dropout — leave as
          a no-op until field tests show it's needed)
  - Self-correcting candidate-goal mechanism from §10.6 (one
    checkbox per layer):
    - [ ] Candidate-goal smoother — EWMA (α = 0.15) on accepted raw
          candidates, with a 5 m hard-snap bypass for big jumps;
          `intermediate_goal_world()` returns the smoothed value
    - [ ] 1/r envelope filter — `d_env = max(0.4, 0.5·L/r)`, reject
          if `d_raw > 4·d_env`; dormant for `r < 3 m`
    - [ ] Moving-away detector on raw `‖GPS − GPS_goal‖` — 3 s
          window, 1 m threshold, 4 s envelope-off window; estimator-
          side only, no controller side effects (no smoother reset,
          no forced replan)
    - [ ] Force-resync triggered by moving-away — 500-sample window,
          baseline ≥ 3 m, snap only if `|new θ − ekf.θ| > 20°`,
          post-snap `σ_θ = 10°`
  - [ ] **Two-timer architecture**:
    - [ ] EKF heartbeat callback on `/odometry/filtered`
          (`MutuallyExclusiveCallbackGroup`)
    - [ ] `/goal_pose` republisher timer at `NAV2_GOAL_HZ = 1.0`
          (gated on `_active_goal_handle is not None`)
    - [ ] `Feedback` publisher timer at 2 Hz (gated similarly)
  - [ ] `/navigate_to_waypoint` **action server** that:
    - [ ] Accepts goals via `goal_type` enum + `PoseStamped`
    - [ ] Validates `frame_id` matches `goal_type` (rejects with
          `STATUS_INVALID_GOAL` otherwise)
    - [ ] **Preempt-with-cancel** on concurrent goals: cancel prior
          handle, wait for terminal, accept new
    - [ ] Lives in a `ReentrantCallbackGroup` so cancel can interrupt
    - [ ] On terminal (success / cancel / abort): stop the
          `/goal_pose` republisher *before* returning the cancel-
          accepted response (anti-pattern §13 #13)
    - [ ] Returns full `Result` with `terminal_status` enum +
          `failure_reason` addendum
  - [ ] Convergence-lock gate (`‖ekf_pos − goal‖ < k·σ_GPS`) —
        gates re-publish, *not* arrival
  - [ ] EKF-state thread safety: `threading.Lock` around all reads /
        writes of `ekf_pos`, `θ_offset`, `θ_offset_std`
  - [ ] `/gps_waypoint/heading_offset`, `/gps_waypoint/heading_offset_std_deg`
        diagnostic publishers (always-on; not gated by active goal)
  - [ ] `/gps_waypoint/debug` verbose JSON publisher (mirrors
        `agent.debug` from the sim, useful for offline analysis)
  - [ ] `gps_to_local` / `local_to_gps` service handlers (same
        callback group as EKF; share the lock)
- [ ] Update `setup.py` — add `gps_handler_node` console_script,
      drop the dead `tester_publisher` entry
- [ ] Rewrite `send_GPS_waypoint.sh` to call
      `ros2 action send_goal /navigate_to_waypoint ... --feedback`
- [ ] Smoke-test in simulation (`gps_sim_gui.py --real --single`)
- [ ] Field-test at the surveyed waypoint
- [ ] Verify the existing `bt_nav.xml` `GoalUpdated()` hook fires on
      each 1 Hz `/goal_pose` re-publish (no BT changes expected, but
      confirm)
- [ ] Add a 5-line mission-script example showing how to chain
      heterogeneous waypoints (one GPS, one local, one GPS) through
      the single action client — drop into
      `isaac_ros-dev/src/gps_waypoint_handler/examples/`

---

## 9. GPS hardware: real-world calibration

This section captures what the AutoNav robot's GPS *actually does* in
the field, so the simulator's noise model and the action server's
filter constants stay anchored to reality instead of drifting into
academic correctness.

### 9.1 Receiver: u-blox ZED-F9P (ELT0156 module)

Per the spec sheet
(https://gnss.store/products/elt0156?variant=55851756945740):

| Spec | Value |
|---|---|
| Engine | 184-channel u-blox F9 |
| Constellations | GPS L1C/A + L2C + L5, GLONASS L1OF/L2OF, Galileo E1B/C + E5a/E5b, BeiDou B1I/B2I/B2a, QZSS L1C/A + L1S + L5, SBAS L1C/A, NavIC L5 |
| Update rate (chipset cap) | up to 20 Hz with RTK |
| Convergence | cold start 24 s · hot start 2 s · RTK lock < 10 s |
| Tracking sensitivity | −167 dBm |
| RTK accuracy | 0.01 m + 1 ppm CEP |
| SPARTN accuracy | 0.06 m horizontal, 0.12 m vertical |
| Standalone accuracy | not stated; typical ~30–100 cm CEP |

### 9.2 Operating mode in this codebase: **STANDALONE** (no corrections)

`isaac_ros-dev/src/gps_handler/src/gps_publisher.cpp` reads NMEA over
USB-serial at 38400 baud. Repo audit confirms:

- **No NTRIP client.**
- **No RTCM input wiring.**
- **No SPARTN subscription.**
- **No UBX-CFG configuration** sent — the receiver runs in whatever
  default mode it boots into.
- Only `$GNGGA` / `$GPGGA` sentences are parsed; everything else is
  dropped (which is what limits the *effective* rate to ≪ chipset cap).

The receiver is therefore in **standalone mode**, with SBAS (WAAS /
EGNOS) implicitly active because that's a u-blox factory default. No
other corrections are applied to the field robot.

Position covariance arrives downstream via HDOP:
`horizontal_variance = hdop²`. For typical good-sky HDOP 0.7–1.5 that
maps to σ ≈ 0.7–1.5 m, which is consistent with an F9P running
standalone+SBAS.

### 9.3 Calibration source for the simulator

Recorded GPS log at
`AutoNav-GUI-Standalone/example-playback-csv/t000_20260427_185211/`.
The simulator's `--real` mode is tuned to match what the log
**actually** shows on a clean sky F9P+SBAS, not a worst-case
conservative envelope. There are two operating points: the global
defaults (used by `--random` and the scripted scenario) which keep
the older conservative numbers, and `--real` (set inside
`_apply_real_overrides()` in `gps_sim_gui.py`) which tightens to the
recorded values:

| Constant | Default / `--random` | `--real` (matches log) | What the log shows |
|---|---|---|---|
| `GPS_SAMPLE_HZ` | 10 | 10 | chipset cap; effective rate ≈ 0.45 Hz due to non-GGA NMEA filter |
| `GPS_NOISE_STD` | 0.30 m | **0.10 m** | <10 cm stationary jitter |
| `GPS_BIAS_AMPL_M` | 0.5 m | **0.20 m** | slow ±20 cm wobble over ~1 min |
| `GPS_OUTLIER_STD` | 6 m | **4 m** | ~5 m occasional hop |
| `GPS_OUTLIER_PROB` | ≈ 0.0005 / sample | same | ~1 outlier per 200 s |
| `GPS_DROPOUT_HZ_PER_S` | 0.01 | same | ~1 multi-second dropout per 100 s |

In `--real` mode, the GPS visualization scatter cloud sits in a
~30 cm 95 % spread around truth — visually consistent with what a
stationary F9P+SBAS receiver actually does in clear sky. The default
mode keeps a more conservative ~1 m spread for stress testing the
EKF / heading-resync against unmodelled drift.

### 9.4 Magnetometer status: confirmed absent

`slam/config/dual_ekf_navsat_params.yaml`:

- `magnetic_declination_radians: 0.0`
- `use_odometry_yaw: true`

Confirms the no-magnetometer operating model that this entire plan and
the simulator are built around. Heading offset is recovered solely
from observed GPS-vs-odom direction over distance — exactly what
`gps_ekf.py` will do.

### 9.5 Tuning guidance if the publisher is ever wired for corrections

If `gps_publisher.cpp` is later updated to ingest RTCM / SPARTN /
NTRIP corrections, retune both the simulator and the action server's
EKF expectations:

| Mode | `GPS_NOISE_STD` | `GPS_BIAS_AMPL_M` | `EKF_GPS_SIGMA` | Notes |
|---|---|---|---|---|
| **Standalone (current field)** | 0.30 m | 0.5 m | 1.2 m | implicit SBAS, what the field robot runs today |
| SBAS only (explicit config) | 0.20 m | 0.3 m | 0.8 m | tighter than implicit, requires UBX-CFG verification |
| SPARTN | 0.05 m | 0.05 m | 0.20 m | requires subscription + DPT firmware |
| RTK (network or local base) | 0.01 m | negligible | 0.05 m | requires NTRIP client + base station |

`--real` mode in `gps_sim_gui.py` already keeps the **standalone**
defaults — appropriate for field deployment. `--crazy` adds
adversarial RF (jammers, spoofers, multipath leaks, cycle slips) that
do **not** model normal AutoNav conditions; it's a robustness stress
test, not a realism mode.

### 9.6 Sim hazards available for testing the action server

`gps_sim_gui.py` exposes the following GPS hazards that the action
server's EKF + heading-resync should survive (all are off in `--real`
unless explicitly listed in CLI flags):

| Hazard | What it models |
|---|---|
| Roof / blackout | full GPS dropout under structures |
| Projector triangle | fixed multipath bias near building corners |
| Hex jammer | sparse fixes inside a region (RF jammer) |
| Foliage | elevated noise under canopy |
| Spoofer | adversarial RF pinning GPS to a fake location |
| Cycle slip | persistent multi-second offset (receiver phase loss) |
| Noise burst | transient ionospheric / GDOP excursion |
| Transient on/off | per-tick probability that a hazard briefly disables |

Each has a corresponding recovery mechanism on the agent side
(blackout reconnect window, heading resync, EKF lock-in recovery,
P-floor, magnitude-ratio filter on the heading fit). When porting
into the action server, port the recoveries too — they are the
difference between 99.96 % and ~16 % arrival in the
`--crazy --agents 1000` smoke runs.

---

## 10. Algorithmic findings from large-scale stress testing

This section records the recoveries that were added to the simulator
after Section 3 was written. Each one fixed a distinct failure class
observed in 2,000+ agent-trial sweeps; **all of them must be ported
to the action server** along with the base EKF or arrival rates fall
back to the level we started at.

### 10.1 Headline numbers

`gps_sim_gui.py --crazy --agents N`, all hazards (jammers, spoofers,
cycle slips, noise bursts, projector multipath, roof blackouts) at
maximum intensity:

| Stage | Arrived |
|---|---|
| Initial repo state | ~16 / 1000 |
| LOS shortcut + replan throttle | 161 / 1000 |
| Bootstrap fix + speed bump | 661 / 1000 |
| Live candidate-goal at path end | 834 / 1000 |
| Blackout reconnect | 942 / 1000 |
| EKF lock-in recovery + P-floor | 999 / 1000 |
| Continuous heading resync | 1000 / 1000 |
| + magnitude-ratio filter | 1000 / 1000 |
| + transient hazards | 1000 / 1000 |
| **20-seed × 500-agent sweep** | **9 996 / 10 000 (99.96 %)** |
| **`--real` 10-seed × 200 sweep** | **2 000 / 2 000 (100 %)** |

The ~84-percentage-point delta from "base EKF" to "everything ported"
is the cost of skipping any individual recovery.

### 10.2 Recoveries that must be ported alongside `gps_ekf.py`

#### EKF lock-in recovery
After enough consecutive Mahalanobis rejections (`EKF_REJ_STREAK_RESET = 25` ≈ 2.5 s at 10 Hz), force-accept the next reading and re-inflate `P[0,0]/P[1,1]` to `EKF_GPS_SIGMA²`. Without this, the filter can lock onto a confident-but-wrong estimate and gate every correcting fix forever. Symptom: agent's EKF position pinned to a phantom location while every fresh GPS reading is rejected.

#### EKF position-variance floor
Clamp `P[0,0]/P[1,1]` to a minimum of `(1 m)²` in `predict()`. Real GPS has irreducible m-scale variation; claiming sub-decimeter certainty is fiction and produces the "parked just outside the goal ring at sub-cm/s" symptom. With the floor, GPS keeps pulling the EKF toward truth at a useful rate.

#### Continuous heading resync
After bootstrap completes, the EKF's Kalman gain on θ shrinks toward zero. If bootstrap converged on a multipath-poisoned heading, that wrong θ is permanent and the agent **orbits the goal** because every world-frame command rotates by `heading_err`. Fix: every GPS tick, recompute the closed-form heading from a sliding window of the last ~10 s of GPS-vs-odom pairs; if it disagrees with the EKF's θ by more than `HEADING_RESYNC_THRESHOLD_DEG = 10°` (with cooldown), snap θ to the closed-form value and re-widen `θ_var` to `(5°)²` so the EKF resumes refining. In simulator tests, `_heading_resync_count` averaged ~4 firings per agent across 10 000 trials.

#### Magnitude-ratio filter on the heading fit
Reject GPS-vs-odom pairs from the heading fit when `|Δgps| / |Δodom|` is outside `[1/3, 3]`. These are spoofer-pinned readings (GPS doesn't move while odom does → ratio → 0) or jam-degraded samples (GPS displacement is mostly noise). Without this filter, the closed-form fit converges to a heading biased *toward the spoofer's lie*; with it, those pairs are dropped and the fit recovers from clean motion alone.

#### Blackout reconnect window
On the False→True edge of "robot is geometrically inside a roof", open a 5 s window of fresh GPS even while shadowed (cooldown 20 s before another can fire). On the real robot this models the moment the antenna catches a window or sky-glance; the algorithmic point is that blackouts must be **recoverable** rather than "the EKF dead-reckons until the agent exits."

#### A* failure fallback
If A* is asked to plan from a cell that's currently lethal (rare but happens when EKF noise nudges the plan anchor into an inflated obstacle), snap to the nearest non-lethal cell within an 8-cell spiral; if even that fails, set `_stuck_until = sim_time + 1.0 s` to back off rather than spinning. NAV2's planner has its own equivalents but the action server should still bound failure modes.

### 10.3 NAV2 goal-publication rate

Sending NAV2 a fresh `PoseStamped` every GPS tick (10 Hz) thrashes
its global planner. The simulator now models this:

| Constant | Value | Meaning |
|---|---|---|
| `NAV2_GOAL_HZ` | 1.0 | how often `published_goal_world` updates |
| `intermediate_goal_world()` | every step | live candidate, free; not published |
| `published_goal_world` | every 1 / NAV2_GOAL_HZ s | what A\* / the controller drive toward |
| First-publish offset | random in `[0, 1/HZ]` per agent | desyncs an N-agent ensemble |

In ROS, this maps to **rate-limit the `nav2_msgs/NavigateToPose` goal-update publisher to 1 Hz**, even when the EKF's belief has shifted. The internal candidate goal can keep updating at the EKF rate; only the *commitment* is throttled. Concretely:

- Action server's EKF runs at the GPS callback rate (≈ 10 Hz expected, ~0.45 Hz actually-observed-on-the-robot).
- A "publish" timer fires at 1 Hz, snapshots the current candidate, and re-sends the action goal to NAV2's `NavigateToPose` if the snapshot has shifted by > the min-action-resend threshold.

The "stops re-emitting" gate from §1 still applies — once `‖ekf_pos − goal_in_world‖ < k · σ_GPS`, hold the last published goal regardless of the timer.

### 10.4 Robot kinematics: non-holonomic constraint

The simulator's body has been switched from a holonomic point-mass
to a **Chaplygin sleigh**: the body can only apply force along its
own heading axis (the knife edge) and a moment about its center.
This is realistic for a ground-vehicle and exposes a behavior the
holonomic version masked — the body has to **rotate before it can
drive** toward a target.

Implications for the action server:

- NAV2's local controller (DWB / RPP / TEB) already enforces a
  non-holonomic constraint via `Differential` or `Ackermann` motion
  models, so this point doesn't change the action server design —
  but it does mean the simulator is now a **faithful** test bed.
  Trajectories the simulator clears will clear on the real robot;
  trajectories the simulator orbits on (heading-bootstrap-poisoned)
  the real robot will also orbit on.
- `MAX_ANGULAR_VEL = 1.5 rad/s` (≈ 86 °/s, 1.2 s for a 180° turn) is
  consistent with most wheeled outdoor robots and matches what NAV2
  parameter files default to.
- The pure-pursuit driver in the simulator scales forward thrust by
  `cos(heading_err)` so the body decelerates / stops rather than
  drifting sideways when the target is off-axis. This is roughly
  what NAV2's RPP controller does; the simulator and the real
  controller should converge toward similar trajectories.

### 10.5 Stuck-class taxonomy (and the recovery that fixed each)

| Class | Symptom | Root cause | Recovery |
|---|---|---|---|
| A | Off-map / orbiting around goal | Bootstrap converged on multipath-poisoned heading; EKF locked on wrong θ | §10.2 continuous heading resync + magnitude-ratio filter |
| B | Stopped 1–10 m short of goal | EKF position biased; controller's `dist_to_goal` collapses to zero before truth-arrival | §10.2 EKF P-floor + drop `MIN_SEARCH_SPEED` after bootstrap |
| C | Off-map | Initial candidate-goal computed off-map (heading_offset_est = 0 rotates goal by full true_heading) | replan-on-goal-drift (was bootstrap-gated; ungated, with `REPLAN_GOAL_DRIFT_M_BOOT = 15 m`) |
| D | Frozen at path's planned end while live candidate kept drifting | Lookahead targeted stale `path[-1]` while `dist_to_goal` was computed off the live candidate | When `best_i ≥ n - 1`, target the live candidate goal directly |
| E | Limit-cycle oscillation around a biased EKF goal | `MIN_SEARCH_SPEED = 0.4 m/s` floored speed at zero `dist_to_goal`; direction vector noise drove the wobble | After bootstrap, drop the floor and let the P-controller taper naturally |
| F | Compound-zone trap (roof × foliage overlap) | Reconnect gave noisy fixes once per blackout entry; cooldown was 20 s; agent dead-reckoned a multi-second blackout | Per-tick transient "off" rolls (5 % roof, 20 % jammer, 30 % spoofer) so a stuck agent gets periodic clean fixes |

This is the order issues were discovered and fixed. The action server
will encounter A and B first (they're the failure modes that survive
basic operation); C–F only show up under maximally-adversarial GPS
and may not need to be handled in production unless field tests
surface them.

### 10.6 Self-correcting candidate-goal mechanism **[PORTED]**

Layered on top of the §3 algorithm. The §10.2 recoveries fix the EKF
itself; this section fixes how the candidate goal — the rotated
projection of the GPS waypoint into the robot's local frame — gets
*derived* from the EKF and *output* to NAV2. Without these layers the
candidate goal jitters on every GPS-noise wiggle (lever-arm
amplification scaled by `|goal − ekf_pos|`) and, more importantly, can
*lock onto the wrong spot* when the EKF heading is biased — a failure
mode that the §10.2 recoveries can't catch because they're internal
EKF mechanisms and the candidate-goal output is downstream.

All four mechanisms are estimator-side and ship per Rule 7.

#### 10.6.1 Candidate-goal smoother (EWMA + 5 m snap)

Raw candidate goal:

```
raw = ekf.pos + R(true_θ − θ_offset_est) · (goal_world − ekf.pos)
```

This is a "moving-carrot" form (steady-state fixed point at
`ekf = goal`, regardless of small residual heading bias). Anchoring
on the live `ekf.pos` rather than the spawn point gives the
self-correcting steady state, but it also feeds 10 Hz GPS-driven EKF
position noise into the candidate scaled by the goal lever arm. The
smoother dampens that:

| Constant | Value | Meaning |
|---|---|---|
| `CANDIDATE_SMOOTH_ALPHA` | 0.15 | EWMA gain — small-step time constant ≈ 0.6 s |
| `CANDIDATE_SNAP_M` | 5.0 m | step-detect threshold; bypasses EWMA for big jumps |

Per-tick: if `‖raw − last_smoothed‖ > SNAP`, replace verbatim
(heading resync, A* re-target). Otherwise EWMA-track. The snap
threshold is what keeps the bootstrap → resync transient (a 50–100 m
drop in one tick) from lagging a few seconds behind.

`intermediate_goal_world()` — the public accessor — returns the
smoothed value. The 1 Hz `published_goal_world` (the actual `/goal_pose`
output) is a sample of the smoothed candidate.

#### 10.6.2 1/r envelope filter (your `θ ∝ 1/r` insight)

Heading observability scales with travel distance: a single GPS
sample's angular precision is `σ_θ ≈ σ_GPS / r` where `r` is travel
since spawn. The candidate-goal distance from the real goal is then
bounded by

```
d_env(r, L) = max(d_floor, GAIN · L / r)
```

with `r` = cumulative odom distance, `L` = current robot-to-goal
distance. Both quantities are agent-local (Rule 1 conformant). A raw
candidate that escapes `K · d_env` is treated as an outlier and
dropped — the smoother holds the previous value.

| Constant | Value | Meaning |
|---|---|---|
| `CANDIDATE_ENV_GAIN_M` | 0.5 m | ≈ σ_GPS lateral noise |
| `CANDIDATE_ENV_FLOOR_M` | 0.4 m | irreducible noise floor |
| `CANDIDATE_ENV_REJECT_K` | 4.0 | reject if `d_raw > K · d_env` |
| `CANDIDATE_ENV_MIN_R_M` | 3.0 m | filter dormant for `r < MIN_R` (bootstrap) |

The `MIN_R` guard keeps the filter dormant during bootstrap so the
legitimate 50–100 m candidate jump at the heading-resync isn't
gated. Once the agent has moved 3 m+, the filter is the primary
defense against multipath-driven candidate jumps.

#### 10.6.3 Moving-away detector → envelope suspension

Failure case the envelope filter alone cannot solve: the EKF locks
onto a biased heading, the closed-form refit gives the *same* biased
heading (because the agent's actual motion is consistent with the
wrong θ), and the candidate-goal sits at a wrong location indefinitely.

Pre-EKF trip wire that catches it:

```
trigger:  ‖GPS_pos(t) − GPS_goal‖  −  ‖GPS_pos(t − W) − GPS_goal‖  > Δ
action:   _envelope_suspended_until = sim_time + S
```

| Constant | Value | Meaning |
|---|---|---|
| `MOVING_AWAY_WINDOW_S` | 3.0 s | sliding window for the trip wire |
| `MOVING_AWAY_THRESHOLD_M` | 1.0 m | net delta over window; +ve = farther |
| `MOVING_AWAY_ENV_SUSPEND_S` | 4.0 s | envelope-off window after trigger |

While suspended, the smoother's normal EWMA + 5 m snap still runs —
just without the envelope-rejection layer — so corrections from a
fresh heading-resync (or the force-resync below) flow through to the
published candidate cleanly, no abrupt step-change for NAV2's
controller.

The trip wire uses raw GPS (`/gps_fix`) and the input GPS goal — *not*
the EKF — so it can flag a biased EKF without circularly trusting it.

#### 10.6.4 Force-resync triggered by moving-away

The standard `_maybe_resync_heading` (§10.2) requires a 2 m baseline
over a 100-sample window and a 10° disagreement with `EKF.θ`. Agents
in a heading-poisoned limit cycle don't accumulate that baseline —
they oscillate in a small bounded region — so the standard resync
never fires for them. The same moving-away trigger that suspends the
envelope also kicks an aggressive resync attempt with relaxed
windowing:

```
def force_heading_resync():
    bs_θ, baseline = closed_form_theta_window(
        n_samples=HEADING_FORCE_RESYNC_WINDOW,
        min_baseline=HEADING_FORCE_RESYNC_MIN_BASELINE_M)
    if bs_θ is None:
        return False                        # no baseline yet
    diff = wrap_pi(bs_θ − ekf.θ)
    if abs(diff) < HEADING_FORCE_RESYNC_DIFF_DEG:
        return False                        # change too small to act
    ekf.reset_theta(bs_θ,
        theta_var=radians(HEADING_FORCE_RESYNC_VAR_DEG)²)
    return True
```

| Constant | Value | Meaning |
|---|---|---|
| `HEADING_FORCE_RESYNC_WINDOW` | 500 samples (~ 50 s) | wider than standard's 100 — accumulates baseline for limit-cyclers |
| `HEADING_FORCE_RESYNC_MIN_BASELINE_M` | 3.0 m | conservative; sub-3 m fits are noisy and must be rejected |
| `HEADING_FORCE_RESYNC_DIFF_DEG` | 20° | only snap on substantial disagreement; small-correction noise wouldn't reach 20° |
| `HEADING_FORCE_RESYNC_VAR_DEG` | 10° | post-snap σ_θ — wide enough that subsequent EKF updates can refine |

Iter-by-iter empirical tuning was needed (see §10.6.6) — both the
baseline and the diff threshold matter:
- baseline = 0.5 m gave noisy fits that sometimes snapped to wrong
  values (regressed arrival from 9964 → 9959 / 10000)
- baseline = 3.0 m + diff > 20° hits 9972 / 10000

#### 10.6.5 Stuck-detector with forward-thrust override **[SIM]**

(Listed for completeness; **does not ship**.) The simulator's
pure-pursuit controller has a limit-cycle failure mode where the
body sweeps past the candidate goal, alignment goes negative, and
forward thrust collapses while angular thrust stays saturated. A
sim-side stuck-detector (track GPS progress over 4 s; on no-progress,
override the post-bootstrap zero-floor speed cap with 0.8 m/s for
2 s) breaks the cycle for the GIF visualization. NAV2's local
controller has its own equivalents (TEB/RPP/DWB recovery
behaviors) so this is **not** ported.

#### 10.6.6 Headline numbers — 10k-agent `--crazy` headless

The recoveries above were tuned over an iterated empirical sweep on
`--crazy --agents 10000 --seed 7` (the hardest seed in our test
matrix). Cumulative arrival rates:

| Configuration | Arrived |
|---|---|
| Pre-mechanisms (envelope filter only, candidate jumping) | 9821 / 10000 (98.21 %) |
| + smoother (EWMA + 5 m snap) | 9964 / 10000 (99.64 %) |
| + moving-away → suspend envelope | 9964 / 10000 (no change without resync) |
| + force-resync, baseline = 0.5 m (iter 1) | 9959 / 10000 (regressed — noisy fits) |
| + force-resync, baseline = 3.0 m, diff > 20° (iter 2) | 9971 / 10000 |
| + envelope `K = 4` (iter 3) | **9972 / 10000 (99.72 %)** |
| same config, seed 42 (independent confirm) | 9998 / 10000 (99.98 %) |
| same config, `--real --scatter --agents 30000` | **30000 / 30000 (100 %)** |

The `--real --scatter` 30k run is the deployment-relevant figure:
realistic outdoor GPS hazards (no jammers / spoofers / cycle slips —
those are `--crazy`-only adversarial conditions), agents scattered
across the map. Arrival is 100 %.

The remaining 28 misses on `--crazy` seed 7 are dominated by two
sub-modes that no estimator-side fix can reach without controller
side-effects:
- ~10 with `|herr| < 5°` but `d > 20 m` — heading was eventually
  fixed, but too late; the agent drove the wrong way long enough to
  run out of sim-time-budget before getting back. NAV2 on the real
  robot has different time budgets and different recoveries; these
  are sim-bound.
- ~10 with `|herr| > 100°` and tightly-bounded motion — limit-cycle
  motion below the 3 m baseline floor; closed-form can't fit
  reliably, and a "Hail Mary" 180° flip was tested (iter 6) and
  regressed arrival by injecting bad flips on agents that had only
  briefly diverged.

Per Rule 7, simulator artifacts in adversarial scenarios that don't
represent the real robot are out of scope; the estimator side has
been pushed to its useful limit.

---

## 11. Tools and artifacts produced during stress testing

- `scripts/bake_gif.py` — headless GIF baker. Renders an N-agent
  ensemble in any mode (`--real` / `--crazy` / `--random`) to a
  PIL-compressed GIF without needing a display. Used to produce
  the visual benchmarks: `crazy_10000.gif`, `crazy_10000_chaplygin.gif`,
  `real_10000_chaplygin.gif` at the repo root.
- `agent.debug` — per-agent dataclass split into `self_view` (what
  the onboard code can see — Rule 1 conformant) and `true_view`
  (sim-only ground truth, for diagnosing stuck agents). Printable
  with `print(agent.debug)`. The action server should expose an
  equivalent diagnostic topic so on-vehicle issues can be inspected
  the same way — `gps_waypoint/debug` (`std_msgs/String` JSON, or a
  custom message type) at low rate.
- 3×3 m goal-cam in `--single` mode tracks the published yellow
  waypoint and auto-expands when the live candidate is far from
  truth. Useful for visually confirming heading-resync is firing.
- `--scatter` flag — spawns each agent at a random valid start
  position instead of clustering at origin. Combines with `--real`
  for outdoor-realistic ensemble tests.

---

## 12. Behavior-tree integration **[REAL]**

The branch `fix/behavior-tree-triggering` already has the BT plumbing
the GPS handler node needs. **No BT changes are required for the
1 Hz goal re-publish to take effect.**

### 12.1 Active production tree: `bt_nav.xml`

Already wires:

- `GoalUpdated()` — fires on every change to `/goal_pose`. The
  handler node's 1 Hz publish is exactly what this hook is
  designed to consume.
- `ReactiveFallback` — runs the global planner + local controller in
  a loop with recovery branches. Re-publishing the goal at 1 Hz
  triggers a fresh `ComputePathToPose` without restarting the tree.
- Custom plugins `GoalBender` and `GradientEscape` (in
  `custom_behavior_tree_plugins/`):
  - **`GoalBender`** — bends a goal that lands behind the robot to a
    forward-facing approach. Operates at the BT layer, on the
    `map`-frame `PoseStamped` we publish. **Complementary** to our
    handler — it solves a different problem (NAV2 controllers
    struggle with reverse-facing goals) and our handler doesn't
    duplicate it.
  - **`GradientEscape`** — costmap-gradient stuck recovery. Fires
    when the local planner can't make progress. Again,
    complementary; our handler's recoveries fire on
    *GPS-estimation* problems, this fires on *navigation* problems.

### 12.2 Where the handler node plugs in

```
/gps_fix ──┐
           ├──► gps_handler_node ─┐
/odom    ──┘                      ├──► /goal_pose (1 Hz) ──► bt_nav.xml ──► NAV2 ──► motors
                                  │                            │
/gps_waypoint/target ─────────────┘                            ├──► GoalBender (if goal is behind)
                                                                ├──► ComputePathToPose
                                                                ├──► FollowPath
                                                                └──► ReactiveFallback
                                                                       └──► GradientEscape
```

The handler node feeds the *input* to the existing tree. The
BT/NAV2 stack handles motion, recoveries, and stuck-detection on
the navigation side. The two recovery domains do not overlap.

---

## 13. What NOT to do **[REAL]**

Anti-patterns surfaced by the audit. Following the Lidar-Simulation
template (which has a similar block).

1. **Don't replace `navsat_transform_node`.** The SLAM stack uses
   it for `/odometry/gps`. Keep it running; the handler node lives
   alongside it, not in its place.

2. **Don't trust `/fromLL` for the live goal projection.**
   `navsat_transform_node` requires either a magnetometer or a
   known starting heading. The whole point of the handler node is
   that the robot has neither. `/fromLL` is fine for the SLAM
   bootstrap or one-off conversions; it is **not** the answer for
   the live goal stream.

3. **Don't publish `/goal_pose` faster than `NAV2_GOAL_HZ = 1.0`.**
   NAV2's global planner can't keep up; you'll thrash A* /
   Smac. The simulator §10.3 makes this explicit.

4. **Don't compute the candidate goal once and reuse it.** The
   live candidate is a function of `ekf_pos` and `θ_offset_est`,
   *both* of which update every GPS fix. Compute it every step,
   sample it at 1 Hz for publication. Storing a single
   `current_goal_world` causes the orbit / oscillation failure
   modes from §10.5.

5. **Don't graduate from bootstrap on `θ_std < 3°` alone.** The
   simulator's earlier criterion was unreachable because
   `reset_theta` slammed `θ_var` to 15° every tick. Use a
   baseline-dependent variance (`σ_θ = σ_GPS / baseline`) and
   graduate on `odom_dist > 5 m` AND `bs_baseline > 5 m` (see
   §10.5 Class C).

6. **Don't drop the magnitude-ratio filter in the heading fit.**
   Without it, GPS samples taken inside a multipath zone (or near
   a building face) will skew the closed-form fit toward the bias
   direction. §10.2 explains; in the field this would manifest as
   "robot heads in a slightly wrong direction even with good GPS
   for the rest of the run."

7. **Don't store waypoints to disk.** The existing
   `gps_waypoint_bringup.py` + `waypoint_commander.py` pattern
   writes converted goals to `stored_waypoints.txt` — leftover
   from when the package was a one-shot batch converter.
   Replaced by in-memory state in the new node.

8. **Don't write the 8 simulator hazards into the production
   handler.** Jammers / spoofers / cycle slips / noise bursts /
   foliage / projectors / roof blackouts / transient on-off rolls
   are **[SIM]**-only. The recoveries that handle them
   (heading resync, lock-in recovery, P-floor, magnitude-ratio
   filter, blackout reconnect) are **[PORTED]** because they
   handle GPS pathologies that occur naturally in the field too,
   just at lower intensity.

9. **Don't rebuild the BT.** `bt_nav.xml` already has
   `GoalUpdated()`. Plug into it rather than authoring a new tree.
   `GoalBender` and `GradientEscape` solve navigation problems,
   not GPS-estimation problems — leave them alone.

10. **Don't rename the existing package.** The handler node lives
    inside `gps_waypoint_handler/`. Other parts of the AutoNav
    launch graph reference it by name; renaming breaks them.

11. **Don't expose two separate actions for GPS vs local goals.**
    `NavigateToWaypoint.action` is one unified action with a
    discriminator field (`latitude = NaN` → use `local_x/local_y`).
    A mission node that wants to chain a heterogeneous list — GPS
    waypoint, local waypoint, GPS waypoint, return-to-start in
    odom — uses one client, one feedback subscription, one result
    parser. Splitting into two actions doubles every mission's
    glue code for no benefit.

12. **Don't use bare topics for the goal interface.** An earlier
    iteration of this plan had the shell publish to
    `/gps_waypoint/target` and block on `/gps_waypoint/status`.
    That works but loses cancellation, structured feedback, and
    the standard ROS goal-state machine. The action server is
    cheap (~40 lines extra) and the right shape for goal-driven
    operation. Use it.

13. **Don't republish `/goal_pose` while an action is canceling.**
    On `cancel_goal_callback`, stop the 1 Hz publisher *before*
    returning the cancel-accepted response. Otherwise NAV2's BT
    sees a stale goal and may start driving toward a cancelled
    waypoint between the cancel signal and the next replan tick.

14. **Don't anchor the candidate-goal projection on the spawn
    point.** It is geometrically the agent's odom-frame projection
    of the GPS goal back into world coords — but the resulting form
    `R(ε)·(goal − spawn) + spawn` has a *fixed* fixed-point that
    isn't the goal, so any residual heading bias `ε` becomes an
    `ε × lever_arm` permanent offset (≈ 1.3 m at 60 m lever and
    1.2°). Use the live-EKF anchor `R(ε)·(goal − ekf) + ekf`; its
    fixed point is `ekf = goal`, which gives self-correcting
    convergence regardless of small residual `ε`. (See §10.6.1.)

15. **Don't reset the candidate smoother on moving-away.** An
    earlier iteration of §10.6.3 also called `_smoothed_candidate
    = None` on the trip wire; on the real robot this would make the
    next published `/goal_pose` step-jump to the raw EKF value,
    which NAV2's controller sees as a "weird movement" command.
    Keep the trip wire information-only — *only* lift the envelope
    and (optionally) fire the force-resync. Let the smoother's
    EWMA + 5 m snap handle convergence smoothly.

16. **Don't use a tight baseline floor on the force-resync.** The
    closed-form heading fit on `< 1.5 m` of motion is dominated by
    GPS angular noise (~σ_GPS / baseline ≈ 30°). Iter-1 testing
    with `min_baseline = 0.5 m` *regressed* arrival from 9964 →
    9959 / 10000 because the noisy fits snapped to wrong values.
    The current 3 m baseline + 20° diff gate combination is the
    iterated empirical sweet spot — relaxing either hurts.

17. **Don't publish a forced 180° heading flip on persistent
    moving-away.** Tested in iter 6 as a Hail Mary for tightly-
    bounded heading-flipped agents that the closed-form fit can't
    catch (because their motion is below the baseline floor). It
    fires in too many cases where heading was actually mostly
    correct — an agent dodging an obstacle for a few seconds
    triggers moving-away, the flip injects a wrong heading, and
    the EKF then has to recover from a worse state. Regressed
    arrival from 9972 → 9959 / 10000. Skip flips; let the closed-
    form do its work when it can, accept the residual failure
    rate when it can't.

---

# Real Robot vs Simulation — Reconciliation Notes (2026-05-09)

## Why this section exists

After a day of field testing the AutoNav robot with the
`feature/self-correcting-gps-waypoint` branch, **the dominant
failure mode is the robot consistently converging to the *wrong*
GPS point** — not the user-supplied target, but a biased nearby
location. The robot does drive purposefully and stops cleanly; it
just stops in the wrong place. This section maps that failure
back to architectural differences between the simulator (which
converges reliably) and the real-robot pipeline.

Two important context items first:

1. **The Jetson was on a stale branch for part of the day.**
   `fix/gps_connection` (an old branch with none of the day's
   self-correction fixes) was the active checkout when several
   "robot drove 43–48 m off" runs were observed. Those runs do
   *not* validate the latest pipeline — only the runs after
   switching to `feature/self-correcting-gps-waypoint` do. Any
   reasoning below should distinguish "regression caused by the
   self-correcting code" from "pre-existing bug from the old
   pipeline that the self-correcting code was meant to address."

2. **Wrong-point convergence is the priority.** Smooth motion to
   a wrong point means the controller, action server, EKF
   *predict*, and TF chain are working. The error has to be in
   the EKF *update* (or its inputs) producing a biased
   position/heading that the candidate-goal projector then
   faithfully tracks.

## Architectural parity (what is the same on both)

The simulator and the real robot share the core algorithmic
spine. The following are not the source of the difference:

- 3-state EKF on `[x_world, y_world, θ_offset]` with odometry
  predict and GPS update.
- Closed-form bootstrap of `θ_offset` via weighted circular mean
  of `atan2(Δgps) − atan2(Δodom)` over an accumulating baseline
  (5 m on both sides).
- EWMA candidate smoother with a 1/r envelope filter.
- "Moving-away" detector firing on a 3 s sliding window of
  GPS-distance-to-goal.
- Force-resync action: closed-form refit on a wider window with
  σ_θ re-inflation when a detector fires.
- Goal projection that pivots around the *EKF position* (not the
  spawn / map origin), so the projected candidate has its
  fixed-point at the true goal as `ekf_pos → goal`.

If the algorithm spine is the same and the simulator agents
converge while the real robot does not, the divergence has to be
in (a) what the EKF *consumes* (sensor model), (b) what the EKF
*locks into* (variance floor / gating), or (c) what consumes the
EKF *output* (frame chain into NAV2). The next sections walk each.

## A. Sensor model differences

| Aspect | Sim | Real robot |
|---|---|---|
| Wheel odometry noise | **Zero** — odometry delta is the ground truth | ~1–2% per-metre slip + IMU drift; correlated with terrain |
| GPS noise model | σ = 0.30 m white + slow sinusoidal bias (0.5 m amplitude, 60 s period) + 0.5%-per-sample 6 m outliers + structured projector zones | Antenna multipath near buildings/trees, near-DC bias from antenna placement, occasional UART byte corruption (now mitigated in `gps_publisher.cpp`) |
| GPS rate | 10 Hz constant | ~0.5–10 Hz, depends on constellation/multipath |
| Common-cause error between odom and GPS | None | Possible — both can be biased by the same physical setup (e.g. on a slope, encoders over-read AND GPS multipath leans) |

The closed-form heading bootstrap is the most fragile consumer of
this difference. Its weighted circular mean on
`atan2(Δgps) − atan2(Δodom)` assumes **odometry is a clean
ruler**. With ~1–2% per-metre wheel slip over a 5 m baseline,
each pair contributes ~5–10 cm of odometry-axis error against a
~30 cm GPS-axis error → the angular fit is already noisy. With
*correlated* errors (slope, slip on a turn), the fit is also
*biased*. The simulator never sees this regime.

## B. State estimation guards present in sim but absent on the real robot

These are the most likely "smoking guns" for wrong-point
convergence. Each one is implemented in the simulator and is
either missing or weaker on the real robot.

### B1. Position variance floor (`EKF_POS_VAR_FLOOR`)

- **Sim**: After every update, `P[0,0]` and `P[1,1]` are clamped
  to ≥ `1.0² m²` (i.e. σ ≥ 1.0 m). Code in `gps_sim_gui.py`
  around lines 1024–1035 and 2411–2420.
- **Real robot**: No analogous floor in `gps_ekf.py`.
- **Failure mechanism**: Without the floor, after enough samples
  with consistent (biased) GPS, the EKF position covariance
  shrinks below the actual bias magnitude. The Kalman gain on
  position approaches zero, the EKF *stops* trusting new GPS,
  and it locks onto a biased position. The candidate-goal
  projector then faithfully tracks that biased position, and the
  robot drives to and "succeeds" on the wrong point. This
  matches the observed failure exactly.
- **Recommended fix**: Add a position-variance floor to
  `gps_ekf.py update()`, mirroring the sim. Tune the floor to
  the empirically observed multipath bias magnitude (probably
  0.5–1.0 m).

### B2. Mahalanobis lock-in escape

- **Sim**: After 25 consecutive Mahalanobis rejections, force-
  accept the next reading and re-inflate `P[0,0]`, `P[1,1]` to
  `max(current, 1.2² m²)`.
- **Real robot**: Mahalanobis gate exists but no escape clause.
- **Failure mechanism**: If the EKF locks to a wrong position
  (B1 chain above), every subsequent GPS reading at the *true*
  position is far enough from the EKF mean to fail the gate.
  Updates stop entirely; the EKF can never escape the lock-in.
- **Recommended fix**: Mirror the sim's escape — count
  consecutive rejections, force-accept after a threshold, and
  re-inflate position covariance.

### B3. Periodic heading refit

- **Sim**: Every 3 s post-bootstrap, refit `θ` via closed-form
  on the last 100 GPS samples (~10 s window). If the refit
  disagrees with the EKF's `θ` by >10°, snap and re-widen
  `σ_θ` to 5°.
- **Real robot**: Force-resync only fires when a detector
  triggers (moving-away or local-vs-world divergence). No
  unconditional periodic refit.
- **Failure mechanism**: A small-but-persistent heading bias
  (say 5–8°) can be below the moving-away threshold (1 m over
  3 s) but still produce ~10–20 cm of cross-track error per
  metre of forward progress. Over a 50 m goal, that's 5–10 m of
  wrong-direction settling — exactly the magnitude of "near but
  not right." The real robot has no mechanism to self-correct in
  this regime.
- **Recommended fix**: Add a periodic refit timer. The sim's
  cadence (every 3 s) is the empirical sweet spot per
  `RULES.md`.

## C. Frame chain on the real robot

The simulator runs in a single direct frame: world (truth) and
odom (robot-local). A `θ_offset` transform connects them. There
is no map frame, no TF tree, no NAV2 plugin chain.

The real robot has, *as designed*:

```
utm  ── (gps_transform / navsat_transform_node)
 │
 ├── map  ── (slam_toolbox publishes map→odom)
 │    │
 │    └── odom  ── (ekf_local fuses /odom + IMU → /local_ekf/odom)
 │         │
 │         └── base_link  ── (robot_state_publisher)
```

But `slam.launch.py` lines 170–172 currently have **`ekf_global`,
`gps_transform`, and `nav2` commented out**. So in the deployed
launch:

- No `/odometry/filtered` (was the ekf_global topic).
- No `utm → map` transform from navsat_transform.
- No NAV2 stack run by *this* launch (a separate launch may run
  it; the field testing showed multiple NAV2 instances, so this
  needs to be sorted out).

The `gps_handler_node`'s consumer side was patched in commit
`cadcc33b` to subscribe to `/local_ekf/odom` instead of
`/odometry/filtered`. That fixes the *input* — the EKF gets odom
deltas — but it leaves the *output* path ambiguous: the
republished `goal_pose` to NAV2 needs to be in the frame NAV2 is
configured to consume (typically `map`). With no
`gps_transform`, there is no clean way to express the
GPS-derived goal in the `map` frame. This is a strong candidate
for "robot drives smoothly to a consistent wrong point" — the
goal is being sent to NAV2 in the wrong frame, and NAV2 plans a
valid path to that wrong location.

**Recommended fix**: Resurrect `ekf_global` and `gps_transform`
in `slam.launch.py`, OR rework the `gps_handler` to emit goals
in the frame NAV2 actually subscribes to (with the appropriate
TF lookup at publish time). Either way, the frame mismatch must
be resolved before B1/B2/B3 fixes can be evaluated cleanly.

## D. What the simulator has that the real robot deliberately doesn't (don't port)

These are sim-only conveniences that should *not* be added to
the real robot — they were either rejected in sim testing or are
flagged as "not representative" in `RULES.md`:

- **Stuck-detector forward-thrust override** (sim lines
  ~2078+). Per `RULES.md` Rule 7, the real robot is meant to
  rely on controller design + heading estimation alone.
- **180° heading flip Hail Mary**. The sim tested this at iter
  6 and *regressed* arrival from 9972 → 9959 / 10000. See
  anti-pattern 17 above. Skip.

## E. Tuning that's already different and probably correct

- `SUCCESS_RADIUS_M`: sim 1.0 m, real 0.25 m. Tightening makes
  sense for a real-world fielded vehicle, *but only after* B1
  (variance floor) is in place; with a biased EKF position, a
  tight radius just makes it harder to "succeed" on the wrong
  point.
- `max_vel_x`: sim 2.24 m/s, real 0.55 m/s. Slower is safer
  during this debugging phase — leave at 0.55.
- Bootstrap baseline: 5 m on both. May want to extend on the
  real robot (10 m?) given noisier odometry, but that delays
  initial convergence and the sim's choice is well-tuned.

## F. Preempt / state-machine bugs that don't exist in sim

The simulator runs a single physics loop and has no
action-server lifecycle. The real robot's `gps_handler_node` is
a `rclpy_action.ActionServer` and has had multiple
preempt-related defects today (e.g. `goal_handle.canceled()`
called from EXECUTING state → `InvalidStateTransition`). These
are now fixed (see commit `63681c91`), but they were a confound
during earlier field runs and should not be conflated with the
EKF / convergence story when reasoning about test results.

## Priority order for fixing wrong-point convergence

1. **Add EKF position variance floor** (B1). Lowest risk,
   highest expected payoff for the observed failure mode. Mirror
   the sim's `EKF_POS_VAR_FLOOR = 1.0 m²` (σ ≥ 1.0 m).
2. **Resolve the frame chain** (C). Without this, no controlled
   field test can distinguish "EKF is wrong" from "goal is in
   the wrong frame." Either uncomment `ekf_global` and
   `gps_transform` in `slam.launch.py` and verify they actually
   come up clean, or change where `gps_handler` publishes goals.
3. **Add Mahalanobis lock-in escape** (B2). Defense in depth
   against the bias-lock-in regime; cheap to add.
4. **Add periodic closed-form heading refit** (B3). Catches the
   residual sub-detector heading bias.
5. **Re-validate on real hardware**, on the *correct* branch,
   with the actual fixes deployed (not the stale
   `fix/gps_connection` chain).

## What this comparison does not address

- IMU calibration on the real robot. The sim has no IMU — the
  real robot fuses IMU into `/local_ekf/odom`. If the IMU bias
  is large or the calibration is stale, the bootstrap heading
  fit inherits that error. Worth a separate audit.
- Hardware bring-up failures (Arduino symlink, lidar
  reachability, X-button). These are orthogonal to the
  GPS-convergence story but block end-to-end testing.
- Whether the wheel encoder scale factor is correct. A 1–2%
  systematic odometry scale error would directly bias the
  closed-form heading fit and would manifest as exactly this
  kind of consistent wrong-point convergence. Calibrate over a
  known straight-line run before assuming the EKF is the
  problem.
