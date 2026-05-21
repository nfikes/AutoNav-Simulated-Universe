# Local EKF sensor expansion — KISS-ICP + ZED IMU

**Status:** parked. Use later when wheel-only odometry on the Local EKF is
no longer adequate (slippy terrain, encoder drift, single-IMU yaw bias).
**Target repo:** `AutoNav_25-26/isaac_ros-dev` (Jetson, ROS 2 Humble).
**Target filter:** `ekf_filter_node` (the Local EKF) defined in
`isaac_ros-dev/src/slam/config/ekf_local.yaml`. Map EKF / SLAM
unaffected.

This plan covers two sensor additions that travel together because their
error sources are mutually independent and they share the same EKF
wiring + validation flow:

1. **KISS-ICP** — scan-to-scan lidar odometry from the SICK multiScan
   pointcloud, fused as a second linear-velocity source.
2. **ZED IMU** — the camera's onboard IMU, fused as a second yaw-rate
   source alongside the SICK lidar IMU. Has a TF prerequisite that
   `ekf_local.yaml` already calls out.

Either can land alone, but doing them in one pass means tuning the
EKF gates once, validating once, and reasoning about covariance
shrinkage holistically.

---

## 1. Why these two sensors

The Local EKF (`robot_localization/ekf_node`, `world_frame: odom`) is the
filter Nav2 controllers latch onto. It must stay **jump-free**. Today its
inputs are:

| Sensor | Topic | Fields fused |
|---|---|---|
| Wheel encoders | `/odom` | `vx, vy, vyaw` |
| SICK lidar IMU | `/multiScan/imu` | `vpitch, vyaw` |

That gives the EKF exactly **one** linear-velocity source (wheels) and
**one** yaw-rate source (lidar IMU). Both have known failure modes that
take down the *only* observation of their dimension at the same time.

### KISS-ICP — second linear-velocity source

When the wheels slip, both the encoder velocity and the encoder yaw rate
go bad simultaneously. The lidar IMU helps with yaw but cannot recover
linear velocity. KISS-ICP gives a third, **independent** velocity
estimate derived from the lidar pointcloud — scan-to-scan ICP, no
learned weights, no map dependency. It's the lightest "lidar odometry"
option that fits the Jetson budget while staying jump-free (it does not
do loop closure — that's the Map EKF / SLAM job). Error source:
geometric correspondence between consecutive clouds. Independent of
wheel encoders, IMUs, and GPS.

### ZED IMU — second yaw-rate source

A single IMU's yaw bias drift is unobservable to the Local EKF — the
filter can't distinguish "the world is rotating slowly" from "this gyro
is drifting." Today, if the SICK IMU's bias wanders, encoder yaw is the
only thing pulling it back, and encoders are exactly the source we
distrust during slip. The ZED2i's onboard IMU sits on a different
physical chip with different temperature curves, mount flex, and bias
characteristics; fusing both yaw rates lets the EKF cross-check them
and tighten yaw covariance legitimately. **Linear acceleration from
either IMU is intentionally not fused** — ground robots gain nothing
useful from accel and pick up a lot of mount-vibration noise.

The map-frame correction path (slam_toolbox `/pose` → Map EKF) is
untouched. Both new sensors feed **only** the Local EKF.

---

## 2. Package choice (KISS-ICP)

The ZED IMU needs no new package — the ZED ROS 2 wrapper already
publishes `/zed/zed_node/imu/data`. KISS-ICP is the only build-from-source
piece. Three viable options, ranked:

1. **`kiss-icp` ROS 2 wrapper** — official, CMake-built, Humble branch
   exists on `kiss-icp` repo (`PRBonn/kiss-icp`, `ros` subfolder).
   Publishes `nav_msgs/Odometry` on `/kiss/odometry` and a `tf`
   `odom_kiss → base_link` (configurable). **Pick this.**
2. **`kiss_icp` Python via `pip`** + a thin custom node — works but
   requires writing the ROS plumbing ourselves and the Python wrapper
   is slower than the C++ entry point.
3. **`dlo` / `dlio`** — tighter accuracy, much heavier on Jetson and
   pulls in PCL / Sophus dependency chains that don't currently live
   in our Docker image. Out of scope for "lightest viable add."

`FAST-LIO2` was considered but rejected: it tightly couples the lidar
IMU and produces excellent results, however it expects a different
pointcloud preprocessing pipeline than the SICK multiScan driver hands
us, and its tuning surface is large enough to be a project of its own.

---

## 3. Inputs and outputs

### KISS-ICP

**Inputs**
- **Pointcloud:** `/multiScan/cloud_unstructured_fullframe` (SICK
  multiScan default; verify on the Jetson with
  `ros2 topic list | grep -i cloud`). Must be `sensor_msgs/PointCloud2`
  with timestamps and an unambiguous `frame_id` (typically
  `cloud_unstructured_fullframe` or `multiScan` depending on the
  driver build).
- **Initial pose / TF:** the static transform `base_link → <lidar
  frame>` must already be published by the URDF (`shogi.urdf` chain
  through `multiScan_link`). If KISS-ICP sees no TF for the cloud's
  `frame_id`, every cloud is rejected silently.

**Outputs**
- **Odometry topic:** `/kiss/odometry` (default; we'll remap to
  `/lidar_odom` for clarity).
- **TF:** KISS-ICP wants to publish `odom_kiss → base_link`. **We
  must disable this** (`publish_odom_tf: false`) because the Local EKF
  already owns the `odom → base_link` TF. Two publishers on the same
  edge is the fastest way to break Nav2.

**Frame / covariance contract**
- Output frame: `odom` (after remap). Child frame: `base_link`.
- Covariance: KISS-ICP reports a fixed pose covariance. Override it on
  the way into the EKF — see §6.

### ZED IMU

**Inputs (as published by ZED wrapper)**
- **Topic:** `/zed/zed_node/imu/data` — `sensor_msgs/Imu`.
- **Frame ID on the message:** `zed2i_imu_link` (verify on the Jetson;
  the wrapper version drives the exact name).

**TF prerequisite — the actual blocker**
The current `shogi.urdf` does not connect `base_link → zed2i_imu_link`.
Without that chain, `robot_localization` rejects every ZED IMU sample
with a TF lookup failure and the imu1 entry below silently starves.
Two acceptable fixes (pick one before the YAML edits land):

1. **URDF route (preferred):** add a fixed joint
   `base_link → zed_camera_link → zed2i_camera_link → zed2i_imu_link`
   with the measured offsets from the camera mount. The ZED's
   internal frames already exist in the camera's URDF fragments; we
   just need to attach the camera root to `base_link`.
2. **Launch route:** in the ZED launch, pass `publish_imu_tf:=true`
   AND attach `zed_camera_link` to `base_link` via a static
   transform publisher in `slam.launch.py`. Less clean but no URDF
   edit.

**Covariance gotcha**
The ZED IMU's covariance fields are sometimes published as zeros (or
worse, NaNs on certain wrapper versions) — `robot_localization`
treats zero covariance as "perfect," and zero-covariance IMU input
can collapse P. Mitigation in §6: declare an explicit
`imu1_remove_gravitational_acceleration: true` is irrelevant since we
don't fuse accel; what matters is overriding the angular-velocity
covariance via either (a) a small `imu_filter_madgwick` republisher
that injects a sane covariance, or (b) the
`imu1_twist_rejection_threshold` knob to gate degenerate samples.

---

## 4. Wiring into `ekf_local.yaml`

Add a second odom source. Current file (relevant excerpt):

```yaml
ekf_node:
  ros__parameters:
    frequency: 30.0
    world_frame: odom
    two_d_mode: true

    odom0: /odom
    odom0_config: [false, false, false,
                   false, false, false,
                   true,  true,  false,
                   false, false, true,
                   false, false, false]
    odom0_differential: true

    imu0: /multiScan/imu
    imu0_config: [false, false, false,
                  false, false, false,
                  false, false, false,
                  false, true,  true,
                  false, false, false]
    imu0_differential: true
```

Proposed additions (below `imu0_differential`):

```yaml
    # Lidar odometry from KISS-ICP. Velocity-only on linear axes —
    # we trust the lidar for vx/vy when the wheels slip, but do
    # NOT fuse its absolute pose (would make the Local EKF jump on
    # ICP convergence shifts). Yaw rate is left to the IMUs.
    odom1: /lidar_odom
    odom1_config: [false, false, false,
                   false, false, false,
                   true,  true,  false,
                   false, false, false,
                   false, false, false]
    odom1_differential: true
    # KISS-ICP's covariance is uniformly tiny; the EKF would
    # otherwise weight lidar above wheel + IMU even when ICP is
    # struggling on a featureless wall. Inflate at the EKF.
    odom1_pose_rejection_threshold: 5.0
    odom1_twist_rejection_threshold: 5.0

    # ZED2i camera IMU. Yaw-rate only — physically separate gyro
    # from the SICK IMU, so the EKF gets a genuine second
    # observation of yaw rate (independent bias drift, independent
    # mount flex). Pitch-rate is also reasonable to fuse if the
    # camera is rigidly mounted; leave it off in v1 until the
    # mount has been characterized. Linear acceleration stays off
    # for the same reason imu0 disables it.
    imu1: /zed/zed_node/imu/data
    imu1_config: [false, false, false,
                  false, false, false,
                  false, false, false,
                  false, false, true,
                  false, false, false]
    imu1_differential: true
    # Guard against the ZED publishing degenerate (zero/NaN)
    # covariance on certain wrapper versions — a zero-covariance
    # IMU sample collapses P. Tune empirically.
    imu1_twist_rejection_threshold: 5.0
```

**Why velocity-only (vx, vy) for KISS-ICP:** the Local EKF must remain
jump-free. KISS-ICP's *pose* drifts deterministically over time (no
loop closure) and would inject those drifts into the `odom` TF.
KISS-ICP's *velocity* between consecutive scans, by contrast, is
exactly the kind of instantaneous estimate the EKF is built to fuse.

**Why not vyaw from KISS-ICP:** between the SICK IMU and the ZED IMU,
the EKF already has two low-latency, tight-noise yaw-rate observations.
Adding a third from ICP residuals mostly inflates the tuning surface
without proportionate information gain — and ICP yaw is correlated with
deskewing, which uses one of the IMUs we're already fusing
(double-counting risk; see §10).

**Why differential on imu1:** robot_localization's `differential` mode
takes successive samples and uses their delta as the measurement,
which strips out absolute-orientation drift. Since we only fuse
angular velocity from the ZED IMU, this is conservative — it's the
same treatment imu0 already gets.

---

## 5. Launch wiring

Add a node entry to `slam.launch.py` between `ekf_local` and
`slam_toolbox`:

```python
kiss_icp = Node(
    package='kiss_icp',
    executable='kiss_icp_node',  # check exact name after building
    name='kiss_icp',
    output='screen',
    parameters=[{
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        # — pointcloud input
        'topic': '/multiScan/cloud_unstructured_fullframe',
        # — frames
        'odom_frame': 'odom_kiss',     # internal; we don't publish TF
        'child_frame': 'base_link',
        'publish_odom_tf': False,      # CRITICAL — see §3
        'publish_debug_clouds': False, # save Jetson cycles
        # — algorithm
        'max_range': 25.0,             # tune to SICK multiScan's range
        'min_range': 0.5,
        'voxel_size': 0.5,             # m; bigger = faster, less precise
        'deskew': True,                # use IMU+timestamps to undistort
        'max_points_per_voxel': 20,
    }],
    remappings=[
        ('/kiss/odometry', '/lidar_odom'),
    ],
)
```

Place it **after** `ekf_local` in the `LaunchDescription` list so the
EKF is up and bound to the topic by the time KISS-ICP starts publishing.

---

## 6. TF chain audit (the part that always bites)

KISS-ICP needs the cloud's `frame_id` to be reachable from `base_link`
via static TF. The current chain (per `shogi.urdf`) is approximately:

```
base_link → multiScan_link → cloud_unstructured_fullframe
                          (or whatever frame_id the driver stamps)
```

**Verify before merging:**

```bash
ros2 run tf2_ros tf2_echo base_link <cloud_frame_id>
```

If that lookup fails, KISS-ICP silently rejects every cloud. If the
SICK driver stamps the cloud with the lidar's optical frame name, add
a static publisher in `slam.launch.py`:

```python
static_tf_lidar = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    arguments=['0', '0', '0', '0', '0', '0',
               'multiScan_link', '<cloud_frame_id>'],
)
```

(prefer fixing it in the URDF if at all possible.)

---

## 7. Jetson resource budget

KISS-ICP at `voxel_size: 0.5` on a Velodyne-class cloud (~70 k
points/scan) eats roughly **15–25 % of one Orin core** at 10 Hz. SICK
multiScan publishes ~30 k points/scan in its full-frame mode, so we
expect ~10–15 % of one core. That sits comfortably alongside
slam_toolbox (~30 % of one core) and the existing EKFs (<5 %).

The ZED IMU adds **negligible** load — the ZED wrapper is already
running for the camera; the EKF subscription is one extra ~200 Hz
callback. No measurable budget impact.

If the Jetson is already CPU-saturated when this lands, raise
KISS-ICP's `voxel_size` to 0.7–1.0 first — much more impact than any
other knob. Drop deskewing as the next step.

---

## 8. Tuning knobs that matter

| Parameter | Effect | First-pass value |
|---|---|---|
| `voxel_size` | Throughput vs. accuracy. Doubling it ~halves the CPU. | 0.5 m |
| `max_range` | Cuts long, sparse points that don't help ICP and hurt the matrix conditioning. | 25 m |
| `min_range` | Drops self-returns from the chassis. | 0.5 m |
| `deskew` | Undistorts the cloud using the lidar IMU + per-point timestamps. Helps a lot when the robot rotates fast. | `true` |
| `initial_threshold` | Per-point inlier distance bootstrap. | leave default |
| `min_motion_th` | Skip-publish threshold when robot is stationary. | leave default |

EKF-side knobs:

| Parameter | Effect | First-pass value |
|---|---|---|
| `odom1_pose_rejection_threshold` | Mahalanobis gate on the inflated lidar pose. | 5.0 |
| `odom1_twist_rejection_threshold` | Same for velocity. | 5.0 |
| `odom1_differential` | Treat as relative velocity, don't latch absolute pose. | `true` |
| `imu1_twist_rejection_threshold` | Mahalanobis gate on ZED gyro samples. Low values protect against zero-covariance / NaN bursts; raise once the ZED stream has been characterized. | 5.0 |
| `imu1_differential` | Strip absolute-orientation drift from the ZED IMU. | `true` |

---

## 9. Validation plan

A. **Static stand-in:** robot powered, wheels on jacks. `/odom`
   reports zero, `/lidar_odom` reports zero ± noise, both IMUs report
   gyro readings near zero (a few mrad/s drift is fine), EKF output
   is zero. If lidar reports drift here, `voxel_size` is too small or
   `min_range` is too low (chassis returns).

B. **Push test:** push the robot 1 m. `/odom` and `/lidar_odom` should
   agree to ~3 cm. Disagreement >10 cm → check the lidar TF, then the
   `voxel_size`.

C. **Slip test:** drive on a lubricated tile or hold the robot in
   place while commanding velocity. `/odom` reports motion that didn't
   happen; `/lidar_odom` does not. EKF output should follow lidar
   (this is the whole point of the integration). If the EKF still
   tracks the wheels, the gate thresholds are too tight or the wheel
   covariance is too small.

D. **Featureless-wall test:** drive parallel to a long blank wall.
   ICP convergence is degenerate along the wall direction. Check that
   `/lidar_odom` stops publishing or its rejection rate spikes —
   `odom1_pose_rejection_threshold` exists for exactly this case.

E. **Cycle test:** drive a 50 m loop back to start. EKF output
   position drift < 2 % of distance traveled. (Loop closure is *not*
   KISS-ICP's job; the Map EKF + slam_toolbox handles that path.)

F. **IMU yaw cross-check (ZED-specific):** rotate the robot in place
   ~360°. Plot `/multiScan/imu`'s vyaw and `/zed/zed_node/imu/data`'s
   vyaw on the same axis. They should agree to within a few %; a
   constant offset > 1°/s indicates a frame mis-alignment in the URDF
   chain (re-check §6). A *scaled* mismatch indicates one IMU's
   self-cal isn't done.

G. **ZED IMU starvation check:** with the ZED IMU plugged into the
   EKF, run `ros2 topic echo /diagnostics` (or
   `ros2 topic hz /local_ekf/odom`) while watching the EKF's stderr.
   If you see `Could not transform message from zed2i_imu_link to
   base_link`, the TF chain (§6) is still broken — fix the URDF
   before believing any other validation result.

H. **Single-IMU fallback:** unplug the ZED. EKF should keep producing
   `/local_ekf/odom` with the SICK IMU + wheels + KISS-ICP only —
   verify it doesn't lock or stall. This proves the ZED IMU is
   additive, not load-bearing.

---

## 10. Risks and known unknowns

### KISS-ICP
- **Pointcloud topic name on the deployed robot.** The plan assumes
  `/multiScan/cloud_unstructured_fullframe`. Confirm via
  `ros2 topic list` on the Jetson before any of the YAML edits land.
- **Cloud `frame_id` may not match URDF.** SICK driver names have
  changed across releases; the URDF was last fixed up for an older
  driver. §6 covers the fallback.
- **KISS-ICP ROS 2 Humble package may not be in our Docker image.**
  May need to add it to the image or build from source under
  `isaac_ros-dev/src/`. Building from source is the safer path —
  pinned to a specific commit, no surprise ABI breaks.
- **`two_d_mode: true` interaction.** The Local EKF zeros out z, roll,
  pitch. KISS-ICP runs in 3-D internally; its output gets projected
  to 2-D at the EKF input. Fine in theory; verify experimentally that
  the EKF doesn't reject unusually many lidar messages on slopes.
- **SICK IMU + KISS-ICP deskewing competition.** KISS-ICP can
  optionally use an external IMU for deskewing. We're not wiring that
  in v1 — KISS-ICP will deskew using only point timestamps. Consider
  passing `/multiScan/imu` to KISS-ICP in v2 if rotation-rate motion
  blur is visible in residuals.

### ZED IMU
- **TF chain is the headline blocker.** Without `base_link →
  zed2i_imu_link`, every IMU sample is silently rejected. §6 has the
  two fix paths; pick one before merging.
- **Zero / NaN covariance bursts.** Some ZED wrapper versions publish
  IMU covariance fields as zeros, which `robot_localization` treats as
  "perfect measurement." If P collapses on this filter, it's almost
  certainly this. `imu1_twist_rejection_threshold` is the first line
  of defense; a covariance-injecting republisher is the second.
- **Mount vibration.** The ZED is bolted higher up than the SICK
  lidar; chassis flex shows up as gyro noise on the ZED IMU before
  it shows up on the SICK. If the ZED IMU's variance (when stationary)
  is more than ~3× the SICK IMU's, treat that as the floor for any
  R-tuning we do later.
- **Double-counting via KISS-ICP deskewing.** If/when v2 deskewing
  passes the SICK IMU into KISS-ICP, the SICK IMU appears twice in
  the EKF's information set (directly via `imu0`, indirectly via
  KISS-ICP's deskewed velocity). This is an *information correlation*
  the EKF doesn't model. Mitigation: deskew KISS-ICP with the **ZED**
  IMU instead of the SICK IMU when the time comes — the SICK IMU
  remains a pure direct input, the ZED IMU remains a pure direct
  input, and KISS-ICP's velocity is then derived from a third source
  (cloud + ZED IMU) that's still mostly orthogonal to both.
- **Time sync.** The ZED IMU and the SICK IMU are clocked separately.
  `robot_localization` assumes monotonic per-source timestamps; check
  that the ZED's `header.stamp` uses Jetson system time, not its
  internal sensor clock. The ZED wrapper has a `sim_time`-style flag
  for this; verify per the wrapper version in our image.

---

## 11. Explicitly out of scope

- **Map-frame fusion.** Neither KISS-ICP nor the ZED IMU feeds the Map
  EKF. slam_toolbox's `/pose` remains the only `map`-frame correction
  source.
- **Loop closure / drift correction.** KISS-ICP is intentionally
  drift-bounded only by ICP residuals; long-term drift is the Map
  EKF's problem.
- **Replacing wheel odometry.** Wheel `/odom` stays in. All velocity
  sources cross-check each other and the EKF picks the weighted blend.
- **ZED visual-inertial odometry (the camera-derived pose track).**
  *Different* workstream from the IMU-only fusion done here. ZED VIO
  publishes a 6-DoF pose; using it would require all the same
  jump-free analysis we did for KISS-ICP, plus dealing with VIO scale
  drift on featureless scenes. Park until a future plan.
- **ZED IMU linear acceleration.** The Imu config bits enable only
  vyaw. Adding accel would require a careful gravity-removal pass and
  buys little for a ground robot.

---

## 12. Rollback

Both sensors roll back independently — there is no shared state
between them at the EKF level.

- **KISS-ICP misbehaving:** comment out the `kiss_icp` Node entry in
  `slam.launch.py`, or set `odom1_pose_rejection_threshold` and
  `odom1_twist_rejection_threshold` to `0.001` to make the EKF
  reject every lidar sample. The Local EKF falls back to wheel +
  SICK IMU + ZED IMU.

- **ZED IMU misbehaving:** delete the `imu1: /zed/zed_node/imu/data`
  block from `ekf_local.yaml`, or set `imu1_twist_rejection_threshold`
  to `0.001`. The Local EKF falls back to wheel + SICK IMU
  (+ KISS-ICP if that's still in).

- **Total rollback to today's config:** revert both edits to
  `ekf_local.yaml` and remove the `kiss_icp` Node from
  `slam.launch.py`. ZED IMU TF/URDF changes can stay — they are
  benign side effects useful for any future ZED work.
