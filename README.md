# AutoNav-Simulated-Universe

Sister repo to **AutoNav_2025-2026**. Hosts the standalone simulators we
use to develop and validate individual subsystems of the AutoNav robot
before running them on the real hardware, plus the source-of-truth
course asset for full-stack Fortress (Gazebo) simulation.

```
.
├── LiDAR_Sim/         — Grade-aware LiDAR perception + navigation
├── GPS_Sim/           — GPS waypoint navigation without a magnetometer
├── GUI_Sim/           — Real HUD running against synthetic ROS topics
├── BEHAVIOR_TREE_Sim/ — Recovery-state decision tree on a Chaplygin sleigh
├── SPEED_Sim/         — Per-wheel PID velocity control (MPH setpoint)
└── Mock_Course_Asset/ — Blender course scene for Gazebo Fortress
```

Each Python sim is self-contained: its own sources under
`simulated_world/`, its own `requirements.txt`, and its own Windows
`.bat` / macOS `.command` launchers at the top of the sim's folder.

---

## What each sim does

### LiDAR Sim — `LiDAR_Sim/`
Drops one or many agents onto a 3D STL terrain and navigates with a
**grade-based costmap** built from a simulated LiDAR point cloud. Two modes:

- **Interactive GUI** — click to place a robot, shift-click to place a goal,
  G to go. Renders the live LiDAR, the costmap, and the planner's path.
- **Ramp benchmark** — 5/10/15 parallel agents crossing the five graded
  ramps to verify the costmap respects the grade-tolerance threshold.

The contract the perception layer must obey lives in
[`LiDAR_Sim/RULES.md`](LiDAR_Sim/RULES.md). The terrain-grade-layer design
notes are in `LiDAR_Sim/terrain-grade-layer-plan.md`.

### GPS Sim — `GPS_Sim/`
A GPS-without-magnetometer waypoint simulator. The robot has no compass, so
it estimates the rotation between its local odom frame and the geographic
frame on the fly by fusing GPS-vs-odom displacement through a 3-state EKF
(`x_world, y_world, θ_offset`). Includes scripted/random/real/crazy scenarios
with obstacles, GPS spoofers, projectors, jammers, roof multipath, and
foliage.

The agent + simulator contract lives in
[`GPS_Sim/RULES.md`](GPS_Sim/RULES.md). The placement plan and waypoint
methodology are in `GPS_Sim/GPS_waypoint_placement_plan.md`.

### GUI Sim — `GUI_Sim/`
Runs the **real** AutoNav HUD (`hud_node.py`) against a faked-up ROS stack
(`fake_ros.install()`). The HUD imports as-is with `_HAS_ROS=True`; the
runner generates synthetic camera / lidar / GPS / odom / electrical messages
at ~10 Hz and delivers them through the recorded subscription callbacks.
Useful for changing HUD code without booting the Jetson stack.

`bake_offscreen.py` (entry: `Bake_Bag_Video.command` / `.bat`) renders a
30 fps MP4 of the live-sensor panel from a recorded CSV or rosbag2 `.db3`.

### Behavior Tree Sim — `BEHAVIOR_TREE_Sim/`
Chaplygin-sleigh robot in a corridor maze. Pairs the per-wheel force
controller and a Dijkstra planner with the **8-state recovery decision
tree** ported from the `path_following` branch of `AutoNav_2025-2026`
(`bt_nav.xml`). Useful for shaking out recovery transitions
(`NORMAL_FOLLOWING` → `FORWARD_BLOCKED_…` → `BACKUP_RECOVERY` etc.)
without the rest of the stack in the loop.

### SPEED Sim — `SPEED_Sim/`
Per-wheel PID velocity control on the same Chaplygin sleigh dynamics
as the BT sim. Replaces AutoNav's legacy 0–75 arbitrary-units speed
system with a direct **MPH setpoint** (0.0 – 5.0 MPH, 0.1 MPH ticks).
Number-row keys jump to half-MPH grid points; `[` / `]` nudge by 0.1.
Drive with the arrow keys; the HUD shows commanded-vs-measured wheel
velocity scopes. Runs at 60 FPS render / 240 Hz physics / 50 Hz native
control rate. Used to tune Kp / Ki / Kd against the real robot mass +
track width before deploying to hardware.

---

## Course asset for full-stack Fortress sim

### Mock Course Asset — `Mock_Course_Asset/`
`Course.blend` is the source-of-truth Blender scene for the AutoNav
mock course — geometry, ramps, GPS waypoints, obstacles, and surface
tags. It's the input to a **Gazebo Fortress** (Ignition) simulation
that boots the actual AutoNav ROS 2 stack against a virtual world,
so perception + navigation + GPS fusion + HUD can be exercised
together without hardware. Each scene object carries Blender custom
properties (grade, GPS coords, surface class, spoofer zones) that the
Fortress exporter stamps onto the SDF world.

See [`Mock_Course_Asset/README.md`](Mock_Course_Asset/README.md) for
the export pipeline and the property-dumping snippet.

---

## Quick start

### Prerequisites
- **Python 3.12** (the pinned versions in each `requirements.txt` were
  resolved against 3.12.4).
- [`uv`](https://github.com/astral-sh/uv) — recommended. The `.bat` /
  `.command` launchers expect `simulated_world/.venv/` to exist and assume
  `uv` was used to create it. Plain `python -m venv` also works.
- **Git LFS** — the LiDAR terrain `.stl`, the course `.blend`, and any
  other `.blend` assets are stored via LFS (see `.gitattributes`). Run
  `git lfs install && git lfs pull` after cloning or those files will
  be tiny pointer files and the sims will crash on load.

### One-time setup, per sim

Each sim has its own venv at `<sim>/simulated_world/.venv`. Pick the sim(s)
you want, `cd` into its `simulated_world/`, then:

```bash
# macOS / Linux
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

```cmd
REM Windows
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

If you don't have `uv`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # macOS / Linux
.venv\Scripts\pip install -r requirements.txt      # Windows
```

That's it — the launchers find Python via `.venv/bin/python` (Unix) or
`.venv\Scripts\python.exe` (Windows) automatically.

### Running

| Sim           | macOS                                          | Windows                                  | Entry script                           |
|---------------|------------------------------------------------|------------------------------------------|----------------------------------------|
| LiDAR         | `LiDAR_Sim/Run_LIDAR_SIM.command`              | `LiDAR_Sim\Run_LIDAR_SIM.bat`            | `launcher.py` → `lidar_sim_gui.py`     |
| LiDAR (skip launcher) | `LiDAR_Sim/Run_LIDAR_SIM_direct.command` | `LiDAR_Sim\Run_LIDAR_SIM_direct.bat` | `lidar_sim_gui.py`                  |
| GPS           | `GPS_Sim/Run_GPS_SIM.command`                  | `GPS_Sim\Run_GPS_SIM.bat`                | `launcher.py` → `gps_sim_gui.py`       |
| GPS (skip launcher)   | `GPS_Sim/Run_GPS_SIM_direct.command`     | `GPS_Sim\Run_GPS_SIM_direct.bat`     | `gps_sim_gui.py`                       |
| GUI           | `GUI_Sim/Run_GUI.command`                      | `GUI_Sim\Run_GUI.bat`                    | `runner.py`                            |
| Bag → MP4     | `GUI_Sim/Bake_Bag_Video.command`               | `GUI_Sim\Bake_Bag_Video.bat`             | `bake_offscreen.py` (file-pick prompt) |
| Behavior Tree | `BEHAVIOR_TREE_Sim/Run_BT_SIM.command`         | `BEHAVIOR_TREE_Sim\Run_BT_SIM.bat`       | `launcher.py` → `bt_sim_gui.py`        |
| SPEED         | `SPEED_Sim/Run_SPEED_SIM.command`              | (macOS only for now)                     | `speed_sim.py`                         |

Double-click the `.command` (macOS Finder) or `.bat` (Windows Explorer), or
run it from a terminal. Any extra CLI flags are forwarded to the underlying
script — e.g. `Run_GPS_SIM_direct.command --real --single --seed 7`.

---

## Repo layout

```
<Sim>/
├── Run_<SIM>.bat              ← Windows launcher (panel)
├── Run_<SIM>.command          ← macOS launcher (panel)
├── Run_<SIM>_direct.bat       ← Windows, skip the panel
├── Run_<SIM>_direct.command   ← macOS, skip the panel
├── RULES.md                   ← Contract the sim must satisfy (LiDAR, GPS)
├── *.md                       ← Design / planning docs
├── 3d_assets/  or  data/      ← Meshes, recorded runs, baked artefacts
└── simulated_world/
    ├── requirements.txt       ← uv pip freeze; install here
    ├── .venv/                 ← Created locally; gitignored
    ├── launcher.py            ← Flag-picker panel (LiDAR, GPS, BT)
    ├── <sim>_gui.py           ← The actual simulator
    └── scripts/               ← Benchmarks, plotters, calibration tools
```

`.venv/`, baked playbacks, the `tile_cache/`, `example-playback-csv/`,
Blender autosave backups (`*.blend1`), and per-sim
`.claude/settings.local.json` are all gitignored — see `.gitignore`.

---

## Known issues

- **macOS, LiDAR launcher panel** — Qt 5.15's `setDefault(True)` on
  `launch_btn` auto-fires the click on window-show, so the panel
  immediately spawns `lidar_sim_gui.py` with the default options instead
  of waiting for input. Workaround: use `Run_LIDAR_SIM_direct.command`
  and pass flags directly. (Doesn't affect Windows.)
- **`Consolas` font warning** on macOS / Linux is harmless — the
  monospace preview falls back to the system mono font.

---

## Related

- Main robot stack: [`AutoNav_2025-2026`](../AutoNav_2025-2026) (the
  hardware-side ROS 2 / Jetson workspace this sim repo feeds into).
