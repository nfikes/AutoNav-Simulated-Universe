# AutoNav-Simulated-Universe

Sister repo to **AutoNav_2025-2026**. Hosts the three standalone simulators we
use to develop and validate individual subsystems of the AutoNav robot before
running them on the real hardware.

```
.
├── LiDAR Sim/   — Grade-aware LiDAR perception + navigation
├── GPS Sim/     — GPS waypoint navigation without a magnetometer
└── GUI Sim/     — Real HUD running against synthetic ROS topics
```

Each sim is self-contained: its own Python sources under `simulated_world/`,
its own `requirements.txt`, and its own Windows `.bat` / macOS `.command`
launchers at the top of the sim's folder.

---

## What each sim does

### LiDAR Sim — `LiDAR Sim/`
Drops one or many agents onto a 3D STL terrain and navigates with a
**grade-based costmap** built from a simulated LiDAR point cloud. Two modes:

- **Interactive GUI** — click to place a robot, shift-click to place a goal,
  G to go. Renders the live LiDAR, the costmap, and the planner's path.
- **Ramp benchmark** — 5/10/15 parallel agents crossing the five graded
  ramps to verify the costmap respects the grade-tolerance threshold.

The contract the perception layer must obey lives in
[`LiDAR Sim/RULES.md`](LiDAR%20Sim/RULES.md). The terrain-grade-layer design
notes are in `LiDAR Sim/terrain-grade-layer-plan.md`.

### GPS Sim — `GPS Sim/`
A GPS-without-magnetometer waypoint simulator. The robot has no compass, so
it estimates the rotation between its local odom frame and the geographic
frame on the fly by fusing GPS-vs-odom displacement through a 3-state EKF
(`x_world, y_world, θ_offset`). Includes scripted/random/real/crazy scenarios
with obstacles, GPS spoofers, projectors, jammers, roof multipath, and
foliage.

The agent + simulator contract lives in
[`GPS Sim/RULES.md`](GPS%20Sim/RULES.md). The placement plan and waypoint
methodology are in `GPS Sim/GPS_waypoint_placement_plan.md`.

### GUI Sim — `GUI Sim/`
Runs the **real** AutoNav HUD (`hud_node.py`) against a faked-up ROS stack
(`fake_ros.install()`). The HUD imports as-is with `_HAS_ROS=True`; the
runner generates synthetic camera / lidar / GPS / odom / electrical messages
at ~10 Hz and delivers them through the recorded subscription callbacks.
Useful for changing HUD code without booting the Jetson stack.

`bake_offscreen.py` (entry: `Bake_Bag_Video.command` / `.bat`) renders a
30 fps MP4 of the live-sensor panel from a recorded CSV or rosbag2 `.db3`.

---

## Quick start

### Prerequisites
- **Python 3.12** (the pinned versions in each `requirements.txt` were
  resolved against 3.12.4).
- [`uv`](https://github.com/astral-sh/uv) — recommended. The `.bat` /
  `.command` launchers expect `simulated_world/.venv/` to exist and assume
  `uv` was used to create it. Plain `python -m venv` also works.
- **Git LFS** — the LiDAR terrain `.stl` and `.blend` files are stored via
  LFS (see `.gitattributes`). Run `git lfs install && git lfs pull` after
  cloning or the LiDAR terrain mesh will be a tiny pointer file and the
  sim will crash on load.

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

| Sim       | macOS                                       | Windows                                  | Entry script                           |
|-----------|---------------------------------------------|------------------------------------------|----------------------------------------|
| LiDAR     | `LiDAR Sim/Run_LIDAR_SIM.command`           | `LiDAR Sim\Run_LIDAR_SIM.bat`            | `launcher.py` → `lidar_sim_gui.py`     |
| LiDAR (skip launcher) | `LiDAR Sim/Run_LIDAR_SIM_direct.command` | `LiDAR Sim\Run_LIDAR_SIM_direct.bat` | `lidar_sim_gui.py`                  |
| GPS       | `GPS Sim/Run_GPS_SIM.command`               | `GPS Sim\Run_GPS_SIM.bat`                | `launcher.py` → `gps_sim_gui.py`       |
| GPS (skip launcher)   | `GPS Sim/Run_GPS_SIM_direct.command`     | `GPS Sim\Run_GPS_SIM_direct.bat`     | `gps_sim_gui.py`                       |
| GUI       | `GUI Sim/Run_GUI.command`                   | `GUI Sim\Run_GUI.bat`                    | `runner.py`                            |
| Bag → MP4 | `GUI Sim/Bake_Bag_Video.command`            | `GUI Sim\Bake_Bag_Video.bat`             | `bake_offscreen.py` (file-pick prompt) |

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
    ├── launcher.py            ← Flag-picker panel (LiDAR, GPS)
    ├── <sim>_gui.py           ← The actual simulator
    └── scripts/               ← Benchmarks, plotters, calibration tools
```

`.venv/`, baked playbacks, the `tile_cache/`, and `example-playback-csv/` are
all gitignored — see `.gitignore`.

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
