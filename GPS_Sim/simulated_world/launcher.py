#!/usr/bin/env python3
"""Flag-picker panel that launches gps_sim_gui.py.

Pops a small PyQt5 window with the most useful CLI flags grouped into
sections. Click "Launch" and the picker exits, replaced by the actual
sim process with the chosen argv.
"""

import os
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


HERE = Path(__file__).resolve().parent
SIM_SCRIPT = HERE / "gps_sim_gui.py"
BAKES_DIR = HERE / "bakes"
BAKE_FPS = 60


SCENARIO_PRESETS = [
    ("Scripted (default)", None,
     "Built-in scripted layout — deterministic, good for first run."),
    ("Random", "--random",
     "Random scenario; honors the obstacle / roof / projector counts."),
    ("Real (calibrated outdoor)", "--real",
     "Sparse rocks + one roof + one projector + foliage. Tuned GPS noise."),
    ("Crazy (all hazards)", "--crazy",
     "Dense obstacles + jammers + spoofers + projector multipath + roofs."),
]


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPS Sim — Launch Options")
        self.setMinimumWidth(980)

        root = QtWidgets.QVBoxLayout(self)

        cols_row = QtWidgets.QHBoxLayout()
        left_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()
        cols_row.addLayout(left_col, 1)
        cols_row.addLayout(right_col, 1)
        root.addLayout(cols_row)

        # ── Scenario preset ───────────────────────────────────────
        preset_box = QtWidgets.QGroupBox("Scenario")
        preset_layout = QtWidgets.QVBoxLayout(preset_box)
        self.preset_group = QtWidgets.QButtonGroup(self)
        for i, (label, flag, tip) in enumerate(SCENARIO_PRESETS):
            rb = QtWidgets.QRadioButton(label)
            rb.setToolTip(tip)
            if i == 0:
                rb.setChecked(True)
            self.preset_group.addButton(rb, i)
            preset_layout.addWidget(rb)
        # When the user picks a preset other than Random, the World
        # Counts spinboxes are ignored by the sim; grey them out and
        # refresh the preview so the command-line doesn't lie.
        self.preset_group.buttonToggled.connect(
            lambda _btn, _on: self._on_preset_changed())
        left_col.addWidget(preset_box)

        # ── Agent options ─────────────────────────────────────────
        agents_box = QtWidgets.QGroupBox("Agents")
        agents_form = QtWidgets.QFormLayout(agents_box)

        self.single_cb = QtWidgets.QCheckBox(
            "Single-agent mode (--single)  ·  full visualization")
        self.single_cb.setChecked(True)
        agents_form.addRow(self.single_cb)

        self.scatter_cb = QtWidgets.QCheckBox(
            "Scatter starts across map (--scatter)")
        agents_form.addRow(self.scatter_cb)

        self.agents_spin = QtWidgets.QSpinBox()
        self.agents_spin.setRange(1, 10000)
        self.agents_spin.setValue(25)
        agents_form.addRow("Number of agents (--agents)", self.agents_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        agents_form.addRow("Random seed (--seed)", self.seed_spin)

        left_col.addWidget(agents_box)

        # Disable --agents while --single is checked.
        def _sync_single():
            self.agents_spin.setEnabled(not self.single_cb.isChecked())
        self.single_cb.toggled.connect(_sync_single)
        _sync_single()

        # ── World counts ──────────────────────────────────────────
        # These spinboxes only feed --random. Scripted uses a fixed
        # layout (1 obstacle / 1 roof / 1 projector / 0 / 0 / 0) and
        # Real / Crazy use their preset counts. The whole group is
        # greyed out when one of those presets is active so the user
        # isn't fooled into thinking these numbers matter.
        self.world_box = QtWidgets.QGroupBox(
            "World counts (used by Random only)")
        world_box = self.world_box
        world_form = QtWidgets.QFormLayout(world_box)

        self.obstacles_spin = QtWidgets.QSpinBox()
        self.obstacles_spin.setRange(0, 200)
        self.obstacles_spin.setValue(12)
        world_form.addRow("Obstacles (--obstacles)", self.obstacles_spin)

        self.roofs_spin = QtWidgets.QSpinBox()
        self.roofs_spin.setRange(0, 50)
        self.roofs_spin.setValue(3)
        world_form.addRow("Roofs (--roofs)", self.roofs_spin)

        self.projectors_spin = QtWidgets.QSpinBox()
        self.projectors_spin.setRange(0, 50)
        self.projectors_spin.setValue(4)
        world_form.addRow("Projectors (--projectors)", self.projectors_spin)

        self.jammers_spin = QtWidgets.QSpinBox()
        self.jammers_spin.setRange(0, 50)
        self.jammers_spin.setValue(0)
        world_form.addRow("Jammers (--jammers)", self.jammers_spin)

        self.foliage_spin = QtWidgets.QSpinBox()
        self.foliage_spin.setRange(0, 50)
        self.foliage_spin.setValue(0)
        world_form.addRow("Foliage (--foliage)", self.foliage_spin)

        self.spoofers_spin = QtWidgets.QSpinBox()
        self.spoofers_spin.setRange(0, 50)
        self.spoofers_spin.setValue(0)
        world_form.addRow("Spoofers (--spoofers)", self.spoofers_spin)

        right_col.addWidget(world_box)

        # ── Robot / EKF ───────────────────────────────────────────
        algo_box = QtWidgets.QGroupBox("Algorithm / robot")
        algo_form = QtWidgets.QFormLayout(algo_box)

        self.no_ekf_cb = QtWidgets.QCheckBox(
            "Disable θ-EKF + LIDAR/IMU fusion (--no-ekf)")
        self.no_ekf_cb.setToolTip(
            "Field-test config: raw biased wheel odom, no fusion. "
            "Reproduces the orbit signature from May 9 runs.")
        algo_form.addRow(self.no_ekf_cb)

        # ── New robust algorithm — cold-start θ snap ──
        # ON by default because the new design intent
        # (GPS Sim/DESIGN_INTENT.md) treats this as the standard
        # one-shot θ seed on the first GPS goal. Without it the
        # 4-heading invariant has no seed and the agent's first GPS
        # update Kalman-gains hard onto whatever the lever-arm-
        # corrupted first fix says.
        self.coldstart_cb = QtWidgets.QCheckBox(
            "Cold-start θ snap on first goal (--coldstart-bias-enable)")
        self.coldstart_cb.setChecked(True)
        self.coldstart_cb.setToolTip(
            "Real-robot ROS param: coldstart_bias_enabled. The "
            "deployed run-gps.sh on improve/gps-waypoint-continuity "
            "flips this to True. On the very first GPS goal accepted, "
            "snap the EKF's θ_offset so that the goal projects "
            "directly in front of base_link. Bootstrap_theta then "
            "overwrites the seed once GPS-vs-odom baseline > 1.5 m. "
            "Required for the new direction-irrelevant converging "
            "algorithm to work as designed.")
        algo_form.addRow(self.coldstart_cb)

        self.heading_cb = QtWidgets.QCheckBox("Force initial heading")
        self.heading_spin = QtWidgets.QDoubleSpinBox()
        self.heading_spin.setRange(-360.0, 360.0)
        self.heading_spin.setValue(0.0)
        self.heading_spin.setSuffix(" °")
        self.heading_spin.setEnabled(False)
        self.heading_cb.toggled.connect(self.heading_spin.setEnabled)
        h_row = QtWidgets.QHBoxLayout()
        h_row.addWidget(self.heading_cb)
        h_row.addWidget(self.heading_spin, 1)
        h_wrap = QtWidgets.QWidget()
        h_wrap.setLayout(h_row)
        algo_form.addRow("Heading (--heading-deg)", h_wrap)

        right_col.addWidget(algo_box)

        # ── GPS noise / bias ──────────────────────────────────────
        gps_box = QtWidgets.QGroupBox("GPS noise")
        gps_form = QtWidgets.QFormLayout(gps_box)

        self.gps_bias_cb = QtWidgets.QCheckBox(
            "Per-agent fixed bias (--gps-bias)")
        self.gps_bias_cb.setToolTip(
            "Give each agent a persistent installation-offset on its "
            "GPS readings. The magnitude is drawn uniformly from "
            "[0, METERS] per agent at init; direction is uniform on "
            "[0, 2π). The bias is added to every fix for the agent's "
            "whole session and is indistinguishable from a shifted "
            "world frame — the 3-state EKF cannot observe it out, so "
            "the agent converges to a waypoint that's offset by ~bias "
            "metres from the true one. Models antenna-placement / "
            "receiver-clock / site-multipath errors.")
        gps_form.addRow(self.gps_bias_cb)

        self.gps_bias_spin = QtWidgets.QDoubleSpinBox()
        self.gps_bias_spin.setRange(0.0, 10.0)
        self.gps_bias_spin.setSingleStep(0.25)
        self.gps_bias_spin.setValue(1.0)
        self.gps_bias_spin.setDecimals(2)
        self.gps_bias_spin.setSuffix(" m")
        self.gps_bias_spin.setEnabled(False)
        gps_form.addRow("Max magnitude", self.gps_bias_spin)
        self.gps_bias_cb.toggled.connect(self.gps_bias_spin.setEnabled)

        right_col.addWidget(gps_box)

        # ── Launch-stack subsystems ──────────────────────────────
        # Toggle individual ROS2 launch-stack packages on/off to
        # simulate field scenarios where a node failed to start. Each
        # checkbox maps to a single CLI flag in gps_sim_gui.py.
        stack_box = QtWidgets.QGroupBox("Launch stack (uncheck = node down)")
        stack_form = QtWidgets.QVBoxLayout(stack_box)

        self.stack_preslam_cb = QtWidgets.QCheckBox(
            "PRESLAM / LIDAR-IMU fusion")
        self.stack_preslam_cb.setChecked(True)
        self.stack_preslam_cb.setToolTip(
            "Real-robot nodes: robot_localization::ekf_node "
            "(ekf_local) + imu_cov_inflator. When unchecked: passes "
            "--no-preslam. The local EKF doesn't fuse IMU yaw with "
            "wheel-encoder yaw → gps_handler receives raw biased "
            "odom. Reproduces the May 9 orbit signature.\n\n"
            "Note: the SICK MultiScan supplies the IMU on this "
            "robot, so a LIDAR-driver outage looks identical — they "
            "are physically inseparable. No separate '--no-lidar' "
            "toggle for this reason.")
        stack_form.addWidget(self.stack_preslam_cb)

        self.stack_gps_cb = QtWidgets.QCheckBox("GPS receiver")
        self.stack_gps_cb.setChecked(True)
        self.stack_gps_cb.setToolTip(
            "Real-robot node: gps_handler::gps_publisher (serial "
            "→ NavSatFix). When unchecked: passes --no-gps. The "
            "/gps_fix publisher never comes up — no samples reach "
            "the handler EKF, agent runs dead-reckoning only.")
        stack_form.addWidget(self.stack_gps_cb)

        self.stack_gps_ekf_cb = QtWidgets.QCheckBox(
            "GPS handler θ-EKF")
        self.stack_gps_ekf_cb.setChecked(True)
        self.stack_gps_ekf_cb.setToolTip(
            "Real-robot node: gps_waypoint_handler::gps_handler_node "
            "(Python — WGS84→local projection + 3-state EKF). When "
            "unchecked: passes --no-gps-ekf. /gps_fix samples still "
            "arrive but the (x_world, y_world, θ) EKF doesn't fuse "
            "them — heading-offset estimate stays at its init "
            "value.")
        stack_form.addWidget(self.stack_gps_ekf_cb)

        self.stack_lever_arm_cb = QtWidgets.QCheckBox(
            "Antenna lever-arm (URDF TF)")
        self.stack_lever_arm_cb.setChecked(True)
        self.stack_lever_arm_cb.setToolTip(
            "Real-robot dependency: robot_state_publisher publishing "
            "the URDF static transform antenna_link → base_link "
            "(NOT slam_toolbox — that's a different TF chain). "
            "When unchecked: passes --no-lever-arm. The antenna "
            "offset isn't subtracted in the gps_callback, so a "
            "heading-locked systematic bias shows up in every fix "
            "and the EKF locks onto it.")
        stack_form.addWidget(self.stack_lever_arm_cb)

        self.stack_nav2_cb = QtWidgets.QCheckBox("NAV2 (A* planning)")
        self.stack_nav2_cb.setChecked(True)
        self.stack_nav2_cb.setToolTip(
            "Real-robot nodes: nav2_planner_server (NavFn), "
            "nav2_controller_server (DWB), nav2_bt_navigator. "
            "When unchecked: passes --no-nav2. No A* replans — "
            "the agent keeps its initial path if any, else the "
            "controller bee-lines toward the candidate goal.")
        stack_form.addWidget(self.stack_nav2_cb)

        not_modeled = QtWidgets.QLabel(
            "<span style='color: #888888'>Not modeled in this sim: "
            "CAMERA (ZED), LINEDETECT (line_detector), PCADETECT "
            "(grade_detector) — these feed Nav2's costmap obstacle "
            "layer on the robot, but this sim uses a static "
            "obstacle layout. SLAM proper (slam_toolbox publishing "
            "map→odom) also isn't modeled — the sim works in a "
            "single world frame.</span>")
        not_modeled.setWordWrap(True)
        stack_form.addWidget(not_modeled)

        right_col.addWidget(stack_box)

        # ── Goal override ─────────────────────────────────────────
        goal_box = QtWidgets.QGroupBox("Goal override (optional)")
        goal_form = QtWidgets.QFormLayout(goal_box)

        # Chained 3-waypoint mission (exercises the preemptive
        # next-goal cache + per-leg baseline reset path on
        # ``origin/improve/gps-waypoint-continuity``). When checked
        # the explicit single-goal lat/lon inputs are disabled and
        # ``--mission three-waypoint`` is passed instead.
        self.mission_cb = QtWidgets.QCheckBox(
            "Three-waypoint mission (chained + cached)")
        self.mission_cb.setToolTip(
            "Run the canonical 3 GPS waypoints from the deployed "
            "stored_waypoints.txt as a chained mission. Exercises "
            "the preemptive next-goal cache + per-leg baseline "
            "reset path from origin/improve/gps-waypoint-continuity "
            "— EKF / heading-fit state are preserved across legs "
            "while the candidate-goal smoother and moving-away "
            "window are cleared on each leg switch. Passes "
            "--mission three-waypoint; overrides --goal-lat / "
            "--goal-lon.")
        goal_form.addRow(self.mission_cb)

        self.goal_cb = QtWidgets.QCheckBox("Set explicit goal lat/lon")
        goal_form.addRow(self.goal_cb)

        self.goal_lat = QtWidgets.QDoubleSpinBox()
        self.goal_lat.setDecimals(6)
        self.goal_lat.setRange(-90.0, 90.0)
        self.goal_lat.setValue(37.23027)
        self.goal_lat.setEnabled(False)
        goal_form.addRow("Latitude (--goal-lat)", self.goal_lat)

        self.goal_lon = QtWidgets.QDoubleSpinBox()
        self.goal_lon.setDecimals(6)
        self.goal_lon.setRange(-180.0, 180.0)
        self.goal_lon.setValue(-80.42504)
        self.goal_lon.setEnabled(False)
        goal_form.addRow("Longitude (--goal-lon)", self.goal_lon)

        self.goal_cb.toggled.connect(self.goal_lat.setEnabled)
        self.goal_cb.toggled.connect(self.goal_lon.setEnabled)

        # Three-waypoint mission and explicit single-goal override
        # are mutually exclusive. When the mission box is checked,
        # disable the single-goal control AND its child spin boxes.
        def _sync_mission():
            mission_on = self.mission_cb.isChecked()
            self.goal_cb.setEnabled(not mission_on)
            if mission_on:
                # Uncheck the single-goal toggle so its preview
                # contribution drops, AND force the spin boxes
                # disabled regardless of the toggle's prior state.
                self.goal_cb.setChecked(False)
                self.goal_lat.setEnabled(False)
                self.goal_lon.setEnabled(False)
        self.mission_cb.toggled.connect(_sync_mission)
        _sync_mission()

        left_col.addWidget(goal_box)

        # ── Headless (advanced) ──────────────────────────────────
        headless_box = QtWidgets.QGroupBox("Headless (no window)")
        headless_form = QtWidgets.QFormLayout(headless_box)

        self.headless_cb = QtWidgets.QCheckBox("Headless run (--headless)")
        headless_form.addRow(self.headless_cb)

        self.headless_steps = QtWidgets.QSpinBox()
        self.headless_steps.setRange(1, 100000)
        self.headless_steps.setValue(200)
        self.headless_steps.setEnabled(False)
        headless_form.addRow("Steps (--headless-steps)", self.headless_steps)

        self.full_steps_cb = QtWidgets.QCheckBox(
            "Force full step count (--full-steps)")
        self.full_steps_cb.setEnabled(False)
        headless_form.addRow(self.full_steps_cb)

        self.headless_cb.toggled.connect(self.headless_steps.setEnabled)
        self.headless_cb.toggled.connect(self.full_steps_cb.setEnabled)

        right_col.addWidget(headless_box)

        left_col.addStretch(1)
        right_col.addStretch(1)

        # ── Preview + launch ─────────────────────────────────────
        self.preview = QtWidgets.QLineEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(_mono_font())
        root.addWidget(self.preview)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.bake_btn = QtWidgets.QPushButton("Bake MP4…")
        self.bake_btn.setToolTip(
            "Render the chosen scenario directly to an MP4 at 60 fps. "
            "No GUI window opens — the sim runs fully headless. Uses "
            "all current options (scenario, agents, seed, gps-bias, "
            "no-ekf, etc.).")
        self.launch_btn = QtWidgets.QPushButton("Launch")
        self.launch_btn.setDefault(True)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.bake_btn)
        buttons.addWidget(self.launch_btn)
        root.addLayout(buttons)

        self.cancel_btn.clicked.connect(self.close)
        self.bake_btn.clicked.connect(self._bake)
        self.launch_btn.clicked.connect(self._launch)

        # Refresh preview on every change.
        for w in self.findChildren(QtWidgets.QAbstractButton):
            w.toggled.connect(self._refresh_preview)
        for w in self.findChildren(QtWidgets.QAbstractSpinBox):
            w.valueChanged.connect(self._refresh_preview)
        # Initial pass — sync the World Counts group with the default
        # (Scripted) preset so it starts greyed-out and labelled.
        self._on_preset_changed()
        self._refresh_preview()

    def _build_args(self):
        args = []
        idx = self.preset_group.checkedId()
        flag = SCENARIO_PRESETS[idx][1]
        if flag:
            args.append(flag)

        if self.single_cb.isChecked():
            args.append("--single")
        else:
            args.extend(["--agents", str(self.agents_spin.value())])

        if self.scatter_cb.isChecked():
            args.append("--scatter")

        args.extend(["--seed", str(self.seed_spin.value())])

        # World counts only matter for --random. Scripted uses a
        # fixed layout (1 obstacle / 1 roof / 1 projector); --real /
        # --crazy override these counts with their preset values.
        # Including the flags in any other case is misleading —
        # they show in the command line but the sim ignores them.
        idx = self.preset_group.checkedId()
        flag = SCENARIO_PRESETS[idx][1]
        if flag == "--random":
            args.extend(["--obstacles",  str(self.obstacles_spin.value())])
            args.extend(["--roofs",      str(self.roofs_spin.value())])
            args.extend(["--projectors", str(self.projectors_spin.value())])
            args.extend(["--jammers",    str(self.jammers_spin.value())])
            args.extend(["--foliage",    str(self.foliage_spin.value())])
            args.extend(["--spoofers",   str(self.spoofers_spin.value())])

        if self.no_ekf_cb.isChecked():
            args.append("--no-ekf")
        if self.coldstart_cb.isChecked():
            args.append("--coldstart-bias-enable")
        if self.gps_bias_cb.isChecked() and self.gps_bias_spin.value() > 0.0:
            args.extend(["--gps-bias", f"{self.gps_bias_spin.value():g}"])
        # Per-subsystem stack toggles (unchecked = node down).
        if not self.stack_preslam_cb.isChecked():
            args.append("--no-preslam")
        if not self.stack_gps_cb.isChecked():
            args.append("--no-gps")
        if not self.stack_gps_ekf_cb.isChecked():
            args.append("--no-gps-ekf")
        if not self.stack_lever_arm_cb.isChecked():
            args.append("--no-lever-arm")
        if not self.stack_nav2_cb.isChecked():
            args.append("--no-nav2")
        if self.heading_cb.isChecked():
            args.extend(["--heading-deg", f"{self.heading_spin.value():g}"])
        if self.mission_cb.isChecked():
            # Mission mode overrides single-goal lat/lon (the
            # documented precedence in gps_sim_gui.py).
            args.extend(["--mission", "three-waypoint"])
        elif self.goal_cb.isChecked():
            args.extend(["--goal-lat", f"{self.goal_lat.value():.6f}"])
            args.extend(["--goal-lon", f"{self.goal_lon.value():.6f}"])

        if self.headless_cb.isChecked():
            args.append("--headless")
            args.extend(["--headless-steps",
                         str(self.headless_steps.value())])
            if self.full_steps_cb.isChecked():
                args.append("--full-steps")

        return args

    def _on_preset_changed(self):
        """Update the World Counts group's enabled state + caption
        every time the scenario radio button changes. The sim only
        consumes those spinboxes when the ``--random`` preset is
        active; everything else (Scripted / Real / Crazy) uses a
        fixed layout. Greying out the group makes that obvious."""
        idx = self.preset_group.checkedId()
        if idx < 0 or idx >= len(SCENARIO_PRESETS):
            return
        label, flag, _tip = SCENARIO_PRESETS[idx]
        is_random = (flag == "--random")
        self.world_box.setEnabled(is_random)
        if is_random:
            self.world_box.setTitle("World counts (used by Random)")
        elif flag is None:
            self.world_box.setTitle(
                "World counts (Scripted uses fixed layout — "
                "1 obstacle / 1 roof / 1 projector)")
        elif flag == "--real":
            self.world_box.setTitle(
                "World counts (Real preset uses its own counts)")
        else:
            self.world_box.setTitle(
                "World counts (Crazy preset uses its own counts)")
        self._refresh_preview()

    def _refresh_preview(self):
        args = self._build_args()
        self.preview.setText("gps_sim_gui.py " + " ".join(args))

    def _launch(self):
        args = self._build_args()
        argv = [sys.executable, str(SIM_SCRIPT)] + args
        # Hand off to the sim. subprocess handles argv[0]-with-spaces
        # correctly on Windows (CreateProcess gets a properly quoted
        # command line); os.execvp does not, which breaks any install
        # path containing a space like "GPS Sim".
        self.close()
        try:
            subprocess.Popen(argv, cwd=str(HERE))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                None, "Launch failed", f"Could not start sim:\n{exc}")
            sys.exit(1)
        # Force the Qt event loop to exit cleanly so the launcher
        # process actually terminates after handing off to the sim.
        # Without this, ``app.exec_()`` can linger on macOS even
        # after every window is closed, leaving an orphan launcher
        # process visible in the dock / Activity Monitor.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def _bake(self):
        """Save dialog → exec gps_sim_gui.py --bake-mp4 PATH.

        Pipes every current launcher selection into the bake (scenario,
        agents, seed, gps-bias, no-ekf, …) plus the --bake-mp4 flag, so
        the offscreen render exactly matches whatever the user had
        configured for a live run. No GUI window appears.

        Replaces this process via ``os.execvp`` instead of detaching a
        child with ``Popen`` — that's the LiDAR-sim fix verbatim. On
        macOS, a Popen child orphans when the bash wrapper exits and
        the Terminal's pty tears down, so the bake silently dies with
        no output. ``execvp`` keeps a single process attached to the
        terminal the whole way, so all the ``[bake-mp4] …`` progress
        prints show up live."""
        from datetime import datetime
        BAKES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(BAKES_DIR / f"gps_sim_{stamp}.mp4")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Bake MP4", default_path,
            "MP4 video (*.mp4);;All files (*)")
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"
        args = self._build_args()
        # --bake-mp4 takes precedence over --headless inside
        # gps_sim_gui; we don't strip --headless if the user happened
        # to pick it.
        args.extend(["--bake-mp4", out_path, "--bake-fps", str(BAKE_FPS)])
        argv = [sys.executable, str(SIM_SCRIPT)] + args
        self.close()
        os.chdir(str(HERE))
        try:
            os.execvp(argv[0], argv)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                None, "Bake failed",
                f"Could not start bake:\n{exc}")
            sys.exit(1)


def _mono_font():
    f = QtCore.QFileInfo  # noqa: F841 - placeholder if we ever need it
    font = QtWidgets.QApplication.font()
    font.setFamily("Consolas")
    font.setStyleHint(font.Monospace)
    return font


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
