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
        self.setMinimumWidth(540)

        root = QtWidgets.QVBoxLayout(self)

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
        root.addWidget(preset_box)

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

        root.addWidget(agents_box)

        # Disable --agents while --single is checked.
        def _sync_single():
            self.agents_spin.setEnabled(not self.single_cb.isChecked())
        self.single_cb.toggled.connect(_sync_single)
        _sync_single()

        # ── World counts ──────────────────────────────────────────
        world_box = QtWidgets.QGroupBox(
            "World counts (overridden by Real / Crazy presets)")
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

        root.addWidget(world_box)

        # ── Robot / EKF ───────────────────────────────────────────
        algo_box = QtWidgets.QGroupBox("Algorithm / robot")
        algo_form = QtWidgets.QFormLayout(algo_box)

        self.no_ekf_cb = QtWidgets.QCheckBox(
            "Disable θ-EKF + LIDAR/IMU fusion (--no-ekf)")
        self.no_ekf_cb.setToolTip(
            "Field-test config: raw biased wheel odom, no fusion. "
            "Reproduces the orbit signature from May 9 runs.")
        algo_form.addRow(self.no_ekf_cb)

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

        root.addWidget(algo_box)

        # ── Goal override ─────────────────────────────────────────
        goal_box = QtWidgets.QGroupBox("Goal override (optional)")
        goal_form = QtWidgets.QFormLayout(goal_box)

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

        root.addWidget(goal_box)

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

        root.addWidget(headless_box)

        # ── Preview + launch ─────────────────────────────────────
        self.preview = QtWidgets.QLineEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(_mono_font())
        root.addWidget(self.preview)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.launch_btn = QtWidgets.QPushButton("Launch")
        self.launch_btn.setDefault(True)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.launch_btn)
        root.addLayout(buttons)

        self.cancel_btn.clicked.connect(self.close)
        self.launch_btn.clicked.connect(self._launch)

        # Refresh preview on every change.
        for w in self.findChildren(QtWidgets.QAbstractButton):
            w.toggled.connect(self._refresh_preview)
        for w in self.findChildren(QtWidgets.QAbstractSpinBox):
            w.valueChanged.connect(self._refresh_preview)
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

        # World counts only matter for scripted/random presets — but
        # the sim's --real / --crazy paths override them anyway, so
        # always pass them through. They're cheap and the user might
        # want to see what they were set to.
        args.extend(["--obstacles",  str(self.obstacles_spin.value())])
        args.extend(["--roofs",      str(self.roofs_spin.value())])
        args.extend(["--projectors", str(self.projectors_spin.value())])
        args.extend(["--jammers",    str(self.jammers_spin.value())])
        args.extend(["--foliage",    str(self.foliage_spin.value())])
        args.extend(["--spoofers",   str(self.spoofers_spin.value())])

        if self.no_ekf_cb.isChecked():
            args.append("--no-ekf")
        if self.heading_cb.isChecked():
            args.extend(["--heading-deg", f"{self.heading_spin.value():g}"])
        if self.goal_cb.isChecked():
            args.extend(["--goal-lat", f"{self.goal_lat.value():.6f}"])
            args.extend(["--goal-lon", f"{self.goal_lon.value():.6f}"])

        if self.headless_cb.isChecked():
            args.append("--headless")
            args.extend(["--headless-steps",
                         str(self.headless_steps.value())])
            if self.full_steps_cb.isChecked():
                args.append("--full-steps")

        return args

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
