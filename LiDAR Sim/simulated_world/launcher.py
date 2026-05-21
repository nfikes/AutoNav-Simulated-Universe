#!/usr/bin/env python3
"""Flag-picker panel that launches the LiDAR sim.

Two modes:
  * Interactive GUI    — click-to-place agent + scan (lidar_sim_gui.py)
  * Ramp benchmark     — 5/10/15 parallel agents crossing ramps
                         (scripts/test_ramp_crossing.py)
"""

import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


HERE = Path(__file__).resolve().parent
GUI_SCRIPT = HERE / "lidar_sim_gui.py"
BENCH_SCRIPT = HERE / "scripts" / "test_ramp_crossing.py"
ASSETS_DIR = HERE.parent / "3d_assets"


MODES = [
    ("gui",   "Interactive GUI  ·  click to place agent"),
    ("bench", "Ramp benchmark  ·  parallel agents on the 5 ramps"),
]
DEFAULT_MODE = "gui"

THRESHOLDS = [
    (15, "15% grade  ·  8.5°  ·  strict"),
    (20, "20% grade  ·  11.3°  ·  moderate"),
    (30, "30% grade  ·  16.7°  ·  real-robot default (competition)"),
]
DEFAULT_THRESHOLD = 30

RAY_PRESETS = [
    (500,   "500   · fastest"),
    (1000,  "1000  · light"),
    (2000,  "2000  · medium"),
    (5000,  "5000  · benchmark-ish"),
    (10000, "10000 · full fidelity"),
    (11520, "11520 · real robot (SICK multiScan165: 16 × 720)"),
]
DEFAULT_RAYS_INDEX = 5  # 11520

BENCH_AGENT_COUNTS = [
    (5,  "5  · first row only (fastest)"),
    (10, "10  · first two rows"),
    (15, "15  · all rows (full benchmark)"),
]
DEFAULT_BENCH_AGENTS = 15

BENCH_PHASES = [
    ("1",    "Phase 1 only  ·  cross the ramps"),
    ("both", "Both phases  ·  cross + route around walls"),
]
DEFAULT_BENCH_PHASE = "both"


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Sim — Launch Options")
        self.setMinimumWidth(560)

        root = QtWidgets.QVBoxLayout(self)

        # ── Mode (GUI vs benchmark) ──────────────────────────────
        mode_box = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        self.mode_group = QtWidgets.QButtonGroup(self)
        for i, (key, label) in enumerate(MODES):
            rb = QtWidgets.QRadioButton(label)
            rb.setProperty("mode_key", key)
            if key == DEFAULT_MODE:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i)
            mode_layout.addWidget(rb)
        root.addWidget(mode_box)

        # ── Grade threshold (both modes) ─────────────────────────
        thr_box = QtWidgets.QGroupBox("Grade tolerance (--threshold)")
        thr_layout = QtWidgets.QVBoxLayout(thr_box)
        self.thr_group = QtWidgets.QButtonGroup(self)
        for i, (val, label) in enumerate(THRESHOLDS):
            rb = QtWidgets.QRadioButton(label)
            rb.setProperty("threshold", val)
            if val == DEFAULT_THRESHOLD:
                rb.setChecked(True)
            self.thr_group.addButton(rb, i)
            thr_layout.addWidget(rb)
        root.addWidget(thr_box)

        # ── Ray count (both modes) ───────────────────────────────
        rays_box = QtWidgets.QGroupBox("Rays per scan (--rays)")
        rays_form = QtWidgets.QFormLayout(rays_box)

        self.rays_combo = QtWidgets.QComboBox()
        for val, label in RAY_PRESETS:
            self.rays_combo.addItem(label, val)
        self.rays_combo.setCurrentIndex(DEFAULT_RAYS_INDEX)
        rays_form.addRow("Preset", self.rays_combo)

        self.rays_custom_cb = QtWidgets.QCheckBox("Use custom value")
        rays_form.addRow(self.rays_custom_cb)

        self.rays_spin = QtWidgets.QSpinBox()
        self.rays_spin.setRange(100, 100000)
        self.rays_spin.setSingleStep(500)
        self.rays_spin.setValue(11520)
        self.rays_spin.setEnabled(False)
        rays_form.addRow("Custom rays", self.rays_spin)

        self.rays_custom_cb.toggled.connect(self.rays_spin.setEnabled)
        self.rays_custom_cb.toggled.connect(
            lambda on: self.rays_combo.setEnabled(not on))

        root.addWidget(rays_box)

        # ── GUI-only section: STL override ───────────────────────
        self.stl_box = QtWidgets.QGroupBox("Terrain mesh (optional, GUI mode)")
        stl_form = QtWidgets.QFormLayout(self.stl_box)

        self.stl_cb = QtWidgets.QCheckBox("Use a different STL")
        stl_form.addRow(self.stl_cb)

        stl_row = QtWidgets.QHBoxLayout()
        self.stl_path = QtWidgets.QLineEdit()
        self.stl_path.setEnabled(False)
        self.stl_browse = QtWidgets.QPushButton("Browse…")
        self.stl_browse.setEnabled(False)
        stl_row.addWidget(self.stl_path, 1)
        stl_row.addWidget(self.stl_browse)
        stl_wrap = QtWidgets.QWidget()
        stl_wrap.setLayout(stl_row)
        stl_form.addRow("STL path", stl_wrap)

        self.stl_cb.toggled.connect(self.stl_path.setEnabled)
        self.stl_cb.toggled.connect(self.stl_browse.setEnabled)
        self.stl_browse.clicked.connect(self._pick_stl)

        root.addWidget(self.stl_box)

        # ── Benchmark-only section ───────────────────────────────
        self.bench_box = QtWidgets.QGroupBox("Benchmark options")
        bench_form = QtWidgets.QFormLayout(self.bench_box)

        self.bench_agents_combo = QtWidgets.QComboBox()
        for val, label in BENCH_AGENT_COUNTS:
            self.bench_agents_combo.addItem(label, val)
        default_agents_idx = next(
            i for i, (v, _) in enumerate(BENCH_AGENT_COUNTS)
            if v == DEFAULT_BENCH_AGENTS)
        self.bench_agents_combo.setCurrentIndex(default_agents_idx)
        bench_form.addRow("Agents", self.bench_agents_combo)

        self.bench_phase_combo = QtWidgets.QComboBox()
        for key, label in BENCH_PHASES:
            self.bench_phase_combo.addItem(label, key)
        default_phase_idx = next(
            i for i, (k, _) in enumerate(BENCH_PHASES)
            if k == DEFAULT_BENCH_PHASE)
        self.bench_phase_combo.setCurrentIndex(default_phase_idx)
        bench_form.addRow("Phase", self.bench_phase_combo)

        self.bench_no_video_cb = QtWidgets.QCheckBox(
            "Skip gif export (--no-video, faster)")
        bench_form.addRow(self.bench_no_video_cb)

        root.addWidget(self.bench_box)

        # ── Preview + launch ─────────────────────────────────────
        self.preview = QtWidgets.QLineEdit()
        self.preview.setReadOnly(True)
        font = QtWidgets.QApplication.font()
        font.setFamily("Consolas")
        font.setStyleHint(font.Monospace)
        self.preview.setFont(font)
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

        for w in self.findChildren(QtWidgets.QAbstractButton):
            w.toggled.connect(self._refresh)
        self.rays_combo.currentIndexChanged.connect(self._refresh)
        self.rays_spin.valueChanged.connect(self._refresh)
        self.stl_path.textChanged.connect(self._refresh)
        self.bench_agents_combo.currentIndexChanged.connect(self._refresh)
        self.bench_phase_combo.currentIndexChanged.connect(self._refresh)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        self._on_mode_changed()
        self._refresh()

    def _pick_stl(self):
        start_dir = str(ASSETS_DIR) if ASSETS_DIR.exists() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick an STL", start_dir, "STL files (*.stl);;All files (*)")
        if path:
            self.stl_path.setText(path)

    def _selected_mode(self):
        btn = self.mode_group.checkedButton()
        return btn.property("mode_key") if btn else DEFAULT_MODE

    def _selected_threshold(self):
        btn = self.thr_group.checkedButton()
        return btn.property("threshold") if btn else DEFAULT_THRESHOLD

    def _selected_rays(self):
        if self.rays_custom_cb.isChecked():
            return self.rays_spin.value()
        return self.rays_combo.currentData()

    def _on_mode_changed(self):
        is_gui = self._selected_mode() == "gui"
        self.stl_box.setVisible(is_gui)
        self.bench_box.setVisible(not is_gui)
        self._refresh()

    def _build_args(self):
        if self._selected_mode() == "gui":
            args = ["--threshold", str(self._selected_threshold()),
                    "--rays",      str(self._selected_rays())]
            if self.stl_cb.isChecked() and self.stl_path.text().strip():
                args.append(self.stl_path.text().strip())
            return args
        # Benchmark mode
        args = ["--threshold", str(self._selected_threshold()),
                "--rays",      str(self._selected_rays()),
                "--agents",    str(self.bench_agents_combo.currentData()),
                "--phase",     str(self.bench_phase_combo.currentData())]
        if self.bench_no_video_cb.isChecked():
            args.append("--no-video")
        return args

    def _script_name(self):
        return ("lidar_sim_gui.py" if self._selected_mode() == "gui"
                else "scripts/test_ramp_crossing.py")

    def _refresh(self):
        self.preview.setText(self._script_name() + " " + " ".join(self._build_args()))

    def _launch(self):
        target = (GUI_SCRIPT if self._selected_mode() == "gui"
                  else BENCH_SCRIPT)
        argv = [sys.executable, str(target)] + self._build_args()
        self.close()
        try:
            subprocess.Popen(argv, cwd=str(HERE))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                None, "Launch failed", f"Could not start sim:\n{exc}")
            sys.exit(1)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
