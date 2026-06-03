#!/usr/bin/env python3
"""Flag-picker panel that launches bt_sim_gui.py.

Mirrors the pattern in `GPS Sim/simulated_world/launcher.py`: a small PyQt5
window with the most useful CLI flags grouped into sections. Click "Launch"
and the picker exits, replaced by the actual sim process.
"""

import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


HERE = Path(__file__).resolve().parent
SIM_SCRIPT = HERE / "bt_sim_gui.py"
DATA_DIR = HERE.parent / "data"
BAKES_DIR = HERE.parent / "bakes"


def _autodiscover_sprite() -> Path:
    """Same priority as bt_sim_gui._resolve_sprite_path:
    robot_top.png if present, else first *.png in data/, else the
    default path (won't exist — used as a placeholder)."""
    default = DATA_DIR / "robot_top.png"
    if default.exists():
        return default
    if DATA_DIR.exists():
        pngs = sorted(DATA_DIR.glob("*.png"))
        if pngs:
            return pngs[0]
    return default


SPRITE_DEFAULT = _autodiscover_sprite()


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BEHAVIOR TREE Sim — Launch Options")
        self.setMinimumWidth(640)

        root = QtWidgets.QVBoxLayout(self)

        # ── World layout ──
        maze_box = QtWidgets.QGroupBox("World")
        maze_form = QtWidgets.QFormLayout(maze_box)

        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItem("Circular track (bisected, one dead end)",
                                  userData="track")
        self.layout_combo.addItem("DFS maze + obstacles", userData="maze")
        self.layout_combo.setCurrentIndex(0)
        self.layout_combo.setToolTip(
            "Track = wavy annular track with parallel divider and one "
            "sealed end (robot must pick the right direction). Maze = "
            "DFS perfect maze with random obstacle rectangles.")
        maze_form.addRow("Layout (--layout)", self.layout_combo)

        self.cells_spin = QtWidgets.QSpinBox()
        self.cells_spin.setRange(2, 12)
        self.cells_spin.setValue(7)
        self.cells_spin.setToolTip(
            "Maze is N×N cells of 5 m each (only used by maze layout).")
        maze_form.addRow("Cells per side (--maze-cells)", self.cells_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(7)
        maze_form.addRow("Random seed (--seed)", self.seed_spin)

        self.obstacles_spin = QtWidgets.QSpinBox()
        self.obstacles_spin.setRange(0, 200)
        self.obstacles_spin.setValue(6)
        self.obstacles_spin.setToolTip(
            "Random rectangular obstacles dropped into corridor cells. "
            "Creates surprise dead-ends that fire the BT recovery "
            "behaviors (BACKUP, CLEAR_AROUND_ROBOT, GOAL_BEND, …).")
        maze_form.addRow("Obstacles (--obstacles)", self.obstacles_spin)

        self.heading_cb = QtWidgets.QCheckBox("Override initial heading")
        self.heading_spin = QtWidgets.QDoubleSpinBox()
        self.heading_spin.setRange(-360.0, 360.0)
        self.heading_spin.setValue(45.0)
        self.heading_spin.setSuffix(" °")
        self.heading_spin.setEnabled(False)
        self.heading_cb.toggled.connect(self.heading_spin.setEnabled)
        h_row = QtWidgets.QHBoxLayout()
        h_row.addWidget(self.heading_cb)
        h_row.addWidget(self.heading_spin, 1)
        h_wrap = QtWidgets.QWidget()
        h_wrap.setLayout(h_row)
        maze_form.addRow("Heading (--heading-deg)", h_wrap)
        root.addWidget(maze_box)

        # ── Sprite ──
        sprite_box = QtWidgets.QGroupBox("Robot sprite")
        sprite_form = QtWidgets.QFormLayout(sprite_box)
        self.sprite_edit = QtWidgets.QLineEdit(str(SPRITE_DEFAULT))
        self.sprite_edit.setToolTip(
            "Top-down PNG of the robot with #E48787 marker pixels at the "
            "two knife edges, COM, and front caster. Leave at the default "
            "path to use data/robot_top.png; uncheck below to force stick "
            "figures instead.")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._pick_sprite)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.sprite_edit, 1)
        srow.addWidget(browse)
        srow_wrap = QtWidgets.QWidget()
        srow_wrap.setLayout(srow)
        sprite_form.addRow("Path (--sprite)", srow_wrap)
        self.no_sprite_cb = QtWidgets.QCheckBox(
            "Force stick figure (--no-sprite)")
        sprite_form.addRow(self.no_sprite_cb)
        root.addWidget(sprite_box)

        # ── Sensor noise ──
        noise_box = QtWidgets.QGroupBox("Sensor noise")
        noise_form = QtWidgets.QFormLayout(noise_box)
        self.detection_spin = QtWidgets.QDoubleSpinBox()
        self.detection_spin.setRange(0.10, 1.00)
        self.detection_spin.setSingleStep(0.05)
        self.detection_spin.setDecimals(2)
        self.detection_spin.setValue(0.85)
        self.detection_spin.setToolTip(
            "Per-ray probability that a wall voxel hit is actually "
            "detected. < 1.0 simulates noisy lidar — missed hits leak "
            "through as false negatives.")
        noise_form.addRow("Wall detect prob (--detection-prob)",
                          self.detection_spin)
        root.addWidget(noise_box)

        # ── Headless / scatter / bake ──
        headless_box = QtWidgets.QGroupBox(
            "Batch modes (mutually exclusive with the GUI)")
        headless_form = QtWidgets.QFormLayout(headless_box)
        self.headless_cb = QtWidgets.QCheckBox("Headless run (--headless)")
        headless_form.addRow(self.headless_cb)
        self.headless_steps = QtWidgets.QSpinBox()
        self.headless_steps.setRange(100, 1_000_000)
        self.headless_steps.setSingleStep(1000)
        self.headless_steps.setValue(20000)
        self.headless_steps.setEnabled(False)
        headless_form.addRow("Physics ticks (--headless-steps)",
                             self.headless_steps)
        self.headless_cb.toggled.connect(self.headless_steps.setEnabled)

        self.scatter_spin = QtWidgets.QSpinBox()
        self.scatter_spin.setRange(0, 1000)
        self.scatter_spin.setValue(0)
        self.scatter_spin.setToolTip(
            "Run N robots in headless scatter mode. Seeds are offset by "
            "7919 per robot. End-of-run prints per-algorithm BT firing "
            "counts. 0 = scatter disabled.")
        headless_form.addRow("Scatter N robots (--scatter)",
                             self.scatter_spin)
        self.scatter_secs = QtWidgets.QDoubleSpinBox()
        self.scatter_secs.setRange(30.0, 3600.0)
        self.scatter_secs.setSingleStep(30.0)
        self.scatter_secs.setValue(400.0)
        self.scatter_secs.setSuffix(" s")
        headless_form.addRow("Per-robot sim cap (--scatter-max-secs)",
                             self.scatter_secs)
        root.addWidget(headless_box)

        # ── Preview + buttons ──
        self.preview = QtWidgets.QLineEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(_mono_font())
        root.addWidget(self.preview)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.bake_btn = QtWidgets.QPushButton("Bake MP4…")
        self.bake_btn.setToolTip(
            "Render the chosen scenario to an MP4 offscreen (no window). "
            "Uses --bake-mp4 + imageio-ffmpeg. All other options on this "
            "panel are honoured.")
        self.launch_btn = QtWidgets.QPushButton("Launch")
        # Do NOT setDefault(True) on the launch button — on macOS Qt 5.15,
        # the default button auto-fires when the window is shown, so the
        # panel immediately spawns the sim with the default args before the
        # user can even interact. Same bug the LiDAR sim README documents.
        self.launch_btn.setAutoDefault(False)
        self.cancel_btn.setAutoDefault(False)
        self.bake_btn.setAutoDefault(False)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.bake_btn)
        buttons.addWidget(self.launch_btn)
        root.addLayout(buttons)
        self.cancel_btn.clicked.connect(self.close)
        self.launch_btn.clicked.connect(self._launch)
        self.bake_btn.clicked.connect(self._bake)

        for w in self.findChildren(QtWidgets.QAbstractButton):
            w.toggled.connect(self._refresh_preview)
        for w in self.findChildren(QtWidgets.QAbstractSpinBox):
            w.valueChanged.connect(self._refresh_preview)
        for w in self.findChildren(QtWidgets.QComboBox):
            w.currentIndexChanged.connect(self._refresh_preview)
        self.sprite_edit.textChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _pick_sprite(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pick robot sprite", str(SPRITE_DEFAULT.parent),
            "PNG (*.png);;All files (*)")
        if path:
            self.sprite_edit.setText(path)

    def _build_args(self) -> list[str]:
        args: list[str] = []
        args.extend(["--layout", self.layout_combo.currentData()])
        args.extend(["--maze-cells", str(self.cells_spin.value())])
        args.extend(["--seed", str(self.seed_spin.value())])
        args.extend(["--obstacles", str(self.obstacles_spin.value())])
        args.extend(["--detection-prob", f"{self.detection_spin.value():.2f}"])
        if self.heading_cb.isChecked():
            args.extend(["--heading-deg", f"{self.heading_spin.value():g}"])
        if self.no_sprite_cb.isChecked():
            args.append("--no-sprite")
        elif self.sprite_edit.text().strip() and \
                Path(self.sprite_edit.text().strip()) != SPRITE_DEFAULT:
            args.extend(["--sprite", self.sprite_edit.text().strip()])
        # Batch modes (scatter > 0 takes precedence over headless inside
        # the sim's dispatch).
        if self.scatter_spin.value() > 0:
            args.extend(["--scatter", str(self.scatter_spin.value())])
            args.extend(["--scatter-max-secs",
                         f"{self.scatter_secs.value():g}"])
        elif self.headless_cb.isChecked():
            args.append("--headless")
            args.extend(["--headless-steps", str(self.headless_steps.value())])
        return args

    def _bake(self):
        """Pick a save path then exec the sim in MP4-bake mode."""
        BAKES_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(BAKES_DIR / f"bt_sim_{stamp}.mp4")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Bake MP4", default_path,
            "MP4 video (*.mp4);;All files (*)")
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"
        args = self._build_args()
        args.extend(["--bake-mp4", out_path])
        argv = [sys.executable, str(SIM_SCRIPT)] + args
        self.close()
        os.chdir(str(HERE))
        try:
            os.execvp(argv[0], argv)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                None, "Bake failed", f"Could not start bake:\n{exc}")
            sys.exit(1)

    def _refresh_preview(self):
        self.preview.setText("bt_sim_gui.py " + " ".join(self._build_args()))

    def _launch(self):
        """Replace this process with the sim via os.execvp.

        Using subprocess.Popen + app.quit() detaches the sim as a child
        process; macOS sends SIGHUP to the process group when the .command
        Terminal window closes, which kills the freshly-spawned sim before
        its window appears. exec preserves the PID — the .command's bash
        waits on the sim directly, Terminal stays open while the sim runs,
        and everything closes cleanly when the sim exits.
        """
        argv = [sys.executable, str(SIM_SCRIPT)] + self._build_args()
        self.close()
        os.chdir(str(HERE))
        try:
            os.execvp(argv[0], argv)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                None, "Launch failed", f"Could not start sim:\n{exc}")
            sys.exit(1)


def _mono_font():
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
