"""Run the real AutoNav HUD (`hud_node.py`) in a faked-up ROS universe.

Workflow:
    1. Register fake `rclpy` / `sensor_msgs` / `nav_msgs` / etc. modules in
       sys.modules via `fake_ros.install()`. After that, `import hud_node`
       picks the real GUI's source as-is with `_HAS_ROS = True`.
    2. Build a `HudNode` (the real ROS Node subclass). Every
       `create_subscription` it makes is recorded by the fake Node base
       at `node._fake_subs[topic]` — we use that to deliver synthetic
       messages directly to the real `_cb_*` callbacks.
    3. Build a `HudWindow` (the real Qt main window).
    4. Spin up a Qt timer that ticks ~10 Hz, generates synthetic messages
       (camera frame, lidar scan, GPS, odom, electrical), and delivers
       them through the recorded callbacks. The HudWindow's `_live_tick`
       then renders from `node.latest_*` exactly the way it would in
       production.

Run from this directory:
    python runner.py
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
import time
from types import SimpleNamespace

# Step 1: install fake ROS BEFORE importing the real GUI.
import fake_ros
fake_ros.install(verbose=False)

# Step 1b: fake the Docker side. The real GUI's "Connect to Container"
# runs `docker ps --filter status=running ... name=^/<container>$` and
# only flips `_container_connected = True` when the output is non-empty.
# We don't have Docker on this dev machine, so we patch subprocess.run
# to return a fake "container is up" result for docker invocations and
# Popen to return a no-op handle for docker exec — that gets us into
# Test Mode + the other container-gated pages without crashing the
# real subprocess pipeline.
_real_subprocess_run = subprocess.run
_real_subprocess_popen = subprocess.Popen


class _FakeCompletedProcess:
    def __init__(self, stdout='fake-container-id\n', returncode=0):
        self.stdout = stdout
        self.stderr = ''
        self.returncode = returncode


class _FakePopen:
    """Minimal stand-in for subprocess.Popen — pretends the child is
    running, exposes a never-closing stdout pipe, and accepts terminate
    / kill / wait without error."""
    def __init__(self):
        self.stdout = iter(())  # nothing to read, callers iterate cleanly
        self.stderr = None
        self.returncode = None
        self.pid = -1
    def poll(self):
        return None  # still "running"
    def terminate(self):
        self.returncode = -15
    def kill(self):
        self.returncode = -9
    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode
    def communicate(self, *a, **kw):
        return ('', '')


def _cmd_is_docker(cmd):
    if isinstance(cmd, (list, tuple)) and cmd:
        return 'docker' in str(cmd[0])
    if isinstance(cmd, str):
        return 'docker' in cmd
    return False


def _fake_subprocess_run(cmd, *args, **kwargs):
    if _cmd_is_docker(cmd):
        return _FakeCompletedProcess()
    return _real_subprocess_run(cmd, *args, **kwargs)


def _fake_subprocess_popen(cmd, *args, **kwargs):
    if _cmd_is_docker(cmd):
        return _FakePopen()
    return _real_subprocess_popen(cmd, *args, **kwargs)


subprocess.run = _fake_subprocess_run
subprocess.Popen = _fake_subprocess_popen

import numpy as np

# Step 2: now safe to import the real GUI.
import hud_node


# ── Fake-message builders ────────────────────────────────────────────────

def _header(stamp_ns):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9,
        ),
        frame_id='base_link',
    )


def make_image_msg(rgb_array, stamp_ns):
    """Wrap an HxWx3 uint8 RGB array as a sensor_msgs/Image clone.
    Encoding is set to 'rgb8'; the real `_cb_image` flips R/B for bgr8/bgra8
    branches and leaves rgb8 alone. step = W * 3 for the rgb8 layout.
    """
    h, w = rgb_array.shape[:2]
    return SimpleNamespace(
        header=_header(stamp_ns),
        height=h, width=w,
        encoding='rgb8',
        is_bigendian=0,
        step=w * 3,
        data=rgb_array.tobytes(),
    )


def make_mono_image_msg(mono_array, stamp_ns):
    """sensor_msgs/Image with MONO8 encoding (mask topic)."""
    h, w = mono_array.shape[:2]
    return SimpleNamespace(
        header=_header(stamp_ns),
        height=h, width=w,
        encoding='mono8',
        is_bigendian=0,
        step=w,
        data=mono_array.tobytes(),
    )


def make_laser_scan(ranges, stamp_ns,
                     angle_min=-math.pi, angle_max=math.pi,
                     range_min=0.1, range_max=10.0):
    inc = (angle_max - angle_min) / max(1, len(ranges))
    return SimpleNamespace(
        header=_header(stamp_ns),
        angle_min=angle_min, angle_max=angle_max, angle_increment=inc,
        time_increment=0.0, scan_time=0.1,
        range_min=range_min, range_max=range_max,
        ranges=np.asarray(ranges, dtype=np.float32),
        intensities=np.zeros(len(ranges), dtype=np.float32),
    )


def make_navsat_fix(lat, lon, alt, cov_ee, cov_nn, stamp_ns):
    """NavSatFix with a valid ENU position_covariance so the GUI graduates
    to its "GPS Covariance [fix]" tier."""
    cov = [0.0] * 9
    cov[0] = cov_ee
    cov[4] = cov_nn
    cov[8] = 1.0
    return SimpleNamespace(
        header=_header(stamp_ns),
        status=SimpleNamespace(status=0, service=1),
        latitude=lat, longitude=lon, altitude=alt,
        position_covariance=cov,
        position_covariance_type=2,
    )


def make_odom(x, y, theta, stamp_ns):
    qz = math.sin(theta / 2.0)
    qw = math.cos(theta / 2.0)
    return SimpleNamespace(
        header=_header(stamp_ns),
        child_frame_id='base_link',
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            ),
            covariance=[0.0] * 36,
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
            covariance=[0.0] * 36,
        ),
    )


def make_float32(value):
    return SimpleNamespace(data=float(value))


def make_bool(value):
    return SimpleNamespace(data=bool(value))


# ── Synthetic-data driver ────────────────────────────────────────────────

class FakeDataDriver:
    """Periodically generates and delivers synthetic ROS messages to the
    real HudNode's callbacks. Lives on the main thread; ticked by a QTimer.

    Generated streams (each delivered IF the GUI has subscribed):
        /zed/zed_node/rgb/color/rect/image  -> shifting RGB gradient
        /scan_fullframe                     -> circular room w/ moving blobs
        /gps_fix                            -> slow drift around VT campus,
                                                with valid covariance
        /odom + /local_ekf/odom             -> slow figure-8
        /electrical/{voltage,current,power,soc}
        /autonomous_mode                    -> False
    """

    def __init__(self, node):
        self._node = node
        self._t0 = time.monotonic()
        self._frame = 0

        # GPS base: Virginia Tech campus.
        self._gps_lat0 = 37.2296
        self._gps_lon0 = -80.4139

        # Odom state — accumulates a slow figure-8.
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0

    def _deliver(self, topic, msg):
        """Call the registered callback for `topic`, if any."""
        sub = self._node._fake_subs.get(topic)
        if sub is None:
            return
        try:
            sub.callback(msg)
        except Exception as e:  # noqa: BLE001
            print(f'WARN: driver callback {topic} raised: {e}', file=sys.stderr)

    def tick(self):
        """Generate one tick of fake data. Call at ~10 Hz."""
        t = time.monotonic() - self._t0
        self._frame += 1
        now_ns = int(time.time() * 1e9)

        # --- Camera frame: shifting colour gradient at 480 wide ---
        w, h = 480, 360
        xs = np.linspace(0, 1, w, dtype=np.float32)
        ys = np.linspace(0, 1, h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        r = np.clip((np.sin(xg * 4 + t) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        g = np.clip((np.cos(yg * 4 + t * 0.7) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        b = np.clip((np.sin((xg + yg) * 3 - t * 0.5) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        self._deliver('/zed/zed_node/rgb/color/rect/image',
                       make_image_msg(rgb, now_ns))

        # --- LaserScan: 360-beam circular room with moving objects ---
        ranges = []
        for i in range(360):
            a = -math.pi + i * (2 * math.pi / 360)
            r_ = 6.0 + random.uniform(-0.05, 0.05)
            # Two slowly orbiting "obstacles"
            for obj_a in (math.fmod(t * 0.3, 2 * math.pi) - math.pi,
                          math.fmod(t * 0.5 + math.pi, 2 * math.pi) - math.pi):
                if abs(a - obj_a) < 0.12:
                    r_ = 3.0
            ranges.append(r_)
        self._deliver('/scan_fullframe', make_laser_scan(ranges, now_ns))

        # --- GPS: slow drift around campus, with valid covariance ---
        drift_lat = 0.0001 * math.sin(t * 0.05)
        drift_lon = 0.0001 * math.cos(t * 0.07)
        cov_ee = 0.5 + 0.2 * math.sin(t * 0.4)
        cov_nn = 0.5 + 0.2 * math.cos(t * 0.4)
        self._deliver('/gps_fix', make_navsat_fix(
            self._gps_lat0 + drift_lat, self._gps_lon0 + drift_lon,
            621.0, cov_ee, cov_nn, now_ns,
        ))

        # --- Odom + EKF odom: slow figure-8 ---
        self._odom_theta += 0.02 * math.sin(t * 0.2)
        speed = 0.3
        self._odom_x += speed * 0.1 * math.cos(self._odom_theta)
        self._odom_y += speed * 0.1 * math.sin(self._odom_theta)
        odom_msg = make_odom(self._odom_x, self._odom_y, self._odom_theta, now_ns)
        self._deliver('/odom', odom_msg)
        self._deliver('/local_ekf/odom', odom_msg)

        # --- Electrical ---
        v = 24.0 + 0.3 * math.sin(t * 0.1)
        i = max(0.0, 3.0 + 1.5 * math.sin(t * 0.3))
        p = v * i
        soc = max(0.0, min(100.0, 75.0 - t * 0.05))
        self._deliver('/electrical/voltage', make_float32(v))
        self._deliver('/electrical/current', make_float32(i))
        self._deliver('/electrical/power', make_float32(p))
        self._deliver('/electrical/soc', make_float32(soc))

        # --- Autonomous mode flag — kept off by default ---
        self._deliver('/autonomous_mode', make_bool(False))


# ── Main ─────────────────────────────────────────────────────────────────

def _install_debug_shortcuts(window, driver):
    """Bind keys to fake-message helpers so we can exercise the GUI's
    state-driven overlays without a real robot:

        F9   — toggle /data/toggle_collect  (drives the REC overlay)
        F10  — toggle /autonomous_mode      (drives the AUTO badge)
    """
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtWidgets import QShortcut

    def toggle_rec():
        driver._sim_rec = not getattr(driver, '_sim_rec', False)
        driver._deliver('/data/toggle_collect', make_bool(driver._sim_rec))
        print(f'[debug] REC -> {driver._sim_rec}', flush=True)

    def toggle_auto():
        driver._sim_auto = not getattr(driver, '_sim_auto', False)
        driver._deliver('/autonomous_mode', make_bool(driver._sim_auto))
        print(f'[debug] AUTO -> {driver._sim_auto}', flush=True)

    QShortcut(QKeySequence('F9'), window, activated=toggle_rec)
    QShortcut(QKeySequence('F10'), window, activated=toggle_auto)


def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Import Qt symbols that the real hud_node also pulls in.
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    node = hud_node.HudNode()
    window = hud_node.HudWindow(ros_node=node)
    window.show()

    driver = FakeDataDriver(node)
    sim_timer = QTimer()
    sim_timer.timeout.connect(driver.tick)
    sim_timer.start(100)  # ~10 Hz

    # Pause the synthetic data driver while Performance Mode is on so
    # the CPU/GPU readout reflects the GUI's idle cost instead of the
    # simulator's image-generation work. The Performance button's own
    # handler flips `window._performance_mode` first; this slot reads
    # the post-flip state and gates the sim_timer accordingly.
    def _sync_sim_timer_with_perf_mode():
        if getattr(window, '_performance_mode', False):
            if sim_timer.isActive():
                sim_timer.stop()
        else:
            if not sim_timer.isActive():
                sim_timer.start(100)
    window.btn_performance.clicked.connect(_sync_sim_timer_with_perf_mode)

    _install_debug_shortcuts(window, driver)
    print('[debug] F9 toggles REC overlay, F10 toggles AUTO badge',
          flush=True)

    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
