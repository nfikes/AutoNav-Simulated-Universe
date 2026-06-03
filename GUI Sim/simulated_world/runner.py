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

import csv
import glob
import math
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

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
sys.path.insert(0, "hud_node_socket")
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
    return _build_odom(x, y, qz, qw, stamp_ns)


def make_odom_qz(x, y, qz, stamp_ns):
    """Build an Odometry message from a position and an *already-quaternion*
    z component (matches the `orient_z` column in the HUD's CSV format,
    which stores msg.pose.pose.orientation.z verbatim — see
    bag_reader._odom_vals). Use this when the source data is a quaternion
    component; use make_odom() when the source is a yaw angle in rad."""
    qz = max(-1.0, min(1.0, float(qz)))
    qw = math.sqrt(max(0.0, 1.0 - qz * qz))
    return _build_odom(x, y, qz, qw, stamp_ns)


def _build_odom(x, y, qz, qw, stamp_ns):
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

# Example recording shipped with the sim — used to source real camera +
# lidar BEV footage so Live Mode shows a realistic picture of the GUI
# instead of synthetic gradients. Selected as the first directory under
# ../data/ that has both *_camera.mp4 and *_lidar_bev.mp4 sidecars.
_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data',
))


def _find_example_recording(data_dir):
    """Return (camera_mp4_path, lidar_mp4_path) for the first complete
    recording found under `data_dir`, or (None, None) if nothing usable
    is present."""
    if not os.path.isdir(data_dir):
        return None, None
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if not os.path.isdir(sub):
            continue
        cams = glob.glob(os.path.join(sub, '*_camera.mp4'))
        lids = glob.glob(os.path.join(sub, '*_lidar_bev.mp4'))
        if cams and lids:
            return cams[0], lids[0]
    return None, None


# Candidate CSVs for sensor replay, tried in order. First match wins.
# t002_20251114_171257.csv is the picture-worthy choice: 1184 /gps_fix
# rows tracing a ~60×60 m outdoor route at VT over 131 s, with matching
# /odom. The 03-28-26 sessions are kept as fallbacks — they're shorter
# and the GPS trail there is only ~15×25 m (the robot barely moved),
# but they're the only sessions with real /electrical/voltage/current/
# power so they're useful if you want the power panel to read from real
# data. Override with SIM_REPLAY_CSV if you want a specific session.
def _ts_repo_data(*parts):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'AutoNav-Data-Visualizer', 'TestingData', *parts,
    ))

_CSV_REPLAY_CANDIDATES = [
    _ts_repo_data('t002_20251114_171257.csv'),
    _ts_repo_data('02-27-26-Data', 't002_20260227_211715.csv'),
    _ts_repo_data('03-28-26-Data', 't000_20260328_144058.csv'),
    _ts_repo_data('03-28-26-Data', 't000_20260328_154444.csv'),
]


def _find_replay_csv():
    """Return a path to a CSV with /gps_fix + /odom rows for Live-mode
    replay, or None if nothing's available."""
    env = os.environ.get('SIM_REPLAY_CSV')
    if env and os.path.isfile(env):
        return env
    for p in _CSV_REPLAY_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


class _BagReplay:
    """Pre-loads selected LaserScan-style topics from a rosbag2
    directory and yields them at the recorded relative time, looping
    at EOF. The bag is the only place real /scan_fullframe data lives
    — LaserScans are too big to log as CSV — so this fills the Lidar
    Heightband panel in Live Mode. If `rosbags` isn't installed, or
    the bag doesn't contain any of `_SUPPORTED`, the instance is
    truthy-False and FakeDataDriver falls back to synthetic.
    """

    _SUPPORTED = ('/scan_fullframe',)

    def __init__(self, bag_dir):
        self.path = bag_dir
        self._messages = []  # list of (rel_ns, topic, msg) — sorted
        self._duration_s = 0.0
        self._idx = 0
        self._loop_t0 = time.monotonic()
        self.counts = {t: 0 for t in self._SUPPORTED}
        try:
            from rosbags.rosbag2 import Reader
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            return
        try:
            ts = get_typestore(Stores.ROS2_HUMBLE)
            with Reader(bag_dir) as r:
                conns = [c for c in r.connections if c.topic in self._SUPPORTED]
                if not conns:
                    return
                t0 = r.start_time
                msgs = []
                for c, t, raw in r.messages(connections=conns):
                    try:
                        msg = ts.deserialize_cdr(raw, c.msgtype)
                    except Exception:  # noqa: BLE001
                        continue
                    msgs.append((t - t0, c.topic, msg))
                    self.counts[c.topic] += 1
                msgs.sort(key=lambda m: m[0])
                self._messages = msgs
                self._duration_s = (r.end_time - r.start_time) / 1e9
        except Exception:  # noqa: BLE001
            self._messages = []

    def __bool__(self):
        return bool(self._messages)

    def due(self):
        """Yield (topic, msg) for every message whose recorded time is
        at or before the current playback offset. Wraps at EOF."""
        if not self._messages:
            return
        play_t_ns = (time.monotonic() - self._loop_t0) * 1e9
        while self._idx < len(self._messages):
            rel_ns, topic, msg = self._messages[self._idx]
            if rel_ns > play_t_ns:
                break
            yield topic, msg
            self._idx += 1
        if self._idx >= len(self._messages):
            self._idx = 0
            self._loop_t0 = time.monotonic()


class _CsvReplay:
    """Loads a t00x-style HUD CSV (ROS2_Clock,Topic_Name,Data_Keys,Value_…)
    and yields the rows that are due at the current playback time. Loops
    back to t=0 when the trail ends so the GUI keeps a continuous trace.

    Only the topics this class knows how to convert to fake ROS messages
    are returned (/gps_fix, /odom). Everything else in the CSV is ignored
    here — `FakeDataDriver` keeps owning the synthetic streams for them."""

    _SUPPORTED = (
        '/gps_fix', '/odom',
        '/electrical/voltage', '/electrical/current', '/electrical/power',
    )

    def __init__(self, csv_path):
        self.path = csv_path
        rows = []
        with open(csv_path, 'r', newline='') as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    ts_ns = int(row[0])
                except ValueError:
                    continue
                topic = row[1]
                if topic not in self._SUPPORTED:
                    continue
                keys = row[2].split(',')
                vals = row[3:3 + len(keys)]
                rows.append((ts_ns, topic, keys, vals))
        rows.sort(key=lambda r: r[0])
        self._rows = rows
        self._t0_ns = rows[0][0] if rows else 0
        self._duration_s = (rows[-1][0] - rows[0][0]) / 1e9 if rows else 0.0
        self._idx = 0
        self._loop_t0 = time.monotonic()
        # Counts of each supported topic, for the startup log line.
        self.counts = {t: sum(1 for r in rows if r[1] == t) for t in self._SUPPORTED}

    def __bool__(self):
        return bool(self._rows)

    def due(self):
        """Yield (topic, keys, vals) tuples for every row whose recorded
        time is at or before the current playback offset. Wraps at EOF."""
        if not self._rows:
            return
        play_t = time.monotonic() - self._loop_t0
        while self._idx < len(self._rows):
            rel_ns = self._rows[self._idx][0] - self._t0_ns
            if rel_ns / 1e9 > play_t:
                break
            _, topic, keys, vals = self._rows[self._idx]
            yield topic, keys, vals
            self._idx += 1
        if self._idx >= len(self._rows):
            self._idx = 0
            self._loop_t0 = time.monotonic()


class FakeDataDriver:
    """Periodically generates and delivers synthetic ROS messages to the
    real HudNode's callbacks. Lives on the main thread; ticked by a QTimer.

    Generated streams (each delivered IF the GUI has subscribed):
        /zed/zed_node/rgb/color/rect/image  -> example recording camera
                                                frames (looped) when an
                                                MP4 is available; falls
                                                back to a shifting RGB
                                                gradient otherwise
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

        # Open the example recording's camera mp4. Frames feed the
        # Camera RAW panel through /zed/.../rgb so the GUI shows a real
        # picture in Live Mode instead of the synthetic gradient. The
        # lidar mp4 isn't piped into /scan_fullframe — that topic is a
        # LaserScan, not an image — but we still report it as found so
        # the operator can confirm the recording was picked up.
        self._cam_cap = None
        self._cam_path, self._lidar_path = (None, None)
        if _HAS_CV2:
            self._cam_path, self._lidar_path = _find_example_recording(_DATA_DIR)
            if self._cam_path is not None:
                cap = cv2.VideoCapture(self._cam_path)
                if cap.isOpened():
                    self._cam_cap = cap
                    print(f'[sim] Camera replay: {os.path.basename(self._cam_path)}',
                          flush=True)
                else:
                    cap.release()
                    print(f'[sim] WARN: could not open {self._cam_path}',
                          flush=True)
        if self._cam_cap is None:
            print('[sim] Camera: synthetic gradient (no recording / cv2 unavailable)',
                  flush=True)

        # Bag replay for /scan_fullframe. The CSVs have no LaserScan
        # data (too big to log as CSV), so the bag is the only source
        # of real lidar. Same recording the camera mp4 was extracted
        # from, so the lidar sweep is time-aligned with the camera.
        self._bag_replay = None
        bag_dir = None
        if self._cam_path is not None:
            cand = os.path.join(os.path.dirname(self._cam_path), 'bag')
            if os.path.isdir(cand):
                bag_dir = cand
        if bag_dir:
            try:
                self._bag_replay = _BagReplay(bag_dir)
            except Exception as e:  # noqa: BLE001
                print(f'[sim] WARN: bag replay failed: {e}', flush=True)
                self._bag_replay = None
        if self._bag_replay:
            counts = self._bag_replay.counts
            print(
                f'[sim] Bag replay: {os.path.basename(os.path.dirname(bag_dir))}/bag '
                f'({counts.get("/scan_fullframe", 0)} lidar, '
                f'{self._bag_replay._duration_s:.1f}s loop)',
                flush=True,
            )
        else:
            print('[sim] Lidar: synthetic circular room (no bag found / rosbags unavailable)',
                  flush=True)

        # CSV replay for /gps_fix + /odom. Pulls real GPS coordinates +
        # wheel odometry from a t00x-style HUD CSV so the GPS panel
        # shows a real outdoor trail and the odom panel shows the
        # matching wheel track. Falls back to synthetic drift / figure-8
        # when no CSV is found.
        self._csv_replay = None
        csv_path = _find_replay_csv()
        if csv_path is not None:
            try:
                self._csv_replay = _CsvReplay(csv_path)
            except Exception as e:  # noqa: BLE001
                print(f'[sim] WARN: failed to load {csv_path}: {e}', flush=True)
                self._csv_replay = None
        if self._csv_replay:
            counts = self._csv_replay.counts
            print(
                f'[sim] CSV replay: {os.path.basename(csv_path)} '
                f'({counts.get("/gps_fix", 0)} gps, '
                f'{counts.get("/odom", 0)} odom, '
                f'{counts.get("/electrical/voltage", 0)} v, '
                f'{self._csv_replay._duration_s:.1f}s loop)',
                flush=True,
            )
        else:
            print('[sim] GPS/odom: synthetic (no CSV — set SIM_REPLAY_CSV to override)',
                  flush=True)

    def _deliver(self, topic, msg):
        """Call the registered callback for `topic`, if any."""
        sub = self._node._fake_subs.get(topic)
        if sub is None:
            return
        try:
            sub.callback(msg)
        except Exception as e:  # noqa: BLE001
            print(f'WARN: driver callback {topic} raised: {e}', file=sys.stderr)

    def _next_camera_frame(self):
        """Decode the next frame from the example recording's camera mp4
        and return it as an RGB ndarray, or None if no capture is open.
        Loops back to frame 0 on EOF."""
        cap = self._cam_cap
        if cap is None:
            return None
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
            if not ok:
                return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def tick(self):
        """Generate one tick of fake data. Call at ~10 Hz."""
        t = time.monotonic() - self._t0
        self._frame += 1
        now_ns = int(time.time() * 1e9)

        # --- Camera frame: prefer the example recording's mp4; fall
        # back to a shifting colour gradient when no recording is
        # available so the panel still has something to draw.
        rgb = self._next_camera_frame()
        if rgb is None:
            w, h = 480, 360
            xs = np.linspace(0, 1, w, dtype=np.float32)
            ys = np.linspace(0, 1, h, dtype=np.float32)
            xg, yg = np.meshgrid(xs, ys)
            r = np.clip((np.sin(xg * 4 + t) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            g = np.clip((np.cos(yg * 4 + t * 0.7) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            b = np.clip((np.sin((xg + yg) * 3 - t * 0.5) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            rgb = np.stack([r, g, b], axis=-1)
        rgb = np.ascontiguousarray(rgb)
        self._deliver('/zed/zed_node/rgb/color/rect/image',
                       make_image_msg(rgb, now_ns))

        # --- LaserScan: prefer bag-replayed /scan_fullframe (real
        # 720-beam sweep with actual obstacle returns). When the bag
        # is loaded we let the recording's own rate drive the panel
        # (~8-9 Hz on this dataset), so most 10 Hz ticks will deliver
        # 0 or 1 message — that's expected. Only fall back to the
        # synthetic circular room when no bag is loaded at all.
        if self._bag_replay:
            for topic, msg in self._bag_replay.due():
                if topic == '/scan_fullframe':
                    self._deliver('/scan_fullframe', msg)
        else:
            ranges = []
            for i in range(360):
                a = -math.pi + i * (2 * math.pi / 360)
                r_ = 6.0 + random.uniform(-0.05, 0.05)
                for obj_a in (math.fmod(t * 0.3, 2 * math.pi) - math.pi,
                              math.fmod(t * 0.5 + math.pi, 2 * math.pi) - math.pi):
                    if abs(a - obj_a) < 0.12:
                        r_ = 3.0
                ranges.append(r_)
            self._deliver('/scan_fullframe', make_laser_scan(ranges, now_ns))

        # --- CSV-driven streams: GPS + odom + electrical telemetry.
        # The CSV iterator returns each due row exactly once, so we
        # record which topics were delivered this tick to know which
        # synthetic fallbacks to skip. Topics not in _SUPPORTED never
        # appear, so /electrical/soc always uses the synthetic curve.
        served = set()
        if self._csv_replay:
            for topic, keys, vals in self._csv_replay.due():
                d = dict(zip(keys, vals))
                if topic == '/gps_fix':
                    try:
                        lat = float(d['latitude'])
                        lon = float(d['longitude'])
                        alt = float(d.get('altitude', 0.0))
                    except (KeyError, ValueError):
                        continue
                    self._deliver('/gps_fix', make_navsat_fix(
                        lat, lon, alt, 0.5, 0.5, now_ns,
                    ))
                    served.add(topic)
                elif topic == '/odom':
                    try:
                        x = float(d['pos_x'])
                        y = float(d['pos_y'])
                        qz = float(d['orient_z'])
                    except (KeyError, ValueError):
                        continue
                    # CSV's orient_z is a quaternion component, NOT a
                    # yaw angle — bag_reader._odom_vals stores
                    # msg.pose.pose.orientation.z verbatim. Treating it
                    # as yaw squashed the heading by ~half.
                    odom_msg = make_odom_qz(x, y, qz, now_ns)
                    self._deliver('/odom', odom_msg)
                    self._deliver('/local_ekf/odom', odom_msg)
                    served.add(topic)
                elif topic in ('/electrical/voltage', '/electrical/current',
                               '/electrical/power'):
                    try:
                        val = float(vals[0])
                    except (IndexError, ValueError):
                        continue
                    self._deliver(topic, make_float32(val))
                    served.add(topic)

        if '/gps_fix' not in served:
            drift_lat = 0.0001 * math.sin(t * 0.05)
            drift_lon = 0.0001 * math.cos(t * 0.07)
            cov_ee = 0.5 + 0.2 * math.sin(t * 0.4)
            cov_nn = 0.5 + 0.2 * math.cos(t * 0.4)
            self._deliver('/gps_fix', make_navsat_fix(
                self._gps_lat0 + drift_lat, self._gps_lon0 + drift_lon,
                621.0, cov_ee, cov_nn, now_ns,
            ))
        if '/odom' not in served:
            self._odom_theta += 0.02 * math.sin(t * 0.2)
            speed = 0.3
            self._odom_x += speed * 0.1 * math.cos(self._odom_theta)
            self._odom_y += speed * 0.1 * math.sin(self._odom_theta)
            odom_msg = make_odom(
                self._odom_x, self._odom_y, self._odom_theta, now_ns,
            )
            self._deliver('/odom', odom_msg)
            self._deliver('/local_ekf/odom', odom_msg)

        # --- Electrical: voltage/current/power come from CSV when the
        # session has them; SOC is always synthesized because the
        # logged sessions don't include /electrical/soc.
        if '/electrical/voltage' not in served:
            self._deliver('/electrical/voltage',
                          make_float32(24.0 + 0.3 * math.sin(t * 0.1)))
        if '/electrical/current' not in served:
            self._deliver('/electrical/current',
                          make_float32(max(0.0, 3.0 + 1.5 * math.sin(t * 0.3))))
        if '/electrical/power' not in served:
            self._deliver('/electrical/power',
                          make_float32(24.0 * 3.0))
        soc = max(0.0, min(100.0, 75.0 - t * 0.05))
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


def _probe_screen_size():
    """Return (width, height) of the host display in *logical* pixels,
    or (0, 0) if every detection path fails. Tries, in order:
        1. macOS `system_profiler SPDisplaysDataType` (no automation
           permissions required, unlike osascript+Finder).
        2. tkinter winfo_screenwidth/height (Linux + Python builds
           that ship Tk; the sim's uv-managed venv does not).
    On macOS the system_profiler output is in physical pixels — Retina
    displays report 2x the logical size — so we halve when the width
    looks like Retina (>2000 px). Without that the GUI thinks it has
    twice the room and the scale factor clamps to 1.0.
    """
    if sys.platform == 'darwin':
        try:
            out = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType', '-json'],
                capture_output=True, text=True, timeout=5,
            )
            import json
            d = json.loads(out.stdout)
            for adapter in d.get('SPDisplaysDataType', []):
                for screen in adapter.get('spdisplays_ndrvs', []) or []:
                    pix = screen.get('_spdisplays_pixels', '')
                    parts = [p.strip() for p in pix.replace('x', ' ').split()]
                    if len(parts) >= 2:
                        sw, sh = int(parts[0]), int(parts[1])
                        if sw > 2000:  # Retina → halve to logical
                            sw, sh = sw // 2, sh // 2
                        return sw, sh
        except Exception:  # noqa: BLE001
            pass
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        sw = r.winfo_screenwidth()
        sh = r.winfo_screenheight()
        r.destroy()
        return sw, sh
    except Exception:  # noqa: BLE001
        return 0, 0


def _compute_qt_scale_factor():
    """Pick a uniform Qt scale factor so the HUD's native 1920x720
    layout fits the host display with ~10% padding. Honors the
    SIM_QT_SCALE_FACTOR env var as an override; falls back to a
    laptop-friendly 0.7 if the screen probe fails entirely so the
    window is never wider than ~1344 px."""
    override = os.environ.get('SIM_QT_SCALE_FACTOR')
    if override:
        try:
            return max(0.1, float(override))
        except ValueError:
            pass
    sw, sh = _probe_screen_size()
    if sw <= 0 or sh <= 0:
        return 0.7
    return min(1.0, (sw * 0.9) / 1920.0, (sh * 0.9) / 720.0)


def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Scale the whole Qt application so the HUD's native 1920x720
    # layout fits the host display without squeezing the button text.
    # QT_SCALE_FACTOR must be set BEFORE QApplication is constructed
    # — Qt reads it during init.
    scale = _compute_qt_scale_factor()
    if scale < 0.999:
        os.environ['QT_SCALE_FACTOR'] = f'{scale:.4f}'
        print(f'[sim] Qt scale factor: {scale:.2f} '
              f'(HUD renders at native 1920x720, scaled to fit)',
              flush=True)

    # Import Qt symbols that the real hud_node also pulls in.
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    node = hud_node.HudNode()
    window = hud_node.HudWindow(ros_node=node)

    # HudWindow.__init__ pins itself as a frameless 1920x720 kiosk
    # window with X11BypassWindowManagerHint and a blank cursor —
    # right on the Jetson's panel, but on a dev laptop the user can't
    # drag/close it and can't see the cursor. Restore a normal Qt
    # window frame and the cursor without touching the internal layout
    # (size stays 1920x720 logical; QT_SCALE_FACTOR above scales the
    # rendered pixels so it fits the screen).
    window.hide()
    window.setWindowFlags(Qt.Window)
    window.unsetCursor()
    window.show()
    window.raise_()
    window.activateWindow()

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

    # Auto-connect to the (faked) container and enter Live Mode so the
    # GUI's _live_tick starts draining node.latest_* into the data
    # buffers. Without this, _odom_buf / _gps_buf stay empty and the
    # new goal-pick buttons ("Set Local Goal…" / "Set GPS Goal…")
    # refuse to enter their crosshair mode. The sim driver above is
    # already feeding /odom and /gps_fix; we just need the consumer
    # loop turned on. Deferred via singleShot so the Qt event loop
    # finishes laying out the window before we start mutating state.
    def _auto_enter_live():
        try:
            if not window._container_connected:
                window._connect_container()
            if not getattr(window, '_live_active', False):
                window._start_live_mode()
            print('[debug] Auto-connected and entered Live Mode', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'[debug] Auto-enter Live failed: {e}', flush=True)
    QTimer.singleShot(300, _auto_enter_live)

    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
