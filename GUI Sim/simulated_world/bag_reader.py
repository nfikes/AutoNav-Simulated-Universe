"""Read a ROS2 rosbag2 directory (sqlite3) without a ROS install.

Uses the pure-Python `rosbags` package to enumerate connections, deserialize
CDR-encoded messages, and produce the same row format the standalone HUD's
CSV playback path already consumes.

Public API:
    iter_bag(bag_dir)
        Iterate every message in chronological order. Yields
        (ts_ns, topic, msgtype, msg) tuples where `msg` is the
        deserialized message object (with attributes like .data, .pose, ...).

    bag_to_rows(bag_dir)
        High-level helper. Returns (rows, gps_lats, gps_lons) where rows is
        a list of (rel_ts_ns, topic, keys, values) tuples — exactly the
        format _load_csv builds for `_pb_rows`. Image and LaserScan topics
        are intentionally excluded (those become sidecar mp4s instead).

    extract_bag(bag_dir, out_dir=None, fps=30, force=False)
        Extract a bag to:
            <out_dir>/<bag_stem>.csv
            <out_dir>/<bag_stem>_camera.mp4
            <out_dir>/<bag_stem>_lidar_bev.mp4
        and return the CSV path. If `out_dir` is None, writes alongside the
        bag directory's parent. If the CSV already exists and `force` is
        False, returns the existing path without re-running extraction.

    extract_bag_cli()
        Argparse entry point — run `python bag_reader.py <bag_dir>`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Iterator, Tuple

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    _HAS_ROSBAGS = True
except ImportError:
    _HAS_ROSBAGS = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ── Topics ────────────────────────────────────────────────────────────
# Scalar/vector topics → CSV rows. Each entry: topic → (keys_csv, value_fn(msg)).
# value_fn returns a tuple of stringified values matching keys order.
def _odom_vals(msg):
    p = msg.pose.pose.position
    qz = msg.pose.pose.orientation.z
    return (f'{p.x:.4f}', f'{p.y:.4f}', f'{qz:.4f}')


def _gps_vals(msg):
    """Return (lat, lon, alt) and, when the message carries a valid 2x2
    ENU position covariance, the (cov_ee, cov_en, cov_nn) block too.

    A "valid" covariance means all three values are finite and the
    diagonal is strictly positive — matching the live GUI's gate at
    hud_node._cb_gps so synthetic-zero covariances don't graduate the
    panel into the covariance tier.
    """
    base = (
        f'{msg.latitude:.8f}',
        f'{msg.longitude:.8f}',
        f'{msg.altitude:.4f}',
    )
    cov = getattr(msg, 'position_covariance', None)
    if cov is not None and len(cov) >= 5:
        try:
            cov_ee = float(cov[0])
            cov_en = float(cov[1])
            cov_nn = float(cov[4])
        except (TypeError, ValueError):
            return base
        if (math.isfinite(cov_ee) and math.isfinite(cov_en)
                and math.isfinite(cov_nn)
                and cov_ee > 0.0 and cov_nn > 0.0):
            return base + (f'{cov_ee:.6e}', f'{cov_en:.6e}', f'{cov_nn:.6e}')
    return base


def _float32_vals(msg):
    return (f'{float(msg.data):.6f}',)


def _bool_vals(msg):
    return (str(bool(msg.data)),)


# msgtype → (keys, value_fn). Indexed by ROS msgtype rather than topic
# so the same handler covers e.g. both /odom and /local_ekf/odom.
_CSV_HANDLERS_BY_TYPE = {
    'nav_msgs/msg/Odometry':       ('pos_x,pos_y,orient_z', _odom_vals),
    'sensor_msgs/msg/NavSatFix':   ('latitude,longitude,altitude', _gps_vals),
    'std_msgs/msg/Float32':        ('data', _float32_vals),
    'std_msgs/msg/Bool':           ('data', _bool_vals),
}

# Some topics use a domain-specific key name in the existing CSVs
# (e.g. /electrical/voltage → "voltage_V"). Override by topic where it
# matters for downstream key matching.
_CSV_KEY_OVERRIDES = {
    '/electrical/voltage': 'voltage_V',
    '/electrical/current': 'current_A',
    '/electrical/power':   'power_W',
    '/electrical/soc':     'soc_pct',
}

# Topics we explicitly do NOT write to CSV (handled as videos/sidecars,
# or simply too verbose to keep).
_SKIP_FOR_CSV = {
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/LaserScan',
    'sensor_msgs/msg/PointCloud2',
    'nav_msgs/msg/OccupancyGrid',
    'nav_msgs/msg/Path',
    'geometry_msgs/msg/PoseWithCovarianceStamped',
    'std_msgs/msg/Int32MultiArray',
}

# Image topics that get extracted to mp4 sidecars
_CAMERA_TOPIC = '/zed/zed_node/rgb/color/rect/image'
# LaserScan topic for the BEV sidecar mp4
_LIDAR_TOPIC = '/scan_fullframe'

# Output mp4 framerate. Must match the standalone's _video_fps.
DEFAULT_FPS = 30


# ── Reader primitives ────────────────────────────────────────────────

def _open_reader(bag_dir):
    if not _HAS_ROSBAGS:
        raise RuntimeError(
            'The `rosbags` package is not installed. '
            'Install it with: uv pip install --system rosbags'
        )
    bag_dir = str(bag_dir)
    if not os.path.isdir(bag_dir):
        raise FileNotFoundError(f'Bag directory not found: {bag_dir}')
    if not os.path.isfile(os.path.join(bag_dir, 'metadata.yaml')):
        raise FileNotFoundError(
            f'metadata.yaml not found in {bag_dir} — not a rosbag2 directory.'
        )
    reader = Reader(bag_dir)
    reader.open()
    return reader


def iter_bag(bag_dir) -> Iterator[Tuple[int, str, str, object]]:
    """Yield (ts_ns, topic, msgtype, msg) for every message in chronological order."""
    reader = _open_reader(bag_dir)
    try:
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        for conn, ts_ns, raw in reader.messages():
            try:
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
            except Exception as e:  # noqa: BLE001 — one bad message shouldn't kill the stream
                print(f'WARN: failed to deserialize {conn.topic} ({conn.msgtype}): {e}',
                      file=sys.stderr)
                continue
            yield ts_ns, conn.topic, conn.msgtype, msg
    finally:
        reader.close()


def bag_to_rows(bag_dir):
    """Read a bag into the standalone's CSV-equivalent row format.

    Returns (rows, gps_lats, gps_lons).

    rows is a list of (rel_ts_ns, topic, keys, values) — absolute ns
    timestamps are re-based to start at 0, matching what _load_csv does
    after sorting.
    """
    rows = []
    gps_lats, gps_lons = [], []

    for ts_ns, topic, msgtype, msg in iter_bag(bag_dir):
        if msgtype in _SKIP_FOR_CSV:
            continue
        handler = _CSV_HANDLERS_BY_TYPE.get(msgtype)
        if handler is None:
            continue
        keys, value_fn = handler
        keys = _CSV_KEY_OVERRIDES.get(topic, keys)
        try:
            values = value_fn(msg)
        except Exception as e:  # noqa: BLE001
            print(f'WARN: failed to extract {topic}: {e}', file=sys.stderr)
            continue
        rows.append((ts_ns, topic, keys, list(values)))

        if topic == '/gps_fix':
            try:
                gps_lats.append(float(values[0]))
                gps_lons.append(float(values[1]))
            except (ValueError, IndexError):
                pass

    rows.sort(key=lambda r: r[0])
    if rows:
        t0 = rows[0][0]
        rows = [(ts - t0, t, k, v) for (ts, t, k, v) in rows]
    return rows, gps_lats, gps_lons


# ── BEV renderers ────────────────────────────────────────────────────
# Match the source GUI's _LidarFrameWorker exactly: gray background,
# white rays, black shadows, green hit dots (heightband); or white rays
# + red 3x3 hit stamps (PCA). Robot origin marker is a red triangle
# pointing +x.
#
# Orientation: the source GUI's live worker applies
# `np.rot90(img, 1); np.fliplr(img)` before painting, yielding
# "+x forward / +y right on screen". Both the standalone HUD's
# playback path and bake_offscreen apply `np.rot90(rgb, 2); np.fliplr(rgb)`
# to the mp4 frames they read — equivalent to a vertical flip.
# To make the mp4 reproduce the live display orientation after that flip,
# we write `np.rot90(raw, -1)`: live-display = flipud(np.rot90(raw, -1))
# = source's `fliplr(rot90(raw, 1))`. Verified algebraically.

def _draw_robot_triangle(img, cx, cy):
    """Red forward-pointing triangle at the robot origin. Pre-rotation
    space: apex points +x. After the consumer's rot90(2)+fliplr, the
    apex shows up pointing "up = forward" on screen."""
    if _HAS_CV2:
        tri = np.array([
            [cx - 4, cy - 4],
            [cx - 4, cy + 4],
            [cx + 5, cy],
        ], dtype=np.int32)
        cv2.fillPoly(img, [tri], (0, 0, 255))  # BGR red


def _render_heightband_raw(scan, size, max_range_default=10.0):
    """Source-fidelity heightband BEV before the orientation pass.
    Returns a BGR uint8 image."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)  # gray bg
    cx, cy = size // 2, size // 2
    max_range = float(scan.range_max) if scan.range_max and scan.range_max > 0 else max_range_default
    if not math.isfinite(max_range) or max_range <= 0:
        max_range = max_range_default
    scale = (size // 2 - 2) / max_range

    angles = np.arange(len(scan.ranges)) * scan.angle_increment + scan.angle_min
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    valid = np.isfinite(ranges) & (ranges >= scan.range_min)

    if np.any(valid):
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        safe_ranges = np.where(valid, ranges, 0.0)
        sx_all = (cx + max_range * cos_a * scale).astype(np.int32)
        sy_all = (cy - max_range * sin_a * scale).astype(np.int32)
        ex_all = (cx + safe_ranges * cos_a * scale).astype(np.int32)
        ey_all = (cy - safe_ranges * sin_a * scale).astype(np.int32)
        is_hit = valid & (ranges < max_range)
        is_clear = valid & (ranges >= max_range)

        if _HAS_CV2:
            white = (255, 255, 255)
            black = (0, 0, 0)
            for i in np.where(is_clear)[0]:
                cv2.line(img, (cx, cy),
                         (int(sx_all[i]), int(sy_all[i])), white, 1)
            for i in np.where(is_hit)[0]:
                ex = int(ex_all[i]); ey = int(ey_all[i])
                cv2.line(img, (cx, cy), (ex, ey), white, 1)
                cv2.line(img, (ex, ey),
                         (int(sx_all[i]), int(sy_all[i])), black, 1)
            hit_xs = ex_all[is_hit]
            hit_ys = ey_all[is_hit]
            in_bounds = ((hit_xs >= 0) & (hit_xs < size) &
                         (hit_ys >= 0) & (hit_ys < size))
            img[hit_ys[in_bounds], hit_xs[in_bounds]] = (0, 255, 0)

    _draw_robot_triangle(img, cx, cy)
    return img


def _render_pca_raw(xy, size, max_range=10.0):
    """Source-fidelity PCA BEV from (N, 2) obstacle XY array.

    White rays from the robot origin to each hit plus 3x3 red stamps."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)  # gray bg
    cx, cy = size // 2, size // 2
    scale = (size // 2 - 2) / max_range

    if xy is not None and len(xy) > 0:
        xs = (cx + xy[:, 0] * scale).astype(np.int32)
        ys = (cy - xy[:, 1] * scale).astype(np.int32)
        in_bounds = ((xs >= 1) & (xs < size - 1) &
                     (ys >= 1) & (ys < size - 1))
        xs = xs[in_bounds]; ys = ys[in_bounds]

        if _HAS_CV2:
            white = (255, 255, 255)
            for xi, yi in zip(xs, ys):
                cv2.line(img, (cx, cy), (int(xi), int(yi)), white, 1)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    img[ys + dy, xs + dx] = (0, 0, 255)  # BGR red

    _draw_robot_triangle(img, cx, cy)
    return img


def _orient_for_mp4(raw):
    """Pre-apply the inverse of the consumer's `rot90(2)+fliplr` so that
    after the consumer applies it, the displayed orientation matches what
    the source GUI's worker emits: +x forward / +y right on screen.

    Equivalent: writes np.rot90(raw, -1) to the mp4.
    """
    return np.ascontiguousarray(np.rot90(raw, -1))


def render_lidar_bev(scan, size=480):
    """Public heightband renderer (oriented for mp4 writing)."""
    return _orient_for_mp4(_render_heightband_raw(scan, size))


def render_pca_bev(xy, size=480, max_range=10.0):
    """Public PCA renderer (oriented for mp4 writing)."""
    return _orient_for_mp4(_render_pca_raw(xy, size, max_range=max_range))


def pca_xy_from_scan(msg):
    """Reproduce the source GUI's _cb_pca_scan: filter valid hits,
    polar → cartesian, and flip y so the worker's orientation pass lands
    the points on the same screen side as the heightband view."""
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    if ranges.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    angles = (np.arange(ranges.size, dtype=np.float32)
              * msg.angle_increment + msg.angle_min)
    rmax = float(msg.range_max) if msg.range_max and msg.range_max > 0 else 10.0
    valid = (np.isfinite(ranges) &
             (ranges >= msg.range_min) & (ranges < rmax))
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    rs = ranges[valid]; as_ = angles[valid]
    return np.column_stack([
        (rs * np.cos(as_)).astype(np.float32),
        -(rs * np.sin(as_)).astype(np.float32),  # sign flip per source
    ])


# ── Image decoding ───────────────────────────────────────────────────

def _image_to_bgr(msg):
    """Convert a sensor_msgs/Image message to a contiguous BGR uint8 ndarray.

    For mono8 / mask images this returns a 3-channel grayscale BGR so it
    can be written directly to an mp4 alongside RGB camera frames.
    """
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or '').lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == 'bgra8':
        arr = raw.reshape(h, w, 4)
        return np.ascontiguousarray(arr[..., :3])
    if enc == 'rgba8':
        arr = raw.reshape(h, w, 4)
        return np.ascontiguousarray(arr[..., [2, 1, 0]])
    if enc == 'bgr8':
        return np.ascontiguousarray(raw.reshape(h, w, 3))
    if enc == 'rgb8':
        return np.ascontiguousarray(raw.reshape(h, w, 3)[..., ::-1])
    if enc in ('mono8', '8uc1'):
        arr = raw.reshape(h, w)
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
    # Fall back to interpreting by step
    try:
        arr = raw.reshape(h, int(msg.step) // max(w, 1) * w).reshape(h, w, -1)
        if arr.shape[2] >= 3:
            return np.ascontiguousarray(arr[..., :3])
    except Exception:
        pass
    return None


def _decode_line_pixels(msg):
    """Decode Int32MultiArray /line_detection/line_pixels into (xs, ys, w, h).

    Layout (matches the source GUI's _cb_line_pixels):
        data[0] = source image width
        data[1] = source image height
        data[2:] = interleaved x0, y0, x1, y1, ...

    Returns (None, None, 0, 0) if the message is malformed or empty.
    """
    data = msg.data
    if len(data) < 2:
        return None, None, 0, 0
    w = int(data[0])
    h = int(data[1])
    pairs = np.asarray(data[2:], dtype=np.int32)
    if pairs.size % 2 != 0:
        pairs = pairs[:-1]
    if pairs.size == 0:
        return None, None, w, h
    pairs = pairs.reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1], w, h


def _overlay_line_pixels(bgr, xs, ys, src_w, src_h):
    """Stamp red dots on a BGR camera frame at the given pixel coords.

    Coords come in source-image space (src_w x src_h). Rescales onto the
    bgr frame's actual size before drawing — the bag's raw camera and the
    mask image can differ in resolution.
    """
    if xs is None or ys is None or xs.size == 0:
        return
    h, w = bgr.shape[:2]
    if src_w > 0 and src_h > 0 and (src_w != w or src_h != h):
        xs = (xs.astype(np.float32) * (w / src_w)).astype(np.int32)
        ys = (ys.astype(np.float32) * (h / src_h)).astype(np.int32)
    in_b = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
    xs = xs[in_b]; ys = ys[in_b]
    # 3x3 red stamp per pixel (BGR red = (0, 0, 255))
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            bgr[ys + dy, xs + dx] = (0, 0, 255)


# ── Extraction (bag → CSV + sidecar mp4s) ────────────────────────────

def _csv_header(max_value_cols=12):
    cols = ['ROS2_Clock', 'Topic_Name', 'Data_Keys'] + [
        f'Value_{i}' for i in range(max_value_cols)
    ]
    return cols


def _scan_for_max_value_cols(rows):
    return max((len(v) for _, _, _, v in rows), default=1)


class _FixedFpsVideoEmitter:
    """Emit mp4 frames at a fixed wall-clock fps from irregularly-timed inputs.

    The bag's image frames arrive at whatever cadence the robot recorded them
    — often laggy / non-uniform. The standalone HUD's CSV playback assumes
    sidecar mp4s are exact `_video_fps` so it can seek via
    `frame_idx = int(elapsed_s * fps)`. To keep timing honest (a 5 fps
    recording should *look* like 5 fps at 30 fps mp4 playback, not get
    smoothed to 30), we duplicate whatever the most-recent decoded frame
    is until the next real frame arrives.

    Usage:
        em = _FixedFpsVideoEmitter(path, fps, t0_ns, duration_ns)
        for (ts_ns, frame_bgr) in stream:
            em.add(ts_ns, frame_bgr)
        em.finalize()
    """

    def __init__(self, path, fps, t0_ns, duration_ns):
        self._path = path
        self._fps = fps
        self._t0 = t0_ns
        self._total = max(1, int(round(duration_ns / 1e9 * fps)))
        self._writer = None
        self._size = None
        self._last_frame = None
        self._emitted = 0
        self._real_frames = 0

    def _ensure_writer(self, sample_bgr):
        if self._writer is not None:
            return
        h, w = sample_bgr.shape[:2]
        self._size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, self._size)
        if not self._writer.isOpened():
            print(f'WARN: could not open VideoWriter for {self._path}',
                  file=sys.stderr)
            self._writer = None

    def add(self, ts_ns, frame_bgr):
        """Record a new real frame at absolute timestamp ts_ns."""
        self._ensure_writer(frame_bgr)
        if self._writer is None:
            return
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self._size:
            frame_bgr = cv2.resize(frame_bgr, self._size)
        # Pad with copies of the previous frame up to (but not including)
        # this new frame's output slot.
        rel_ns = max(0, ts_ns - self._t0)
        target_idx = min(self._total - 1, int(rel_ns / 1e9 * self._fps))
        while self._emitted < target_idx and self._last_frame is not None:
            self._writer.write(self._last_frame)
            self._emitted += 1
        # If we have no last_frame yet (this is the very first), pad with
        # blank frames so the timeline still aligns.
        if self._last_frame is None and self._emitted < target_idx:
            blank = np.zeros_like(frame_bgr)
            while self._emitted < target_idx:
                self._writer.write(blank)
                self._emitted += 1
        self._writer.write(frame_bgr)
        self._emitted += 1
        self._last_frame = frame_bgr
        self._real_frames += 1

    def finalize(self):
        """Pad to the bag's full duration with the last real frame, then close."""
        if self._writer is None:
            return self._real_frames
        while self._emitted < self._total:
            if self._last_frame is None:
                # Bag had zero frames on this stream — leave the mp4 empty.
                break
            self._writer.write(self._last_frame)
            self._emitted += 1
        self._writer.release()
        return self._real_frames


def _bag_time_bounds(bag_dir):
    """Return (start_ns, duration_ns) from the bag's metadata."""
    reader = _open_reader(bag_dir)
    try:
        start = int(reader.start_time)
        duration = int(reader.duration)
    finally:
        reader.close()
    return start, duration


_MASK_TOPIC = '/line_detection/debug/mask'
_OVERLAY_TOPIC = '/line_detection/debug/overlay'
_LINE_PIXELS_TOPIC = '/line_detection/line_pixels'
_PCA_TOPIC = '/scan_pca_filtered'
_EKF_ODOM_TOPIC = '/local_ekf/odom'
_ODOM_TOPIC = '/odom'
_PLAN_TOPIC = '/plan'
_COSTMAP_TOPIC = '/global_costmap/costmap'

# Freshness windows (seconds) for source switching, matching the live GUI.
_MASK_FRESH_S = 1.0
_OVERLAY_FRESH_S = 1.0
_PCA_FRESH_S = 1.0
_LINES_FRESH_S = 1.0

# Default BEV canvas size (matches the standalone GUI's lidar panel).
_LIDAR_BEV_SIZE = 480


def _bag_topic_counts(bag_dir):
    """Return {topic_name: msg_count} from the bag's connections."""
    reader = _open_reader(bag_dir)
    try:
        return {c.topic: c.msgcount for c in reader.connections}
    finally:
        reader.close()


def extract_bag(bag_dir, out_dir=None, fps=DEFAULT_FPS, force=False, verbose=True):
    """Convert a bag directory into CSV + sidecar mp4s.

    Layout: given <session>/bag (the typical t000 layout), outputs are
    written to <session>/ with stem equal to the session folder name.
    For a bag at the top level (no parent session dir), outputs are
    written next to the bag dir using its own name.

    Returns the produced CSV path. If `force` is False and the CSV
    already exists, returns it without doing anything.

    Mp4 sidecars are written at exactly `fps` fps with timestamp-aware
    framing — duplicating the most-recent real bag frame across output
    slots — so recording lag is preserved at playback time. Source
    upgrades match the live GUI:

        Camera   raw  ← mask          (when /line_detection/debug/mask is
                                       fresh within the past 1 s)
        Camera   +  line_pixels overlay (when fresh within 1 s)
        Lidar    heightband ← PCA     (when /scan_pca_filtered is fresh
                                       within the past 1 s)
        Odom    /odom ← /local_ekf/odom  (whenever the bag carries any
                                          ekf rows; emitted as /odom in
                                          the CSV so the existing
                                          playback consumer picks it up)
    """
    bag_dir = os.path.abspath(str(bag_dir))
    if not os.path.isfile(os.path.join(bag_dir, 'metadata.yaml')):
        raise FileNotFoundError(
            f'metadata.yaml not found in {bag_dir}; not a rosbag2 directory.'
        )

    # Decide output stem.
    parent = os.path.dirname(bag_dir)
    bag_basename = os.path.basename(bag_dir)
    if bag_basename == 'bag' and parent:
        stem_dir = parent
        stem_name = os.path.basename(parent)
    else:
        stem_dir = parent
        stem_name = bag_basename
    if out_dir is not None:
        stem_dir = os.path.abspath(out_dir)
        os.makedirs(stem_dir, exist_ok=True)

    csv_path = os.path.join(stem_dir, f'{stem_name}.csv')
    cam_mp4_path = os.path.join(stem_dir, f'{stem_name}_camera.mp4')
    lidar_mp4_path = os.path.join(stem_dir, f'{stem_name}_lidar_bev.mp4')

    if (not force) and os.path.isfile(csv_path):
        if verbose:
            print(f'extract_bag: CSV already exists, skipping ({csv_path})')
        return csv_path

    start_ns, duration_ns = _bag_time_bounds(bag_dir)
    counts = _bag_topic_counts(bag_dir)
    has_ekf = counts.get(_EKF_ODOM_TOPIC, 0) > 0
    has_mask = counts.get(_MASK_TOPIC, 0) > 0
    has_overlay = counts.get(_OVERLAY_TOPIC, 0) > 0
    has_pca = counts.get(_PCA_TOPIC, 0) > 0
    has_line_pixels = counts.get(_LINE_PIXELS_TOPIC, 0) > 0

    total_frames = max(1, int(round(duration_ns / 1e9 * fps)))
    if verbose:
        upgrades = []
        if has_ekf: upgrades.append('odom->EKF')
        if has_overlay:
            upgrades.append('cam->OVERLAY')
        elif has_mask:
            upgrades.append('cam->MASK')
        if has_pca: upgrades.append('lidar->PCA')
        if has_line_pixels: upgrades.append('+line_pixels')
        tag = ' '.join(upgrades) or '(no upgrades)'
        print(f'extract_bag: reading {bag_dir}')
        print(f'extract_bag: duration {duration_ns/1e9:.2f}s, '
              f'{total_frames} mp4 frames @ {fps}fps, upgrades: {tag}')

    rows = []
    # State buffers for streaming smart emission.
    last_raw_bgr = None
    last_mask_bgr = None
    last_mask_ts = -10**18
    last_overlay_bgr = None
    last_overlay_ts = -10**18
    last_line_pixels = None  # (xs, ys, src_w, src_h)
    last_line_pixels_ts = -10**18
    last_pca_xy = None
    last_pca_ts = -10**18
    last_scan = None  # full-frame LaserScan
    next_idx = 0
    cam_writer = None
    cam_size = None  # (w, h)
    lidar_writer = None
    # Per-frame camera-source counters for the meta sidecar.
    cam_src_counts = {'overlay': 0, 'mask': 0, 'raw': 0}
    # Per-frame lidar-source counters (pca-vs-heightband) for the same.
    lidar_src_counts = {'pca': 0, 'heightband': 0}
    # Plan + costmap presence — set as messages arrive; drives the
    # odom-panel title in the bake.
    has_plan_msgs = False
    # All costmap snapshots seen during the bag, in publish order.
    # Each entry is the OccupancyGrid info + raw int8 data + the bag
    # timestamp (ns). The bake selects the most-recent snapshot per
    # frame so the costmap evolves during playback the same way the
    # live GUI's _cb_global_costmap updates it.
    costmap_snaps = []  # list[dict]

    def _ts_for_idx(idx):
        return start_ns + int(idx * 1e9 / fps)

    def _open_lidar_writer():
        nonlocal lidar_writer
        if lidar_writer is not None or not _HAS_CV2:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        lidar_writer = cv2.VideoWriter(
            lidar_mp4_path, fourcc, fps, (_LIDAR_BEV_SIZE, _LIDAR_BEV_SIZE)
        )
        if not lidar_writer.isOpened():
            print(f'WARN: could not open VideoWriter for {lidar_mp4_path}',
                  file=sys.stderr)
            lidar_writer = None

    def _open_cam_writer(sample_bgr):
        nonlocal cam_writer, cam_size
        if cam_writer is not None or not _HAS_CV2:
            return
        h, w = sample_bgr.shape[:2]
        cam_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cam_writer = cv2.VideoWriter(cam_mp4_path, fourcc, fps, cam_size)
        if not cam_writer.isOpened():
            print(f'WARN: could not open VideoWriter for {cam_mp4_path}',
                  file=sys.stderr)
            cam_writer = None

    def _emit_one(idx):
        """Write one frame each to cam + lidar mp4s at output index idx."""
        out_ts = _ts_for_idx(idx)

        # --- Camera frame selection: overlay (fresh) → mask (fresh) → raw ---
        # When the overlay path is chosen, the line_pixels scatter is NOT
        # drawn on top — the overlay frames already contain their own dots.
        cam_src = None
        cam_src_kind = None
        if ((out_ts - last_overlay_ts) / 1e9 < _OVERLAY_FRESH_S
                and last_overlay_bgr is not None):
            cam_src = last_overlay_bgr
            cam_src_kind = 'overlay'
        elif ((out_ts - last_mask_ts) / 1e9 < _MASK_FRESH_S
                and last_mask_bgr is not None):
            cam_src = last_mask_bgr
            cam_src_kind = 'mask'
        elif last_raw_bgr is not None:
            cam_src = last_raw_bgr
            cam_src_kind = 'raw'

        if cam_src is not None and _HAS_CV2:
            _open_cam_writer(cam_src)
            if cam_writer is not None:
                frame = cam_src.copy()
                if cam_size and (frame.shape[1], frame.shape[0]) != cam_size:
                    frame = cv2.resize(frame, cam_size)
                # Overlay line_pixels (red dots) when fresh — but skip when
                # the overlay topic itself was the source, since those frames
                # already include the dots.
                if (cam_src_kind != 'overlay'
                        and last_line_pixels is not None
                        and (out_ts - last_line_pixels_ts) / 1e9 < _LINES_FRESH_S):
                    xs, ys, src_w, src_h = last_line_pixels
                    _overlay_line_pixels(frame, xs, ys, src_w, src_h)
                cam_writer.write(frame)
                if cam_src_kind is not None:
                    cam_src_counts[cam_src_kind] += 1

        # --- Lidar frame selection: PCA if fresh, else heightband ---
        if _HAS_CV2:
            _open_lidar_writer()
            if lidar_writer is not None:
                if ((out_ts - last_pca_ts) / 1e9 < _PCA_FRESH_S
                        and last_pca_xy is not None):
                    bev = render_pca_bev(last_pca_xy, size=_LIDAR_BEV_SIZE)
                    lidar_src_counts['pca'] += 1
                elif last_scan is not None:
                    bev = render_lidar_bev(last_scan, size=_LIDAR_BEV_SIZE)
                    lidar_src_counts['heightband'] += 1
                else:
                    bev = np.full(
                        (_LIDAR_BEV_SIZE, _LIDAR_BEV_SIZE, 3), 128, np.uint8
                    )
                lidar_writer.write(bev)

    # --- Single chronological pass over the bag ---
    for ts_ns, topic, msgtype, msg in iter_bag(bag_dir):
        # Catch the mp4 timeline up to (but not including) this message's
        # output slot, then update state.
        target_idx = min(total_frames - 1,
                         max(0, int((ts_ns - start_ns) / 1e9 * fps)))
        while next_idx < target_idx:
            _emit_one(next_idx)
            next_idx += 1

        # CSV row collection. EKF rows are tagged as /odom so the
        # downstream playback (which only watches /odom) inherits the
        # filtered pose automatically.
        if msgtype in _CSV_HANDLERS_BY_TYPE:
            csv_topic = topic
            if has_ekf and topic == _ODOM_TOPIC:
                continue  # drop raw odom in favour of EKF rows
            if topic == _EKF_ODOM_TOPIC:
                csv_topic = _ODOM_TOPIC
            keys, value_fn = _CSV_HANDLERS_BY_TYPE[msgtype]
            keys = _CSV_KEY_OVERRIDES.get(csv_topic, keys)
            try:
                values = value_fn(msg)
            except Exception as e:  # noqa: BLE001
                print(f'WARN: {topic}: {e}', file=sys.stderr)
            else:
                # GPS rows have variable width — extend the keys list when
                # the message carried a valid covariance block. parse_csv
                # uses the per-row `keys` field to discover which columns
                # mean what, so this lets the bake graduate to the
                # covariance tier without a CSV-wide schema change.
                if (msgtype == 'sensor_msgs/msg/NavSatFix'
                        and len(values) >= 6):
                    keys = 'latitude,longitude,altitude,cov_ee,cov_en,cov_nn'
                rows.append((ts_ns, csv_topic, keys, list(values)))
            continue

        # /plan — nav_msgs/Path. Variable-length poses serialised into
        # one row: keys = "n_poses,x0,y0,x1,y1,...". The bake's
        # render_odom picks up the latest fresh plan and draws it as a
        # green line on the encoders panel.
        if topic == _PLAN_TOPIC and msgtype == 'nav_msgs/msg/Path':
            try:
                poses = msg.poses
                n = len(poses)
                if n > 0:
                    flat = []
                    for ps in poses:
                        flat.append(f'{ps.pose.position.x:.4f}')
                        flat.append(f'{ps.pose.position.y:.4f}')
                    rows.append((ts_ns, topic, 'n_poses,xy_pairs',
                                 [str(n)] + flat))
                    has_plan_msgs = True
            except Exception as e:  # noqa: BLE001
                print(f'WARN: {topic}: {e}', file=sys.stderr)
            continue

        # /global_costmap/costmap — store every snapshot so the bake can
        # show the costmap building up over time. Snapshots are ~10-20 KB
        # of int8 at typical map_padder corridor sizes (140x130 cells), so
        # 5 min of 2 Hz data is well under 10 MB and trivial to compress.
        if (topic == _COSTMAP_TOPIC
                and msgtype == 'nav_msgs/msg/OccupancyGrid'):
            try:
                costmap_snaps.append({
                    'ts_ns': int(ts_ns),
                    'height': int(msg.info.height),
                    'width': int(msg.info.width),
                    'resolution': float(msg.info.resolution),
                    'origin_x': float(msg.info.origin.position.x),
                    'origin_y': float(msg.info.origin.position.y),
                    'data': np.asarray(msg.data, dtype=np.int8).copy(),
                })
            except Exception as e:  # noqa: BLE001
                print(f'WARN: {topic}: {e}', file=sys.stderr)
            continue

        # State updates for the smart emitter.
        if topic == _CAMERA_TOPIC and msgtype == 'sensor_msgs/msg/Image':
            bgr = _image_to_bgr(msg)
            if bgr is not None:
                last_raw_bgr = bgr
        elif topic == _MASK_TOPIC and msgtype == 'sensor_msgs/msg/Image':
            bgr = _image_to_bgr(msg)
            if bgr is not None:
                last_mask_bgr = bgr
                last_mask_ts = ts_ns
        elif topic == _OVERLAY_TOPIC and msgtype == 'sensor_msgs/msg/Image':
            bgr = _image_to_bgr(msg)
            if bgr is not None:
                last_overlay_bgr = bgr
                last_overlay_ts = ts_ns
        elif topic == _LINE_PIXELS_TOPIC:
            xs, ys, src_w, src_h = _decode_line_pixels(msg)
            last_line_pixels = (xs, ys, src_w, src_h)
            last_line_pixels_ts = ts_ns
        elif topic == _LIDAR_TOPIC and msgtype == 'sensor_msgs/msg/LaserScan':
            last_scan = msg
        elif topic == _PCA_TOPIC and msgtype == 'sensor_msgs/msg/LaserScan':
            last_pca_xy = pca_xy_from_scan(msg)
            last_pca_ts = ts_ns

    # Tail: pad the rest of the mp4 timeline up to total_frames.
    while next_idx < total_frames:
        _emit_one(next_idx)
        next_idx += 1

    if cam_writer is not None:
        cam_writer.release()
    if lidar_writer is not None:
        lidar_writer.release()

    cam_frames = total_frames if (
        cam_writer
        or last_raw_bgr is not None
        or last_mask_bgr is not None
        or last_overlay_bgr is not None
    ) else 0
    lidar_frames = total_frames if last_scan is not None or last_pca_xy is not None else 0

    # Write the meta sidecar so the bake can pick per-panel titles + know
    # which optional overlays are available. The "*_source" fields report
    # whichever variant produced the most frames during this extract.
    meta_path = os.path.join(stem_dir, f'{stem_name}_meta.json')
    if cam_frames > 0:
        cam_source = max(cam_src_counts, key=lambda k: cam_src_counts[k])
        if cam_src_counts[cam_source] == 0:
            cam_source = 'raw'
    else:
        cam_source = 'raw'

    if lidar_frames > 0:
        lidar_source = max(lidar_src_counts, key=lambda k: lidar_src_counts[k])
        if lidar_src_counts[lidar_source] == 0:
            lidar_source = 'heightband'
    else:
        lidar_source = 'heightband'

    # Odom tier — costmap requires both an EKF stream AND a costmap
    # snapshot to render (the bake centres the costmap crop on the EKF
    # pose). If the bag has a costmap but no EKF the tier collapses to
    # "ekf" — same rule the live GUI applies via the costmap-stale gate.
    has_costmap = len(costmap_snaps) > 0
    if has_ekf and has_costmap:
        odom_tier = 'ekf_costmap'
    elif has_ekf:
        odom_tier = 'ekf'
    else:
        odom_tier = 'raw'

    # GPS covariance — present iff at least one /gps_fix row carried a
    # valid (finite, positive-diagonal) covariance block. _gps_vals
    # already does the validity gate; here we just count.
    gps_has_covariance = any(
        topic == '/gps_fix' and len(values) >= 6
        for _ts, topic, _keys, values in rows
    )

    meta_payload = {
        'camera_source': cam_source,
        'lidar_source': lidar_source,
        'odom_tier': odom_tier,
        'gps_has_covariance': bool(gps_has_covariance),
        'has_plan': bool(has_plan_msgs),
        'has_costmap': bool(has_costmap),
    }
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta_payload, f, indent=2)
        if verbose:
            print(f'extract_bag: wrote meta to {meta_path} '
                  f'(cam={cam_source}, lidar={lidar_source}, '
                  f'odom={odom_tier}, gps_cov={gps_has_covariance}, '
                  f'plan={has_plan_msgs}, costmap={has_costmap})')
    except OSError as e:
        print(f'WARN: could not write meta sidecar {meta_path}: {e}',
              file=sys.stderr)

    # Re-base timestamps to start at 0 and sort. The costmap snapshots
    # below get the same shift so they share the bake's relative timeline.
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0] if rows else 0
    if rows:
        rows = [(ts - t0, t, k, v) for (ts, t, k, v) in rows]

    # Costmap snapshots: every /global_costmap/costmap message from the
    # bag is preserved with its relative timestamp. The bake selects the
    # most-recent snapshot per output frame, so the cost overlay updates
    # during playback (matching the live GUI's _cb_global_costmap loop).
    # Layout in the npz is flat to avoid object-dtype arrays:
    #   timestamps[N]    int64    publish ns, relative to row t0
    #   shapes[N,2]      int32    (h, w) per snapshot
    #   resolutions[N]   float32  metres/cell
    #   origins[N,2]     float32  (ox, oy) metres
    #   data_offsets[N+1] int64   start index of each snapshot in data_flat
    #   data_flat[total] int8     concatenated OccupancyGrid cells
    if costmap_snaps:
        costmap_path = os.path.join(stem_dir, f'{stem_name}_costmap.npz')
        try:
            n = len(costmap_snaps)
            timestamps = np.array(
                [s['ts_ns'] - t0 for s in costmap_snaps], dtype=np.int64)
            shapes = np.array(
                [[s['height'], s['width']] for s in costmap_snaps],
                dtype=np.int32)
            resolutions = np.array(
                [s['resolution'] for s in costmap_snaps], dtype=np.float32)
            origins = np.array(
                [[s['origin_x'], s['origin_y']] for s in costmap_snaps],
                dtype=np.float32)
            sizes = [int(s['height']) * int(s['width']) for s in costmap_snaps]
            data_offsets = np.zeros(n + 1, dtype=np.int64)
            data_offsets[1:] = np.cumsum(sizes)
            data_flat = np.empty(int(data_offsets[-1]), dtype=np.int8)
            for i, s in enumerate(costmap_snaps):
                start, end = int(data_offsets[i]), int(data_offsets[i + 1])
                data_flat[start:end] = s['data'][:end - start]
            np.savez_compressed(
                costmap_path,
                version=np.int32(2),
                timestamps=timestamps,
                shapes=shapes,
                resolutions=resolutions,
                origins=origins,
                data_offsets=data_offsets,
                data_flat=data_flat,
            )
            if verbose:
                last = costmap_snaps[-1]
                print(f'extract_bag: wrote {n} costmap snapshot(s) to '
                      f'{costmap_path} (latest {last["width"]}x'
                      f'{last["height"]} cells @ {last["resolution"]} '
                      f'm/cell)')
        except OSError as e:
            print(f'WARN: could not write costmap snapshot {costmap_path}: '
                  f'{e}', file=sys.stderr)

    max_cols = _scan_for_max_value_cols(rows)
    header = _csv_header(max_value_cols=max_cols)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for ts, topic, keys, values in rows:
            w.writerow([ts, topic, keys] + values)

    if verbose:
        print(f'extract_bag: wrote {len(rows)} rows to {csv_path}')
        if cam_frames:
            print(f'extract_bag: wrote {cam_frames} camera frames to {cam_mp4_path}')
        if lidar_frames:
            print(f'extract_bag: wrote {lidar_frames} lidar BEV frames to {lidar_mp4_path}')

    return csv_path


def find_bag_dir(path):
    """Resolve a user-supplied path to the actual bag directory.

    Accepts either the bag dir itself or a session dir containing a `bag/`
    subdirectory. Returns the bag dir, or None if not a rosbag2.
    """
    path = os.path.abspath(str(path))
    if os.path.isfile(os.path.join(path, 'metadata.yaml')):
        return path
    nested = os.path.join(path, 'bag')
    if os.path.isfile(os.path.join(nested, 'metadata.yaml')):
        return nested
    return None


# ── CLI ──────────────────────────────────────────────────────────────

def extract_bag_cli(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract a rosbag2 directory to CSV + sidecar mp4 files '
                    'consumable by the AutoNav HUD standalone.'
    )
    parser.add_argument('bag_dir', help='Path to bag dir (or session dir containing bag/)')
    parser.add_argument('--out-dir', default=None,
                        help='Where to write CSV + mp4 (default: alongside the bag)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help=f'Output mp4 framerate (default: {DEFAULT_FPS})')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing CSV if present')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress logs')
    args = parser.parse_args(argv)

    bag_dir = find_bag_dir(args.bag_dir)
    if bag_dir is None:
        print(f'ERROR: not a rosbag2 directory: {args.bag_dir}', file=sys.stderr)
        return 1

    extract_bag(
        bag_dir,
        out_dir=args.out_dir,
        fps=args.fps,
        force=args.force,
        verbose=not args.quiet,
    )
    return 0


if __name__ == '__main__':
    sys.exit(extract_bag_cli())
