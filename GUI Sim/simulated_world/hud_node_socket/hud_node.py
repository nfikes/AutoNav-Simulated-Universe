import csv
import io
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, LaserScan, NavSatFix, Imu, PointCloud2, Joy
    from nav_msgs.msg import Odometry, OccupancyGrid, Path
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from std_msgs.msg import Float32, Int32MultiArray, Bool
    try:
        from sensor_msgs_py import point_cloud2 as _pc2
        _HAS_PC2 = True
    except ImportError:
        _HAS_PC2 = False
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False
    _HAS_PC2 = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# psutil + NVML are only consulted while the Performance mode veil is up,
# so missing imports just blank out the matching stat rather than blocking
# the rest of the GUI.
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

import numpy as np
from PIL import Image as PILImage

import warnings
warnings.filterwarnings('ignore', message='.*fixed.*data aspect.*')

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Ellipse


# ── OSM tile helpers ──────────────────────────────────────────────────
_TILE_CACHE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'tile_cache',
))
_GPS_VIEW_RADIUS_FT = 100  # feet on each side of current position
_GPS_VIEW_RADIUS_M = _GPS_VIEW_RADIUS_FT * 0.3048  # ≈ 30.48 m
_GPS_TILE_ZOOM = 19  # high zoom for ~100 ft view


def _latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to OSM tile (x, y)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds(tx, ty, zoom):
    """Return (lat_min, lat_max, lon_min, lon_max) for a tile."""
    n = 2 ** zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lat_min, lat_max, lon_min, lon_max


def _fetch_tile(tx, ty, zoom):
    """Download one 256×256 hybrid satellite+road tile, with disk cache."""
    os.makedirs(_TILE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_TILE_CACHE_DIR, f'{zoom}_{tx}_{ty}.png')
    if os.path.exists(cache_path):
        return PILImage.open(cache_path).convert('RGB')
    # Google hybrid: satellite imagery + road/label overlay (lyrs=y)
    url = (f'https://mt1.google.com/vt/lyrs=y&x={tx}&y={ty}&z={zoom}')
    req = urllib.request.Request(url, headers={'User-Agent': 'AutoNavGUI/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        with open(cache_path, 'wb') as f:
            f.write(data)
        return PILImage.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None


def _fetch_map_for_gps(lats, lons, zoom=_GPS_TILE_ZOOM):
    """Fetch & stitch OSM tiles covering the given GPS points.

    Returns (numpy_rgb, (lon_min, lon_max, lat_min, lat_max)) or (None, None).
    """
    if not lats or not lons:
        return None, None
    lat_min_d, lat_max_d = min(lats), max(lats)
    lon_min_d, lon_max_d = min(lons), max(lons)
    # Add padding of ~200 ft in degrees
    pad_lat = _GPS_VIEW_RADIUS_M * 3 / 111320.0
    pad_lon = _GPS_VIEW_RADIUS_M * 3 / (111320.0 * math.cos(math.radians((lat_min_d + lat_max_d) / 2)))
    tx_min, ty_min = _latlon_to_tile(lat_max_d + pad_lat, lon_min_d - pad_lon, zoom)
    tx_max, ty_max = _latlon_to_tile(lat_min_d - pad_lat, lon_max_d + pad_lon, zoom)
    # Clamp tile range to avoid huge downloads
    if (tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 100:
        return None, None

    rows = []
    for ty in range(ty_min, ty_max + 1):
        row_imgs = []
        for tx in range(tx_min, tx_max + 1):
            tile = _fetch_tile(tx, ty, zoom)
            if tile is None:
                return None, None
            row_imgs.append(np.array(tile))
        rows.append(np.concatenate(row_imgs, axis=1))
    img = np.concatenate(rows, axis=0)

    # Compute overall extent
    bnd_lat_min, _, bnd_lon_min, _ = _tile_bounds(tx_min, ty_max + 1, zoom)
    _, bnd_lat_max, _, bnd_lon_max = _tile_bounds(tx_max + 1, ty_min, zoom)
    # Fix: tile_bounds for ty_max+1 gives lat below last tile
    bnd_lat_min2, _, _, _ = _tile_bounds(tx_min, ty_max, zoom)
    _, bnd_lat_max2, _, _ = _tile_bounds(tx_min, ty_min, zoom)
    bnd_lon_min2, _, bnd_lon_min3, _ = _tile_bounds(tx_min, ty_min, zoom)
    _, _, _, bnd_lon_max3 = _tile_bounds(tx_max, ty_min, zoom)

    # Simpler: compute from tile coords directly
    n = 2 ** zoom
    bnd_lon_min = tx_min / n * 360.0 - 180.0
    bnd_lon_max = (tx_max + 1) / n * 360.0 - 180.0
    bnd_lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty_min / n))))
    bnd_lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty_max + 1) / n))))

    extent = (bnd_lon_min, bnd_lon_max, bnd_lat_min, bnd_lat_max)
    return img, extent


from PyQt5.QtCore import QObject, pyqtSignal


# Competition operates in VA or MI. GPS fixes outside these boxes are bogus
# (parser glitch, GPS in fault state reporting 0/0, etc.) — refuse to fetch
# tiles for them. Otherwise a single bad fix during LIVE mode triggers a
# multi-tile download for the middle of the Atlantic, which can pin the
# worker for tens of seconds and never produces a useful image.
_GPS_VALID_REGIONS = (
    # (lat_min, lat_max, lon_min, lon_max)
    (36.4, 39.7, -83.8, -75.1),    # Virginia (padded)
    (41.5, 48.4, -90.5, -82.0),    # Michigan (Lower + Upper, padded)
)


def _gps_in_valid_region(lat, lon):
    for lat_min, lat_max, lon_min, lon_max in _GPS_VALID_REGIONS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


def _probe_tile_server(timeout=1.0):
    """Return True if the tile server is reachable. 1 s default keeps the
    offline-detection path snappy — failing this fast means we mark the
    network down without blocking subsequent requests behind a 10 s tile
    timeout."""
    try:
        req = urllib.request.Request(
            'https://mt1.google.com/vt/lyrs=y&x=0&y=0&z=0',
            headers={'User-Agent': 'AutoNavGUI/1.0'},
            method='HEAD',
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


class _TileFetcher(QObject):
    """Off-thread GPS tile fetcher.

    Lives as a QObject with a daemon worker thread. The main (Qt) thread
    posts requests via ``request()``; results come back on the main thread
    via the ``finished`` signal (cross-thread emission is queued by Qt).

    Network state is tri-state (None=unknown, True=online, False=offline)
    so the first request always probes. While offline, requests short-
    circuit and emit ``finished(None, None, ...)`` without attempting the
    multi-tile fetch. A 60 s idle re-probe re-arms when wifi/hotspot
    becomes available, and the last successful-area request is automatic-
    ally retried on the offline → online transition so cached tiles for
    the current area get populated as soon as the radio comes back.
    """

    finished = pyqtSignal(object, object, int, str)  # img, extent, req_id, tag
    online_changed = pyqtSignal(bool)

    _PROBE_INTERVAL_S = 60.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._online = None
        self._last_probe_t = 0.0
        self._last_request = None  # (lats, lons, tag) for reconnect retry
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name='gps-tile-fetcher', daemon=True)
        self._worker.start()

    def request(self, lats, lons, req_id, tag):
        self._queue.put(('request', list(lats), list(lons), int(req_id), str(tag)))

    def probe(self):
        """Force a non-blocking probe (used by the GUI watchdog)."""
        self._queue.put(('probe',))

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(('stop',))

    def _run(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self._PROBE_INTERVAL_S)
            except queue.Empty:
                # Idle wakeup: probe + auto-retry on reconnect
                self._auto_probe_and_retry()
                continue
            kind = item[0]
            if kind == 'stop':
                return
            if kind == 'probe':
                self._do_probe()
                continue
            if kind == 'request':
                _, lats, lons, req_id, tag = item
                self._handle_request(lats, lons, req_id, tag)

    def _handle_request(self, lats, lons, req_id, tag):
        if self._online is None:
            self._do_probe()
        elif self._online is False and \
                (time.monotonic() - self._last_probe_t) > self._PROBE_INTERVAL_S:
            self._do_probe()

        if self._online is False:
            self.finished.emit(None, None, req_id, tag)
            return

        img, extent = _fetch_map_for_gps(lats, lons)
        if img is None:
            # Fetch failed mid-flight — re-probe to mark offline if so
            self._do_probe()
        else:
            self._last_request = (lats, lons, tag)
        self.finished.emit(img, extent, req_id, tag)

    def _do_probe(self):
        self._last_probe_t = time.monotonic()
        new_online = _probe_tile_server(timeout=1.0)
        if new_online != self._online:
            self._online = new_online
            self.online_changed.emit(bool(new_online))
        else:
            self._online = new_online

    def _auto_probe_and_retry(self):
        if self._online is not False:
            return
        prev = self._online
        self._do_probe()
        if self._online is True and prev is False and self._last_request is not None:
            lats, lons, tag = self._last_request
            # req_id = -1 marks this as a worker-initiated retry; the main
            # thread accepts it regardless of its current id counter so
            # the just-reconnected fetch isn't dropped as stale.
            img, extent = _fetch_map_for_gps(lats, lons)
            self.finished.emit(img, extent, -1, f'{tag}+reconnect')


class _CameraFrameWorker(QObject):
    """Off-thread camera display-prep worker.

    Takes the ROS-thread-decoded RGB array, downscales to a bounded max
    dimension, scales the optional line-detector overlay by the same
    factor, and emits a ready-to-paint payload back to the main thread.
    Latest-wins: if the worker is still busy when a new submit arrives,
    the new frame overwrites the pending slot and the old one is dropped
    — the right behavior for a live view where stale frames have no value.

    On a 1920×1200 ZED frame the matplotlib AxesImage.set_data +
    subsequent Agg paint cost 5–20 ms on the Qt main thread per frame.
    Downscaling to ~720 px max brings that to ~1 ms, leaving the Qt event
    loop free for keyboard/mouse and the other sensor boxes' paints.
    """

    frame_ready = pyqtSignal(object, object, bool, str)
    # array (downscaled, contiguous; HxW for mask, HxWx3 for raw/overlay),
    # overlay_xy (Nx2) or None, line_fresh, source ('raw'|'mask'|'overlay')

    def __init__(self, max_dim=320, min_interval_s=0.0, parent=None):
        super().__init__(parent)
        # 320 px max dim — panel is ~400 px wide on screen, but the
        # canvas is now dpi=40 so anything bigger just wastes set_data
        # bytes that matplotlib will resample down anyway. Combined
        # with the canvas DPI drop this keeps a single paint < ~5 ms
        # on Jetson.
        # min_interval_s=0 lets the worker emit as fast as the source can
        # supply; latest-wins drops backpressure. The active source rate
        # is now determined upstream — typically the line detector's
        # debug/mask topic (4 Hz at publish_interval_ms=250) or the raw
        # ZED stream (15 Hz) when the detector is silent.
        self._max_dim = int(max_dim)
        self._min_interval_s = float(min_interval_s)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending = None      # (arr, line_xy, line_fresh, source)
        self._last_emit_t = 0.0
        # Backpressure handshake with the main-thread slot. When set,
        # the worker is free to emit. The slot calls slot_ready() at the
        # start of each invocation to re-arm. Initially set so the first
        # emit isn't blocked. Without this the cross-thread queued
        # signal piles up paint events when the source rate (15 Hz
        # camera) exceeds the main-thread paint capacity, which is what
        # was making the displayed frame visibly lag the camera.
        self._slot_ready = threading.Event()
        self._slot_ready.set()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name='camera-frame-worker', daemon=True)
        self._worker.start()

    def submit(self, arr, line_xy=None, line_fresh=False, source='raw'):
        with self._cv:
            self._pending = (arr, line_xy, bool(line_fresh), str(source))
            self._cv.notify()

    def shutdown(self):
        self._stop_event.set()
        # Wake the worker if it's idling on either the submit cv or the
        # rate-limit sleep. _slot_ready isn't currently waited on but
        # set it anyway for symmetry — costs nothing.
        self._slot_ready.set()
        with self._cv:
            self._cv.notify()

    def slot_ready(self):
        """Main thread calls this at the start of each frame_ready slot
        invocation. The worker uses it to decide whether to emit the
        next frame or drop it (latest-wins). Effective rate equals
        whichever of the source / slot is slower — no signal backlog.
        """
        self._slot_ready.set()

    def _run(self):
        while not self._stop_event.is_set():
            with self._cv:
                while self._pending is None and not self._stop_event.is_set():
                    self._cv.wait()
                if self._stop_event.is_set():
                    return
                arr_full, line_xy, line_fresh, source = self._pending
                self._pending = None

            # Rate-limit emissions: drop into a sleep that yields to any
            # newer submits queueing in _pending. When we wake, if a new
            # frame arrived we'll pick it up on the next loop instead of
            # processing the stale one we just took.
            wait = self._min_interval_s - (time.monotonic() - self._last_emit_t)
            if wait > 0:
                # Sleep but watch for shutdown; new submits overwrite
                # _pending and we'll grab them on the next iteration.
                self._stop_event.wait(timeout=wait)
                if self._stop_event.is_set():
                    return
                with self._lock:
                    if self._pending is not None:
                        # A fresher frame is waiting — discard the one we
                        # took and let the next loop iteration handle it.
                        continue

            # Backpressure check before doing CPU work. If the previous
            # frame_ready is still being painted, drop this submit;
            # latest-wins will deliver a fresher one when slot_ready()
            # is called again. Skipping the downscale here also keeps
            # the worker thread idle during slow paints instead of
            # spending cycles on frames that will be discarded.
            if not self._slot_ready.is_set():
                continue

            try:
                arr_small, scale = self._downscale(arr_full)
            except Exception:
                continue

            overlay = None
            if (line_xy is not None and line_fresh and
                    getattr(line_xy[0], 'size', 0) > 0):
                xs = np.asarray(line_xy[0], dtype=np.float32) * scale
                ys = np.asarray(line_xy[1], dtype=np.float32) * scale
                overlay = np.column_stack([xs, ys])

            self._slot_ready.clear()
            self._last_emit_t = time.monotonic()
            self.frame_ready.emit(arr_small, overlay, line_fresh, source)

    def _downscale(self, rgb):
        h, w = rgb.shape[:2]
        largest = max(h, w)
        if largest <= self._max_dim:
            # Ensure contiguous so matplotlib's set_data avoids a copy.
            return np.ascontiguousarray(rgb), 1.0
        scale = self._max_dim / float(largest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # PIL bilinear is good quality for live monitoring and runs in C.
        pil = PILImage.fromarray(rgb)
        pil = pil.resize((new_w, new_h), PILImage.BILINEAR)
        return np.ascontiguousarray(np.asarray(pil)), scale


def _bresenham_line(img, x0, y0, x1, y1, color):
    """Draw a line on a numpy image using Bresenham's algorithm. Module
    scope so it's reachable from both _LidarFrameWorker and the legacy
    static _render_lidar_bev (which the worker has now superseded but
    that we keep around for the playback BEV path).
    """
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


class _JoyNavBus(QObject):
    """Cross-thread bridge for Xbox-controller GUI navigation.

    HudNode's /joy callback runs on the rclpy executor thread and
    cannot poke Qt widgets directly. It pushes navigation events
    here via ``push(direction)``; the ``nav`` pyqtSignal auto-
    marshals to the main thread (queued connection by default for
    cross-thread emits) where HudWindow synthesises the matching
    QKeyEvent so the existing keyboard-navigation paths fire
    unchanged.
    """
    nav = pyqtSignal(str)  # 'up' | 'down' | 'left' | 'right' | 'select'

    def push(self, direction: str) -> None:
        self.nav.emit(direction)


class _LidarFrameWorker(QObject):
    """Off-thread lidar BEV rasterizer.

    Two rendering modes, selected by the source tag at submit time:

    * 'heightband' — full SICK 360° scan as a BEV with green hit dots,
      white driveable lines, and black obstacle shadows. The classic
      lidar debug view, but a few hundred Python Bresenham line draws
      per frame; runs here on the worker so the main thread never
      blocks on it.
    * 'pca' — PCA-filtered 180° LaserScan, i.e. the same costmap-ready
      scan that nav2's obstacle layer consumes from
      pointcloud_to_laserscan in slam.launch.py. Just hits, no
      shadows. Fully vectorized in numpy, sub-ms — what makes the
      lidar panel keep pace with the lidar's true publish rate when
      the grade detector is running.

    Backpressure / latest-wins / slot_ready handshake mirror
    _CameraFrameWorker so the lidar paint can't pile up Qt events
    behind the camera or itself.
    """

    frame_ready = pyqtSignal(object, str)   # img (HxWx3 RGB), source

    def __init__(self, size=320, parent=None):
        super().__init__(parent)
        # 320 px BEV — half the previous 480 px, ~4x fewer pixels for
        # matplotlib to set_data and Agg to paint. Combined with the
        # cam/lidar canvas dpi=40 drop, this is what makes the GUI keep
        # its 30 FPS target even with both sensor panels active.
        self._size = int(size)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending = None
        self._slot_ready = threading.Event()
        self._slot_ready.set()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name='lidar-frame-worker', daemon=True)
        self._worker.start()

    def submit(self, scan, source='heightband'):
        with self._cv:
            self._pending = (scan, str(source))
            self._cv.notify()

    def slot_ready(self):
        """Main thread calls this at the start of each frame_ready slot
        invocation to flow-control the worker (see _CameraFrameWorker).
        """
        self._slot_ready.set()

    def shutdown(self):
        self._stop_event.set()
        self._slot_ready.set()
        with self._cv:
            self._cv.notify()

    def _run(self):
        while not self._stop_event.is_set():
            with self._cv:
                while self._pending is None and not self._stop_event.is_set():
                    self._cv.wait()
                if self._stop_event.is_set():
                    return
                scan, source = self._pending
                self._pending = None

            # Backpressure: if the slot is still painting, drop this
            # submit; latest-wins will supply a fresher scan when the
            # slot acknowledges.
            if not self._slot_ready.is_set():
                continue

            try:
                if source == 'pca':
                    img = self._render_pca_bev(scan, self._size)
                else:
                    img = self._render_heightband_bev(scan, self._size)
            except Exception:
                continue

            # Match the orientation the previous in-place renderer
            # established (rot90 + fliplr → +x forward / +y right on
            # screen) so the panel looks identical to before.
            img = np.rot90(img, 1)
            img = np.fliplr(img)
            img = np.ascontiguousarray(img)

            self._slot_ready.clear()
            self.frame_ready.emit(img, source)

    @staticmethod
    def _draw_robot_marker(img, cx, cy):
        """Forward-pointing red triangle at the robot origin. Replaces
        the old 3x3 red square so it can't be mistaken for a PCA
        obstacle hit (which is also red, but square). cv2.fillPoly is
        a single C call that releases the GIL — no per-frame cost
        worth measuring. The triangle's apex points +x in render
        space, which lands as 'up = forward' on screen after the
        worker's rot90 + fliplr orientation pass.
        """
        if _HAS_CV2:
            tri = np.array([
                [cx - 4, cy - 4],
                [cx - 4, cy + 4],
                [cx + 5, cy],
            ], dtype=np.int32)
            cv2.fillPoly(img, [tri], (255, 0, 0))
            return
        # Fallback for cv2-less environments — a column scan that
        # narrows linearly from base (dx=-4) to apex (dx=+5).
        h, w = img.shape[:2]
        for dx in range(-4, 6):
            t = (dx + 4) / 9.0
            half_h = max(0, int(round(4 * (1.0 - t))))
            nx = cx + dx
            for dy in range(-half_h, half_h + 1):
                ny = cy + dy
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = (255, 0, 0)

    @staticmethod
    def _render_pca_bev(xy, size, max_range=10.0):
        """BEV from a PCA (N, 2) obstacle XY array.

        White rays from the robot origin to each hit (same visual
        language as the heightband view's white driveable lines) plus
        a 3x3 red stamp at each hit. Robot origin marker draws last so
        it always overlays cleanly on top of overlapping rays.
        """
        img = np.full((size, size, 3), 128, dtype=np.uint8)
        cx, cy = size // 2, size // 2
        scale = (size // 2 - 2) / max_range

        if xy is not None and len(xy) > 0:
            xs = (cx + xy[:, 0] * scale).astype(np.int32)
            ys = (cy - xy[:, 1] * scale).astype(np.int32)
            # 1-pixel margin matches the 3x3 stamp so the broadcast
            # writes below can't run off the canvas edges.
            in_bounds = ((xs >= 1) & (xs < size - 1) &
                         (ys >= 1) & (ys < size - 1))
            xs = xs[in_bounds]
            ys = ys[in_bounds]

            # White rays from origin → each hit. Drawn first so the
            # red dots stamp on top of the line endpoints. cv2.line is
            # a C call that releases the GIL during each invocation,
            # so the main thread can keep painting the camera between
            # rays even at high PCA point counts.
            if _HAS_CV2:
                white = (255, 255, 255)
                for xi, yi in zip(xs, ys):
                    cv2.line(img, (cx, cy),
                             (int(xi), int(yi)), white, 1)
            else:
                for xi, yi in zip(xs, ys):
                    _bresenham_line(
                        img, cx, cy, int(xi), int(yi), (255, 255, 255))

            # 3x3 red stamp per hit (was 5x5 — too chunky at clustered
            # hits and overlapping into adjacent points).
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    img[ys + dy, xs + dx] = (255, 0, 0)

        _LidarFrameWorker._draw_robot_marker(img, cx, cy)
        return img

    @staticmethod
    def _render_heightband_bev(scan, size):
        """BEV: white driveable + black shadows + green hits.

        Uses cv2.line for the per-ray drawing when available. The pure
        Python Bresenham inner loop was the cause of camera FPS
        collapsing whenever the lidar panel was on — a Python loop
        across ~1000 rays × ~200 pixel writes per ray holds the GIL
        for tens of milliseconds, starving the main thread of paint
        cycles. cv2.line is a C call that releases the GIL during each
        invocation, so the main thread can paint the camera between
        rays. Falls back to the pure-Python path if cv2 isn't on the
        Jetson's Python path.
        """
        img = np.full((size, size, 3), 128, dtype=np.uint8)
        cx, cy = size // 2, size // 2
        max_range = scan.range_max if scan.range_max > 0 else 10.0
        scale = (size // 2 - 2) / max_range

        angles = (np.arange(len(scan.ranges)) * scan.angle_increment
                  + scan.angle_min)
        ranges = np.array(scan.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges >= scan.range_min)

        if not np.any(valid):
            _LidarFrameWorker._draw_robot_marker(img, cx, cy)
            return img

        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        sx_all = (cx + max_range * cos_a * scale).astype(np.int32)
        sy_all = (cy - max_range * sin_a * scale).astype(np.int32)
        ex_all = (cx + ranges * cos_a * scale).astype(np.int32)
        ey_all = (cy - ranges * sin_a * scale).astype(np.int32)
        is_hit = valid & (ranges < max_range)
        is_clear = valid & (ranges >= max_range)

        if _HAS_CV2:
            white = (255, 255, 255)
            black = (0, 0, 0)
            clear_idx = np.where(is_clear)[0]
            for i in clear_idx:
                cv2.line(img, (cx, cy),
                         (int(sx_all[i]), int(sy_all[i])), white, 1)
            hit_idx = np.where(is_hit)[0]
            for i in hit_idx:
                ex = int(ex_all[i])
                ey = int(ey_all[i])
                cv2.line(img, (cx, cy), (ex, ey), white, 1)
                cv2.line(img, (ex, ey),
                         (int(sx_all[i]), int(sy_all[i])), black, 1)
            # Green hit dots, fully vectorized.
            hit_xs = ex_all[is_hit]
            hit_ys = ey_all[is_hit]
            in_bounds = ((hit_xs >= 0) & (hit_xs < size) &
                         (hit_ys >= 0) & (hit_ys < size))
            img[hit_ys[in_bounds], hit_xs[in_bounds]] = (0, 255, 0)
        else:
            # GIL-bound fallback. Only hit when cv2 isn't installed —
            # the lidar panel will likely hammer the camera's paint
            # rate here, but at least the geometry still draws.
            for i in range(len(ranges)):
                r = ranges[i]
                if not np.isfinite(r) or r < scan.range_min:
                    continue
                sx = int(sx_all[i])
                sy = int(sy_all[i])
                if r >= max_range:
                    _bresenham_line(img, cx, cy, sx, sy, (255, 255, 255))
                else:
                    ex = int(ex_all[i])
                    ey = int(ey_all[i])
                    _bresenham_line(img, cx, cy, ex, ey, (255, 255, 255))
                    _bresenham_line(img, ex, ey, sx, sy, (0, 0, 0))
                    if 0 <= ex < size and 0 <= ey < size:
                        img[ey, ex] = (0, 255, 0)

        _LidarFrameWorker._draw_robot_marker(img, cx, cy)
        return img


from PyQt5.QtCore import (
    QEasingCurve, QEvent, QParallelAnimationGroup, QPoint,
    QPropertyAnimation, QRect, Qt, QTimer,
)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QKeyEvent, QPainter, QPalette, QPen, QPolygon,
    QRegion,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _make_dark_canvas(nrows=1, ncols=1, figsize=(3.6, 2.4), dpi=80):
    """Create a small matplotlib figure + canvas with a dark theme.

    The camera and lidar panels pass dpi=40 to halve the rasterized
    pixmap size (and roughly quarter the Agg paint cost) — without
    that, two image panels updating at 15+ Hz starve the main thread
    and the GUI FPS collapses below the 30 Hz target. Plot panels
    keep the default 80 dpi so axis ticks/labels stay crisp.
    """
    fig = Figure(figsize=figsize, dpi=dpi, facecolor='#1e1e1e')
    fig.subplots_adjust(left=0.18, right=0.94, top=0.88, bottom=0.22)
    axes = fig.subplots(nrows, ncols)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet("border: none;")
    return fig, axes, canvas


def _make_mini_canvas(figsize=(1.4, 1.0)):
    """Create a tiny single-axis canvas for individual power traces."""
    fig = Figure(figsize=figsize, dpi=80, facecolor='#1e1e1e')
    fig.subplots_adjust(left=0.22, right=0.94, top=0.78, bottom=0.28)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setStyleSheet("border: none;")
    return fig, ax, canvas


def _style_ax(ax, title=None):
    """Apply dark styling to a matplotlib axes."""
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.xaxis.label.set_color('#888')
    ax.yaxis.label.set_color('#888')
    if title:
        ax.set_title(title, fontsize=8, color='#ccc')


class _ModeOverlay(QWidget):
    """Unified PERF + REC overlay.

    One widget that's wider than the parent (≈ 2× window width + the 60°
    diagonal width). It paints a blue trapezoid on the left, a red
    trapezoid on the right, with the diagonal between them. Horizontal
    translation chooses what's visible:

        PERF — widget at x = 0; blue trapezoid covers the window.
        REC  — widget at x = -(W + diag); red trapezoid covers the window.
        BOTH — widget at x = -(W + diag) / 2; diagonal centred, both
               trapezoids' visible slabs sit either side.

    Two child labels (PERF + REC) shimmy between solo-centred and
    split-trapezoid-centred positions as the state changes. A third
    label (CPU/GPU stats) tracks the PERF label.

    All transitions are QParallelAnimationGroups with InOutCubic easing
    on the widget pos and on each label pos, so the whole composition
    moves as one piece.
    """

    OFF = 'off'
    PERF = 'perf'
    REC = 'rec'
    BOTH = 'both'

    _BLUE_FILL = QColor(40, 90, 200, 110)
    _BLUE_BORDER = QColor(20, 50, 140, 220)
    _RED_FILL = QColor(255, 0, 0, 14)
    _RED_BORDER = QColor(255, 0, 0, 200)
    _BORDER_PX = 8
    _ANIM_MS = 300

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        title_font = QFont()
        title_font.setPointSize(96)
        title_font.setBold(True)

        self.perf_label = QLabel('PERF', self)
        self.perf_label.setFont(title_font)
        self.perf_label.setStyleSheet(
            'color: #ffffff; background: transparent; border: none;'
        )
        self.perf_label.setAlignment(Qt.AlignCenter)
        self.perf_label.adjustSize()

        self.rec_label = QLabel('REC', self)
        self.rec_label.setFont(title_font)
        self.rec_label.setStyleSheet(
            'color: rgba(255, 80, 80, 230);'
            ' background: transparent; border: none;'
        )
        self.rec_label.setAlignment(Qt.AlignCenter)
        self.rec_label.adjustSize()

        stats_font = QFont()
        stats_font.setPointSize(20)
        stats_font.setFamily('Consolas')
        self.perf_stats_label = QLabel(
            'CPU   Before: --     Now: --\n'
            'GPU   Before: --     Now: --',
            self,
        )
        self.perf_stats_label.setFont(stats_font)
        self.perf_stats_label.setStyleSheet(
            'color: #cfe4ff; background: transparent; border: none;'
        )
        self.perf_stats_label.setAlignment(Qt.AlignCenter)
        self.perf_stats_label.adjustSize()

        # Screen dims — set by configure_for_screen, also recomputed on
        # window resize. Until then, harmless placeholder values.
        self._sw = 1
        self._sh = 1
        self._diag = 1
        self._anim = None
        self._state = self.OFF
        self.hide()

    # ── Geometry helpers ─────────────────────────────────────────────

    def configure_for_screen(self, sw, sh):
        """Set / refresh the parent (window) dimensions and recompute the
        widget's total width + diagonal width. Snaps to the current state
        without animation so resize jumps are atomic."""
        import math as _math
        sw = max(1, int(sw))
        sh = max(1, int(sh))
        self._sw = sw
        self._sh = sh
        # 60° diagonal — width = h * cot(60°).
        self._diag = max(1, int(sh / _math.tan(_math.radians(60.0))))
        widget_w = 2 * sw + self._diag
        self.setFixedSize(widget_w, sh)
        # Recenter labels for whatever state we're in.
        self._apply_state(self._state, animate=False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W, H, D = self._sw, self._sh, self._diag

        blue_poly = QPolygon([
            QPoint(0, 0), QPoint(W + D, 0),
            QPoint(W, H), QPoint(0, H),
        ])
        red_poly = QPolygon([
            QPoint(W + D, 0), QPoint(2 * W + D, 0),
            QPoint(2 * W + D, H), QPoint(W, H),
        ])

        # Fills first.
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._BLUE_FILL)
        painter.drawPolygon(blue_poly)
        painter.setBrush(self._RED_FILL)
        painter.drawPolygon(red_poly)

        # Then thick polygon outlines. Both colors stamp their own
        # border along the shared diagonal — in BOTH mode that reads as
        # an emphatic split-stripe; in solo modes the diagonal sits at
        # one screen edge and acts as that side's border.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(self._BLUE_BORDER, self._BORDER_PX))
        painter.drawPolygon(blue_poly)
        painter.setPen(QPen(self._RED_BORDER, self._BORDER_PX))
        painter.drawPolygon(red_poly)

    # ── State machine ───────────────────────────────────────────────

    def _widget_pos_for(self, state):
        if state == self.PERF:
            return QPoint(0, 0)
        if state == self.REC:
            return QPoint(-(self._sw + self._diag), 0)
        if state == self.BOTH:
            return QPoint(-(self._sw + self._diag) // 2, 0)
        # OFF — pinned past the window; exact value irrelevant since
        # we hide right after the animation finishes.
        return self.pos()

    def _centre_label(self, label, cx, cy):
        return QPoint(cx - label.width() // 2, cy - label.height() // 2)

    def _label_positions_for(self, state):
        """Return (perf, stats, rec) widget-local top-left QPoints for
        the three labels at the given state."""
        W, H, D = self._sw, self._sh, self._diag
        offset_both = (W + D) // 2  # widget local x of the screen origin in BOTH

        if state == self.PERF:
            perf_cx = W // 2
            rec_cx = W + D + W // 2  # off-screen-right — never visible
        elif state == self.REC:
            perf_cx = W // 2  # off-screen-left — never visible
            rec_cx = W + D + W // 2
        elif state == self.BOTH:
            # Trapezoid centroids in screen coords: (W/4, H/2) and (3W/4, H/2).
            perf_cx = W // 4 + offset_both
            rec_cx = 3 * W // 4 + offset_both
        else:  # OFF — leave wherever they were
            return (self.perf_label.pos(), self.perf_stats_label.pos(),
                    self.rec_label.pos())

        cy = H // 2
        # Stack the stats label below the PERF title.
        stats_cy = cy + self.perf_label.height() // 2 + 12 \
            + self.perf_stats_label.height() // 2
        return (
            self._centre_label(self.perf_label, perf_cx, cy),
            self._centre_label(self.perf_stats_label, perf_cx, stats_cy),
            self._centre_label(self.rec_label, rec_cx, cy),
        )

    def state(self):
        return self._state

    def set_state(self, new_state, animate=True):
        if new_state == self._state:
            return
        prev = self._state
        if self._anim is not None:
            self._anim.stop()
            self._anim = None

        if new_state == self.OFF:
            # Slide off in the direction that matches the colour just
            # deactivated, then hide.
            if prev == self.PERF:
                off_pos = QPoint(self._sw, 0)  # right
            elif prev == self.REC:
                off_pos = QPoint(-(2 * self._sw + self._diag), 0)  # left
            else:
                off_pos = self.pos()  # snap-hide for unusual paths

            def _on_done():
                self.hide()

            self._animate_widget_pos(off_pos, animate, _on_done)
            self._state = new_state
            return

        if prev == self.OFF:
            # Coming from OFF — snap to target without sliding from
            # nowhere; the visible slide-in happens only when going
            # between active states.
            self._apply_state(new_state, animate=False)
            self.show()
            self.raise_()
            self._state = new_state
            return

        # Active-to-active transition — full animation of widget pos
        # and all label positions in parallel.
        self._apply_state(new_state, animate=animate)
        self._state = new_state

    def _apply_state(self, state, animate=True):
        target_widget = self._widget_pos_for(state)
        perf_pos, stats_pos, rec_pos = self._label_positions_for(state)
        if not animate:
            self.move(target_widget)
            self.perf_label.move(perf_pos)
            self.perf_stats_label.move(stats_pos)
            self.rec_label.move(rec_pos)
            return
        group = QParallelAnimationGroup(self)
        for widget, target in [
            (self, target_widget),
            (self.perf_label, perf_pos),
            (self.perf_stats_label, stats_pos),
            (self.rec_label, rec_pos),
        ]:
            a = QPropertyAnimation(widget, b'pos')
            a.setDuration(self._ANIM_MS)
            a.setStartValue(widget.pos())
            a.setEndValue(target)
            a.setEasingCurve(QEasingCurve.InOutCubic)
            group.addAnimation(a)
        self._anim = group
        group.start()

    def _animate_widget_pos(self, target, animate, on_finished=None):
        if not animate:
            self.move(target)
            if on_finished is not None:
                on_finished()
            return
        a = QPropertyAnimation(self, b'pos', self)
        a.setDuration(self._ANIM_MS)
        a.setStartValue(self.pos())
        a.setEndValue(target)
        a.setEasingCurve(QEasingCurve.InOutCubic)
        if on_finished is not None:
            a.finished.connect(on_finished)
        self._anim = a
        a.start()


class HudWindow(QMainWindow):
    """1920x720 dark-themed HUD window for the AutoNav bar display."""

    # Two complete color stylesheets. The whole UI is built in dark colors;
    # at the end of __init__ we walk the widget tree and substitute hex
    # codes via _DARK_TO_LIGHT to switch to light. Toggling the theme button
    # walks again with the inverse map. The maps are kept injective so
    # substitution can run in either direction.
    _DARK_TO_LIGHT = {
        '#141414': '#ededed',  # window bg
        '#1a1a1a': '#e0e0e0',  # pressed/disabled bg
        '#1e1e1e': '#fafafa',  # panel bg / sensor cell
        '#2a2a2a': '#e8e8e8',  # button bg
        '#3a3a3a': '#d0d0d0',  # button hover
        '#4a1a1a': '#f5dada',  # quit/exit bg
        '#6a2a2a': '#e8b8b8',  # quit hover
        '#300a0a': '#d8a8a8',  # quit pressed
        '#1a2a3a': '#dde8f5',  # connect bg
        '#2a3a4a': '#c5d8e8',  # connect hover
        '#0a1a2a': '#b0c8de',  # connect pressed
        '#8b0000': '#bb1010',  # E-stop bg (still red but lighter)
        '#a00000': '#cc2020',  # E-stop hover
        '#600000': '#990000',  # E-stop pressed
        '#f00':    '#aa1111',  # E-stop border
        '#fff':    '#fefefe',  # E-stop text (kept near-white on red)
        '#333':    '#cccccc',  # disabled border
        '#444':    '#b8b8b8',  # general border
        '#555':    '#9c9c9c',  # button border
        '#666':    '#5e5e5e',  # group label fg — must stay readable on white
        '#888':    '#6c6c6c',  # dim fg
        '#aaa':    '#4e4e4e',  # subtle label fg
        '#ccc':    '#525252',  # mpl ax title
        '#dcdcdc': '#202020',  # text fg
        '#ffffff': '#000000',  # title text (full 6-char form)
        '#0f0':    '#0a8800',  # active green text
        '#0af':    '#0a5a9a',  # info blue
        '#ff0':    '#a06000',  # yellow text status (dots use #ffff00 below)
        '#f44':    '#cc3030',  # red dot
        '#4f4':    '#0db000',  # green dot — distinct from #0f0 so reverse map is bijective
        '#111111': '#f5f5f5',  # mpl axes facecolor
        '#0a0a0a': '#fbfbfb',  # process terminal bg (overridden via _restyle_terminal too)
    }

    def __init__(self, ros_node=None):
        super().__init__()
        self._ros_node = ros_node
        self.setWindowTitle('AutoNav HUD')
        self.resize(1920, 720)
        self.showFullScreen()
        self.setCursor(Qt.BlankCursor)

        # GUI defaults to light theme. The widget tree is built in dark
        # colors below, then _apply_theme() is called at the end of
        # __init__ to flip the widget tree + QPalette + matplotlib canvases
        # to whichever theme is selected.
        self._theme = 'light'

        # Build the regex once. Matches a 6-char hex first, then a 3-char
        # hex (negative lookahead so #fff doesn't capture inside #fffabc).
        self._hex_re = re.compile(
            r'#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])'
        )

        # Initial QPalette (dark). _apply_theme overwrites this at the end.
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(20, 20, 20))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(40, 40, 40))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

        # Playback state
        self._pb_rows = []
        self._pb_duration_ns = 0
        self._pb_timer = None
        self._pb_wall_start = 0.0
        self._pb_row_idx = 0
        self._pb_state = 'idle'
        self._pb_pause_elapsed_ns = 0
        self._pb_elapsed_ns = 0
        self.sensor_value_labels = {}

        # GPS tile fetcher — runs on a worker thread so the Qt event loop
        # never blocks on the multi-tile HTTP fetch. Without this every
        # cache-miss in LIVE mode stalled the whole HUD for N×10 s while
        # urllib.urlopen waited for a tile that never came (no wifi). The
        # main thread now posts a request and gets the result via signal.
        self._tile_fetcher = _TileFetcher(self)
        self._tile_fetcher.finished.connect(self._on_tile_fetch_finished)
        self._tile_fetcher.online_changed.connect(self._on_tile_network_changed)
        self._tile_request_id = 0
        self._tile_request_pending = False
        self._tile_network_online = None  # tri-state; mirrors fetcher's view
        self._tile_offline_logged = False

        # Camera frame worker — decouples display prep (downscale, overlay
        # scaling) from the Qt main thread so the camera can paint at a
        # stable ~7.5 Hz (half the ZED 2i source rate) without the main
        # thread spending 5–20 ms per frame on full-res matplotlib
        # renders. The ROS image callback submits directly to this worker,
        # bypassing the 5 Hz _live_tick polling cap that previously
        # capped camera FPS.
        self._camera_worker = _CameraFrameWorker(parent=self)
        self._camera_worker.frame_ready.connect(self._on_camera_frame_ready)
        if self._ros_node is not None:
            self._ros_node.camera_worker = self._camera_worker
        # Xbox controller D-pad → arrow keys, A button → Enter (when
        # not in auto mode and not in test recording). HudNode's joy
        # callback pushes into this bus; the queued-connection signal
        # marshals onto the main thread where we synthesise a
        # QKeyEvent.
        self._joy_nav_bus = _JoyNavBus(parent=self)
        self._joy_nav_bus.nav.connect(self._on_joy_nav)
        if self._ros_node is not None:
            self._ros_node.joy_nav_bus = self._joy_nav_bus
        # Rolling window of recent frame-arrival timestamps for the FPS
        # counter overlaid on the camera panel. deque is bounded so the
        # average tracks the last ~10 frames (~1.3 s window at 7.5 Hz).
        self._cam_fps_deque = deque(maxlen=10)

        # Lidar frame worker — same pattern as the camera worker. The
        # 'heightband' source is the slow Bresenham BEV (kept for the
        # raw mode); the 'pca' source is a vectorized BEV from the
        # PCA-filtered LaserScan that already feeds nav2's costmap.
        # Submission decision is made on the ROS thread in _cb_scan
        # and _cb_pca_scan: PCA wins when fresh, heightband fills in
        # otherwise.
        self._lidar_worker = _LidarFrameWorker(parent=self)
        self._lidar_worker.frame_ready.connect(self._on_lidar_frame_ready)
        if self._ros_node is not None:
            self._ros_node.lidar_worker = self._lidar_worker
        self._lidar_fps_deque = deque(maxlen=10)

        # Global GUI responsiveness metric. The QTimer is scheduled at
        # 30 Hz (33 ms interval); if the main thread is busy, the slot
        # fires late and the inter-tick rate drops below 30 FPS. Reading
        # this lets the operator catch sensor paints starving the event
        # loop in real time — keyboard/cursor responsiveness tracks
        # this number directly.
        self._gui_fps_deque = deque(maxlen=15)
        self._gui_fps_timer = QTimer()
        self._gui_fps_timer.setInterval(33)
        self._gui_fps_timer.timeout.connect(self._gui_fps_tick)
        self._gui_fps_timer.start()

        # Live mode state (defined before the rolling buffers so the
        # maxlen constants are available for deque construction below).
        self._live_active = False
        self._live_timer = None      # QTimer for 10 Hz refresh
        self._live_t0 = 0.0          # wall-clock start for power X axis
        self._live_gps_maxlen = 100
        self._live_odom_maxlen = 100

        # Rolling data buffers for plots. deque(maxlen=...) auto-trims
        # on append — eliminates the per-message O(N) slice-and-reassign
        # that the old list-based buffers paid for every GPS / odom
        # message in live mode.
        self._power_buf = {
            't': deque(),
            'V': deque(),
            'I': deque(),
            'P': deque(),
        }
        self._gps_buf = {
            'lat': deque(maxlen=self._live_gps_maxlen),
            'lon': deque(maxlen=self._live_gps_maxlen),
        }
        self._odom_buf = {
            'x':     deque(maxlen=self._live_odom_maxlen),
            'y':     deque(maxlen=self._live_odom_maxlen),
            'theta': deque(maxlen=self._live_odom_maxlen),
        }

        # ETA estimation state (mirrors ina226_monitor approach)
        self._CAPACITY_AH = 20.476       # empirical usable capacity
        self._ETA_ALPHA = 0.05           # EMA smoothing factor
        self._ema_eta_hours = None       # smoothed time-remaining estimate
        self._latest_soc_pct = None      # latest SOC from electrical publisher (0-100)
        self._odom_tri_patch = None
        # Global costmap snippet overlay (Tier 3 of the odom display).
        # Created lazily by _redraw_plots() the first time a fresh
        # cropped snippet is ready; updated in-place after that via
        # set_data/set_extent. zorder=2 puts it under the NAV2 plan
        # (4) the trail (3 — wait, see comment below) and triangle (5)
        # so the robot stays visible.
        self._odom_costmap_img = None
        # NAV2 global plan as a green Line2D on the odom canvas.
        # zorder=4 — above the costmap (2) and trail (3), below the
        # red triangle (5) so the operator can see the planner's
        # intent without it eclipsing the robot.
        self._odom_plan_line = None
        self._plots_dirty = False

        # EKF participation pulse state. Each device dot whose raw
        # input AND the EKF that consumes it are both fresh gets
        # painted with a hue cycling between green (135°) and
        # purple (270°) — the visible "this device is currently
        # being fused" indicator. Phase advances on every live
        # tick (5 Hz); a 0.5 Hz cycle gives 10 frames per period
        # which reads as a smooth breathing pulse.
        self._ekf_pulse_phase = 0.0
        self._ekf_pulse_freq_hz = 0.5
        # Hue endpoints: green (#33ff66 ~ HSL 135) ↔ purple
        # (#aa55ff ~ HSL 270). Sweep along the +135° arc.
        self._ekf_pulse_hue_lo = 135
        self._ekf_pulse_hue_hi = 270
        self._ekf_msg_age_max_s = 1.0
        # Dot-name → (input_msg_key, ekf_output_key). A dot pulses
        # iff last_msg_t[input_msg_key] AND last_msg_t[ekf_output_key]
        # are both within max_age — i.e. the input is being produced
        # AND the EKF that consumes it is publishing fused output.
        # Mapping rationale:
        #   * Encoders → /odom feeds the Local EKF (ekf_local.yaml
        #     odom0). Pulses while /local_ekf/odom is alive.
        #   * Lidar    → /sick_scansegment_xd/imu (SICK lidar's onboard IMU)
        #     feeds the Local EKF (ekf_local.yaml imu0). The
        #     /scan_fullframe stream feeds slam_toolbox separately
        #     (map↔odom correction) — only the IMU is actually
        #     fused into the Local EKF, so stamp here is the IMU's.
        #   * GPS      → /gps_fix feeds gps_handler_node's
        #     magnetometer-less θ EKF, which depends on
        #     /local_ekf/odom for its predict heartbeat.
        #   * SLAM     → /pose is slam_toolbox's localized pose,
        #     consumed by the Map EKF (ekf_global.yaml pose0).
        #     Pulses while /global_ekf/odom is alive.
        #   * Camera   → /zed/zed_node/imu/data is NOT yet wired
        #     into either EKF (see ekf_local.yaml comment about the
        #     missing TF bridge). Until that lands, Camera has no
        #     entry here and its dot keeps the existing static
        #     style.
        self._ekf_pulse_devices = {
            'Encoders': ('odom', 'ekf_local_odom'),
            'Lidar':    ('imu',  'ekf_local_odom'),
            'GPS':      ('gps',  'ekf_local_odom'),
            'SLAM':     ('pose', 'ekf_global_odom'),
        }
        # Per-device override of the "input is still considered
        # fresh" window. The default ``_ekf_msg_age_max_s`` (1 s) is
        # right for high-rate streams like /odom and the SICK IMU —
        # losing freshness within a second means the source genuinely
        # stopped. /gps_fix publishes at 0.5–10 Hz, so a 1 s window
        # repeatedly times out between fixes and the GPS dot snaps
        # in and out of the pulse instead of breathing smoothly. A
        # 5 s window keeps the dot pulsing across the slow intervals
        # while still going stale promptly if GPS actually dies.
        self._ekf_pulse_input_max_age_s = {
            'GPS': 5.0,
        }
        # EKF status rows (the "is this filter publishing?" indicators
        # added to the device list). Maps dot name → ekf-output
        # last_msg_t key. _ekf_pulse_tick drives these solid green
        # while the topic is fresh and gray when stale.
        self._ekf_status_rows = {
            'Local EKF': 'ekf_local_odom',
            'Map EKF':   'ekf_global_odom',
        }

        # Screen lock state. When _screen_locked is True an opaque
        # overlay covers the central widget and an application-wide
        # event filter (installed at end of __init__) intercepts every
        # KeyPress / mouse event so the operator can neither click
        # buttons nor type into focused widgets. Ctrl+Shift+L toggles
        # the lock; while locked, Ctrl+Shift+L reveals the password
        # field, and the field unlocks on a correct password + Enter.
        self._screen_locked = False
        self._lock_password = "@cro123"
        self._lock_overlay = None
        # Test-recording overlay (REC mode). Same window-fill /
        # mouse-pass-through pattern as the lock overlay, but driven
        # by /data/toggle_collect instead of a hotkey. The device dots
        # also color-ramp black↔red at 2 Hz so the operator can see
        # both peripherally (overlay) and dot-by-dot (which topics
        # are actually in the bag).
        self._rec_overlay = None
        self._rec_active = False
        self._rec_phase_t0 = None
        # Snapshot of every status_dots stylesheet taken at the moment
        # REC turns on, so we can restore the pre-REC look when it
        # turns off (the participation-pulse code reasserts its own
        # styling on the next tick either way; the snapshot handles
        # the gap).
        self._dot_styles_before_rec = None
        self._lock_password_input = None
        self._lock_password_visible = False
        self._lock_hint_label = None
        self._lock_status_label = None

        # Performance Mode — opt-in low-CPU veil. When on, every QTimer
        # the window owns is paused (live tick, playback, EKF pulse,
        # process polling, etc.) so the GUI hovers at near-zero CPU
        # behind a translucent blue panel with a giant centred
        # "PERFORMANCE" label. _perf_resume_timers stores the timers
        # we stopped so toggle-off resumes exactly those that were
        # running on entry — anything started while in performance
        # mode is left alone.

        self._performance_mode = False
        self._perf_resume_timers = []
        # Unified mode overlay — built lazily; carries both PERF + REC
        # veils with a 60° diagonal split. self._perf_stats_label is
        # aliased to the overlay's stats QLabel after construction so
        # the existing CPU/GPU update path keeps working unchanged.
        self._mode_overlay = None
        self._perf_stats_label = None
        # 5 s sampler runs CONTINUOUSLY (not just in Perf mode) so the
        # overlay can show a Before/Now comparison — Before being the
        # most recent sample taken before the operator entered Perf
        # mode, Now being the latest sample taken while paused. The
        # sampler is also the one timer skipped by the pause loop.
        self._perf_stats_timer = QTimer(self)
        self._perf_stats_timer.setInterval(5000)
        self._perf_stats_timer.timeout.connect(self._sample_perf_stats)
        self._perf_proc = None
        self._perf_cpu_count = 1
        if _HAS_PSUTIL:
            try:
                self._perf_proc = psutil.Process()
                # Prime per-process cpu_percent so the first reading
                # reflects real work, not the all-time-since-start
                # fallback. (System cpu_percent doesn't need priming
                # because we're not reading it any more.)
                self._perf_proc.cpu_percent(interval=None)
                self._perf_cpu_count = max(1, psutil.cpu_count(logical=True))
            except Exception:
                self._perf_proc = None
        self._perf_nvml_handle = None  # populated lazily on first sample
        # Latest samples — refreshed every 5 s by _sample_perf_stats.
        self._last_gui_cpu_pct = None
        self._last_sys_gpu_pct = None
        # Baseline — snapshotted at the moment Perf mode is entered.
        self._baseline_gui_cpu_pct = None
        self._baseline_sys_gpu_pct = None
        # Start the sampler. QTimer is queued to fire from the event
        # loop, so this is safe even though __init__ hasn't returned.
        self._perf_stats_timer.start()

        # Container connection state
        self._container_connected = False
        self._container_name = 'koopa-kingdom'
        self._container_workdir = '/autonav/isaac_ros-dev'
        self._container_user = 'admin'

        # Video playback state (camera + lidar mp4s)
        self._camera_cap = None   # cv2.VideoCapture
        self._lidar_cap = None    # cv2.VideoCapture
        self._cam_im = None       # matplotlib imshow handle
        # Source tag for the current _cam_im so we know whether it's
        # showing 3-channel RGB (raw / overlay) or 1-channel grayscale
        # (mask). Used by _on_camera_frame_ready to decide whether to
        # recreate the AxesImage with a different colormap.
        self._cam_im_source = None
        self._lidar_im = None     # matplotlib imshow handle
        # Camera/lidar overlays: scatter for detected line pixels and a
        # patches.Ellipse for the GPS covariance — created lazily on
        # first data, hidden when the source topic goes stale.
        self._cam_lines_scatter = None
        self._gps_cov_ellipse = None
        # Freshness window for "is this overlay topic live right now?"
        # The camera/lidar/GPS overlays clear themselves once the source
        # has been silent for longer than this, so an inactive detector
        # leaves the raw image / plain BEV / plain map untouched.
        self._overlay_fresh_s = 1.0
        self._video_fps = 30      # must match recorder's VIDEO_FPS

        # --- Central widget + top-level 3-column layout ---
        central = QWidget()
        self.setCentralWidget(central)
        top_layout = QHBoxLayout(central)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.setSpacing(4)

        # Shared helpers
        section_title_font = QFont()
        section_title_font.setPointSize(14)
        section_title_font.setBold(True)

        button_style = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 4px;"
            "  padding: 10px; font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

        frame_style = (
            "QFrame#sensorCell {"
            "  border: 1px solid #444;"
            "  background-color: #1e1e1e;"
            "  border-radius: 3px;"
            "}"
        )

        val_label_style = (
            "border: none; color: #0f0; font-size: 11px;"
            " font-family: monospace;"
        )

        section_title_style = "color: #ffffff; border: none;"
        group_label_style = (
            "border: none; font-size: 9px; color: #666;"
            " font-weight: bold; text-transform: uppercase;"
        )

        # =====================================================================
        # SECTION 1: OPTIONS (left column) – uses QStackedWidget for sub-pages
        # =====================================================================
        options_frame = QFrame()
        options_frame.setStyleSheet("QFrame { border: 1px solid #444; }")
        options_outer = QVBoxLayout(options_frame)
        options_outer.setContentsMargins(10, 6, 10, 10)

        header_font = QFont()
        header_font.setPointSize(20)
        header_font.setBold(True)
        lbl_header = QLabel("AutoNav GUI")
        lbl_header.setFont(header_font)
        lbl_header.setAlignment(Qt.AlignCenter)
        lbl_header.setStyleSheet("color: #ffffff; border: none;")
        options_outer.addWidget(lbl_header)

        self._options_stack = QStackedWidget()
        options_outer.addWidget(self._options_stack, stretch=1)

        # --- Page 0: Main OPTIONS ---
        page_main = QWidget()
        options_layout = QVBoxLayout(page_main)
        options_layout.setContentsMargins(6, 0, 6, 0)

        lbl_options = QLabel("OPTIONS")
        lbl_options.setFont(section_title_font)
        lbl_options.setAlignment(Qt.AlignCenter)
        lbl_options.setStyleSheet(section_title_style)
        options_layout.addWidget(lbl_options)

        self._nav_buttons = []  # (QPushButton, base_label, base_style)

        disabled_btn_style = (
            "QPushButton {"
            "  background-color: #1a1a1a; color: #555;"
            "  border: 1px solid #333; border-radius: 4px;"
            "  padding: 10px; font-size: 13px;"
            "}"
        )
        self._disabled_btn_style = disabled_btn_style

        connect_style = (
            button_style.replace("#2a2a2a", "#1a2a3a")
                        .replace("#3a3a3a", "#2a3a4a")
                        .replace("#1a1a1a", "#0a1a2a")
        )
        self.btn_connect = QPushButton("Connect to Container")
        self.btn_connect.setStyleSheet(connect_style)
        self.btn_connect.setFocusPolicy(Qt.NoFocus)
        self.btn_connect.clicked.connect(self._on_connect_container)
        options_layout.addWidget(self.btn_connect)
        self._nav_buttons.append((self.btn_connect, "Connect to Container", connect_style))
        self._connect_style = connect_style

        # -- Container Dependent Actions --
        lbl_dep = QLabel("Container Dependent")
        lbl_dep.setAlignment(Qt.AlignCenter)
        lbl_dep.setStyleSheet(group_label_style)
        options_layout.addWidget(lbl_dep)

        self._container_dots = []

        def _make_container_btn_row(label, style, click_handler):
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                "background-color: #f44; border-radius: 5px; border: none;"
            )
            btn = QPushButton(label)
            btn.setStyleSheet(style)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setEnabled(False)
            btn.clicked.connect(click_handler)
            row.addWidget(dot)
            row.addWidget(btn, stretch=1)
            self._container_dots.append(dot)
            return row, btn

        row_launch, self.btn_launch = _make_container_btn_row(
            "Launch/End Processes", disabled_btn_style, self._show_launch_page)
        options_layout.addLayout(row_launch)
        self._nav_buttons.append((self.btn_launch, "Launch/End Processes", button_style))

        row_live, self.btn_live = _make_container_btn_row(
            "Live Mode", disabled_btn_style, self._on_live_clicked)
        options_layout.addLayout(row_live)
        self._nav_buttons.append((self.btn_live, "Live Mode", button_style))

        row_test, self.btn_test = _make_container_btn_row(
            "Test Mode", disabled_btn_style, self._on_test_clicked)
        options_layout.addLayout(row_test)
        self._nav_buttons.append((self.btn_test, "Test Mode", button_style))

        row_build, self.btn_build = _make_container_btn_row(
            "Build Workspace", disabled_btn_style, self._on_build_clicked)
        options_layout.addLayout(row_build)
        self._nav_buttons.append((self.btn_build, "Build Workspace", button_style))

        # Map of container-gated buttons to their base labels
        self._container_buttons = {
            self.btn_launch: "Launch/End Processes",
            self.btn_live: "Live Mode",
            self.btn_test: "Test Mode",
            self.btn_build: "Build Workspace",
        }

        # -- Container Independent Actions --
        lbl_indep = QLabel("Container Independent")
        lbl_indep.setAlignment(Qt.AlignCenter)
        lbl_indep.setStyleSheet(group_label_style)
        options_layout.addWidget(lbl_indep)

        self.btn_developer = QPushButton("Developer")
        self.btn_developer.setStyleSheet(button_style)
        self.btn_developer.setFocusPolicy(Qt.NoFocus)
        self.btn_developer.clicked.connect(self._show_developer_page)
        options_layout.addWidget(self.btn_developer)
        self._nav_buttons.append((self.btn_developer, "Developer", button_style))

        self.btn_playback = QPushButton("Playback Mode")
        self.btn_playback.setStyleSheet(button_style)
        self.btn_playback.setFocusPolicy(Qt.NoFocus)
        self.btn_playback.clicked.connect(self._on_playback_clicked)
        options_layout.addWidget(self.btn_playback)
        self._nav_buttons.append((self.btn_playback, "Playback Mode", button_style))

        # ROS bag playback state — populated by the Playback Mode
        # page's row scanner, driven by `ros2 bag play` subprocesses.
        # The poll timer detects natural end-of-bag and clears the
        # row so the operator doesn't have to.
        self._bag_play_proc = None
        self._bag_play_poll_timer = QTimer(self)
        self._bag_play_poll_timer.setInterval(500)
        self._bag_play_poll_timer.timeout.connect(self._bag_play_poll)

        # Performance Mode — last entry in the action stack. Toggles a
        # near-zero-CPU veil that pauses every QTimer the window owns.
        # The button + veil are container-independent (no ROS needed),
        # so it sits alongside Playback Mode.
        self._perf_off_style = button_style
        self._perf_on_style = (
            button_style.replace("#2a2a2a", "#1a3a6a")
                        .replace("#3a3a3a", "#2a4a7a")
                        .replace("#1a1a1a", "#0a2a4a")
        )
        self.btn_performance = QPushButton("Performance Mode")
        self.btn_performance.setStyleSheet(self._perf_off_style)
        self.btn_performance.setFocusPolicy(Qt.NoFocus)
        self.btn_performance.clicked.connect(self._on_performance_clicked)
        options_layout.addWidget(self.btn_performance)
        self._nav_buttons.append(
            (self.btn_performance, "Performance Mode", self._perf_off_style)
        )

        options_layout.addStretch()

        # Theme toggle (label reflects what clicking will do, not current).
        self.btn_theme = QPushButton("Switch to Dark Mode")
        self.btn_theme.setStyleSheet(button_style)
        self.btn_theme.setFocusPolicy(Qt.NoFocus)
        self.btn_theme.clicked.connect(self._toggle_theme)
        options_layout.addWidget(self.btn_theme)
        self._nav_buttons.append((self.btn_theme, "Switch to Dark Mode", button_style))

        quit_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_quit = QPushButton("Quit GUI")
        btn_quit.setStyleSheet(quit_style)
        btn_quit.setFocusPolicy(Qt.NoFocus)
        # self.close() triggers closeEvent → _kill_process for every
        # running launch. QApplication.quit() bypasses closeEvent, which
        # left in-container ros2 stacks orphaned and accumulating as
        # zombies across GUI restarts.
        btn_quit.clicked.connect(self.close)
        options_layout.addWidget(btn_quit)
        self._nav_buttons.append((btn_quit, "Quit GUI", quit_style))

        self._options_stack.addWidget(page_main)  # index 0

        # --- Page 1: Launch/End Processes ---
        page_launch = QWidget()
        launch_layout = QVBoxLayout(page_launch)
        launch_layout.setContentsMargins(6, 0, 6, 0)

        lbl_launch = QLabel("LAUNCH / END PROCESSES")
        lbl_launch.setFont(section_title_font)
        lbl_launch.setAlignment(Qt.AlignCenter)
        lbl_launch.setStyleSheet(section_title_style)
        launch_layout.addWidget(lbl_launch)

        # Queue status label
        self._queue_label = QLabel("Queue: idle")
        self._queue_label.setStyleSheet(
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )
        self._queue_label.setAlignment(Qt.AlignCenter)
        self._queue_label.setWordWrap(True)
        launch_layout.addWidget(self._queue_label)

        # Toggle style: green border when active
        toggle_on_style = (
            button_style.replace("border: 1px solid #555", "border: 1px solid #0f0")
                        .replace("color: #dcdcdc", "color: #0f0")
        )
        self._toggle_on_style = toggle_on_style

        # Device definitions: (button label, status_dot key(s), real command)
        # Every command goes through a script that prints "[GUI_READY] <label>"
        # once its readiness condition (a topic publishing) is met. The HUD
        # blocks the queue on that sentinel — see READY_SENTINEL below.
        self._launch_devices = [
            ("Pre-SLAM", ["Encoders", "CONTROL"],
             "./config/run-pre-slam.sh"),
            ("Camera", ["Camera"], "./config/run-zed.sh"),
            ("Lidar", ["Lidar"], "./config/run-lidar.sh"),
            ("GPS", ["GPS"], "./config/run-gps.sh"),
            # PCA DETECT must come up before SLAM. slam_toolbox is
            # configured to subscribe to /scan_pca_filtered (the
            # grade detector's obstacle cloud collapsed to 2D via
            # pca_pc2_to_scan in slam.launch.py). If SLAM starts
            # first, slam_toolbox starves of scans, never produces
            # /pose or the map→odom TF, and the Nav2 lifecycle
            # stalls at "Activating planner_server."
            # LINE DETECT also goes before SLAM so the line-pixel
            # stream is producing by the time the nav2 / line_layer
            # plugins come up — same reasoning as PCA above.
            ("PCA DETECT", ["PCA DETECT"], "./config/run-pca.sh"),
            ("LINE DETECT", ["LINE DETECT"], "./config/run-lines.sh"),
            ("SLAM", ["SLAM"], "ros2 launch slam slam.launch.py"),
            ("NAV2", ["NAV2"], "./config/run-nav2.sh"),
            ("Power PCB", ["Power PCB"], "./config/run-electrical.sh"),
        ]

        self._launch_nav_buttons = []  # same tuple format as _nav_buttons
        self._launch_states = {}   # label -> False | 'starting' | True
        self._flash_timers = {}    # label -> QTimer (flashing animation)
        self._startup_timers = {}  # label -> QTimer (readiness poll)
        self._launch_queue = []    # list of labels waiting to start
        self._ready_events = {}    # label -> bool (set when [GUI_READY] seen on stdout)
        self._startup_deadlines = {}  # label -> monotonic seconds; readiness must arrive by then

        # Scripts/launches that opt into the readiness handshake print this
        # token on stdout when they reach steady state. The reader thread
        # flips _ready_events[label] true; _check_startup waits on that.
        self.READY_SENTINEL = "[GUI_READY]"

        # Per-device readiness timeout (seconds). Devices not listed here
        # use DEFAULT_READY_TIMEOUT. If the sentinel never arrives within
        # the window, the device is marked failed and the queue advances.
        self.DEFAULT_READY_TIMEOUT = 60.0
        self._ready_timeouts = {
            "Pre-SLAM":  60.0,
            "Camera":    45.0,
            "Lidar":     45.0,
            "SLAM":      120.0,  # waits for /scan_fullframe + first /map_padded
            "NAV2":      90.0,
            "GPS":       300.0,  # outdoor GPS lock can take minutes
            "Power PCB": 30.0,
            "LINE DETECT": 45.0,
            "PCA DETECT": 45.0,
        }

        cmd_label_style = (
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )

        # Compact button style for launch grid (narrower in X)
        launch_btn_compact = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 4px;"
            "  padding: 10px 4px; font-size: 11px;"
            "}"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

        # Two-column grid: buttons left, commands right
        launch_grid = QGridLayout()
        launch_grid.setSpacing(4)
        for i, (label, _dot_keys, cmd) in enumerate(self._launch_devices):
            btn = QPushButton(label)
            btn.setStyleSheet(launch_btn_compact)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setFixedWidth(110)
            btn.clicked.connect(lambda checked=False, l=label: self._toggle_device(l))
            launch_grid.addWidget(btn, i, 0)
            self._launch_nav_buttons.append((btn, label, launch_btn_compact))
            self._launch_states[label] = False

            cmd_lbl = QLabel(cmd)
            cmd_lbl.setStyleSheet(cmd_label_style)
            cmd_lbl.setWordWrap(True)
            launch_grid.addWidget(cmd_lbl, i, 1)

        launch_grid.setColumnStretch(0, 0)
        launch_grid.setColumnStretch(1, 1)
        launch_layout.addLayout(launch_grid)

        # Separator bar
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #444; border: none; max-height: 1px;")
        sep.setFixedHeight(1)
        launch_layout.addWidget(sep)

        # "Launch All in Sequence" button
        launch_all_style = (
            button_style.replace("border: 1px solid #555", "border: 1px solid #0af")
                        .replace("color: #dcdcdc", "color: #0af")
        )
        btn_launch_all = QPushButton("Launch All in Sequence")
        btn_launch_all.setStyleSheet(launch_all_style)
        btn_launch_all.setFocusPolicy(Qt.NoFocus)
        btn_launch_all.clicked.connect(self._launch_all_in_sequence)
        launch_layout.addWidget(btn_launch_all)
        self._launch_nav_buttons.append(
            (btn_launch_all, "Launch All in Sequence", launch_all_style)
        )

        # "Run Mission" toggle — runs ./config/run_mission.sh inside the
        # container; click again to abort.
        mission_off_style = (
            button_style.replace("border: 1px solid #555", "border: 1px solid #fa0")
                        .replace("color: #dcdcdc", "color: #fa0")
        )
        mission_on_style = (
            button_style.replace("border: 1px solid #555", "border: 1px solid #0f0")
                        .replace("color: #dcdcdc", "color: #0f0")
        )
        self._mission_off_style = mission_off_style
        self._mission_on_style = mission_on_style
        self._mission_running = False
        self._mission_proc = None
        self._mission_check_timer = QTimer(self)
        self._mission_check_timer.setInterval(500)
        self._mission_check_timer.timeout.connect(self._check_mission_finished)
        self._mission_btn = QPushButton("Run Mission")
        self._mission_btn.setStyleSheet(mission_off_style)
        self._mission_btn.setFocusPolicy(Qt.NoFocus)
        self._mission_btn.clicked.connect(self._toggle_mission)
        launch_layout.addWidget(self._mission_btn)
        self._launch_nav_buttons.append(
            (self._mission_btn, "Run Mission", mission_off_style)
        )

        # --- One-shot script runners: send_goal.sh / send_GPS_waypoint.sh ---
        # Each row is [label | QLineEdit args | Send button]. The button
        # fires the script (wrapped for the container) with the args
        # string appended; output is captured into _process_buffers
        # under the label so it shows up in the terminal display when
        # the device dot is selected (no dot is wired for these, so the
        # operator selects via the existing dot UI — output is mostly
        # informational anyway).
        sep_send = QFrame()
        sep_send.setFrameShape(QFrame.HLine)
        sep_send.setStyleSheet("background-color: #444; border: none; max-height: 1px;")
        sep_send.setFixedHeight(1)
        launch_layout.addWidget(sep_send)

        send_field_style = (
            "QLineEdit {"
            "  background-color: #1a1a1a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 3px;"
            "  padding: 4px 6px; font-size: 11px; font-family: monospace;"
            "}"
            "QLineEdit:focus { border: 1px solid #0af; }"
        )
        send_btn_style = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #0af;"
            "  border: 1px solid #0af; border-radius: 4px;"
            "  padding: 6px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )

        # Selected-state style for the input fields: green border so
        # the operator can see which row arrow-nav is on. Stored on
        # self because _update_selection reads it.
        self._send_field_base_style = send_field_style
        self._send_field_sel_style = (
            "QLineEdit {"
            "  background-color: #1a2a1a; color: #dcdcdc;"
            "  border: 2px solid #0f0; border-radius: 3px;"
            "  padding: 3px 5px; font-size: 11px; font-family: monospace;"
            "}"
        )
        self._send_btn_base_style = send_btn_style

        send_grid = QGridLayout()
        send_grid.setSpacing(4)

        lbl_goal = QLabel("Send Goal")
        lbl_goal.setStyleSheet("border: none; color: #dcdcdc; font-size: 11px;")
        self._send_goal_input = QLineEdit()
        self._send_goal_input.setStyleSheet(send_field_style)
        btn_send_goal = QPushButton("Send")
        btn_send_goal.setStyleSheet(send_btn_style)
        btn_send_goal.setFocusPolicy(Qt.NoFocus)
        btn_send_goal.setFixedWidth(60)
        btn_send_goal.clicked.connect(self._on_send_goal_clicked)
        self._send_goal_input.returnPressed.connect(self._on_send_goal_clicked)
        send_grid.addWidget(lbl_goal, 0, 0)
        send_grid.addWidget(self._send_goal_input, 0, 1)
        send_grid.addWidget(btn_send_goal, 0, 2)
        # Arrow-key nav participation. Up/Down through these in col 0
        # of the launch-page nav grid. Enter on the QLineEdit focuses
        # it (for typing); Enter on the button submits. See
        # _update_selection / keyPressEvent for the QLineEdit-aware
        # paths.
        self._launch_nav_buttons.append(
            (self._send_goal_input, "Send Goal Args", send_field_style)
        )
        self._launch_nav_buttons.append(
            (btn_send_goal, "Send", send_btn_style)
        )

        lbl_gps = QLabel("Send GPS")
        lbl_gps.setStyleSheet("border: none; color: #dcdcdc; font-size: 11px;")
        self._send_gps_input = QLineEdit()
        self._send_gps_input.setStyleSheet(send_field_style)
        btn_send_gps = QPushButton("Send")
        btn_send_gps.setStyleSheet(send_btn_style)
        btn_send_gps.setFocusPolicy(Qt.NoFocus)
        btn_send_gps.setFixedWidth(60)
        btn_send_gps.clicked.connect(self._on_send_gps_clicked)
        self._send_gps_input.returnPressed.connect(self._on_send_gps_clicked)
        send_grid.addWidget(lbl_gps, 1, 0)
        send_grid.addWidget(self._send_gps_input, 1, 1)
        send_grid.addWidget(btn_send_gps, 1, 2)
        self._launch_nav_buttons.append(
            (self._send_gps_input, "Send GPS Args", send_field_style)
        )
        self._launch_nav_buttons.append(
            (btn_send_gps, "Send", send_btn_style)
        )

        send_grid.setColumnStretch(0, 0)
        send_grid.setColumnStretch(1, 1)
        send_grid.setColumnStretch(2, 0)
        launch_layout.addLayout(send_grid)

        launch_layout.addStretch()

        # "Exit Launch/End Processes" button at the bottom
        exit_launch_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_exit_launch = QPushButton("Exit Launch/End Processes")
        btn_exit_launch.setStyleSheet(exit_launch_style)
        btn_exit_launch.setFocusPolicy(Qt.NoFocus)
        btn_exit_launch.clicked.connect(self._show_main_page)
        launch_layout.addWidget(btn_exit_launch)
        self._launch_nav_buttons.append(
            (btn_exit_launch, "Exit Launch/End Processes", exit_launch_style)
        )

        self._options_stack.addWidget(page_launch)  # index 1

        # Store styles for toggle (use compact style for launch grid buttons)
        self._launch_btn_style = launch_btn_compact
        self._launch_on_style = (
            launch_btn_compact
            .replace("border: 1px solid #555", "border: 1px solid #0f0")
            .replace("color: #dcdcdc", "color: #0f0")
        )
        self._launch_wait_style = (
            launch_btn_compact
            .replace("border: 1px solid #555", "border: 1px solid #ff0")
            .replace("color: #dcdcdc", "color: #ff0")
        )

        # --- Page 2: Playback CSV selection ---
        page_playback = QWidget()
        playback_layout = QVBoxLayout(page_playback)
        playback_layout.setContentsMargins(6, 0, 6, 0)

        lbl_pb = QLabel("PLAYBACK")
        lbl_pb.setFont(section_title_font)
        lbl_pb.setAlignment(Qt.AlignCenter)
        lbl_pb.setStyleSheet(section_title_style)
        playback_layout.addWidget(lbl_pb)

        csv_label_style = (
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )
        self._playback_nav_buttons = []
        self._csv_grid = QGridLayout()
        self._csv_grid.setSpacing(4)
        self._csv_grid.setColumnStretch(0, 0)
        self._csv_grid.setColumnStretch(1, 1)
        playback_layout.addLayout(self._csv_grid)
        self._scan_csv_files(button_style, csv_label_style)

        playback_layout.addStretch()

        exit_pb_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_exit_pb = QPushButton("Exit Playback")
        btn_exit_pb.setStyleSheet(exit_pb_style)
        btn_exit_pb.setFocusPolicy(Qt.NoFocus)
        btn_exit_pb.clicked.connect(self._show_main_page)
        playback_layout.addWidget(btn_exit_pb)
        self._playback_nav_buttons.append(
            (btn_exit_pb, "Exit Playback", exit_pb_style)
        )

        self._options_stack.addWidget(page_playback)  # index 2
        self._pb_button_style = button_style
        self._pb_csv_label_style = csv_label_style

        # --- Page 3: Test Mode ---
        page_test = QWidget()
        test_layout = QVBoxLayout(page_test)
        test_layout.setContentsMargins(6, 0, 6, 0)

        lbl_test = QLabel("TEST MODE")
        lbl_test.setFont(section_title_font)
        lbl_test.setAlignment(Qt.AlignCenter)
        lbl_test.setStyleSheet(section_title_style)
        test_layout.addWidget(lbl_test)

        self._test_status_label = QLabel("Status: idle")
        self._test_status_label.setStyleSheet(
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )
        self._test_status_label.setAlignment(Qt.AlignCenter)
        self._test_status_label.setWordWrap(True)
        test_layout.addWidget(self._test_status_label)

        # Test definitions: (id, title, launch command, description)
        self._test_defs = [
            ("t000", "DAQ Mode",
             "ros2 launch autonav_automated_testing t000_DAQ_MODE.launch.py",
             "Data acquisition — operator drives, system logs all sensor data"),
            ("t002", "Line Compliance",
             "ros2 launch autonav_automated_testing t002_Line_Comp.launch.py",
             "Autonomous line-following — robot drives 110 ft along white lines"),
        ]

        self._test_nav_buttons = []
        self._active_test = None  # id of currently running test

        test_btn_style = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 4px;"
            "  padding: 10px 4px; font-size: 11px;"
            "}"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )
        self._test_btn_style = test_btn_style
        self._test_on_style = (
            test_btn_style
            .replace("border: 1px solid #555", "border: 1px solid #0f0")
            .replace("color: #dcdcdc", "color: #0f0")
        )

        test_desc_style = (
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )

        test_grid = QGridLayout()
        test_grid.setSpacing(4)
        for i, (tid, title, cmd, desc) in enumerate(self._test_defs):
            btn = QPushButton(f"{tid}: {title}")
            btn.setStyleSheet(test_btn_style)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setFixedWidth(140)
            btn.clicked.connect(
                lambda checked=False, t=tid: self._toggle_test(t)
            )
            test_grid.addWidget(btn, i, 0)
            self._test_nav_buttons.append((btn, f"{tid}: {title}", test_btn_style))

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(test_desc_style)
            desc_lbl.setWordWrap(True)
            test_grid.addWidget(desc_lbl, i, 1)

        test_grid.setColumnStretch(0, 0)
        test_grid.setColumnStretch(1, 1)
        test_layout.addLayout(test_grid)

        # Separator
        sep_test = QFrame()
        sep_test.setFrameShape(QFrame.HLine)
        sep_test.setStyleSheet("background-color: #444; border: none; max-height: 1px;")
        sep_test.setFixedHeight(1)
        test_layout.addWidget(sep_test)

        # E-STOP button
        estop_style = (
            "QPushButton {"
            "  background-color: #8b0000; color: #fff;"
            "  border: 2px solid #f00; border-radius: 4px;"
            "  padding: 12px 4px; font-size: 13px; font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #a00000; }"
            "QPushButton:pressed { background-color: #600000; }"
        )
        btn_estop = QPushButton("E-STOP")
        btn_estop.setStyleSheet(estop_style)
        btn_estop.setFocusPolicy(Qt.NoFocus)
        btn_estop.clicked.connect(self._on_estop)
        test_layout.addWidget(btn_estop)
        self._test_nav_buttons.append((btn_estop, "E-STOP", estop_style))

        test_layout.addStretch()

        # Exit Test Mode button
        exit_test_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_exit_test = QPushButton("Exit Test Mode")
        btn_exit_test.setStyleSheet(exit_test_style)
        btn_exit_test.setFocusPolicy(Qt.NoFocus)
        btn_exit_test.clicked.connect(self._show_main_page)
        test_layout.addWidget(btn_exit_test)
        self._test_nav_buttons.append(
            (btn_exit_test, "Exit Test Mode", exit_test_style)
        )

        self._options_stack.addWidget(page_test)  # index 3

        # --- Page 4: Developer (git + container lifecycle) ---
        # GUI runs natively on the Jetson; all commands here run on the host.
        self._dev_host_repo = os.path.expanduser('~/AutoNav_25-26')
        self._dev_run_script = os.path.join(
            self._dev_host_repo, 'env/docker/run-container.sh'
        )
        self._dev_container_running = False
        self._dev_branch_rows = []  # (QPushButton, QLabel) per branch

        page_dev = QWidget()
        dev_layout = QVBoxLayout(page_dev)
        dev_layout.setContentsMargins(6, 0, 6, 0)

        lbl_dev = QLabel("DEVELOPER")
        lbl_dev.setFont(section_title_font)
        lbl_dev.setAlignment(Qt.AlignCenter)
        lbl_dev.setStyleSheet(section_title_style)
        dev_layout.addWidget(lbl_dev)

        # Current branch header
        self._dev_branch_label = QLabel("Current branch: …")
        self._dev_branch_label.setAlignment(Qt.AlignCenter)
        self._dev_branch_label.setStyleSheet(
            "border: none; color: #0af; font-size: 11px;"
            " font-family: monospace;"
        )
        dev_layout.addWidget(self._dev_branch_label)

        # Status / output line (errors, pull results, etc.)
        self._dev_status_label = QLabel("")
        self._dev_status_label.setAlignment(Qt.AlignCenter)
        self._dev_status_label.setWordWrap(True)
        self._dev_status_label.setStyleSheet(
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )
        dev_layout.addWidget(self._dev_status_label)

        self._dev_nav_buttons = []  # same tuple format as other pages

        dev_btn_compact = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 4px;"
            "  padding: 8px 4px; font-size: 11px;"
            "}"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:pressed { background-color: #1a1a1a; }"
        )
        self._dev_btn_style = dev_btn_compact

        # Container start/stop section
        lbl_container = QLabel("CONTAINER")
        lbl_container.setAlignment(Qt.AlignCenter)
        lbl_container.setStyleSheet(group_label_style)
        dev_layout.addWidget(lbl_container)

        container_row = QHBoxLayout()
        container_row.setSpacing(6)
        self._dev_container_dot = QLabel()
        self._dev_container_dot.setFixedSize(10, 10)
        self._dev_container_dot.setStyleSheet(
            "background-color: #f44; border-radius: 5px; border: none;"
        )
        self._dev_container_btn = QPushButton("Start Container")
        self._dev_container_btn.setStyleSheet(dev_btn_compact)
        self._dev_container_btn.setFocusPolicy(Qt.NoFocus)
        self._dev_container_btn.clicked.connect(self._dev_toggle_container)
        container_row.addWidget(self._dev_container_dot)
        container_row.addWidget(self._dev_container_btn, stretch=1)
        dev_layout.addLayout(container_row)
        self._dev_nav_buttons.append(
            (self._dev_container_btn, "Container", dev_btn_compact)
        )

        # Separator
        sep_a = QFrame()
        sep_a.setFrameShape(QFrame.HLine)
        sep_a.setStyleSheet("background-color: #444; border: none; max-height: 1px;")
        sep_a.setFixedHeight(1)
        dev_layout.addWidget(sep_a)

        # One-shot: stop container → pull current branch → start container
        # → connect → colcon build. Long-running (minutes); runs in a
        # worker thread and streams status into _dev_set_status. The
        # button disables itself while the chain is in flight so a
        # double-click can't kick off two builds in parallel.
        lbl_quick = QLabel("QUICK ACTIONS")
        lbl_quick.setAlignment(Qt.AlignCenter)
        lbl_quick.setStyleSheet(group_label_style)
        dev_layout.addWidget(lbl_quick)

        self._dev_full_rebuild_btn = QPushButton("Load branch changes and build")
        self._dev_full_rebuild_btn.setStyleSheet(dev_btn_compact)
        self._dev_full_rebuild_btn.setFocusPolicy(Qt.NoFocus)
        self._dev_full_rebuild_btn.clicked.connect(self._dev_full_rebuild)
        dev_layout.addWidget(self._dev_full_rebuild_btn)
        self._dev_nav_buttons.append(
            (self._dev_full_rebuild_btn, "Load branch changes and build", dev_btn_compact)
        )
        self._dev_full_rebuild_running = False

        # Separator
        sep_b = QFrame()
        sep_b.setFrameShape(QFrame.HLine)
        sep_b.setStyleSheet("background-color: #444; border: none; max-height: 1px;")
        sep_b.setFixedHeight(1)
        dev_layout.addWidget(sep_b)

        # Git pull
        lbl_git = QLabel("GIT")
        lbl_git.setAlignment(Qt.AlignCenter)
        lbl_git.setStyleSheet(group_label_style)
        dev_layout.addWidget(lbl_git)

        self._dev_pull_btn = QPushButton("git pull")
        self._dev_pull_btn.setStyleSheet(dev_btn_compact)
        self._dev_pull_btn.setFocusPolicy(Qt.NoFocus)
        self._dev_pull_btn.clicked.connect(self._dev_git_pull)
        dev_layout.addWidget(self._dev_pull_btn)
        self._dev_nav_buttons.append(
            (self._dev_pull_btn, "git pull", dev_btn_compact)
        )

        # Branch switching is on its own sub-page (one click deeper) so
        # accidentally tapping near the dev page does not switch branches.
        self._dev_branches_btn = QPushButton("Switch Branch…")
        self._dev_branches_btn.setStyleSheet(dev_btn_compact)
        self._dev_branches_btn.setFocusPolicy(Qt.NoFocus)
        self._dev_branches_btn.clicked.connect(self._show_branches_page)
        dev_layout.addWidget(self._dev_branches_btn)
        self._dev_nav_buttons.append(
            (self._dev_branches_btn, "Switch Branch…", dev_btn_compact)
        )

        self._dev_processes_btn = QPushButton("Manage Running Processes")
        self._dev_processes_btn.setStyleSheet(dev_btn_compact)
        self._dev_processes_btn.setFocusPolicy(Qt.NoFocus)
        self._dev_processes_btn.clicked.connect(self._show_processes_page)
        dev_layout.addWidget(self._dev_processes_btn)
        self._dev_nav_buttons.append(
            (self._dev_processes_btn, "Manage Running Processes", dev_btn_compact)
        )

        dev_layout.addStretch()

        # Exit Developer button at the bottom
        exit_dev_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_exit_dev = QPushButton("Exit Developer")
        btn_exit_dev.setStyleSheet(exit_dev_style)
        btn_exit_dev.setFocusPolicy(Qt.NoFocus)
        btn_exit_dev.clicked.connect(self._show_main_page)
        dev_layout.addWidget(btn_exit_dev)
        self._dev_nav_buttons.append(
            (btn_exit_dev, "Exit Developer", exit_dev_style)
        )

        self._options_stack.addWidget(page_dev)  # index 4

        # --- Page 5: Branch switcher (sub-page of Developer) ---
        page_branches = QWidget()
        branches_layout = QVBoxLayout(page_branches)
        branches_layout.setContentsMargins(6, 0, 6, 0)

        lbl_br_title = QLabel("SWITCH BRANCH")
        lbl_br_title.setFont(section_title_font)
        lbl_br_title.setAlignment(Qt.AlignCenter)
        lbl_br_title.setStyleSheet(section_title_style)
        branches_layout.addWidget(lbl_br_title)

        # Current branch + status (shared with dev page wiring)
        self._br_branch_label = QLabel("Current branch: …")
        self._br_branch_label.setAlignment(Qt.AlignCenter)
        self._br_branch_label.setStyleSheet(
            "border: none; color: #0af; font-size: 11px;"
            " font-family: monospace;"
        )
        branches_layout.addWidget(self._br_branch_label)

        self._br_status_label = QLabel("")
        self._br_status_label.setAlignment(Qt.AlignCenter)
        self._br_status_label.setWordWrap(True)
        self._br_status_label.setStyleSheet(
            "border: none; color: #888; font-size: 10px;"
            " font-family: monospace;"
        )
        branches_layout.addWidget(self._br_status_label)

        # Refresh button (re-fetches and rebuilds branch list)
        self._br_refresh_btn = QPushButton("Refresh")
        self._br_refresh_btn.setStyleSheet(dev_btn_compact)
        self._br_refresh_btn.setFocusPolicy(Qt.NoFocus)
        self._br_refresh_btn.clicked.connect(self._dev_refresh_branches)
        branches_layout.addWidget(self._br_refresh_btn)

        self._branches_nav_buttons = []
        self._branches_nav_buttons.append(
            (self._br_refresh_btn, "Refresh", dev_btn_compact)
        )

        # Scrollable branch grid: button left, ahead/behind + GUI flag right
        self._dev_branch_grid = QGridLayout()
        self._dev_branch_grid.setSpacing(4)
        branch_holder = QWidget()
        branch_holder.setLayout(self._dev_branch_grid)
        branch_scroll = QScrollArea()
        branch_scroll.setWidgetResizable(True)
        branch_scroll.setWidget(branch_holder)
        branch_scroll.setStyleSheet("QScrollArea { border: none; }")
        branch_scroll.setMinimumHeight(280)
        branches_layout.addWidget(branch_scroll, stretch=1)

        # Back to Developer
        exit_branches_style = (
            button_style.replace("#2a2a2a", "#4a1a1a")
                        .replace("#3a3a3a", "#6a2a2a")
                        .replace("#1a1a1a", "#300a0a")
        )
        btn_exit_br = QPushButton("Back to Developer")
        btn_exit_br.setStyleSheet(exit_branches_style)
        btn_exit_br.setFocusPolicy(Qt.NoFocus)
        btn_exit_br.clicked.connect(self._show_developer_page)
        branches_layout.addWidget(btn_exit_br)
        self._branches_nav_buttons.append(
            (btn_exit_br, "Back to Developer", exit_branches_style)
        )

        self._options_stack.addWidget(page_branches)  # index 5

        # --- Page 6: Manage Running Processes (sub-page of Developer) ---
        page_processes = QWidget()
        processes_layout = QVBoxLayout(page_processes)
        processes_layout.setContentsMargins(6, 0, 6, 0)

        lbl_pr_title = QLabel("MANAGE RUNNING PROCESSES")
        lbl_pr_title.setFont(section_title_font)
        lbl_pr_title.setAlignment(Qt.AlignCenter)
        lbl_pr_title.setStyleSheet(section_title_style)
        processes_layout.addWidget(lbl_pr_title)

        proc_banner = QLabel(
            "Foreign PIDs resolved heuristically (node name → /proc/*/cmdline). "
            "Display only — the GUI never kills foreign processes."
        )
        proc_banner.setWordWrap(True)
        proc_banner.setAlignment(Qt.AlignCenter)
        proc_banner.setStyleSheet(
            "border: none; color: #ff0; font-size: 10px;"
            " font-family: monospace; padding: 4px;"
        )
        processes_layout.addWidget(proc_banner)

        # Header row for the process table
        proc_header = QGridLayout()
        proc_header.setSpacing(4)
        proc_header_style = (
            "border: none; color: #aaa; font-size: 10px;"
            " font-family: monospace; font-weight: bold;"
        )
        # One row per PID — aggregated across all watched topics. The
        # Topics column lists every watched topic that PID publishes to.
        # Column widths kept in sync with _render_process_table below.
        for col, text in enumerate(["PID", "Source", "Topics", ""]):
            h = QLabel(text)
            h.setStyleSheet(proc_header_style)
            proc_header.addWidget(h, 0, col)
        proc_header.setColumnMinimumWidth(0, 110)
        proc_header.setColumnMinimumWidth(1, 150)
        proc_header.setColumnMinimumWidth(3, 60)
        proc_header.setColumnStretch(0, 0)
        proc_header.setColumnStretch(1, 0)
        proc_header.setColumnStretch(2, 1)
        proc_header.setColumnStretch(3, 0)
        proc_header_holder = QWidget()
        proc_header_holder.setLayout(proc_header)
        processes_layout.addWidget(proc_header_holder)

        # Body rows live in their own scroll area, repopulated on every poll.
        self._proc_table = QGridLayout()
        self._proc_table.setSpacing(4)
        self._proc_table.setColumnStretch(2, 1)
        proc_holder = QWidget()
        proc_holder.setLayout(self._proc_table)
        proc_scroll = QScrollArea()
        proc_scroll.setWidgetResizable(True)
        proc_scroll.setWidget(proc_holder)
        proc_scroll.setStyleSheet("QScrollArea { border: none; }")
        proc_scroll.setMinimumHeight(320)
        proc_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        proc_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        processes_layout.addWidget(proc_scroll, stretch=1)

        # Empty-state placeholder shown when no watched-topic publishers exist.
        self._proc_empty_label = QLabel("No publishers detected on watched topics.")
        self._proc_empty_label.setAlignment(Qt.AlignCenter)
        self._proc_empty_label.setStyleSheet(
            "border: none; color: #666; font-size: 10px;"
            " font-family: monospace; padding: 6px;"
        )
        processes_layout.addWidget(self._proc_empty_label)

        self._processes_nav_buttons = []

        btn_exit_pr = QPushButton("Back to Developer")
        btn_exit_pr.setStyleSheet(exit_branches_style)
        btn_exit_pr.setFocusPolicy(Qt.NoFocus)
        btn_exit_pr.clicked.connect(self._show_developer_page)
        processes_layout.addWidget(btn_exit_pr)
        self._processes_nav_buttons.append(
            (btn_exit_pr, "Back to Developer", exit_branches_style)
        )

        self._options_stack.addWidget(page_processes)  # index 6

        # Poll container status once a second whenever the dev page is up
        self._dev_status_timer = QTimer(self)
        self._dev_status_timer.setInterval(1000)
        self._dev_status_timer.timeout.connect(self._dev_update_container_status)

        top_layout.addWidget(options_frame, stretch=2)

        # =====================================================================
        # SECTION 2: LIVE SENSOR DATA (center column)
        # =====================================================================
        sensor_frame = QFrame()
        sensor_frame.setStyleSheet("QFrame { border: 1px solid #444; }")
        sensor_outer = QVBoxLayout(sensor_frame)
        sensor_outer.setContentsMargins(10, 6, 10, 10)

        lbl_sensor = QLabel("LIVE SENSOR DATA")
        lbl_sensor.setFont(section_title_font)
        lbl_sensor.setAlignment(Qt.AlignCenter)
        lbl_sensor.setStyleSheet(section_title_style)
        sensor_outer.addWidget(lbl_sensor)

        sensor_body = QHBoxLayout()
        sensor_outer.addLayout(sensor_body, stretch=1)

        # -- 2a: Status Indicators (narrow left strip) --
        status_col = QVBoxLayout()
        status_col.setSpacing(4)
        self.status_dots = {}
        self._status_nav_buttons = []  # (QPushButton, name, base_style)

        status_btn_style = (
            "QPushButton {"
            "  background-color: #2a2a2a; color: #aaa;"
            "  border: 1px solid #555; border-radius: 3px;"
            "  padding: 1px 4px; font-size: 11px;"
            "}"
        )

        def _add_status_row(name, parent_layout):
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(
                "background-color: #555; border-radius: 7px; border: none;"
            )
            row.addWidget(dot)
            btn = QPushButton(name)
            btn.setStyleSheet(status_btn_style)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setFixedWidth(95)
            btn.clicked.connect(lambda checked=False, n=name: self._on_status_dot_clicked(n))
            row.addWidget(btn)
            row.addStretch()
            parent_layout.addLayout(row)
            self.status_dots[name] = dot
            self._status_nav_buttons.append((btn, name, status_btn_style))

        # Style for non-clickable rows (EKF status indicators). Same
        # height/width as the button rows above so the columns line up,
        # but a flat QLabel — there's nothing for the user to navigate
        # to from these rows, they're pure freshness indicators driven
        # by _ekf_pulse_tick.
        status_text_style = (
            "QLabel {"
            "  background-color: transparent; color: #aaa;"
            "  border: none;"
            "  padding: 1px 4px; font-size: 11px;"
            "}"
        )

        def _add_status_text_row(name, parent_layout):
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(
                "background-color: #555; border-radius: 7px; border: none;"
            )
            row.addWidget(dot)
            lbl = QLabel(name)
            lbl.setStyleSheet(status_text_style)
            lbl.setFixedWidth(95)
            row.addWidget(lbl)
            row.addStretch()
            parent_layout.addLayout(row)
            self.status_dots[name] = dot

        # Physical Devices
        phys_label = QLabel("Physical Devices")
        phys_label.setStyleSheet(group_label_style)
        status_col.addWidget(phys_label)
        physical_names = [
            "Camera", "Lidar", "GPS", "Encoders",
            "Power PCB",
        ]
        for name in physical_names:
            _add_status_row(name, status_col)

        # Virtual Devices
        virt_label = QLabel("Virtual Devices")
        virt_label.setStyleSheet(group_label_style + " margin-top: 6px;")
        status_col.addWidget(virt_label)
        virtual_names = ["SLAM", "CONTROL", "NAV2", "LINE DETECT", "PCA DETECT"]
        for name in virtual_names:
            _add_status_row(name, status_col)

        # EKF Filters — pure freshness indicators driven by
        # _ekf_pulse_tick. Solid green while their fused-output
        # topic is fresh, gray when stale. Not tied to a launch
        # device and not clickable — these answer "is this filter
        # publishing?", so they use the text-row helper.
        ekf_label = QLabel("EKF Filters")
        ekf_label.setStyleSheet(group_label_style + " margin-top: 6px;")
        status_col.addWidget(ekf_label)
        for name in ("Local EKF", "Map EKF"):
            _add_status_text_row(name, status_col)

        # Test System
        test_label = QLabel("Test System")
        test_label.setStyleSheet(group_label_style + " margin-top: 6px;")
        status_col.addWidget(test_label)
        _add_status_row("TEST", status_col)

        # Build dot-name → launch-device reverse mapping
        self._dot_to_device = {}
        for dev_label, dot_keys, _cmd in self._launch_devices:
            for key in dot_keys:
                self._dot_to_device[key] = dev_label
        self._dot_to_device["TEST"] = None

        left_col = QVBoxLayout()
        self._left_col = left_col
        left_col.addLayout(status_col)
        left_col.addStretch()

        # -- 2b: Sensor Display Grid --
        grid = QGridLayout()
        grid.setSpacing(4)

        def _sensor_cell(title_text):
            f = QFrame()
            f.setObjectName("sensorCell")
            f.setStyleSheet(frame_style)
            f.setFrameShape(QFrame.StyledPanel)
            f.setFrameShadow(QFrame.Raised)
            v = QVBoxLayout(f)
            v.setContentsMargins(6, 4, 6, 4)
            t = QLabel(title_text)
            t.setStyleSheet("border: none; font-weight: bold; color: #ccc;")
            t.setAlignment(Qt.AlignCenter)
            v.addWidget(t)
            val = QLabel("")
            val.setStyleSheet(val_label_style)
            val.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            val.setWordWrap(True)
            v.addWidget(val)
            v.addStretch()
            return f, v, val

        def _plot_cell(title_text):
            f = QFrame()
            f.setObjectName("sensorCell")
            f.setStyleSheet(frame_style)
            f.setFrameShape(QFrame.StyledPanel)
            f.setFrameShadow(QFrame.Raised)
            v = QVBoxLayout(f)
            v.setContentsMargins(4, 2, 4, 2)
            t = QLabel(title_text)
            t.setStyleSheet("border: none; font-weight: bold; color: #ccc;")
            t.setAlignment(Qt.AlignCenter)
            v.addWidget(t)
            return f, v

        # Camera — image canvas for video playback. The title flips
        # between "Camera RAW" and "Camera CUDA" depending on whether
        # the CUDA line detector is publishing /line_detection/line_pixels.
        cam_cell, cam_layout = _plot_cell("Camera RAW")
        self._cam_title_label = cam_layout.itemAt(0).widget()
        self._cam_title_text = "Camera RAW"
        # dpi=40 (instead of the default 80) so the rasterized pixmap is
        # half the size and Agg paint is ~4x cheaper. Sensor panels are
        # ~400 px wide on screen so we lose almost no perceptible detail.
        self._cam_fig, self._cam_ax, self._cam_canvas = _make_dark_canvas(dpi=40)
        self._cam_fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._cam_ax.set_facecolor('#111111')
        self._cam_ax.axis('off')
        self._cam_no_video_txt = self._cam_ax.text(
            0.5, 0.5, 'VIDEO NOT AVAILABLE\nFOR THIS SET OF DATA',
            transform=self._cam_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._cam_no_video_txt.set_visible(False)
        self._cam_live_txt = self._cam_ax.text(
            0.5, 0.5, 'NO DATA AVAILABLE\nAT THE MOMENT',
            transform=self._cam_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._cam_live_txt.set_visible(False)
        # Live FPS overlay (bottom-right). Updated from the frame-ready
        # slot; shows how close we're actually running to the 7.5 Hz
        # target so the operator can see paint-rate health at a glance.
        # fontsize bumped to 14 because the canvas dpi dropped to 40 —
        # matplotlib renders text at fontsize * dpi / 72 device pixels,
        # so the previous fontsize=8 rendered shrunk after the dpi
        # cut. 14 is the value that visually matches the old 8 at the
        # previous dpi=80.
        self._cam_fps_txt = self._cam_ax.text(
            0.985, 0.02, '', transform=self._cam_ax.transAxes,
            fontsize=14, color='#0f0', family='monospace',
            ha='right', va='bottom', zorder=20,
        )
        self._cam_canvas.setMinimumSize(50, 50)
        cam_layout.addWidget(self._cam_canvas, stretch=1)

        # Lidar — image canvas for video playback. Title flips between
        # "LIDAR Heightband" (plain BEV) and "LIDAR PCA" when the grade
        # detector is publishing /scan_pca_filtered_points.
        lidar_cell, lidar_layout = _plot_cell("LIDAR Heightband")
        self._lidar_title_label = lidar_layout.itemAt(0).widget()
        self._lidar_title_text = "LIDAR Heightband"
        # Square figsize so aspect='equal' on the BEV doesn't waste
        # axes space. dpi=60 (vs camera's 40) because the lidar shows
        # small ~few-pixel obstacle markers — at dpi=40 the resampled
        # pixmap is so small the marks disappear. dpi=60 still gives a
        # ~3x paint speedup over the original dpi=80 while keeping
        # individual PCA dots clearly visible.
        self._lidar_fig, self._lidar_ax, self._lidar_canvas = _make_dark_canvas(
            figsize=(2.4, 2.4), dpi=60)
        self._lidar_fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._lidar_ax.set_facecolor('#111111')
        self._lidar_ax.axis('off')
        self._lidar_no_video_txt = self._lidar_ax.text(
            0.5, 0.5, 'VIDEO NOT AVAILABLE\nFOR THIS SET OF DATA',
            transform=self._lidar_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._lidar_no_video_txt.set_visible(False)
        self._lidar_live_txt = self._lidar_ax.text(
            0.5, 0.5, 'NO DATA AVAILABLE\nAT THE MOMENT',
            transform=self._lidar_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._lidar_live_txt.set_visible(False)
        # Same dpi-compensation reasoning as the camera FPS text.
        self._lidar_fps_txt = self._lidar_ax.text(
            0.985, 0.02, '', transform=self._lidar_ax.transAxes,
            fontsize=14, color='#0f0', family='monospace',
            ha='right', va='bottom', zorder=20,
        )
        self._lidar_canvas.setMinimumSize(50, 50)
        lidar_layout.addWidget(self._lidar_canvas, stretch=1)

        # GPS — map plot with satellite background (no axis clutter).
        # Title flips to "GPS Covariance [fix]" when the message
        # carries a usable position_covariance. The red dot is the
        # latest NavSatFix (not a rolling mean); the ellipse is the
        # 2σ contour of that fix's reported error distribution.
        gps_cell, gps_layout = _plot_cell("GPS")
        self._gps_title_label = gps_layout.itemAt(0).widget()
        self._gps_title_text = "GPS"
        self._gps_fig, self._gps_ax, self._gps_canvas = _make_dark_canvas()
        self._gps_fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._gps_ax.set_facecolor('#111111')
        self._gps_ax.set_xticklabels([])
        self._gps_ax.set_yticklabels([])
        self._gps_ax.tick_params(axis='both', length=0, pad=0)
        for spine in self._gps_ax.spines.values():
            spine.set_visible(False)
        self._gps_coord_label = self._gps_ax.text(
            0.02, 0.96, '', transform=self._gps_ax.transAxes,
            fontsize=7, color='#0f0', verticalalignment='top',
            family='monospace',
            bbox=dict(facecolor='#222222', alpha=0.6, edgecolor='none', pad=2),
        )
        self._gps_no_data_txt = self._gps_ax.text(
            0.5, 0.5, 'NO GPS FOR\nTHIS DATA SET',
            transform=self._gps_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._gps_no_data_txt.set_visible(False)
        self._gps_live_txt = self._gps_ax.text(
            0.5, 0.5, 'NO DATA AVAILABLE\nAT THE MOMENT',
            transform=self._gps_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._gps_live_txt.set_visible(False)
        # Shown when we have GPS but no map tile (offline + uncached area).
        # Replaces the bare matplotlib background that would otherwise look
        # like a blank/white screen and made operators think the GUI froze.
        self._gps_offline_txt = self._gps_ax.text(
            0.5, 0.5, 'OFFLINE — NO TILES CACHED\nFOR THIS AREA',
            transform=self._gps_ax.transAxes, fontsize=8, color='#888',
            ha='center', va='center', family='monospace',
        )
        self._gps_offline_txt.set_visible(False)
        self._gps_map_im = None          # imshow handle for map background
        self._gps_map_extent = None       # (lon_min, lon_max, lat_min, lat_max)
        self._gps_map_img = None          # numpy RGB array
        self._gps_trail, = self._gps_ax.plot([], [], 'c-', linewidth=1, alpha=0.6)
        self._gps_dot, = self._gps_ax.plot([], [], 'ro', markersize=5, zorder=5)
        self._gps_canvas.setMinimumSize(50, 50)
        gps_layout.addWidget(self._gps_canvas, stretch=1)

        # Encoders — XY odometry plot with white trail + 1m grid.
        # Title graduates from "Encoders (Odom)" to "Encoders (Odom EKF)"
        # whenever /local_ekf/odom is fresh — see _set_enc_title and
        # the live-tick odom block. Raw /odom is what we plot before
        # the Local EKF is up; the filtered stream takes over once
        # the EKF starts publishing, giving the operator the same
        # estimate that the rest of the stack actually uses.
        enc_cell, enc_layout = _plot_cell("Encoders (Odom)")
        # Hold a reference to the title QLabel so the live tick can
        # flip the text. _plot_cell adds the title as the first
        # widget in the returned layout.
        self._enc_title_label = enc_layout.itemAt(0).widget()
        self._enc_title_text = "Encoders (Odom)"
        self._odom_fig, self._odom_ax, self._odom_canvas = _make_dark_canvas()
        self._odom_fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self._odom_ax.set_facecolor('#111111')
        self._odom_ax.tick_params(axis='both', length=0, pad=-12,
                                   labelsize=6, colors='#666', direction='in')
        for spine in self._odom_ax.spines.values():
            spine.set_visible(False)
        self._odom_ax.set_aspect('equal', adjustable='box')
        self._odom_ax.grid(True, which='both', color='#333', linewidth=0.5)
        self._odom_scatter = None  # Line2D trail, updated in-place
        # Inside axis label + distance readout
        self._odom_xy_label = self._odom_ax.text(
            0.02, 0.96, 'x / y (m)', transform=self._odom_ax.transAxes,
            fontsize=7, color='#888', verticalalignment='top',
            bbox=dict(facecolor='#222222', alpha=0.6, edgecolor='none', pad=2),
        )
        self._odom_dist_label = self._odom_ax.text(
            0.98, 0.96, '', transform=self._odom_ax.transAxes,
            fontsize=7, color='#0f0', verticalalignment='top',
            horizontalalignment='right', family='monospace',
            bbox=dict(facecolor='#222222', alpha=0.6, edgecolor='none', pad=2),
        )
        self._odom_live_txt = self._odom_ax.text(
            0.5, 0.5, 'NO DATA AVAILABLE\nAT THE MOMENT',
            transform=self._odom_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        self._odom_live_txt.set_visible(False)
        self._odom_canvas.setMinimumSize(50, 50)
        enc_layout.addWidget(self._odom_canvas, stretch=1)

        # Power PCB — vertical column with 3 mini oscilloscopes + SOC gauge
        power_cell, power_layout = _plot_cell("Power PCB")

        pwr_val_style = (
            "border: none; color: #0f0; font-size: 10px;"
            " font-family: monospace;"
        )
        pwr_title_style = "border: none; font-weight: bold; color: #ccc; font-size: 9px;"

        # Voltage mini-graph — fixed Y: 20-30
        self._pwr_v_fig, self._pwr_v_ax, self._pwr_v_canvas = _make_mini_canvas()
        _style_ax(self._pwr_v_ax)
        self._pwr_v_ax.set_ylim(20, 30)
        # Hide x-axis labels permanently — set once at construction so
        # _redraw_plots doesn't need to call set_xticklabels([]) every
        # tick (which otherwise undoes itself after each set_xlim).
        self._pwr_v_ax.tick_params(labelbottom=False)
        self._pwr_line_v, = self._pwr_v_ax.plot([], [], color='#4af', linewidth=1)
        self._pwr_v_live_txt = self._pwr_v_ax.text(
            0.5, 0.5, 'NO DATA', transform=self._pwr_v_ax.transAxes,
            fontsize=7, color='#555', ha='center', va='center', family='monospace',
        )
        self._pwr_v_live_txt.set_visible(False)
        self._pwr_v_canvas.setMinimumSize(50, 20)

        pwr_v_title_row = QHBoxLayout()
        pwr_v_title_lbl = QLabel("Voltage (V)")
        pwr_v_title_lbl.setStyleSheet(pwr_title_style)
        self._pwr_val_v = QLabel("V: --")
        self._pwr_val_v.setAlignment(Qt.AlignRight)
        self._pwr_val_v.setStyleSheet(pwr_val_style.replace("#0f0", "#4af"))
        pwr_v_title_row.addWidget(pwr_v_title_lbl)
        pwr_v_title_row.addWidget(self._pwr_val_v)
        power_layout.addLayout(pwr_v_title_row)
        power_layout.addWidget(self._pwr_v_canvas, stretch=1)

        # Current mini-graph — fixed Y: 0-6
        self._pwr_i_fig, self._pwr_i_ax, self._pwr_i_canvas = _make_mini_canvas()
        _style_ax(self._pwr_i_ax)
        self._pwr_i_ax.set_ylim(0, 6)
        self._pwr_i_ax.tick_params(labelbottom=False)
        self._pwr_line_i, = self._pwr_i_ax.plot([], [], color='#f44', linewidth=1)
        self._pwr_i_live_txt = self._pwr_i_ax.text(
            0.5, 0.5, 'NO DATA', transform=self._pwr_i_ax.transAxes,
            fontsize=7, color='#555', ha='center', va='center', family='monospace',
        )
        self._pwr_i_live_txt.set_visible(False)
        self._pwr_i_canvas.setMinimumSize(50, 20)

        pwr_i_title_row = QHBoxLayout()
        pwr_i_title_lbl = QLabel("Current (A)")
        pwr_i_title_lbl.setStyleSheet(pwr_title_style)
        self._pwr_val_i = QLabel("I: --")
        self._pwr_val_i.setAlignment(Qt.AlignRight)
        self._pwr_val_i.setStyleSheet(pwr_val_style.replace("#0f0", "#f44"))
        pwr_i_title_row.addWidget(pwr_i_title_lbl)
        pwr_i_title_row.addWidget(self._pwr_val_i)
        power_layout.addLayout(pwr_i_title_row)
        power_layout.addWidget(self._pwr_i_canvas, stretch=1)

        # Power mini-graph — fixed Y: 0-100
        self._pwr_p_fig, self._pwr_p_ax, self._pwr_p_canvas = _make_mini_canvas()
        _style_ax(self._pwr_p_ax)
        self._pwr_p_ax.set_ylim(0, 100)
        self._pwr_p_ax.tick_params(labelbottom=False)
        self._pwr_line_p, = self._pwr_p_ax.plot([], [], color='#4f4', linewidth=1)
        self._pwr_p_live_txt = self._pwr_p_ax.text(
            0.5, 0.5, 'NO DATA', transform=self._pwr_p_ax.transAxes,
            fontsize=7, color='#555', ha='center', va='center', family='monospace',
        )
        self._pwr_p_live_txt.set_visible(False)
        self._pwr_p_canvas.setMinimumSize(50, 20)

        pwr_p_title_row = QHBoxLayout()
        pwr_p_title_lbl = QLabel("Power (W)")
        pwr_p_title_lbl.setStyleSheet(pwr_title_style)
        self._pwr_val_p = QLabel("P: --")
        self._pwr_val_p.setAlignment(Qt.AlignRight)
        self._pwr_val_p.setStyleSheet(pwr_val_style.replace("#0f0", "#4f4"))
        pwr_p_title_row.addWidget(pwr_p_title_lbl)
        pwr_p_title_row.addWidget(self._pwr_val_p)
        power_layout.addLayout(pwr_p_title_row)
        power_layout.addWidget(self._pwr_p_canvas, stretch=1)

        # SOC fuel gauge bar — horizontal bar with info row beneath
        soc_row = QHBoxLayout()
        soc_title = QLabel("SOC")
        soc_title.setStyleSheet("border: none; font-weight: bold; color: #ccc; font-size: 9px;")
        soc_row.addWidget(soc_title)
        self._soc_bar = QFrame()
        self._soc_bar.setFixedHeight(18)
        self._soc_bar.setStyleSheet(
            "background-color: #252525; border: 1px solid #555; border-radius: 2px;"
        )
        self._soc_bar_layout = QHBoxLayout(self._soc_bar)
        self._soc_bar_layout.setContentsMargins(2, 2, 2, 2)
        self._soc_bar_layout.setSpacing(0)
        self._soc_fill = QFrame()
        self._soc_fill.setStyleSheet("background-color: #4f4; border: none; border-radius: 1px;")
        self._soc_bar_layout.addWidget(self._soc_fill, stretch=0)  # fill at 0%
        self._soc_bar_layout.addStretch(1)  # empty space on right
        soc_row.addWidget(self._soc_bar, stretch=1)
        power_layout.addLayout(soc_row)
        # SOC info row: percentage left-aligned, ETA right-aligned
        soc_info_row = QHBoxLayout()
        soc_info_row.setContentsMargins(0, 0, 0, 0)
        self._soc_label = QLabel("0%")
        self._soc_label.setStyleSheet("border: none; color: #888; font-size: 9px; font-family: monospace;")
        self._eta_label = QLabel("--:--")
        self._eta_label.setAlignment(Qt.AlignRight)
        self._eta_label.setStyleSheet("border: none; color: #888; font-size: 9px; font-family: monospace;")
        soc_info_row.addWidget(self._soc_label)
        soc_info_row.addStretch(1)
        soc_info_row.addWidget(self._eta_label)
        power_layout.addLayout(soc_info_row)

        # Add Power PCB below the device status list
        left_col.addWidget(power_cell, stretch=1)
        sensor_body.addLayout(left_col)

        # -- 2b continued: Sensor grid (Camera, Lidar, GPS, Encoders) --
        self._sensor_grid = grid
        self._sensor_cells = [cam_cell, lidar_cell, gps_cell, enc_cell]
        self._sensor_grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self._expanded_cell = None

        grid.addWidget(cam_cell, 0, 0)
        grid.addWidget(lidar_cell, 0, 1)
        grid.addWidget(gps_cell, 1, 0)
        grid.addWidget(enc_cell, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        # Click to expand/collapse sensor cells
        self._canvas_to_cell = {
            self._cam_canvas: cam_cell,
            self._lidar_canvas: lidar_cell,
            self._gps_canvas: gps_cell,
            self._odom_canvas: enc_cell,
        }
        for canvas in self._canvas_to_cell:
            canvas.installEventFilter(self)
        for cell in self._sensor_cells:
            cell.mousePressEvent = lambda event, c=cell: self._toggle_sensor_expand(c)
            cell.setCursor(Qt.PointingHandCursor)

        self._power_cell = power_cell
        power_cell.mousePressEvent = lambda event, c=power_cell: self._toggle_sensor_expand(c)
        power_cell.setCursor(Qt.PointingHandCursor)

        # Sensor frame style for keyboard nav selection
        self._sensor_frame_style = frame_style
        self._sensor_sel_style = (
            "QFrame#sensorCell {"
            "  border: 2px solid #0af;"
            "  background-color: #1a2a3a;"
            "  border-radius: 3px;"
            "}"
        )

        # Nav columns for sensor cells — split into left/right columns
        self._status_nav_buttons.append((power_cell, "Power PCB", frame_style))
        self._sensor_left_col = [
            (cam_cell, "Camera", frame_style),
            (gps_cell, "GPS", frame_style),
        ]
        self._sensor_right_col = [
            (lidar_cell, "Lidar", frame_style),
            (enc_cell, "Encoders", frame_style),
        ]

        sensor_body.addLayout(grid, stretch=1)

        # -- Playback time slider --
        slider_row = QHBoxLayout()

        # Stacked time + frame labels on the left
        slider_info = QVBoxLayout()
        slider_info.setSpacing(0)
        self.pb_time_label = QLabel("0.0s / 0.0s")
        self.pb_time_label.setStyleSheet(
            "border: none; color: #888; font-size: 10px; font-family: monospace;"
        )
        self.pb_time_label.setFixedWidth(100)
        slider_info.addWidget(self.pb_time_label)
        self.pb_frame_label = QLabel("F: -- / --")
        self.pb_frame_label.setStyleSheet(
            "border: none; color: #555; font-size: 10px; font-family: monospace;"
        )
        self.pb_frame_label.setFixedWidth(100)
        slider_info.addWidget(self.pb_frame_label)
        slider_row.addLayout(slider_info)

        self.pb_slider = QSlider(Qt.Horizontal)
        self.pb_slider.setRange(0, 0)
        self.pb_slider.setEnabled(False)
        self.pb_slider.setStyleSheet(
            "QSlider { border: none; }"
            "QSlider::groove:horizontal {"
            "  background: #333; height: 6px; border-radius: 3px;"
            "}"
            "QSlider::handle:horizontal {"
            "  background: #0af; width: 14px; margin: -4px 0;"
            "  border-radius: 7px;"
            "}"
        )
        self.pb_slider.sliderPressed.connect(self._on_slider_pressed)
        self.pb_slider.sliderReleased.connect(self._on_slider_released)
        self.pb_slider.sliderMoved.connect(self._on_slider_seek)
        slider_row.addWidget(self.pb_slider, stretch=1)

        self._play_pause_style = (
            "QPushButton { background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 3px;"
            "  padding: 4px 10px; font-size: 16px;"
            "  min-width: 36px; min-height: 24px; }"
            "QPushButton:hover { background-color: #3a3a3a; }"
            "QPushButton:disabled { color: #555; }"
        )
        play_pause_style = self._play_pause_style
        self.btn_pp = QPushButton("\u25B6")
        self.btn_pp.setFixedSize(40, 28)
        self.btn_pp.setStyleSheet(play_pause_style)
        self.btn_pp.setFocusPolicy(Qt.NoFocus)
        self.btn_pp.setEnabled(False)
        self.btn_pp.clicked.connect(self._on_play_pause)
        slider_row.addWidget(self.btn_pp)

        # Playback speed button
        self._pb_speed_options = [1.0, 2.0, 4.0]
        self._pb_speed_idx = 0
        self._pb_speed = 1.0
        speed_btn_style = (
            "QPushButton { background-color: #2a2a2a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 3px;"
            "  padding: 4px 6px; font-size: 11px;"
            "  min-width: 36px; min-height: 24px; }"
            "QPushButton:hover { background-color: #3a3a3a; }"
        )
        self._speed_btn_style = speed_btn_style
        self.btn_speed = QPushButton("1x")
        self.btn_speed.setFixedSize(40, 28)
        self.btn_speed.setStyleSheet(speed_btn_style)
        self.btn_speed.setFocusPolicy(Qt.NoFocus)
        self.btn_speed.clicked.connect(self._cycle_playback_speed)
        slider_row.addWidget(self.btn_speed)

        sensor_outer.addLayout(slider_row)

        top_layout.addWidget(sensor_frame, stretch=5)

        # =====================================================================
        # SECTION 3: PROCESS TERMINAL (right column)
        # =====================================================================
        viz_frame = QFrame()
        viz_frame.setStyleSheet("QFrame { border: 1px solid #444; }")
        viz_layout = QVBoxLayout(viz_frame)
        viz_layout.setContentsMargins(10, 6, 10, 10)

        # Title row with the GUI-FPS readout pinned to the LEFT so it
        # doesn't sit underneath the AUTO ON/OFF badge that floats over
        # the upper-right of the central widget. A right-side spacer
        # of the same fixed width balances the layout so the title
        # itself stays visually centered. Operator uses the number to
        # verify the Qt event loop is keeping its 30 FPS target —
        # drops mean something downstream (sensor paints, plot redraws,
        # etc.) is starving the main thread.
        viz_title_row = QHBoxLayout()
        viz_title_row.setContentsMargins(0, 0, 0, 0)
        _gui_fps_width = 110
        self._gui_fps_label = QLabel("GUI -- FPS")
        self._gui_fps_label.setFixedWidth(_gui_fps_width)
        self._gui_fps_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        viz_title_row.addWidget(self._gui_fps_label)
        lbl_viz = QLabel("PROCESS TERMINAL")
        lbl_viz.setFont(section_title_font)
        lbl_viz.setAlignment(Qt.AlignCenter)
        lbl_viz.setStyleSheet(section_title_style)
        viz_title_row.addWidget(lbl_viz, stretch=1)
        # Invisible balancing spacer so the PROCESS TERMINAL title is
        # centered across the *whole* row, not centered relative to the
        # remaining space after the FPS label.
        _gui_fps_spacer = QLabel("")
        _gui_fps_spacer.setFixedWidth(_gui_fps_width)
        viz_title_row.addWidget(_gui_fps_spacer)
        viz_layout.addLayout(viz_title_row)
        # Initial color follows the current theme. _apply_theme() at
        # the end of __init__ re-applies on the first dark→light flip.
        self._apply_gui_fps_label_style()

        self._term_header = QLabel("Select a process from the status dots")
        self._term_header.setAlignment(Qt.AlignCenter)
        self._term_header.setStyleSheet(
            "border: none; color: #888; font-size: 12px; font-family: monospace;"
        )
        viz_layout.addWidget(self._term_header)

        self._term_display = QTextEdit()
        self._term_display.setReadOnly(True)
        self._term_display.setFocusPolicy(Qt.NoFocus)
        self._term_display.setLineWrapMode(QTextEdit.WidgetWidth)
        self._term_display.setStyleSheet(
            "QTextEdit {"
            "  background-color: #0a0a0a; color: #0f0;"
            "  border: 1px solid #333; border-radius: 3px;"
            "  font-family: monospace; font-size: 11px;"
            "  padding: 4px;"
            "}"
        )
        viz_layout.addWidget(self._term_display, stretch=1)

        self._selected_process = None
        self._gui_log = []  # GUI info log lines
        self._term_last_text = ''  # cache to skip redundant updates

        # Process management state
        self._process_objects = {}   # label → subprocess.Popen
        self._process_buffers = {}   # label → list of str lines
        self._process_readers = {}   # label → threading.Thread

        # 2 Hz timer to poll process output and refresh terminal
        self._process_poll_timer = QTimer()
        self._process_poll_timer.setInterval(1000)  # 1 Hz process polling
        self._process_poll_timer.timeout.connect(self._poll_process_output)
        self._process_poll_timer.start()

        # Container health check — every 5 seconds
        self._container_health_timer = QTimer()
        self._container_health_timer.setInterval(5000)
        self._container_health_timer.timeout.connect(self._check_container_health)
        self._container_health_timer.start()

        # Duplicate-publisher detector — polls topics from config/
        # watched_topics.yaml at 0.1 Hz. Flags device dots red when
        # >1 distinct publisher source (GUI device / foreign PID) is
        # detected; feeds the Manage Running Processes dev-page window.
        self._watched_topics = self._load_watched_topics()
        self._watched_pub_state = {}         # topic -> [pub_entry, ...]
        self._dot_duplicate_flagged = set()  # dot keys currently red-flagged
        self._duplicate_poll_timer = QTimer()
        self._duplicate_poll_timer.setInterval(10000)
        self._duplicate_poll_timer.timeout.connect(self._poll_watched_publishers)
        self._duplicate_poll_timer.start()

        # REC indicator timer — reads HudNode.latest_recording_active
        # each tick. When the bag recorder is active, ramps device
        # dots black↔red at 2 Hz and shows the transparent overlay.
        # Always-on (cheap when REC is off — early returns after one
        # bool read + one comparison).
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(50)  # 20 Hz tick = 10 frames per 2 Hz cycle
        self._rec_timer.timeout.connect(self._rec_tick)
        self._rec_timer.start()

        top_layout.addWidget(viz_frame, stretch=2)

        # -- Keyboard navigation --
        self._sel_idx = 0
        # Slider highlight styles (normal vs selected)
        self._slider_base_style = (
            "QSlider { border: none; }"
            "QSlider::groove:horizontal {"
            "  background: #333; height: 6px; border-radius: 3px;"
            "}"
            "QSlider::handle:horizontal {"
            "  background: #0af; width: 14px; margin: -4px 0;"
            "  border-radius: 7px;"
            "}"
        )
        self._slider_sel_style = (
            "QSlider { border: 1px solid #0af; border-radius: 4px; }"
            "QSlider::groove:horizontal {"
            "  background: #333; height: 6px; border-radius: 3px;"
            "}"
            "QSlider::handle:horizontal {"
            "  background: #fff; width: 14px; margin: -4px 0;"
            "  border-radius: 7px;"
            "}"
        )
        self._slider_scrub_style = (
            "QSlider { border: none; }"
            "QSlider::groove:horizontal {"
            "  background: #333; height: 6px; border-radius: 3px;"
            "}"
            "QSlider::handle:horizontal {"
            "  background: #0f0; width: 14px; margin: -4px 0;"
            "  border: 2px solid #0f0; border-radius: 7px;"
            "}"
        )
        # 4-column, 14-row nav grid:
        #   Col 0: Connect(r1), Launch(r6), Live(r8), Test(r10), Playback(r12), Quit(r14)
        #   Col 1: dots(r1-r12), Power PCB plot(r13), Scrub Bar(r14)
        #   Col 2: Camera(r1), GPS(r13), Play/Pause(r14)
        #   Col 3: Lidar(r1), Odom(r13), Speed(r14)
        self._status_nav_buttons.append(
            (self.pb_slider, "Scrub Bar", self._slider_base_style))
        self._sensor_left_col.append(
            (self.btn_pp, "\u25B6", play_pause_style))
        self._sensor_right_col.append(
            (self.btn_speed, "1x", speed_btn_style))
        self._nav_groups = [
            self._nav_buttons,          # col 0
            self._status_nav_buttons,   # col 1
            self._sensor_left_col,      # col 2
            self._sensor_right_col,     # col 3
        ]
        # Logical row numbers for row-matched Left/Right navigation
        _col0_rows = [1, 6, 8, 10, 12, 14]
        _col1_rows = list(range(1, len(self._status_nav_buttons) + 1))
        _col2_rows = [1, 13, 14]
        _col3_rows = [1, 13, 14]
        self._nav_logical_rows = [_col0_rows, _col1_rows, _col2_rows, _col3_rows]
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row = [0, 0, 0, 0]
        self._scrub_mode = False  # True when actively scrubbing with arrows
        self._speed_mode = False  # True when selecting playback speed with arrows

        self._sel_frames_r = [' <', '< ']
        self._sel_frames_l = ['> ', ' >']
        self._sel_frame_durations = [3, 3]
        self._sel_frame_idx = 0
        self._sel_tick_count = 0
        self._sel_arrow_l = self._sel_frames_l[0]
        self._sel_arrow_r = self._sel_frames_r[0]

        # Floating directional indicators (overlays on the central widget)
        indicator_style = (
            "color: #0af; font-size: 12px; font-weight: bold;"
            " font-family: monospace; background: transparent; border: none;"
        )
        self._ind_left = QLabel("<<", central)
        self._ind_right = QLabel(">>", central)
        for ind in (self._ind_left, self._ind_right):
            ind.setStyleSheet(indicator_style)
            ind.setAlignment(Qt.AlignCenter)
            ind.adjustSize()
            ind.raise_()
            ind.hide()

        # Auto-mode badge in the upper-right corner. The 'A' key toggles
        # _auto_mode (handled in keyPressEvent); the badge re-paints to
        # reflect the state. Parent is the central widget so it floats
        # over the layout; _position_auto_badge keeps it pinned to the
        # top-right on resize.
        self._auto_mode = False
        self._auto_badge_on_style = (
            "color: #0f0; font-size: 13px; font-weight: bold;"
            " font-family: monospace; background-color: rgba(0, 40, 0, 200);"
            " border: 1px solid #0f0; border-radius: 4px; padding: 3px 8px;"
        )
        self._auto_badge_off_style = (
            "color: #666; font-size: 13px; font-weight: bold;"
            " font-family: monospace; background-color: rgba(20, 20, 20, 180);"
            " border: 1px solid #444; border-radius: 4px; padding: 3px 8px;"
        )
        self._auto_badge = QLabel("AUTO OFF", central)
        self._auto_badge.setStyleSheet(self._auto_badge_off_style)
        self._auto_badge.setAlignment(Qt.AlignCenter)
        self._auto_badge.adjustSize()
        self._auto_badge.raise_()
        self._auto_badge.show()
        # central hasn't been laid out yet at __init__ time, so its
        # width() is still 0 and an immediate _position_auto_badge would
        # clamp x to 0 (badge stuck in the top-left). Defer to after the
        # event loop processes the show + first layout pass, when
        # parent.width() reflects the real geometry.
        QTimer.singleShot(0, self._position_auto_badge)

        self._update_selection()

        self._nav_timer = QTimer()
        self._nav_timer.setInterval(250)  # 4 Hz animation
        self._nav_timer.timeout.connect(self._nav_anim_tick)
        self._nav_timer.start()

        # EKF status timer — drives both the green↔purple participation
        # pulse on device dots AND the "Local EKF / Map EKF" status
        # rows. Runs whenever the GUI is up, NOT only in Live mode:
        # the operator should be able to glance at the launch screen
        # and see whether the filters are publishing without first
        # having to enter Live. _ekf_pulse_tick reads only last_msg_t
        # + status_dots, so it's safe to call before live mode is
        # entered (gracefully no-ops when the ROS node is None).
        self._ekf_status_timer = QTimer()
        self._ekf_status_timer.setInterval(200)  # 5 Hz
        self._ekf_status_timer.timeout.connect(self._ekf_status_tick)
        self._ekf_status_timer.start()

        # The widget tree was built using dark hex codes; flip to whichever
        # theme is selected (light by default).
        self._apply_theme()

        # Application-wide event filter. This is what makes the screen
        # lock actually block input: when _screen_locked is True the
        # filter swallows every KeyPress / mouse event in the GUI
        # process, except those targeted at the lock overlay's password
        # field. Installed unconditionally so we don't have to manage
        # install/uninstall on lock/unlock.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -----------------------------------------------------------------
    # Theme helpers
    # -----------------------------------------------------------------
    def _light_to_dark_map(self):
        return {v: k for k, v in self._DARK_TO_LIGHT.items()}

    def _translate_to_theme(self, s):
        """Translate a stylesheet authored in dark hex codes to whichever
        theme is currently active. Sites that build inline styles at
        runtime (selection highlight, _dev_refresh_branches, status
        labels) pipe through this so the active theme survives ticks
        of _update_selection that re-write `base_style`."""
        if not s or self._theme == 'dark':
            return s
        cm = self._DARK_TO_LIGHT
        return self._hex_re.sub(lambda m: cm.get(m.group(0), m.group(0)), s)

    def _restyle_terminal(self):
        """The process terminal is special-cased: it shouldn't pick up
        the regex substitution because we want a hard white-on-black
        (dark) or black-on-white (light) look, not a muted version."""
        if not hasattr(self, '_term_display'):
            return
        if self._theme == 'light':
            self._term_display.setStyleSheet(
                "QTextEdit {"
                "  background-color: #ffffff; color: #000000;"
                "  border: 1px solid #b8b8b8; border-radius: 3px;"
                "  font-family: monospace; font-size: 11px;"
                "  padding: 4px;"
                "}"
            )
        else:
            self._term_display.setStyleSheet(
                "QTextEdit {"
                "  background-color: #0a0a0a; color: #0f0;"
                "  border: 1px solid #333; border-radius: 3px;"
                "  font-family: monospace; font-size: 11px;"
                "  padding: 4px;"
                "}"
            )

    def _color_map(self, target):
        """Returns hex→hex substitution map for current → `target` theme."""
        if target == 'light':
            return dict(self._DARK_TO_LIGHT)
        return self._light_to_dark_map()

    def _recolor_widget_tree(self, root, color_map):
        """Walk root + descendants and rewrite hex codes in stylesheets."""
        def repl(m):
            return color_map.get(m.group(0), m.group(0))
        widgets = [root] + list(root.findChildren(QWidget))
        for w in widgets:
            s = w.styleSheet()
            if not s:
                continue
            new_s = self._hex_re.sub(repl, s)
            if new_s != s:
                w.setStyleSheet(new_s)

    def _set_qpalette_for_theme(self):
        """Rebuild and install QPalette for the current theme."""
        p = QPalette()
        if self._theme == 'light':
            p.setColor(QPalette.Window, QColor(237, 237, 237))
            p.setColor(QPalette.WindowText, QColor(32, 32, 32))
            p.setColor(QPalette.Base, QColor(250, 250, 250))
            p.setColor(QPalette.AlternateBase, QColor(232, 232, 232))
            p.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
            p.setColor(QPalette.ToolTipText, QColor(32, 32, 32))
            p.setColor(QPalette.Text, QColor(32, 32, 32))
            p.setColor(QPalette.Button, QColor(232, 232, 232))
            p.setColor(QPalette.ButtonText, QColor(32, 32, 32))
            p.setColor(QPalette.BrightText, QColor(180, 0, 0))
            p.setColor(QPalette.Highlight, QColor(42, 130, 218))
            p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        else:
            p.setColor(QPalette.Window, QColor(20, 20, 20))
            p.setColor(QPalette.WindowText, QColor(220, 220, 220))
            p.setColor(QPalette.Base, QColor(30, 30, 30))
            p.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
            p.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
            p.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
            p.setColor(QPalette.Text, QColor(220, 220, 220))
            p.setColor(QPalette.Button, QColor(40, 40, 40))
            p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
            p.setColor(QPalette.BrightText, QColor(255, 50, 50))
            p.setColor(QPalette.Highlight, QColor(42, 130, 218))
            p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(p)

    def _restyle_canvases(self):
        """Recolor matplotlib figures and axes for the current theme.
        Stylesheet substitution doesn't reach matplotlib internals."""
        if self._theme == 'light':
            fig_bg = '#fafafa'
            ax_bg = '#ffffff'
            tick_color = '#555'
            spine_color = '#bbb'
            label_color = '#444'
            odom_trail = '#000000'
            odom_grid = '#cccccc'
        else:
            fig_bg = '#1e1e1e'
            ax_bg = '#111111'
            tick_color = '#888'
            spine_color = '#444'
            label_color = '#888'
            odom_trail = 'white'
            odom_grid = '#333'
        canvases = self.findChildren(FigureCanvasQTAgg)
        for canvas in canvases:
            fig = canvas.figure
            fig.set_facecolor(fig_bg)
            for ax in fig.get_axes():
                ax.set_facecolor(ax_bg)
                ax.tick_params(colors=tick_color)
                for spine in ax.spines.values():
                    spine.set_color(spine_color)
                ax.xaxis.label.set_color(label_color)
                ax.yaxis.label.set_color(label_color)
                if ax.get_title():
                    ax.title.set_color(label_color)
            canvas.draw_idle()
        # Odom plot specifics: the trail line and grid don't fit the
        # generic semantic-color palette (they're contrast-on-bg, not
        # value-encoding), so they swap with the theme.
        if hasattr(self, '_odom_scatter') and self._odom_scatter is not None:
            self._odom_scatter.set_color(odom_trail)
        if hasattr(self, '_odom_ax'):
            for line in (self._odom_ax.get_xgridlines()
                         + self._odom_ax.get_ygridlines()):
                line.set_color(odom_grid)
            # Re-color the corner texts (x/y, distance, live indicator)
            for txt_attr in ('_odom_xy_label', '_odom_dist_label',
                             '_odom_live_txt'):
                if hasattr(self, txt_attr):
                    getattr(self, txt_attr).set_color(label_color)
        if hasattr(self, '_odom_canvas'):
            self._odom_canvas.draw_idle()

    def _apply_theme(self):
        """Push the current theme to QPalette + the widget tree + canvases.
        Called once at end of __init__ (light by default) and again on every
        toggle. The widget tree is always built in dark colors during
        __init__, so on first call this flips dark → light if theme is
        light. On subsequent calls, the tree already reflects the previous
        theme, so we substitute previous → current."""
        # The widget tree was last styled for the OPPOSITE of the current
        # theme (we just toggled), so we substitute the inverse.
        prev = 'dark' if self._theme == 'light' else 'light'
        # Map: prev → current
        if self._theme == 'light':
            color_map = dict(self._DARK_TO_LIGHT)
        else:
            color_map = self._light_to_dark_map()
        self._set_qpalette_for_theme()
        self._recolor_widget_tree(self, color_map)
        self._restyle_canvases()
        self._restyle_terminal()
        self._apply_gui_fps_label_style()
        # Re-run _update_selection so the selected button picks up the
        # newly-translated theme (its stylesheet was overwritten by the
        # last tick using the stored dark base_style).
        if hasattr(self, '_nav_groups'):
            self._update_selection()
        # Update the theme button label to reflect what clicking does next.
        if hasattr(self, 'btn_theme'):
            self.btn_theme.setText(
                "Switch to Dark Mode" if self._theme == 'light'
                else "Switch to Light Mode"
            )

    def _toggle_theme(self):
        self._theme = 'dark' if self._theme == 'light' else 'light'
        self._apply_theme()

    # -----------------------------------------------------------------
    # Keyboard navigation
    # -----------------------------------------------------------------
    @staticmethod
    def _make_sel_style(base_style):
        """Derive a selected style from the base style by swapping bg/border colors."""
        s = base_style
        # Replace background colors with blue-highlighted versions
        for old, new in [
            ('background-color: #2a2a2a', 'background-color: #1a3a5a'),
            ('background-color: #4a1a1a', 'background-color: #1a3a5a'),
            ('background-color: #8b0000', 'background-color: #1a3a5a'),
            ('border: 1px solid #555', 'border: 1px solid #0af'),
            ('border: 2px solid #f00', 'border: 2px solid #0af'),
            ('color: #dcdcdc', 'color: #ffffff'),
            ('color: #fff', 'color: #ffffff'),
        ]:
            s = s.replace(old, new)
        return s

    # -- Launch/End Processes sub-page ------------------------------------------

    def _show_launch_page(self):
        """Switch OPTIONS column to the Launch/End Processes sub-page."""
        self._options_stack.setCurrentIndex(1)
        # Swap nav group 0 to the launch buttons
        self._nav_groups[0] = self._launch_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()

    # -- Playback CSV sub-page -------------------------------------------------

    def _show_playback_page(self):
        """Switch OPTIONS column to the Playback CSV selection sub-page."""
        # Rescan for any new CSV files
        self._scan_csv_files(self._pb_button_style, self._pb_csv_label_style)
        self._options_stack.setCurrentIndex(2)
        self._nav_groups[0] = self._playback_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()

    def _scan_csv_files(self, button_style, csv_label_style):
        """Populate the Playback grid with every replayable session
        under _CSV_DIR. Two source kinds are supported, both shown in
        the same list (newest sessions first) so legacy CSV/MP4 runs
        and new ROS bag runs can coexist while we transition:

          * ROS bag — a directory containing ``metadata.yaml``
            (either the entry itself, or a nested ``<entry>/bag/``
            which is the t000 layout). Loads via ``ros2 bag play``
            into the live canvases.
          * Legacy CSV — a ``.csv`` file at the top level OR the
            first CSV inside a session subdirectory. Loads via the
            scrubber-driven _load_and_play_csv path.

        Each row is tagged in the label so the operator can tell
        which path will run. Name kept (``_scan_csv_files``) for
        compatibility with the page-show plumbing.
        """
        # Clear existing grid and nav buttons (except the exit button at the end)
        exit_btn_entry = None
        if self._playback_nav_buttons:
            last = self._playback_nav_buttons[-1]
            if last[1] == "Exit Playback":
                exit_btn_entry = last

        self._playback_nav_buttons.clear()

        while self._csv_grid.count():
            item = self._csv_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        logs_dir = self._CSV_DIR
        # Each entry: (sort_key, display_name, kind, path)
        #   kind == 'bag' → path is the bag directory
        #   kind == 'csv' → path is the CSV file
        entries = []
        if os.path.isdir(logs_dir):
            # Top-level .csv files (legacy flat layout)
            for f in os.listdir(logs_dir):
                if f.lower().endswith('.csv'):
                    full = os.path.join(logs_dir, f)
                    entries.append((
                        os.path.getmtime(full),
                        f'CSV  {f}',
                        'csv',
                        full,
                    ))
            # Session sub-directories — could be a bag, a CSV folder,
            # or a hybrid t000 session (bag/ subdir + sidecar csv).
            for entry in os.listdir(logs_dir):
                sess_dir = os.path.join(logs_dir, entry)
                if not os.path.isdir(sess_dir):
                    continue
                # ROS bag at the session level?
                if os.path.isfile(os.path.join(sess_dir, 'metadata.yaml')):
                    entries.append((
                        os.path.getmtime(sess_dir),
                        f'BAG  {entry}',
                        'bag',
                        sess_dir,
                    ))
                    continue
                # t000 nested layout: <session>/bag/metadata.yaml.
                # If we find one, register that as the playable
                # source AND still look for legacy CSVs in the same
                # session folder so the operator can pick either.
                nested = os.path.join(sess_dir, 'bag')
                if (os.path.isdir(nested)
                        and os.path.isfile(os.path.join(nested, 'metadata.yaml'))):
                    entries.append((
                        os.path.getmtime(nested),
                        f'BAG  {entry}',
                        'bag',
                        nested,
                    ))
                # Legacy: first CSV in the session folder.
                csvs = sorted(
                    f for f in os.listdir(sess_dir)
                    if f.lower().endswith('.csv')
                )
                if csvs:
                    full = os.path.join(sess_dir, csvs[0])
                    entries.append((
                        os.path.getmtime(full),
                        f'CSV  {entry}',
                        'csv',
                        full,
                    ))

        # Newest sessions first so the just-recorded run is easy to find.
        entries.sort(key=lambda e: e[0], reverse=True)

        for i, (_t, display_name, kind, full_path) in enumerate(entries):
            btn = QPushButton("Load")
            btn.setStyleSheet(button_style)
            btn.setFocusPolicy(Qt.NoFocus)
            if kind == 'bag':
                btn.clicked.connect(
                    lambda checked=False, p=full_path: self._load_and_play_bag(p)
                )
            else:
                btn.clicked.connect(
                    lambda checked=False, p=full_path: self._load_and_play_csv(p)
                )
            self._csv_grid.addWidget(btn, i, 0)
            self._playback_nav_buttons.append((btn, "Load", button_style))

            lbl = QLabel(display_name)
            lbl.setStyleSheet(csv_label_style)
            self._csv_grid.addWidget(lbl, i, 1)

        if exit_btn_entry:
            self._playback_nav_buttons.append(exit_btn_entry)

    def _load_and_play_csv(self, path):
        """Load a CSV and start playback, then return to main page."""
        self._stop_playback()
        self._load_csv(path)
        self._start_playback()
        self._show_main_page()

    def _toggle_mission(self):
        """Start ./config/run_mission.sh, or kill it if already running."""
        if self._mission_running and self._mission_proc is not None:
            self._kill_process(self._mission_proc, "Mission")
            return  # _check_mission_finished will reset the button

        if not self._container_connected:
            self._gui_log_msg("Cannot run mission: container not connected")
            return

        cmd = "./config/run_mission.sh"
        label = "Mission"
        exec_cmd = self._wrap_container_cmd(cmd, label=label)
        buf = []
        self._process_buffers[label] = buf
        buf.append(f"$ {cmd}\n")
        try:
            proc = subprocess.Popen(
                exec_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            self._process_objects[label] = proc
            self._mission_proc = proc
            self._mission_running = True
            self._mission_btn.setText("Stop Mission")
            self._mission_btn.setStyleSheet(self._mission_on_style)
            for i, (b, lbl, _s) in enumerate(self._launch_nav_buttons):
                if b is self._mission_btn:
                    self._launch_nav_buttons[i] = (
                        b, lbl, self._mission_on_style
                    )
                    break

            def _reader():
                try:
                    for line in proc.stdout:
                        buf.append(line)
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            self._process_readers[label] = t
            self._mission_check_timer.start()
        except Exception as e:
            buf.append(f"[Failed to start mission: {e}]\n")
            self._mission_running = False
            self._mission_proc = None

        self._selected_process = label
        self._term_last_text = ''
        self._refresh_terminal_display()

    def _check_mission_finished(self):
        """Polled by _mission_check_timer; resets the button when the
        run_mission.sh subprocess exits (either naturally or via kill)."""
        if self._mission_proc is None:
            self._mission_check_timer.stop()
            return
        rc = self._mission_proc.poll()
        if rc is None:
            return
        self._mission_check_timer.stop()
        buf = self._process_buffers.get("Mission")
        if buf is not None:
            buf.append(f"[Mission finished with code {rc}]\n")
        self._process_objects.pop("Mission", None)
        self._process_readers.pop("Mission", None)
        self._mission_running = False
        self._mission_proc = None
        self._mission_btn.setText("Run Mission")
        self._mission_btn.setStyleSheet(self._mission_off_style)
        for i, (b, lbl, _s) in enumerate(self._launch_nav_buttons):
            if b is self._mission_btn:
                self._launch_nav_buttons[i] = (b, lbl, self._mission_off_style)
                break
        self._refresh_terminal_display()

    def _on_send_goal_clicked(self):
        """Run ./config/send_goal.sh with the args from the textfield."""
        args = self._send_goal_input.text().strip()
        # Hand focus back to the main window so arrow-key nav resumes
        # instead of staying trapped inside the QLineEdit.
        self._send_goal_input.clearFocus()
        self.setFocus(Qt.OtherFocusReason)
        self._run_one_shot_script(
            label="Send Goal",
            script="./config/send_goal.sh",
            args=args,
        )

    def _on_send_gps_clicked(self):
        """Run ./config/send_GPS_waypoint.sh with the args from the textfield.

        Commas in the field (e.g. ``37.23028, -80.42502``) are normalized
        to spaces so the script's positional <lat> <lon> [radius] parser
        accepts the input as-pasted from a maps app.
        """
        raw = self._send_gps_input.text().strip()
        args = re.sub(r'[,\s]+', ' ', raw).strip()
        self._send_gps_input.clearFocus()
        self.setFocus(Qt.OtherFocusReason)
        self._run_one_shot_script(
            label="Send GPS",
            script="./config/send_GPS_waypoint.sh",
            args=args,
        )

    def _run_one_shot_script(self, label, script, args):
        """Fire-and-forget runner for the send_goal/send_GPS scripts.

        Wraps the command for the container the same way device launches
        do, captures stdout into the per-label buffer, and selects the
        label so the terminal display shows the output. The Popen is
        kept in _process_objects so _poll_process_output reaps it when
        it exits.
        """
        if not self._container_connected:
            self._gui_log_msg(f"Cannot run {label}: container not connected")
            return

        # Shell-safe arg pass-through. The args string is appended
        # verbatim to the script invocation inside the docker exec
        # bash -lc '...' wrapper; the user owns the quoting of their
        # own input (matches how they'd run it from a shell).
        cmd = f"{script} {args}".strip()
        exec_cmd = self._wrap_container_cmd(cmd, label=label)
        buf = self._process_buffers.setdefault(label, [])
        buf.append(f"$ {cmd}\n")
        try:
            proc = subprocess.Popen(
                exec_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            self._process_objects[label] = proc

            def _reader():
                try:
                    for line in proc.stdout:
                        buf.append(line)
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            self._process_readers[label] = t
            self._gui_log_msg(f"{label}: {cmd}")
        except Exception as e:
            buf.append(f"[Failed to start: {e}]\n")

        self._selected_process = label
        self._term_last_text = ''
        self._refresh_terminal_display()

    def _show_main_page(self):
        """Switch OPTIONS column back to the main page."""
        self._options_stack.setCurrentIndex(0)
        # Restore nav group 0 to the main option buttons
        self._nav_groups[0] = self._nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        # Stop the dev page's container-status poll if it was running
        timer = getattr(self, '_dev_status_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._update_selection()

    # -- Test Mode sub-page ---------------------------------------------------

    def _on_build_clicked(self):
        """Run colcon build inside the container and source the workspace."""
        if not self._container_connected:
            return
        self._gui_log_msg("Building workspace...")
        build_cmd = "colcon build --symlink-install && source install/setup.bash"
        label = "Build Workspace"
        exec_cmd = self._wrap_container_cmd(build_cmd, label=label)
        buf = []
        self._process_buffers[label] = buf
        buf.append(f"$ {build_cmd}\n")
        try:
            proc = subprocess.Popen(
                exec_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            self._process_objects[label] = proc

            def _reader():
                try:
                    for line in proc.stdout:
                        buf.append(line)
                except Exception:
                    pass
                buf.append(f"[Build finished with code {proc.returncode}]\n")
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            self._process_readers[label] = t
        except Exception as e:
            buf.append(f"[Failed to start build: {e}]\n")

        # Show build output in terminal
        self._selected_process = label
        self._term_last_text = ''
        self._refresh_terminal_display()

    def _on_test_clicked(self):
        """Show the Test Mode page."""
        if self._live_active:
            self._stop_live_mode()
        if self._pb_state != 'idle':
            self._stop_playback()
        self._show_test_page()

    def _show_test_page(self):
        """Switch OPTIONS column to the Test Mode sub-page."""
        self._options_stack.setCurrentIndex(3)
        self._nav_groups[0] = self._test_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()

    # -- Developer sub-page ---------------------------------------------------

    def _show_developer_page(self):
        """Switch OPTIONS column to the Developer sub-page."""
        self._options_stack.setCurrentIndex(4)
        self._nav_groups[0] = self._dev_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()
        self._dev_update_branch_label()
        self._dev_update_container_status()
        self._dev_status_timer.start()
        # Audit the Load-branch-changes-and-build state each time the
        # operator returns to this page — if the previous worker died
        # without resetting the flag, the button would otherwise stay
        # disabled and the status label stuck at "Starting…".
        self._dev_rebuild_self_heal()

    def _show_branches_page(self):
        """Switch OPTIONS column to the Branch switcher sub-page (index 5)."""
        self._options_stack.setCurrentIndex(5)
        self._nav_groups[0] = self._branches_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()
        self._dev_update_branch_label()
        self._dev_refresh_branches()

    def _show_processes_page(self):
        """Switch OPTIONS column to the Manage Running Processes sub-page (index 6)."""
        self._options_stack.setCurrentIndex(6)
        self._nav_groups[0] = self._processes_nav_buttons
        self._nav_col = 0
        self._nav_row = 0
        self._nav_last_row[0] = 0
        self._update_selection()
        self._render_process_table()

    def _dev_run_git(self, args, timeout=15):
        """Run a git command in the host repo. Returns (rc, stdout, stderr)."""
        try:
            r = subprocess.run(
                ['git', '-C', self._dev_host_repo] + list(args),
                capture_output=True, text=True, timeout=timeout,
            )
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except FileNotFoundError:
            return 127, '', 'git not found on PATH'
        except subprocess.TimeoutExpired:
            return 124, '', f'git {args[0] if args else ""} timed out'
        except Exception as e:
            return 1, '', f'{e}'

    def _dev_update_branch_label(self):
        rc, out, err = self._dev_run_git(['branch', '--show-current'])
        if rc == 0 and out:
            text = f"Current branch: {out}"
        else:
            text = f"Current branch: ? ({err or 'unknown'})"
        self._dev_branch_label.setText(text)
        if hasattr(self, '_br_branch_label'):
            self._br_branch_label.setText(text)

    def _dev_set_status(self, text, color='#888'):
        # Callers pass dark-theme hex codes; translate to the active theme.
        if self._theme == 'light':
            color = self._DARK_TO_LIGHT.get(color, color)
        style = (
            f"border: none; color: {color}; font-size: 10px;"
            " font-family: monospace;"
        )
        self._dev_status_label.setStyleSheet(style)
        self._dev_status_label.setText(text)
        if hasattr(self, '_br_status_label'):
            self._br_status_label.setStyleSheet(style)
            self._br_status_label.setText(text)

    def _branch_has_gui(self, branch):
        """True if `branch`'s tree contains the GUI source file. Used to
        block switches that would leave the running GUI without source.
        Tries the local ref first, then origin/<branch>."""
        gui_path = (
            'isaac_ros-dev/src/autonav-gui-hud/autonav_gui_hud/hud_node.py'
        )
        for ref in (branch, f'origin/{branch}'):
            rc, out, _err = self._dev_run_git(
                ['ls-tree', '--name-only', ref, '--', gui_path]
            )
            if rc == 0 and out:
                return True
        return False

    def _dev_git_pull(self):
        self._dev_set_status("Running git pull --ff-only…", color='#ff0')
        QApplication.processEvents()
        rc, out, err = self._dev_run_git(['pull', '--ff-only'], timeout=60)
        if rc == 0:
            summary = out.splitlines()[-1] if out else 'Up to date'
            self._dev_set_status(f"Pull OK: {summary}", color='#0f0')
        else:
            msg = (err or out or 'unknown error').splitlines()[-1]
            self._dev_set_status(f"Pull failed: {msg}", color='#f44')
        self._dev_update_branch_label()
        self._dev_refresh_branches()

    def _dev_refresh_branches(self):
        # Clear the grid (lives on the branches sub-page now)
        while self._dev_branch_grid.count():
            item = self._dev_branch_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._dev_branch_rows = []

        # Reset the branches sub-page nav to just Refresh + Back.
        keep_labels = {"Refresh", "Back to Developer"}
        self._branches_nav_buttons = [
            e for e in self._branches_nav_buttons if e[1] in keep_labels
        ]

        # Fetch with --prune so origin/<x> refs accurately reflect the
        # current remote. Local branches are NEVER pruned by this call;
        # local-only branches stay and are labeled "(old)" below.
        self._dev_run_git(['fetch', '--prune', '--quiet'], timeout=30)

        # List local branches
        rc, out, _err = self._dev_run_git([
            'for-each-ref', '--format=%(refname:short)', 'refs/heads/'
        ])
        local_branches = [b for b in (out or '').splitlines() if b]
        local_set = set(local_branches)

        # List remote branches; strip the "origin/" prefix and skip the
        # symbolic origin/HEAD entry.
        rc, out, _err = self._dev_run_git([
            'for-each-ref', '--format=%(refname:short)', 'refs/remotes/origin/'
        ])
        remote_refs = [b for b in (out or '').splitlines() if b]
        remote_branches = [
            r[len('origin/'):] for r in remote_refs
            if r.startswith('origin/') and not r.startswith('origin/HEAD')
        ]
        remote_set = set(remote_branches)

        # Auto-create a local tracking branch for any origin branch we
        # don't have yet. `git branch <name> origin/<name>` creates the
        # ref without touching the working tree, so it's safe even with
        # uncommitted work on the current branch.
        for rb in remote_branches:
            if rb not in local_set:
                self._dev_run_git(['branch', rb, f'origin/{rb}'])

        # Re-list locals so the newly-created tracking branches show up.
        rc, out, _err = self._dev_run_git([
            'for-each-ref', '--format=%(refname:short)', 'refs/heads/'
        ])
        branches = [b for b in (out or '').splitlines() if b]

        rc_cur, current, _ = self._dev_run_git(['branch', '--show-current'])
        current = current if rc_cur == 0 else ''

        on_style = (
            self._dev_btn_style
            .replace("border: 1px solid #555", "border: 1px solid #0f0")
            .replace("color: #dcdcdc", "color: #0f0")
        )
        # "no-GUI" branches are visually disabled — dim text + no hover.
        nogui_style = (
            self._dev_btn_style
            .replace("color: #dcdcdc", "color: #555")
        )

        ab_style = (
            "border: none; color: #aaa; font-size: 10px;"
            " font-family: monospace;"
        )
        ab_warn_style = (
            "border: none; color: #f44; font-size: 10px;"
            " font-family: monospace;"
        )
        ab_old_style = (
            "border: none; color: #ff0; font-size: 10px;"
            " font-family: monospace;"
        )

        # Rebuild order: [Refresh, <branches…>, Back to Developer].
        head_entries = [
            e for e in self._branches_nav_buttons if e[1] != "Back to Developer"
        ]
        exit_entry = next(
            (e for e in self._branches_nav_buttons if e[1] == "Back to Developer"),
            None,
        )

        T = self._translate_to_theme
        new_branch_entries = []
        for i, branch in enumerate(branches):
            has_gui = self._branch_has_gui(branch)
            is_old = branch not in remote_set
            btn = QPushButton(branch)
            if branch == current:
                style = on_style
            elif not has_gui:
                style = nogui_style
            else:
                style = self._dev_btn_style
            btn.setStyleSheet(T(style))
            btn.setFocusPolicy(Qt.NoFocus)
            if has_gui:
                btn.clicked.connect(
                    lambda checked=False, b=branch: self._dev_switch_branch(b)
                )
            else:
                btn.setEnabled(False)
                btn.setToolTip(
                    "This branch's tree does not contain the GUI source. "
                    "Switching would leave the running GUI without code."
                )
            self._dev_branch_grid.addWidget(btn, i, 0)

            # Right column: "(old)" for local-only, "no-gui" for missing GUI,
            # otherwise ahead/behind versus origin.
            if is_old:
                ab_text = "(old)"
                ab_lbl_style = ab_old_style
            else:
                ab_text = self._dev_ahead_behind(branch)
                ab_lbl_style = ab_style
            if not has_gui:
                ab_text = (ab_text + " no-gui").strip()
                ab_lbl_style = ab_warn_style
            ab_lbl = QLabel(ab_text)
            ab_lbl.setStyleSheet(T(ab_lbl_style))
            self._dev_branch_grid.addWidget(ab_lbl, i, 1)

            self._dev_branch_rows.append((btn, ab_lbl))
            # Only nav-add branches that are actually clickable.
            if has_gui:
                new_branch_entries.append((btn, branch, style))

        self._dev_branch_grid.setColumnStretch(0, 0)
        self._dev_branch_grid.setColumnStretch(1, 1)

        self._branches_nav_buttons = head_entries + new_branch_entries + (
            [exit_entry] if exit_entry else []
        )

    def _dev_ahead_behind(self, branch):
        rc, out, _err = self._dev_run_git([
            'rev-list', '--left-right', '--count',
            f'{branch}...origin/{branch}',
        ])
        if rc != 0 or not out:
            return "(no upstream)"
        try:
            ahead_str, behind_str = out.split()
            ahead, behind = int(ahead_str), int(behind_str)
        except ValueError:
            return ""
        if ahead == 0 and behind == 0:
            return "in sync"
        parts = []
        if ahead:
            parts.append(f"↑{ahead}")
        if behind:
            parts.append(f"↓{behind}")
        return " ".join(parts)

    def _dev_switch_branch(self, branch):
        rc_cur, current, _ = self._dev_run_git(['branch', '--show-current'])
        if rc_cur == 0 and current == branch:
            self._dev_set_status(f"Already on {branch}", color='#888')
            return

        # Defense-in-depth: never switch into a branch that lacks the GUI
        # source, even if a stale UI button somehow lets the click through.
        if not self._branch_has_gui(branch):
            self._dev_set_status(
                f"Refused: {branch} has no GUI source.", color='#f44'
            )
            return

        # Auto-stash if dirty so the user's in-progress edits follow them.
        rc_st, st_out, _ = self._dev_run_git(['status', '--porcelain'])
        stashed = False
        if rc_st == 0 and st_out:
            self._dev_set_status("Stashing local changes…", color='#ff0')
            QApplication.processEvents()
            rc_s, _o, err_s = self._dev_run_git(
                ['stash', 'push', '-u', '-m', 'auto-stash from GUI']
            )
            if rc_s != 0:
                self._dev_set_status(
                    f"Stash failed: {(err_s or 'unknown').splitlines()[-1]}",
                    color='#f44',
                )
                return
            stashed = True

        self._dev_set_status(f"Switching to {branch}…", color='#ff0')
        QApplication.processEvents()
        rc_sw, _o, err_sw = self._dev_run_git(['switch', branch])
        if rc_sw != 0:
            tail = (err_sw or 'unknown error').splitlines()[-1]
            self._dev_set_status(f"Switch failed: {tail}", color='#f44')
            if stashed:
                self._dev_run_git(['stash', 'pop'])
            return

        if stashed:
            rc_p, _o, err_p = self._dev_run_git(['stash', 'pop'])
            if rc_p != 0:
                self._dev_set_status(
                    f"On {branch}; stash pop conflict — resolve manually",
                    color='#ff0',
                )
            else:
                self._dev_set_status(
                    f"On {branch}; stashed changes restored", color='#0f0'
                )
        else:
            self._dev_set_status(f"On {branch}", color='#0f0')

        self._dev_update_branch_label()
        self._dev_refresh_branches()

    def _dev_toggle_container(self):
        if self._dev_container_running:
            self._dev_set_status(
                f"Stopping container {self._container_name}…", color='#ff0'
            )
            QApplication.processEvents()
            try:
                r = subprocess.run(
                    ['docker', 'stop', self._container_name],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    self._dev_set_status("Container stopped", color='#0f0')
                else:
                    msg = (r.stderr or r.stdout or 'unknown').splitlines()[-1]
                    self._dev_set_status(f"Stop failed: {msg}", color='#f44')
            except Exception as e:
                self._dev_set_status(f"Stop failed: {e}", color='#f44')
        else:
            if not os.path.isfile(self._dev_run_script):
                self._dev_set_status(
                    f"Missing script: {self._dev_run_script}", color='#f44'
                )
                return
            self._dev_set_status(
                f"Starting container {self._container_name}…", color='#ff0'
            )
            QApplication.processEvents()
            try:
                # Detached run-container.sh; --no-attach keeps it from
                # waiting for an interactive shell.
                subprocess.Popen(
                    ['bash', self._dev_run_script, '--no-attach'],
                    cwd=self._dev_host_repo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._dev_set_status(
                    "Container start initiated", color='#0f0'
                )
            except Exception as e:
                self._dev_set_status(f"Start failed: {e}", color='#f44')
        # Status will update on the next 1s poll
        self._dev_update_container_status()

    def _dev_update_container_status(self):
        try:
            r = subprocess.run(
                ['docker', 'ps', '--quiet', '--filter', 'status=running',
                 '--filter', f'name=^/{self._container_name}$'],
                capture_output=True, text=True, timeout=3,
            )
            running = bool(r.stdout.strip())
        except Exception:
            running = False
        self._dev_container_running = running
        T = self._translate_to_theme
        if running:
            self._dev_container_dot.setStyleSheet(T(
                "background-color: #4f4; border-radius: 5px; border: none;"
            ))
            desired = "Stop Container"
        else:
            self._dev_container_dot.setStyleSheet(T(
                "background-color: #f44; border-radius: 5px; border: none;"
            ))
            desired = "Start Container"
        # Sync the base_label stored in _dev_nav_buttons so the next
        # _update_selection tick doesn't revert to a stale label and cause
        # the button to flash between "Start Container" and the selection-
        # wrapped form like "> Container <". Gate on actual state change so
        # we don't restyle every nav button every poll.
        if getattr(self, '_dev_container_btn_label', None) != desired:
            self._dev_container_btn_label = desired
            self._set_btn_label(self._dev_container_btn, desired)

    # ── Quick action: stop → pull → start → connect → build ──────────
    # Runs the whole chain in a worker thread so the multi-minute
    # colcon build doesn't freeze the GUI event loop. Status updates
    # come back to the UI thread via QTimer.singleShot(0, ...) so the
    # _dev_set_status label stays in sync without subclassing QThread.
    _DEV_REBUILD_LABEL = "Load branch changes and build"

    def _dev_rebuild_self_heal(self):
        """Detect a stale 'running' state and reset it.

        ``_dev_full_rebuild_running`` is cleared by the worker thread's
        ``finally`` block. That ``finally`` does not run when the
        worker thread dies abnormally — a hung ``subprocess.run`` /
        ``proc.wait()``, the main process being force-killed mid-build,
        the daemon worker being torn down by the OS on GUI close, etc.
        When that happens the flag is stuck True forever: the button
        is permanently disabled and the status label is permanently
        "Starting…", even though nothing is actually running.

        We use the worker Thread's own ``is_alive()`` as the source of
        truth — if a build was supposedly in flight but the thread
        isn't alive, the state is by definition stale. Returns True
        if it had to clean up.
        """
        if not getattr(self, '_dev_full_rebuild_running', False):
            return False
        t = getattr(self, '_dev_full_rebuild_thread', None)
        if t is not None and t.is_alive():
            return False  # genuinely running, leave it alone
        # Stale — the worker died without clearing the flag.
        self._dev_full_rebuild_running = False
        if hasattr(self, '_dev_full_rebuild_btn'):
            self._dev_full_rebuild_btn.setEnabled(True)
        # Clear the misleading "Starting…" status so the user knows
        # the GUI noticed.
        self._dev_set_status(
            "Previous build state was stale — reset.", color='#888')
        return True

    def _dev_full_rebuild(self):
        # Self-heal first: if the previous run died abnormally, this
        # click should be treated as fresh, not silently ignored.
        self._dev_rebuild_self_heal()
        if getattr(self, '_dev_full_rebuild_running', False):
            return  # genuinely in flight — ignore the click
        self._dev_full_rebuild_running = True
        self._dev_full_rebuild_btn.setEnabled(False)
        # Allocate the per-process terminal buffer the way every other
        # launched device does, switch the terminal selection to it,
        # and refresh. The worker thread streams every step's output
        # into this buffer so the user sees the colcon build line by
        # line in the side terminal instead of one delayed dump at the
        # end (which is what subprocess.run with capture_output would
        # give us).
        label = self._DEV_REBUILD_LABEL
        buf = [f"=== {label} ===\n"]
        self._process_buffers[label] = buf
        self._selected_process = label
        self._term_last_text = ''
        self._refresh_terminal_display()
        self._dev_set_status("Starting…", color='#ff0')
        # Bounce back to the main options screen — pressing this button
        # is the operator saying "get the robot ready"; they want to be
        # back on the launch screen so the streaming colcon build is
        # visible in the side terminal while they wait.
        self._show_main_page()
        QApplication.processEvents()
        # Stash the worker Thread so _dev_rebuild_self_heal can use
        # is_alive() to detect that a previous build died without
        # firing its finally block.
        self._dev_full_rebuild_thread = threading.Thread(
            target=self._dev_full_rebuild_worker,
            args=(label, buf),
            daemon=True,
        )
        self._dev_full_rebuild_thread.start()

    def _dev_ui_status(self, text, color='#888'):
        """Thread-safe wrapper around _dev_set_status. Schedules the
        actual setText call on the Qt event loop so worker threads
        don't touch QLabel widgets directly."""
        QTimer.singleShot(
            0, lambda t=text, c=color: self._dev_set_status(t, color=c))

    def _dev_full_rebuild_worker(self, label, buf):
        """Background chain for the Load-branch-changes-and-build
        button. Every step's stdout/stderr is appended to ``buf`` so
        the side terminal (which reads ``_process_buffers[label]``)
        shows the build live, line by line, the same way every other
        launched device does.
        """
        def log(text):
            """Append text to the per-process buffer with a trailing
            newline. Safe to call from this worker thread — list.append
            is atomic in CPython and the UI thread only reads."""
            if not text:
                return
            if not text.endswith('\n'):
                text += '\n'
            buf.append(text)

        try:
            # 0. Tear down every running launch via the GUI's normal
            # stop path. Without this, buttons stayed green after a
            # rebuild because the inner ROS nodes died with the
            # container faster than the GUI noticed. _toggle_device
            # touches Qt widgets, so schedule on the UI thread.
            running_labels = [
                L for L, st in self._launch_states.items() if st
            ]
            self._dev_ui_status(
                f"[0/5] Tearing down {len(running_labels)} launches…",
                color='#ff0')
            log(f"Tearing down launches: {running_labels}")
            for L in running_labels:
                QTimer.singleShot(0, lambda lab=L: self._toggle_device(lab))
            time.sleep(1.5)  # let the UI-thread teardowns drain

            # 1. Stop container (no-op if already stopped).
            self._dev_ui_status(
                f"[1/5] Stopping container {self._container_name}…",
                color='#ff0')
            log(f"$ docker stop {self._container_name}")
            try:
                r = subprocess.run(
                    ['docker', 'stop', self._container_name],
                    capture_output=True, text=True, timeout=30,
                )
                log(r.stdout)
                log(r.stderr)
            except Exception as e:
                # Non-fatal — the container may already be down. The
                # start step below will surface real issues.
                log(f"[stop] {e}")

            # 2. git pull --ff-only on the host repo.
            self._dev_ui_status(
                "[2/5] git pull --ff-only…", color='#ff0')
            log("$ git pull --ff-only")
            rc, out, err = self._dev_run_git(
                ['pull', '--ff-only'], timeout=120)
            log(out)
            log(err)
            if rc != 0:
                msg = (err or out or 'unknown').splitlines()[-1]
                self._dev_ui_status(
                    f"Pull failed: {msg}", color='#f44')
                log(f"[!] Pull failed: {msg}")
                return

            # 3. Start container detached.
            self._dev_ui_status(
                f"[3/5] Starting container {self._container_name}…",
                color='#ff0')
            log(f"$ bash {self._dev_run_script} --no-attach")
            if not os.path.isfile(self._dev_run_script):
                self._dev_ui_status(
                    f"Missing script: {self._dev_run_script}",
                    color='#f44')
                log(f"[!] Missing script: {self._dev_run_script}")
                return
            try:
                subprocess.Popen(
                    ['bash', self._dev_run_script, '--no-attach'],
                    cwd=self._dev_host_repo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                self._dev_ui_status(
                    f"Start failed: {e}", color='#f44')
                log(f"[!] Start failed: {e}")
                return

            # 4. Wait until docker reports the container running.
            #    The 1 s poll loop matches _dev_update_container_status.
            self._dev_ui_status(
                "[4/5] Waiting for container to come up…",
                color='#ff0')
            log("Waiting for container to come up (60 s ceiling)…")
            ready = False
            for _ in range(60):
                try:
                    r = subprocess.run(
                        ['docker', 'ps', '--quiet', '--filter',
                         'status=running', '--filter',
                         f'name=^/{self._container_name}$'],
                        capture_output=True, text=True, timeout=3,
                    )
                    if r.stdout.strip():
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            if not ready:
                self._dev_ui_status(
                    "Container did not come up within 60 s.",
                    color='#f44')
                log("[!] Container did not come up within 60 s.")
                return
            log("Container is up.")
            # Connect on the UI thread (touches widgets + state).
            QTimer.singleShot(0, self._connect_container)

            # 5. colcon build — STREAMING. Popen with PIPE + an inline
            # reader feeds each line into the per-process buffer as it
            # arrives, so the side terminal shows progress live. We
            # also register the Popen in _process_objects[label] so
            # the GUI's 2 Hz poll timer cleans up the entry and writes
            # the "[Process exited with code N]" trailer the way it
            # does for any other launched device. closeEvent will
            # SIGTERM/SIGKILL this Popen if the user closes the GUI
            # mid-build (the local docker-exec proc forwards signals
            # to the in-container colcon).
            self._dev_ui_status(
                "[5/5] colcon build --symlink-install (this can "
                "take several minutes)…", color='#ff0')
            build_cmd = (
                "cd /autonav/isaac_ros-dev && "
                "source /opt/ros/humble/setup.bash && "
                "colcon build --symlink-install"
            )
            log(
                f"$ docker exec -u admin {self._container_name} "
                f"/bin/bash -lc '{build_cmd}'"
            )
            try:
                proc = subprocess.Popen(
                    ['docker', 'exec', '-u', 'admin',
                     self._container_name, '/bin/bash', '-lc',
                     build_cmd],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except Exception as e:
                self._dev_ui_status(
                    f"Build launch failed: {e}", color='#f44')
                log(f"[!] Build launch failed: {e}")
                return
            self._process_objects[label] = proc
            try:
                for line in proc.stdout:
                    buf.append(line)
                proc.wait()
            except Exception as e:
                log(f"[reader] {e}")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            rc = proc.returncode
            if rc == 0:
                self._dev_ui_status(
                    "Load branch changes and build: OK",
                    color='#0f0')
                log("[OK] colcon build finished cleanly")
            else:
                self._dev_ui_status(
                    f"Build failed (rc={rc})", color='#f44')
                log(f"[!] Build failed (rc={rc})")
        finally:
            # Re-enable the button on the UI thread no matter how the
            # chain exited — failure paths must not leave it disabled.
            def _reset():
                self._dev_full_rebuild_running = False
                self._dev_full_rebuild_btn.setEnabled(True)
            QTimer.singleShot(0, _reset)

    def _toggle_test(self, tid):
        """Start or stop a test by its ID."""
        if self._active_test == tid:
            # Stop the running test
            self._stop_test(tid)
        else:
            # Stop any other running test first
            if self._active_test is not None:
                self._stop_test(self._active_test)
            self._start_test(tid)

    def _start_test(self, tid):
        """Launch a test subprocess."""
        tdef = None
        for d in self._test_defs:
            if d[0] == tid:
                tdef = d
                break
        if tdef is None:
            return
        tid, title, cmd, _desc = tdef

        self._gui_log_msg(f"Starting test {tid}: {title}")
        self._active_test = tid
        self._test_status_label.setText(f"Status: running {tid}")

        # Highlight the active test button green
        for btn, label, base_style in self._test_nav_buttons:
            if label.startswith(tid):
                btn.setStyleSheet(self._test_on_style)
                # Update nav list entry
                for i, (b, l, _s) in enumerate(self._test_nav_buttons):
                    if b is btn:
                        self._test_nav_buttons[i] = (b, l, self._test_on_style)
                        break
                break

        # Launch subprocess (wrapped for container if connected)
        test_label = f"test:{tid}"
        exec_cmd = self._wrap_container_cmd(cmd, label=test_label)
        buf = []
        self._process_buffers[test_label] = buf
        try:
            proc = subprocess.Popen(
                exec_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self._process_objects[f"test:{tid}"] = proc

            def _reader():
                try:
                    for line in proc.stdout:
                        buf.append(line)
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            self._process_readers[f"test:{tid}"] = t
        except Exception as e:
            buf.append(f"[ERROR] Failed to launch: {e}\n")
            self._gui_log_msg(f"Test {tid} failed to launch: {e}")

        self._update_selection()

    def _stop_test(self, tid):
        """Stop a running test."""
        self._gui_log_msg(f"Stopping test {tid}")
        key = f"test:{tid}"
        proc = self._process_objects.pop(key, None)
        if proc is not None:
            self._kill_process(proc, key)
        buf = self._process_buffers.get(key)
        if buf is not None:
            buf.append("[Test stopped by user]\n")
        self._process_readers.pop(key, None)

        self._active_test = None
        self._test_status_label.setText("Status: idle")

        # Restore button style
        for btn, label, base_style in self._test_nav_buttons:
            if label.startswith(tid):
                btn.setStyleSheet(self._test_btn_style)
                for i, (b, l, _s) in enumerate(self._test_nav_buttons):
                    if b is btn:
                        self._test_nav_buttons[i] = (b, l, self._test_btn_style)
                        break
                break

        self._update_selection()
        self._refresh_terminal_display()

    def _on_estop(self):
        """E-STOP: stop active test and publish estop if ROS available."""
        self._gui_log_msg("E-STOP activated!")
        if self._active_test is not None:
            self._stop_test(self._active_test)
        # Publish to /estop topic if ROS node exists
        node = self._ros_node
        if node is not None and hasattr(node, '_estop_pub'):
            from std_msgs.msg import String
            msg = String()
            msg.data = "STOP"
            node._estop_pub.publish(msg)

    _DOT_ON = "background-color: #0f0; border-radius: 7px; border: none;"
    _DOT_OFF = "background-color: #555; border-radius: 7px; border: none;"

    # Some ROS2 nodes are launched via an executable whose binary name
    # doesn't match the node name (e.g., sick_scansegment_xd runs inside
    # the sick_generic_caller binary with no __node:= remap). Map them
    # here so /proc/*/cmdline matching still finds the right process.
    _NODE_EXEC_ALIASES = {
        'sick_scansegment_xd': ('sick_generic_caller',),
    }
    # Use 6-char #ffff00 (NOT #ff0) so the regex walker leaves it alone in
    # light mode — the user wants the process yellow dots to stay yellow,
    # while #ff0 used for status text still translates to dark orange.
    _DOT_YELLOW = "background-color: #ffff00; border-radius: 7px; border: none;"

    def _dot_keys_for(self, label):
        """Return the status-dot keys for a device label."""
        for dev_label, keys, _cmd in self._launch_devices:
            if dev_label == label:
                return keys
        return []

    def _load_watched_topics(self):
        """Load watched_topics from the package's config YAML.

        Prefers ament_index lookup so an installed colcon build is found;
        falls back to a source-relative path so dev runs from the source
        tree still work.
        """
        if not _HAS_YAML:
            return []
        path = None
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('autonav_gui_hud')
            cand = os.path.join(share, 'config', 'watched_topics.yaml')
            if os.path.isfile(cand):
                path = cand
        except Exception:
            pass
        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.normpath(
                os.path.join(here, '..', 'config', 'watched_topics.yaml')
            )
            if os.path.isfile(cand):
                path = cand
        if path is None:
            return []
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            topics = data.get('watched_topics', []) or []
            return [t for t in topics if isinstance(t, str)]
        except Exception:
            return []

    def _base_dot_style_for(self, dot_key):
        """Correct base style for a dot from current launch state."""
        dev_label = self._dot_to_device.get(dot_key)
        if dev_label is None:
            return self._DOT_OFF
        st = self._launch_states.get(dev_label, False)
        if st == 'starting':
            return self._DOT_YELLOW
        if st is True:
            return self._DOT_ON
        return self._DOT_OFF

    def _snapshot_proc_cmdlines(self):
        """Read every /proc/<pid>/cmdline once per poll.

        Returns {pid: cmdline_bytes}. Cheaper than re-scanning /proc for
        each node-name lookup.
        """
        out = {}
        try:
            entries = os.listdir('/proc')
        except OSError:
            return out
        for ent in entries:
            if not ent.isdigit():
                continue
            try:
                pid = int(ent)
                with open(f'/proc/{ent}/cmdline', 'rb') as f:
                    out[pid] = f.read()
            except (OSError, ValueError):
                continue
        return out

    def _find_pids_for_node(self, short_name, full_name, proc_map):
        """All PIDs whose cmdline matches a ROS2 node name.

        Returns a list (possibly empty). ROS2 Humble doesn't expose
        participant PIDs via the graph API, so we scrape argv. When
        multiple processes share a node name (e.g., two wheel_odom from
        two pre_slam launches) all are returned so the caller can
        attribute one publisher endpoint per distinct PID.
        """
        needles = [n.encode() for n in (short_name, full_name) if n]
        for alias in self._NODE_EXEC_ALIASES.get(short_name, ()):
            needles.append(alias.encode())
        if not needles:
            return []
        return [
            pid for pid, blob in proc_map.items()
            if blob and any(n in blob for n in needles)
        ]

    def _gui_owner_for_pid(self, pid):
        """Walk PPid chain; return (label, root_pid) or (None, None).

        Bounded walk against the in-container ros2 launch PIDs the GUI
        writes to /tmp/gui_pid_<sanitized_label> before exec'ing the
        launch. The container runs with --pid=host so those PIDs are
        host PIDs too. We deliberately do NOT use popen.pid here: that's
        the host-side `docker exec` wrapper, which sits in a separate
        process tree (parented by containerd-shim, not by the caller)
        and is never an ancestor of the in-container ros2 launch.

        Returning the root PID lets duplicate detection key sources on
        the launch tree, not the publisher process — so two distinct
        publisher nodes from the same launch (e.g. Nav2's behavior_server
        and velocity_smoother both publishing /cmd_vel) collapse to one
        source instead of falsely flagging the device as duplicated.
        """
        if pid is None:
            return (None, None)
        owned = {}
        for lbl, popen in self._process_objects.items():
            if popen is None:
                continue
            try:
                if popen.poll() is not None:
                    continue
            except Exception:
                continue
            pid_tag = lbl.replace(' ', '_').replace('/', '_')
            try:
                with open(f'/tmp/gui_pid_{pid_tag}', 'r') as f:
                    launch_pid = int(f.read().strip())
            except (OSError, ValueError):
                continue
            try:
                os.stat(f'/proc/{launch_pid}')
            except OSError:
                continue
            owned[launch_pid] = lbl
        cur = pid
        for _ in range(64):
            if cur in owned:
                return (owned[cur], cur)
            try:
                with open(f'/proc/{cur}/status', 'r') as f:
                    ppid = None
                    for line in f:
                        if line.startswith('PPid:'):
                            ppid = int(line.split()[1])
                            break
                if ppid is None or ppid <= 1:
                    return (None, None)
                cur = ppid
            except (OSError, ValueError):
                return (None, None)
        return (None, None)

    def _poll_watched_publishers(self):
        """0.1 Hz: attribute every watched-topic publisher to a process.

        For each topic, group endpoints by (namespace, node_name). When
        multiple endpoints share a node name they could come from one
        process publishing several internal endpoints (e.g., Nav2's
        behavior_server creates ~6 on /cmd_vel) OR from several distinct
        processes that happen to share a name (the bug case: two
        pre_slam launches → two wheel_odom processes). We disambiguate
        by /proc scan: if N endpoints share a name and we find N distinct
        matching processes, emit N rows; if fewer processes match, emit
        one row per process (collapsing endpoints from the same PID).

        ``node_to_pids`` carries already-attributed PIDs across topics
        so a node that legitimately publishes to multiple watched topics
        (e.g., ekf_node → /odom + /odometry/filtered) reuses the same
        PID for both rather than skipping it as "claimed".
        """
        node = self._ros_node
        if node is None or not self._watched_topics:
            return

        proc_map = self._snapshot_proc_cmdlines()
        state = {}
        node_to_pids = {}  # (ns, name) -> list of PIDs already attributed

        for topic in self._watched_topics:
            try:
                infos = node.get_publishers_info_by_topic(topic)
            except Exception:
                continue

            groups = {}  # (ns, name, full) -> endpoint count
            for info in infos:
                ns = getattr(info, 'node_namespace', '') or ''
                name = getattr(info, 'node_name', '') or ''
                if ns and not ns.endswith('/'):
                    ns = ns + '/'
                full = (ns + name) if name else (ns or '?')
                full = full.replace('//', '/')
                key = (ns, name, full)
                groups[key] = groups.get(key, 0) + 1

            entries = []
            for (ns, name, full), count in groups.items():
                already = node_to_pids.get((ns, name), [])
                if len(already) >= count:
                    pids_for_topic = already[:count]
                else:
                    all_matches = self._find_pids_for_node(name, full, proc_map)
                    extra = [p for p in all_matches if p not in already]
                    take = max(0, count - len(already))
                    already = already + extra[:take]
                    node_to_pids[(ns, name)] = already
                    pids_for_topic = already[:count]

                if pids_for_topic:
                    for pid in pids_for_topic:
                        owner, owner_root = self._gui_owner_for_pid(pid)
                        entries.append({
                            'node': full,
                            'topic': topic,
                            'pid': pid,
                            'gui_owner': owner,
                            'gui_owner_root': owner_root,
                        })
                else:
                    entries.append({
                        'node': full,
                        'topic': topic,
                        'pid': None,
                        'gui_owner': None,
                        'gui_owner_root': None,
                    })
            state[topic] = entries

        self._watched_pub_state = state
        self._apply_duplicate_dots(state)
        self._refresh_process_table()

    def _apply_duplicate_dots(self, state):
        """Flip dots red when their device participates in a topic collision.

        Sources are keyed by the launch ROOT PID for GUI-owned publishers
        (the value from /tmp/gui_pid_<label>), not the publisher's own
        PID. Two publisher nodes spawned by the same launch (e.g. Nav2's
        behavior_server + velocity_smoother both on /cmd_vel) descend
        from the same root, share one source tuple, and don't trip a
        false duplicate. Two launches of the same device (real duplicate)
        have different roots and DO trip it.
        """
        flagged = set()
        for _topic, entries in state.items():
            sources = set()
            for e in entries:
                pid = e['pid']
                owner = e['gui_owner']
                owner_root = e.get('gui_owner_root')
                if owner is not None:
                    sources.add(('gui', owner, owner_root))
                elif pid is not None:
                    sources.add(('ext', pid))
                else:
                    sources.add(('unknown', e['node']))
            if len(sources) <= 1:
                continue
            for e in entries:
                owner = e['gui_owner']
                if owner is None:
                    continue
                for dev_label, dot_keys, _cmd in self._launch_devices:
                    if dev_label == owner:
                        flagged.update(dot_keys)
                        break

        for k in flagged - self._dot_duplicate_flagged:
            dot = self.status_dots.get(k)
            if dot is not None:
                dot.setStyleSheet(
                    "background-color: #f44; border-radius: 7px; border: none;"
                )
        for k in self._dot_duplicate_flagged - flagged:
            dot = self.status_dots.get(k)
            if dot is not None:
                dot.setStyleSheet(self._base_dot_style_for(k))
        self._dot_duplicate_flagged = flagged

    def _refresh_process_table(self):
        """Hook invoked from _poll_watched_publishers each 0.1 Hz tick."""
        if not hasattr(self, '_proc_table'):
            return
        self._render_process_table()

    def _render_process_table(self):
        """Repopulate the Manage Running Processes grid, aggregated by PID.

        Columns: [PID | Source | Topics | Action]. Each process gets one
        row; the Topics cell lists every watched topic that PID publishes
        to. Unknown-PID publishers cluster by node name so they don't
        collapse together. Kill button only on GUI-owned rows.
        """
        if not hasattr(self, '_proc_table'):
            return

        while self._proc_table.count():
            item = self._proc_table.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        cell_style = (
            "border: none; color: #ccc; font-size: 10px;"
            " font-family: monospace;"
        )
        foreign_style = (
            "border: none; color: #f80; font-size: 10px;"
            " font-family: monospace;"
        )
        kill_style = (
            "QPushButton {"
            "  background-color: #4a1a1a; color: #fbb;"
            "  border: 1px solid #722; border-radius: 3px;"
            "  padding: 2px 8px; font-size: 10px;"
            "}"
            "QPushButton:hover { background-color: #6a2a2a; }"
            "QPushButton:pressed { background-color: #300a0a; }"
        )

        # Aggregate per process. Resolved PIDs collapse cleanly; unresolved
        # publishers cluster by node name as a fallback so distinct unknown
        # processes don't merge into a single row.
        agg = {}  # key -> {pid, node, gui_owner, topics: set}
        for topic in self._watched_topics:
            for e in self._watched_pub_state.get(topic, []):
                pid = e.get('pid')
                node = e.get('node', '?')
                owner = e.get('gui_owner')
                key = pid if pid is not None else ('unk', node)
                if key not in agg:
                    agg[key] = {
                        'pid': pid,
                        'node': node,
                        'gui_owner': owner,
                        'topics': set(),
                    }
                agg[key]['topics'].add(e.get('topic', topic))

        # Sort: GUI-owned first, then by numeric PID, then unknowns by node.
        def sort_key(item):
            _key, v = item
            owner = v.get('gui_owner')
            pid = v.get('pid')
            return (
                0 if owner is not None else 1,
                pid if pid is not None else 10**9,
                v.get('node', ''),
            )

        # Fixed widths for PID / Source / Action so Topics absorbs slack.
        # Without these caps a long unknown-node fallback (e.g. "? /…")
        # forces the row wider than the viewport and we have to scroll
        # horizontally — operator can't read the table.
        col_pid_width = 110
        col_source_width = 150
        col_action_width = 60

        row = 0
        for _key, v in sorted(agg.items(), key=sort_key):
            owner = v['gui_owner']
            pid = v['pid']
            pid_text = str(pid) if pid is not None else f"?  {v['node']}"
            pid_lbl = QLabel(pid_text)
            pid_lbl.setStyleSheet(
                cell_style if owner is not None else foreign_style
            )
            pid_lbl.setWordWrap(True)
            pid_lbl.setMaximumWidth(col_pid_width)
            self._proc_table.addWidget(pid_lbl, row, 0)

            if owner is not None:
                source_lbl = QLabel(f"GUI: {owner}")
                source_lbl.setStyleSheet(cell_style)
            else:
                source_lbl = QLabel("External")
                source_lbl.setStyleSheet(foreign_style)
            source_lbl.setWordWrap(True)
            source_lbl.setMaximumWidth(col_source_width)
            self._proc_table.addWidget(source_lbl, row, 1)

            topics_text = ", ".join(sorted(v['topics']))
            topics_lbl = QLabel(topics_text)
            topics_lbl.setStyleSheet(
                cell_style if owner is not None else foreign_style
            )
            topics_lbl.setWordWrap(True)
            self._proc_table.addWidget(topics_lbl, row, 2)

            if owner is not None:
                kill_btn = QPushButton("Kill")
                kill_btn.setStyleSheet(kill_style)
                kill_btn.setFocusPolicy(Qt.NoFocus)
                kill_btn.setFixedWidth(col_action_width)
                kill_btn.clicked.connect(
                    lambda _checked=False, lbl=owner:
                        self._kill_gui_process(lbl)
                )
                self._proc_table.addWidget(kill_btn, row, 3)
            row += 1

        # Pin column widths so header and body line up.
        self._proc_table.setColumnMinimumWidth(0, col_pid_width)
        self._proc_table.setColumnMinimumWidth(1, col_source_width)
        self._proc_table.setColumnMinimumWidth(3, col_action_width)
        self._proc_table.setColumnStretch(0, 0)
        self._proc_table.setColumnStretch(1, 0)
        self._proc_table.setColumnStretch(3, 0)

        if hasattr(self, '_proc_empty_label'):
            self._proc_empty_label.setVisible(row == 0)

    def _kill_gui_process(self, device_label):
        """Terminate a GUI-spawned process by its device label.

        Reuses the existing toggle path so launch state, dot, and queue
        all reset cleanly — _toggle_device(label) when label is already
        on / starting routes to the stop branch.

        Optimistically drops the killed device's rows from the table so
        the user sees the change immediately instead of waiting up to
        10 s for the next graph poll. DDS may still report the dying
        publisher for a beat; the next poll reconciles authoritatively.
        """
        if device_label not in self._launch_states:
            return
        try:
            self._toggle_device(device_label)
        except Exception as ex:
            self._gui_log_msg(f"Failed to stop {device_label}: {ex}")
            return
        self._watched_pub_state = {
            topic: [e for e in entries if e.get('gui_owner') != device_label]
            for topic, entries in self._watched_pub_state.items()
        }
        self._render_process_table()

    def _update_queue_label(self):
        """Refresh the queue status text at the top of the launch page."""
        # Find what's currently starting
        starting = None
        for lbl, st in self._launch_states.items():
            if st == 'starting':
                starting = lbl
                break
        parts = []
        if starting:
            parts.append(f"Starting: {starting}")
        if self._launch_queue:
            q_str = " > ".join(self._launch_queue)
            parts.append(f"Queued: {q_str}")
        if parts:
            self._queue_label.setText(" | ".join(parts))
            self._queue_label.setStyleSheet(
                "border: none; color: #ff0; font-size: 10px;"
                " font-family: monospace;"
            )
        else:
            self._queue_label.setText("Queue: idle")
            self._queue_label.setStyleSheet(
                "border: none; color: #888; font-size: 10px;"
                " font-family: monospace;"
            )

    def _toggle_device(self, label):
        """Toggle a device on/off. Uses a queue so only one starts at a time."""
        state = self._launch_states.get(label, False)

        # --- Turning off (from any state) ---
        if state:  # True or 'starting'
            self._gui_log_msg(f"Stopping: {label}")
            self._launch_states[label] = False
            dot_keys = self._dot_keys_for(label)
            for key in dot_keys:
                if key in self.status_dots:
                    self.status_dots[key].setStyleSheet(self._DOT_OFF)
            # Cancel timers and reset readiness state
            if label in self._startup_timers:
                self._startup_timers[label].stop()
                del self._startup_timers[label]
            self._startup_deadlines.pop(label, None)
            self._ready_events.pop(label, None)
            if label in self._flash_timers:
                self._flash_timers[label].stop()
                del self._flash_timers[label]
            # Terminate subprocess
            proc = self._process_objects.pop(label, None)
            if proc is not None:
                self._kill_process(proc, label)
                buf = self._process_buffers.get(label)
                if buf is not None:
                    buf.append("[Process terminated by user]\n")
                self._refresh_terminal_display()
            self._process_readers.pop(label, None)
            # Remove from queue if queued
            if label in self._launch_queue:
                self._launch_queue.remove(label)
            # Restore button label and style
            for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
                if blabel == label:
                    btn.setText(label)
                    self._launch_nav_buttons[i] = (btn, blabel, self._launch_btn_style)
                    btn.setStyleSheet(self._launch_btn_style)
                    break
            self._update_selection()
            self._update_queue_label()
            # If we cancelled an active startup, process next in queue
            if state == 'starting':
                self._process_queue()
            return

        # --- Turning on ---
        # If something is already starting, queue this one
        any_starting = any(
            s == 'starting' for s in self._launch_states.values()
        )
        if any_starting:
            if label not in self._launch_queue:
                self._launch_queue.append(label)
                # Show "Waiting" on the button
                for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
                    if blabel == label:
                        btn.setText("Waiting")
                        btn.setStyleSheet(self._launch_wait_style)
                        self._launch_nav_buttons[i] = (btn, blabel, self._launch_wait_style)
                        break
                self._update_selection()
            self._update_queue_label()
            return

        # Nothing starting — launch immediately
        self._start_device(label)

    def _start_device(self, label):
        """Begin the startup sequence: flash dots, launch subprocess, check after 1.5s."""
        self._gui_log_msg(f"Launching: {label}")
        dot_keys = self._dot_keys_for(label)
        self._launch_states[label] = 'starting'
        self._ready_events[label] = False
        timeout = self._ready_timeouts.get(label, self.DEFAULT_READY_TIMEOUT)
        self._startup_deadlines[label] = time.monotonic() + timeout
        # Restore button text (may have been "Waiting")
        for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
            if blabel == label:
                btn.setText(label)
                break
        self._update_queue_label()

        # Flash timer: alternate yellow/gray every 200ms
        flash_state = [True]

        def flash():
            if self._launch_states.get(label) != 'starting':
                return
            style = self._DOT_YELLOW if flash_state[0] else self._DOT_OFF
            for key in dot_keys:
                if key in self.status_dots:
                    self.status_dots[key].setStyleSheet(style)
            flash_state[0] = not flash_state[0]

        flash_timer = QTimer()
        flash_timer.setInterval(200)
        flash_timer.timeout.connect(flash)
        flash_timer.start()
        self._flash_timers[label] = flash_timer
        flash()

        # Find the command for this device
        cmd = None
        for dev_label, _keys, dev_cmd in self._launch_devices:
            if dev_label == label:
                cmd = dev_cmd
                break

        # Launch real subprocess (wrapped for container if connected)
        exec_cmd = self._wrap_container_cmd(cmd, label=label)
        buf = []
        self._process_buffers[label] = buf
        buf.append(f"$ {exec_cmd}\n")
        try:
            proc = subprocess.Popen(
                exec_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            self._process_objects[label] = proc

            # Daemon reader thread: appends stdout lines to buffer and
            # watches for the readiness sentinel printed by the launch script.
            sentinel = self.READY_SENTINEL
            ready_events = self._ready_events

            def _reader():
                try:
                    for line in proc.stdout:
                        buf.append(line)
                        if sentinel in line:
                            ready_events[label] = True
                except Exception:
                    pass
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            self._process_readers[label] = t
        except Exception as e:
            buf.append(f"[Failed to start: {e}]\n")
            self._finish_device_startup(label, success=False)
            return

        # Poll every 250ms: succeed when [GUI_READY] arrives on stdout,
        # fail when the process exits or the readiness deadline passes.
        startup_timer = QTimer()
        startup_timer.setInterval(250)
        startup_timer.timeout.connect(lambda: self._check_startup(label))
        startup_timer.start()
        self._startup_timers[label] = startup_timer

        self._update_selection()

    def _launch_all_in_sequence(self):
        """Queue all devices that aren't already on or starting."""
        for label, _keys, _cmd in self._launch_devices:
            state = self._launch_states.get(label, False)
            if not state:  # not on, not starting
                self._toggle_device(label)

    def _process_queue(self):
        """Start the next queued device if nothing is currently starting."""
        any_starting = any(
            s == 'starting' for s in self._launch_states.values()
        )
        if any_starting:
            self._update_queue_label()
            return
        while self._launch_queue:
            next_label = self._launch_queue.pop(0)
            # Skip if it was turned off while queued
            if self._launch_states.get(next_label, False):
                continue
            self._start_device(next_label)
            return
        self._update_queue_label()

    def _check_startup(self, label):
        """Polled every 250ms while a device is 'starting'.

        Outcomes:
          ready sentinel seen → green, advance queue.
          process exited     → red,   advance queue.
          deadline passed    → red,   advance queue (process keeps running
                                       so the user can inspect logs and
                                       turn it off manually).
        """
        if self._launch_states.get(label) != 'starting':
            self._stop_startup_timer(label)
            return

        # 1. readiness handshake fired
        if self._ready_events.get(label):
            self._stop_startup_timer(label)
            self._finish_device_startup(label, success=True)
            return

        # 2. process exited before signaling ready
        proc = self._process_objects.get(label)
        if proc is not None and proc.poll() is not None:
            buf = self._process_buffers.get(label)
            if buf is not None:
                buf.append(f"[Process exited with code {proc.returncode}]\n")
            self._process_objects.pop(label, None)
            self._process_readers.pop(label, None)
            self._stop_startup_timer(label)
            self._finish_device_startup(label, success=False)
            return

        # 3. readiness deadline passed
        deadline = self._startup_deadlines.get(label)
        if deadline is not None and time.monotonic() >= deadline:
            buf = self._process_buffers.get(label)
            if buf is not None:
                buf.append(f"[Readiness timeout — no {self.READY_SENTINEL} after "
                           f"{self._ready_timeouts.get(label, self.DEFAULT_READY_TIMEOUT):.0f}s]\n")
            self._stop_startup_timer(label)
            self._finish_device_startup(label, success=False)

    def _stop_startup_timer(self, label):
        timer = self._startup_timers.pop(label, None)
        if timer is not None:
            timer.stop()
        self._startup_deadlines.pop(label, None)

    def _finish_device_startup(self, label, success):
        """Stop flashing, set dots green or gray, update button style, process queue."""
        self._gui_log_msg(f"{'Started' if success else 'Failed'}: {label}")
        dot_keys = self._dot_keys_for(label)
        # Stop flash timer
        if label in self._flash_timers:
            self._flash_timers[label].stop()
            del self._flash_timers[label]
        if success:
            self._launch_states[label] = True
            for key in dot_keys:
                if key in self.status_dots:
                    self.status_dots[key].setStyleSheet(self._DOT_ON)
            for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
                if blabel == label:
                    self._launch_nav_buttons[i] = (btn, blabel, self._launch_on_style)
                    btn.setStyleSheet(self._launch_on_style)
                    break
        else:
            self._launch_states[label] = False
            for key in dot_keys:
                if key in self.status_dots:
                    self.status_dots[key].setStyleSheet(self._DOT_OFF)
            for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
                if blabel == label:
                    self._launch_nav_buttons[i] = (btn, blabel, self._launch_btn_style)
                    btn.setStyleSheet(self._launch_btn_style)
                    break
        self._update_selection()
        self._refresh_terminal_display()
        self._process_queue()

    def _gui_log_msg(self, msg):
        """Append a timestamped info line to the GUI log and refresh terminal."""
        ts = time.strftime("%H:%M:%S")
        self._gui_log.append(f"[{ts}] {msg}\n")
        self._refresh_terminal_display()

    # -----------------------------------------------------------------
    # Screen lock
    # -----------------------------------------------------------------
    def _build_lock_overlay(self):
        """Lazily build the fullscreen lock overlay widget."""
        overlay = QWidget(self)
        overlay.setObjectName("lockOverlay")
        overlay.setStyleSheet(
            "QWidget#lockOverlay { background-color: rgba(0, 0, 0, 235); }"
        )
        overlay.setAutoFillBackground(True)
        overlay.setGeometry(0, 0, self.width(), self.height())

        v = QVBoxLayout(overlay)
        v.setContentsMargins(0, 0, 0, 0)
        v.setAlignment(Qt.AlignCenter)
        v.addStretch()

        title = QLabel("\U0001F512  SCREEN LOCKED")
        f = QFont()
        f.setPointSize(36)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #dcdcdc; background: transparent; border: none;")
        v.addWidget(title)

        hint = QLabel("Press Ctrl+Shift+L to unlock")
        hf = QFont()
        hf.setPointSize(14)
        hint.setFont(hf)
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #888; background: transparent; border: none;")
        v.addWidget(hint)
        self._lock_hint_label = hint

        pw = QLineEdit()
        pw.setEchoMode(QLineEdit.Password)
        pw.setPlaceholderText("password")
        pw.setAlignment(Qt.AlignCenter)
        pw.setFixedWidth(320)
        pw.setStyleSheet(
            "QLineEdit {"
            "  background-color: #1a1a1a; color: #dcdcdc;"
            "  border: 1px solid #555; border-radius: 4px;"
            "  padding: 8px 10px; font-size: 16px; font-family: monospace;"
            "}"
            "QLineEdit:focus { border: 1px solid #0af; }"
        )
        pw.returnPressed.connect(self._try_unlock)
        # Strong focus + visible from the start. The previous two-step
        # design (Ctrl+Shift+L to reveal) was fragile — keystrokes went
        # to whichever widget had focus before the lock and the app
        # event filter swallowed them. Showing the field immediately
        # means the password input is always the obvious target.
        pw.setFocusPolicy(Qt.StrongFocus)
        # Center the line edit horizontally inside the QVBoxLayout via
        # a wrapping QHBoxLayout — QLineEdit otherwise stretches.
        pw_row = QHBoxLayout()
        pw_row.addStretch()
        pw_row.addWidget(pw)
        pw_row.addStretch()
        v.addLayout(pw_row)
        self._lock_password_input = pw

        status = QLabel("")
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet("color: #f55; background: transparent; border: none;"
                             " font-size: 12px;")
        status.setVisible(False)
        v.addWidget(status)
        self._lock_status_label = status

        v.addStretch()
        overlay.hide()
        self._lock_overlay = overlay
        return overlay

    # -----------------------------------------------------------------
    # Performance mode — opt-in low-CPU veil
    # -----------------------------------------------------------------
    def _build_mode_overlay(self):
        """Lazy-construct the unified PERF + REC overlay. One wide widget
        carries both veils with a 60° diagonal split between them; it
        translates horizontally to expose whichever combination of modes
        is active. See `_ModeOverlay` for the geometry."""
        central = self.centralWidget()
        if central is None:
            return None
        overlay = _ModeOverlay(central)
        overlay.configure_for_screen(central.width(), central.height())
        self._mode_overlay = overlay
        # Alias the stats label so the existing CPU/GPU update path
        # (_sample_perf_stats → _refresh_perf_overlay_text) keeps
        # working with no other changes.
        self._perf_stats_label = overlay.perf_stats_label
        return overlay

    def _update_mode_overlay_state(self):
        """Compute the target overlay state from the perf + rec flags
        and animate the unified overlay to it."""
        if self._mode_overlay is None:
            if self._build_mode_overlay() is None:
                return
        perf_on = self._performance_mode
        rec_on = bool(self._rec_active)
        if perf_on and rec_on:
            target = _ModeOverlay.BOTH
        elif perf_on:
            target = _ModeOverlay.PERF
        elif rec_on:
            target = _ModeOverlay.REC
        else:
            target = _ModeOverlay.OFF
        self._mode_overlay.set_state(target)

    def _sample_perf_stats(self):
        """Take a fresh CPU/GPU sample and (when the veil is up) refresh
        the overlay. Runs every 5 s for the lifetime of the window so
        we always have a fresh Before snapshot the moment Perf mode is
        entered.

        GUI CPU is per-process, normalised by total logical-core count
        so a single saturated core reads as 100/N% (matches Task
        Manager's column). GPU is whole-device — per-process GPU isn't
        reliably queryable across drivers, and on a dev laptop the GUI
        is the dominant graphics client so this is a fair proxy for
        what the HUD is costing the GPU.
        """
        if self._perf_proc is not None:
            try:
                raw = self._perf_proc.cpu_percent(interval=None)
                self._last_gui_cpu_pct = raw / self._perf_cpu_count
            except Exception:
                pass
        if _HAS_NVML:
            try:
                if self._perf_nvml_handle is None:
                    pynvml.nvmlInit()
                    self._perf_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(
                    self._perf_nvml_handle
                )
                self._last_sys_gpu_pct = float(util.gpu)
            except Exception:
                pass
        if self._performance_mode:
            self._refresh_perf_overlay_text()

    def _refresh_perf_overlay_text(self):
        """Write the Before/Now values into the overlay label."""
        if self._perf_stats_label is None:
            return

        def fmt(v):
            return '--' if v is None else f'{v:.1f}%'

        self._perf_stats_label.setText(
            f'CPU   Before: {fmt(self._baseline_gui_cpu_pct)}     '
            f'Now: {fmt(self._last_gui_cpu_pct)}\n'
            f'GPU   Before: {fmt(self._baseline_sys_gpu_pct)}     '
            f'Now: {fmt(self._last_sys_gpu_pct)}'
        )

    def _on_performance_clicked(self):
        """Toggle Performance Mode."""
        if self._performance_mode:
            self._exit_performance_mode()
        else:
            self._enter_performance_mode()

    def _enter_performance_mode(self):
        """Pause every QTimer the window owns + raise the PERFORMANCE veil.

        Iterating __dict__ catches new timers added in the future
        automatically; isActive() filters down to the ones that were
        genuinely running so toggle-off doesn't kick idle timers awake.
        The perf-stats ticker is skipped here and started right after
        — it's the one timer that stays alive while the veil is up so
        the CPU/GPU readout keeps refreshing.
        """
        if self._performance_mode:
            return
        # Force a fresh sample BEFORE flipping the mode flag so the
        # Before baseline isn't None for users who hit Perf within
        # the first 5 s (before the periodic sampler has ticked). The
        # sample's refresh-overlay branch is gated on _performance_mode,
        # so calling it here with the flag still False is safe.
        self._sample_perf_stats()

        self._performance_mode = True
        self._perf_resume_timers = []
        # Two timers stay alive in Perf mode:
        #   _perf_stats_timer  — drives the 5 s CPU/GPU readout.
        #   _rec_timer         — keeps the REC overlay reactive to
        #                        /data/toggle_collect so the operator
        #                        can split the screen even while
        #                        throttled. Both are cheap.
        always_on = {self._perf_stats_timer, self._rec_timer}
        for val in self.__dict__.values():
            if (isinstance(val, QTimer) and val.isActive()
                    and val not in always_on):
                val.stop()
                self._perf_resume_timers.append(val)

        # Snapshot the last-seen samples as the Before baseline before
        # the GUI throttles down. The sampler keeps running so Now
        # will refresh at the next 5 s tick.
        self._baseline_gui_cpu_pct = self._last_gui_cpu_pct
        self._baseline_sys_gpu_pct = self._last_sys_gpu_pct

        # Pause the camera + lidar worker threads via their slot_ready
        # backpressure. With the event cleared and the slots gated to
        # not re-set it, the workers drain submissions in their loop
        # without doing downscale/render work. This is the biggest
        # remaining CPU sink after the timer pause.
        for w in (getattr(self, '_camera_worker', None),
                  getattr(self, '_lidar_worker', None)):
            if w is not None and hasattr(w, '_slot_ready'):
                try:
                    w._slot_ready.clear()
                except Exception:
                    pass

        # Slide the unified overlay to its new state (PERF or BOTH
        # depending on whether REC is also active). Paint the stats
        # immediately so the operator sees Before / -- before the
        # 5 s tick lands.
        self._update_mode_overlay_state()
        self._refresh_perf_overlay_text()

        self.btn_performance.setText("Exit Performance")
        self.btn_performance.setStyleSheet(self._perf_on_style)
        for i, (b, lbl, _s) in enumerate(self._nav_buttons):
            if b is self.btn_performance:
                self._nav_buttons[i] = (
                    b, "Exit Performance", self._perf_on_style,
                )
                break

    def _exit_performance_mode(self):
        if not self._performance_mode:
            return
        self._performance_mode = False
        # Leave the sampler running — it's the source of the Before
        # snapshot the next time we enter Perf mode.
        # Slide overlay back: REC stays full-screen if REC is still on,
        # otherwise slides off entirely.
        self._update_mode_overlay_state()

        # Wake the camera + lidar workers — calling slot_ready() sets
        # the event so the next submission processes normally. Without
        # this, both workers would stay drained forever.
        for w in (getattr(self, '_camera_worker', None),
                  getattr(self, '_lidar_worker', None)):
            if w is not None and hasattr(w, 'slot_ready'):
                try:
                    w.slot_ready()
                except Exception:
                    pass

        for t in self._perf_resume_timers:
            try:
                t.start()
            except RuntimeError:
                # Timer may have been deleted while we were paused.
                pass
        self._perf_resume_timers = []

        self.btn_performance.setText("Performance Mode")
        self.btn_performance.setStyleSheet(self._perf_off_style)
        for i, (b, lbl, _s) in enumerate(self._nav_buttons):
            if b is self.btn_performance:
                self._nav_buttons[i] = (
                    b, "Performance Mode", self._perf_off_style,
                )
                break

    # -----------------------------------------------------------------
    # REC mode — driven by /data/toggle_collect (base_automator from a
    # test script, or the HUD itself via _request_record_toggle / R key).
    # -----------------------------------------------------------------
    def _request_record_toggle(self):
        """Flip ROS bag recording. Publishes the inverse of the current
        latest_recording_active on /data/toggle_collect; the existing
        subscriber updates state and _rec_tick reacts on the next tick.
        Works whether or not a test script is also driving the topic."""
        node = getattr(self, '_ros_node', None)
        if node is None:
            return
        pub = getattr(node, 'toggle_collect_pub', None)
        if pub is None:
            return
        msg = Bool()
        msg.data = not bool(getattr(node, 'latest_recording_active', False))
        pub.publish(msg)

    def _rec_tick(self):
        """20 Hz tick. Reads HudNode.latest_recording_active and:
          • On OFF→ON edge: snapshot existing dot styles, show the
            overlay.
          • While ON: ramp every status dot's background between
            #000000 and #ff0000 on a 2 Hz cosine.
          • On ON→OFF edge: restore the snapshotted styles, hide the
            overlay. (The existing participation-pulse code will
            reassert correct colors on its own next tick; the
            snapshot just spares us a flash of stale red.)
        """
        node = getattr(self, '_ros_node', None)
        active = bool(getattr(node, 'latest_recording_active', False))

        if active and not self._rec_active:
            # OFF → ON
            try:
                self._dot_styles_before_rec = {
                    k: w.styleSheet() for k, w in self.status_dots.items()
                }
            except Exception:
                self._dot_styles_before_rec = None
            self._rec_phase_t0 = time.monotonic()
            self._rec_active = True  # set BEFORE update so state mapping is right
            self._update_mode_overlay_state()

        elif not active and self._rec_active:
            # ON → OFF
            if self._dot_styles_before_rec is not None:
                for k, style in self._dot_styles_before_rec.items():
                    w = self.status_dots.get(k)
                    if w is not None:
                        try:
                            w.setStyleSheet(style)
                        except Exception:
                            pass
            self._dot_styles_before_rec = None
            self._rec_phase_t0 = None
            self._rec_active = False
            self._update_mode_overlay_state()

        self._rec_active = active
        if not active:
            return

        # Cosine ramp: spends a beat at each extreme rather than just
        # crossing through. 2 Hz cycle ⇒ period 500 ms.
        t = time.monotonic() - (self._rec_phase_t0 or time.monotonic())
        ramp = 0.5 * (1.0 - math.cos(2.0 * math.pi * 2.0 * t))
        r = int(255 * ramp)
        style = (
            f"background-color: rgb({r}, 0, 0); "
            "border-radius: 7px; border: none;"
        )
        for w in self.status_dots.values():
            try:
                w.setStyleSheet(style)
            except Exception:
                pass

    def _toggle_screen_lock(self):
        """Ctrl+Shift+L entry point (only called when unlocked)."""
        if self._screen_locked:
            return
        self._lock_screen()

    def _lock_screen(self):
        if self._lock_overlay is None:
            self._build_lock_overlay()
        # Password field is visible immediately. Keeping it hidden
        # behind a second Ctrl+Shift+L stranded the operator when
        # setFocus didn't take effect — without focus, the app event
        # filter swallowed every keystroke and there was no obvious
        # way to recover.
        self._lock_password_visible = True
        self._lock_password_input.setVisible(True)
        self._lock_password_input.setEnabled(True)
        self._lock_password_input.clear()
        self._lock_status_label.setVisible(False)
        self._lock_hint_label.setText("Type password and press Enter")
        self._lock_overlay.setGeometry(0, 0, self.width(), self.height())
        self._lock_overlay.show()
        self._lock_overlay.raise_()
        # Show the cursor again so the operator can see they're locked
        # (the HUD normally hides it).
        self.setCursor(Qt.ArrowCursor)
        self._screen_locked = True
        self._gui_log_msg("Screen locked")
        # Force focus to the password field. Deferred to the next event
        # loop iteration: setFocus on a widget that was just shown — or
        # one whose top-level didn't own keyboard focus before the lock
        # — silently no-ops when called inline. The 50ms follow-up
        # handles the case where Qt re-routes focus after the first
        # tick (e.g. a sibling widget's deferred FocusIn races with us).
        self._focus_lock_password()

    def _focus_lock_password(self):
        """Visually focus the password field. Typing is NOT routed via
        Qt focus — see eventFilter, which manually relays printable
        characters and Backspace/Enter into the QLineEdit while
        _screen_locked is True. We tried grabKeyboard() here; on
        Linux/X11 it does an X-server-wide keyboard grab that locks
        the entire OS keyboard until release, which stranded the
        operator from killing the GUI in another terminal. Don't."""
        pw = self._lock_password_input
        if pw is None or not self._screen_locked:
            return
        if self._lock_overlay is not None:
            self._lock_overlay.raise_()
        pw.raise_()
        # setFocus is best-effort — used only to show the caret.
        # The manual relay in eventFilter is the real input path.
        pw.setFocus(Qt.OtherFocusReason)

    def _show_lock_password_input(self):
        """Ctrl+Shift+L while locked → re-focus the password field.
        Kept for the legacy two-step flow's keystroke (and as a
        rescue path if the operator clicked outside the field and
        focus drifted)."""
        if not self._screen_locked or self._lock_password_input is None:
            return
        if not self._lock_password_visible:
            self._lock_password_visible = True
            self._lock_password_input.setVisible(True)
            self._lock_status_label.setVisible(False)
            self._lock_hint_label.setText("Type password and press Enter")
        self._focus_lock_password()

    def _try_unlock(self):
        if not self._screen_locked or self._lock_password_input is None:
            return
        if self._lock_password_input.text() == self._lock_password:
            self._unlock_screen()
        else:
            self._lock_password_input.clear()
            self._lock_status_label.setText("Incorrect password")
            self._lock_status_label.setVisible(True)

    def _unlock_screen(self):
        self._screen_locked = False
        if self._lock_password_input is not None:
            self._lock_password_input.clear()
            self._lock_password_input.setVisible(False)
        if self._lock_overlay is not None:
            self._lock_overlay.hide()
        self._lock_password_visible = False
        # Restore the HUD's default blank cursor.
        self.setCursor(Qt.BlankCursor)
        self._gui_log_msg("Screen unlocked")

    def resizeEvent(self, event):
        """Keep the lock + REC overlays sized to the window and the
        auto badge pinned.

        Qt fires resizeEvent during __init__ before the overlay widgets
        exist, so getattr-guard every attribute we touch — otherwise the
        first resize during construction raises AttributeError and kills
        the Python process.
        """
        super().resizeEvent(event)
        lock = getattr(self, '_lock_overlay', None)
        if lock is not None:
            lock.setGeometry(0, 0, self.width(), self.height())
        # Unified mode overlay handles its own geometry — just hand it
        # the new screen dimensions and it'll resize + snap to its
        # current state.
        mode = getattr(self, '_mode_overlay', None)
        if mode is not None:
            central = self.centralWidget()
            if central is not None:
                mode.configure_for_screen(central.width(), central.height())
        if hasattr(self, '_position_auto_badge'):
            self._position_auto_badge()

    def showEvent(self, event):
        """Re-pin the auto badge on first show and on any hide/show cycle.
        resizeEvent alone has been seen to fire before the central widget
        finishes its first layout pass, leaving the badge stranded in the
        top-left corner until the next user action triggered a re-pin.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._position_auto_badge)

    # -----------------------------------------------------------------
    # Auto mode
    # -----------------------------------------------------------------
    def _ekf_status_tick(self):
        """5 Hz tick: drive the EKF participation pulse and pull the
        latest /autonomous_mode into the corner badge. Combined into
        one slot to keep timer count down."""
        self._ekf_pulse_tick(time.monotonic())
        self._sync_auto_mode_from_topic()

    def _toggle_auto_mode(self):
        """A-key handler. Publish a fake X-button rising edge on /joy
        so control.cpp toggles its autonomousMode (it watches button
        index 3). Update the badge optimistically; _sync_auto_mode_from_topic
        will overwrite from the real /autonomous_mode topic once
        control.cpp confirms — that's the authoritative state."""
        self._auto_mode = not self._auto_mode
        self._apply_auto_badge()
        self._publish_fake_x_button_press()
        self._gui_log_msg(f"Auto mode: {'ON' if self._auto_mode else 'OFF'}")

    def _apply_auto_badge(self):
        """Re-paint the auto-mode badge from _auto_mode."""
        if self._auto_mode:
            self._auto_badge.setText("AUTO ON")
            self._auto_badge.setStyleSheet(self._auto_badge_on_style)
        else:
            self._auto_badge.setText("AUTO OFF")
            self._auto_badge.setStyleSheet(self._auto_badge_off_style)
        self._auto_badge.adjustSize()
        self._position_auto_badge()
        self._auto_badge.raise_()

    def _publish_fake_x_button_press(self):
        """Publish Joy(buttons[3]=1) and a release 200ms later. The
        rising edge from 0→1 is what control.cpp triggers on; the
        release lets a subsequent press fire again. Matches
        t002_automator._send_x_button_to_control. No-op if ROS isn't
        available."""
        node = self._ros_node
        if node is None or not hasattr(node, 'joy_pub'):
            return
        try:
            # Joy is only importable when _HAS_ROS — pull the class
            # off the publisher's msg_type so this method doesn't need
            # the import in HudWindow's scope.
            JoyMsg = node.joy_pub.msg_type
            press = JoyMsg()
            press.buttons = [0] * 8
            press.axes = [0.0] * 4
            press.buttons[3] = 1
            node.joy_pub.publish(press)

            def _release():
                try:
                    rel = JoyMsg()
                    rel.buttons = [0] * 8
                    rel.axes = [0.0] * 4
                    node.joy_pub.publish(rel)
                except Exception as e:
                    self._gui_log_msg(f"Auto release publish failed: {e}")
            QTimer.singleShot(200, _release)
        except Exception as e:
            self._gui_log_msg(f"Failed to publish fake X-button to /joy: {e}")

    def _sync_auto_mode_from_topic(self):
        """Read the latest /autonomous_mode value the HudNode has seen
        and reconcile _auto_mode with it. Called from the existing
        5 Hz status timer. Skips silently if ROS isn't wired up or
        the topic hasn't published yet."""
        node = self._ros_node
        if node is None:
            return
        latest = getattr(node, 'latest_autonomous_mode', None)
        if latest is None:
            return
        if latest != self._auto_mode:
            self._auto_mode = bool(latest)
            self._apply_auto_badge()
            self._gui_log_msg(
                f"Auto mode (from /autonomous_mode): "
                f"{'ON' if self._auto_mode else 'OFF'}"
            )

    def _position_auto_badge(self):
        """Pin the auto-mode badge to the upper-right of the central widget."""
        if not hasattr(self, '_auto_badge') or self._auto_badge is None:
            return
        parent = self._auto_badge.parentWidget()
        if parent is None:
            return
        margin = 8
        self._auto_badge.adjustSize()
        x = parent.width() - self._auto_badge.width() - margin
        y = margin
        self._auto_badge.move(max(x, 0), y)
        self._auto_badge.raise_()

    _STATUS_BTN_SELECTED = (
        "QPushButton {"
        "  background-color: #1a3a1a; color: #0f0;"
        "  border: 1px solid #0f0; border-radius: 3px;"
        "  padding: 1px 4px; font-size: 11px;"
        "}"
    )

    def eventFilter(self, obj, event):
        """Block input while locked; otherwise forward canvas clicks.

        Installed application-wide (see __init__), so this is called
        for every event in the process. The lock path runs first and
        must be cheap when unlocked — it's a single attribute read.
        """
        # Ctrl+Shift+L must work from anywhere, including while focus
        # is in a QLineEdit (the send_goal/send_GPS fields). Without
        # this app-level intercept, the line edit would swallow the
        # key event and HudWindow.keyPressEvent would never see it.
        if (event.type() == QEvent.KeyPress and
                event.key() == Qt.Key_L and
                (event.modifiers() & Qt.ControlModifier) and
                (event.modifiers() & Qt.ShiftModifier)):
            if self._screen_locked:
                self._show_lock_password_input()
            else:
                self._toggle_screen_lock()
            return True

        if self._screen_locked:
            et = event.type()
            if et == QEvent.KeyPress:
                # Manual key relay: don't depend on Qt's focus system
                # to deliver keystrokes to the password QLineEdit.
                # Instead, *intercept* every key event in the app and
                # drive the QLineEdit directly via insert() /
                # backspace() / returnPressed handler. This makes
                # typing work regardless of which widget actually has
                # focus or whether Qt routes events somewhere unexpected.
                pw = self._lock_password_input
                if pw is None:
                    return True
                k = event.key()
                if k in (Qt.Key_Return, Qt.Key_Enter):
                    self._try_unlock()
                    return True
                if k == Qt.Key_Backspace:
                    pw.backspace()
                    return True
                if k == Qt.Key_Delete:
                    pw.del_()
                    return True
                if k in (Qt.Key_Left,):
                    pw.cursorBackward(False, 1)
                    return True
                if k in (Qt.Key_Right,):
                    pw.cursorForward(False, 1)
                    return True
                if k == Qt.Key_Home:
                    pw.home(False)
                    return True
                if k == Qt.Key_End:
                    pw.end(False)
                    return True
                # Printable characters: insert into the password field.
                # event.text() carries the OS-decoded character (e.g.
                # '@' for Shift+2), so we don't have to reinvent
                # keyboard layout handling.
                text = event.text()
                if text and text.isprintable():
                    pw.insert(text)
                    return True
                # Non-printable / modifier-only keys: swallow.
                return True
            if et in (QEvent.KeyRelease, QEvent.ShortcutOverride,
                      QEvent.InputMethod, QEvent.InputMethodQuery):
                # All other keyboard-related events: swallow.
                return True
            if et in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                      QEvent.MouseButtonDblClick, QEvent.Wheel):
                # Clicks: only allow them on widgets inside the lock
                # overlay (the password field gets caret placement /
                # focus). Everything else swallowed.
                in_overlay = (
                    self._lock_overlay is not None and
                    isinstance(obj, QWidget) and
                    (obj is self._lock_overlay or
                     self._lock_overlay.isAncestorOf(obj))
                )
                if in_overlay:
                    if et == QEvent.MouseButtonPress:
                        self._focus_lock_password()
                    return False
                return True

        if event.type() == QEvent.MouseButtonPress and obj in self._canvas_to_cell:
            cell = self._canvas_to_cell[obj]
            self._toggle_sensor_expand(cell)
            return True
        return super().eventFilter(obj, event)

    def _toggle_sensor_expand(self, cell):
        """Toggle a sensor cell between expanded (full grid) and normal size."""
        grid = self._sensor_grid
        cells = self._sensor_cells
        positions = self._sensor_grid_positions

        if self._expanded_cell is cell:
            # Collapse: restore normal layout
            self._expanded_cell = None
            grid.removeWidget(cell)
            if cell is self._power_cell:
                self._left_col.addWidget(cell, stretch=1)
            for c, (r, col) in zip(cells, positions):
                c.setVisible(True)
                grid.addWidget(c, r, col)
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
        else:
            # Expand: hide grid cells, make this one fill the whole grid
            self._expanded_cell = cell
            for c in cells:
                grid.removeWidget(c)
                if c is not cell:
                    c.setVisible(False)
            if cell is self._power_cell:
                self._left_col.removeWidget(cell)
            grid.addWidget(cell, 0, 0, 2, 2)
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 0)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 0)

    def _on_status_dot_clicked(self, name):
        """Toggle device selection — show process output or return to info log."""
        dev_label = self._dot_to_device.get(name)
        target = dev_label if dev_label is not None else name

        if self._selected_process == target:
            # Clicking the same device again — deselect, show info log
            self._selected_process = None
        else:
            self._selected_process = target

        # Update button styles — highlight selected, reset others
        for btn, btn_name, base_style in self._status_nav_buttons:
            mapped = self._dot_to_device.get(btn_name, btn_name)
            if mapped == self._selected_process:
                btn.setStyleSheet(self._STATUS_BTN_SELECTED)
            else:
                btn.setStyleSheet(base_style)

        self._term_last_text = ''  # force refresh
        self._refresh_terminal_display()

    _MAX_TERMINAL_LINES = 100  # only show last N lines to avoid lag

    def _refresh_terminal_display(self):
        """Update the terminal QTextEdit with the selected process's output."""
        label = self._selected_process
        if label is None:
            self._term_header.setText("GUI Info Log")
            lines = self._gui_log if self._gui_log else ["No events yet."]
        else:
            self._term_header.setText(f"Process: {label}")
            buf = self._process_buffers.get(label)
            if buf is not None:
                lines = buf
            else:
                lines = [f"Process '{label}' is not running.\n"]
        # Only render the last N lines to keep the widget fast
        tail = lines[-self._MAX_TERMINAL_LINES:]
        text = "".join(tail)
        # Skip update if text hasn't changed
        if text == self._term_last_text:
            return
        self._term_last_text = text
        self._term_display.setPlainText(text)
        sb = self._term_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    _MAX_BUF_LINES = 500  # cap per-process buffer to avoid memory growth

    def _poll_process_output(self):
        """2 Hz: clean up exited process handles, trim buffers, refresh terminal."""
        for label in list(self._process_objects.keys()):
            proc = self._process_objects[label]
            if proc.poll() is not None:
                buf = self._process_buffers.get(label)
                if buf is not None:
                    buf.append(f"[Process exited with code {proc.returncode}]\n")
                self._process_objects.pop(label, None)
                self._process_readers.pop(label, None)
        # Trim buffers that have grown too large
        for buf in self._process_buffers.values():
            if len(buf) > self._MAX_BUF_LINES:
                del buf[:-self._MAX_BUF_LINES]
        self._refresh_terminal_display()

    def closeEvent(self, event):
        """Terminate all subprocesses and stop all modes on window
        close. Blocks until every kill thread has reported done (or
        hits its timeout) — historically these ran as daemon threads
        that the OS tore down mid-flight, which left orphan ros2
        binaries reparented to PID 1 in the container and showing as
        "external" processes in the next GUI session.
        """
        # Shut down the GPS tile fetcher worker thread.
        try:
            if hasattr(self, '_tile_fetcher') and self._tile_fetcher is not None:
                self._tile_fetcher.shutdown()
        except Exception:
            pass
        # Detach + shut down the camera worker. Detach first so the ROS
        # callback stops submitting before we tear the worker down.
        try:
            if self._ros_node is not None:
                self._ros_node.camera_worker = None
            if hasattr(self, '_camera_worker') and self._camera_worker is not None:
                self._camera_worker.shutdown()
        except Exception:
            pass
        # Same for the lidar worker.
        try:
            if self._ros_node is not None:
                self._ros_node.lidar_worker = None
            if hasattr(self, '_lidar_worker') and self._lidar_worker is not None:
                self._lidar_worker.shutdown()
        except Exception:
            pass

        # Stop live mode if active
        if self._live_active:
            self._stop_live_mode()

        # Kick off every kill in parallel, then join. Per-thread cap
        # of 12 s ≈ Phase 1 SIGINT (5 s) + Phase 2 SIGKILL (5 s) +
        # slack for the name-based pkill fallback.
        kill_threads = []
        for label, proc in list(self._process_objects.items()):
            t = self._kill_process(proc, label)
            if t is not None:
                kill_threads.append((label, t))
        for _label, t in kill_threads:
            t.join(timeout=12)
        self._process_objects.clear()

        # Disconnect container
        if self._container_connected:
            self._container_connected = False

        super().closeEvent(event)

    def _cur_btn(self):
        return self._nav_groups[self._nav_col][self._nav_row]

    def _is_slider_selected(self):
        """Return True if the slider group is currently selected."""
        widget, _, _ = self._cur_btn()
        return widget is self.pb_slider

    def _is_speed_selected(self):
        """Return True if the speed button is currently selected."""
        widget, _, _ = self._cur_btn()
        return widget is self.btn_speed

    def _is_sensor_selected(self):
        """Return True if a sensor cell or power cell is currently selected."""
        widget, _, _ = self._cur_btn()
        return widget in self._sensor_cells or widget is self._power_cell

    def _on_joy_nav(self, direction):
        """Slot fired by HudNode._cb_joy_nav → _JoyNavBus.nav.
        Synthesises a QKeyEvent and dispatches via the normal
        keyPressEvent path so the existing arrow-key navigation
        (selection grid, scrub mode, speed-select mode) works
        identically for the Xbox D-pad."""
        key_map = {
            'up':     Qt.Key_Up,
            'down':   Qt.Key_Down,
            'left':   Qt.Key_Left,
            'right':  Qt.Key_Right,
            'select': Qt.Key_Return,
        }
        qkey = key_map.get(direction)
        if qkey is None:
            return
        try:
            ev = QKeyEvent(QEvent.KeyPress, qkey, Qt.NoModifier)
            self.keyPressEvent(ev)
        except Exception:
            # Don't let a synthesised key event crash the GUI.
            pass

    def keyPressEvent(self, event):
        key = event.key()

        # Ctrl+Shift+L is intercepted at the application-level event
        # filter (see eventFilter) so it works from any focus context,
        # including QLineEdit. No handling needed here.

        # 'A' toggles auto mode. No modifiers so it doesn't fight Ctrl+A
        # or similar; intentionally only handled at the window level so
        # that typing 'a' into a focused QLineEdit (send_goal / GPS
        # field / lock password) still enters the character. 'P' and 'R'
        # follow the same pattern: 'P' toggles Performance mode, 'R'
        # toggles ROS-bag recording. Both work from any screen and at
        # any time — no test script required.
        if key == Qt.Key_A and not event.modifiers():
            self._toggle_auto_mode()
            return
        if key == Qt.Key_P and not event.modifiers():
            self._on_performance_clicked()
            return
        if key == Qt.Key_R and not event.modifiers():
            self._request_record_toggle()
            return

        # --- Scrub mode: arrows move the slider, Enter exits ---
        if self._scrub_mode:
            slider = self.pb_slider
            rng = slider.maximum() - slider.minimum()
            if rng <= 0:
                # No data loaded, exit scrub mode
                self._scrub_mode = False
                self._update_selection()
                return
            big_step = max(1, rng // 20)   # ~5% jump
            if key == Qt.Key_Right:
                slider.setValue(min(slider.value() + big_step, slider.maximum()))
                self._on_slider_seek(slider.value())
                self._position_indicators()
            elif key == Qt.Key_Left:
                slider.setValue(max(slider.value() - big_step, slider.minimum()))
                self._on_slider_seek(slider.value())
                self._position_indicators()
            elif key in (Qt.Key_Down, Qt.Key_Up):
                # Fine scrub: advance/retreat exactly one data row.
                # _pb_row_idx points to the NEXT unplayed row after a seek,
                # so Down uses it directly, Up goes back two (one before last played).
                idx = self._pb_row_idx
                if key == Qt.Key_Down:
                    idx = min(idx, len(self._pb_rows) - 1)
                else:
                    idx = max(idx - 2, 0)
                self._seek_to_row(idx)
                self._position_indicators()
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                self._scrub_mode = False
                self._update_selection()
            return

        # --- Speed select mode: Up increases speed, Down decreases, Enter exits ---
        if self._speed_mode:
            if key == Qt.Key_Up:
                self._pb_speed_idx = min(self._pb_speed_idx + 1, len(self._pb_speed_options) - 1)
                self._apply_speed()
            elif key == Qt.Key_Down:
                self._pb_speed_idx = max(self._pb_speed_idx - 1, 0)
                self._apply_speed()
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                self._speed_mode = False
                self._update_selection()
            return

        # --- Normal navigation (4-column, 14-row grid with logical row matching) ---
        group = self._nav_groups[self._nav_col]
        if key == Qt.Key_Up:
            self._nav_row = (self._nav_row - 1) % len(group)
            self._nav_last_row[self._nav_col] = self._nav_row
            self._update_selection()
        elif key == Qt.Key_Down:
            self._nav_row = (self._nav_row + 1) % len(group)
            self._nav_last_row[self._nav_col] = self._nav_row
            self._update_selection()
        elif key in (Qt.Key_Right, Qt.Key_Left):
            n_cols = len(self._nav_groups)
            new_col = (self._nav_col + (1 if key == Qt.Key_Right else -1)) % n_cols
            # Clamp nav_row into _nav_logical_rows for the source col.
            # The launch page swaps in a longer button list than the
            # main page (we just added more rows for send_goal/GPS),
            # so _nav_logical_rows[0]'s length can lag the group size.
            src_rows = self._nav_logical_rows[self._nav_col]
            src_idx = min(self._nav_row, len(src_rows) - 1)
            cur_logical = src_rows[src_idx]
            tgt_rows = self._nav_logical_rows[new_col]
            # Find best match: closest logical row <= current, prefer upper
            best_idx = 0
            for i, lr in enumerate(tgt_rows):
                if lr <= cur_logical:
                    best_idx = i
            self._nav_last_row[self._nav_col] = self._nav_row
            self._nav_col = new_col
            self._nav_row = best_idx
            self._nav_last_row[self._nav_col] = self._nav_row
            self._update_selection()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self._is_slider_selected():
                # Enter scrub mode — auto-pause if playing
                if self._pb_state == 'playing':
                    self._pause_playback()
                self._scrub_mode = True
                self._update_selection()
            elif self._is_speed_selected():
                # Enter speed select mode
                self._speed_mode = True
                self._update_selection()
            elif self._is_sensor_selected():
                cell, _, _ = self._cur_btn()
                self._toggle_sensor_expand(cell)
            else:
                widget, _, _ = self._cur_btn()
                if isinstance(widget, QLineEdit):
                    # Hand keyboard focus to the field so the operator
                    # can type. Pressing Enter inside the field fires
                    # returnPressed → _on_send_*_clicked, which calls
                    # clearFocus to hand control back to nav.
                    widget.setFocus(Qt.OtherFocusReason)
                    widget.selectAll()
                elif widget.isEnabled():
                    widget.click()
        elif key == Qt.Key_Space:
            if self._pb_state in ('playing', 'paused', 'ended'):
                self._on_play_pause()
        else:
            super().keyPressEvent(event)

    def _update_selection(self):
        """Restyle all buttons: selected gets highlight + mirrored arrows, others reset."""
        T = self._translate_to_theme
        for g, group in enumerate(self._nav_groups):
            for r, (widget, base_label, base_style) in enumerate(group):
                is_selected = (g == self._nav_col and r == self._nav_row)
                # Slider uses stylesheet only (no setText)
                if widget is self.pb_slider:
                    if is_selected and self._scrub_mode:
                        widget.setStyleSheet(T(self._slider_scrub_style))
                    elif is_selected:
                        widget.setStyleSheet(T(self._slider_sel_style))
                    else:
                        widget.setStyleSheet(T(self._slider_base_style))
                elif widget is self.btn_speed and is_selected and self._speed_mode:
                    widget.setText(
                        f"\u25B2  {base_label}  \u25BC"
                    )
                    widget.setStyleSheet(T(
                        self._speed_btn_style
                        .replace("border: 1px solid #555", "border: 1px solid #0f0")
                        .replace("color: #dcdcdc", "color: #0f0")
                    ))
                elif widget in self._sensor_cells or widget is self._power_cell:
                    if is_selected:
                        widget.setStyleSheet(T(self._sensor_sel_style))
                    else:
                        widget.setStyleSheet(T(self._sensor_frame_style))
                elif isinstance(widget, QLineEdit):
                    # QLineEdit nav participation. Don't call setText —
                    # that would clobber the user's typed value. Highlight
                    # via stylesheet instead.
                    if is_selected:
                        widget.setStyleSheet(T(self._send_field_sel_style))
                    else:
                        widget.setStyleSheet(T(base_style))
                else:
                    # Check if this is the selected device button
                    is_selected_device = False
                    if self._selected_process is not None:
                        for _sb, sn, _ss in self._status_nav_buttons:
                            if _sb is widget:
                                mapped = self._dot_to_device.get(sn, sn)
                                if mapped == self._selected_process:
                                    is_selected_device = True
                                break

                    if is_selected:
                        widget.setText(
                            f"{self._sel_arrow_l}  {base_label}  {self._sel_arrow_r}"
                        )
                        if is_selected_device:
                            widget.setStyleSheet(T(self._make_sel_style(self._STATUS_BTN_SELECTED)))
                        else:
                            widget.setStyleSheet(T(self._make_sel_style(base_style)))
                    else:
                        widget.setText(base_label)
                        if is_selected_device:
                            widget.setStyleSheet(T(self._STATUS_BTN_SELECTED))
                        else:
                            widget.setStyleSheet(T(base_style))
        # Position floating directional indicators around the selected widget
        self._position_indicators()

    def _position_indicators(self):
        """Show << and >> around the slider knob only when in scrub mode."""
        if not self._scrub_mode:
            self._ind_left.hide()
            self._ind_right.hide()
            return

        slider = self.pb_slider
        central = self.centralWidget()

        try:
            # Use Qt's style system to get the exact handle rect
            opt = QStyleOptionSlider()
            slider.initStyleOption(opt)
            handle_rect = slider.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, slider
            )
            # Map handle center to central widget coordinates
            handle_center = slider.mapTo(central, handle_rect.center())
            hx = handle_center.x()
            hy = handle_center.y()
            hw = handle_rect.width()
        except RuntimeError:
            self._ind_left.hide()
            self._ind_right.hide()
            return

        gap = 4

        # Left << (big step back)
        self._ind_left.adjustSize()
        self._ind_left.move(hx - hw // 2 - self._ind_left.width() - gap,
                            hy - self._ind_left.height() // 2)
        self._ind_left.show()
        self._ind_left.raise_()

        # Right >> (big step forward)
        self._ind_right.adjustSize()
        self._ind_right.move(hx + hw // 2 + gap,
                             hy - self._ind_right.height() // 2)
        self._ind_right.show()
        self._ind_right.raise_()

    def _nav_anim_tick(self):
        """Advance the spinning arrow animation."""
        self._sel_tick_count += 1
        if self._sel_tick_count >= self._sel_frame_durations[self._sel_frame_idx]:
            self._sel_tick_count = 0
            self._sel_frame_idx = (self._sel_frame_idx + 1) % len(self._sel_frames_r)
            self._sel_arrow_l = self._sel_frames_l[self._sel_frame_idx]
            self._sel_arrow_r = self._sel_frames_r[self._sel_frame_idx]
            widget, base_label, _ = self._cur_btn()
            # Slider / sensor cells / power cell don't support setText.
            # QLineEdit DOES support setText, but calling it would
            # overwrite the operator's typed args every 250ms — that
            # was the "text keeps reverting to default" bug. Skip it.
            if (widget is not self.pb_slider
                    and widget not in self._sensor_cells
                    and widget is not self._power_cell
                    and not isinstance(widget, QLineEdit)):
                widget.setText(
                    f"{self._sel_arrow_l}  {base_label}  {self._sel_arrow_r}"
                )

    def _set_btn_label(self, btn, new_label):
        """Update a nav button's base label and refresh display."""
        for g, group in enumerate(self._nav_groups):
            for r, (b, old_label, style) in enumerate(group):
                if b is btn:
                    self._nav_groups[g][r] = (b, new_label, style)
                    self._update_selection()
                    return

    # -- Topic -> sensor cell mapping --
    _TOPIC_CELL_MAP = {
        # Camera and Lidar handled via video mp4 playback
        '/gps_fix': 'GPS',
        '/encoders': 'Encoders',
        '/odom': 'Encoders',
        '/electrical/voltage': 'Power PCB',
        '/electrical/current': 'Power PCB',
        '/electrical/power': 'Power PCB',
        '/electrical/soc': 'Power PCB',
    }

    POWER_WINDOW_S = 3.0

    # Default directory for playback CSVs
    # Native Jetson: ~/AutoNav_25-26/logs (same as /autonav/logs inside container)
    _CSV_DIR = os.path.join(os.path.expanduser('~'), 'AutoNav_25-26', 'logs')

    # -----------------------------------------------------------------
    # Playback engine
    # -----------------------------------------------------------------
    # -- Container connection ------------------------------------------------

    def _on_connect_container(self):
        """Toggle connection to the Docker container."""
        if self._container_connected:
            self._disconnect_container()
        else:
            self._connect_container()

    def _connect_container(self):
        """Check if the container is running and enable container features."""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--quiet', '--filter', 'status=running',
                 '--filter', f'name=^/{self._container_name}$'],
                capture_output=True, text=True, timeout=5,
            )
            if not result.stdout.strip():
                self._gui_log_msg(
                    f"Container '{self._container_name}' is not running. "
                    "Start it with ./env/docker/run-container.sh"
                )
                return
        except FileNotFoundError:
            self._gui_log_msg("Docker not found on this system.")
            return
        except subprocess.TimeoutExpired:
            self._gui_log_msg("Docker command timed out.")
            return

        self._container_connected = True
        self._gui_log_msg(f"Connected to container '{self._container_name}'")

        # Update button to show connected state. Note that the nav tuple
        # stores the dark-original style; _update_selection runs each tick
        # and pipes it through _translate_to_theme, so light mode survives.
        T = self._translate_to_theme
        connected_style = (
            self._connect_style
            .replace("#1a2a3a", "#1a3a1a")
            .replace("#2a3a4a", "#2a4a2a")
            .replace("#0a1a2a", "#0a2a0a")
        )
        self.btn_connect.setText("Disconnect Container")
        self.btn_connect.setStyleSheet(T(connected_style))
        for i, (b, lbl, _s) in enumerate(self._nav_buttons):
            if b is self.btn_connect:
                self._nav_buttons[i] = (b, "Disconnect Container", connected_style)
                break

        # Enable container-dependent buttons
        for btn_ref, base_label in self._container_buttons.items():
            btn_ref.setEnabled(True)
            for _b, _l, base_s in self._nav_buttons:
                if _b is btn_ref:
                    btn_ref.setStyleSheet(T(base_s))
                    break
        # Turn container dots green
        for dot in self._container_dots:
            dot.setStyleSheet(T(
                "background-color: #4f4; border-radius: 5px; border: none;"
            ))

    def _disconnect_container(self):
        """Disconnect from the container and disable container features."""
        self._container_connected = False
        self._gui_log_msg(f"Disconnected from container '{self._container_name}'")

        T = self._translate_to_theme
        # Restore connect button
        self.btn_connect.setText("Connect to Container")
        self.btn_connect.setStyleSheet(T(self._connect_style))
        for i, (b, lbl, _s) in enumerate(self._nav_buttons):
            if b is self.btn_connect:
                self._nav_buttons[i] = (b, "Connect to Container", self._connect_style)
                break

        # Disable container-dependent buttons and show warnings
        for btn_ref, base_label in self._container_buttons.items():
            btn_ref.setEnabled(False)
            btn_ref.setStyleSheet(T(self._disabled_btn_style))
        # Turn container dots red
        for dot in self._container_dots:
            dot.setStyleSheet(T(
                "background-color: #f44; border-radius: 5px; border: none;"
            ))

        # Stop any active container modes
        if self._live_active:
            self._stop_live_mode()

        # Reset launch panel: any green/yellow buttons must return to gray
        # since their in-container processes are dead with the container.
        self._reset_all_launch_states()

    def _reset_all_launch_states(self):
        """Clear launch-page state when the container drops away.
        Cancels timers, kills any tracked host-side wrappers, resets every
        device to gray. Called from _disconnect_container so launch buttons
        do not stay locked green after the container stops."""
        for label in list(self._launch_states.keys()):
            if not self._launch_states.get(label):
                continue
            self._launch_states[label] = False
            # Status dots back to off
            for key in self._dot_keys_for(label):
                if key in self.status_dots:
                    self.status_dots[key].setStyleSheet(self._DOT_OFF)
            # Cancel readiness/flash timers
            t = self._startup_timers.pop(label, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            self._startup_deadlines.pop(label, None)
            self._ready_events.pop(label, None)
            t = self._flash_timers.pop(label, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
            # Kill the host-side wrapper subprocess (the in-container
            # children are already gone with the container).
            proc = self._process_objects.pop(label, None)
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                buf = self._process_buffers.get(label)
                if buf is not None:
                    buf.append("[Process ended: container disconnected]\n")
            self._process_readers.pop(label, None)
            # Restore button text + style
            for i, (btn, blabel, _s) in enumerate(self._launch_nav_buttons):
                if blabel == label:
                    btn.setText(label)
                    btn.setStyleSheet(self._launch_btn_style)
                    self._launch_nav_buttons[i] = (btn, blabel, self._launch_btn_style)
                    break
        # Empty the queue and refresh the queue label
        self._launch_queue.clear()
        self._update_queue_label()
        self._refresh_terminal_display()

    def _check_container_health(self):
        """Periodic check: if connected, verify the container is still running."""
        if not self._container_connected:
            return
        try:
            result = subprocess.run(
                ['docker', 'ps', '--quiet', '--filter', 'status=running',
                 '--filter', f'name=^/{self._container_name}$'],
                capture_output=True, text=True, timeout=3,
            )
            if not result.stdout.strip():
                self._gui_log_msg(f"Container '{self._container_name}' stopped — disconnecting")
                self._disconnect_container()
        except Exception:
            pass

    def _wrap_container_cmd(self, cmd, label=None):
        """Wrap a command to run inside the Docker container via docker exec.
        Writes the shell PID to /tmp/gui_pid_{label} so we can kill the
        entire process tree later."""
        if not self._container_connected:
            return cmd
        # Escape single quotes in cmd for safe embedding
        safe_cmd = cmd.replace("'", "'\\''")
        # Use a sanitized label for the PID file
        pid_tag = (label or 'unknown').replace(' ', '_').replace('/', '_')
        return (
            f"docker exec "
            f"-u {self._container_user} "
            f"-e HOME=/home/{self._container_user} "
            f"-e USER={self._container_user} "
            f"--workdir {self._container_workdir} "
            f"{self._container_name} "
            f"/bin/bash -lc "
            f"'echo $$ > /tmp/gui_pid_{pid_tag} && "
            f"source /opt/ros/humble/setup.bash && "
            f"if [ -f {self._container_workdir}/install/setup.bash ]; then "
            f"source {self._container_workdir}/install/setup.bash; fi && "
            f"exec {safe_cmd}'"
        )

    def _kill_process(self, proc, label):
        """Kill a launched process AND every descendant/orphan it
        spawned, returning the kill Thread so callers can ``.join()``
        when they need to block on completion (closeEvent does).
        Fire-and-forget callers may discard the return value.

        Kill plan, in order:
          1. SIGINT to the wrapper PID and its process group — gives
             bash traps + the ``ros2 run`` python wrapper a window to
             clean up gracefully.
          2. Recursive ps-tree walk → SIGKILL every descendant. Catches
             grandchildren the previous ``--ppid $PID`` one-liner
             missed.
          3. SIGKILL the wrapper's process group (``kill -- -$PID``) —
             catches anything that retained the group ID after being
             reparented to PID 1 in the container.
          4. Name-based ``pkill -f`` fallback for known orphan cases
             (the CUDA ``line_detector`` and Eigen ``grade_detector``
             survive their ``ros2 run`` python parent's SIGINT and
             escape steps 2–3 because reparenting drops them off the
             tree and out of the original PGID).
        """
        pid_tag = label.replace(' ', '_').replace('/', '_')

        def _do_kill():
            if self._container_connected:
                try:
                    # Phase 1: SIGINT — graceful path.
                    sigint_cmd = (
                        f"docker exec -u root {self._container_name} "
                        f"/bin/bash -c "
                        f"'PID=$(cat /tmp/gui_pid_{pid_tag} 2>/dev/null) && "
                        f"kill -INT $PID 2>/dev/null ; "
                        f"kill -INT -- -$PID 2>/dev/null'"
                    )
                    subprocess.run(sigint_cmd, shell=True, timeout=5,
                                   capture_output=True)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # Phase 2 + 3: SIGKILL the recursive descendant
                    # tree and the process group in one container hop.
                    force_cmd = (
                        f"docker exec -u root {self._container_name} "
                        f"/bin/bash -c '"
                        f"PID=$(cat /tmp/gui_pid_{pid_tag} 2>/dev/null) ; "
                        f"_desc() {{ local p=$1 ; "
                        f"for k in $(ps -o pid= --ppid \"$p\" 2>/dev/null) ; "
                        f"do echo $k ; _desc $k ; done ; }} ; "
                        f"ALL=$(_desc $PID 2>/dev/null) ; "
                        f"kill -9 $PID $ALL 2>/dev/null ; "
                        f"kill -9 -- -$PID 2>/dev/null ; "
                        f"rm -f /tmp/gui_pid_{pid_tag}'"
                    )
                    subprocess.run(force_cmd, shell=True, timeout=5,
                                   capture_output=True)
                    # Phase 4: name-based fallback for known orphans.
                    for pattern in self._orphan_pkill_patterns(label):
                        orphan_cmd = (
                            f"docker exec -u root {self._container_name} "
                            f"/bin/bash -c "
                            f"\"pkill -KILL -f '{pattern}' 2>/dev/null\""
                        )
                        subprocess.run(orphan_cmd, shell=True, timeout=5,
                                       capture_output=True)
                except Exception:
                    pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        t = threading.Thread(target=_do_kill, daemon=True)
        t.start()
        return t

    def _orphan_pkill_patterns(self, label):
        """Cmdline substrings for processes that escape the ps-tree +
        process-group kill in ``_kill_process``. Each entry is matched
        via ``pkill -KILL -f`` inside the container, so it must be a
        unique substring of the full cmdline — broad matches risk
        killing unrelated processes that happen to share a token. Keep
        the list narrow; add entries only for orphans observed in
        practice.
        """
        return {
            'LINE DETECT': [
                'autonav_detection line_detector',
                'autonav_detection/line_detector',
            ],
            'PCA DETECT': [
                'autonav_detection grade_detector',
                'autonav_detection/grade_detector',
            ],
        }.get(label, [])

    def _launch_cmd_for(self, label):
        """Return the raw command string for a device/test label."""
        for dev_label, _keys, dev_cmd in self._launch_devices:
            if dev_label == label:
                return dev_cmd
        return label

    def _on_playback_clicked(self):
        """Show the playback CSV selection page."""
        if self._live_active:
            self._stop_live_mode()
        self._show_playback_page()

    # -----------------------------------------------------------------
    # ROS bag playback — wraps `ros2 bag play` as a subprocess. The
    # bag's messages replay onto the same topic names the GUI's live
    # subscriptions watch, so the live canvases reanimate exactly as
    # they looked during the original test session.
    # -----------------------------------------------------------------
    # Visual-only topic subset, used as the --topics filter for
    # `ros2 bag play`. Mirrors base_automator.DEFAULT_BAG_TOPICS in
    # the testing package — recording and playback should always
    # match. If a topic is dropped from one list, drop it from the
    # other.
    #
    # Notably ABSENT: raw + inflated SICK IMU, /joy, /cmd_vel, /odom.
    # The IMU/joy/cmd_vel streams produce no pixel; together they were
    # ~83 % of the earlier bag's message volume and were the reason
    # camera/lidar paint got shoved off-rhythm. Raw /odom is dropped
    # because the live tick prefers /local_ekf/odom for the Encoders
    # panel whenever it's fresh, and the standalone bake also drops
    # raw /odom rows in favour of EKF — so replaying it adds no
    # extra signal on either consumer.
    _BAG_PLAYBACK_TOPICS = [
        '/zed/zed_node/rgb/color/rect/image',
        '/line_detection/debug/mask',
        '/line_detection/debug/overlay',
        '/line_detection/line_pixels',
        '/scan_fullframe',
        '/scan_pca_filtered',
        '/gps_fix',
        '/local_ekf/odom',
        '/global_ekf/odom',
        '/pose',
        '/global_costmap/costmap',
        '/plan',
        '/autonomous_mode',
        '/electrical/voltage',
        '/electrical/current',
        '/electrical/power',
        '/electrical/soc',
        '/data/toggle_collect',
    ]

    def _load_and_play_bag(self, bag_dir):
        """Replace any running playback / live source with this bag.
        Called by the "Load" buttons in the Playback Mode grid."""
        # Make sure nothing else is fighting for the topics: stop a
        # prior bag, stop any CSV-based playback timer (the historical
        # PB state machine is still here, just dormant), and DO NOT
        # stop live mode — live mode IS what makes the bag's replayed
        # messages reach the GUI canvases. Start live mode if it
        # isn't already on.
        self._stop_bag_play()
        try:
            self._stop_playback()
        except Exception:
            pass
        if not self._live_active:
            self._start_live_mode()
        cmd = [
            'ros2', 'bag', 'play', bag_dir,
            '--rate', '1.0',
            '--topics', *self._BAG_PLAYBACK_TOPICS,
        ]
        try:
            self._bag_play_proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._gui_log_msg(
                "ros2 not on PATH — can't start bag playback. "
                "Source the ROS2 setup before launching the GUI."
            )
            self._bag_play_proc = None
            return
        self._gui_log_msg(
            f"Playing bag: {bag_dir} "
            f"({len(self._BAG_PLAYBACK_TOPICS)} visual topics)"
        )
        self._bag_play_poll_timer.start()
        # Pop back to the main page so the operator sees the live
        # canvases reanimate from the bag.
        self._show_main_page()

    def _stop_bag_play(self):
        """SIGINT the bag-playback subprocess. Idempotent."""
        proc = self._bag_play_proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        self._bag_play_proc = None
        self._bag_play_poll_timer.stop()

    def _bag_play_poll(self):
        """Detect natural end-of-bag and clear state."""
        if self._bag_play_proc is None:
            self._bag_play_poll_timer.stop()
            return
        if self._bag_play_proc.poll() is not None:
            self._gui_log_msg("Bag playback finished")
            self._bag_play_proc = None
            self._bag_play_poll_timer.stop()

    def _load_csv(self, path):
        self._gui_log_msg(f"Loading CSV: {os.path.basename(path)}")
        rows = []
        gps_lats, gps_lons = [], []
        odom_xs, odom_ys = [], []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for r in reader:
                if len(r) < 4:
                    continue
                ts = int(r[0])
                topic = r[1]
                keys = r[2].split(',')
                values = r[3:]
                rows.append((ts, topic, keys, values))
                if topic == '/gps_fix' and len(values) >= 2:
                    try:
                        gps_lats.append(float(values[0]))
                        gps_lons.append(float(values[1]))
                    except ValueError:
                        pass
                elif topic == '/odom' and len(values) >= 2:
                    try:
                        odom_xs.append(float(values[0]))
                        odom_ys.append(float(values[1]))
                    except ValueError:
                        pass
        rows.sort(key=lambda x: x[0])
        t0 = rows[0][0]
        self._pb_rows = [(ts - t0, topic, keys, vals) for ts, topic, keys, vals in rows]
        self._pb_duration_ns = self._pb_rows[-1][0]
        duration_ms = int(self._pb_duration_ns / 1e6)
        self.pb_slider.setRange(0, duration_ms)
        self.pb_slider.setValue(0)
        self.pb_slider.setEnabled(True)
        total_s = self._pb_duration_ns / 1e9
        self.pb_time_label.setText(f"0.0s / {total_s:.1f}s")
        self._pb_total_frames = len(self._pb_rows)
        self.pb_frame_label.setText(f"F: 0 / {self._pb_total_frames}")

        # GPS and Odom windows adjust dynamically in _redraw_plots

        # Discover companion mp4 video files
        self._release_video_captures()
        if _HAS_CV2:
            csv_stem = os.path.splitext(path)[0]
            cam_mp4 = csv_stem + '_camera.mp4'
            lidar_mp4 = csv_stem + '_lidar_bev.mp4'
            if os.path.isfile(cam_mp4):
                self._camera_cap = cv2.VideoCapture(cam_mp4)
            if os.path.isfile(lidar_mp4):
                self._lidar_cap = cv2.VideoCapture(lidar_mp4)

        # Show "no video" text when mp4 files are missing
        has_cam = self._camera_cap is not None
        has_lidar = self._lidar_cap is not None
        self._cam_no_video_txt.set_visible(not has_cam)
        self._lidar_no_video_txt.set_visible(not has_lidar)
        self._cam_canvas.draw_idle()
        self._lidar_canvas.draw_idle()

        # Pre-fetch OSM map tiles for GPS background. Dispatched off-thread
        # so the "Start Playback" click returns immediately even on a
        # bag with many GPS points; the imshow goes up when the fetch
        # callback fires.
        if gps_lats and gps_lons:
            self._gps_no_data_txt.set_visible(False)
            self._request_gps_map(gps_lats, gps_lons, 'playback-start')
        else:
            self._gps_no_data_txt.set_visible(True)
        self._gps_canvas.draw_idle()

    def _clear_buffers(self):
        self._power_buf = {
            't': deque(),
            'V': deque(),
            'I': deque(),
            'P': deque(),
        }
        self._gps_buf = {
            'lat': deque(maxlen=self._live_gps_maxlen),
            'lon': deque(maxlen=self._live_gps_maxlen),
        }
        self._odom_buf = {
            'x':     deque(maxlen=self._live_odom_maxlen),
            'y':     deque(maxlen=self._live_odom_maxlen),
            'theta': deque(maxlen=self._live_odom_maxlen),
        }
        self._ema_eta_hours = None
        self._latest_soc_pct = None
        self._update_soc(0.0, force=True)
        if self._odom_tri_patch:
            self._odom_tri_patch.remove()
            self._odom_tri_patch = None
        if self._odom_costmap_img is not None:
            self._odom_costmap_img.remove()
            self._odom_costmap_img = None
        if self._odom_plan_line is not None:
            self._odom_plan_line.remove()
            self._odom_plan_line = None
        # Reset dot tracking
        self._active_dots = set()
        # Clear video imshow handles
        self._cam_im = None
        self._cam_im_source = None
        self._lidar_im = None

    def _release_video_captures(self):
        """Release any open VideoCapture objects."""
        if self._camera_cap is not None:
            self._camera_cap.release()
            self._camera_cap = None
        if self._lidar_cap is not None:
            self._lidar_cap.release()
            self._lidar_cap = None

    def _update_video_frames(self, elapsed_ns):
        """Seek both video captures to the correct frame and update canvases."""
        if not _HAS_CV2:
            return
        elapsed_s = elapsed_ns / 1e9
        frame_idx = int(elapsed_s * self._video_fps)

        for cap, ax, canvas, attr, no_txt, rotate in [
            (self._camera_cap, self._cam_ax, self._cam_canvas, '_cam_im',
             self._cam_no_video_txt, False),
            (self._lidar_cap, self._lidar_ax, self._lidar_canvas, '_lidar_im',
             self._lidar_no_video_txt, True),
        ]:
            if cap is None or not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rotate:
                rgb = np.rot90(rgb, 2)
                rgb = np.fliplr(rgb)
                # Display full 360° BEV with meter-based extent
                half_m = 10.0  # max range in meters
                extent = [-half_m, half_m, -half_m, half_m]
                no_txt.set_visible(False)
                im_handle = getattr(self, attr)
                if im_handle is None:
                    ax.axis('on')
                    im_handle = ax.imshow(rgb, aspect='equal', extent=extent)
                    ax.set_xlim(-half_m, half_m)
                    ax.set_ylim(-half_m, half_m)
                    ax.tick_params(axis='both', length=2, pad=2,
                                  labelsize=6, colors='#888', direction='in')
                    ax.set_xlabel('m', fontsize=6, color='#888', labelpad=1)
                    for spine in ax.spines.values():
                        spine.set_color('#444')
                    setattr(self, attr, im_handle)
                else:
                    im_handle.set_data(rgb)
                    im_handle.set_extent(extent)
                canvas.draw_idle()
                continue
            no_txt.set_visible(False)
            im_handle = getattr(self, attr)
            if im_handle is None:
                im_handle = ax.imshow(rgb, aspect='equal')
                setattr(self, attr, im_handle)
            else:
                im_handle.set_data(rgb)
            canvas.draw_idle()

        # Light up Camera/Lidar dots if videos are playing
        if self._camera_cap is not None and self._camera_cap.isOpened():
            dot = self.status_dots.get('Camera')
            if dot:
                dot.setStyleSheet(self._DOT_ON)
        if self._lidar_cap is not None and self._lidar_cap.isOpened():
            dot = self.status_dots.get('Lidar')
            if dot:
                dot.setStyleSheet(self._DOT_ON)

    def _on_play_pause(self):
        if self._pb_state == 'playing':
            self._pause_playback()
        elif self._pb_state == 'paused':
            self._resume_playback()
        elif self._pb_state == 'ended':
            # Restart from beginning
            self._seek_to_row(0)
            self._resume_playback()

    def _cycle_playback_speed(self):
        """Cycle through playback speed options (used by mouse click)."""
        self._pb_speed_idx = (self._pb_speed_idx + 1) % len(self._pb_speed_options)
        self._apply_speed()

    def _apply_speed(self):
        """Apply the current speed index: update label, adjust wall-clock."""
        self._pb_speed = self._pb_speed_options[self._pb_speed_idx]
        self._gui_log_msg(f"Playback speed: {self._pb_speed:g}x")
        label = f"{self._pb_speed:g}x"
        self.btn_speed.setText(label)
        self._set_btn_label(self.btn_speed, label)
        # Adjust wall-clock start so position stays consistent
        if self._pb_state == 'playing':
            elapsed_ns = self._pb_elapsed_ns
            self._pb_wall_start = time.monotonic() - elapsed_ns / (1e9 * self._pb_speed)
        elif self._pb_state == 'paused':
            elapsed_ns = self._pb_pause_elapsed_ns
            self._pb_wall_start = time.monotonic() - elapsed_ns / (1e9 * self._pb_speed)

    def _start_playback(self):
        self._gui_log_msg("Playback started")
        if self._live_active:
            self._stop_live_mode()
        self._pb_state = 'playing'
        self._pb_row_idx = 0
        self._pb_elapsed_ns = 0
        self._pb_wall_start = time.monotonic()
        self._clear_buffers()
        self.btn_pp.setEnabled(True)
        self._set_btn_label(self.btn_pp, "\u275A\u275A")
        # Highlight Playback Mode button green
        self._set_nav_btn_style(self.btn_playback, self._toggle_on_style)
        # Mark TEST dot as active (green) during playback
        if "TEST" in self.status_dots:
            self.status_dots["TEST"].setStyleSheet(
                "background-color: #0f0; border-radius: 7px; border: none;"
            )
        self._pb_timer = QTimer()
        self._pb_timer.setInterval(50)
        self._pb_timer.timeout.connect(self._playback_tick)
        self._pb_timer.start()
        self._pb_redraw_counter = 0

    def _pause_playback(self):
        self._gui_log_msg("Playback paused")
        self._pb_state = 'paused'
        if self._pb_timer:
            self._pb_timer.stop()
        self._pb_pause_elapsed_ns = (time.monotonic() - self._pb_wall_start) * 1e9 * self._pb_speed
        self._set_btn_label(self.btn_pp, "\u25B6")

    def _resume_playback(self):
        self._gui_log_msg("Playback resumed")
        self._pb_state = 'playing'
        self._pb_wall_start = time.monotonic() - self._pb_pause_elapsed_ns / (1e9 * self._pb_speed)
        self._set_btn_label(self.btn_pp, "\u275A\u275A")
        if self._pb_timer:
            self._pb_timer.start()

    def _stop_playback(self):
        self._pb_state = 'idle'
        if self._pb_timer:
            self._pb_timer.stop()
            self._pb_timer = None
        self._release_video_captures()
        self._reset_all_plots()
        self._set_btn_label(self.btn_pp, "\u25B6")
        self.btn_pp.setEnabled(False)
        # Restore Playback Mode button to normal style
        self._set_nav_btn_style(self.btn_playback, self._pb_button_style)
        for dot in self.status_dots.values():
            dot.setStyleSheet(
                "background-color: #555; border-radius: 7px; border: none;"
            )

    def _reset_all_plots(self):
        """Reset all sensor plots back to their default empty state."""
        # -- Camera --
        if self._cam_im is not None:
            self._cam_im.remove()
            self._cam_im = None
        self._cam_im_source = None
        if self._cam_lines_scatter is not None:
            self._cam_lines_scatter.remove()
            self._cam_lines_scatter = None
        self._cam_no_video_txt.set_visible(False)
        self._cam_live_txt.set_visible(False)
        self._cam_ax.set_facecolor('#111111')
        self._cam_canvas.draw_idle()

        # -- Lidar --
        if self._lidar_im is not None:
            self._lidar_im.remove()
            self._lidar_im = None
        self._lidar_no_video_txt.set_visible(False)
        self._lidar_live_txt.set_visible(False)
        self._lidar_ax.set_facecolor('#111111')
        self._lidar_canvas.draw_idle()

        # -- GPS --
        self._gps_trail.set_data([], [])
        self._gps_dot.set_data([], [])
        self._gps_coord_label.set_text('')
        if self._gps_map_im is not None:
            self._gps_map_im.remove()
            self._gps_map_im = None
        self._gps_map_img = None
        self._gps_map_extent = None
        self._gps_no_data_txt.set_visible(False)
        self._gps_live_txt.set_visible(False)
        self._gps_offline_txt.set_visible(False)
        if self._gps_cov_ellipse is not None:
            self._gps_cov_ellipse.remove()
            self._gps_cov_ellipse = None
        self._set_cell_title(
            '_cam_title_text', '_cam_title_label', 'Camera RAW')
        self._set_cell_title(
            '_lidar_title_text', '_lidar_title_label', 'LIDAR Heightband')
        self._set_cell_title(
            '_gps_title_text', '_gps_title_label', 'GPS')
        self._gps_ax.set_facecolor('#111111')
        self._gps_canvas.draw_idle()

        # -- Encoders (Odom) --
        if self._odom_scatter is not None:
            self._odom_scatter.remove()
            self._odom_scatter = None
        if self._odom_tri_patch is not None:
            self._odom_tri_patch.remove()
            self._odom_tri_patch = None
        if self._odom_costmap_img is not None:
            self._odom_costmap_img.remove()
            self._odom_costmap_img = None
        if self._odom_plan_line is not None:
            self._odom_plan_line.remove()
            self._odom_plan_line = None
        self._odom_dist_label.set_text('')
        self._odom_live_txt.set_visible(False)
        self._odom_ax.set_xlim(-5, 5)
        self._odom_ax.set_ylim(-5, 5)
        self._odom_canvas.draw_idle()

        # -- Power PCB --
        self._pwr_line_v.set_data([], [])
        self._pwr_line_i.set_data([], [])
        self._pwr_line_p.set_data([], [])
        self._pwr_v_live_txt.set_visible(False)
        self._pwr_i_live_txt.set_visible(False)
        self._pwr_p_live_txt.set_visible(False)
        self._pwr_val_v.setText("V: --")
        self._pwr_val_i.setText("I: --")
        self._pwr_val_p.setText("P: --")
        self._pwr_v_canvas.draw_idle()
        self._pwr_i_canvas.draw_idle()
        self._pwr_p_canvas.draw_idle()
        self._update_soc(0.0, force=True)

        # -- Clear data buffers --
        self._clear_buffers()

    def _playback_tick(self):
        elapsed_ns = (time.monotonic() - self._pb_wall_start) * 1e9 * self._pb_speed
        self._pb_elapsed_ns = elapsed_ns

        while self._pb_row_idx < len(self._pb_rows):
            rel_ns, topic, keys, vals = self._pb_rows[self._pb_row_idx]
            if rel_ns > elapsed_ns:
                break
            self._apply_row(rel_ns, topic, keys, vals)
            self._pb_row_idx += 1

        # Throttle plot redraws to every 5th tick (~4Hz) to reduce lag
        # Video frames update every tick for smooth playback
        self._pb_redraw_counter += 1
        if self._plots_dirty and self._pb_redraw_counter >= 5:
            self._redraw_plots()
            self._plots_dirty = False
            self._pb_redraw_counter = 0

        self._update_video_frames(elapsed_ns)

        elapsed_ms = int(elapsed_ns / 1e6)
        duration_ms = int(self._pb_duration_ns / 1e6)
        self.pb_slider.blockSignals(True)
        self.pb_slider.setValue(min(elapsed_ms, duration_ms))
        self.pb_slider.blockSignals(False)
        elapsed_s = elapsed_ns / 1e9
        total_s = self._pb_duration_ns / 1e9
        self.pb_time_label.setText(f"{elapsed_s:.1f}s / {total_s:.1f}s")
        self.pb_frame_label.setText(
            f"F: {self._pb_row_idx} / {self._pb_total_frames}"
        )

        if self._pb_row_idx >= len(self._pb_rows):
            # Pause at the end — user can press play to restart from beginning
            self._pb_state = 'ended'
            if self._pb_timer:
                self._pb_timer.stop()
            self._set_btn_label(self.btn_pp, "\u25B6")

    def _apply_row(self, rel_ns, topic, keys, values):
        cell_name = self._TOPIC_CELL_MAP.get(topic)
        if not cell_name:
            return

        if not hasattr(self, '_active_dots'):
            self._active_dots = set()
        if cell_name not in self._active_dots:
            dot = self.status_dots.get(cell_name)
            if dot:
                dot.setStyleSheet(
                    "background-color: #0f0; border-radius: 7px; border: none;"
                )
                self._active_dots.add(cell_name)

        t_s = rel_ns / 1e9
        self._plots_dirty = True

        if topic == '/electrical/voltage':
            self._power_buf['t'].append(t_s)
            self._power_buf['V'].append(float(values[0]))
            self._power_buf['I'].append(self._power_buf['I'][-1] if self._power_buf['I'] else 0)
            self._power_buf['P'].append(self._power_buf['P'][-1] if self._power_buf['P'] else 0)
            self._trim_power_buf(t_s)
        elif topic == '/electrical/current':
            self._power_buf['t'].append(t_s)
            self._power_buf['V'].append(self._power_buf['V'][-1] if self._power_buf['V'] else 0)
            self._power_buf['I'].append(float(values[0]))
            self._power_buf['P'].append(self._power_buf['P'][-1] if self._power_buf['P'] else 0)
            self._trim_power_buf(t_s)
        elif topic == '/electrical/power':
            self._power_buf['t'].append(t_s)
            self._power_buf['V'].append(self._power_buf['V'][-1] if self._power_buf['V'] else 0)
            self._power_buf['I'].append(self._power_buf['I'][-1] if self._power_buf['I'] else 0)
            self._power_buf['P'].append(float(values[0]))
            self._trim_power_buf(t_s)
        elif topic == '/electrical/soc':
            self._latest_soc_pct = float(values[0])
        elif topic == '/gps_fix':
            try:
                lat = float(values[0])
                lon = float(values[1])
                self._gps_buf['lat'].append(lat)
                self._gps_buf['lon'].append(lon)
            except (IndexError, ValueError):
                pass
        elif topic == '/odom':
            try:
                x = float(values[0])
                y = float(values[1])
                qz = float(values[2]) if len(values) > 2 else 0.0
                # Convert quaternion z to yaw: yaw = 2 * asin(qz)
                qz_clamped = max(-1.0, min(1.0, qz))
                yaw = 2.0 * math.asin(qz_clamped)
                self._odom_buf['x'].append(x)
                self._odom_buf['y'].append(y)
                self._odom_buf['theta'].append(yaw)
            except (IndexError, ValueError):
                pass
        # Camera and Lidar handled via video mp4 playback

    def _trim_power_buf(self, t_now):
        cutoff = t_now - self.POWER_WINDOW_S
        while self._power_buf['t'] and self._power_buf['t'][0] < cutoff:
            self._power_buf['t'].popleft()
            self._power_buf['V'].popleft()
            self._power_buf['I'].popleft()
            self._power_buf['P'].popleft()

    def _update_soc(self, fraction, force=False):
        """Update SOC gauge bar and ETA. fraction is 0.0–1.0.
        Hysteresis: only update display if change > 1% to avoid flicker."""
        pct = int(round(fraction * 100))
        if not force and hasattr(self, '_soc_display_pct') and abs(pct - self._soc_display_pct) <= 1:
            return
        self._soc_display_pct = pct
        fill = max(pct, 0)
        empty = max(100 - pct, 0)
        # Fill widget (index 0) = filled portion, spacer (index 1) = empty portion
        self._soc_bar_layout.setStretch(0, fill)
        self._soc_bar_layout.setStretch(1, empty)
        self._soc_label.setText(f"{pct}%")
        # Color: green > 50%, yellow 20-50%, red < 20%
        if fraction > 0.5:
            color = "#4f4"
        elif fraction > 0.2:
            color = "#ff4"
        else:
            color = "#f44"
        self._soc_fill.setStyleSheet(f"background-color: {color}; border: none; border-radius: 1px;")

        # Estimated time remaining — EMA-smoothed current draw
        avg_i = 0.0
        if self._power_buf['I']:
            vals = self._power_buf['I']
            avg_i = sum(abs(v) for v in vals) / len(vals)
        if avg_i > 0.01:
            raw_hours = fraction * self._CAPACITY_AH / avg_i
            if self._ema_eta_hours is None:
                self._ema_eta_hours = raw_hours
            else:
                smoothed = self._ETA_ALPHA * raw_hours + (1 - self._ETA_ALPHA) * self._ema_eta_hours
                # Monotonic: only allow the displayed estimate to decrease
                self._ema_eta_hours = min(self._ema_eta_hours, smoothed)
            h = int(self._ema_eta_hours)
            m = int((self._ema_eta_hours - h) * 60)
            self._eta_label.setText(f"{h}h {m:02d}m")
        else:
            self._eta_label.setText("--:--")

    def _redraw_plots(self):
        # --- Power mini oscilloscopes ---
        pwr_t = self._power_buf['t']
        if pwr_t:
            # Guard set_visible/draw_idle behind a state transition so
            # we don't invalidate Qt layout every 3 Hz tick. matplotlib
            # text artists are already invisible after the first hide.
            if getattr(self, '_pwr_live_txt_visible', True):
                self._pwr_v_live_txt.set_visible(False)
                self._pwr_i_live_txt.set_visible(False)
                self._pwr_p_live_txt.set_visible(False)
                self._pwr_live_txt_visible = False
            t_last = pwr_t[-1]
            xlim = (t_last - self.POWER_WINDOW_S, t_last)
            # matplotlib.Line2D.set_data accepts deques directly; no
            # need to materialize to list every redraw.
            for line, buf_key, ax, canvas in (
                (self._pwr_line_v, 'V', self._pwr_v_ax, self._pwr_v_canvas),
                (self._pwr_line_i, 'I', self._pwr_i_ax, self._pwr_i_canvas),
                (self._pwr_line_p, 'P', self._pwr_p_ax, self._pwr_p_canvas),
            ):
                line.set_data(pwr_t, self._power_buf[buf_key])
                ax.set_xlim(*xlim)
                canvas.draw_idle()
            # Update numeric readouts with latest values
            self._pwr_val_v.setText(f"V: {self._power_buf['V'][-1]:.2f}")
            self._pwr_val_i.setText(f"I: {self._power_buf['I'][-1]:.2f}")
            self._pwr_val_p.setText(f"P: {self._power_buf['P'][-1]:.2f}")
            # Use real SOC from electrical publisher if available,
            # otherwise fall back to voltage-derived estimate
            if self._latest_soc_pct is not None:
                soc = max(0.0, min(1.0, self._latest_soc_pct / 100.0))
            else:
                v = self._power_buf['V'][-1]
                soc = max(0.0, min(1.0, (v - 20.0) / (29.4 - 20.0)))
            self._update_soc(soc)
        else:
            if not getattr(self, '_pwr_live_txt_visible', True):
                self._pwr_v_live_txt.set_visible(True)
                self._pwr_i_live_txt.set_visible(True)
                self._pwr_p_live_txt.set_visible(True)
                self._pwr_live_txt_visible = True
                self._pwr_v_canvas.draw_idle()
                self._pwr_i_canvas.draw_idle()
                self._pwr_p_canvas.draw_idle()
            self._pwr_val_v.setText("V: --")
            self._pwr_val_i.setText("I: --")
            self._pwr_val_p.setText("P: --")

        # --- GPS with satellite map ---
        lats = self._gps_buf['lat']
        lons = self._gps_buf['lon']
        if lons:
            # Faint trail of all previous points
            self._gps_trail.set_data(lons, lats)
            # Current position dot
            self._gps_dot.set_data([lons[-1]], [lats[-1]])
            # 100 ft window centered on current position
            cur_lat, cur_lon = lats[-1], lons[-1]
            dlat = _GPS_VIEW_RADIUS_M / 111320.0
            dlon = _GPS_VIEW_RADIUS_M / (111320.0 * math.cos(math.radians(cur_lat)))
            self._gps_ax.set_xlim(cur_lon - dlon, cur_lon + dlon)
            self._gps_ax.set_ylim(cur_lat - dlat, cur_lat + dlat)
            # Show lat/lon truncated to 4 decimals
            self._gps_coord_label.set_text(
                f"Lat: {cur_lat:.4f}\nLon: {cur_lon:.4f}"
            )
            self._gps_canvas.draw_idle()

        # --- Odom XY with trail line + direction triangle ---
        xs = self._odom_buf['x']
        ys = self._odom_buf['y']
        thetas = self._odom_buf['theta']
        if xs:
            # Use a simple line instead of scatter (much faster)
            if self._odom_scatter is None:
                trail_color = '#000000' if self._theme == 'light' else 'white'
                self._odom_scatter, = self._odom_ax.plot(
                    xs, ys, '-', color=trail_color, linewidth=1.5, zorder=3,
                )
            else:
                self._odom_scatter.set_data(xs, ys)

            # --- Global costmap snippet (Tier 3 — robot-centered crop
            # of the global costmap so persistent line_layer +
            # map_padder corridor data are visible). Extent is in map
            # frame; treated as ≈ odom on this canvas (map↔odom drift
            # < 1 m in a healthy mission).
            costmap_alive = bool(getattr(self, '_costmap_alive', False))
            node_obj = getattr(self, '_ros_node', None)
            costmap_data = (
                getattr(node_obj, 'latest_costmap', None)
                if (costmap_alive and node_obj is not None) else None
            )
            cm_extent = None
            if costmap_data is not None:
                rgba, cm_extent = costmap_data
                if self._odom_costmap_img is None:
                    self._odom_costmap_img = self._odom_ax.imshow(
                        rgba, extent=cm_extent, origin='lower',
                        interpolation='nearest', zorder=2,
                    )
                else:
                    self._odom_costmap_img.set_data(rgba)
                    self._odom_costmap_img.set_extent(cm_extent)
                    self._odom_costmap_img.set_visible(True)
            elif self._odom_costmap_img is not None:
                self._odom_costmap_img.set_visible(False)

            # --- NAV2 global plan as a green Line2D. Same frame
            # treatment as the costmap (map ≈ odom).
            plan_alive = bool(getattr(self, '_plan_alive', False))
            plan_data = (
                getattr(node_obj, 'latest_plan_xy', None)
                if (plan_alive and node_obj is not None) else None
            )
            if plan_data is not None:
                plan_xs, plan_ys = plan_data
                if self._odom_plan_line is None:
                    self._odom_plan_line, = self._odom_ax.plot(
                        plan_xs, plan_ys, '-',
                        color='#33ff66', linewidth=1.5, zorder=4,
                    )
                else:
                    self._odom_plan_line.set_data(plan_xs, plan_ys)
                    self._odom_plan_line.set_visible(True)
            elif self._odom_plan_line is not None:
                self._odom_plan_line.set_visible(False)

            # Adjust window to fit all current points with padding.
            # When the costmap overlay is live, also include its
            # extent so the operator sees the full ring of cost cells
            # around the red triangle instead of a clipped corner.
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if cm_extent is not None:
                cm_x0, cm_x1, cm_y0, cm_y1 = cm_extent
                min_x = min(min_x, cm_x0)
                max_x = max(max_x, cm_x1)
                min_y = min(min_y, cm_y0)
                max_y = max(max_y, cm_y1)
            dx = max_x - min_x
            dy = max_y - min_y
            span = max(dx, dy) * 1.3
            span = max(span, 1.0)
            half = span / 2
            cx_view = (min_x + max_x) / 2
            cy_view = (min_y + max_y) / 2
            self._odom_ax.set_xlim(cx_view - half, cx_view + half)
            self._odom_ax.set_ylim(cy_view - half, cy_view + half)
            # Distance traveled — vectorized for speed
            if len(xs) > 1:
                ax_arr = np.array(xs)
                ay_arr = np.array(ys)
                dist = float(np.sum(np.hypot(np.diff(ax_arr), np.diff(ay_arr))))
            else:
                dist = 0.0
            self._odom_dist_label.set_text(f"Dist: {dist:.2f} m")

            cx, cy = xs[-1], ys[-1]
            theta = thetas[-1] if thetas else 0.0
            s = span * 0.05
            s = max(s, 0.1)
            nose = (cx + s * math.cos(theta),
                    cy + s * math.sin(theta))
            bl = (cx + s * 0.5 * math.cos(theta + 2.5),
                  cy + s * 0.5 * math.sin(theta + 2.5))
            br = (cx + s * 0.5 * math.cos(theta - 2.5),
                  cy + s * 0.5 * math.sin(theta - 2.5))
            if self._odom_tri_patch:
                self._odom_tri_patch.set_xy([nose, bl, br])
            else:
                tri = Polygon([nose, bl, br], closed=True,
                              facecolor='red', edgecolor='white',
                              linewidth=0.8, zorder=5)
                self._odom_ax.add_patch(tri)
                self._odom_tri_patch = tri
            self._odom_canvas.draw_idle()

    def _on_slider_pressed(self):
        if self._pb_state == 'playing' and self._pb_timer:
            self._pb_timer.stop()

    def _on_slider_released(self):
        if self._pb_state == 'playing' and self._pb_timer:
            self._pb_timer.start()

    def _seek_to_row(self, target_idx):
        """Seek directly to a specific row index (nanosecond-precise)."""
        target_idx = max(0, min(target_idx, len(self._pb_rows) - 1))
        # Replay all rows up to and including target_idx
        target_ns = self._pb_rows[target_idx][0]
        self._pb_wall_start = time.monotonic() - target_ns / 1e9
        if self._pb_state == 'paused':
            self._pb_pause_elapsed_ns = target_ns
        self._pb_elapsed_ns = target_ns

        self._clear_buffers()
        for i in range(target_idx + 1):
            rel_ns, topic, keys, vals = self._pb_rows[i]
            self._apply_row(rel_ns, topic, keys, vals)
        self._pb_row_idx = target_idx + 1

        self._redraw_plots()
        self._update_video_frames(target_ns)

        # Update slider and labels
        self.pb_slider.blockSignals(True)
        self.pb_slider.setValue(int(target_ns / 1e6))
        self.pb_slider.blockSignals(False)
        elapsed_s = target_ns / 1e9
        total_s = self._pb_duration_ns / 1e9
        self.pb_time_label.setText(f"{elapsed_s:.1f}s / {total_s:.1f}s")
        self.pb_frame_label.setText(
            f"F: {self._pb_row_idx} / {self._pb_total_frames}"
        )

    def _on_slider_seek(self, position_ms):
        target_ns = position_ms * 1e6
        self._pb_wall_start = time.monotonic() - target_ns / 1e9
        if self._pb_state == 'paused':
            self._pb_pause_elapsed_ns = target_ns
        self._pb_elapsed_ns = target_ns

        self._clear_buffers()
        self._pb_row_idx = 0
        for i, (rel_ns, topic, keys, vals) in enumerate(self._pb_rows):
            if rel_ns > target_ns:
                self._pb_row_idx = i
                break
            self._apply_row(rel_ns, topic, keys, vals)
        else:
            self._pb_row_idx = len(self._pb_rows)

        # Reset all dots then re-light active ones
        for dot in self.status_dots.values():
            dot.setStyleSheet(
                "background-color: #555; border-radius: 7px; border: none;"
            )
        active_cells = set()
        if self._gps_buf['lat']:
            active_cells.add('GPS')
        if self._odom_buf['x']:
            active_cells.add('Encoders')
        if self._power_buf['t']:
            active_cells.add('Power PCB')
        if self._camera_cap is not None:
            active_cells.add('Camera')
        if self._lidar_cap is not None:
            active_cells.add('Lidar')
        for cell in active_cells:
            dot = self.status_dots.get(cell)
            if dot:
                dot.setStyleSheet(
                    "background-color: #0f0; border-radius: 7px; border: none;"
                )

        self._redraw_plots()
        self._update_video_frames(target_ns)

        elapsed_s = target_ns / 1e9
        total_s = self._pb_duration_ns / 1e9
        self.pb_time_label.setText(f"{elapsed_s:.1f}s / {total_s:.1f}s")
        self.pb_frame_label.setText(
            f"F: {self._pb_row_idx} / {self._pb_total_frames}"
        )

    # -----------------------------------------------------------------
    # Live Mode
    # -----------------------------------------------------------------
    def _on_live_clicked(self):
        """Toggle Live Mode on/off."""
        if self._live_active:
            self._stop_live_mode()
        else:
            self._start_live_mode()

    def _start_live_mode(self):
        """Activate Live Mode: subscribe to ROS topics at 10 Hz."""
        self._gui_log_msg("Live Mode activated")
        if self._ros_node is None:
            self._gui_log_msg("WARNING: No ROS node — rclpy not available, live data disabled")
        # Stop playback if running
        if self._pb_state != 'idle':
            self._stop_playback()

        self._live_active = True
        self._live_t0 = time.monotonic()
        self._clear_buffers()

        # Button styles: Live green, Playback normal
        self._set_nav_btn_style(self.btn_live, self._toggle_on_style)
        self._set_nav_btn_style(self.btn_playback, self._pb_button_style)

        # Slider: show LIVE label, disable scrubbing
        self.pb_time_label.setText("LIVE")
        self.pb_frame_label.setText("")
        self.pb_slider.setEnabled(False)
        self.pb_slider.setRange(0, 0)
        self.btn_pp.setEnabled(False)

        # Show "NO DATA AVAILABLE" placeholders (hidden when data arrives)
        self._cam_live_txt.set_visible(True)
        self._cam_no_video_txt.set_visible(False)
        if self._cam_lines_scatter is not None:
            self._cam_lines_scatter.remove()
            self._cam_lines_scatter = None
        # Reset FPS trackers for the new session.
        self._cam_fps_deque.clear()
        self._cam_fps_txt.set_text('')
        self._lidar_fps_deque.clear()
        self._lidar_fps_txt.set_text('')
        self._set_cell_title(
            '_cam_title_text', '_cam_title_label', 'Camera RAW')
        self._set_cell_title(
            '_lidar_title_text', '_lidar_title_label', 'LIDAR Heightband')
        self._set_cell_title(
            '_gps_title_text', '_gps_title_label', 'GPS')
        self._cam_canvas.draw_idle()
        self._lidar_live_txt.set_visible(True)
        self._lidar_no_video_txt.set_visible(False)
        self._lidar_canvas.draw_idle()
        self._gps_live_txt.set_visible(True)
        self._gps_no_data_txt.set_visible(False)
        self._gps_offline_txt.set_visible(False)
        # Reset per-session tile-fetch bookkeeping so log lines fire again
        # on the next offline transition / first out-of-bounds fix.
        self._tile_oob_logged = False
        self._tile_request_pending = False
        # Clear GPS map background for a clean black screen
        if self._gps_map_im is not None:
            self._gps_map_im.remove()
            self._gps_map_im = None
        self._gps_map_extent = None
        self._gps_trail.set_data([], [])
        self._gps_dot.set_data([], [])
        self._gps_coord_label.set_text('')
        if self._gps_cov_ellipse is not None:
            self._gps_cov_ellipse.remove()
            self._gps_cov_ellipse = None
        self._gps_canvas.draw_idle()
        # Encoders: show live placeholder and clear plot
        self._odom_live_txt.set_visible(True)
        self._odom_ax.grid(False)
        self._odom_ax.tick_params(labelbottom=False, labelleft=False)
        self._odom_xy_label.set_visible(False)
        if self._odom_scatter is not None:
            self._odom_scatter.remove()
            self._odom_scatter = None
        if self._odom_tri_patch is not None:
            self._odom_tri_patch.remove()
            self._odom_tri_patch = None
        if self._odom_costmap_img is not None:
            self._odom_costmap_img.remove()
            self._odom_costmap_img = None
        if self._odom_plan_line is not None:
            self._odom_plan_line.remove()
            self._odom_plan_line = None
        self._odom_dist_label.set_text('')
        self._odom_canvas.draw_idle()
        self._pwr_v_live_txt.set_visible(True)
        self._pwr_i_live_txt.set_visible(True)
        self._pwr_p_live_txt.set_visible(True)
        self._pwr_v_canvas.draw_idle()
        self._pwr_i_canvas.draw_idle()
        self._pwr_p_canvas.draw_idle()

        # Flash only the live-relevant sensor dots
        self._live_received = set()
        self._live_sensors = {"Camera", "Lidar", "GPS", "Encoders", "Power PCB"}
        waiting = [n for n in self._live_sensors if n in self.status_dots]
        self._gui_log_msg("Awaiting live data from: " + ", ".join(waiting))
        self._live_flash_state = True
        self._live_dot_flash_timer = QTimer()
        self._live_dot_flash_timer.setInterval(200)
        self._live_dot_flash_timer.timeout.connect(self._live_dot_flash_tick)
        self._live_dot_flash_timer.start()
        for name, dot in self.status_dots.items():
            if name in self._live_sensors:
                dot.setStyleSheet(self._DOT_YELLOW)

        # Start 10 Hz timer
        self._live_timer = QTimer()
        self._live_timer.setInterval(200)  # 5 Hz live updates
        self._live_timer.timeout.connect(self._live_tick)
        self._live_timer.start()

    def _live_dot_flash_tick(self):
        """Flash dots yellow/gray for live sensors that haven't received data yet."""
        self._live_flash_state = not self._live_flash_state
        style = self._DOT_YELLOW if self._live_flash_state else self._DOT_OFF
        for name in self._live_sensors:
            if name not in self._live_received:
                dot = self.status_dots.get(name)
                if dot:
                    dot.setStyleSheet(style)

    def _live_set_dot_received(self, name):
        """Mark a sensor dot as having received data — stop flashing, go green."""
        if name not in self._live_received:
            self._live_received.add(name)
            self._gui_log_msg(f"Receiving live data from: {name}")
            dot = self.status_dots.get(name)
            if dot:
                dot.setStyleSheet(self._DOT_ON)

    def _stop_live_mode(self):
        """Deactivate Live Mode."""
        self._gui_log_msg("Live Mode deactivated")
        self._live_active = False
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None

        # Stop dot flash timer
        if hasattr(self, '_live_dot_flash_timer') and self._live_dot_flash_timer is not None:
            self._live_dot_flash_timer.stop()
            self._live_dot_flash_timer = None

        # Restore button style
        self._set_nav_btn_style(self.btn_live, self._pb_button_style)

        # Reset dots only for sensors that don't have a running process
        running_labels = set(self._process_objects.keys())
        for name, dot in self.status_dots.items():
            # Check if any running process matches this dot's device
            dev_label = self._dot_to_device.get(name, name)
            if dev_label not in running_labels:
                dot.setStyleSheet(self._DOT_OFF)

        # Hide live placeholders, clear imshow handles
        self._cam_live_txt.set_visible(False)
        self._cam_fps_txt.set_text('')
        self._cam_fps_deque.clear()
        self._cam_canvas.draw_idle()
        self._lidar_fps_txt.set_text('')
        self._lidar_fps_deque.clear()
        self._lidar_live_txt.set_visible(False)
        self._lidar_canvas.draw_idle()
        self._gps_live_txt.set_visible(False)
        self._gps_offline_txt.set_visible(False)
        self._gps_canvas.draw_idle()
        self._odom_live_txt.set_visible(False)
        self._odom_ax.grid(True, which='both', color='#333', linewidth=0.5)
        self._odom_ax.tick_params(labelbottom=True, labelleft=True)
        self._odom_xy_label.set_visible(True)
        self._odom_canvas.draw_idle()
        self._pwr_v_live_txt.set_visible(False)
        self._pwr_i_live_txt.set_visible(False)
        self._pwr_p_live_txt.set_visible(False)
        self._pwr_v_canvas.draw_idle()
        self._pwr_i_canvas.draw_idle()
        self._pwr_p_canvas.draw_idle()

        self._cam_im = None
        self._cam_im_source = None
        self._lidar_im = None

        # Reset time/frame labels
        self.pb_time_label.setText("0.0s / 0.0s")
        self.pb_frame_label.setText("F: -- / --")

    def _set_nav_btn_style(self, btn, style):
        """Update a navigation button's base style in all nav groups."""
        for g, group in enumerate(self._nav_groups):
            for r, (b, label, _old) in enumerate(group):
                if b is btn:
                    self._nav_groups[g][r] = (b, label, style)
                    break
        # Also update the master _nav_buttons list
        for i, (b, label, _old) in enumerate(self._nav_buttons):
            if b is btn:
                self._nav_buttons[i] = (b, label, style)
                break
        self._update_selection()

    # -----------------------------------------------------------------
    # GPS tile fetch coordination (off-thread)
    # -----------------------------------------------------------------
    def _request_gps_map(self, lats, lons, tag):
        """Post an async tile-fetch request; return False if rejected.

        Rejections (validation, already pending, GPS out of VA/MI) are
        silent on the hot path so the 5 Hz live tick doesn't spam logs.
        Out-of-bounds GPS is logged once per LIVE session.
        """
        if not lats or not lons:
            return False
        last_lat, last_lon = lats[-1], lons[-1]
        if not _gps_in_valid_region(last_lat, last_lon):
            if not getattr(self, '_tile_oob_logged', False):
                self._gui_log_msg(
                    f'GPS fix ({last_lat:.4f}, {last_lon:.4f}) outside '
                    'VA/MI bounds; skipping tile fetch')
                self._tile_oob_logged = True
            return False
        if self._tile_request_pending:
            return False
        self._tile_request_id += 1
        self._tile_request_pending = True
        self._tile_fetcher.request(lats, lons, self._tile_request_id, tag)
        return True

    def _on_tile_fetch_finished(self, img, extent, req_id, tag):
        """Main-thread slot: receives the fetched (or empty) tile image."""
        # req_id == -1 marks a worker-initiated reconnect retry; accept
        # regardless. Otherwise drop stale results.
        if req_id != -1 and req_id != self._tile_request_id:
            self._tile_request_pending = False
            return
        if req_id != -1:
            self._tile_request_pending = False
        if img is None or extent is None:
            # No map this round — show the offline placeholder if we don't
            # have any cached map at all yet.
            if self._gps_map_im is None:
                self._gps_offline_txt.set_visible(True)
                self._gps_canvas.draw_idle()
            return
        self._gps_map_img = img
        self._gps_map_extent = extent
        if self._gps_map_im is not None:
            self._gps_map_im.remove()
        self._gps_map_im = self._gps_ax.imshow(
            self._gps_map_img,
            extent=[extent[0], extent[1], extent[2], extent[3]],
            aspect='auto', zorder=0,
        )
        self._gps_offline_txt.set_visible(False)
        self._gps_canvas.draw_idle()

    def _on_camera_frame_ready(self, arr, overlay, line_fresh, source):
        """Main-thread slot: receives a downscaled camera frame from the
        camera worker and applies it to the matplotlib widgets. The heavy
        decoding / downscaling already happened off-thread; here we only
        do the matplotlib set_data + draw_idle, which on a ~480 px frame
        costs ~1 ms.

        Two array shapes are accepted:
          * HxWx3 RGB — for source='raw'/'overlay' (default colormap)
          * HxW MONO8 — for source='mask' (cmap='gray', vmin/vmax pinned
            so matplotlib doesn't autoscale every frame)
        Changing source rebuilds the AxesImage; set_data alone won't
        change the colormap and would render the mask in 'viridis'.
        """
        # In Performance Mode we deliberately DON'T re-arm the worker —
        # leaving _slot_ready cleared keeps the worker's backpressure
        # check tripping `continue`, so it drains submissions without
        # downscaling or emitting. Returning here also short-circuits
        # the matplotlib paint below.
        if self._performance_mode:
            return
        # Re-arm the worker first thing so it can downscale the next
        # frame in parallel with our paint here. This is what bounds
        # the Qt event queue to 1 paint deep regardless of source rate.
        self._camera_worker.slot_ready()
        if not self._live_active:
            return  # Ignore late frames if we exited live mode.
        self._cam_live_txt.set_visible(False)

        prev_source = getattr(self, '_cam_im_source', None)
        need_recreate = (
            self._cam_im is None or source != prev_source or
            (self._cam_im.get_array() is None) or
            (self._cam_im.get_array().shape != arr.shape)
        )
        if need_recreate:
            if self._cam_im is not None:
                self._cam_im.remove()
            if source == 'mask':
                self._cam_im = self._cam_ax.imshow(
                    arr, cmap='gray', vmin=0, vmax=255, aspect='equal')
            else:
                self._cam_im = self._cam_ax.imshow(arr, aspect='equal')
            self._cam_im_source = source
        else:
            self._cam_im.set_data(arr)
        self._live_set_dot_received('Camera')

        if source == 'overlay':
            title = 'Camera CUDA OVERLAY'
        elif source == 'mask':
            title = 'Camera MASK'
        elif line_fresh:
            title = 'Camera CUDA'
        else:
            title = 'Camera RAW'
        self._set_cell_title(
            '_cam_title_text', '_cam_title_label', title,
        )
        if overlay is not None and getattr(overlay, 'size', 0) > 0:
            if self._cam_lines_scatter is None:
                self._cam_lines_scatter = self._cam_ax.scatter(
                    overlay[:, 0], overlay[:, 1], s=4, c='red', marker='.',
                    linewidths=0, zorder=10,
                )
            else:
                self._cam_lines_scatter.set_offsets(overlay)
                self._cam_lines_scatter.set_visible(True)
        elif self._cam_lines_scatter is not None:
            self._cam_lines_scatter.set_visible(False)

        # FPS counter — IN = ROS-callback arrival rate for the currently
        # active source (raw or mask); OUT = rate this slot is firing at,
        # which is what the operator actually sees painted. With the
        # worker backpressure in place IN ≥ OUT always — if they diverge
        # the main thread paint is the bottleneck, not the camera.
        now = time.monotonic()
        self._cam_fps_deque.append(now)
        out_fps = 0.0
        if len(self._cam_fps_deque) >= 2:
            out_span = self._cam_fps_deque[-1] - self._cam_fps_deque[0]
            if out_span > 0:
                out_fps = (len(self._cam_fps_deque) - 1) / out_span
        in_fps = 0.0
        node = self._ros_node
        if node is not None:
            if source == 'overlay':
                in_ts = node._overlay_in_ts
            elif source == 'mask':
                in_ts = node._mask_in_ts
            else:
                in_ts = node._camera_in_ts
            if len(in_ts) >= 2:
                in_span = in_ts[-1] - in_ts[0]
                if in_span > 0:
                    in_fps = (len(in_ts) - 1) / in_span
        self._cam_fps_txt.set_text(f'IN {in_fps:4.1f}  OUT {out_fps:4.1f}')

        self._cam_canvas.draw_idle()

    def _apply_gui_fps_label_style(self):
        """Theme-aware foreground: white text in dark mode, black in
        light. Called on theme toggle + once at construction time. The
        per-tick update only changes setText, never the stylesheet, so
        color stays stable across paints.
        """
        if not hasattr(self, '_gui_fps_label') or self._gui_fps_label is None:
            return
        fg = '#ffffff' if self._theme == 'dark' else '#000000'
        self._gui_fps_label.setStyleSheet(
            f"border: none; color: {fg}; font-size: 10px;"
            f" font-family: monospace; padding: 0 8px;"
        )

    def _gui_fps_tick(self):
        """30 Hz QTimer tick. Records arrival time and updates the GUI
        FPS readout. Because QTimer is event-loop driven, late fires
        (caused by other slots monopolizing the main thread) directly
        lower the measured rate — that's the whole point.

        setText/f-string only fire when the integer FPS actually
        changes — saves ~25–30 redundant Qt label updates per second
        which themselves trigger layout invalidation.
        """
        now = time.monotonic()
        self._gui_fps_deque.append(now)
        if len(self._gui_fps_deque) < 2:
            return
        span = self._gui_fps_deque[-1] - self._gui_fps_deque[0]
        if span <= 0:
            return
        fps_int = int(round((len(self._gui_fps_deque) - 1) / span))
        if fps_int == getattr(self, '_gui_fps_last_int', -1):
            return
        self._gui_fps_last_int = fps_int
        self._gui_fps_label.setText(f"GUI {fps_int:2d} FPS")

    def _on_lidar_frame_ready(self, img, source):
        """Main-thread slot: receives a fully-rendered lidar BEV from
        the lidar worker. Heightband and PCA rendering both produced
        HxWx3 RGB arrays of the same size, so set_data without recreate
        is the steady-state path; recreate only fires if size changed.
        """
        # Same backpressure pause as the camera slot — leave the
        # worker's _slot_ready cleared in Performance Mode so it
        # drains submissions without rendering BEVs.
        if self._performance_mode:
            return
        self._lidar_worker.slot_ready()
        if not self._live_active:
            return
        self._lidar_live_txt.set_visible(False)
        if self._lidar_im is None:
            self._lidar_im = self._lidar_ax.imshow(img, aspect='equal')
        else:
            current = self._lidar_im.get_array()
            if current is None or current.shape != img.shape:
                self._lidar_im.remove()
                self._lidar_im = self._lidar_ax.imshow(img, aspect='equal')
            else:
                self._lidar_im.set_data(img)
        self._live_set_dot_received('Lidar')

        self._set_cell_title(
            '_lidar_title_text', '_lidar_title_label',
            'LIDAR PCA' if source == 'pca' else 'LIDAR Heightband',
        )

        # FPS overlay: IN = ROS-callback arrival rate for the active
        # lidar source; OUT = rate this slot is firing at.
        now = time.monotonic()
        self._lidar_fps_deque.append(now)
        out_fps = 0.0
        if len(self._lidar_fps_deque) >= 2:
            out_span = self._lidar_fps_deque[-1] - self._lidar_fps_deque[0]
            if out_span > 0:
                out_fps = (len(self._lidar_fps_deque) - 1) / out_span
        in_fps = 0.0
        node = self._ros_node
        if node is not None:
            in_ts = (node._pca_in_ts if source == 'pca'
                     else node._lidar_in_ts)
            if len(in_ts) >= 2:
                in_span = in_ts[-1] - in_ts[0]
                if in_span > 0:
                    in_fps = (len(in_ts) - 1) / in_span
        self._lidar_fps_txt.set_text(f'IN {in_fps:4.1f}  OUT {out_fps:4.1f}')

        self._lidar_canvas.draw_idle()

    def _on_tile_network_changed(self, online):
        """Main-thread slot: log network state transitions once each."""
        self._tile_network_online = bool(online)
        if online:
            if self._tile_offline_logged:
                self._gui_log_msg('GPS tile server reachable; map fetches re-enabled')
            self._tile_offline_logged = False
        else:
            if not self._tile_offline_logged:
                self._gui_log_msg('GPS tile server unreachable; map fetches deferred until reconnect')
                self._tile_offline_logged = True

    def _live_tick(self):
        """10 Hz poll: consume latest ROS data and update GUI cells."""
        node = self._ros_node
        if node is None:
            return  # Placeholders stay visible

        now = time.monotonic()
        t_s = now - self._live_t0
        any_scalar_changed = False

        # --- GPS ---
        gps = node.latest_gps
        if gps is not None:
            node.latest_gps = None
            lat, lon = gps
            self._gps_buf['lat'].append(lat)
            self._gps_buf['lon'].append(lon)
            # deque(maxlen=N) auto-trims on append — no manual slice.
            self._gps_live_txt.set_visible(False)
            self._live_set_dot_received('GPS')
            # Fetch map tiles on first point or if position leaves extent.
            # Both paths route through the off-thread fetcher; the result
            # arrives asynchronously in _on_tile_fetch_finished and does
            # not block the live tick.
            if self._gps_map_extent is None:
                self._request_gps_map([lat], [lon], 'live-first')
            else:
                e = self._gps_map_extent
                if not (e[2] <= lat <= e[3] and e[0] <= lon <= e[1]):
                    self._request_gps_map(
                        self._gps_buf['lat'], self._gps_buf['lon'], 'live-extent')
            # 2σ covariance ellipse around the current fix. Drawn in
            # degrees on the lat/lon axes — width/height come from the
            # eigendecomposition of the East/North block, converted
            # from meters to degrees with the local meridian scale.
            cov = node.latest_gps_cov
            self._set_cell_title(
                '_gps_title_text', '_gps_title_label',
                'GPS Covariance [fix]' if cov is not None else 'GPS',
            )
            if cov is not None:
                cov_ee, cov_en, cov_nn = cov
                trace = cov_ee + cov_nn
                det = cov_ee * cov_nn - cov_en * cov_en
                disc = max(0.0, trace * trace / 4.0 - det)
                root = math.sqrt(disc)
                lam1 = trace / 2.0 + root  # larger eigenvalue (m^2)
                lam2 = max(0.0, trace / 2.0 - root)
                # 2σ axes in meters.
                a_m = 2.0 * math.sqrt(max(0.0, lam1))
                b_m = 2.0 * math.sqrt(max(0.0, lam2))
                # Orientation of the major axis. atan2 against the
                # East/North block — angle measured CCW from East.
                angle_rad = 0.5 * math.atan2(2.0 * cov_en,
                                             cov_ee - cov_nn)
                angle_deg = math.degrees(angle_rad)
                m_per_deg_lat = 111320.0
                m_per_deg_lon = m_per_deg_lat * max(
                    1e-6, math.cos(math.radians(lat)))
                # We render on a lat/lon axis where 1 deg lon ≠ 1 deg
                # lat in meters; convert each principal axis as a
                # vector and use the diameter (2 × radius) for Ellipse.
                ca = math.cos(angle_rad)
                sa = math.sin(angle_rad)
                # Major-axis vector (a_m along the major dir) in deg.
                maj_dlon = (a_m * ca) / m_per_deg_lon
                maj_dlat = (a_m * sa) / m_per_deg_lat
                min_dlon = (-b_m * sa) / m_per_deg_lon
                min_dlat = (b_m * ca) / m_per_deg_lat
                width_deg = 2.0 * math.hypot(maj_dlon, maj_dlat)
                height_deg = 2.0 * math.hypot(min_dlon, min_dlat)
                if self._gps_cov_ellipse is None:
                    self._gps_cov_ellipse = Ellipse(
                        (lon, lat), width=width_deg, height=height_deg,
                        angle=angle_deg, facecolor='none',
                        edgecolor='#ff5050', linewidth=1.2, alpha=0.85,
                        zorder=4,
                    )
                    self._gps_ax.add_patch(self._gps_cov_ellipse)
                else:
                    self._gps_cov_ellipse.set_center((lon, lat))
                    self._gps_cov_ellipse.width = width_deg
                    self._gps_cov_ellipse.height = height_deg
                    self._gps_cov_ellipse.angle = angle_deg
                    self._gps_cov_ellipse.set_visible(True)
            elif self._gps_cov_ellipse is not None:
                self._gps_cov_ellipse.set_visible(False)
            any_scalar_changed = True

        # --- Odom: graduate raw /odom → /local_ekf/odom → +costmap.
        # Same buffer either way; the title flip is what tells the
        # operator which stream is on screen. The costmap tier also
        # turns on the global-costmap snippet overlay in _redraw_plots.
        ekf_odom_t = node.last_msg_t.get('ekf_local_odom', 0.0)
        ekf_odom_alive = (
            ekf_odom_t > 0.0
            and (now - ekf_odom_t) < self._ekf_msg_age_max_s
        )
        costmap_t = node.last_msg_t.get('costmap_global', 0.0)
        # nav2 publishes the global costmap at ~2 Hz, so allow a 2 s
        # window before declaring stale — survives a single missed
        # update without the overlay strobing on/off.
        self._costmap_alive = (
            costmap_t > 0.0
            and (now - costmap_t) < 2.0
            and node.latest_costmap is not None
        )
        # Plan freshness — nav2 republishes /plan whenever the global
        # planner runs (~1 Hz). 2 s tolerance same as the costmap.
        plan_t = node.last_msg_t.get('plan', 0.0)
        self._plan_alive = (
            plan_t > 0.0
            and (now - plan_t) < 2.0
            and node.latest_plan_xy is not None
        )
        if ekf_odom_alive and node.latest_ekf_odom is not None:
            odom = node.latest_ekf_odom
            node.latest_ekf_odom = None
            # Drop any raw /odom we picked up in the same tick — we
            # don't want to interleave the two streams in the trail.
            node.latest_odom = None
            if self._costmap_alive:
                self._set_enc_title("Encoders (Odom EKF Costmap)")
            else:
                self._set_enc_title("Encoders (Odom EKF)")
        else:
            odom = node.latest_odom
            if odom is not None:
                node.latest_odom = None
            self._set_enc_title("Encoders (Odom)")
        if odom is not None:
            self._odom_live_txt.set_visible(False)
            self._odom_ax.grid(True, which='both', color='#333', linewidth=0.5)
            self._odom_ax.tick_params(labelbottom=True, labelleft=True)
            self._odom_xy_label.set_visible(True)
            x, y, qz = odom
            qz_clamped = max(-1.0, min(1.0, qz))
            yaw = 2.0 * math.asin(qz_clamped)
            self._odom_buf['x'].append(x)
            self._odom_buf['y'].append(y)
            self._odom_buf['theta'].append(yaw)
            # deque(maxlen=N) auto-trims on append.
            self._live_set_dot_received('Encoders')
            any_scalar_changed = True

        # --- Power (Voltage / Current / Power) ---
        voltage = node.latest_voltage
        current = node.latest_current
        power = node.latest_power
        if voltage is not None or current is not None or power is not None:
            self._power_buf['t'].append(t_s)
            self._power_buf['V'].append(
                voltage if voltage is not None
                else (self._power_buf['V'][-1] if self._power_buf['V'] else 0)
            )
            self._power_buf['I'].append(
                current if current is not None
                else (self._power_buf['I'][-1] if self._power_buf['I'] else 0)
            )
            self._power_buf['P'].append(
                power if power is not None
                else (self._power_buf['P'][-1] if self._power_buf['P'] else 0)
            )
            node.latest_voltage = None
            node.latest_current = None
            node.latest_power = None
            self._trim_power_buf(t_s)
            self._pwr_v_live_txt.set_visible(False)
            self._pwr_i_live_txt.set_visible(False)
            self._pwr_p_live_txt.set_visible(False)
            self._live_set_dot_received('Power PCB')
            any_scalar_changed = True

        # --- SOC (from electrical publisher) ---
        soc_val = node.latest_soc
        if soc_val is not None:
            node.latest_soc = None
            self._latest_soc_pct = soc_val

        # --- Camera ---
        # Display is driven by the off-thread _CameraFrameWorker which the
        # ROS callback feeds directly at the source rate (~15 Hz). The
        # only thing we still do here is drain node.latest_image_rgb so
        # external probes that read it (e.g. screenshot path) don't see
        # stale data accumulate. The actual paint happens in
        # _on_camera_frame_ready when the worker emits.
        if node.latest_image_rgb is not None:
            node.latest_image_rgb = None

        # --- LiDAR ---
        # The BEV render is now driven entirely by _LidarFrameWorker via
        # the _on_lidar_frame_ready slot. Drain latest_scan so external
        # probes don't see a stale ref; everything else (paint, title,
        # FPS, dot) happens when the worker emits.
        if node.latest_scan is not None:
            node.latest_scan = None

        # --- Redraw scalar plots (throttled to ~3 Hz) ---
        if any_scalar_changed and (now - getattr(self, '_last_live_redraw', 0)) > 0.33:
            self._redraw_plots()
            self._last_live_redraw = now

        # NOTE: the EKF participation pulse used to run here. It now
        # has a dedicated always-on timer (self._ekf_status_timer in
        # __init__) so the pulse and the EKF Filters status rows
        # update whether or not Live mode is active.

    def _set_cell_title(self, attr_text, attr_label, text):
        """Update one of the sensor-cell title QLabels only when the
        text changes — cheap no-op on every tick that the state
        doesn't move, so we can call it every live tick without
        triggering Qt restyle work.
        """
        if getattr(self, attr_text, None) == text:
            return
        lbl = getattr(self, attr_label, None)
        if lbl is None:
            return
        lbl.setText(text)
        setattr(self, attr_text, text)

    def _set_enc_title(self, text):
        """Flip the Encoders cell title between '(Odom)' and
        '(Odom EKF)'. Cheap-no-op when the text is unchanged so the
        live tick can call this every iteration without churn.
        """
        if getattr(self, '_enc_title_text', None) == text:
            return
        lbl = getattr(self, '_enc_title_label', None)
        if lbl is None:
            return
        lbl.setText(text)
        self._enc_title_text = text

    def _ekf_pulse_tick(self, now_s):
        """Apply the green↔purple EKF-participation pulse to device
        dots whose input AND the EKF that consumes it are both fresh,
        and drive the "Local EKF" / "Map EKF" status rows from the
        freshness of their fused-output topics.

        REC mode owns the dots while a test recording is active —
        early-return so the participation pulse doesn't flicker
        against the black↔red ramp. Snapshot/restore in _rec_tick
        repaints the right colors on REC-off.

        Devices NOT participating in any EKF fall through this
        method untouched — their styling is owned by the existing
        _live_set_dot_received / _live_dot_flash_tick path. This
        means a freshly-running-but-not-fused device stays plain
        green, an EKF-fused device pulses, and a stalled device
        keeps its yellow flash. The transition between states is
        what tells the operator "the EKF just elevated this
        device" or "the EKF just lost it."
        """
        # REC ownership of dots — see method docstring.
        if getattr(self, '_rec_active', False):
            return
        node = self._ros_node
        if node is None:
            return
        stamps = getattr(node, 'last_msg_t', None)
        if stamps is None:
            return

        def _alive(key, max_age_s=None):
            if max_age_s is None:
                max_age_s = self._ekf_msg_age_max_s
            t = stamps.get(key, 0.0)
            return t > 0.0 and (now_s - t) < max_age_s

        # Hue interpolated along the green→purple arc by a sine
        # wave at ``_ekf_pulse_freq_hz``. Phase advanced by the
        # tick interval (5 Hz live tick → 0.2 s).
        self._ekf_pulse_phase = (
            (self._ekf_pulse_phase + 0.2 * self._ekf_pulse_freq_hz) % 1.0)
        t = 0.5 * (1.0 + math.sin(2.0 * math.pi * self._ekf_pulse_phase))
        hue = int(self._ekf_pulse_hue_lo
                  + (self._ekf_pulse_hue_hi - self._ekf_pulse_hue_lo) * t)
        pulse_style = (
            f"background-color: hsl({hue}, 100%, 60%); "
            "border-radius: 7px; border: none;")

        for dot_name, (input_key, ekf_key) in self._ekf_pulse_devices.items():
            dot = self.status_dots.get(dot_name)
            if dot is None:
                continue
            # Per-device input-freshness override — slow inputs (GPS)
            # use a longer window so the dot doesn't flicker between
            # fixes. The EKF output side always uses the default
            # window because it should be high-rate regardless.
            input_max_age = self._ekf_pulse_input_max_age_s.get(
                dot_name, self._ekf_msg_age_max_s)
            if _alive(input_key, input_max_age) and _alive(ekf_key):
                dot.setStyleSheet(pulse_style)
            # Else: leave whatever the prior tick set — the existing
            # alive/flash logic owns the dot's appearance in those
            # states.

        # EKF status rows: solid green while the filter is publishing,
        # gray when stale. These don't pulse — they're the "is this
        # filter alive?" answer, mirroring the device-list rows above.
        # Cache last alive state per row so setStyleSheet only fires on
        # transitions; without this every 5 Hz tick re-applies the same
        # stylesheet on every row, triggering Qt style invalidation for
        # no visible change.
        if not hasattr(self, '_ekf_row_last_alive'):
            self._ekf_row_last_alive = {}
        for dot_name, ekf_key in self._ekf_status_rows.items():
            dot = self.status_dots.get(dot_name)
            if dot is None:
                continue
            alive = _alive(ekf_key)
            if self._ekf_row_last_alive.get(dot_name) == alive:
                continue
            self._ekf_row_last_alive[dot_name] = alive
            dot.setStyleSheet(self._DOT_ON if alive else self._DOT_OFF)

    @staticmethod
    def _render_lidar_bev(scan, size=480, pca_xy=None):
        """Render a LaserScan as a bird's-eye-view RGB image.

        Gray background = outside lidar range / unknown.
        White = driveable (clear path from robot to hit).
        Black = obstacle shadow (from hit outward to max range).
        Green dots = hit points. Red dot = robot origin.

        ``pca_xy`` is an optional (N, 2) array of obstacle XY positions
        in the same frame as the scan (lidar_footprint). When provided,
        they are drawn as 3x3 red squares on top of everything else —
        this is the PCA obstacle overlay.
        """
        img = np.full((size, size, 3), 128, dtype=np.uint8)  # gray background
        cx, cy = size // 2, size // 2

        # Determine scale: fit max range into half the canvas
        max_range = scan.range_max
        if max_range <= 0 or not np.isfinite(max_range):
            max_range = 10.0
        scale = (size // 2 - 2) / max_range

        angles = np.arange(len(scan.ranges)) * scan.angle_increment + scan.angle_min
        ranges = np.array(scan.ranges, dtype=np.float32)

        # Cap for "no detection" rays: draw white line to this distance
        no_detect_range = min(10.0, max_range)

        for i in range(len(ranges)):
            r = ranges[i]
            a = angles[i]
            if not np.isfinite(r) or r < scan.range_min:
                continue  # NaN/inf = no data, leave as gray
            # Max range endpoint along this ray
            sx = int(cx + max_range * math.cos(a) * scale)
            sy = int(cy - max_range * math.sin(a) * scale)
            if r >= max_range:
                # Clear to max range — entire ray is driveable (white)
                _bresenham_line(img, cx, cy, sx, sy, (255, 255, 255))
            else:
                # Hit point
                ex = int(cx + r * math.cos(a) * scale)
                ey = int(cy - r * math.sin(a) * scale)
                # White line: robot to hit (driveable space)
                _bresenham_line(img, cx, cy, ex, ey, (255, 255, 255))
                # Black line: hit to max range (obstacle shadow)
                _bresenham_line(img, ex, ey, sx, sy, (0, 0, 0))
                # Green hit dot
                if 0 <= ex < size and 0 <= ey < size:
                    img[ey, ex] = (0, 255, 0)

        # PCA obstacle overlay — drawn before the robot dot so the
        # origin stays distinguishable when obstacles are clustered
        # near the robot.
        if pca_xy is not None and len(pca_xy) > 0:
            for px, py in pca_xy:
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                ex = int(cx + px * scale)
                ey = int(cy - py * scale)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = ey + dy, ex + dx
                        if 0 <= ny < size and 0 <= nx < size:
                            img[ny, nx] = (255, 0, 0)

        # Robot origin (red dot)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < size and 0 <= nx < size:
                    img[ny, nx] = (255, 0, 0)

        return img


if _HAS_ROS:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

    from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

    _SENSOR_QOS = qos_profile_sensor_data
    _RELIABLE_QOS = qos_profile_system_default

    class HudNode(Node):
        """ROS2 node for the AutoNav HUD with live sensor subscriptions."""

        def __init__(self):
            super().__init__('autonav_hud')
            self.get_logger().info('AutoNav HUD node started')

            # Latest values (consumed by _live_tick on the GUI thread)
            self.latest_image_rgb = None   # numpy RGB array
            self.latest_scan = None        # LaserScan message
            self.latest_gps = None         # (lat, lon)
            # GPS position covariance (3x3, ENU, in m^2). Stashed
            # separately from latest_gps so the live tick can draw the
            # 2σ ellipse without changing the fix-consumption path.
            self.latest_gps_cov = None     # (cov_ee, cov_en, cov_nn) m^2
            # Detected line pixels overlay (cleared by the live tick
            # once consumed; the topic-staleness check in HudWindow
            # decides whether to draw).
            self.latest_line_pixels = None # (xs, ys, width, height)
            self.latest_line_pixels_t = 0.0
            # Set by HudWindow once it constructs the camera worker.
            # _cb_image hands frames here directly so the camera updates
            # at the ROS publish rate, not the live-tick poll rate.
            self.camera_worker = None
            # Receipt time of the most recent /line_detection/debug/mask
            # message. _cb_image checks this to decide whether to submit
            # the raw camera (mask is silent) or defer to the mask path
            # (detector is active).
            self.latest_mask_image_t = 0.0
            # Receipt time of the most recent /line_detection/debug/overlay
            # message — the raw camera frame with red/yellow/green dots
            # already drawn by the detector (see line/node.cpp ~L1114-1138).
            # _cb_image / _cb_mask_image both defer to this when fresh so
            # the overlay variant wins the precedence at the camera panel.
            self.latest_overlay_image_t = 0.0
            # Per-source IN-rate timestamp deques. Each ROS callback
            # appends the wall-clock time of arrival; the GUI reads these
            # to render the "IN" half of the camera FPS overlay so the
            # operator can verify the source is truly running at the
            # expected rate (15 Hz for /zed/.../image, 4 Hz for the
            # line-detector debug mask under publish_interval_ms: 250).
            self._camera_in_ts = deque(maxlen=20)
            self._mask_in_ts = deque(maxlen=20)
            self._overlay_in_ts = deque(maxlen=20)

            # Lidar worker, set by HudWindow once the worker is built.
            # _cb_scan (heightband) and _cb_pca_scan (PCA) submit
            # directly to it so the BEV renders at the lidar source
            # rate (~20 Hz) without going through the 5 Hz live tick.
            self.lidar_worker = None
            self._lidar_in_ts = deque(maxlen=20)
            self._pca_in_ts = deque(maxlen=20)
            # PCA-filtered lidar obstacles (PointCloud2 → list of
            # (x, y) tuples in the lidar_footprint frame).
            self.latest_pca_xy = None      # numpy (N, 2) array
            self.latest_pca_t = 0.0
            self.latest_odom = None        # (x, y, qz) — raw /odom
            self.latest_ekf_odom = None    # (x, y, qz) — /local_ekf/odom (filtered)
            # Persistent (never drained) copy of the EKF-fused robot XY
            # in odom frame. The live tick drains ``latest_ekf_odom``
            # to None each iteration, so the global-costmap callback
            # needs its own latched copy to know where to crop.
            self.latest_ekf_pose_xy = None  # (x, y) in odom frame
            # Global costmap as a pre-rendered (rgba, extent) pair. The
            # ROS callback crops a robot-centered snippet of the global
            # costmap (so the persistent lines + map_padder corridor
            # are visible, but only the patch near the robot) and
            # converts to grayscale RGBA. The GUI thread only does
            # set_data/set_extent on an existing AxesImage.
            #   rgba:   uint8 (H, W, 4) — grayscale, alpha baked in
            #   extent: (x_min, x_max, y_min, y_max) in map frame metres
            #           — treated as ≈ odom on the GUI's odom canvas;
            #           map↔odom drift is normally < 1 m in a healthy
            #           mission so the snippet reads under the red
            #           triangle.
            self.latest_costmap = None
            # NAV2 global plan as a pair of equal-length 1D numpy
            # arrays (xs, ys) in map frame. Drawn as a green Line2D
            # over the odom canvas. None until first /plan arrives;
            # cleared when /plan goes stale.
            self.latest_plan_xy = None  # Optional[Tuple[np.ndarray, np.ndarray]]
            self.latest_voltage = None     # float
            self.latest_current = None     # float
            self.latest_power = None       # float
            self.latest_soc = None         # float (0-100%)

            # ── EKF participation tracking (per-message wall-clock
            # timestamps). The HudWindow's pulse tick reads this to
            # decide which device dots are "elevated" (= the local
            # robot_localization EKF is currently fusing this
            # device's data). A device is considered participating
            # iff (a) its raw input topic is fresh AND (b)
            # /local_ekf/odom is fresh — i.e. the EKF is alive AND
            # consuming this device. Stays at 0.0 until first
            # message; the pulse skips devices whose stamp == 0.0.
            self.last_msg_t = {
                'image':           0.0,   # /zed/zed_node/rgb/...
                'scan':            0.0,   # /scan_fullframe
                'gps':             0.0,   # /gps_fix
                'odom':            0.0,   # /odom (raw wheel)
                'imu':             0.0,   # /sick_scansegment_xd/imu (SICK lidar IMU)
                'pose':            0.0,   # /pose (slam_toolbox → Map EKF)
                'ekf_local_odom':  0.0,   # /local_ekf/odom (Local EKF out)
                'ekf_global_odom': 0.0,   # /global_ekf/odom (Map EKF out)
                'costmap_global':  0.0,   # /global_costmap/costmap (nav2)
                'plan':            0.0,   # /plan (nav2 global planner)
            }

            self._cv_bridge = CvBridge() if _HAS_CV_BRIDGE else None

            self.create_subscription(
                Image,
                '/zed/zed_node/rgb/color/rect/image',
                self._cb_image, 10,
            )
            # Debug MASK from the line detector. Single-channel binary
            # threshold image; gated on subscriber count on the publisher
            # side so subscribing here is what makes the detector start
            # publishing it. When this stream goes silent (>1 s) _cb_image
            # resumes feeding the raw camera to the worker.
            self.create_subscription(
                Image,
                '/line_detection/debug/mask',
                self._cb_mask_image, 10,
            )
            # Debug OVERLAY from the line detector — the raw camera frame
            # with red dots at every detected line pixel and yellow/green
            # dots for accepted/confirmed candidates (see line/node.cpp
            # ~L1114-1138). Same subscriber-count gating as the mask, so
            # subscribing here is what makes the detector publish it.
            # When this stream is fresh (1 s window) _cb_image AND
            # _cb_mask_image both defer to this variant — the dots are
            # already baked into the frame so the post-paint line_pixels
            # scatter is also skipped in _on_camera_frame_ready.
            self.create_subscription(
                Image,
                '/line_detection/debug/overlay',
                self._cb_overlay_image, 10,
            )
            self.create_subscription(
                LaserScan, '/scan_fullframe', self._cb_scan, _SENSOR_QOS,
            )
            self.create_subscription(
                NavSatFix, '/gps_fix', self._cb_gps, _SENSOR_QOS,
            )
            self.create_subscription(
                Odometry, '/odom', self._cb_odom, _SENSOR_QOS,
            )
            self.create_subscription(
                Float32, '/electrical/voltage', self._cb_voltage, _SENSOR_QOS,
            )
            self.create_subscription(
                Float32, '/electrical/current', self._cb_current, _SENSOR_QOS,
            )
            self.create_subscription(
                Float32, '/electrical/power', self._cb_power, _SENSOR_QOS,
            )
            self.create_subscription(
                Float32, '/electrical/soc', self._cb_soc, _SENSOR_QOS,
            )
            # ── EKF participation: monitor the EKF's input streams
            # and its output. The IMU input feeding ekf_local is the
            # SICK multiScan's onboard IMU at /sick_scansegment_xd/imu
            # — the SICK driver hardcodes its ROS2 node name to
            # ``sick_scansegment_xd`` and advertises IMU under
            # ``<nodename>/imu``. (Earlier code subscribed to
            # /multiScan/imu based on the launch file's ``nodename``
            # default, but the running driver ignores that arg —
            # verified via ros2 topic info, 0 publishers on
            # /multiScan/imu.) The ZED camera's IMU is excluded for
            # now: its TF chain to base_link is not currently
            # bridged, so robot_localization rejects every ZED IMU
            # message. When the URDF/launch is fixed to publish
            # base_link → zed2i_imu_link, add a second subscription
            # here and a corresponding dot mapping.
            self.create_subscription(
                Imu, '/sick_scansegment_xd/imu', self._cb_imu, _SENSOR_QOS,
            )
            self.create_subscription(
                Odometry, '/local_ekf/odom',
                self._cb_ekf_local_odom, _SENSOR_QOS,
            )
            # Map EKF participation: /pose is slam_toolbox's localization
            # output and is the Map EKF's pose0 input (ekf_global.yaml).
            # /global_ekf/odom is the Map EKF's fused output. The GUI's
            # SLAM dot pulses iff both are fresh; the "Map EKF" status
            # row goes solid green iff /global_ekf/odom is fresh.
            self.create_subscription(
                PoseWithCovarianceStamped, '/pose',
                self._cb_pose, _SENSOR_QOS,
            )
            self.create_subscription(
                Odometry, '/global_ekf/odom',
                self._cb_ekf_global_odom, _SENSOR_QOS,
            )
            # PCA-filtered lidar obstacles. Single source — the 2D
            # LaserScan that pca_cloud_to_laserscan publishes in
            # slam.launch.py (target_frame: base_link). Mirroring the
            # heightband path's "one input, one frame" design so the
            # BEV dots can't flip-flop. Requires SLAM (specifically
            # pca_cloud_to_laserscan) to be running.
            self.create_subscription(
                LaserScan, '/scan_pca_filtered',
                self._cb_pca_scan, _SENSOR_QOS,
            )
            # 2D line-pixel array from the line detector (red dots on
            # the camera image). Published as std_msgs/Int32MultiArray
            # so the native HUD only needs /opt/ros/humble on its
            # Python path — no custom interface dependency. Wire
            # format: data[0]=width, data[1]=height, data[2:] is
            # interleaved [x0, y0, x1, y1, ...].
            self.create_subscription(
                Int32MultiArray, '/line_detection/line_pixels',
                self._cb_line_pixels, _SENSOR_QOS,
            )

            # nav2's global_costmap. Publisher QoS is RELIABLE +
            # TRANSIENT_LOCAL + depth 1 (nav2_costmap_2d's standard
            # latched publisher), so subscribe with the matching
            # profile or rclpy will warn "incompatible QoS, no messages
            # will be received". The TRANSIENT_LOCAL on our side means
            # we get the most recent costmap immediately on subscribe
            # rather than waiting for the next 2 Hz tick.
            #
            # Global (not local) because the line_layer + map_padder
            # corridor only persist there — the local costmap is a
            # rolling 5 m window that wipes line memory the moment the
            # robot drives past. We crop a robot-centered snippet of
            # the global costmap in the callback so the GUI overlay
            # stays visually local while the data comes from the
            # persistent global side.
            #
            # All nav2 configs in this repo set
            # `always_send_full_costmap: true`, so the full grid lands
            # on every publish — no need to fold in /costmap_updates.
            _COSTMAP_QOS = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                OccupancyGrid, '/global_costmap/costmap',
                self._cb_global_costmap, _COSTMAP_QOS,
            )
            # NAV2 global plan (green line on the odom canvas). Same
            # RELIABLE QoS as the planner publishes with.
            self.create_subscription(
                Path, '/plan',
                self._cb_plan, _RELIABLE_QOS,
            )

            # Auto-mode wiring. The control node toggles its internal
            # autonomousMode on a rising edge of /joy buttons[3] (X
            # button on the Xbox controller) and publishes the result
            # to /autonomous_mode. The GUI publishes a fake X-button
            # press to /joy when the operator presses 'A' on the
            # keyboard, and subscribes to /autonomous_mode to keep the
            # corner badge in sync with the actual robot state — same
            # pattern as t002_automator._send_x_button_to_control.
            self.latest_autonomous_mode = None  # None until first /autonomous_mode msg
            self.joy_pub = self.create_publisher(Joy, '/joy', 10)
            self.create_subscription(
                Bool, '/autonomous_mode',
                self._cb_autonomous_mode, _RELIABLE_QOS,
            )

            # Test recording state — base_automator publishes True on
            # the A-button press that starts a test, False on the press
            # that ends it. The HUD itself also publishes here from the
            # R key (see HudWindow._request_record_toggle) so an operator
            # can record without a test script. Either source drives the
            # GUI's REC indicator: device-dot black↔red ramp + transparent
            # red overlay over the central widget.
            self.latest_recording_active = False
            self.create_subscription(
                Bool, '/data/toggle_collect',
                self._cb_recording_toggle, _RELIABLE_QOS,
            )
            self.toggle_collect_pub = self.create_publisher(
                Bool, '/data/toggle_collect', _RELIABLE_QOS,
            )

            # Xbox D-pad → GUI arrow keys. HudWindow assigns
            # ``joy_nav_bus`` after constructing the bus; until then
            # the callback no-ops. The D-pad lives on axes[6] (left=+1,
            # right=-1) and axes[7] (up=+1, down=-1) — verified in the
            # field with a controller dump. We hold previous axis
            # values to fire on rising-edge crossings only, so holding
            # the D-pad doesn't auto-repeat (matches single-key
            # arrow-press semantics).
            self.joy_nav_bus = None
            self._joy_prev_dpad_x = 0.0
            self._joy_prev_dpad_y = 0.0
            self._joy_prev_select_btn = 0
            self.create_subscription(
                Joy, '/joy',
                self._cb_joy_nav, _SENSOR_QOS,
            )

        def _cb_image(self, msg):
            self.last_msg_t['image'] = time.monotonic()
            self._camera_in_ts.append(self.last_msg_t['image'])
            try:
                channels = max(1, msg.step // msg.width) if msg.width > 0 else 3
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape((msg.height, msg.width, channels))
                if msg.encoding == 'bgra8':
                    img = img[:, :, [2, 1, 0]]  # BGRA to RGB
                elif msg.encoding in ('bgr8', 'bgr16'):
                    img = img[:, :, ::-1]  # BGR to RGB
                elif msg.encoding == 'rgba8':
                    img = img[:, :, :3]  # RGBA to RGB
                self.latest_image_rgb = img
                worker = getattr(self, 'camera_worker', None)
                if worker is None:
                    return
                # Prefer the line detector's debug OVERLAY when fresh —
                # it's the raw frame with the detector's red/yellow/green
                # dots already drawn, so it's the most informative camera
                # variant. _cb_overlay_image drives the display on that
                # path; we silently drop the raw frame here so the worker
                # isn't fighting two source streams for the same panel.
                now_t = time.monotonic()
                if (now_t - self.latest_overlay_image_t) < 1.0:
                    return
                # Otherwise prefer the line detector's debug MASK when
                # it's actively publishing — single-channel and matches
                # what the detector "sees" as line. _cb_mask_image drives
                # the display on that path; the raw camera is the
                # fallback only when both overlay and mask are silent.
                if (now_t - self.latest_mask_image_t) < 1.0:
                    return
                worker.submit(img, None, False, source='raw')
            except Exception:
                pass

        def _cb_mask_image(self, msg):
            """/line_detection/debug/mask — MONO8 binary mask from the
            detector's brightness threshold. Hand straight to the camera
            worker; the slot will imshow it with cmap='gray'. Deferred to
            the overlay variant when /line_detection/debug/overlay is
            fresh — overlay sits above mask in the panel precedence.
            """
            self.latest_mask_image_t = time.monotonic()
            self._mask_in_ts.append(self.latest_mask_image_t)
            try:
                # If the overlay variant is publishing, let it drive the
                # panel — same fight-for-the-source guard as in _cb_image.
                if (self.latest_mask_image_t
                        - self.latest_overlay_image_t) < 1.0:
                    return
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape((msg.height, msg.width))
                worker = getattr(self, 'camera_worker', None)
                if worker is not None:
                    worker.submit(img, None, True, source='mask')
            except Exception:
                pass

        def _cb_overlay_image(self, msg):
            """/line_detection/debug/overlay — BGR8 raw camera frame with
            the detector's red/yellow/green dots already drawn (see
            line/node.cpp ~L1114-1138). Decode like _cb_image (BGR→RGB)
            and hand to the camera worker with source='overlay' so the
            slot suppresses the post-paint line_pixels scatter (the dots
            are already baked in) and uses the 'Camera CUDA OVERLAY'
            title.
            """
            self.latest_overlay_image_t = time.monotonic()
            self._overlay_in_ts.append(self.latest_overlay_image_t)
            try:
                channels = max(1, msg.step // msg.width) if msg.width > 0 else 3
                img = np.frombuffer(msg.data, dtype=np.uint8)
                img = img.reshape((msg.height, msg.width, channels))
                if msg.encoding == 'bgra8':
                    img = img[:, :, [2, 1, 0]]  # BGRA to RGB
                elif msg.encoding in ('bgr8', 'bgr16'):
                    img = img[:, :, ::-1]  # BGR to RGB
                elif msg.encoding == 'rgba8':
                    img = img[:, :, :3]  # RGBA to RGB
                worker = getattr(self, 'camera_worker', None)
                if worker is not None:
                    worker.submit(img, None, False, source='overlay')
            except Exception:
                pass

        def _cb_scan(self, msg):
            self.last_msg_t['scan'] = time.monotonic()
            self._lidar_in_ts.append(self.last_msg_t['scan'])
            self.latest_scan = msg
            worker = getattr(self, 'lidar_worker', None)
            if worker is None:
                return
            # Defer to the PCA path whenever the grade detector is
            # actively publishing /scan_pca_filtered_points (same
            # mask/raw fallback pattern as the camera). Heightband only
            # renders when the PCA stream has been silent for >1 s.
            if (time.monotonic() - self.latest_pca_t) < 1.0:
                return
            worker.submit(msg, source='heightband')

        def _cb_gps(self, msg):
            self.last_msg_t['gps'] = time.monotonic()
            self.latest_gps = (msg.latitude, msg.longitude)
            # position_covariance is row-major 3x3 in ENU (m^2). Pull
            # the 2x2 East/North block — that's what the GUI's lat/lon
            # map view can render as a 2σ ellipse.
            cov = msg.position_covariance
            if cov is not None and len(cov) >= 5:
                cov_ee = float(cov[0])
                cov_en = float(cov[1])
                cov_nn = float(cov[4])
                if (math.isfinite(cov_ee) and math.isfinite(cov_en)
                        and math.isfinite(cov_nn) and cov_ee > 0.0
                        and cov_nn > 0.0):
                    self.latest_gps_cov = (cov_ee, cov_en, cov_nn)

        def _cb_odom(self, msg):
            self.last_msg_t['odom'] = time.monotonic()
            p = msg.pose.pose.position
            qz = msg.pose.pose.orientation.z
            self.latest_odom = (p.x, p.y, qz)

        def _cb_imu(self, msg):
            # SICK multiScan onboard IMU (the EKF's only IMU input
            # on Bowser today). Stamp drives the Lidar dot's EKF-
            # participation pulse in HudWindow's _ekf_pulse_tick —
            # pulses only while /local_ekf/odom is also fresh.
            self.last_msg_t['imu'] = time.monotonic()

        def _cb_ekf_local_odom(self, msg):
            # The local robot_localization EKF's output. When
            # this stops publishing, every "EKF-participating"
            # pulse on the GUI freezes to its last hue and the
            # device dots fall back to their plain-alive style —
            # the operator's "the EKF is dead" cue.
            #
            # We also stash the filtered pose so the Encoders cell
            # can graduate from raw /odom to the EKF stream — the
            # live tick prefers latest_ekf_odom whenever this topic
            # is fresh.
            self.last_msg_t['ekf_local_odom'] = time.monotonic()
            p = msg.pose.pose.position
            qz = msg.pose.pose.orientation.z
            self.latest_ekf_odom = (p.x, p.y, qz)
            # Persistent (never drained) copy for the global-costmap
            # crop centering. Atomic tuple swap under the GIL.
            self.latest_ekf_pose_xy = (p.x, p.y)

        def _cb_pose(self, msg):
            # slam_toolbox's localized pose. Drives the SLAM dot's
            # EKF-participation pulse — pulses iff /global_ekf/odom
            # is also fresh (i.e. the Map EKF is fusing this pose).
            self.last_msg_t['pose'] = time.monotonic()

        def _cb_ekf_global_odom(self, msg):
            # The global (map-frame) robot_localization EKF's output.
            # Drives the "Map EKF" status dot and is the second-half
            # gate for the SLAM dot's pulse.
            self.last_msg_t['ekf_global_odom'] = time.monotonic()

        def _cb_global_costmap(self, msg):
            # /global_costmap/costmap — nav2 global costmap (whatever
            # the map_padder corridor currently covers; can be 10s of
            # metres on a side, 0.10 m/cell). We crop a ±3 m window
            # around the EKF-fused robot pose so the GUI overlay reads
            # local-sized but inherits the global costmap's persistent
            # line_layer + map_padder corridor. Without the crop, a
            # mission-scale 100×100 costmap rasterises to a tiny patch
            # under the red triangle; with the crop it fills the
            # auto-pan window the same way the local costmap did.
            #
            # Conversion is ~0.5 ms of vectorised numpy on a 60×60
            # crop. At the global costmap's 2 Hz publish rate the ROS
            # thread cost is well under 2 ms/sec.
            self.last_msg_t['costmap_global'] = time.monotonic()
            try:
                h = int(msg.info.height)
                w = int(msg.info.width)
                if h <= 0 or w <= 0:
                    return
                res = float(msg.info.resolution)
                if res <= 0.0:
                    return
                ox = float(msg.info.origin.position.x)
                oy = float(msg.info.origin.position.y)

                # Crop centre — robot's EKF-fused odom-frame XY,
                # treated as map ≈ odom for visualisation. If the EKF
                # hasn't published yet just skip; next callback will
                # try again.
                robot_xy = self.latest_ekf_pose_xy
                if robot_xy is None:
                    return
                rx, ry = robot_xy

                # ±3 m half-width — matches the local costmap's
                # default extent so the operator gets the same
                # visual scale they're used to.
                crop_half = 3.0
                col_min = max(0, int((rx - crop_half - ox) / res))
                col_max = min(w, int(math.ceil((rx + crop_half - ox) / res)))
                row_min = max(0, int((ry - crop_half - oy) / res))
                row_max = min(h, int(math.ceil((ry + crop_half - oy) / res)))
                if col_max <= col_min or row_max <= row_min:
                    # Robot's outside the global costmap entirely —
                    # nothing useful to draw this tick.
                    return

                # np.frombuffer reads array.array's buffer-protocol view
                # directly — no copy. msg.data only needs to outlive this
                # callback (it does; we only persist the freshly-allocated
                # rgba below, not any view onto msg).
                raw = np.frombuffer(msg.data, dtype=np.int8)
                if raw.size != h * w:
                    return
                grid = raw.reshape((h, w))
                # Slice is a view, not a copy — bytes are still owned by
                # the msg buffer, so the np.clip etc. below need to run
                # before the callback returns (which they do).
                cropped = grid[row_min:row_max, col_min:col_max]
                ch, cw = cropped.shape

                # OccupancyGrid encoding:
                #   -1   = unknown   → fully transparent
                #    0   = free      → faint dark grey
                #   1-100 = cost     → linear ramp toward near-white solid
                #
                # Grayscale keeps the red triangle as the only red on the
                # canvas. We emit uint8 RGBA so matplotlib's imshow does
                # zero norm/cmap work — the conversion is fully done here.
                clipped = np.clip(cropped, 0, 100).astype(np.uint16)
                gray = (40 + clipped * 200 // 100).astype(np.uint8)
                alpha = (60 + clipped * 160 // 100).astype(np.uint8)
                unknown = cropped < 0
                gray[unknown] = 0
                alpha[unknown] = 0

                rgba = np.empty((ch, cw, 4), dtype=np.uint8)
                rgba[..., :3] = gray[..., None]  # broadcast into R, G, B
                rgba[..., 3] = alpha

                extent = (
                    ox + col_min * res,
                    ox + col_max * res,
                    oy + row_min * res,
                    oy + row_max * res,
                )
                self.latest_costmap = (rgba, extent)
            except Exception:
                # Don't let a malformed costmap take down the GUI;
                # leaving the previous image in place is fine.
                pass

        def _cb_plan(self, msg):
            # /plan — nav2's global planner output. Drawn as a green
            # Line2D on the odom canvas (map frame, treated as ≈ odom
            # for visualisation). Empty/zero-length plans clear the
            # line; the freshness gate in the live tick hides it when
            # /plan goes silent.
            self.last_msg_t['plan'] = time.monotonic()
            try:
                poses = msg.poses
                n = len(poses)
                if n == 0:
                    self.latest_plan_xy = None
                    return
                xs = np.empty(n, dtype=np.float32)
                ys = np.empty(n, dtype=np.float32)
                for i, ps in enumerate(poses):
                    xs[i] = ps.pose.position.x
                    ys[i] = ps.pose.position.y
                self.latest_plan_xy = (xs, ys)
            except Exception:
                # Same defensive policy as the costmap callback —
                # never let a malformed path kill the GUI.
                pass

        def _cb_voltage(self, msg):
            self.latest_voltage = msg.data

        def _cb_current(self, msg):
            self.latest_current = msg.data

        def _cb_power(self, msg):
            self.latest_power = msg.data

        def _cb_soc(self, msg):
            self.latest_soc = msg.data

        def _cb_pca_scan(self, msg):
            """/scan_pca_filtered (2D LaserScan from
            pca_cloud_to_laserscan, in base_link frame). Single PCA
            input — mirroring heightband's one-input model so the BEV
            dots can't flip between frames. Polar → Cartesian for the
            valid hits, then hand the (N, 2) array to the lidar worker.
            """
            now = time.monotonic()
            self.latest_pca_t = now
            self._pca_in_ts.append(now)
            try:
                ranges = np.asarray(msg.ranges, dtype=np.float32)
                if ranges.size == 0:
                    xy = np.empty((0, 2), dtype=np.float32)
                else:
                    angles = (np.arange(ranges.size, dtype=np.float32)
                              * msg.angle_increment + msg.angle_min)
                    rmax = (msg.range_max
                            if msg.range_max and msg.range_max > 0 else 10.0)
                    valid = (np.isfinite(ranges) &
                             (ranges >= msg.range_min) & (ranges < rmax))
                    if not np.any(valid):
                        xy = np.empty((0, 2), dtype=np.float32)
                    else:
                        rs = ranges[valid]
                        as_ = angles[valid]
                        # Y sign flipped to match the heightband view's
                        # left/right orientation on screen. The PCA
                        # LaserScan comes in base_link frame while
                        # /scan_fullframe is in lidar_footprint —
                        # whatever yaw mismatch exists between those
                        # frames was inverting the rendered y axis for
                        # the PCA path only. Negating sin() here mirrors
                        # the points horizontally before the worker's
                        # rot90+fliplr orientation pass, so they land
                        # on the same screen side as the equivalent
                        # heightband hits.
                        xy = np.column_stack([
                            (rs * np.cos(as_)).astype(np.float32),
                            -(rs * np.sin(as_)).astype(np.float32),
                        ])
                self.latest_pca_xy = xy
                worker = getattr(self, 'lidar_worker', None)
                if worker is not None:
                    worker.submit(xy, source='pca')
            except Exception:
                pass

        def _cb_line_pixels(self, msg):
            data = msg.data
            if len(data) < 2:
                return
            w = int(data[0])
            h = int(data[1])
            pairs = np.asarray(data[2:], dtype=np.int32)
            # data[2:] is interleaved [x0, y0, x1, y1, ...]; drop the
            # tail if a malformed message arrives with an odd count.
            if pairs.size % 2 != 0:
                pairs = pairs[:-1]
            if pairs.size == 0:
                xs = np.empty(0, dtype=np.int32)
                ys = np.empty(0, dtype=np.int32)
            else:
                pairs = pairs.reshape(-1, 2)
                xs = pairs[:, 0]
                ys = pairs[:, 1]
            self.latest_line_pixels = (xs, ys, w, h)
            self.latest_line_pixels_t = time.monotonic()

        def _cb_autonomous_mode(self, msg):
            # control.cpp publishes this whenever autonomousMode flips
            # (and once on startup with the initial value). HudWindow's
            # 5 Hz sync tick reads this and updates the corner badge,
            # so the GUI badge always reflects the *actual* robot state
            # rather than what we last published.
            self.latest_autonomous_mode = bool(msg.data)

        def _cb_recording_toggle(self, msg):
            # base_automator.start_test/stop_test publish True/False
            # here. HudWindow's REC-ramp timer reads this every tick.
            self.latest_recording_active = bool(msg.data)

        def _cb_joy_nav(self, msg):
            """Xbox D-pad → arrow-key signals on the nav bus.

            D-pad layout (verified in field with /joy dumps):
              axes[6]: left=+1.0, right=-1.0, neutral=0.0
              axes[7]: up=+1.0, down=-1.0, neutral=0.0

            Pure edge detection: emit only on the 0→±1 transition so
            holding the D-pad doesn't auto-repeat. A configurable
            "select" button (TBD) will get the same treatment, gated
            on NOT in auto mode AND NOT in test recording so it
            doesn't fight the t000 record toggle or surprise the
            autonomy stack.
            """
            bus = self.joy_nav_bus
            if bus is None:
                return
            axes = msg.axes
            if len(axes) > 7:
                dx = float(axes[6])
                dy = float(axes[7])
                # 0.5 threshold filters analog jitter; the D-pad is
                # effectively digital so values are basically -1/0/+1.
                if dy > 0.5 and self._joy_prev_dpad_y <= 0.5:
                    bus.push('up')
                elif dy < -0.5 and self._joy_prev_dpad_y >= -0.5:
                    bus.push('down')
                if dx > 0.5 and self._joy_prev_dpad_x <= 0.5:
                    bus.push('left')
                elif dx < -0.5 and self._joy_prev_dpad_x >= -0.5:
                    bus.push('right')
                self._joy_prev_dpad_x = dx
                self._joy_prev_dpad_y = dy
            # Select button. Index 15 — verified in the field with a
            # /joy dump (17-button array, button[15]=1 on press).
            # Picked specifically to avoid A (button[0]) which t000
            # already uses for "start/stop recording". Gated on NOT
            # in auto mode AND NOT in test recording so we never
            # synthesise an Enter that could trigger something
            # during autonomy or a live test capture.
            SELECT_BTN_INDEX = 15
            if len(msg.buttons) > SELECT_BTN_INDEX:
                b = int(msg.buttons[SELECT_BTN_INDEX])
                if (b == 1
                        and self._joy_prev_select_btn == 0
                        and not self.latest_autonomous_mode
                        and not self.latest_recording_active):
                    bus.push('select')
                self._joy_prev_select_btn = b


def main(args=None):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    node = None
    spin_thread = None
    if _HAS_ROS:
        rclpy.init(args=args)
        node = HudNode()

        # Spin in a background thread so DDS discovery and callbacks
        # run continuously without depending on QTimer timing
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

    app = QApplication(sys.argv)
    window = HudWindow(ros_node=node)
    window.show()

    exit_code = app.exec_()

    if node:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
