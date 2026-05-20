"""Offscreen multi-process bake renderer for AutoNav HUD.

Renders a 30 fps MP4 that replicates the LIVE SENSOR DATA panel
using matplotlib + OpenCV compositing.  No Qt required — each worker
process renders its chunk of frames independently, then the main
process concatenates the temp files into the final output.

Usage (standalone):
    python bake_offscreen.py <csv_path> [--output <out.mp4>] [--workers N]

The GUI can also launch this as a subprocess via:
    subprocess.Popen([sys.executable, 'bake_offscreen.py', csv_path, ...])
and read stdout lines like "PROGRESS:<frame>/<total>" to track progress.
"""

import argparse
import bisect
import csv
import io
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — no display needed
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Ellipse, Polygon
import matplotlib.ticker as mticker

# ── Constants matching the GUI ───────────────────────────────────────
BAKE_FPS = 30
VIDEO_FPS = 30          # source video FPS (must match recorder)
POWER_WINDOW_S = 3.0
CAPACITY_AH = 20.476
ETA_ALPHA = 0.05

# Layout (pixels) — 1920x1080 output for crisp video
FRAME_W = 1920
FRAME_H = 1080
LEFT_W = 340          # status + power strip
GRID_W = FRAME_W - LEFT_W
GRID_H = FRAME_H - 60  # bottom bar
CELL_W = GRID_W // 2
CELL_H = GRID_H // 2
BOTTOM_H = 60

# Matplotlib render sizes (inches at 150 dpi for sharp plots)
DPI = 150
CELL_FIG_W = CELL_W / DPI
CELL_FIG_H = (CELL_H - 30) / DPI
MINI_FIG_W = (LEFT_W - 24) / DPI
MINI_FIG_H = 100 / DPI

# Colors (BGR for OpenCV)
BG_COLOR = (30, 30, 30)
DARK_BG = (17, 17, 17)
BORDER_COLOR = (68, 68, 68)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT = (136, 136, 136)
GREEN = (0, 255, 0)
CYAN_ACCENT = (255, 170, 0)

# GPS
GPS_VIEW_RADIUS_M = 100 * 0.3048  # 100 ft

# Encoder correction constants
LEFT_ENCODER_SCALE = 1.0 / 1.016335
WHEEL_RADIUS = 0.12946
WHEEL_BASE = 0.6858
TICKS_PER_REV = 81923


# ── Camera title resolution ──────────────────────────────────────────

def _cam_title_for_source(camera_source):
    """Map the meta JSON's `camera_source` to the GUI's panel-title precedence.

    Returns the bake camera panel title text. Legacy CSVs without a
    `<stem>_meta.json` sidecar pass `None` here and get plain "Camera".
    """
    if camera_source == 'overlay':
        return 'Camera CUDA OVERLAY'
    if camera_source == 'mask':
        return 'Camera MASK'
    if camera_source == 'raw':
        return 'Camera RAW'
    return 'Camera'


def _lidar_title_for_source(lidar_source):
    """Map meta JSON's `lidar_source` to the panel-title text.
    Live GUI uses "LIDAR PCA" / "LIDAR Heightband" — keep the same words."""
    if lidar_source == 'pca':
        return 'Lidar PCA'
    if lidar_source == 'heightband':
        return 'Lidar Heightband'
    return 'Lidar'


def _gps_title_for_cov(has_covariance):
    """`GPS Covariance [fix]` when the bag carried valid covariance,
    else plain `GPS`. Same wording as hud_node._live_tick."""
    return 'GPS Covariance [fix]' if has_covariance else 'GPS'


def _odom_title_for_tier(odom_tier):
    """Three-tier title from meta JSON's `odom_tier`.
    Matches hud_node._live_tick lines 7762-7770 exactly."""
    if odom_tier == 'ekf_costmap':
        return 'Encoders (Odom EKF Costmap)'
    if odom_tier == 'ekf':
        return 'Encoders (Odom EKF)'
    return 'Encoders (Odom)'


# ── Data parsing ─────────────────────────────────────────────────────

def parse_csv(path):
    """Parse CSV and return (rows, gps_lats, gps_lons) with absolute timestamps."""
    rows = []
    gps_lats, gps_lons = [], []
    has_encoders = False
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
            if topic == '/encoders':
                has_encoders = True
            if topic == '/gps_fix' and len(values) >= 2:
                try:
                    gps_lats.append(float(values[0]))
                    gps_lons.append(float(values[1]))
                except ValueError:
                    pass
    if has_encoders:
        _recompute_odom_from_encoders(rows)
    return rows, gps_lats, gps_lons


def _recompute_odom_from_encoders(rows):
    """Recompute /odom from raw /encoders with left encoder correction.
    Matches the GUI's _recompute_odom_from_encoders exactly."""
    enc_rows = [(ts, keys, vals) for ts, t, keys, vals in rows if t == '/encoders']
    if len(enc_rows) < 2:
        return
    m_per_tick = (2 * math.pi * WHEEL_RADIUS) / TICKS_PER_REV
    prev_l = float(enc_rows[0][2][0])
    prev_r = float(enc_rows[0][2][1])
    x, y, theta = 0.0, 0.0, 0.0
    odom_insert = []
    for ts, keys, vals in enc_rows[1:]:
        cur_l = float(vals[0])
        cur_r = float(vals[1])
        dl = -(cur_l - prev_l) * m_per_tick * LEFT_ENCODER_SCALE
        dr = (cur_r - prev_r) * m_per_tick
        prev_l, prev_r = cur_l, cur_r
        fwd = (dl + dr) / 2.0
        ang = (dr - dl) / WHEEL_BASE
        x += fwd * math.cos(theta)
        y += fwd * math.sin(theta)
        theta += ang
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi
        qz = math.sin(theta / 2.0)
        odom_insert.append((ts, '/odom', ['pos_x', 'pos_y', 'orient_z'],
                            [str(x), str(y), str(qz)]))
    rows[:] = [(ts, t, k, v) for ts, t, k, v in rows if t != '/odom']
    rows.extend(odom_insert)
    rows.sort(key=lambda r: r[0])


def build_seek_index(rows):
    """Build precomputed index for O(log n) seeking."""
    gps_ts, gps_lat, gps_lon = [], [], []
    gps_cov_ts, gps_cov_ee, gps_cov_en, gps_cov_nn = [], [], [], []
    odom_ts, odom_x, odom_y, odom_theta = [], [], [], []
    pwr_ts, pwr_V, pwr_I, pwr_P = [], [], [], []
    soc_ts, soc_val = [], []
    # /plan — one entry per nav2 plan publication. The xy list grows as
    # the bag's plan messages arrive. seek_data returns the most recent.
    plan_ts, plan_xy = [], []
    last_V, last_I, last_P = 0.0, 0.0, 0.0

    for rel_ns, topic, keys, values in rows:
        if topic == '/gps_fix' and len(values) >= 2:
            try:
                gps_ts.append(rel_ns)
                gps_lat.append(float(values[0]))
                gps_lon.append(float(values[1]))
                # bag_reader writes 6 values when covariance is valid
                # (lat, lon, alt, cov_ee, cov_en, cov_nn).
                if len(values) >= 6:
                    gps_cov_ts.append(rel_ns)
                    gps_cov_ee.append(float(values[3]))
                    gps_cov_en.append(float(values[4]))
                    gps_cov_nn.append(float(values[5]))
            except ValueError:
                pass
        elif topic == '/odom' and len(values) >= 2:
            try:
                x = float(values[0])
                y = float(values[1])
                qz = float(values[2]) if len(values) > 2 else 0.0
                qz_c = max(-1.0, min(1.0, qz))
                yaw = 2.0 * math.asin(qz_c)
                odom_ts.append(rel_ns)
                odom_x.append(x)
                odom_y.append(y)
                odom_theta.append(yaw)
            except (IndexError, ValueError):
                pass
        elif topic == '/plan' and len(values) >= 1:
            try:
                n = int(values[0])
                if n > 0 and len(values) >= 1 + 2 * n:
                    coords = values[1:1 + 2 * n]
                    xs = [float(coords[2 * i]) for i in range(n)]
                    ys = [float(coords[2 * i + 1]) for i in range(n)]
                    plan_ts.append(rel_ns)
                    plan_xy.append((xs, ys))
            except (IndexError, ValueError):
                pass
        elif topic == '/electrical/voltage':
            last_V = float(values[0])
            pwr_ts.append(rel_ns); pwr_V.append(last_V); pwr_I.append(last_I); pwr_P.append(last_P)
        elif topic == '/electrical/current':
            last_I = float(values[0])
            pwr_ts.append(rel_ns); pwr_V.append(last_V); pwr_I.append(last_I); pwr_P.append(last_P)
        elif topic == '/electrical/power':
            last_P = float(values[0])
            pwr_ts.append(rel_ns); pwr_V.append(last_V); pwr_I.append(last_I); pwr_P.append(last_P)
        elif topic == '/electrical/soc':
            soc_ts.append(rel_ns)
            soc_val.append(float(values[0]))

    return {
        'gps_ts': gps_ts, 'gps_lat': gps_lat, 'gps_lon': gps_lon,
        'gps_cov_ts': gps_cov_ts, 'gps_cov_ee': gps_cov_ee,
        'gps_cov_en': gps_cov_en, 'gps_cov_nn': gps_cov_nn,
        'odom_ts': odom_ts, 'odom_x': odom_x, 'odom_y': odom_y, 'odom_theta': odom_theta,
        'plan_ts': plan_ts, 'plan_xy': plan_xy,
        'pwr_ts': pwr_ts, 'pwr_V': pwr_V, 'pwr_I': pwr_I, 'pwr_P': pwr_P,
        'soc_ts': soc_ts, 'soc_val': soc_val,
    }


def seek_data(idx, target_ns):
    """Extract all data buffers at a given timestamp from the preindex."""
    gi = bisect.bisect_right(idx['gps_ts'], target_ns)
    oi = bisect.bisect_right(idx['odom_ts'], target_ns)
    t_s = target_ns / 1e9
    cutoff = t_s - POWER_WINDOW_S
    pi_ = bisect.bisect_right(idx['pwr_ts'], target_ns)
    pwr_start = pi_
    for j in range(pi_ - 1, -1, -1):
        if idx['pwr_ts'][j] / 1e9 < cutoff:
            break
        pwr_start = j
    si = bisect.bisect_right(idx['soc_ts'], target_ns)

    # Latest covariance / plan within the 1s freshness window used by
    # the live GUI for tier graduation. Returning None when stale lets
    # the renderers fall back to the lower-tier display.
    ci = bisect.bisect_right(idx['gps_cov_ts'], target_ns)
    gps_cov = None
    if ci > 0 and (target_ns - idx['gps_cov_ts'][ci - 1]) / 1e9 < 1.0:
        gps_cov = (idx['gps_cov_ee'][ci - 1],
                   idx['gps_cov_en'][ci - 1],
                   idx['gps_cov_nn'][ci - 1])
    plan_i = bisect.bisect_right(idx['plan_ts'], target_ns)
    plan_xy = None
    if plan_i > 0 and (target_ns - idx['plan_ts'][plan_i - 1]) / 1e9 < 2.0:
        plan_xy = idx['plan_xy'][plan_i - 1]

    return {
        'gps_lat': idx['gps_lat'][:gi], 'gps_lon': idx['gps_lon'][:gi],
        'gps_cov': gps_cov,
        'odom_x': idx['odom_x'][:oi], 'odom_y': idx['odom_y'][:oi],
        'odom_theta': idx['odom_theta'][:oi],
        'plan_xy': plan_xy,
        'pwr_t': [ts / 1e9 for ts in idx['pwr_ts'][pwr_start:pi_]],
        'pwr_V': idx['pwr_V'][pwr_start:pi_],
        'pwr_I': idx['pwr_I'][pwr_start:pi_],
        'pwr_P': idx['pwr_P'][pwr_start:pi_],
        'soc_pct': idx['soc_val'][si - 1] if si > 0 else None,
    }


# ── OSM tile fetching ────────────────────────────────────────────────

def _latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _fetch_tile(tx, ty, zoom, tile_cache_dir):
    from PIL import Image as PILImage
    os.makedirs(tile_cache_dir, exist_ok=True)
    cache_path = os.path.join(tile_cache_dir, f'{zoom}_{tx}_{ty}.png')
    if os.path.exists(cache_path):
        return PILImage.open(cache_path).convert('RGB')
    import urllib.request
    url = f'https://mt1.google.com/vt/lyrs=y&x={tx}&y={ty}&z={zoom}'
    req = urllib.request.Request(url, headers={'User-Agent': 'AutoNavGUI/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        with open(cache_path, 'wb') as f:
            f.write(data)
        return PILImage.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None


def fetch_map_for_gps(lats, lons, tile_cache_dir, zoom=19):
    if not lats or not lons:
        return None, None
    lat_min_d, lat_max_d = min(lats), max(lats)
    lon_min_d, lon_max_d = min(lons), max(lons)
    pad_lat = GPS_VIEW_RADIUS_M * 3 / 111320.0
    pad_lon = GPS_VIEW_RADIUS_M * 3 / (111320.0 * math.cos(math.radians((lat_min_d + lat_max_d) / 2)))
    tx_min, ty_min = _latlon_to_tile(lat_max_d + pad_lat, lon_min_d - pad_lon, zoom)
    tx_max, ty_max = _latlon_to_tile(lat_min_d - pad_lat, lon_max_d + pad_lon, zoom)
    if (tx_max - tx_min + 1) * (ty_max - ty_min + 1) > 100:
        return None, None
    tile_rows = []
    for ty in range(ty_min, ty_max + 1):
        row_imgs = []
        for tx in range(tx_min, tx_max + 1):
            tile = _fetch_tile(tx, ty, zoom, tile_cache_dir)
            if tile is None:
                return None, None
            row_imgs.append(np.array(tile))
        tile_rows.append(np.concatenate(row_imgs, axis=1))
    img = np.concatenate(tile_rows, axis=0)
    n = 2 ** zoom
    bnd_lon_min = tx_min / n * 360.0 - 180.0
    bnd_lon_max = (tx_max + 1) / n * 360.0 - 180.0
    bnd_lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty_min / n))))
    bnd_lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty_max + 1) / n))))
    return img, (bnd_lon_min, bnd_lon_max, bnd_lat_min, bnd_lat_max)


# ── Matplotlib figure helpers ────────────────────────────────────────

def fig_to_array(fig):
    """Render a matplotlib figure to a BGR numpy array."""
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _style_ax(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#888', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#444')


class PlotRenderers:
    """Persistent matplotlib figures for one worker — create once, update per frame."""

    def __init__(self, gps_map_img, gps_map_extent, cam_title='Camera',
                 lidar_title='Lidar', gps_title='GPS',
                 odom_title='Encoders (Odom)', costmap=None):
        self._gps_map_img = gps_map_img
        self._gps_map_extent = gps_map_extent
        self.cam_title = cam_title
        self.lidar_title = lidar_title
        self.gps_title = gps_title
        self.odom_title = odom_title
        # costmap is None or dict {'data', 'shape', 'resolution', 'origin'};
        # render_odom crops a ±3 m window dynamically per output frame.
        self._costmap = costmap
        self._odom_scatter = None
        self._odom_tri = None
        # Persistent artists for the tier upgrades. Created on first use
        # and reused across frames to keep matplotlib's set_data path cheap.
        self._gps_cov_ellipse = None
        self._odom_plan_line = None
        self._odom_costmap_im = None

        # -- Camera canvas --
        self._cam_fig = Figure(figsize=(CELL_FIG_W, CELL_FIG_H), dpi=DPI, facecolor='#1e1e1e')
        self._cam_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._cam_ax = self._cam_fig.add_subplot(111)
        self._cam_ax.set_facecolor('#111111')
        self._cam_ax.axis('off')
        self._cam_im = None
        self._last_cam_frame = None  # cache last rendered BGR array

        # -- Lidar canvas --
        self._lidar_fig = Figure(figsize=(CELL_FIG_W, CELL_FIG_H), dpi=DPI, facecolor='#1e1e1e')
        self._lidar_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._lidar_ax = self._lidar_fig.add_subplot(111)
        self._lidar_ax.set_facecolor('#111111')
        self._lidar_ax.axis('off')
        self._lidar_im = None
        self._last_lidar_frame = None

        # -- GPS canvas --
        self._gps_fig = Figure(figsize=(CELL_FIG_W, CELL_FIG_H), dpi=DPI, facecolor='#1e1e1e')
        self._gps_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._gps_ax = self._gps_fig.add_subplot(111)
        self._gps_ax.set_facecolor('#111111')
        self._gps_ax.set_xticklabels([])
        self._gps_ax.set_yticklabels([])
        self._gps_ax.tick_params(axis='both', length=0, pad=0)
        for sp in self._gps_ax.spines.values():
            sp.set_visible(False)
        self._gps_trail, = self._gps_ax.plot([], [], 'c-', linewidth=1, alpha=0.6)
        self._gps_dot, = self._gps_ax.plot([], [], 'ro', markersize=5, zorder=5)
        self._gps_coord_label = self._gps_ax.text(
            0.02, 0.96, '', transform=self._gps_ax.transAxes,
            fontsize=7, color='#0f0', verticalalignment='top', family='monospace',
            bbox=dict(facecolor='#222222', alpha=0.6, edgecolor='none', pad=2),
        )
        self._gps_no_data_txt = self._gps_ax.text(
            0.5, 0.5, 'NO GPS FOR\nTHIS DATA SET',
            transform=self._gps_ax.transAxes, fontsize=8, color='#555',
            ha='center', va='center', family='monospace',
        )
        if gps_map_img is not None:
            self._gps_ax.imshow(
                gps_map_img,
                extent=[gps_map_extent[0], gps_map_extent[1],
                        gps_map_extent[2], gps_map_extent[3]],
                aspect='auto', zorder=0,
            )
            self._gps_no_data_txt.set_visible(False)

        # -- Odom canvas --
        self._odom_fig = Figure(figsize=(CELL_FIG_W, CELL_FIG_H), dpi=DPI, facecolor='#1e1e1e')
        self._odom_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._odom_ax = self._odom_fig.add_subplot(111)
        self._odom_ax.set_facecolor('#111111')
        self._odom_ax.tick_params(axis='both', length=0, pad=-12,
                                   labelsize=6, colors='#666', direction='in')
        for sp in self._odom_ax.spines.values():
            sp.set_visible(False)
        self._odom_ax.set_aspect('equal', adjustable='box')
        self._odom_ax.grid(True, which='both', color='#333', linewidth=0.5)
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

        # -- Power mini canvases --
        self._pwr_figs = {}
        self._pwr_axes = {}
        self._pwr_lines = {}
        self._pwr_default_ylims = {}
        for key, ylim, color in [('V', (20, 30), '#4af'),
                                   ('I', (0, 6), '#f44'),
                                   ('P', (0, 100), '#4f4')]:
            fig = Figure(figsize=(MINI_FIG_W, MINI_FIG_H), dpi=DPI, facecolor='#1e1e1e')
            fig.subplots_adjust(left=0.22, right=0.94, top=0.78, bottom=0.28)
            ax = fig.add_subplot(111)
            _style_ax(ax)
            ax.set_ylim(*ylim)
            line, = ax.plot([], [], color=color, linewidth=1)
            self._pwr_figs[key] = fig
            self._pwr_axes[key] = ax
            self._pwr_lines[key] = line
            self._pwr_default_ylims[key] = ylim

    def render_camera(self, cam_cap, elapsed_s, duration_s):
        if cam_cap is None:
            return self._placeholder(self._cam_fig, self._cam_ax, 'VIDEO NOT AVAILABLE')
        total = int(cam_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 1:1 mapping — camera video duration matches CSV data span
        frame_idx = min(int(elapsed_s * VIDEO_FPS), total - 1)
        if frame_idx < 0:
            return self._placeholder(self._cam_fig, self._cam_ax, 'VIDEO NOT AVAILABLE')
        cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cam_cap.read()
        if not ret:
            # Hold last frame
            if self._last_cam_frame is not None:
                return self._last_cam_frame
            return self._placeholder(self._cam_fig, self._cam_ax, 'VIDEO NOT AVAILABLE')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._cam_im is None:
            self._cam_im = self._cam_ax.imshow(rgb, aspect='equal')
        else:
            self._cam_im.set_data(rgb)
        result = fig_to_array(self._cam_fig)
        self._last_cam_frame = result
        return result

    def render_lidar(self, lidar_cap, elapsed_s, cam_duration_s):
        if lidar_cap is None:
            return self._placeholder(self._lidar_fig, self._lidar_ax, 'VIDEO NOT AVAILABLE')
        total = int(lidar_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Scale lidar to camera timebase — both cover the same real-world span
        # but lidar may have fewer frames (dropped during recording)
        if cam_duration_s > 0:
            frame_idx = min(int((elapsed_s / cam_duration_s) * total), total - 1)
        else:
            frame_idx = min(int(elapsed_s * VIDEO_FPS), total - 1)
        if frame_idx < 0:
            return self._placeholder(self._lidar_fig, self._lidar_ax, 'VIDEO NOT AVAILABLE')
        lidar_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = lidar_cap.read()
        if not ret:
            if self._last_lidar_frame is not None:
                return self._last_lidar_frame
            return self._placeholder(self._lidar_fig, self._lidar_ax, 'VIDEO NOT AVAILABLE')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.rot90(rgb, 2)
        rgb = np.fliplr(rgb)
        h = rgb.shape[0]
        rgb = rgb[:h * 2 // 3, :]
        half_m = 10.0
        y_bottom = -half_m + (2.0 * half_m) / 3.0
        extent = [-half_m, half_m, y_bottom, half_m]
        if self._lidar_im is None:
            self._lidar_ax.axis('on')
            self._lidar_im = self._lidar_ax.imshow(rgb, aspect='equal', extent=extent)
            self._lidar_ax.set_xlim(-half_m, half_m)
            self._lidar_ax.set_ylim(y_bottom, half_m)
            self._lidar_ax.tick_params(axis='both', length=2, pad=2,
                                        labelsize=6, colors='#888', direction='in')
            self._lidar_ax.set_xlabel('m', fontsize=6, color='#888', labelpad=1)
            for sp in self._lidar_ax.spines.values():
                sp.set_color('#444')
        else:
            self._lidar_im.set_data(rgb)
            self._lidar_im.set_extent(extent)
        result = fig_to_array(self._lidar_fig)
        self._last_lidar_frame = result
        return result

    def render_gps(self, data):
        lats, lons = data['gps_lat'], data['gps_lon']
        if lons:
            self._gps_no_data_txt.set_visible(False)
            self._gps_trail.set_data(lons, lats)
            self._gps_dot.set_data([lons[-1]], [lats[-1]])
            cur_lat, cur_lon = lats[-1], lons[-1]
            dlat = GPS_VIEW_RADIUS_M / 111320.0
            dlon = GPS_VIEW_RADIUS_M / (111320.0 * math.cos(math.radians(cur_lat)))
            self._gps_ax.set_xlim(cur_lon - dlon, cur_lon + dlon)
            self._gps_ax.set_ylim(cur_lat - dlat, cur_lat + dlat)
            self._gps_coord_label.set_text(f"Lat: {cur_lat:.8f}\nLon: {cur_lon:.8f}")

            # Covariance tier — 2σ ellipse from the East/North 2x2 block.
            # Axis lengths from eigendecomposition; orientation in degrees
            # measured from the lon axis. Matches the live GUI's logic
            # (hud_node._live_tick around line 7712).
            cov = data.get('gps_cov')
            if cov is not None:
                cov_ee, cov_en, cov_nn = cov
                # 2x2 eigendecomposition (closed form for symmetric matrix)
                trace = cov_ee + cov_nn
                diff = cov_ee - cov_nn
                disc = math.sqrt(max(0.0, diff * diff + 4 * cov_en * cov_en))
                lam1 = 0.5 * (trace + disc)  # major
                lam2 = 0.5 * (trace - disc)  # minor
                # Eigenvector angle (radians) of major axis
                angle_rad = 0.5 * math.atan2(2 * cov_en, diff) if diff != 0 or cov_en != 0 else 0.0
                # 2σ axes in metres → convert to degrees lat/lon
                axis_e_m = 2.0 * math.sqrt(max(0.0, lam1))
                axis_n_m = 2.0 * math.sqrt(max(0.0, lam2))
                axis_e_deg = axis_e_m / (111320.0 * math.cos(math.radians(cur_lat)))
                axis_n_deg = axis_n_m / 111320.0
                if self._gps_cov_ellipse is None:
                    self._gps_cov_ellipse = Ellipse(
                        (cur_lon, cur_lat),
                        width=2 * axis_e_deg, height=2 * axis_n_deg,
                        angle=math.degrees(angle_rad),
                        fill=False, edgecolor='#ff5555',
                        linewidth=1.2, zorder=4,
                    )
                    self._gps_ax.add_patch(self._gps_cov_ellipse)
                else:
                    self._gps_cov_ellipse.center = (cur_lon, cur_lat)
                    self._gps_cov_ellipse.width = 2 * axis_e_deg
                    self._gps_cov_ellipse.height = 2 * axis_n_deg
                    self._gps_cov_ellipse.angle = math.degrees(angle_rad)
                    self._gps_cov_ellipse.set_visible(True)
            elif self._gps_cov_ellipse is not None:
                self._gps_cov_ellipse.set_visible(False)
        else:
            self._gps_no_data_txt.set_visible(True)
            self._gps_trail.set_data([], [])
            self._gps_dot.set_data([], [])
            if self._gps_cov_ellipse is not None:
                self._gps_cov_ellipse.set_visible(False)
        return fig_to_array(self._gps_fig)

    def render_odom(self, data):
        xs, ys, thetas = data['odom_x'], data['odom_y'], data['odom_theta']
        if self._odom_scatter is not None:
            self._odom_scatter.remove()
            self._odom_scatter = None
        if self._odom_tri is not None:
            self._odom_tri.remove()
            self._odom_tri = None
        if not xs:
            # Hide tier upgrades when no trail exists
            if self._odom_plan_line is not None:
                self._odom_plan_line.set_visible(False)
            if self._odom_costmap_im is not None:
                self._odom_costmap_im.set_visible(False)
            return fig_to_array(self._odom_fig)

        n_pts = len(xs)
        if n_pts > 2000:
            step = n_pts // 2000
            xs_d = xs[::step] + [xs[-1]]
            ys_d = ys[::step] + [ys[-1]]
        else:
            xs_d, ys_d = xs, ys

        colors = np.linspace(0, 1, len(xs_d))
        self._odom_scatter = self._odom_ax.scatter(
            xs_d, ys_d, c=colors, cmap='coolwarm', s=4, zorder=3)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span = max(max_x - min_x, max_y - min_y) * 1.3
        span = max(span, 1.0)
        half = span / 2
        cx_v = (min_x + max_x) / 2
        cy_v = (min_y + max_y) / 2
        self._odom_ax.set_xlim(cx_v - half, cx_v + half)
        self._odom_ax.set_ylim(cy_v - half, cy_v + half)

        nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        target = span / 5
        grid_sp = nice[0]
        for n in nice:
            grid_sp = n
            if n >= target:
                break
        self._odom_ax.xaxis.set_major_locator(mticker.MultipleLocator(grid_sp))
        self._odom_ax.yaxis.set_major_locator(mticker.MultipleLocator(grid_sp))

        if n_pts > 1:
            dist = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        else:
            dist = 0.0
        self._odom_dist_label.set_text(f"Dist: {dist:.2f} m")

        cx, cy = xs[-1], ys[-1]
        theta = thetas[-1] if thetas else 0.0

        # ── Tier 3 visual: ±3 m costmap crop centred on current pose ──
        # The live GUI does the exact same crop in _cb_global_costmap so the
        # global costmap renders local-sized under the red triangle.
        if self._costmap is not None:
            self._render_costmap_crop(cx, cy)
        elif self._odom_costmap_im is not None:
            self._odom_costmap_im.set_visible(False)

        # ── Tier visual (independent): green /plan line overlay ──
        plan_xy = data.get('plan_xy')
        if plan_xy is not None:
            pxs, pys = plan_xy
            if self._odom_plan_line is None:
                self._odom_plan_line, = self._odom_ax.plot(
                    pxs, pys, color='#33ff66', linewidth=1.5,
                    zorder=4, alpha=0.9,
                )
            else:
                self._odom_plan_line.set_data(pxs, pys)
                self._odom_plan_line.set_visible(True)
        elif self._odom_plan_line is not None:
            self._odom_plan_line.set_visible(False)

        s = max(span * 0.05, 0.1)
        nose = (cx + s * math.cos(theta), cy + s * math.sin(theta))
        bl = (cx + s * 0.5 * math.cos(theta + 2.5), cy + s * 0.5 * math.sin(theta + 2.5))
        br = (cx + s * 0.5 * math.cos(theta - 2.5), cy + s * 0.5 * math.sin(theta - 2.5))
        self._odom_tri = Polygon([nose, bl, br], closed=True,
                                  facecolor='red', edgecolor='white',
                                  linewidth=0.8, zorder=5)
        self._odom_ax.add_patch(self._odom_tri)
        return fig_to_array(self._odom_fig)

    def _render_costmap_crop(self, rx, ry, crop_half=3.0):
        """Crop a ±crop_half-metre window around (rx, ry) from the persistent
        global costmap snapshot and draw it under the odom trail. Mirrors
        the cropping/colour-mapping logic in hud_node._cb_global_costmap.
        """
        cm = self._costmap
        h = int(cm['shape'][0])
        w = int(cm['shape'][1])
        res = float(cm['resolution'])
        ox = float(cm['origin'][0])
        oy = float(cm['origin'][1])
        if h <= 0 or w <= 0 or res <= 0.0:
            return
        col_min = max(0, int((rx - crop_half - ox) / res))
        col_max = min(w, int(math.ceil((rx + crop_half - ox) / res)))
        row_min = max(0, int((ry - crop_half - oy) / res))
        row_max = min(h, int(math.ceil((ry + crop_half - oy) / res)))
        if col_max <= col_min or row_max <= row_min:
            if self._odom_costmap_im is not None:
                self._odom_costmap_im.set_visible(False)
            return

        data = cm['data'].reshape((h, w))
        cropped = data[row_min:row_max, col_min:col_max]
        ch, cw = cropped.shape
        clipped = np.clip(cropped, 0, 100).astype(np.uint16)
        gray = (40 + clipped * 200 // 100).astype(np.uint8)
        alpha = (60 + clipped * 160 // 100).astype(np.uint8)
        unknown = cropped < 0
        gray[unknown] = 0
        alpha[unknown] = 0
        rgba = np.empty((ch, cw, 4), dtype=np.uint8)
        rgba[..., :3] = gray[..., None]
        rgba[..., 3] = alpha
        extent = (
            ox + col_min * res,
            ox + col_max * res,
            oy + row_min * res,
            oy + row_max * res,
        )
        if self._odom_costmap_im is None:
            self._odom_costmap_im = self._odom_ax.imshow(
                rgba, extent=extent, origin='lower',
                interpolation='nearest', zorder=1,
            )
        else:
            self._odom_costmap_im.set_data(rgba)
            self._odom_costmap_im.set_extent(extent)
            self._odom_costmap_im.set_visible(True)

    def render_power(self, key, data):
        fig = self._pwr_figs[key]
        ax = self._pwr_axes[key]
        line = self._pwr_lines[key]
        t = data['pwr_t']
        vals = data[f'pwr_{key}']
        if t:
            line.set_data(t, vals)
            ax.set_xlim(t[-1] - POWER_WINDOW_S, t[-1])
            ax.set_xticklabels([])
            # Auto-scale Y if data exceeds default limits
            default_lo, default_hi = self._pwr_default_ylims[key]
            d_min, d_max = min(vals), max(vals)
            y_lo = min(default_lo, d_min * 0.95) if d_min < default_lo else default_lo
            y_hi = max(default_hi, d_max * 1.05) if d_max > default_hi else default_hi
            ax.set_ylim(y_lo, y_hi)
        else:
            line.set_data([], [])
        return fig_to_array(fig)

    def _placeholder(self, fig, ax, text):
        ax.clear()
        ax.set_facecolor('#111111')
        ax.axis('off')
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=8, color='#555',
                ha='center', va='center', family='monospace')
        return fig_to_array(fig)


# ── Frame compositing ────────────────────────────────────────────────

def _text(canvas, text, pos, color=TEXT_COLOR, scale=0.4, thickness=1):
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _status_dot(canvas, x, y, active, label):
    color = (0, 255, 0) if active else (85, 85, 85)
    cv2.circle(canvas, (x + 10, y + 10), 9, color, -1, cv2.LINE_AA)
    _text(canvas, label, (x + 26, y + 16), DIM_TEXT, 0.55)


def _soc_color(pct):
    if pct > 50:
        return (68, 255, 68)
    elif pct > 20:
        return (68, 255, 255)
    else:
        return (68, 68, 255)


def compose_frame(cam_img, lidar_img, gps_img, odom_img, pwr_imgs, data,
                   elapsed_s, duration_s, frame_i, total_frames, active_topics,
                   cam_title='Camera', lidar_title='Lidar', gps_title='GPS',
                   odom_title='Encoders (Odom)'):
    """Composite all elements into FRAME_W x FRAME_H BGR image."""
    canvas = np.full((FRAME_H, FRAME_W, 3), BG_COLOR, dtype=np.uint8)

    # ── Left strip: status dots ──
    y = 12
    _text(canvas, 'LIVE SENSOR DATA', (LEFT_W // 2 - 100, y + 18), TEXT_COLOR, 0.75, 2)
    y += 38
    _text(canvas, 'Physical Devices', (14, y + 16), DIM_TEXT, 0.5)
    y += 26
    for name in ['Camera', 'Lidar', 'GPS', 'Encoders', 'Power PCB', 'Arduino', 'Motor Ctrl']:
        _status_dot(canvas, 14, y, name in active_topics, name)
        y += 28
    y += 6
    _text(canvas, 'Virtual Devices', (14, y + 16), DIM_TEXT, 0.5)
    y += 26
    for name in ['SLAM', 'CONTROL', 'NAV2', 'LINE DETECT']:
        _status_dot(canvas, 14, y, False, name)
        y += 28
    y += 6
    _text(canvas, 'Test System', (14, y + 16), DIM_TEXT, 0.5)
    y += 26
    _status_dot(canvas, 14, y, False, 'TEST')
    y += 36

    # ── Left strip: power ──
    _text(canvas, 'Power PCB', (LEFT_W // 2 - 50, y + 16), TEXT_COLOR, 0.6, 1)
    y += 26
    for key, label, vc in [('V', 'Voltage (V)', (255, 170, 68)),
                            ('I', 'Current (A)', (68, 68, 255)),
                            ('P', 'Power (W)', (68, 255, 68))]:
        vals = data[f'pwr_{key}']
        vs = f"{key}: {vals[-1]:.2f}" if vals else f"{key}: --"
        _text(canvas, label, (14, y + 16), TEXT_COLOR, 0.45)
        _text(canvas, vs, (LEFT_W - 100, y + 16), vc, 0.45)
        y += 22
        pimg = pwr_imgs.get(key)
        if pimg is not None:
            ph, pw = pimg.shape[:2]
            tw = LEFT_W - 28
            th = int(ph * tw / pw)
            resized = cv2.resize(pimg, (tw, th))
            if y + th < FRAME_H - BOTTOM_H:
                canvas[y:y + th, 14:14 + tw] = resized
            y += th + 6

    # SOC
    soc = data['soc_pct']
    if soc is not None:
        _text(canvas, 'SOC', (14, y + 16), TEXT_COLOR, 0.5)
        _text(canvas, f"{int(round(soc))}%", (65, y + 16), DIM_TEXT, 0.5)
        y += 26
        bx, bw, bh = 14, LEFT_W - 40, 22
        cv2.rectangle(canvas, (bx, y), (bx + bw, y + bh), (85, 85, 85), 1)
        fw = int(bw * max(0, min(100, soc)) / 100)
        if fw > 0:
            cv2.rectangle(canvas, (bx + 1, y + 1), (bx + 1 + fw, y + bh - 1), _soc_color(soc), -1)
        y += bh + 6
        eta_hours = data.get('eta_hours')
        if eta_hours is not None and eta_hours > 0:
            h = int(eta_hours)
            m = int((eta_hours - h) * 60)
            _text(canvas, f"ETA: {h}h {m:02d}m", (14, y + 16), DIM_TEXT, 0.5)
        else:
            _text(canvas, "ETA: --:--", (14, y + 16), DIM_TEXT, 0.5)

    # Separator
    cv2.line(canvas, (LEFT_W - 1, 0), (LEFT_W - 1, FRAME_H - BOTTOM_H), BORDER_COLOR, 1)

    # ── 2x2 sensor grid ──
    def _place(img, row, col, title):
        x0 = LEFT_W + col * CELL_W
        y0 = row * CELL_H
        title_h = 30
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W - 1, y0 + title_h), BG_COLOR, -1)
        tw_ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        _text(canvas, title, (x0 + (CELL_W - tw_) // 2, y0 + 22), TEXT_COLOR, 0.6, 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W - 1, y0 + CELL_H - 1), BORDER_COLOR, 1)
        if img is not None:
            tw2, th2 = CELL_W - 10, CELL_H - title_h - 6
            canvas[y0 + title_h + 2:y0 + title_h + 2 + th2, x0 + 5:x0 + 5 + tw2] = cv2.resize(img, (tw2, th2))

    _place(cam_img, 0, 0, cam_title)
    _place(lidar_img, 0, 1, lidar_title)
    _place(gps_img, 1, 0, gps_title)
    _place(odom_img, 1, 1, odom_title)

    # ── Bottom bar ──
    by = FRAME_H - BOTTOM_H
    cv2.line(canvas, (0, by), (FRAME_W, by), BORDER_COLOR, 1)
    _text(canvas, f"{elapsed_s:.1f}s / {duration_s:.1f}s", (14, by + 26), DIM_TEXT, 0.6)
    _text(canvas, f"F: {frame_i} / {total_frames}", (14, by + 48), (85, 85, 85), 0.5)
    bx_s, bx_e = 340, FRAME_W - 80
    bmy = by + 30
    cv2.line(canvas, (bx_s, bmy), (bx_e, bmy), (51, 51, 51), 8)
    fx = bx_s + int((bx_e - bx_s) * frame_i / max(total_frames, 1))
    cv2.line(canvas, (bx_s, bmy), (fx, bmy), CYAN_ACCENT, 8)
    _text(canvas, "1x", (FRAME_W - 60, by + 36), DIM_TEXT, 0.6)

    return canvas


# ── Worker ───────────────────────────────────────────────────────────

def worker_render_chunk(args):
    """Render a chunk of frames to a temp mp4. Returns (temp_path, n_frames)."""
    (chunk_id, frame_start, frame_end, total_frames,
     idx, duration_ns, cam_path, lidar_path,
     gps_map_img, gps_map_extent, temp_dir,
     cam_title, lidar_title, gps_title, odom_title, costmap) = args

    duration_s = duration_ns / 1e9
    cam_cap = cv2.VideoCapture(cam_path) if cam_path and os.path.isfile(cam_path) else None
    lidar_cap = cv2.VideoCapture(lidar_path) if lidar_path and os.path.isfile(lidar_path) else None
    renderers = PlotRenderers(
        gps_map_img, gps_map_extent,
        cam_title=cam_title, lidar_title=lidar_title,
        gps_title=gps_title, odom_title=odom_title,
        costmap=costmap,
    )

    # Camera duration is the reference timebase for syncing lidar
    cam_duration_s = (int(cam_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / VIDEO_FPS) if cam_cap and cam_cap.isOpened() else duration_s

    temp_path = os.path.join(temp_dir, f'chunk_{chunk_id:04d}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, BAKE_FPS, (FRAME_W, FRAME_H))

    # Skip-redundant-redraws tracking
    last_gps_len = -1
    last_odom_len = -1
    last_pwr_len = -1
    cached_gps = None
    cached_odom = None
    cached_pwr = {}
    ema_eta_hours = None  # EMA-smoothed ETA estimate

    for frame_i in range(frame_start, frame_end):
        target_ns = int((frame_i / BAKE_FPS) * 1e9)
        elapsed_s = target_ns / 1e9
        data = seek_data(idx, target_ns)

        # Compute EMA-smoothed ETA (monotonically decreasing)
        soc = data['soc_pct']
        avg_i = sum(abs(v) for v in data['pwr_I']) / len(data['pwr_I']) if data['pwr_I'] else 0
        if avg_i > 0.01 and soc is not None:
            raw_hours = (soc / 100.0) * CAPACITY_AH / avg_i
            if ema_eta_hours is None:
                ema_eta_hours = raw_hours
            else:
                smoothed = ETA_ALPHA * raw_hours + (1 - ETA_ALPHA) * ema_eta_hours
                ema_eta_hours = min(ema_eta_hours, smoothed)
        data['eta_hours'] = ema_eta_hours

        active = set()
        if data['gps_lat']:
            active.add('GPS')
        if data['odom_x']:
            active.add('Encoders')
        if data['pwr_t']:
            active.add('Power PCB')

        # Camera/lidar always re-render (video frames)
        cam_img = renderers.render_camera(cam_cap, elapsed_s, duration_s)
        lidar_img = renderers.render_lidar(lidar_cap, elapsed_s, cam_duration_s)
        active.update(['Camera', 'Lidar'])

        # GPS: re-render if trail length changed OR covariance is fresh
        # (the ellipse can update independently of the trail).
        gl = len(data['gps_lat'])
        gps_has_cov = data.get('gps_cov') is not None
        if gl != last_gps_len or gps_has_cov:
            cached_gps = renderers.render_gps(data)
            last_gps_len = gl

        # Odom: re-render when trail grew OR a plan/costmap is fresh
        # (plan + costmap can change between odom samples).
        ol = len(data['odom_x'])
        plan_fresh = data.get('plan_xy') is not None
        if ol != last_odom_len or plan_fresh or costmap is not None:
            cached_odom = renderers.render_odom(data)
            last_odom_len = ol

        # Power: always re-render — rolling window content shifts every frame
        for k in ('V', 'I', 'P'):
            cached_pwr[k] = renderers.render_power(k, data)

        frame = compose_frame(cam_img, lidar_img, cached_gps, cached_odom,
                               cached_pwr, data, elapsed_s, duration_s,
                               frame_i, total_frames, active,
                               cam_title=cam_title, lidar_title=lidar_title,
                               gps_title=gps_title, odom_title=odom_title)
        writer.write(frame)

        if frame_i % 30 == 0:
            print(f"PROGRESS:{frame_i}/{total_frames}", flush=True)

    writer.release()
    if cam_cap:
        cam_cap.release()
    if lidar_cap:
        lidar_cap.release()
    return temp_path, frame_end - frame_start


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Offscreen multi-process bake renderer')
    parser.add_argument('source', nargs='?', default=None,
                        help='CSV path, or a rosbag2 directory (e.g. <session>/bag '
                             'or a session dir containing bag/)')
    parser.add_argument('--bag', default=None,
                        help='Explicit rosbag2 directory to bake. Equivalent to '
                             'passing a bag dir as the positional source.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output MP4 path (default: Baked_Playbacks/<stem>_baked.mp4)')
    parser.add_argument('--workers', '-w', type=int, default=0,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--force-extract', action='store_true',
                        help='Re-extract CSV+sidecar mp4s even if they already exist (bag inputs only)')
    args = parser.parse_args()

    src = args.bag or args.source
    if not src:
        parser.error('Provide a CSV path or a bag directory (--bag <dir>).')

    src_abs = os.path.abspath(src)
    # If the source is a rosbag2 directory (or a session dir containing one),
    # extract it first to produce the CSV + sidecar mp4s that the rest of
    # the pipeline already knows how to consume.
    if os.path.isdir(src_abs):
        from bag_reader import find_bag_dir, extract_bag
        bag_dir = find_bag_dir(src_abs)
        if bag_dir is None:
            print(f"ERROR: not a rosbag2 directory: {src_abs}", flush=True)
            sys.exit(1)
        print(f"Extracting bag -> CSV+mp4 sidecars: {bag_dir}", flush=True)
        csv_path = extract_bag(bag_dir, force=args.force_extract, verbose=True)
    else:
        csv_path = src_abs

    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", flush=True)
        sys.exit(1)

    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        bake_dir = os.path.normpath(os.path.join(script_dir, '..', 'data', 'Baked_Playbacks'))
        os.makedirs(bake_dir, exist_ok=True)
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(bake_dir, f'{csv_stem}_baked.mp4')

    cam_stem = os.path.splitext(csv_path)[0]
    cam_path = cam_stem + '_camera.mp4'
    lidar_path = cam_stem + '_lidar_bev.mp4'
    meta_path = cam_stem + '_meta.json'
    costmap_path = cam_stem + '_costmap.npz'
    cam_path = cam_path if os.path.isfile(cam_path) else None
    lidar_path = lidar_path if os.path.isfile(lidar_path) else None

    # Resolve panel titles + optional overlay state from the meta sidecar.
    # Legacy CSVs without _meta.json get the default ("Camera"/"Lidar"/
    # "GPS"/"Encoders (Odom)").
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f) or {}
        except (OSError, ValueError) as e:
            print(f"WARN: could not read {meta_path}: {e}", flush=True)
    cam_title = _cam_title_for_source(meta.get('camera_source'))
    lidar_title = _lidar_title_for_source(meta.get('lidar_source'))
    gps_title = _gps_title_for_cov(bool(meta.get('gps_has_covariance')))
    odom_title = _odom_title_for_tier(meta.get('odom_tier'))
    print(f"Panel titles: cam={cam_title!r} lidar={lidar_title!r} "
          f"gps={gps_title!r} odom={odom_title!r}", flush=True)

    # Load the costmap snapshot once (it's tiny — a few KB). Workers
    # receive it via the args tuple; matplotlib creates the imshow once
    # per worker and the crop is updated per output frame.
    costmap = None
    if meta.get('has_costmap') and os.path.isfile(costmap_path):
        try:
            cm = np.load(costmap_path)
            costmap = {
                'data': cm['data'],
                'shape': cm['shape'],
                'resolution': float(cm['resolution']),
                'origin': cm['origin'],
            }
            print(f"Loaded costmap snapshot from {costmap_path}", flush=True)
        except (OSError, ValueError, KeyError) as e:
            print(f"WARN: could not load costmap {costmap_path}: {e}",
                  flush=True)

    tile_cache_dir = os.path.join(script_dir, 'tile_cache')

    print("BAKE_START", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Workers: {n_workers}", flush=True)
    t0 = time.time()

    print("Parsing CSV...", flush=True)
    raw_rows, gps_lats, gps_lons = parse_csv(csv_path)
    if not raw_rows:
        print("ERROR: No data in CSV", flush=True)
        sys.exit(1)
    raw_rows.sort(key=lambda r: r[0])
    t0_ns = raw_rows[0][0]
    rows = [(ts - t0_ns, topic, keys, vals) for ts, topic, keys, vals in raw_rows]
    duration_ns = rows[-1][0]
    total_frames = int((duration_ns / 1e9) * BAKE_FPS)
    print(f"Duration: {duration_ns / 1e9:.1f}s, Frames: {total_frames}", flush=True)
    print(f"TOTAL_FRAMES:{total_frames}", flush=True)

    print("Building seek index...", flush=True)
    idx = build_seek_index(rows)

    print("Fetching GPS map tiles...", flush=True)
    gps_map_img, gps_map_extent = fetch_map_for_gps(gps_lats, gps_lons, tile_cache_dir)

    temp_dir = tempfile.mkdtemp(prefix='autonav_bake_')
    chunk_size = max(1, total_frames // n_workers)
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, total_frames)
        if i == n_workers - 1:
            end = total_frames
        if start >= total_frames:
            break
        chunks.append((i, start, end, total_frames, idx, duration_ns,
                        cam_path, lidar_path, gps_map_img, gps_map_extent,
                        temp_dir, cam_title, lidar_title, gps_title,
                        odom_title, costmap))

    print(f"Launching {len(chunks)} workers...", flush=True)
    with multiprocessing.Pool(len(chunks)) as pool:
        results = pool.map(worker_render_chunk, chunks)

    print("Concatenating chunks...", flush=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, BAKE_FPS, (FRAME_W, FRAME_H))
    for temp_path, n in results:
        cap = cv2.VideoCapture(temp_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        os.remove(temp_path)
    writer.release()
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    elapsed = time.time() - t0
    print(f"BAKE_DONE:{out_path}", flush=True)
    print(f"Completed in {elapsed:.1f}s ({total_frames / elapsed:.1f} fps)", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
