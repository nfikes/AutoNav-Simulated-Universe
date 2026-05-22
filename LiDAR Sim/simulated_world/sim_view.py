#!/usr/bin/env python3
"""Shared figure-construction and 3D-panel update helpers.

Both ``lidar_sim_gui.py`` (interactive single-agent GUI) and
``scripts/test_ramp_crossing.py`` (15-agent ramp benchmark) call into
this module so the two front ends look and feel like one system.

The single source of truth for the three-panel layout, the BEV panel
title/legend, and the 3D point-cloud styling lives here. If you want to
change how the sim *looks*, change it here.

The algorithm constants (Z_GROUND_BAND, TRAVERSABLE_MAX_DEG, etc.) are
still owned by ``lidar_sim_gui.py`` — this module imports them via
``import lidar_sim_gui as sim`` to avoid duplicating any numbers.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Late-bound to avoid circular import at module-load time inside
# lidar_sim_gui.py; we import the sim module on first use instead.
_sim = None


def _get_sim():
    global _sim
    if _sim is None:
        import lidar_sim_gui as sim
        _sim = sim
    return _sim


# ── PyQtGraph 3D view ─────────────────────────────────────────────
#
# The single-agent GUI swaps mplot3d for a PyQtGraph GLViewWidget so
# the 3D point cloud can be updated in-place (single VBO upload) rather
# than rebuilt as a fresh Path3DCollection every scan tick. The heavy
# "scan-tick" frame used to spend ~70 ms in the mplot3d scatter rebuild
# alone; the GL path drops that to single-digit ms.
#
# The benchmark (scripts/test_ramp_crossing.py) still calls
# ``build_three_panel_figure`` and gets a matplotlib ax3d back, so it
# keeps working unchanged in headless Agg batch runs.

# Imported lazily so this module still loads on machines without
# PyQtGraph / OpenGL installed (e.g. the headless harness when the
# 3D-view dependencies are missing — we fall back to a mock view).
_gl = None
_qt = None


def _get_pyqtgraph():
    """Lazy import of pyqtgraph.opengl + Qt. Returns (gl_module, QtWidgets, QtCore)
    or raises ImportError if PyQtGraph / PyOpenGL are unavailable."""
    global _gl, _qt
    if _gl is None:
        import pyqtgraph.opengl as gl  # noqa: WPS433 — lazy import is intentional
        from PyQt5 import QtCore, QtWidgets  # noqa: WPS433
        _gl = gl
        _qt = (QtWidgets, QtCore)
    return _gl, _qt[0], _qt[1]


class ThreeDView:
    """PyQtGraph GLViewWidget wrapper holding the persistent scatter +
    quiver items so each scan tick is a single ``setData`` upload.

    Attributes:
        widget      — the ``GLViewWidget`` (a ``QWidget`` subclass)
        cloud       — ``GLScatterPlotItem`` for the LiDAR point cloud
        lidar       — ``GLScatterPlotItem`` for the LiDAR origin marker
        normal      — ``GLLinePlotItem`` for the surface-normal quiver
        grid        — ``GLGridItem`` at z=0 (matches the explicit-grid
                      request for the 3D panel)
    """

    __slots__ = ("widget", "cloud", "lidar", "normal", "grid", "ghost")

    def __init__(self):
        gl, QtWidgets, QtCore = _get_pyqtgraph()
        # Black/dark background matches the matplotlib 3D panel.
        self.widget = gl.GLViewWidget()
        self.widget.setBackgroundColor((0x11, 0x11, 0x11))
        # Camera mirrors matplotlib's view_init(elev=30, azim=-60) and
        # the ~4 m view radius (distance ≈ 8 m at the GL widget's default
        # field of view).
        self.widget.opts["elevation"] = 30.0
        self.widget.opts["azimuth"] = -60.0
        self.widget.opts["distance"] = 8.0

        # Faint grid at z=0 — explicit grid-on request for the 3D panel.
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=8.0, y=8.0)
        self.grid.setSpacing(x=1.0, y=1.0)
        self.grid.setColor((0x55, 0x55, 0x55, 0x80))
        self.widget.addItem(self.grid)

        # Persistent scatter / quiver items — updated in place each tick.
        self.cloud = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.zeros((1, 4), dtype=np.float32),
            size=2.0,
        )
        # pxMode=True means ``size`` is in screen pixels (matches the
        # matplotlib s=0.5 sense of "small dot"), and the scatter draws
        # cheaply via point sprites. Depth test is OFF so the cloud
        # always renders on top of the translucent ghost mesh, and the
        # depth value is bumped high so the cloud sorts after the ghost
        # in PyQtGraph's per-item draw order.
        try:
            from OpenGL import GL as _gl
            self.cloud.setGLOptions({
                _gl.GL_DEPTH_TEST: False,
                _gl.GL_BLEND: True,
                _gl.GL_ALPHA_TEST: False,
                _gl.GL_CULL_FACE: False,
                "glBlendFunc": (_gl.GL_SRC_ALPHA, _gl.GL_ONE_MINUS_SRC_ALPHA),
            })
        except Exception:
            self.cloud.setGLOptions("opaque")  # fallback
        self.cloud.setDepthValue(10)
        self.widget.addItem(self.cloud)

        self.lidar = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[0.85, 0.0, 0.0, 1.0]], dtype=np.float32),
            size=10.0,
        )
        self.widget.addItem(self.lidar)

        self.normal = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 1.0, 0.0, 1.0),
            width=2.5,
            antialias=True,
        )
        self.widget.addItem(self.normal)

        # Ghost STL — populated via ``set_ghost_mesh`` after construction
        # so the heavy mesh upload (300k faces) only happens when the
        # GUI has a mesh handy. Stays None on the benchmark side that
        # doesn't pass a mesh through.
        self.ghost = None

    def set_ghost_mesh(self, verts, faces,
                       color=(0.62, 0.66, 0.72, 0.25)):
        """Add a translucent STL ghost to the 3D view. Call once after
        the mesh is loaded. ``verts`` is (N, 3) float32, ``faces`` is
        (M, 3) int32. Alpha defaults to 0.25 (the user's spec).
        """
        gl, _, _ = _get_pyqtgraph()
        if self.ghost is not None:
            self.widget.removeItem(self.ghost)
            self.ghost = None
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        self.ghost = gl.GLMeshItem(
            meshdata=meshdata,
            color=color,
            glOptions="translucent",
            smooth=False,
            shader="shaded",      # cheap lambertian — gives shape readability
            drawEdges=False,
        )
        # Render under the point cloud so the cloud sits on top.
        self.ghost.setDepthValue(-1)
        self.widget.addItem(self.ghost)


class MockThreeDView:
    """No-op stand-in for ``ThreeDView`` used by headless harnesses
    that can't construct a real OpenGL context (Agg-only runs).

    Exposes the same attribute surface — ``cloud.setData(...)`` etc are
    silently swallowed so the rest of the GUI / animate loop runs
    unchanged."""

    class _Item:
        def setData(self, *args, **kwargs):  # noqa: D401, N802 — Qt-style API
            return None

    __slots__ = ("widget", "cloud", "lidar", "normal", "grid", "ghost")

    def __init__(self):
        self.widget = None
        self.cloud = self._Item()
        self.lidar = self._Item()
        self.normal = self._Item()
        self.grid = None
        self.ghost = None

    def set_ghost_mesh(self, verts, faces, color=None):
        """No-op for headless harnesses — no GL context to upload into."""
        return None


def build_three_panel_qt_widget(title="LiDAR Sim",
                                 terrain_img=None, terrain_extent=None,
                                 figsize=(8, 7),
                                 ghost_mesh=None):
    """Construct the hybrid Qt widget used by the single-agent GUI.

    Layout (horizontal, equal split):
        Left  — matplotlib FigureCanvasQTAgg with ax_map (terrain +
                 persistent obstacle halo + agent + path + goal)
        Right — PyQtGraph GLViewWidget for the 3D point cloud + ghost STL

    Returns a dict mirroring ``build_three_panel_figure`` so call-sites
    can pick whichever handles they need. Keys:
        widget            — top-level QWidget containing the layout
        fig, gs           — matplotlib figure + gridspec (2 columns now)
        canvas            — FigureCanvasQTAgg embedded in ``widget``
        ax_map, ax_bev    — matplotlib axes (no ax3d!)
        three_d           — ``ThreeDView`` wrapping the GLViewWidget
        global_img        — RGBA imshow on ax_map (set_data + set_extent)
        bev_handle        — RGB imshow on ax_bev (set_data)
        status            — figure-level status text on the mpl canvas
    """
    sim = _get_sim()
    gl, QtWidgets, QtCore = _get_pyqtgraph()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    # Ensure a QApplication exists before we instantiate any QWidget.
    app = QtWidgets.QApplication.instance()
    if app is None:
        import sys as _sys
        app = QtWidgets.QApplication(_sys.argv)

    # ── Matplotlib (map + BEV) ────────────────────────────────────
    fig = Figure(figsize=figsize, facecolor="#1e1e1e")
    # Single-panel matplotlib figure — the BEV was merged into the map.
    # ``GlobalCostmap.render_overlay`` now draws the inflation halo
    # directly on ``global_img`` in world-frame, so the robot-frame BEV
    # was redundant.
    gs = fig.add_gridspec(1, 1)

    # Map panel — terrain background, persistent obstacle halo, agent,
    # path, goal. All in world frame.
    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor("#222")
    if terrain_img is not None and terrain_extent is not None:
        ax_map.imshow(terrain_img, origin="lower", extent=terrain_extent,
                      cmap="gray", aspect="equal", vmin=0, vmax=1)
    # The overlay extent matches GLOBAL_HALF (the persistent costmap's
    # native extent). The map's xlim/ylim of ±8m crops to the
    # interesting region.
    global_img = ax_map.imshow(
        np.zeros((10, 10, 4), dtype=np.uint8),
        origin="lower", extent=[-sim.GLOBAL_HALF, sim.GLOBAL_HALF,
                                 -sim.GLOBAL_HALF, sim.GLOBAL_HALF],
        aspect="equal", alpha=0.85, zorder=3,
        interpolation="none")
    ax_map.set_xlim(-8, 8); ax_map.set_ylim(-8, 8)
    ax_map.set_title("Click=Place | Drag=Sweep | Shift+Click=Goal | G=Go",
                     color="white", fontsize=9)
    ax_map.set_xlabel("X (m)", color="white")
    ax_map.set_ylabel("Y (m)", color="white")
    ax_map.tick_params(colors="white")
    ax_map.grid(False, which="both")
    ax_map.minorticks_off()
    for s in ax_map.spines.values():
        s.set_edgecolor("#555")

    # ax_bev / bev_handle are None now — callers that still touch them
    # need an ``is not None`` guard.
    ax_bev = None
    bev_handle = None

    status = fig.text(0.5, 0.01, "",
                      ha="center", fontsize=10, color="white",
                      style="italic")

    canvas = FigureCanvasQTAgg(fig)
    # Let the matplotlib canvas accept keyboard focus so existing
    # mpl_connect("key_press_event", ...) handlers keep working when
    # focus is on either child widget.
    canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    # ── PyQtGraph (right) ────────────────────────────────────────
    three_d = ThreeDView()
    three_d.widget.setFocusPolicy(QtCore.Qt.StrongFocus)
    # Optional translucent STL ghost so the user can see the terrain
    # the agent is traversing through. ``ghost_mesh`` is a trimesh-like
    # object exposing ``vertices`` (N,3) and ``faces`` (M,3).
    if ghost_mesh is not None:
        try:
            three_d.set_ghost_mesh(ghost_mesh.vertices, ghost_mesh.faces)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[sim_view] ghost mesh upload failed: {exc}")

    # ── Compose into a single QWidget ────────────────────────────
    container = QtWidgets.QWidget()
    container.setWindowTitle(title)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    # After the BEV merge, the layout is two equal panels — matplotlib
    # map on the left, PyQtGraph GL view on the right.
    layout.addWidget(canvas, 1)
    layout.addWidget(three_d.widget, 1)

    return {
        "widget": container,
        "fig": fig,
        "gs": gs,
        "canvas": canvas,
        "ax_map": ax_map,
        "ax_bev": ax_bev,
        "three_d": three_d,
        "global_img": global_img,
        "bev_handle": bev_handle,
        "status": status,
    }


# ── Three-panel figure factory ─────────────────────────────────────

def build_three_panel_figure(title="LiDAR Sim",
                             terrain_img=None, terrain_extent=None,
                             figsize=(22, 7)):
    """Construct the three-panel figure used by both the GUI and the
    multi-agent ramp benchmark.

    Layout (horizontal GridSpec, width ratios 1.3 / 1 / 1):
        Left   — terrain map (top-down)         ax_map
        Middle — BEV PCA-grade costmap          ax_bev
        Right  — 3D point cloud                 ax3d

    All three axes are pre-styled with the dark theme used by the GUI
    (face colors, grid off on the map / on in 3D, tick colors, etc.).
    The caller is responsible for plotting agent markers / paths /
    overlays on top of the styled axes.

    Returns a dict so call-sites can take just the handles they need.

    Keys:
        fig, gs, ax_map, ax_bev, ax3d
        global_img        – RGBA imshow on ax_map (set_data + set_extent)
        bev_handle        – RGB imshow on ax_bev (set_data)
        status            – figure-level status text
    """
    sim = _get_sim()
    fig = plt.figure(figsize=figsize, facecolor="#1e1e1e")
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass  # Headless / Agg backend has no manager
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # ── Left: terrain + global overlay (top-down map) ──────────────
    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor("#222")
    if terrain_img is not None and terrain_extent is not None:
        ax_map.imshow(terrain_img, origin="lower", extent=terrain_extent,
                      cmap="gray", aspect="equal", vmin=0, vmax=1)
    # RGBA overlay imshow — call set_data + set_extent each frame.
    # interpolation='none' bypasses matplotlib's antialiased resample
    # pipeline; otherwise every blit re-resamples the 400×400 RGBA
    # texture (~50-80 ms per call) and heavy frames stall.
    global_img = ax_map.imshow(
        np.zeros((10, 10, 4), dtype=np.uint8),
        origin="lower", extent=[-8, 8, -8, 8],
        aspect="equal", alpha=0.6, zorder=3,
        interpolation="none")
    ax_map.set_xlim(-8, 8); ax_map.set_ylim(-8, 8)
    ax_map.set_xlabel("X (m)", color="white")
    ax_map.set_ylabel("Y (m)", color="white")
    ax_map.tick_params(colors="white")
    # grid() AFTER imshow / xlim / tick_params — those side-effect
    # the matplotlib axes grid back on.
    ax_map.grid(False, which="both")
    ax_map.minorticks_off()
    for s in ax_map.spines.values():
        s.set_edgecolor("#555")

    # ── Middle: PCA grade costmap (BEV) ────────────────────────────
    ax_bev = fig.add_subplot(gs[1])
    ax_bev.set_facecolor("#000")
    ax_bev.axis("off")
    ax_bev.set_title("PCA Grade Costmap (Red = LETHAL above threshold)",
                     color="white", fontsize=9, fontweight="bold")
    bev_handle = ax_bev.imshow(
        np.full((sim.BEV_SIZE, sim.BEV_SIZE, 3), 240, dtype=np.uint8),
        aspect="equal", interpolation="none")
    ax_bev.legend(handles=[
        Patch(facecolor=(0.94, 0.94, 0.94),
              edgecolor="gray", label="Free"),
        Patch(facecolor=(0.86, 0.12, 0.12),
              edgecolor="gray", label="Grade obstacle"),
    ], loc="upper right", fontsize=7, framealpha=0.5,
       facecolor="#222", labelcolor="white", edgecolor="gray")

    # ── Right: 3D point cloud ──────────────────────────────────────
    ax3d = fig.add_subplot(gs[2], projection="3d")
    ax3d.set_facecolor("#111")
    ax3d.grid(True)
    ax3d.set_title("3D Point Cloud (New System)",
                   color="white", fontsize=9, fontweight="bold")
    for axn in ("x", "y", "z"):
        getattr(ax3d, f"{axn}axis").pane.fill = False
        getattr(ax3d, f"{axn}axis").pane.set_edgecolor("#333")
    ax3d.tick_params(colors="white", labelsize=6)
    ax3d.view_init(elev=30, azim=-60)

    status = fig.text(0.5, 0.01, "",
                      ha="center", fontsize=10, color="white", style="italic")

    return {
        "fig": fig,
        "gs": gs,
        "ax_map": ax_map,
        "ax_bev": ax_bev,
        "ax3d": ax3d,
        "global_img": global_img,
        "bev_handle": bev_handle,
        "status": status,
    }


# ── 3D point-cloud panel update ────────────────────────────────────

class ThreeDState:
    """Per-panel scatter handles for the 3D point-cloud view.

    Holding scatter handles externally lets the same update function
    work for both the GUI (one persistent ax3d) and the benchmark
    (also one persistent ax3d). matplotlib 3D scatters can't be
    updated in-place, so each call removes the old handle and adds
    a new one.
    """
    __slots__ = ("cloud", "lidar", "normal")

    def __init__(self):
        self.cloud = None
        self.lidar = None
        self.normal = None


def _classify_cloud_colors(cloud_s, origin, slope_grid, obs_grid):
    """Per-point RGB classification used by both backends.

    Returns ``(colors_rgb, is_obs)`` — green for traversable, red for
    obstacle, gray for unclassified (point outside the local costmap
    window or in a cell where PCA produced no result).

    The colour reflects the algorithm's *final* per-cell decision
    (``obs_grid``), which is also what feeds the global costmap and
    A* planner. Earlier versions OR'd raw PCA slope into the colour
    check, which disagreed with the costmap whenever the bleed-zone
    or spike-override post-processing changed a cell's classification.
    """
    sim = _get_sim()
    gh2, gw2 = slope_grid.shape
    dx = cloud_s[:, 0] - origin[0]
    dy = cloud_s[:, 1] - origin[1]
    gx = ((dx + sim.COSTMAP_HALF) / sim.COSTMAP_RES).astype(int)
    gy = ((dy + sim.COSTMAP_HALF) / sim.COSTMAP_RES).astype(int)
    in_grid = (gx >= 0) & (gx < gw2) & (gy >= 0) & (gy < gh2)

    is_obs = np.zeros(len(cloud_s), dtype=bool)
    is_obs[in_grid] = obs_grid[gy[in_grid], gx[in_grid]]

    # Inside the local costmap, the colour mirrors what the planner sees:
    # red wherever obs_grid says obstacle, green everywhere else (cells
    # with no PCA result default to traversable in the planner too). Out-
    # side the window we simply have no local data, so paint gray rather
    # than misleadingly green.
    colors = np.empty((len(cloud_s), 3), dtype=np.float32)
    colors[:] = (0.45, 0.45, 0.45)               # gray = no local data
    colors[in_grid & ~is_obs] = (0.0, 0.82, 0.2) # green = traversable
    colors[is_obs] = (0.85, 0.0, 0.0)            # red = obstacle
    return colors, is_obs


def update_3d_panel(view, state_or_none, cloud, origin, sn, slope_grid, obs_grid,
                    n_lidar_points_max=3000):
    """Update the 3D point-cloud panel with binary-classified point colors.

    Dispatches on the view type:
      * ``matplotlib.axes.Axes`` (mplot3d) — legacy path, uses ``state``
        (a ``ThreeDState``) to remove + recreate scatters each call.
      * ``ThreeDView`` (PyQtGraph)        — fast path, ``setData`` on the
        persistent ``GLScatterPlotItem`` / ``GLLinePlotItem``.
      * ``MockThreeDView``                — no-op (headless harness).

    For the matplotlib path the second positional argument MUST be the
    ``ThreeDState`` instance (kept for backward compat with the
    benchmark). For the GL path it can be None.

    Coloring matches the algorithm's final per-cell decision (same
    signal that drives the costmap + A* planner):
        red   — ``obs_grid[cell]`` is True (wall, spike, DBSCAN cluster,
                or PCA-steep not cleared by bleed-zone override)
        green — point is inside the local costmap window and the cell
                is traversable (matches the planner's view)
        gray  — point is outside the local costmap window — no local
                data, planner has no opinion on it yet
    """
    # Dispatch on the view type. ThreeDView / MockThreeDView use GL;
    # everything else (matplotlib axes) falls through to the legacy path.
    if isinstance(view, (ThreeDView, MockThreeDView)):
        _update_3d_panel_gl(view, cloud, origin, sn, slope_grid, obs_grid,
                            n_lidar_points_max=n_lidar_points_max)
        return

    _update_3d_panel_mpl(view, state_or_none, cloud, origin, sn,
                         slope_grid, obs_grid,
                         n_lidar_points_max=n_lidar_points_max)


def _update_3d_panel_mpl(ax3d, state, cloud, origin, sn, slope_grid, obs_grid,
                          n_lidar_points_max=3000):
    """Legacy matplotlib (mplot3d) 3D-panel update — used by the
    multi-agent benchmark and any caller that still hands us an
    ``Axes3D``.  Same semantics as the original ``update_3d_panel``.
    """
    # Remove previous scatter / quiver handles.
    if state.cloud is not None:
        try: state.cloud.remove()
        except Exception: pass
    if state.lidar is not None:
        try: state.lidar.remove()
        except Exception: pass
    if state.normal is not None:
        try: state.normal.remove()
        except Exception: pass
    state.cloud = state.lidar = state.normal = None

    if cloud is None or len(cloud) == 0:
        return

    # Subsample for snappier rendering (3D scatter dominates frame cost).
    if len(cloud) > n_lidar_points_max:
        idx = np.random.default_rng(42).choice(
            len(cloud), n_lidar_points_max, replace=False)
        cloud_s = cloud[idx]
    else:
        cloud_s = cloud

    colors, _ = _classify_cloud_colors(cloud_s, origin, slope_grid, obs_grid)

    state.cloud = ax3d.scatter(
        cloud_s[:, 0], cloud_s[:, 1], cloud_s[:, 2],
        s=0.5, c=colors, alpha=0.9, depthshade=False)
    state.lidar = ax3d.scatter(
        [origin[0]], [origin[1]], [origin[2]],
        s=30, c="red", marker="o", zorder=100, depthshade=False)
    n = sn * 2
    state.normal = ax3d.quiver(
        origin[0], origin[1], origin[2],
        n[0], n[1], n[2],
        color="yellow", arrow_length_ratio=0.2, linewidth=2.5)

    # IMPORTANT: 1:1:1 aspect — do NOT change.
    view_r = 4.0
    ax3d.set_xlim(origin[0] - view_r, origin[0] + view_r)
    ax3d.set_ylim(origin[1] - view_r, origin[1] + view_r)
    ax3d.set_zlim(origin[2] - view_r, origin[2] + view_r)
    ax3d.set_box_aspect([1, 1, 1])
    # Keep the 3D grid on — gives the point cloud spatial context.
    ax3d.grid(True)


def _update_3d_panel_gl(view, cloud, origin, sn, slope_grid, obs_grid,
                         n_lidar_points_max=3000):
    """PyQtGraph fast-path 3D update — single VBO upload per item.

    ``view`` is either a real ``ThreeDView`` (drives the GLViewWidget)
    or a ``MockThreeDView`` (headless harness — every setData is a
    no-op so the rest of the GUI / animate loop runs unchanged).
    """
    if cloud is None or len(cloud) == 0:
        # Clear the scatter to an empty single point so the previous
        # frame's cloud doesn't linger when the scan returns nothing.
        empty_pos = np.zeros((1, 3), dtype=np.float32)
        empty_col = np.zeros((1, 4), dtype=np.float32)
        view.cloud.setData(pos=empty_pos, color=empty_col, size=2.0)
        return

    # Subsample (same RNG seed as the mpl path for parity in test runs).
    if len(cloud) > n_lidar_points_max:
        idx = np.random.default_rng(42).choice(
            len(cloud), n_lidar_points_max, replace=False)
        cloud_s = cloud[idx]
    else:
        cloud_s = cloud

    colors_rgb, _ = _classify_cloud_colors(cloud_s, origin, slope_grid, obs_grid)

    # Pad to RGBA — PyQtGraph wants (N, 4) float32 in 0-1.
    colors_rgba = np.empty((len(cloud_s), 4), dtype=np.float32)
    colors_rgba[:, :3] = colors_rgb
    colors_rgba[:, 3] = 0.9  # matches the matplotlib alpha=0.9
    pos = np.ascontiguousarray(cloud_s[:, :3], dtype=np.float32)

    view.cloud.setData(pos=pos, color=colors_rgba, size=2.0)

    # Origin marker — red dot, screen-size 10 px.
    view.lidar.setData(
        pos=np.asarray([origin], dtype=np.float32),
        color=np.array([[0.85, 0.0, 0.0, 1.0]], dtype=np.float32),
        size=10.0,
    )

    # Yellow surface-normal quiver: origin → origin + 2 * sn.
    n = np.asarray(sn, dtype=np.float32) * 2.0
    quiver = np.empty((2, 3), dtype=np.float32)
    quiver[0] = origin
    quiver[1] = (origin[0] + n[0], origin[1] + n[1], origin[2] + n[2])
    view.normal.setData(pos=quiver, color=(1.0, 1.0, 0.0, 1.0), width=2.5,
                        antialias=True)

    # Recenter the camera on the LiDAR origin (matches the 4 m view
    # radius from the matplotlib path). Only update the center — the
    # user can still rotate / zoom freely.
    if hasattr(view, "widget") and view.widget is not None:
        try:
            from pyqtgraph import Vector  # local import: only on the real path
            view.widget.opts["center"] = Vector(
                float(origin[0]), float(origin[1]), float(origin[2]))
        except Exception:
            pass


# ── Multi-agent BEV (used by the ramp benchmark) ───────────────────

def render_bev_multi(agents_obs, size=None, half=None):
    """Per-agent-colored BEV view of the latest local-scan obstacles.

    Each agent contributes its current local PCA-grade obstacle grid;
    the cell is painted with that agent's color. Where multiple agents
    overlap on a cell the latest paint wins (cheap and adequate for a
    visualization).

    ``agents_obs`` is a list of (obs_grid, color_hex) tuples — pass
    None for any agent's obs_grid that hasn't scanned yet and the
    entry will be skipped.
    """
    sim = _get_sim()
    if size is None:
        size = sim.BEV_SIZE
    if half is None:
        half = sim.COSTMAP_HALF
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    scale = size / (2 * half)
    cx, cy = size // 2, size // 2

    for obs, color_hex in agents_obs:
        if obs is None or not obs.any():
            continue
        hx = color_hex.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        gh, gw = obs.shape
        ys, xs = np.where(obs)
        for gy_, gx_ in zip(ys, xs):
            x0 = int(round(cx + (gx_ * sim.COSTMAP_RES - half) * scale))
            x1 = int(round(cx + ((gx_ + 1) * sim.COSTMAP_RES - half) * scale))
            y0 = int(round(cy - ((gy_ + 1) * sim.COSTMAP_RES - half) * scale))
            y1 = int(round(cy - (gy_ * sim.COSTMAP_RES - half) * scale))
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(size, x1); y1 = min(size, y1)
            if x0 < x1 and y0 < y1:
                img[y0:y1, x0:x1] = (r, g, b)

    # Origin marker (small yellow square)
    for d in range(-2, 3):
        for e in range(-2, 3):
            xx, yy = cx + d, cy + e
            if 0 <= xx < size and 0 <= yy < size:
                img[yy, xx] = (255, 255, 0)
    return img
