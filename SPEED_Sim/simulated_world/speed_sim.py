#!/usr/bin/env python3
"""SPEED Sim — per-wheel PID velocity control on a Chaplygin sleigh.

Replaces the legacy 0-75 arbitrary-units speed system on AutoNav with a
direct MPH controller. Internally the setpoint is a tick in 0..50, each
worth 0.1 MPH (so the full range is 0.0 .. 5.0 MPH). The number-row keys
1..9, 0 jump straight to the 0.5-MPH grid (1 -> 0.5, 2 -> 1.0, ..., 9 ->
4.5, 0 -> 5.0). The [ / ] keys nudge by 0.1 MPH for fine tuning.

Drive with the arrow keys:
  Up       both wheels at +setpoint
  Down     both wheels at -setpoint
  Left     differential: bias the right wheel positive (CCW)
  Right    differential: bias the left  wheel positive (CW)

Each wheel runs its own PID at a native 50 Hz control rate. The PID
reads measured wheel velocity from the kinematic relation
v_wheel = R * w_encoder = u +/- omega * TRACK_WIDTH / 2 and emits a
saturated force command for the motor. Physics: two rear knife-edge
wheels + frictionless front caster, copied verbatim from
BEHAVIOR TREE Sim/simulated_world/bt_sim_gui.py so the dynamics match.

The arena is an open box with:
  * orange "rough" patches that add extra linear damping (gravel / grass)
  * blue gradient regions that apply a constant world-frame force at the
    COM (slopes pushing the robot off-course)

The HUD on the right shows two scopes:
  * left wheel: commanded square wave (cyan dash) vs measured (white)
  * right wheel: commanded square wave (cyan dash) vs measured (white)
with a numeric panel below.

Pygame at 60 FPS, physics at 240 Hz (4 substeps/frame). No matplotlib.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


# ============================================================================
#                                Constants
# ============================================================================
# Robot — match BEHAVIOR TREE Sim exactly so the dynamics are comparable.
ROBOT_MASS_KG       = 35.0           # 77 lb
COM_OFFSET_M        = 0.25
WHEELBASE_M         = 0.39
TRACK_WIDTH_M       = 0.54
WHEEL_RADIUS_M      = 0.20
CASTER_RADIUS_M     = 0.09
FOOTPRINT_HALF_W    = 0.21
FOOTPRINT_LEN_BACK  = 0.10
FOOTPRINT_LEN_FWD   = WHEELBASE_M + 0.05
_L_FP = FOOTPRINT_LEN_BACK + FOOTPRINT_LEN_FWD
_W_FP = 2 * FOOTPRINT_HALF_W
INERTIA_COM         = ROBOT_MASS_KG * (_L_FP * _L_FP + _W_FP * _W_FP) / 12.0
INERTIA_REAR        = INERTIA_COM + ROBOT_MASS_KG * COM_OFFSET_M * COM_OFFSET_M

# Motor / wheel-force envelope — symmetric for the speed PID (BT sim
# asymmetrically capped reverse because its planner never demanded large
# reverse forces; here the driver can floor reverse).
F_WHEEL_MAX_N       = 200.0
F_WHEEL_MIN_N       = -200.0

# Baseline damping = drivetrain + air drag.
LIN_DAMP            = 6.0
ANG_DAMP            = 2.0

# Speed system — 0.1 MPH granularity, 0..5.0 MPH range = 50 ticks.
MPS_PER_MPH         = 0.44704
SPEED_TICK_MPH      = 0.1
SPEED_TICKS_MAX     = 50
DEFAULT_TICK        = 30             # 3.0 MPH

# PID — per-wheel, output is force in newtons.
CTRL_RATE_HZ        = 50.0           # native motor-controller rate
CTRL_DT             = 1.0 / CTRL_RATE_HZ

# Gains tuned for a near-critical response on the 35 kg sleigh. The
# per-wheel effective mass on the Chaplygin sleigh is roughly
# 1 / (1/m + T^2/(4*I_rear)) ~ 20 kg, so critical Kd ~ 2*sqrt(Kp*m_eff)
# minus the baseline LIN_DAMP. Without filtering the derivative, raw
# Kd values that large amplify rapid velocity changes during the
# initial bang-bang ramp; we instead use a moderate Kp and a low-passed
# derivative (D_FILTER_ALPHA) to stay critically damped without
# saturating on every measurement bump.
KP_WHEEL_DEFAULT    = 120.0
KI_WHEEL_DEFAULT    = 55.0
KD_WHEEL_DEFAULT    = 28.0
D_FILTER_ALPHA      = 0.35           # EWMA on d_meas (0 = freeze, 1 = raw)
INTEGRAL_LIMIT_N    = 60.0           # anti-windup clamp on Ki * integral
                                     # (max newtons the I term may contribute)

# Arrow-key kinematics: convert held arrow keys -> wheel velocity setpoints.
TURN_SHARE          = 0.6            # how much of the linear setpoint
                                     # is reallocated as a turn delta

# Physics integration.
PHYS_DT             = 1.0 / 240.0
RENDER_FPS          = 60
SUBSTEPS_PER_FRAME  = int(round((1.0 / RENDER_FPS) / PHYS_DT))

# World — open arena.
ARENA_W_M           = 30.0
ARENA_H_M           = 20.0
WALL_THICKNESS_M    = 0.25

# Window layout.
WIN_W               = 1480
WIN_H               = 820
ARENA_PANEL_W       = 880
PLOT_PANEL_X        = ARENA_PANEL_W
PLOT_PANEL_W        = WIN_W - ARENA_PANEL_W
PLOT_HEIGHT         = 230
PLOT_MARGIN         = 14
HUD_HEIGHT          = WIN_H - 2 * PLOT_HEIGHT - 3 * PLOT_MARGIN

# Plot history: 5 seconds at the control rate.
HISTORY_SECS        = 5.0
HISTORY_N           = int(round(HISTORY_SECS * CTRL_RATE_HZ))

# Colors.
BG               = (18, 18, 22)
PANEL_BG         = (24, 26, 30)
WALL_COLOR       = (200, 200, 210)
FLOOR_COLOR      = (38, 42, 48)
PATCH_COLOR      = (180, 110,  40)   # rough/friction patch
PATCH_COLOR_EDGE = (220, 150,  60)
GRAD_COLOR       = ( 60, 110, 180)   # slope/gradient region
GRAD_COLOR_EDGE  = (110, 170, 230)
GRAD_ARROW_COLOR = (210, 230, 255)
ROBOT_BODY       = (230, 230, 230)
ROBOT_OUTLINE    = ( 30,  30,  35)
KNIFE_COLOR      = (255, 200,  90)
CASTER_COLOR     = (130, 200, 255)
HEADING_COLOR    = (255,  90,  90)
TRAIL_COLOR      = ( 90, 200, 255)
CMD_COLOR        = (110, 230, 240)   # commanded velocity
MEAS_COLOR       = (240, 240, 240)   # measured velocity
ZERO_LINE        = ( 80,  80,  85)
GRID_LINE        = ( 50,  52,  58)
TEXT_COLOR       = (235, 235, 240)
TEXT_DIM         = (160, 160, 168)
ACCENT           = (110, 230, 240)


# ============================================================================
#                                  Robot
# ============================================================================
@dataclass
class Robot:
    """Rear-axle midpoint state. Identical to BT Sim's Robot — see
    bt_sim_gui.py line 951. Wheel longitudinal speeds are derived from
    the rigid-body kinematics:
        v_left  = u - omega * TRACK_WIDTH / 2
        v_right = u + omega * TRACK_WIDTH / 2
    which is what an ideal encoder reads (V = R * w).
    """
    x: float
    y: float
    theta: float
    u: float = 0.0
    omega: float = 0.0
    F_left: float = 0.0
    F_right: float = 0.0

    @property
    def v_left(self) -> float:
        return self.u - self.omega * TRACK_WIDTH_M / 2.0

    @property
    def v_right(self) -> float:
        return self.u + self.omega * TRACK_WIDTH_M / 2.0

    def rear_axle(self):
        return self.x, self.y

    def com(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + COM_OFFSET_M * c, self.y + COM_OFFSET_M * s)

    def front_caster(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + WHEELBASE_M * c, self.y + WHEELBASE_M * s)

    def left_knife(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x - s * TRACK_WIDTH_M / 2,
                self.y + c * TRACK_WIDTH_M / 2)

    def right_knife(self):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return (self.x + s * TRACK_WIDTH_M / 2,
                self.y - c * TRACK_WIDTH_M / 2)

    def step(self, F_left: float, F_right: float, dt: float,
             arena: "Arena") -> None:
        F_left  = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_left))
        F_right = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_right))
        self.F_left, self.F_right = F_left, F_right

        F_total     = F_left + F_right
        torque_rear = (F_right - F_left) * TRACK_WIDTH_M / 2.0

        # World disturbance: gradient regions push the COM with a
        # constant world-frame force. Split into body-frame (along, lat)
        # — along feeds linear acceleration, lat * COM_OFFSET creates a
        # rear-axle torque.
        com_x, com_y = self.com()
        fx_world, fy_world = arena.force_at(com_x, com_y)
        c, s = math.cos(self.theta), math.sin(self.theta)
        F_along_dist = fx_world * c + fy_world * s
        F_lat_dist   = -fx_world * s + fy_world * c
        torque_dist  = F_lat_dist * COM_OFFSET_M

        # Friction patches: extra damping when the COM sits on a rough
        # patch. Scales both linear and angular damping (rougher ground
        # also resists yaw).
        extra_damp_lin = arena.friction_at(com_x, com_y)
        extra_damp_ang = extra_damp_lin * 0.4

        du = (F_total + F_along_dist
              + ROBOT_MASS_KG * COM_OFFSET_M * self.omega ** 2
              - (LIN_DAMP + extra_damp_lin) * self.u) / ROBOT_MASS_KG
        dw = (torque_rear + torque_dist
              - ROBOT_MASS_KG * COM_OFFSET_M * self.u * self.omega
              - (ANG_DAMP + extra_damp_ang) * self.omega) / INERTIA_REAR

        self.u     += du * dt
        self.omega += dw * dt
        self.x     += self.u * c * dt
        self.y     += self.u * s * dt
        self.theta += self.omega * dt
        if self.theta > math.pi:
            self.theta -= 2 * math.pi
        elif self.theta <= -math.pi:
            self.theta += 2 * math.pi

        self._resolve_walls(arena)

    def _resolve_walls(self, arena: "Arena") -> None:
        """Push the rear axle and footprint points off the four arena
        walls. Cheap circle-vs-AABB clamp on each reference point."""
        rad = FOOTPRINT_HALF_W + 0.02
        xmin, ymin = rad, rad
        xmax, ymax = arena.W - rad, arena.H - rad
        # Clamp rear axle into a slightly-shrunken AABB. With knife edge
        # offsets we additionally clamp using the most-extreme knife.
        pts = [self.left_knife(), self.right_knife(),
               self.front_caster(), self.rear_axle()]
        nx = ny = 0.0
        push = 0.0
        for (px, py) in pts:
            if px < xmin:
                d = xmin - px
                if d > push:
                    push, nx, ny = d, 1.0, 0.0
            elif px > xmax:
                d = px - xmax
                if d > push:
                    push, nx, ny = d, -1.0, 0.0
            if py < ymin:
                d = ymin - py
                if d > push:
                    push, nx, ny = d, 0.0, 1.0
            elif py > ymax:
                d = ymax - py
                if d > push:
                    push, nx, ny = d, 0.0, -1.0
        if push > 0:
            self.x += nx * push
            self.y += ny * push
            vx = self.u * math.cos(self.theta)
            vy = self.u * math.sin(self.theta)
            vn = vx * nx + vy * ny
            if vn < 0:
                self.u     *= max(0.0, 1.0 - abs(vn) * 0.5)
                self.omega *= 0.7


# ============================================================================
#                                  Arena
# ============================================================================
@dataclass
class FrictionPatch:
    cx: float; cy: float
    w: float; h: float
    extra_damp: float


@dataclass
class GradientRegion:
    cx: float; cy: float
    w: float; h: float
    fx: float; fy: float          # constant world-frame force, N


class Arena:
    """Open arena (ARENA_W_M x ARENA_H_M). Walls at the boundary.
    Random rough patches (raise damping) and gradient regions (push the
    COM) provide the disturbance test bed for the PID."""

    def __init__(self, seed: int):
        self.W = ARENA_W_M
        self.H = ARENA_H_M
        rng = random.Random(seed)
        self.patches: list[FrictionPatch] = []
        self.gradients: list[GradientRegion] = []

        # Friction patches scattered in a rough grid for predictable
        # coverage, then jittered. Patches are bigger than the robot so
        # the disturbance lasts through multiple control cycles.
        for gx in range(2, 6):
            for gy in range(2, 4):
                if rng.random() < 0.55:
                    continue
                cx = gx * (self.W / 7) + rng.uniform(-1.0, 1.0)
                cy = gy * (self.H / 5) + rng.uniform(-0.8, 0.8)
                w  = rng.uniform(2.0, 3.6)
                h  = rng.uniform(1.6, 3.0)
                damp = rng.uniform(18.0, 38.0)
                self.patches.append(FrictionPatch(cx, cy, w, h, damp))

        # Gradient regions — slope-like, pointed roughly along the long
        # axis to disturb forward driving.
        for _ in range(4):
            w = rng.uniform(3.5, 5.5)
            h = rng.uniform(3.5, 5.5)
            cx = rng.uniform(w / 2 + 1.5, self.W - w / 2 - 1.5)
            cy = rng.uniform(h / 2 + 1.5, self.H - h / 2 - 1.5)
            ang = rng.uniform(0, 2 * math.pi)
            mag = rng.uniform(45.0, 85.0)
            self.gradients.append(
                GradientRegion(cx, cy, w, h,
                               mag * math.cos(ang),
                               mag * math.sin(ang)))

    def friction_at(self, x: float, y: float) -> float:
        for p in self.patches:
            if (abs(x - p.cx) <= p.w / 2 and
                    abs(y - p.cy) <= p.h / 2):
                return p.extra_damp
        return 0.0

    def force_at(self, x: float, y: float) -> tuple[float, float]:
        fx = fy = 0.0
        for g in self.gradients:
            if (abs(x - g.cx) <= g.w / 2 and
                    abs(y - g.cy) <= g.h / 2):
                fx += g.fx
                fy += g.fy
        return (fx, fy)


# ============================================================================
#                                   PID
# ============================================================================
@dataclass
class WheelPID:
    """Per-wheel velocity PID. Output is a wheel force in newtons,
    saturated to the motor envelope. Anti-windup via conditional
    integration: don't accumulate if the output is already pinned in
    the direction the error wants to push."""
    kp: float
    ki: float
    kd: float
    d_alpha: float = D_FILTER_ALPHA
    i_limit: float = INTEGRAL_LIMIT_N
    _integral: float = 0.0
    _prev_meas: float = 0.0
    _d_filt: float = 0.0
    _last_out: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_meas = 0.0
        self._d_filt = 0.0
        self._last_out = 0.0

    def step(self, sp: float, meas: float, dt: float,
             u_min: float = F_WHEEL_MIN_N,
             u_max: float = F_WHEEL_MAX_N) -> float:
        err = sp - meas
        # Derivative on measurement + EWMA low-pass (avoids derivative
        # kick on step SP changes — which is exactly what the arrow-key
        # drive produces — and damps measurement spikes during the
        # bang-bang ramp).
        d_meas = (meas - self._prev_meas) / dt if dt > 0 else 0.0
        self._d_filt += self.d_alpha * (d_meas - self._d_filt)
        self._prev_meas = meas

        # i_limit is expressed in newtons (max contribution of the I
        # term). Convert to an integral cap given Ki.
        if self.ki > 1e-9:
            int_cap = self.i_limit / self.ki
        else:
            int_cap = 0.0
        tent_int = self._integral + err * dt
        tent_int = max(-int_cap, min(int_cap, tent_int))
        u_unsat = self.kp * err + self.ki * tent_int - self.kd * self._d_filt
        u_sat = max(u_min, min(u_max, u_unsat))

        # Anti-windup: integrate only if not saturated, OR saturated but
        # the error wants to pull us back out of saturation.
        saturated_high = u_unsat > u_max and err > 0
        saturated_low  = u_unsat < u_min and err < 0
        if not (saturated_high or saturated_low):
            self._integral = tent_int

        self._last_out = u_sat
        return u_sat


# ============================================================================
#                              Plot ring buffer
# ============================================================================
class ScopeBuffer:
    """Ring buffer of (commanded, measured) wheel velocity in m/s.
    Drawn as a single LineCollection-style polyline per series."""

    def __init__(self, n: int):
        self.n = n
        self.cmd = np.zeros(n, dtype=np.float32)
        self.meas = np.zeros(n, dtype=np.float32)
        self.idx = 0  # write head

    def push(self, cmd: float, meas: float) -> None:
        self.cmd[self.idx] = cmd
        self.meas[self.idx] = meas
        self.idx = (self.idx + 1) % self.n

    def ordered(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (cmd, meas) oldest-first so the rightmost sample is
        the most recent (= the time axis flows left -> right)."""
        i = self.idx
        cmd = np.concatenate((self.cmd[i:], self.cmd[:i]))
        meas = np.concatenate((self.meas[i:], self.meas[:i]))
        return cmd, meas


# ============================================================================
#                                 Sim
# ============================================================================
class Sim:
    def __init__(self, args):
        rng = random.Random(args.seed)
        self.arena = Arena(args.seed)
        self.robot = Robot(
            x=2.0, y=self.arena.H / 2.0, theta=0.0)
        self.pid_L = WheelPID(args.kp, args.ki, args.kd)
        self.pid_R = WheelPID(args.kp, args.ki, args.kd)

        self.tick = DEFAULT_TICK
        self.up = self.down = self.left = self.right = False

        # Held commanded wheel velocities (m/s), updated on key events.
        self.cmd_L_mps = 0.0
        self.cmd_R_mps = 0.0

        # Control-loop accumulator (allow physics & render to run at
        # different rates than the 50 Hz controller).
        self._ctrl_acc = 0.0

        # Plot history.
        self.scope_L = ScopeBuffer(HISTORY_N)
        self.scope_R = ScopeBuffer(HISTORY_N)

        # Trail of COM positions for the arena view.
        self.trail: list[tuple[float, float]] = []
        self._trail_last_xy: Optional[tuple[float, float]] = None
        self.TRAIL_DROP_M = 0.10
        self.TRAIL_MAX = 600

        # PID-runtime adjustment via keys (record for HUD).
        self.kp_step = 10.0
        self.ki_step = 5.0
        self.kd_step = 2.0

        # Force-history (for HUD min/max indicators).
        self._F_max_seen = 0.0

        # Live recording (toggled with V).
        self._rec_proc: Optional[subprocess.Popen] = None
        self._rec_path: Optional[str] = None
        self._rec_start_t: float = 0.0
        self._rec_fps: int = RENDER_FPS

    # ── Live MP4 recording ──
    @property
    def recording(self) -> bool:
        return self._rec_proc is not None

    def rec_elapsed(self) -> float:
        if not self.recording:
            return 0.0
        return time.perf_counter() - self._rec_start_t

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        bakes_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "bakes")
        os.makedirs(bakes_dir, exist_ok=True)
        path = os.path.join(bakes_dir, f"speed_sim_live_{ts}.mp4")
        self._rec_proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIN_W}x{WIN_H}",
                "-r", str(self._rec_fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "18",
                path,
            ],
            stdin=subprocess.PIPE,
        )
        self._rec_path = path
        self._rec_start_t = time.perf_counter()
        print(f"[REC] start -> {path}", flush=True)

    def _stop_recording(self) -> None:
        if self._rec_proc is None:
            return
        elapsed = time.perf_counter() - self._rec_start_t
        path = self._rec_path
        try:
            if self._rec_proc.stdin:
                self._rec_proc.stdin.close()
        except BrokenPipeError:
            pass
        self._rec_proc.wait()
        self._rec_proc = None
        self._rec_path = None
        print(f"[REC] stop  -> {path}  ({elapsed:.1f}s)", flush=True)

    def record_frame(self, surface: "pygame.Surface") -> None:
        if not self.recording:
            return
        arr = pygame.surfarray.array3d(surface)
        arr = np.swapaxes(arr, 0, 1)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        try:
            self._rec_proc.stdin.write(arr.tobytes())
        except BrokenPipeError:
            # ffmpeg died — clean up so the rest of the session keeps running.
            self._stop_recording()

    # ── Speed setpoint helpers ──
    def setpoint_mps(self) -> float:
        return self.tick * SPEED_TICK_MPH * MPS_PER_MPH

    def setpoint_mph(self) -> float:
        return self.tick * SPEED_TICK_MPH

    # ── Arrow keys -> per-wheel velocity setpoints ──
    def recompute_cmd(self) -> None:
        sp = self.setpoint_mps()
        fwd = (1 if self.up else 0) - (1 if self.down else 0)
        turn = (1 if self.right else 0) - (1 if self.left else 0)
        # Pure spin: both arrows release the linear component but keep
        # the turn so the robot rotates in place at the selected speed.
        base = fwd * sp
        diff = turn * sp * TURN_SHARE
        # Convention: turn = +1 (right) -> negative ω -> CW -> right.
        # ω = (v_R - v_L) / T,  so for CW we want v_L > v_R.
        # base + diff -> left, base - diff -> right.
        self.cmd_L_mps = base + diff
        self.cmd_R_mps = base - diff

    # ── Step the world ──
    def update(self, dt: float) -> None:
        # Run physics substeps at PHYS_DT until we've consumed dt.
        remaining = dt
        ctrl_dt_acc = self._ctrl_acc
        F_L = self.robot.F_left
        F_R = self.robot.F_right
        while remaining > 1e-9:
            step = min(PHYS_DT, remaining)
            ctrl_dt_acc += step
            # Fire the controller at the native rate.
            if ctrl_dt_acc >= CTRL_DT:
                F_L = self.pid_L.step(self.cmd_L_mps,
                                      self.robot.v_left, CTRL_DT)
                F_R = self.pid_R.step(self.cmd_R_mps,
                                      self.robot.v_right, CTRL_DT)
                ctrl_dt_acc -= CTRL_DT
                # Log to scopes at the controller rate so the time axis
                # corresponds to actual PID cycles.
                self.scope_L.push(self.cmd_L_mps, self.robot.v_left)
                self.scope_R.push(self.cmd_R_mps, self.robot.v_right)
            self.robot.step(F_L, F_R, step, self.arena)
            remaining -= step
        self._ctrl_acc = ctrl_dt_acc

        # Trail
        cx, cy = self.robot.com()
        if (self._trail_last_xy is None or
                math.hypot(cx - self._trail_last_xy[0],
                           cy - self._trail_last_xy[1]) >= self.TRAIL_DROP_M):
            self.trail.append((cx, cy))
            self._trail_last_xy = (cx, cy)
            if len(self.trail) > self.TRAIL_MAX:
                self.trail = self.trail[-self.TRAIL_MAX:]

        self._F_max_seen = max(self._F_max_seen,
                               abs(self.robot.F_left),
                               abs(self.robot.F_right))


# ============================================================================
#                                Renderer
# ============================================================================
class Renderer:
    def __init__(self, sim: Sim):
        self.sim = sim
        pygame.init()
        pygame.display.set_caption("SPEED Sim — per-wheel PID velocity control")
        self.screen = pygame.display.set_mode(
            (WIN_W, WIN_H), pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Fonts.
        self.font = pygame.font.SysFont(
            "Menlo,Consolas,monospace", 14)
        self.font_sm = pygame.font.SysFont(
            "Menlo,Consolas,monospace", 12)
        self.font_lg = pygame.font.SysFont(
            "Menlo,Consolas,monospace", 18, bold=True)

        # Arena-panel world->screen transform.
        margin = 24
        avail_w = ARENA_PANEL_W - 2 * margin
        avail_h = WIN_H - 2 * margin
        self.scale = min(avail_w / ARENA_W_M, avail_h / ARENA_H_M)
        view_w = ARENA_W_M * self.scale
        view_h = ARENA_H_M * self.scale
        self.ox = margin + (avail_w - view_w) / 2
        self.oy = margin + (avail_h - view_h) / 2
        self.view_w = view_w
        self.view_h = view_h

        self._static_arena = self._build_static_arena()

    # ── Coordinate transforms ──
    def w2s(self, x: float, y: float) -> tuple[int, int]:
        # World y is up; screen y is down.
        return (int(self.ox + x * self.scale),
                int(self.oy + (ARENA_H_M - y) * self.scale))

    def w2s_len(self, m: float) -> float:
        return m * self.scale

    # ── Static arena surface ──
    def _build_static_arena(self) -> pygame.Surface:
        surf = pygame.Surface((ARENA_PANEL_W, WIN_H), pygame.SRCALPHA)
        # Floor
        x0, y0 = self.w2s(0.0, ARENA_H_M)
        x1, y1 = self.w2s(ARENA_W_M, 0.0)
        floor_rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        surf.fill((0, 0, 0, 0))
        pygame.draw.rect(surf, FLOOR_COLOR, floor_rect)
        # Grid lines every metre.
        for gx in range(0, int(ARENA_W_M) + 1):
            sx, _ = self.w2s(gx, 0.0)
            _, sy0 = self.w2s(gx, ARENA_H_M)
            _, sy1 = self.w2s(gx, 0.0)
            pygame.draw.line(surf, GRID_LINE, (sx, sy0), (sx, sy1), 1)
        for gy in range(0, int(ARENA_H_M) + 1):
            sx0, sy = self.w2s(0.0, gy)
            sx1, _ = self.w2s(ARENA_W_M, gy)
            pygame.draw.line(surf, GRID_LINE, (sx0, sy), (sx1, sy), 1)

        # Friction patches.
        for p in self.sim.arena.patches:
            self._draw_rect_world(surf, p.cx, p.cy, p.w, p.h,
                                  PATCH_COLOR, PATCH_COLOR_EDGE, alpha=190)
            # Hatching to suggest "rough"
            self._hatch_rect_world(surf, p.cx, p.cy, p.w, p.h)

        # Gradient regions.
        for g in self.sim.arena.gradients:
            self._draw_rect_world(surf, g.cx, g.cy, g.w, g.h,
                                  GRAD_COLOR, GRAD_COLOR_EDGE, alpha=160)
            # Force-direction arrow inside the region.
            cx, cy = self.w2s(g.cx, g.cy)
            mag = math.hypot(g.fx, g.fy) or 1.0
            ax = g.fx / mag
            ay = g.fy / mag
            L = self.w2s_len(min(g.w, g.h)) * 0.40
            ex = cx + ax * L
            ey = cy - ay * L
            pygame.draw.line(surf, GRAD_ARROW_COLOR,
                             (cx, cy), (ex, ey), 3)
            # Arrow head
            head_a = math.atan2(ey - cy, ex - cx)
            for sign in (-1, 1):
                ha = head_a + sign * 0.5
                hx = ex - 10 * math.cos(ha)
                hy = ey - 10 * math.sin(ha)
                pygame.draw.line(surf, GRAD_ARROW_COLOR,
                                 (ex, ey), (hx, hy), 3)

        # Outer walls.
        wall_px = max(2, int(self.w2s_len(WALL_THICKNESS_M)))
        x0, y0 = self.w2s(0.0, ARENA_H_M)
        x1, y1 = self.w2s(ARENA_W_M, 0.0)
        pygame.draw.rect(surf, WALL_COLOR,
                         pygame.Rect(x0, y0, x1 - x0, y1 - y0),
                         wall_px)
        return surf

    def _draw_rect_world(self, surf, cx, cy, w, h,
                         fill, edge, alpha=255):
        x0, y0 = self.w2s(cx - w / 2, cy + h / 2)
        x1, y1 = self.w2s(cx + w / 2, cy - h / 2)
        rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
        tinted = pygame.Surface(rect.size, pygame.SRCALPHA)
        tinted.fill((*fill, alpha))
        surf.blit(tinted, rect.topleft)
        pygame.draw.rect(surf, edge, rect, 2)

    def _hatch_rect_world(self, surf, cx, cy, w, h):
        x0, y0 = self.w2s(cx - w / 2, cy + h / 2)
        x1, y1 = self.w2s(cx + w / 2, cy - h / 2)
        step = max(6, int(self.w2s_len(0.30)))
        col = (245, 200, 140, 90)
        # Diagonal hatches: y = x + b -> b spans (y0-x1) .. (y1-x0)
        for b in range(y0 - x1, y1 - x0, step):
            sx0 = x0
            sy0 = x0 + b
            sx1 = x1
            sy1 = x1 + b
            # Clip into rect.
            if sy0 < y0:
                sx0 += (y0 - sy0); sy0 = y0
            if sy1 > y1:
                sx1 -= (sy1 - y1); sy1 = y1
            if sx0 < x0 or sx1 > x1 or sx0 >= sx1:
                continue
            hatch = pygame.Surface((1, 1), pygame.SRCALPHA)
            pygame.draw.line(surf, col, (sx0, sy0), (sx1, sy1), 1)

    # ── Robot ──
    def _draw_robot(self, surf):
        r = self.sim.robot
        c, s = math.cos(r.theta), math.sin(r.theta)
        # Body rectangle in body frame: x in [-back, +fwd], y in [-hw, +hw]
        corners_body = [
            (FOOTPRINT_LEN_FWD,  FOOTPRINT_HALF_W),
            (FOOTPRINT_LEN_FWD, -FOOTPRINT_HALF_W),
            (-FOOTPRINT_LEN_BACK, -FOOTPRINT_HALF_W),
            (-FOOTPRINT_LEN_BACK,  FOOTPRINT_HALF_W),
        ]
        body_pts = []
        for bx, by in corners_body:
            wx = r.x + bx * c - by * s
            wy = r.y + bx * s + by * c
            body_pts.append(self.w2s(wx, wy))
        pygame.draw.polygon(surf, ROBOT_BODY, body_pts)
        pygame.draw.polygon(surf, ROBOT_OUTLINE, body_pts, 2)

        # Knife edges
        lk = self.w2s(*r.left_knife())
        rk = self.w2s(*r.right_knife())
        kw = max(2, int(self.w2s_len(0.10)))
        # Draw as short bars along body-x at the wheel.
        for (kx, ky) in (lk, rk):
            half = self.w2s_len(WHEEL_RADIUS_M)
            ax = kx + c * half
            ay = ky - s * half
            bx = kx - c * half
            by = ky + s * half
            pygame.draw.line(surf, KNIFE_COLOR,
                             (ax, ay), (bx, by), kw)

        # Caster
        caster = self.w2s(*r.front_caster())
        pygame.draw.circle(surf, CASTER_COLOR, caster,
                           max(2, int(self.w2s_len(CASTER_RADIUS_M))))
        pygame.draw.circle(surf, ROBOT_OUTLINE, caster,
                           max(2, int(self.w2s_len(CASTER_RADIUS_M))), 1)

        # Heading marker — short line from rear axle out the front.
        ra = self.w2s(*r.rear_axle())
        fc = self.w2s(*r.front_caster())
        pygame.draw.line(surf, HEADING_COLOR, ra, fc, 2)

    def _draw_rec_indicator(self, surf):
        if not self.sim.recording:
            return
        elapsed = self.sim.rec_elapsed()
        m = int(elapsed // 60)
        s = elapsed - m * 60
        txt = f"REC  {m:d}:{s:04.1f}"
        rec_color = (240, 80, 80)
        label = self.font_lg.render(txt, True, rec_color)
        margin = 14
        x = ARENA_PANEL_W - label.get_width() - margin
        y = margin
        # Blinking dot (1 Hz).
        if int(elapsed * 2) % 2 == 0:
            pygame.draw.circle(surf, rec_color,
                               (x - 14, y + label.get_height() // 2), 7)
        surf.blit(label, (x, y))

    def _draw_trail(self, surf):
        if len(self.sim.trail) < 2:
            return
        pts = [self.w2s(x, y) for (x, y) in self.sim.trail]
        pygame.draw.aalines(surf, TRAIL_COLOR, False, pts)

    # ── Plot scope ──
    def _draw_scope(self, surf, rect: pygame.Rect, scope: ScopeBuffer,
                    title: str, color_cmd, color_meas):
        # Background panel.
        pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(surf, (60, 62, 70), rect, 1, border_radius=6)

        # Title
        ts = self.font.render(title, True, TEXT_COLOR)
        surf.blit(ts, (rect.x + 10, rect.y + 6))
        legend_x = rect.x + rect.w - 220
        ly = rect.y + 6
        pygame.draw.line(surf, color_cmd,
                         (legend_x, ly + 6), (legend_x + 22, ly + 6), 2)
        surf.blit(self.font_sm.render("commanded", True, TEXT_DIM),
                  (legend_x + 28, ly))
        pygame.draw.line(surf, color_meas,
                         (legend_x + 120, ly + 6),
                         (legend_x + 142, ly + 6), 2)
        surf.blit(self.font_sm.render("measured", True, TEXT_DIM),
                  (legend_x + 148, ly))

        # Plot area
        pad_l, pad_r, pad_t, pad_b = 50, 14, 28, 22
        px0 = rect.x + pad_l
        py0 = rect.y + pad_t
        pw  = rect.w - pad_l - pad_r
        ph  = rect.h - pad_t - pad_b

        # Y range — full MPH range for clarity.
        y_min_mph = -5.5
        y_max_mph = 5.5
        yspan = y_max_mph - y_min_mph

        def y2pix_scalar(v_mph: float) -> int:
            t = (v_mph - y_min_mph) / yspan
            return int(py0 + (1.0 - t) * ph)

        # Grid + axis labels
        for v in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5):
            ypx = y2pix_scalar(v)
            col = ZERO_LINE if v == 0 else GRID_LINE
            pygame.draw.line(surf, col, (px0, ypx), (px0 + pw, ypx), 1)
            lab = self.font_sm.render(f"{v:+d}" if v != 0 else " 0",
                                      True, TEXT_DIM)
            surf.blit(lab, (px0 - 32, ypx - 7))
        # X axis label
        xlab = self.font_sm.render(
            f"last {HISTORY_SECS:.0f} s   (MPH)", True, TEXT_DIM)
        surf.blit(xlab, (px0 + pw - 110, py0 + ph + 4))

        # Vectorised polyline construction.
        cmd, meas = scope.ordered()
        cmd_mph = cmd / MPS_PER_MPH
        meas_mph = meas / MPS_PER_MPH
        n = scope.n
        xs = (px0 + (np.arange(n) / max(1, n - 1)) * pw).astype(np.int32)
        cmd_ys = (py0 + (1.0 - (cmd_mph - y_min_mph) / yspan) * ph
                  ).clip(py0, py0 + ph).astype(np.int32)
        meas_ys = (py0 + (1.0 - (meas_mph - y_min_mph) / yspan) * ph
                   ).clip(py0, py0 + ph).astype(np.int32)
        cmd_pts = list(zip(xs.tolist(), cmd_ys.tolist()))
        meas_pts = list(zip(xs.tolist(), meas_ys.tolist()))
        # Step-look for cmd (square wave): sample-to-sample connections
        # already render as the square edges between held values because
        # cmd is logged at the controller rate (50 Hz).
        if len(cmd_pts) > 1:
            pygame.draw.lines(surf, color_cmd, False, cmd_pts, 2)
        if len(meas_pts) > 1:
            pygame.draw.aalines(surf, color_meas, False, meas_pts)

        # Numeric readout (rightmost sample)
        last_cmd_mph = cmd_mph[-1]
        last_meas_mph = meas_mph[-1]
        info = (f"cmd {last_cmd_mph:+.2f}  "
                f"meas {last_meas_mph:+.2f}  "
                f"err {(last_cmd_mph - last_meas_mph):+.2f}")
        surf.blit(self.font_sm.render(info, True, TEXT_COLOR),
                  (px0 + 4, py0 + ph + 4))

    # ── HUD ──
    def _draw_hud(self, surf, rect: pygame.Rect, dt: float):
        pygame.draw.rect(surf, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(surf, (60, 62, 70), rect, 1, border_radius=6)
        s = self.sim
        x = rect.x + 14
        y = rect.y + 10

        sp_mph = s.setpoint_mph()
        title = self.font_lg.render(
            f"SP  {sp_mph:>4.1f} MPH   tick {s.tick:>2}/50",
            True, ACCENT)
        surf.blit(title, (x, y))
        y += 26
        sub = self.font_sm.render(
            "keys: 1-9,0 set speed   [/] nudge 0.1 MPH   "
            "p/o Kp   l/k Ki   m/n Kd   r reset PID",
            True, TEXT_DIM)
        surf.blit(sub, (x, y))
        y += 18
        sub2 = self.font_sm.render(
            "drive: arrow keys   space = brake (cmd 0)   "
            "v = toggle MP4 record   esc = quit", True, TEXT_DIM)
        surf.blit(sub2, (x, y))
        y += 22

        # Wheel readouts
        vL_mph = s.robot.v_left  / MPS_PER_MPH
        vR_mph = s.robot.v_right / MPS_PER_MPH
        cL_mph = s.cmd_L_mps     / MPS_PER_MPH
        cR_mph = s.cmd_R_mps     / MPS_PER_MPH
        lines = [
            f"L  cmd {cL_mph:+5.2f}  meas {vL_mph:+5.2f} MPH   "
            f"F {s.robot.F_left:+7.1f} N",
            f"R  cmd {cR_mph:+5.2f}  meas {vR_mph:+5.2f} MPH   "
            f"F {s.robot.F_right:+7.1f} N",
            f"u {s.robot.u:+5.2f} m/s   omega {s.robot.omega:+5.2f} rad/s"
            f"   heading {math.degrees(s.robot.theta):+6.1f}°",
            f"PID  Kp {s.pid_L.kp:6.1f}  Ki {s.pid_L.ki:6.1f}"
            f"  Kd {s.pid_L.kd:6.1f}",
        ]
        for line in lines:
            surf.blit(self.font.render(line, True, TEXT_COLOR), (x, y))
            y += 18

        # Render rate
        fps = self.clock.get_fps()
        fps_s = self.font_sm.render(f"render {fps:5.1f} FPS",
                                    True, TEXT_DIM)
        surf.blit(fps_s, (rect.x + rect.w - fps_s.get_width() - 12,
                          rect.y + 10))

    # ── Frame ──
    def draw(self, dt: float):
        self.screen.fill(BG)
        # Arena panel
        self.screen.blit(self._static_arena, (0, 0))
        # Trail + robot drawn on top
        self._draw_trail(self.screen)
        self._draw_robot(self.screen)

        # Plot panel
        px = PLOT_PANEL_X + PLOT_MARGIN
        py = PLOT_MARGIN
        pw = PLOT_PANEL_W - 2 * PLOT_MARGIN

        left_rect = pygame.Rect(px, py, pw, PLOT_HEIGHT)
        self._draw_scope(self.screen, left_rect, self.sim.scope_L,
                         "Left wheel velocity", CMD_COLOR, MEAS_COLOR)

        py += PLOT_HEIGHT + PLOT_MARGIN
        right_rect = pygame.Rect(px, py, pw, PLOT_HEIGHT)
        self._draw_scope(self.screen, right_rect, self.sim.scope_R,
                         "Right wheel velocity", CMD_COLOR, MEAS_COLOR)

        py += PLOT_HEIGHT + PLOT_MARGIN
        hud_rect = pygame.Rect(px, py, pw,
                               WIN_H - py - PLOT_MARGIN)
        self._draw_hud(self.screen, hud_rect, dt)

        self._draw_rec_indicator(self.screen)

        pygame.display.flip()


# ============================================================================
#                                 Main loop
# ============================================================================
KEY_TICK_MAP = {
    pygame.K_1:  5,   # 0.5 MPH
    pygame.K_2: 10,   # 1.0
    pygame.K_3: 15,   # 1.5
    pygame.K_4: 20,   # 2.0
    pygame.K_5: 25,   # 2.5
    pygame.K_6: 30,   # 3.0
    pygame.K_7: 35,   # 3.5
    pygame.K_8: 40,   # 4.0
    pygame.K_9: 45,   # 4.5
    pygame.K_0: 50,   # 5.0
}

# Symbolic key names used in bake scripts -> pygame keycodes.
KEY_NAME_MAP = {
    "up":     pygame.K_UP,
    "down":   pygame.K_DOWN,
    "left":   pygame.K_LEFT,
    "right":  pygame.K_RIGHT,
    "space":  pygame.K_SPACE,
    "0": pygame.K_0, "1": pygame.K_1, "2": pygame.K_2,
    "3": pygame.K_3, "4": pygame.K_4, "5": pygame.K_5,
    "6": pygame.K_6, "7": pygame.K_7, "8": pygame.K_8, "9": pygame.K_9,
    "[": pygame.K_LEFTBRACKET, "]": pygame.K_RIGHTBRACKET,
    "p": pygame.K_p, "o": pygame.K_o,
    "l": pygame.K_l, "k": pygame.K_k,
    "m": pygame.K_m, "n": pygame.K_n,
    "r": pygame.K_r,
}


def handle_keydown(sim: Sim, k: int) -> bool:
    """Apply a keydown to `sim`. Returns False if the key requests quit."""
    if k == pygame.K_ESCAPE:
        return False
    if k == pygame.K_UP:
        sim.up = True; sim.recompute_cmd()
    elif k == pygame.K_DOWN:
        sim.down = True; sim.recompute_cmd()
    elif k == pygame.K_LEFT:
        sim.left = True; sim.recompute_cmd()
    elif k == pygame.K_RIGHT:
        sim.right = True; sim.recompute_cmd()
    elif k == pygame.K_SPACE:
        # Brake — zero everything.
        sim.up = sim.down = sim.left = sim.right = False
        sim.cmd_L_mps = sim.cmd_R_mps = 0.0
    elif k in KEY_TICK_MAP:
        sim.tick = KEY_TICK_MAP[k]
        sim.recompute_cmd()
    elif k == pygame.K_LEFTBRACKET:
        sim.tick = max(0, sim.tick - 1)
        sim.recompute_cmd()
    elif k == pygame.K_RIGHTBRACKET:
        sim.tick = min(SPEED_TICKS_MAX, sim.tick + 1)
        sim.recompute_cmd()
    elif k == pygame.K_p:
        sim.pid_L.kp += sim.kp_step
        sim.pid_R.kp = sim.pid_L.kp
    elif k == pygame.K_o:
        sim.pid_L.kp = max(0.0, sim.pid_L.kp - sim.kp_step)
        sim.pid_R.kp = sim.pid_L.kp
    elif k == pygame.K_l:
        sim.pid_L.ki += sim.ki_step
        sim.pid_R.ki = sim.pid_L.ki
    elif k == pygame.K_k:
        sim.pid_L.ki = max(0.0, sim.pid_L.ki - sim.ki_step)
        sim.pid_R.ki = sim.pid_L.ki
    elif k == pygame.K_m:
        sim.pid_L.kd += sim.kd_step
        sim.pid_R.kd = sim.pid_L.kd
    elif k == pygame.K_n:
        sim.pid_L.kd = max(0.0, sim.pid_L.kd - sim.kd_step)
        sim.pid_R.kd = sim.pid_L.kd
    elif k == pygame.K_r:
        sim.pid_L.reset(); sim.pid_R.reset()
    elif k == pygame.K_v:
        sim.toggle_recording()
    return True


def handle_keyup(sim: Sim, k: int) -> None:
    if k == pygame.K_UP:
        sim.up = False; sim.recompute_cmd()
    elif k == pygame.K_DOWN:
        sim.down = False; sim.recompute_cmd()
    elif k == pygame.K_LEFT:
        sim.left = False; sim.recompute_cmd()
    elif k == pygame.K_RIGHT:
        sim.right = False; sim.recompute_cmd()


def run(args):
    sim = Sim(args)
    renderer = Renderer(sim)

    running = True
    last = time.perf_counter()
    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if not handle_keydown(sim, ev.key):
                        running = False
                elif ev.type == pygame.KEYUP:
                    handle_keyup(sim, ev.key)

            # Step the world.
            now = time.perf_counter()
            dt = min(now - last, 0.05)          # cap massive frame drops
            last = now
            sim.update(dt)

            renderer.draw(dt)
            sim.record_frame(renderer.screen)
            renderer.clock.tick(RENDER_FPS)
    finally:
        if sim.recording:
            sim._stop_recording()
        pygame.quit()


# ============================================================================
#                                  Bake
# ============================================================================
def parse_input_script(path: str) -> list[tuple[float, str, str]]:
    """Parse a keystroke bake script.

    Format (one event per line, whitespace separated):
        <t_seconds>  down  <key>     # arrow key held down
        <t_seconds>  up    <key>     # arrow key released
        <t_seconds>  press <key>     # one-shot key (space, digits, [/], p/o/l/k/m/n/r)
        <t_seconds>  tick  <0..50>   # set speed tick directly
        # comment lines start with '#'; blank lines ignored

    Key names: up, down, left, right, space, 0..9, [, ], p, o, l, k, m, n, r.
    Returns events sorted by time.
    """
    events: list[tuple[float, str, str]] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{lineno}: expected '<t> <verb> <arg>', got {raw!r}")
            try:
                t = float(parts[0])
            except ValueError:
                raise ValueError(f"{path}:{lineno}: bad time {parts[0]!r}")
            events.append((t, parts[1].lower(), parts[2].lower()))
    events.sort(key=lambda e: e[0])
    return events


def apply_event(sim: Sim, verb: str, arg: str) -> None:
    if verb == "tick":
        try:
            n = int(arg)
        except ValueError:
            raise ValueError(f"tick arg must be int, got {arg!r}")
        sim.tick = max(0, min(SPEED_TICKS_MAX, n))
        sim.recompute_cmd()
        return
    if verb not in ("down", "up", "press"):
        raise ValueError(f"unknown event verb {verb!r}")
    k = KEY_NAME_MAP.get(arg)
    if k is None:
        raise ValueError(f"unknown key {arg!r}")
    if verb in ("down", "press"):
        handle_keydown(sim, k)
    if verb == "up":
        handle_keyup(sim, k)


def bake(args) -> None:
    events = parse_input_script(args.bake)
    duration = args.duration
    fps = args.fps

    # Force headless pygame so no window appears during the bake.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    sim = Sim(args)
    renderer = Renderer(sim)

    n_frames = int(round(duration * fps))
    dt = 1.0 / fps

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    print(f"baking {n_frames} frames ({duration:.2f}s @ {fps} fps) "
          f"<- {args.bake}", flush=True)
    print(f"output: {out_path}", flush=True)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIN_W}x{WIN_H}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "18",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )

    ev_idx = 0
    t = 0.0
    t0 = time.perf_counter()
    try:
        for frame in range(n_frames):
            # Apply every scripted event whose time has arrived.
            while ev_idx < len(events) and events[ev_idx][0] <= t + 1e-9:
                _, verb, arg = events[ev_idx]
                ev_idx += 1
                apply_event(sim, verb, arg)

            sim.update(dt)
            renderer.draw(dt)

            # pygame.surfarray.array3d -> (W, H, 3) RGB; ffmpeg wants (H, W, 3).
            arr = pygame.surfarray.array3d(renderer.screen)
            arr = np.swapaxes(arr, 0, 1)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            proc.stdin.write(arr.tobytes())

            t += dt
            if (frame + 1) % fps == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {frame+1:4d}/{n_frames}  "
                      f"({(frame+1)/fps:5.2f}s sim, "
                      f"{elapsed:5.1f}s wall)", flush=True)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        pygame.quit()
    print(f"done in {time.perf_counter() - t0:.1f}s")


def parse_args(argv):
    p = argparse.ArgumentParser(description="SPEED Sim")
    p.add_argument("--seed", type=int, default=7,
                   help="RNG seed for friction patches + gradients")
    p.add_argument("--kp", type=float, default=KP_WHEEL_DEFAULT)
    p.add_argument("--ki", type=float, default=KI_WHEEL_DEFAULT)
    p.add_argument("--kd", type=float, default=KD_WHEEL_DEFAULT)
    p.add_argument("--bake", type=str, default=None, metavar="SCRIPT",
                   help="Bake an offscreen MP4 driven by a keystroke script "
                        "(see parse_input_script docstring for format).")
    p.add_argument("--out", type=str, default=None, metavar="MP4",
                   help="Bake output path "
                        "(default: <repo>/SPEED Sim/bakes/speed_sim_<script>.mp4).")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Bake duration in seconds (default 20).")
    p.add_argument("--fps", type=int, default=RENDER_FPS,
                   help="Bake frame rate (default matches RENDER_FPS).")
    return p.parse_args(argv)


def _default_out_path(script_path: str) -> str:
    base = os.path.splitext(os.path.basename(script_path))[0]
    bakes_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bakes")
    return os.path.join(bakes_dir, f"speed_sim_{base}.mp4")


def main():
    args = parse_args(sys.argv[1:])
    if args.bake:
        if args.out is None:
            args.out = _default_out_path(args.bake)
        bake(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
