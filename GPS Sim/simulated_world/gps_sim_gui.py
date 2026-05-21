#!/usr/bin/env python3
"""
GPS-without-magnetometer waypoint simulator.

The robot has no magnetometer, so the rotation between its local odom
frame and the geographic frame is unknown. It estimates that rotation on
the fly by comparing GPS-derived world displacement to perfect-odom
displacement; the more it travels, the better the fit.

What each step does:
  1. Read GPS (true position + Gaussian noise).
  2. Refit heading_offset_est by circular-mean over GPS-vs-odom deltas.
  3. Re-plan A* in world frame from the latest GPS estimate to the
     candidate goal — the rotated projection of the GPS goal that
     reflects the robot's current heading-estimate error. As the
     estimate refines, the candidate goal converges to the true GPS
     goal. Smart-padded windowed A* — the search box hugs the
     start-goal corridor, doubling its pad if no path is found.
  4. Compute the desired direction toward the next world waypoint and
     translate it to robot/odom frame using the current heading estimate.
  5. Apply a P-controlled thrust force in odom toward the desired
     velocity, capped at MAX_THRUST. Viscous friction limits terminal
     speed to 5 mph. Integrate odom; rotate the resulting odom delta
     by the unknown TRUE heading to get the world delta. Feed both into
     a 3-state EKF (x_world, y_world, θ_offset) — its θ converges as
     the GPS-vs-odom directional disagreement is observed over distance.
     During GPS dropouts the EKF coasts on prediction; outliers get
     rejected via Mahalanobis gating.

Visualization (matplotlib + Qt5):
  - Map: 500 ft (152.4 m) square centered on 37.23027 N, 80.42504 W,
    rendered in world frame (east = +x, north = +y).
  - True robot pose (red dot) and trail (red line).
  - True GPS goal (green star) with 1 m success circle.
  - "Intermediate goal" (yellow X): the world position where the robot
    will actually arrive if its current odom-frame plan is executed
    given its current heading estimate. Equivalent to the true goal
    rotated by (true_heading - heading_offset_est) around the robot's
    current GPS estimate. Converges to the true goal as the estimate
    refines.
  - Recent GPS samples (light-blue scatter) — shows the noise level.
  - A* path (yellow line) — replanned every step; smart padding window
    drawn faintly around the corridor.
"""

import sys
import math
import heapq
import argparse
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass  # fall back to whatever backend is active (e.g. headless Agg)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Patch
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.ndimage import distance_transform_edt


# ── Three-waypoint mission (chained + cached) ─────────────────────
# Canonical 3 GPS waypoints from the deployed
# ``isaac_ros-dev/config/stored_waypoints.txt`` (the actual surveyed
# competition fixture). When ``--mission three-waypoint`` is passed,
# the sim runs these three GPS targets in order while preserving EKF
# state (the "preemptive next-goal cache") across leg switches —
# mirroring ``gps_handler_node.py`` on branch
# ``origin/improve/gps-waypoint-continuity``. Each tuple is
# (latitude_deg, longitude_deg).
THREE_WAYPOINT_MISSION = [
    (37.23027, -80.42504),
    (37.23013, -80.42524),
    (37.22999, -80.42507),
]


# ── Map geometry ──────────────────────────────────────────────────
LAT_CENTER = 37.23027
LON_CENTER = -80.42504
MAP_FT     = 500.0
MAP_M      = MAP_FT * 0.3048      # 152.4 m
MAP_HALF   = MAP_M / 2.0          # 76.2 m
RES        = 0.5                  # 0.5 m grid → 305x305 cells

EARTH_R    = 6_371_000.0          # mean Earth radius (m); matches
                                  # gps_conversions.py EARTH_R_M on the
                                  # deployed robot side so the sim's
                                  # lat/lon ↔ local conversion is
                                  # byte-equivalent to production.

# ── Robot / planner constants ────────────────────────────────────
ROBOT_RADIUS     = 0.30
INFLATION_RADIUS = 0.7
COST_SCALING     = 5.0
GOAL_RADIUS      = 1.0            # competition success circle (m)
# Robot-strict success radius: matches gps_handler_node
# SUCCESS_RADIUS_M = 0.25 m. When ``ROBOT_STRICT_ARRIVAL = True``
# the sim's ``arrived`` fires only when the robot is within this
# tight ring — the IGVC 50 %-footprint rule is ignored. Set by
# ``--real`` to mirror field behavior: with the encoder-yaw bias
# pulling the candidate goal away from truth, the agent never
# crosses the 0.25 m threshold and the run reproduces the
# observed "robot never converges onto the real GPS goal".
ROBOT_SUCCESS_RADIUS_M = 0.25
ROBOT_STRICT_ARRIVAL   = False

# Predicted-convergence early termination.
# Three terminal bins, evaluated per-agent:
#   * arrived            — sim verdict: real GPS inside the
#                          GOAL_RADIUS circle around the waypoint.
#   * predicted_success  — candidate goal stayed within
#                          PREDICT_RADIUS_M of the real goal for
#                          PREDICT_HOLD_TICKS consecutive ticks.
#                          The θ fit has converged, so the
#                          controller will inevitably close the
#                          remaining distance.
#   * predicted_failure  — candidate stayed *more than*
#                          PREDICT_FAIL_RADIUS_M from the real
#                          goal for PREDICT_FAIL_HOLD_TICKS
#                          consecutive ticks while bootstrap was
#                          done — the θ fit has settled at a
#                          wrong value and the agent will never
#                          converge. Also fires if bootstrap
#                          itself never completes within
#                          PREDICT_BOOT_TIMEOUT_TICKS.
# Once a flag is set ``step()`` short-circuits, so the headless
# loop terminates as soon as every agent is in one of the three
# bins. Within ~50 s of sim time the heuristic should classify
# every reasonable agent.
PREDICT_SNAPSHOT_TICKS       = 300  # take the snapshot at 30 s
                                    # sim time, matching the user's
                                    # "after 300 steps" criterion.
PREDICT_CAND_RING_M          = 2.0  # candidate's distance to the
                                    # actual goal (yellow-X to
                                    # green-star, both as plotted on
                                    # the world map).
PREDICT_TRUE_RING_M          = 5.0  # the sim's own GPS-truth
                                    # distance to the goal — the
                                    # second leg of the comparison.
                                    # If both are inside their
                                    # rings at the snapshot tick,
                                    # the agent has converged or
                                    # will converge with a short
                                    # additional drive.

# GPS-heading EKF master switch. When False, the magnetometer-
# less θ EKF is bypassed entirely: bootstrap never graduates,
# ekf.theta stays at 0, and every heading-resync / periodic-refit
# / divergence detector is a no-op. The agent treats the world-
# frame goal as if it were already in odom frame and drives to it
# by raw odometry. Combined with the encoder-yaw bias, this
# faithfully reproduces the field test's "EKF might not have
# been running" / "robot consistently ends up opposite the goal"
# behavior — the agent drives in odom toward (goal_x, goal_y),
# but its true world motion follows ``true_heading + encoder
# drift``, depositing it predictably-far from the real goal.
GPS_HEADING_EKF_ENABLE = True

# ── Physics ──────────────────────────────────────────────────────
# Terminal speed (mph). The robot accelerates from rest, fights
# viscous friction, and asymptotes to v_max = MAX_THRUST / DAMPING.
# Kept low (~5 mph) because A* + a P-controller don't model momentum
# well at high speeds and the agent overshoots into obstacles. To make
# the GUI feel responsive at this slower speed, the GUI runs
# STEPS_PER_FRAME physics ticks per timer fire — i.e. we advance more
# *sim* time per *wallclock* second instead of cranking the agent
# velocity. Headless mode is unaffected (it always runs flat-out).
MAX_SPEED_MPH = 5.0
MAX_SPEED_MPS = MAX_SPEED_MPH * 0.44704      # ≈ 2.2352 m/s
ROBOT_MASS    = 1.0                          # kg (abstract units)
LINEAR_DAMPING = 1.0                         # F_friction = -DAMPING * v
MAX_THRUST    = MAX_SPEED_MPS * LINEAR_DAMPING  # terminal thrust = damping*v
SIM_DT        = 0.10                         # physics tick (10 Hz)

# ── Chaplygin sleigh dynamics ────────────────────────────────────
# The agent is a non-holonomic body. It can apply only two scalar
# controls each tick:
#   • F  — forward force along its body's knife-edge (heading axis).
#   • M  — moment about its center.
# This means it can't side-step or instantly turn around — it has to
# rotate its body first, then drive. EKF / bootstrap / heading-resync
# all still work because they observe odom-vs-GPS *displacement*; the
# constraint just means the odom velocity vector is always parallel
# to the body heading.
ROBOT_INERTIA       = 0.5    # kg·m² — moment of inertia about center
ANGULAR_DAMPING     = 0.5    # rotational viscous friction
MAX_ANGULAR_VEL     = 1.5    # rad/s ≈ 86°/s (180° in ~1.2 s)
MAX_MOMENT          = MAX_ANGULAR_VEL * ANGULAR_DAMPING  # terminal M
MOMENT_KP           = 4.0    # P-gain on angular-velocity error
HEADING_ERR_KP      = 2.0    # P-gain on heading-error → ω_des

# ── Wheel-encoder bias (real-robot parity) ───────────────────────
# Bowser's left encoder over-samples by ~1.6335 % — measured by
# manual calibration in fix/odometry-issues / commit 6485b9f8.
# The fix (multiply left_displacement by 1/1.016335) is NOT
# currently running on the robot, so wheel_odom_pub.cpp publishes
# odom that drifts.
#
# Differential-drive kinematics with WHEEL_BASE_M = 0.6858 m:
#     reported ω = (v_R - 1.016335·v_L) / wheel_base
#                = ((1 - 1.016335) / wheel_base) · v_forward
#                ≈ -0.02382 rad/m  ≈ -1.36 °/m
# Per meter of forward motion the reported body heading drifts
# CW by ~1.36° while the body actually moves straight. Over a
# 50 m drive that's a 68° lag — the regime where the EKF's
# closed-form θ-fit settles tens of degrees off truth and the
# candidate goal lands tens of meters from the GPS waypoint.
# This is the missing source of the ~40 m field-test offset.
WHEEL_BASE_M                 = 0.6858    # match wheel_odom_pub.cpp
LEFT_ENCODER_OVERCOUNT_RATIO = 1.016335  # = 1.0 / left_encoder_scale_
ODOM_YAW_BIAS_RAD_PER_M = (
    (1.0 - LEFT_ENCODER_OVERCOUNT_RATIO) / WHEEL_BASE_M
)
# Default off at module load — flipped to True by
# ``_apply_real_overrides()`` for the field-parity ``--real``
# scenario. Leaving it on by default contaminates the default
# scripted scenario and ``--crazy``, which expect clean wheel
# odom and a working GPS-heading EKF.
ODOM_YAW_BIAS_ENABLE         = False

# ── LIDAR onboard IMU (real-robot parity) ────────────────────────
# The SICK multiScan publishes a 3-axis gyroscope on
# ``/multiScan/imu`` at ~100 Hz. ``ekf_local.yaml`` configures
# ``imu0_config`` to fuse vroll/vpitch/vyaw — and because the
# IMU's yaw-rate variance is much smaller than the wheel-encoder
# yaw-rate variance (which is dominated by the calibrated
# encoder bias above), robot_localization's posterior yaw rate
# is essentially the IMU's. By the time ``/local_ekf/odom``
# emerges, the encoder yaw bias has been fused away.
#
# The sim reproduces this two-stage architecture: the LIDAR IMU
# gyro reading is generated per tick (= true ω + Gaussian noise),
# and an in-sim ``_local_ekf_yaw_fusion`` step yields a corrected
# body_heading that the gps_handler EKF predict consumes. Without
# this stage, the sim was feeding RAW biased wheel odom into
# gps_handler and losing yaw alignment — the orbit signature.
#
# σ_gyro = 0.01 rad/s ≈ 0.6 °/s noise — typical for a MEMS-grade
# rate sensor. Set to 0 for a "perfect IMU" sanity test.
LIDAR_IMU_NOISE_STD_RAD_PER_S = 0.01
# When True, body_heading integrates IMU-corrected ω (= true ω
# + Gaussian noise) rather than encoder-biased ω. Models the
# effect of robot_localization's wheel/IMU fusion. Default False
# at module load; ``_apply_real_overrides()`` flips it True so
# ``--real`` matches the deployed stack.
LIDAR_IMU_FUSION_ENABLE      = False

# How many physics ticks the GUI runs per timer fire. The render
# timer fires at 30 Hz (GUI_FPS below), so each fire steps one
# physics tick — gives smooth ~30 FPS motion at the same sim-time
# rate the old 10-FPS / 3-step combo produced (3 sim-s / 1 wall-s).
STEPS_PER_FRAME = 1
GUI_FPS = 30
GUI_FRAME_DT_MS = int(1000 / GUI_FPS)
LOOKAHEAD     = 1.5                          # path lookahead (m)
SPEED_GAIN    = 1.0                          # slows near goal (1/s)
MIN_SEARCH_SPEED = 0.4                       # m/s — keep moving even
                                              # when EKF says we're at
                                              # the goal but truth-side
                                              # arrival hasn't fired yet
                                              # (handles projector bias
                                              # near the goal).
THRUST_KP     = 4.0                          # P-gain on velocity error

# ── GPS sensor model — u-blox ZED-F9P ────────────────────────────
# Design rate is 10 Hz: matches both the publisher's 100 ms timer in
# isaac_ros-dev/src/gps_handler/src/gps_publisher.cpp and the URDF's
# `<update_rate>10</update_rate>` in
# isaac_ros-dev/src/sim/description/custom_robot/gps.xacro.
# (The recorded log at AutoNav-GUI-Standalone/example-playback-csv/
# t000_20260427_185211 shows ~0.45 Hz observed — a publisher quirk:
# its timer reads one NMEA line per 100 ms tick, but only $GNGGA /
# $GPGGA sentences are forwarded to /gps_fix, so most timer reads see
# a non-GGA sentence and the effective downstream rate is throttled.
# Other noise parameters below are calibrated to that real log:
# stationary jitter < 10 cm, one ~10 m discontinuity per 200 s.)
GPS_SAMPLE_HZ = 10.0
GPS_PERIOD    = 1.0 / GPS_SAMPLE_HZ          # 0.1 s
GPS_NOISE_STD = 0.30                         # white noise σ (m)
# Slow correlated drift (ionosphere / multipath). Modelled as an
# elliptical sinusoid with random phase and per-session axis.
GPS_BIAS_AMPL_M    = 0.5
GPS_BIAS_PERIOD_S  = 60.0
# Per-sample chance of a multi-metre discontinuity. Calibrated so the
# expected outlier rate in seconds is ~0.005/s (one ~10 m jump per
# 200 s, matching the real log) regardless of sample rate.
GPS_OUTLIER_PROB = 0.005 * GPS_PERIOD
GPS_OUTLIER_STD  = 6.0
# Disconnect / signal loss. None observed in the real log; keep rare.
GPS_DROPOUT_HZ_PER_S    = 0.01               # 1 dropout / 100 s
GPS_DROPOUT_DURATION_S  = (1.0, 4.0)         # multi-fix outage

# Roofs: axis-aligned squares the robot can drive under. While the
# robot's body is inside the rectangle's footprint, the GPS antenna's
# sky view is blocked → no fix. They're NOT obstacles for navigation.
ROOF_SIZE_RANGE_M = (6.0, 14.0)

# Projectors: triangular obstacles (e.g. corners of buildings) that
# bounce satellite signals from the wrong direction, biasing the GPS
# reading by a couple of metres while the robot is nearby. They DO
# block A*. Real buildings behave like this near windows / facades.
PROJECTOR_SIDE_RANGE_M       = (3.0, 5.5)
PROJECTOR_BIAS_RANGE_M       = (1.5, 3.0)    # |b|, m
PROJECTOR_INFLUENCE_RADIUS_M = 7.0           # bias tapers linearly to 0

# Roof "blackout leak": with these at zero a roof always cuts the fix
# (default behaviour). Raised by --crazy so the receiver occasionally
# returns a heavily-skewed reflected reading while the antenna is
# shadowed — urban-canyon failure mode where multipath wins over the
# blocked direct signal.
ROOF_BLACKOUT_LEAK_PROB = 0.0
ROOF_BLACKOUT_SKEW_M    = 0.0

# Blackout-zone reconnect: when an agent first enters a roof blackout
# it can fire a one-shot "reconnect" — fresh, clean GPS fixes for
# RECONNECT_DURATION_S regardless of being shadowed. After that, the
# blackout reasserts and the receiver has to wait RECONNECT_COOLDOWN_S
# before another reconnect is allowed. Models a real-world receiver
# that briefly re-acquires through a window, plus an external
# correction service that is rate-limited.
GPS_RECONNECT_DURATION_S  = 5.0
GPS_RECONNECT_COOLDOWN_S  = 20.0

# What variance the EKF *expects* on a GPS measurement. Larger than the
# white-noise σ above so the filter implicitly absorbs the unmodeled
# slow bias drift — without this, the filter locks down on early
# samples and Mahalanobis-gates most of what comes afterwards.
EKF_GPS_SIGMA = 1.2                          # m — bigger than the
                                              # white-noise σ to absorb
                                              # projector bias and slow
                                              # multipath drift
EKF_GATE_CHI2 = 50.0                         # outlier gate, χ²(2) tail
# After this many consecutive Mahalanobis rejections, force-accept the
# next GPS reading and re-inflate position covariance. Catches the
# EKF "lock-in" failure mode where a confident-but-wrong filter gates
# out every correcting fix. 25 rejections @ 10 Hz = 2.5 s of bad gating
# before recovery — short enough to fix lost agents, long enough that
# normal GPS noise doesn't trip it.
EKF_REJ_STREAK_RESET = 25

# Sliding-anchor window for the bootstrap closed-form θ fit
# (``_bootstrap_theta``). Anchoring on the OLDEST sample within the
# trailing N entries — instead of the very first sample of the whole
# history — minimises encoder-yaw-drift contamination of the fit.
# At 10 Hz GPS, 100 samples ≈ 10 s of history. The window must be
# large enough to contain ≥ ``BOOTSTRAP_BASELINE_M = 5 m`` of motion
# at the agent's nominal speed (5 mph ≈ 2.2 m/s → 100 samples covers
# ~22 m), so the fit always has plenty of baseline available; the
# trim only kicks in once history has grown past N entries.
BOOTSTRAP_WINDOW = 100

# D-spread gate for adopting the joint-fit K (encoder yaw bias rate)
# in ``_joint_theta_K_fit``. The slope of the (theta_i − theta_avg)
# vs (D_mid − ⟨D_mid⟩) regression has variance σ² / Σ(x − x̄)², so
# K-estimate quality scales directly with the cumulative-forward-
# distance span of the samples in the fit window.
#
# Without this gate, bootstrap fires after ~5 m of motion and the
# slope estimate is so noisy that the EKF's R(-K·D) per-tick
# derotation compounds error every meter — the experiment regressed
# 1000-agent --real convergence from 70.4 % to 15.0 %.
#
# 15 m is empirically large enough that K_est tracks calibrated
# truth (-0.024 rad/m for the deployed encoder bias) within ~10 %
# noise — the single-agent trace at heading=105° hit -0.027 rad/m
# at D=25 m and converged in 400 steps.
BOOTSTRAP_K_MIN_D_SPREAD_M = 15.0

# Magnitude clamp on the joint-fit's K_est. Twice the calibrated
# encoder bias rate (-0.0238 rad/m) is a defensive ceiling — any
# slope estimate above this is fit-noise from a small-sample
# regime that the D-spread gate didn't catch (e.g., first valid
# fit on a curving trajectory). Hard-clamping here prevents
# unphysical K from corrupting the EKF predict step.
BOOTSTRAP_K_MAX_RAD_PER_M = 0.05

# EMA damping factor for K adoption at the consumer side. Each
# resync site runs ``self.K_est = (1−α)·K_old + α·K_fit``, so a
# single noisy fit can shift K_est by at most α·(noise span).
# α = 0.2 gives a ~5-fit time constant — fast enough that K_est
# tracks slow drifts in the encoder bias (it can change with
# wheel wear / load), slow enough to dampen single-sample noise.
BOOTSTRAP_K_EMA_ALPHA = 0.2

# Floor on the EKF's position variance — the filter can never claim
# tighter than this. Real GPS has ~m-scale irreducible bias from
# multipath / atmospheric variation, so claiming sub-decimetre certainty
# is fiction and causes a Class-B failure mode where the controller
# stops at the (biased) EKF goal while truth is still ~1 m short of
# the success ring. Floored variance keeps GPS pulling the filter
# toward truth at a useful rate even after long convergence runs.
EKF_POS_VAR_FLOOR = 1.0 ** 2                 # σ ≥ 1.0 m

# Heading-resync (Class-A "orbit" recovery). Once bootstrap finishes
# the EKF's Kalman gain on θ shrinks to near zero, so a heading that
# was wrong at bootstrap time stays wrong forever — the agent ends up
# orbiting the goal because every world-frame command rotates by a
# fixed heading_err. We continuously re-run the closed-form GPS-vs-
# odom heading fit on recent samples and, if it disagrees with the
# EKF's θ by more than the threshold, snap θ to the closed-form value
# and re-widen θ_var so the EKF keeps refining. Cooldown gates how
# often this can fire to avoid thrashing.
HEADING_RESYNC_THRESHOLD_DEG  = 15.0   # May-2026 retune (was 10) —
                                        # matches deployed
                                        # gps_handler_node anti-chatter
HEADING_RESYNC_MIN_BASELINE_M = 2.0
HEADING_RESYNC_COOLDOWN_S     = 5.0    # May-2026 retune (was 3)
HEADING_RESYNC_WINDOW         = 100   # GPS samples (≈ 10 s @ 10 Hz)
# Reject GPS-vs-odom pairs from the heading fit when the magnitudes
# disagree by more than this factor. Spoofers pin GPS while odom
# moves (ratio → 0); jam dropouts give noisy GPS displacements with no
# real signal. Either way, those pairs would corrupt the closed-form
# heading via wildly-wrong direction angles. 3.0 keeps the filter
# permissive against ordinary noise while excluding pinned readings.
HEADING_FIT_MAGRATIO_MAX      = 3.0

# ── Additional GPS hazards ──────────────────────────────────────
# Hexagon jammers: zones where each GPS reading is independently
# suppressed with probability JAMMER_DROPOUT_PROB. Agents inside get
# *sparse* but unbiased fixes — opposite failure mode to projectors
# (which preserve fix rate but add bias). The agent's EKF coasts on
# odom prediction more than usual, accumulating heading drift if the
# agent stays in the zone too long.
JAMMER_HEX_RADIUS_RANGE_M = (5.0, 9.0)   # circumradius
JAMMER_DROPOUT_PROB       = 0.7          # 70 % of fixes suppressed inside
# Transient "off" probabilities — per GPS tick. When a hazard rolls
# off, it has no effect for that tick: an agent inside the zone gets
# an unobstructed reading. Mirrors real-life mechanisms — a window
# glance through a roof, a frequency-hopping jammer momentarily on
# the wrong band, a spoofer with variable transmit power. Crucially,
# this gives agents stuck inside compound hazard zones (e.g. roof
# overlapping foliage) periodic chances to recover.
ROOF_TRANSIENT_OFF_PROB    = 0.05        # ~5 % of ticks: window leak
JAMMER_TRANSIENT_OFF_PROB  = 0.20        # ~20 %: jammer sweep
SPOOFER_TRANSIENT_OFF_PROB = 0.30        # ~30 %: spoofer power dip

# Foliage / canopy zones: soft GPS degradation. Fixes still arrive
# but white-noise σ is multiplied — emulates fix degradation under
# tree cover or heavy weather. Different from blackout (still
# connected) and from projectors (no systematic bias, just more
# noise around truth).
FOLIAGE_RADIUS_RANGE_M   = (4.0, 9.0)
FOLIAGE_NOISE_MULT       = 4.0           # σ ×= this while inside

# Spoofers: adversarial RF that pins the receiver to a fixed lie.
# While inside SPOOFER_INFLUENCE_RADIUS_M, the GPS reading is
# replaced with the spoofer's "fake target" world coords plus the
# normal noise. Differs from projector: the pull is to a WORLD-fixed
# point, independent of the goal. Heading-resync can't rescue this
# — readings agree with each other, just at the wrong place. Real
# robots near hostile RF, or near GPS-test transmitters, see this.
SPOOFER_INFLUENCE_RADIUS_M  = 8.0
SPOOFER_FAKE_OFFSET_RANGE_M = (15.0, 35.0)   # how far the spoofer lies

# Cycle slips: rare receiver-side phase-lock loss. When fired, all
# subsequent readings get a persistent random offset for a few
# seconds, then the fix resets. Distinct from one-shot outliers
# (which decay in one tick) — cycle slips persist long enough that
# the EKF actually drifts.
CYCLE_SLIP_HZ_PER_S    = 0.0             # 0 in default mode; raised by --crazy
CYCLE_SLIP_DURATION_S  = (2.0, 5.0)
CYCLE_SLIP_OFFSET_M    = 5.0             # σ of the persistent offset

# Per-agent noise-burst windows (loose model of ionospheric / GDOP
# excursions). Bernoulli per dt to start; while active, σ multiplies
# by NOISE_BURST_MULT for the duration. Per-agent rather than truly
# global because the sim doesn't have shared world state, but the
# distinct failure mode (everyone-occasionally-having-bad-fixes) is
# preserved.
NOISE_BURST_HZ_PER_S    = 0.0            # 0 in default mode; raised by --crazy
NOISE_BURST_DURATION_S  = (3.0, 10.0)
NOISE_BURST_MULT        = 5.0

# ── A* windowing ─────────────────────────────────────────────────
# Start with a moderate pad around the start-goal corridor; double until
# success or the full map is searched. The corridor is a tube of
# half-width `pad` around the (start, goal) line segment, not the
# axis-aligned bbox: for a diagonal path this is ~ length × 2*pad
# instead of length², which is the difference between fast and
# unusable when planning for 1000 agents in parallel. Pad must be wide
# enough to route around the largest crazy-mode obstacle plus the
# robot+inflation margin, otherwise we cascade through three doublings
# before finding a path and the full-map searches dominate.
ASTAR_INITIAL_PAD = 12.0
ASTAR_MAX_PAD     = MAP_M

# Bounded histories for display.
GPS_HISTORY_LEN          = 400
INTENDED_HISTORY_LEN     = 600

# Replan throttling. With 1000 agents at 10 Hz the dominant cost is A*,
# so cut the rate aggressively: only replan when (a) we don't yet have
# a path, (b) path drift exceeds REPLAN_PATH_DRIFT_M, (c) candidate
# goal moved by more than REPLAN_GOAL_DRIFT_M, AND (d) at least
# REPLAN_MIN_INTERVAL_S has passed since the last replan. The
# underlying obstacles are static — the robot only needs to re-plan
# when its belief about *where* it's going has actually shifted.
REPLAN_PATH_DRIFT_M    = 5.0
REPLAN_GOAL_DRIFT_M    = 3.0
# NAV2's global planner typically replans at ~1 Hz — sending it
# fresh goals at the simulator's internal 10 Hz would just thrash
# its A* / Smac. The sim still computes the live candidate goal
# every tick (it costs nothing — just a rotation around ekf_pos);
# we sample it at NAV2_GOAL_HZ into `published_goal_world`, and
# *that* is what A* and the controller actually drive toward. The
# downstream rate is then a faithful match to a real NAV2
# deployment — internal belief stays high-res, commitment stays
# realistic. Override per-run with --nav2-hz.
NAV2_GOAL_HZ           = 1.0
# ── Stop-refining / heartbeat gates (real-robot parity) ───────────
# Mirrors gps_handler_node.py:
#   STOP_REFINE_K · STOP_REFINE_SIGMA_GPS_M = 0.6 m = the "we're
#     close enough — stop nudging the goal" bubble around the true
#     goal. Once the EKF position falls inside the bubble we freeze
#     `published_goal_world` so the controller / planner don't keep
#     chasing sub-σ wobble (which on the real robot triggers
#     bt_navigator replans and a visible start-stop pattern).
#   GOAL_REPUBLISH_HEARTBEAT_S — minimum interval between successive
#     /goal_pose republishes. Deployed-value parity (0.2 s on the
#     real node — keeps the action chain warm without flooding it).
#   GOAL_POSE_HEARTBEAT_S — lifecycle heartbeat: re-issue the goal
#     after this long even if nothing has changed (60 s).
STOP_REFINE_K              = 2.0
STOP_REFINE_SIGMA_GPS_M    = 0.3
GOAL_POSE_HEARTBEAT_S      = 60.0
GOAL_REPUBLISH_HEARTBEAT_S = 0.2
# During bootstrap the candidate goal can swing wildly; use a looser
# threshold so we don't burn A* on each tiny rotation, but still allow
# replan when the goal has moved by more than ~one path length. This is
# what saves agents whose first candidate landed off-map.
REPLAN_GOAL_DRIFT_M_BOOT = 15.0
REPLAN_MIN_INTERVAL_S  = 0.5

# Candidate-goal smoothing. The raw projection (rotate goal around
# ekf_pos by ε) couples the candidate to GPS-driven EKF-position
# jitter through the lever arm (|goal − ekf|), so even after the
# heading has resync'd you see the candidate wobble at multiple
# Hz. The EWMA `α` controls the small-step time constant;
# `CANDIDATE_SNAP_M` is the hard step-detect threshold above which we
# bypass the filter and adopt the new value verbatim — without this,
# the heading-resync transient (a 50–100 m candidate jump in one
# tick) would lag for several seconds. With α = 0.15 / dt = 0.1 the
# small-step time constant is ~0.6 s; with SNAP = 5 m the bootstrap
# → resync drop snaps through immediately.
CANDIDATE_SMOOTH_ALPHA = 0.15
CANDIDATE_SNAP_M       = 10.0    # May-2026 retune (was 5) — matches
                                  # deployed gps_handler_node

# Theoretical-envelope outlier filter on the candidate goal.
#
# Derivation: heading is recovered from comparing GPS-derived world
# displacement to perfect-odom displacement. After travelling `r`
# metres from spawn, a single GPS sample has angular precision
# σ_θ ≈ σ_GPS / r (longer baseline → better triangulation), so the
# residual heading error decays as 1/r. The candidate-goal distance
# from the true goal is then bounded by
#     d_env(r, L) = max(d_floor, GAIN · L / r)
# where L = current ‖robot − goal‖ and r = cumulative odom distance.
# Both quantities are known to the agent (Rule 1: odom + GPS only).
# A raw candidate that exceeds k · d_env is most likely a multipath /
# projector / outlier sample and gets rejected — the smoother holds
# the previous value instead.
#
# `MIN_R` keeps the filter dormant during bootstrap (when r is tiny
# the envelope is huge anyway, and we don't want to gate the
# legitimate snap that resyncs the heading).
CANDIDATE_ENV_GAIN_M    = 0.5    # ≈ σ_GPS lateral noise, metres
CANDIDATE_ENV_FLOOR_M   = 1.0    # noise floor (irreducible). May-2026
                                  # retune (was 0.4) — matches deployed
CANDIDATE_ENV_REJECT_K  = 4.0    # reject if d_raw > K · d_env
                                 # (was 3.0; loosened to let more
                                 # legitimate corrections through —
                                 # high reject counts on stuck agents
                                 # suggested envelope was over-gating)
CANDIDATE_ENV_MIN_R_M   = 3.0    # disable filter until r > MIN_R
CANDIDATE_ENV_ENABLE    = True   # master switch (off → no filter)

# Anti-spin / no-progress recovery.
#
# Failure mode: pure-pursuit gates forward thrust on
# `align = cos(heading_err)`. When the robot overshoots the candidate
# goal and heading_err goes past ±90°, align ≤ 0 ⇒ v_des = 0 while
# omega_des is still saturated at MAX_ANGULAR_VEL. The robot rotates
# in place; as the body sweeps through the target direction, align
# flickers slightly positive, the robot creeps forward, immediately
# overshoots again — limit-cycle spin around the candidate.
#
# Detector: track GPS-derived distance-to-goal over a sliding window.
# If progress (oldest − newest) is below `STUCK_PROGRESS_M` over a
# full `STUCK_WINDOW_S`, declare stuck and trigger recovery.
#
# Recovery: for `STUCK_RECOVERY_S` seconds, override the alignment
# gate with a forward-speed floor of `STUCK_RECOVERY_SPEED_MPS`. Also
# drop the cached A* path so a fresh plan is built against the
# (possibly newly-stable) candidate. This guarantees translation
# even if not perfectly aligned, breaking the limit cycle.
#
# Inhibitors: don't run during bootstrap (the agent legitimately
# spins/orbits while the closed-form fit converges) or once arrived
# (no point).
STUCK_WINDOW_S          = 4.0
STUCK_PROGRESS_M        = 0.4
STUCK_RECOVERY_S        = 2.0
STUCK_RECOVERY_SPEED_MPS = 0.8
STUCK_MIN_HISTORY_TICKS = 25     # need this many samples before a check
STUCK_DETECTOR_ENABLE   = True

# Moving-away detector → envelope-filter suspension (estimator-side,
# ships per Rule 7).
#
# The 1/r envelope filter assumes the EKF heading estimate is roughly
# correct — when it is, big candidate jumps are usually multipath
# outliers and rejecting them is the right call. But when the EKF is
# *biased* (bootstrap fit corrupted by a projector hit, heading-
# resync also confused, etc.) the candidate is parked at the wrong
# projection and the envelope ends up gating away the very updates
# that would correct it.
#
# Pre-EKF trip wire: if `‖GPS-position − GPS-goal‖` is monotonically
# increasing over a sliding window (the agent is moving *away* from
# the real goal), suspend the envelope filter and reset the smoother.
# The next raw candidate is then adopted verbatim, giving the EKF
# a clean chance to refetch. Computable from agent-side data only
# (own GPS + goal GPS). Independent of the EKF whose belief is what
# we're trying to validate.
MOVING_AWAY_WINDOW_S          = 3.0
MOVING_AWAY_THRESHOLD_M       = 1.0   # net delta over window; +ve = farther
MOVING_AWAY_ENV_SUSPEND_S     = 4.0
MOVING_AWAY_DETECTOR_ENABLE   = True
# Mirror gps_handler_node MOVING_AWAY_MIN_HISTORY_TICKS = 8 (was 25)
# and MOVING_AWAY_WINDOW_COVERAGE = 0.6 (was 0.8). Loosened on the
# robot so the detector can trip on a half-filled window — gives
# the pre-bootstrap safety net a chance to fire after just a few
# seconds of wrong-direction driving instead of waiting out the
# full window. Also runs pre-bootstrap (sim-side gate dropped) per
# commit 1399e958: catches "θ still at default 0, robot driving
# off into a phantom goal" before the bootstrap accumulates 5 m
# of wrong-direction baseline.
MOVING_AWAY_MIN_HISTORY_TICKS = 8
MOVING_AWAY_WINDOW_COVERAGE   = 0.6

# Heading-flip Hail Mary tested and rolled back (iter 6 regressed to
# 9959/10000 because the flip fires in too many cases where heading
# was actually mostly correct).
HEADING_FLIP_AFTER_N_EVENTS  = 5
HEADING_FLIP_VAR_DEG         = 30.0
HEADING_FLIP_ENABLE          = False

# Forced heading-resync triggered by moving-away (estimator-side,
# ships per Rule 7).
#
# Standard `_maybe_resync_heading` is cooldown-gated and requires a
# 2 m baseline over a 10 s window. Agents in a wrong-heading limit
# cycle don't accumulate that much net motion (they oscillate around
# the wrong-place candidate), so the closed-form fit returns no
# value and the EKF stays locked at the wrong θ.
#
# When moving-away fires, we *know* the heading is wrong (the agent
# is getting farther from the goal it's trying to reach). Run a
# closed-form fit on a wider window with a much lower baseline
# requirement, and accept any fit it produces — even a small
# correction is better than the locked wrong value.
HEADING_FORCE_RESYNC_WINDOW           = 500   # ≈ 50 s of GPS samples
HEADING_FORCE_RESYNC_MIN_BASELINE_M   = 3.0   # back to iter-2 value;
                                              # 1.5 m ties iter-3
HEADING_FORCE_RESYNC_DIFF_DEG         = 20.0  # only snap if new fit
                                              # disagrees with EKF θ by
                                              # this much
HEADING_FORCE_RESYNC_VAR_DEG          = 10.0  # post-snap σ_θ
HEADING_FORCE_RESYNC_ENABLE           = True

# ── GPS antenna lever-arm (real-robot parity) ────────────────────
# Bowser's URDF puts the GPS antenna (`gps_footprint`) at
# (-0.38, 0.0, 0.56) m relative to `base_link`. The receiver
# reports the antenna's WORLD position; the EKF and the controller
# track base_link's WORLD position. Without compensation, every
# /gps_fix is biased by  R(yaw_world) · antenna_offset_in_baselink
# — a heading-locked bias that locks the EKF onto a fixed wrong
# point. The fusion pipeline subtracts an estimated lever-arm using
# the EKF's own θ. Residual heading error ε leaves a residual bias
# of ~ε · |antenna_offset|, plus the closed-form heading fit itself
# is biased by the antenna's rotational swing during early motion
# (the bootstrap fit consumes raw `gi - g0` differences that
# include  R(yaw(i)) - R(yaw(0)) · offset). Both effects together
# make the candidate goal converge toward a stable wrong spot —
# exactly the field-test failure we need the sim to reproduce.
GPS_ANTENNA_OFFSET_BASELINK = (-0.38, 0.0)
GPS_LEVER_ARM_CORRECTION_ENABLE = True   # robot-side flag

# Periodic heading refit (mirrored from gps_handler_node
# PERIODIC_REFIT_PERIOD_S etc.). Every PERIODIC_REFIT_PERIOD_S,
# unconditionally refit θ from the resync window if the closed-
# form fit disagrees with the EKF by more than the threshold.
# Catches small persistent biases (5-10°) that fall below the
# moving-away (1 m / 3 s) and divergence (5 m) thresholds yet
# still cause meters of cross-track error over a long goal.
PERIODIC_REFIT_PERIOD_S        = 3.0
PERIODIC_REFIT_THRESHOLD_DEG   = 10.0
PERIODIC_REFIT_MIN_BASELINE_M  = 2.0
PERIODIC_REFIT_VAR_DEG         = 5.0
PERIODIC_REFIT_ENABLE          = True

# Local-vs-world divergence detector (companion to moving-away).
# Moving-away catches *radial* drift (raw GPS distance from the
# goal growing). It misses *tangential* drift, where raw GPS
# distance barely changes while the EKF's distance-to-goal can
# collapse rapidly because θ is wrong and predict() rotates odom
# motion into a fictitious "toward goal" direction. This detector
# compares cumulative progress in EKF-frame distance vs raw-GPS-
# frame distance since the goal was first observable; when local
# progress runs ahead of world progress by more than the threshold,
# θ is wrong → force-resync the heading. Mirrored from
# gps_handler_node._update_local_world_divergence().
LOCAL_VS_WORLD_DIVERGENCE_M         = 5.0
LOCAL_VS_WORLD_MIN_LOCAL_PROGRESS_M = 5.0
LOCAL_VS_WORLD_COOLDOWN_S           = 5.0
LOCAL_VS_WORLD_DETECTOR_ENABLE      = True


# ── Geographic ↔ local-tangent meters ────────────────────────────
def latlon_to_meters(lat, lon):
    """Local tangent plane around (LAT_CENTER, LON_CENTER).
    Returns (east_m, north_m) measured from the center.
    """
    lat0 = math.radians(LAT_CENTER)
    de = math.radians(lon - LON_CENTER) * EARTH_R * math.cos(lat0)
    dn = math.radians(lat - LAT_CENTER) * EARTH_R
    return de, dn


def meters_to_latlon(east_m, north_m):
    lat0 = math.radians(LAT_CENTER)
    lat = LAT_CENTER + math.degrees(north_m / EARTH_R)
    lon = LON_CENTER + math.degrees(east_m / (EARTH_R * math.cos(lat0)))
    return lat, lon


# ── World obstacles (random circular blobs) ─────────────────────
def gen_obstacles(rng, n=12, min_r=2.0, max_r=5.5,
                  exclude=((0.0, 0.0, 6.0),)):
    """Random circular obstacles inside the map. Each `exclude` entry
    (cx, cy, r) is a circle obstacles must not overlap (used to keep the
    start zone clear)."""
    out = []
    bound = MAP_HALF - max_r
    tries = 0
    while len(out) < n and tries < n * 80:
        tries += 1
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        r  = rng.uniform(min_r, max_r)
        bad = False
        for ex, ey, er in exclude:
            if math.hypot(cx - ex, cy - ey) < r + er + 0.5:
                bad = True
                break
        if not bad:
            for ox, oy, oR in out:
                if math.hypot(cx - ox, cy - oy) < r + oR + 1.5:
                    bad = True
                    break
        if not bad:
            out.append((cx, cy, r))
    return out


def gen_roofs(rng, n, exclude=((0.0, 0.0, 6.0),)):
    """Random axis-aligned roofs (no overlap with the start zone).
    Stored as (x_min, y_min, x_max, y_max). Roofs are allowed to sit
    on top of obstacles — they're a separate, GPS-only layer."""
    out = []
    smin, smax = ROOF_SIZE_RANGE_M
    bound = MAP_HALF - smax / 2 - 2.0
    for _ in range(n * 60):
        if len(out) >= n:
            break
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        sz = rng.uniform(smin, smax)
        x_min, y_min = cx - sz / 2, cy - sz / 2
        x_max, y_max = cx + sz / 2, cy + sz / 2
        bad = False
        for ex, ey, er in exclude:
            if (x_min < ex + er and x_max > ex - er
                    and y_min < ey + er and y_max > ey - er):
                bad = True
                break
        if not bad:
            out.append((x_min, y_min, x_max, y_max))
    return out


def gen_projectors(rng, n, obstacles=(), exclude=((0.0, 0.0, 6.0),)):
    """Random projector triangles. Each is stored as (verts, bias)
    where verts is a tuple of 3 (x, y) corners and bias is a fixed
    (bx, by) GPS-offset vector applied near the projector. Triangles
    avoid existing obstacles and the start zone."""
    smin, smax = PROJECTOR_SIDE_RANGE_M
    bmin, bmax = PROJECTOR_BIAS_RANGE_M
    bound = MAP_HALF - smax - 2.0
    out = []
    for _ in range(n * 80):
        if len(out) >= n:
            break
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        side = rng.uniform(smin, smax)
        # Equilateral-ish triangle around (cx, cy)
        rot = rng.uniform(0, 2 * math.pi)
        verts = []
        radius = side / math.sqrt(3)        # circumradius of equilateral
        for k in range(3):
            a = rot + k * (2 * math.pi / 3)
            verts.append((cx + radius * math.cos(a),
                          cy + radius * math.sin(a)))
        # Reject overlap with start zone, obstacles, or other projectors
        bad = False
        for ex, ey, er in exclude:
            if math.hypot(cx - ex, cy - ey) < radius + er + 0.5:
                bad = True; break
        if not bad:
            for ox, oy, oR in obstacles:
                if math.hypot(cx - ox, cy - oy) < radius + oR + 1.0:
                    bad = True; break
        if not bad:
            for v_other, _ in out:
                ocx = sum(v[0] for v in v_other) / 3.0
                ocy = sum(v[1] for v in v_other) / 3.0
                if math.hypot(cx - ocx, cy - ocy) < 2 * radius + 1.5:
                    bad = True; break
        if bad:
            continue
        # Random bias direction, magnitude in [bmin, bmax]
        bdir = rng.uniform(0, 2 * math.pi)
        bmag = rng.uniform(bmin, bmax)
        bias = (bmag * math.cos(bdir), bmag * math.sin(bdir))
        out.append((tuple(verts), bias))
    return out


def projector_centroid_radius(verts):
    cx = sum(v[0] for v in verts) / 3.0
    cy = sum(v[1] for v in verts) / 3.0
    r  = max(math.hypot(v[0] - cx, v[1] - cy) for v in verts)
    return cx, cy, r


def hex_vertices(cx, cy, radius, rot=0.0):
    """6 corners of a regular hexagon centered at (cx, cy)."""
    return [(cx + radius * math.cos(rot + k * math.pi / 3.0),
             cy + radius * math.sin(rot + k * math.pi / 3.0))
            for k in range(6)]


def gen_jammers(rng, n, exclude=((0.0, 0.0, 6.0),)):
    """Random hex-jammer regions. Stored as (cx, cy, radius). Like
    roofs, jammers are a GPS-only layer (the robot can drive through
    them); they don't block A*."""
    smin, smax = JAMMER_HEX_RADIUS_RANGE_M
    bound = MAP_HALF - smax - 2.0
    out = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        r  = rng.uniform(smin, smax)
        bad = False
        for ex, ey, er in exclude:
            if math.hypot(cx - ex, cy - ey) < r + er + 0.5:
                bad = True; break
        if not bad:
            for ox, oy, oR in out:
                if math.hypot(cx - ox, cy - oy) < r + oR + 1.0:
                    bad = True; break
        if not bad:
            out.append((cx, cy, r))
    return out


def gen_foliage(rng, n, exclude=((0.0, 0.0, 6.0),)):
    """Random foliage / canopy zones. Stored as (cx, cy, radius). GPS-
    only layer — agent can drive through them; effect is a per-tick
    noise multiplier while inside."""
    smin, smax = FOLIAGE_RADIUS_RANGE_M
    bound = MAP_HALF - smax - 2.0
    out = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        r  = rng.uniform(smin, smax)
        bad = False
        for ex, ey, er in exclude:
            if math.hypot(cx - ex, cy - ey) < r + er + 0.5:
                bad = True; break
        if not bad:
            out.append((cx, cy, r))
    return out


def gen_spoofers(rng, n, exclude=((0.0, 0.0, 6.0),)):
    """Random GPS spoofers. Stored as ((cx, cy), (fx, fy)) where
    (cx, cy) is the spoofer's world position and (fx, fy) is the
    fake-target it pins receivers to while they're inside its
    influence radius. Agents inside the influence radius receive GPS
    readings centered on (fx, fy) instead of their true position."""
    bound = MAP_HALF - SPOOFER_INFLUENCE_RADIUS_M - 2.0
    fmin, fmax = SPOOFER_FAKE_OFFSET_RANGE_M
    out = []
    for _ in range(n * 80):
        if len(out) >= n:
            break
        cx = rng.uniform(-bound, bound)
        cy = rng.uniform(-bound, bound)
        # Reject if too close to start zone or another spoofer.
        bad = False
        for ex, ey, er in exclude:
            if math.hypot(cx - ex, cy - ey) \
                    < SPOOFER_INFLUENCE_RADIUS_M + er + 0.5:
                bad = True; break
        if not bad:
            for (ocx, ocy), _ in out:
                if math.hypot(cx - ocx, cy - ocy) \
                        < 2 * SPOOFER_INFLUENCE_RADIUS_M + 1.0:
                    bad = True; break
        if bad:
            continue
        # Fake target: somewhere on the map, but offset by a noticeable
        # amount from the spoofer's actual position.
        for _ in range(50):
            mag = rng.uniform(fmin, fmax)
            ang = rng.uniform(0, 2 * math.pi)
            fx = cx + mag * math.cos(ang)
            fy = cy + mag * math.sin(ang)
            if abs(fx) < MAP_HALF - 1.0 and abs(fy) < MAP_HALF - 1.0:
                out.append(((cx, cy), (fx, fy)))
                break
    return out


# ── Costmap (world frame; full map) ──────────────────────────────
class Costmap:
    """Full-map binary obstacle grid + NAV2-style inflation."""

    def __init__(self, obstacles, projectors=()):
        self.res  = RES
        self.half = MAP_HALF
        self.w = int(round(2 * self.half / self.res))
        self.h = self.w
        self.obstacles = np.zeros((self.h, self.w), dtype=bool)
        for cx, cy, r in obstacles:
            self._stamp_disk(cx, cy, r)
        # Projectors are obstacles too — approximate the triangle by
        # its circumscribed disk (small triangles, this is plenty
        # tight for A* avoidance).
        for verts, _bias in projectors:
            cx, cy, r = projector_centroid_radius(verts)
            self._stamp_disk(cx, cy, r)
        self._inflate()

    def _stamp_disk(self, cx, cy, r):
        rng_cells = int(math.ceil(r / self.res)) + 1
        ix, iy = self.w2c(cx, cy)
        y0 = max(0, iy - rng_cells); y1 = min(self.h, iy + rng_cells + 1)
        x0 = max(0, ix - rng_cells); x1 = min(self.w, ix + rng_cells + 1)
        if y0 >= y1 or x0 >= x1:
            return
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        wxs = (xs + 0.5) * self.res - self.half
        wys = (ys + 0.5) * self.res - self.half
        sub = (wxs - cx) ** 2 + (wys - cy) ** 2 <= r * r
        self.obstacles[y0:y1, x0:x1] |= sub

    def _inflate(self):
        self.inflated = np.zeros_like(self.obstacles, dtype=np.uint8)
        self.inflated[self.obstacles] = 254
        if not self.obstacles.any():
            return
        dist = distance_transform_edt(~self.obstacles) * self.res
        inscribed = (dist <= ROBOT_RADIUS) & (dist > 0)
        self.inflated[inscribed] = 253
        decay = (dist > ROBOT_RADIUS) & (dist < ROBOT_RADIUS + INFLATION_RADIUS)
        if decay.any():
            costs = (252 * np.exp(
                -COST_SCALING * (dist[decay] - ROBOT_RADIUS))).astype(np.uint8)
            self.inflated[decay] = np.maximum(self.inflated[decay], costs)

    def w2c(self, wx, wy):
        cx = int(np.clip((wx + self.half) / self.res, 0, self.w - 1))
        cy = int(np.clip((wy + self.half) / self.res, 0, self.h - 1))
        return cx, cy

    def c2w(self, cx, cy):
        return (cx + 0.5) * self.res - self.half, (cy + 0.5) * self.res - self.half

    def is_lethal_world(self, wx, wy):
        cx, cy = self.w2c(wx, wy)
        return self.inflated[cy, cx] >= 254


# ── Smart-padded A* ──────────────────────────────────────────────
def _los_clear(cm, sx, sy, gx, gy):
    """True iff the straight world-frame segment (sx,sy)→(gx,gy) crosses
    no lethal inflated cells. Sampled at one step per grid cell. This
    is the "MAP padder" fast path: when the corridor is empty (which
    is the common case in --crazy with sparse-but-large features), the
    A* call is just a 100-cell lookup instead of thousands of heap
    operations."""
    inflated = cm.inflated
    res = cm.res
    half = cm.half
    w, h = cm.w, cm.h
    dx = gx - sx
    dy = gy - sy
    length = math.hypot(dx, dy)
    if length < 1e-9:
        scx = int((sx + half) / res)
        scy = int((sy + half) / res)
        if 0 <= scx < w and 0 <= scy < h:
            return inflated[scy, scx] < 254
        return False
    n = int(length / res) + 2
    inv_n = 1.0 / n
    for i in range(n + 1):
        t = i * inv_n
        x = sx + t * dx
        y = sy + t * dy
        cx = int((x + half) / res)
        cy = int((y + half) / res)
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return False
        if inflated[cy, cx] >= 254:
            return False
    return True


def _nearest_free_cell(cm, cx, cy, max_r=8):
    """Outward ring search for a non-lethal cell. Used to recover when
    the plan anchor or goal lands inside an inflated obstacle. Returns
    (cx, cy) or None if no free cell within `max_r` cells."""
    inflated = cm.inflated
    w, h = cm.w, cm.h
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            # Only check the ring boundary (|dy|==r OR |dx|==r), so we
            # don't re-examine inner cells from prior rings.
            if abs(dy) == r:
                xrange_iter = range(-r, r + 1)
            else:
                xrange_iter = (-r, r)
            for dx in xrange_iter:
                nx = cx + dx
                if nx < 0 or nx >= w:
                    continue
                if inflated[ny, nx] < 254:
                    return nx, ny
    return None


def windowed_astar(cm, sx, sy, gx, gy, pad=ASTAR_INITIAL_PAD):
    """A* on a window around the (start, goal) corridor.

    The window is the axis-aligned bounding box of (start, goal),
    enlarged by `pad` meters on every side. If A* finds no path, double
    the pad and retry until either a path is found or the pad covers
    the entire map.

    Returns (path, used_pad, window_xyxy_world). `path` is a list of
    (x, y) world coords or None. `window_xyxy_world` is the final search
    box for visualization."""
    # MAP-padder fast path: if the straight LOS is clear, skip A*
    # entirely. The vast majority of (start, goal) pairs in --crazy have
    # an obstacle-free corridor — for those, we trade ~10k heap ops for
    # ~200 grid lookups.
    if _los_clear(cm, sx, sy, gx, gy):
        win = (min(sx, gx) - pad, min(sy, gy) - pad,
               max(sx, gx) + pad, max(sy, gy) + pad)
        return [(sx, sy), (gx, gy)], pad, win
    while True:
        x_min = max(-cm.half, min(sx, gx) - pad)
        x_max = min( cm.half, max(sx, gx) + pad)
        y_min = max(-cm.half, min(sy, gy) - pad)
        y_max = min( cm.half, max(sy, gy) + pad)
        path = _astar_in_window(cm, sx, sy, gx, gy,
                                x_min, x_max, y_min, y_max,
                                pad=pad)
        win = (x_min, y_min, x_max, y_max)
        if path is not None:
            return path, pad, win
        if pad >= ASTAR_MAX_PAD:
            return None, pad, win
        pad = min(pad * 2, ASTAR_MAX_PAD)


def _astar_in_window(cm, sx, sy, gx, gy, x_min, x_max, y_min, y_max,
                     pad=None):
    cx_min, cy_min = cm.w2c(x_min, y_min)
    cx_max, cy_max = cm.w2c(x_max, y_max)
    scx, scy = cm.w2c(sx, sy)
    gcx, gcy = cm.w2c(gx, gy)
    inflated = cm.inflated
    res = cm.res
    half = cm.half

    # Corridor (line-of-sight tube) check. A cell is in-corridor if its
    # perpendicular distance from the start→goal segment is ≤ pad. This
    # is the MAP-padder idea: only expand cells that could plausibly be
    # on the way, not the whole bbox. For an axis-aligned (sx==gx or
    # sy==gy) path the corridor and bbox coincide; for diagonals it's a
    # large win.
    if pad is None or pad >= ASTAR_MAX_PAD - 1e-6:
        # Full-map fallback: cheap to skip the corridor test entirely.
        in_corridor = None
    else:
        seg_dx = gx - sx
        seg_dy = gy - sy
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        pad2 = pad * pad
        if seg_len2 < 1e-9:
            in_corridor = None
        else:
            inv_len2 = 1.0 / seg_len2
            # Closures fight Python's attribute lookup overhead; bind
            # locals up front.
            _sx, _sy = sx, sy
            _dx, _dy = seg_dx, seg_dy

            def in_corridor(cx, cy):
                wx = (cx + 0.5) * res - half
                wy = (cy + 0.5) * res - half
                vx = wx - _sx
                vy = wy - _sy
                t = (vx * _dx + vy * _dy) * inv_len2
                if t < 0.0:
                    px, py = _sx, _sy
                elif t > 1.0:
                    px, py = _sx + _dx, _sy + _dy
                else:
                    px = _sx + t * _dx
                    py = _sy + t * _dy
                ddx = wx - px
                ddy = wy - py
                return (ddx * ddx + ddy * ddy) <= pad2

    def in_win(cx, cy):
        return cx_min <= cx <= cx_max and cy_min <= cy <= cy_max

    def lethal(cx, cy):
        if cx < 0 or cx >= cm.w or cy < 0 or cy >= cm.h:
            return True
        return inflated[cy, cx] >= 254

    if not in_win(scx, scy) or not in_win(gcx, gcy):
        return None
    # If the start or goal lands in a lethal cell (EKF noise can drift
    # the plan anchor into an inflated obstacle, especially in --crazy),
    # snap to the nearest non-lethal cell within a short spiral. Without
    # this, A* returns None forever and the agent silently freezes.
    if lethal(scx, scy):
        snapped = _nearest_free_cell(cm, scx, scy, max_r=8)
        if snapped is None:
            return None
        scx, scy = snapped
    if lethal(gcx, gcy):
        snapped = _nearest_free_cell(cm, gcx, gcy, max_r=8)
        if snapped is None:
            return None
        gcx, gcy = snapped

    open_set = [(0.0, scy, scx)]
    came = {}
    gs = {(scy, scx): 0.0}
    closed = set()
    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]
    base = [1.4142, 1.0, 1.4142, 1.0, 1.0, 1.4142, 1.0, 1.4142]
    while open_set:
        _, cy, cx = heapq.heappop(open_set)
        if (cy, cx) in closed:
            continue
        closed.add((cy, cx))
        if cy == gcy and cx == gcx:
            path = []
            p = (cy, cx)
            while p in came:
                path.append(cm.c2w(p[1], p[0]))
                p = came[p]
            path.append(cm.c2w(p[1], p[0]))
            return path[::-1]
        for (dy, dx), c in zip(nbrs, base):
            ny, nx = cy + dy, cx + dx
            if (ny, nx) in closed:
                continue
            if not in_win(nx, ny):
                continue
            if lethal(nx, ny):
                continue
            if in_corridor is not None and not in_corridor(nx, ny):
                continue
            if dy != 0 and dx != 0:
                if lethal(cx + dx, cy) or lethal(cx, cy + dy):
                    continue
            v = inflated[ny, nx]
            extra = 50.0 if v >= 253 else (v / 50.0 if v > 0 else 0.0)
            ng = gs[(cy, cx)] + c + extra
            if (ny, nx) not in gs or ng < gs[(ny, nx)]:
                gs[(ny, nx)] = ng
                # Weighted A* (h_weight > 1) — gives up optimality for
                # speed. The robot doesn't care about a few extra metres
                # of path; it cares about the planner finishing before
                # the next physics tick, especially with 1000 agents
                # contending for CPU time.
                h = 1.6 * math.hypot(ny - gcy, nx - gcx)
                heapq.heappush(open_set, (ng + h, ny, nx))
                came[(ny, nx)] = (cy, cx)
    return None


# ── 3-state EKF: fuse perfect odom with noisy GPS ────────────────
class GPSEKF:
    """Extended Kalman filter on state x = [x_world, y_world, theta].

    Prediction:
        Robot reports an odom-frame delta (Δx_o, Δy_o) per dt. Because
        the rotation between odom and world is the unknown θ, the world
        delta is R(θ) (Δx_o, Δy_o). θ itself is treated as constant.

    Update:
        GPS provides direct measurements of (x_world, y_world). Outliers
        and multipath spikes get rejected by Mahalanobis gating; signal
        loss is handled by simply not calling update() — the EKF coasts
        on its prediction (which is exact in our model because odometry
        is perfect).

    Why this works for a magnetometer-less robot:
        θ is observable through the joint dynamics — when the GPS shows
        the robot moving in a direction that disagrees with the
        odom-frame direction, the only state that explains it is θ.
        The off-diagonal covariance entries grow naturally as the robot
        moves, so a single GPS update propagates information into θ.
    """

    def __init__(self, x0, y0,
                 q_pos=1e-3, q_theta=1e-4,
                 r_gps=EKF_GPS_SIGMA, theta_var0=None):
        self.x = np.array([x0, y0, 0.0], dtype=float)
        if theta_var0 is None:
            theta_var0 = (math.pi) ** 2          # full ±π uncertainty
        self.P = np.diag([r_gps ** 2, r_gps ** 2, theta_var0]).astype(float)
        # Process-noise rate (per unit dt). Small on position (odom is
        # perfect in this sim, but we leave headroom for unmodeled bias
        # drift), small on theta to keep the filter from locking out
        # late corrections.
        self._Q = np.diag([q_pos ** 2, q_pos ** 2, q_theta ** 2])
        self._R = np.diag([r_gps ** 2, r_gps ** 2])
        self._I = np.eye(3)
        self._r_gps = float(r_gps)

        # Diagnostics for the status panel.
        self.last_innovation = (0.0, 0.0)
        self.last_mahalanobis = 0.0
        self.rejected_count = 0
        self.update_count = 0
        # Mirror gps_ekf.GpsEkf: count of consecutive update()
        # rejections. Cleared on every accepted update; consumed
        # by the step() loop to fire force_accept_next() once it
        # crosses EKF_REJ_STREAK_RESET.
        self.consecutive_rejects = 0

    def predict(self, dxo, dyo, dt):
        c = math.cos(self.x[2]); s = math.sin(self.x[2])
        self.x[0] += c * dxo - s * dyo
        self.x[1] += s * dxo + c * dyo
        # F = ∂f/∂x
        F = np.eye(3)
        F[0, 2] = -s * dxo - c * dyo
        F[1, 2] =  c * dxo - s * dyo
        self.P = F @ self.P @ F.T + self._Q * dt
        # Position-variance floor (real-robot parity, gps_ekf.py
        # GpsEkf.predict). When clamping a diagonal entry up, scale
        # the corresponding row and column off-diagonals by
        # sqrt(new/old) so correlation coefficients ρ_ij = P[i,j] /
        # sqrt(P[i,i]·P[j,j]) are preserved and the matrix stays PSD
        # (Cauchy-Schwarz). Scalar-only flooring breaks this and can
        # make the gain on θ explode on subsequent updates.
        for i in (0, 1):
            if self.P[i, i] < EKF_POS_VAR_FLOOR:
                old_var = float(self.P[i, i])
                new_var = EKF_POS_VAR_FLOOR
                scale = math.sqrt(new_var / max(old_var, 1e-12))
                for j in range(self.P.shape[0]):
                    if j != i:
                        self.P[i, j] *= scale
                        self.P[j, i] *= scale
                self.P[i, i] = new_var

    def reset_theta(self, theta, theta_var=None):
        """Replace θ (and decorrelate it) — used to seed the filter
        from a closed-form estimate after the cold-start bootstrap."""
        self.x[2] = (float(theta) + math.pi) % (2 * math.pi) - math.pi
        if theta_var is None:
            theta_var = math.radians(20.0) ** 2
        self.P[2, :] = 0.0
        self.P[:, 2] = 0.0
        self.P[2, 2] = float(theta_var)

    def update_theta_measurement(self, theta_obs, theta_meas_std):
        """Scalar Kalman update on θ as a direct measurement.

        Real-robot parity, mirrored from gps_ekf.GpsEkf.update_theta_measurement
        (gps_ekf.py lines 198-232).

        Use this AFTER bootstrap completes, where we want successive
        observations weighed against the EKF's accumulated confidence
        rather than snap-replacing it. As P[2,2] shrinks across many
        updates the Kalman gain on the next observation also shrinks,
        so a converged θ becomes increasingly resistant to single
        noisy fits. This is what makes the candidate goal in map
        frame actually converge instead of swinging on every resync
        event.

        Measurement model: H = [0, 0, 1], R = theta_meas_std². The
        gain K = P[:, 2] / (P[2,2] + R) is a 3-vector — the
        accumulated cross-covariance entries propagate information
        from the θ observation into x and y too.
        """
        R_theta = float(theta_meas_std) ** 2
        innovation = (float(theta_obs) - self.x[2] + math.pi) \
                     % (2 * math.pi) - math.pi
        S = float(self.P[2, 2]) + R_theta
        if S <= 0.0:
            return False
        K = self.P[:, 2] / S
        self.x[0] += K[0] * innovation
        self.x[1] += K[1] * innovation
        self.x[2] = (self.x[2] + K[2] * innovation + math.pi) \
                    % (2 * math.pi) - math.pi
        # H = [0,0,1] selects row 2: (I - K H) P  ≡  P - outer(K, P[2,:]).
        self.P = self.P - np.outer(K, self.P[2, :])
        # Re-symmetrize for numerical safety.
        self.P = 0.5 * (self.P + self.P.T)
        self.update_count += 1
        return True

    def update(self, zx, zy, gate_chi2=EKF_GATE_CHI2):
        z = np.array([zx, zy], dtype=float)
        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self._R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        m2 = float(y @ S_inv @ y)
        self.last_innovation = (float(y[0]), float(y[1]))
        self.last_mahalanobis = m2
        if m2 > gate_chi2:                      # 99.9% χ²(2) ≈ 13.8
            self.rejected_count += 1
            self.consecutive_rejects += 1
            return False
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        # Joseph form would be more numerically stable but with a 3×3
        # matrix and reasonable Q this simple form is fine.
        self.P = (self._I - K @ H) @ self.P
        # Floor the position variance HERE, immediately after the
        # Kalman gain shrinks it (real-robot parity, gps_ekf.py
        # GpsEkf.update lines 178-187). The predict-step floor still
        # runs, but update() can drive P[0,0]/P[1,1] far below
        # EKF_POS_VAR_FLOOR in a single accepted sample — and if a
        # biased GPS sample (e.g. antenna lever-arm residual) is
        # consistently accepted, the gain on subsequent samples
        # collapses and the EKF locks onto the bias before predict()
        # has a chance to re-inflate. Same off-diagonal scaling as
        # predict(), preserving correlation coefficients.
        for i in (0, 1):
            if self.P[i, i] < EKF_POS_VAR_FLOOR:
                old_var = float(self.P[i, i])
                if old_var > 1e-12:
                    scale = math.sqrt(EKF_POS_VAR_FLOOR / old_var)
                    for j in range(self.P.shape[0]):
                        if j != i:
                            self.P[i, j] *= scale
                            self.P[j, i] *= scale
                self.P[i, i] = EKF_POS_VAR_FLOOR
        # Keep θ in [-π, π]
        self.x[2] = (self.x[2] + math.pi) % (2 * math.pi) - math.pi
        self.update_count += 1
        self.consecutive_rejects = 0
        return True

    def force_accept_next(self):
        """Mirror gps_ekf.GpsEkf.force_accept_next: reinflate the
        position variance to ``r_gps²`` so the next GPS sample
        drags the estimate back toward truth even if the
        Mahalanobis gate would normally reject it. Called by the
        consumer after EKF_REJ_STREAK_RESET consecutive rejections.
        """
        self.P[0, 0] = max(self.P[0, 0], self._r_gps ** 2)
        self.P[1, 1] = max(self.P[1, 1], self._r_gps ** 2)
        self.consecutive_rejects = 0

    @property
    def pos_xy(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def theta(self):
        return float(self.x[2])

    @property
    def pos_std(self):
        return math.sqrt(max(self.P[0, 0], 0.0)), math.sqrt(max(self.P[1, 1], 0.0))

    @property
    def theta_std_rad(self):
        """Standard deviation of θ in radians. Mirror of
        gps_ekf.GpsEkf.theta_std_rad — name kept identical so a
        port across files is a literal copy-paste."""
        return math.sqrt(max(self.P[2, 2], 0.0))

    # Backwards-compat alias for any in-tree consumer that hadn't
    # been migrated yet. Identical numeric value; can be removed
    # once nothing references the un-suffixed form.
    @property
    def theta_std(self):
        return self.theta_std_rad


# ── Per-agent debug view ─────────────────────────────────────────
# Two halves: what the agent itself can see (RULES.md rule 1 — odom +
# GPS sensor outputs and anything derived from them) and what the
# simulator harness sees that the agent does NOT (ground truth used
# only for debugging stuck/oscillating agents).
@dataclass
class AgentSelfView:
    """The view the onboard code has of itself. Building blocks here
    must all be derivable from odometry and GPS sensor readings — no
    sneaking in `true_pos` or `true_heading`."""
    odom_xy: Tuple[float, float]
    odom_vel: Tuple[float, float]
    last_gps_world_xy: Optional[Tuple[float, float]]
    last_gps_latlon: Optional[Tuple[float, float]]
    gps_connected: bool
    gps_reconnect_active: bool
    gps_dropout_active: bool
    ekf_pos: Tuple[float, float]
    ekf_theta_deg: float
    ekf_theta_std_deg: float
    bootstrap_done: bool
    candidate_goal_world: Tuple[float, float]
    path_len: int
    best_i: int
    last_pad_m: float
    # Real-robot parity: STOP_REFINE bubble latched on first entry.
    # True once the EKF position has fallen inside the
    # STOP_REFINE_K · STOP_REFINE_SIGMA bubble around the true goal
    # (post-latch, `published_goal_world` no longer updates).
    refinement_locked: bool


@dataclass
class AgentTrueView:
    """Sim-only ground truth. The agent does NOT have access to any of
    these — they exist purely so we can diagnose which invariant a
    misbehaving agent is violating (drift, heading error, EKF bias,
    backoff state, etc.)."""
    pos_xy: Tuple[float, float]
    heading_deg: float
    goal_xy: Tuple[float, float]
    dist_to_goal_m: float
    heading_err_deg: float
    ekf_pos_err_m: float
    arrived: bool
    coasting: bool
    sim_time_s: float
    steps: int
    no_path: bool
    in_stuck_backoff: bool
    last_replan_age_s: float
    heading_resync_count: int


@dataclass
class AgentDebugView:
    self_view: AgentSelfView
    true_view: AgentTrueView

    def __str__(self) -> str:
        s, t = self.self_view, self.true_view

        def _xy(p):
            return "None" if p is None else f"({p[0]:.2f}, {p[1]:.2f})"

        def _ll(p):
            return "None" if p is None else f"({p[0]:.6f}, {p[1]:.6f})"

        return (
            f"── AGENT DEBUG  t={t.sim_time_s:.1f}s  step={t.steps} ──\n"
            f"  self_view (what the agent knows):\n"
            f"    odom            {_xy(s.odom_xy)}\n"
            f"    odom_vel        ({s.odom_vel[0]:+.2f}, {s.odom_vel[1]:+.2f}) m/s\n"
            f"    last_gps_xy     {_xy(s.last_gps_world_xy)}\n"
            f"    last_gps_latlon {_ll(s.last_gps_latlon)}\n"
            f"    gps             connected={s.gps_connected}  "
            f"reconnect={s.gps_reconnect_active}  "
            f"dropout={s.gps_dropout_active}\n"
            f"    ekf_pos         {_xy(s.ekf_pos)}\n"
            f"    ekf_theta       {s.ekf_theta_deg:+.2f}° "
            f"(σ={s.ekf_theta_std_deg:.2f}°)\n"
            f"    bootstrap_done  {s.bootstrap_done}\n"
            f"    refine_locked   {s.refinement_locked}\n"
            f"    candidate_goal  {_xy(s.candidate_goal_world)}\n"
            f"    path            len={s.path_len}  best_i={s.best_i}  "
            f"last_pad={s.last_pad_m:.1f} m\n"
            f"  true_view (sim-only, agent cannot see):\n"
            f"    true_pos        {_xy(t.pos_xy)}\n"
            f"    true_heading    {t.heading_deg:+.2f}°\n"
            f"    true_goal       {_xy(t.goal_xy)}\n"
            f"    dist_to_goal    {t.dist_to_goal_m:.2f} m\n"
            f"    heading_err     {t.heading_err_deg:+.2f}°\n"
            f"    ekf_pos_err     {t.ekf_pos_err_m:.2f} m  "
            f"(EKF vs truth)\n"
            f"    arrived={t.arrived}  coasting={t.coasting}  "
            f"no_path={t.no_path}  stuck_backoff={t.in_stuck_backoff}\n"
            f"    last_replan_age {t.last_replan_age_s:.2f} s\n"
            f"    heading_resyncs {t.heading_resync_count}"
        )


# ── Truth / Virtual proxy classes ────────────────────────────────
# The simulation has two physical representations of the same robot
# that exchange data through a fixed interface:
#
#   ┌─────────────────────────┐  forces (F, M)  ┌────────────────────┐
#   │  Truth (ground reality) │ ◄────────────── │ Virtual (estimator)│
#   │  ─ true_pos             │                 │ ─ odom (biased)     │
#   │  ─ true_heading         │  body deltas    │ ─ body_heading      │
#   │  ─ body_heading_true    │ (biased rotn)   │ ─ ekf (3-state KF)  │
#   │  ─ forward/angular vel  │ ──────────────► │ ─ gps_history       │
#   └─────────────────────────┘                 └────────────────────┘
#
# Truth holds the physics; the simulator-as-world owns it and the
# robot itself never reads it. Virtual holds the agent's onboard
# belief — biased odometry, EKF state, GPS samples — and feeds the
# controller. The controller's output (thrust + moment) is applied
# back to Truth, closing the loop.
#
# These two classes are *views* over the existing GPSWaypointSim
# attributes — they do not duplicate state. They exist so the
# data-flow contract is visible at the type level rather than buried
# in attribute prefixes (``true_*`` vs everything else).
class Truth:
    """Read-only window onto the simulator's ground-truth state.
    The robot's onboard code (controller, planner, EKF) MUST NOT
    touch this — it represents reality the deployed system cannot
    observe directly. The physics integrator updates these fields
    every tick from the forces produced by Virtual's controller."""
    __slots__ = ("_a",)
    def __init__(self, agent): self._a = agent
    @property
    def pos(self):           return self._a.true_pos
    @property
    def heading(self):       return self._a.true_heading
    @property
    def body_heading(self):  return self._a.body_heading_true
    @property
    def forward_vel(self):   return self._a.forward_vel
    @property
    def angular_vel(self):   return self._a.angular_vel
    @property
    def trail(self):         return self._a.true_trail


class Virtual:
    """Read-only window onto the robot's onboard belief — what the
    deployed software actually has access to: biased wheel odom,
    IMU-fused body heading, GPS samples, and the EKF estimate of
    (x_world, y_world, θ). The controller and A* planner consume
    these. ``Virtual.ekf_pos`` is the gps_handler_node EKF's filtered
    position in world frame; ``Virtual.odom`` is the raw biased
    /odom in odom frame."""
    __slots__ = ("_a",)
    def __init__(self, agent): self._a = agent
    @property
    def odom(self):              return self._a.odom
    @property
    def odom_vel(self):          return self._a.odom_vel
    @property
    def body_heading(self):      return self._a.body_heading
    @property
    def yaw_bias_offset(self):   return self._a._yaw_bias_offset
    @property
    def ekf(self):               return self._a.ekf
    @property
    def ekf_pos(self):
        return self._a.ekf.pos_xy if self._a.ekf is not None \
            else (0.0, 0.0)
    @property
    def ekf_theta(self):
        return self._a.ekf.theta if self._a.ekf is not None else 0.0
    @property
    def heading_offset_est(self): return self._a.heading_offset_est
    @property
    def gps_history(self):       return self._a.gps_history
    @property
    def D_total(self):           return self._a.D_total


# ── GPS-no-magnetometer simulator ────────────────────────────────
class GPSWaypointSim:
    """Encapsulates ground-truth state, robot belief, GPS, A* planning,
    and one-step kinematics. Pure logic — no matplotlib.

    Two proxy views are exposed for clarity:
      * ``self.truth`` — physical ground reality (the simulator owns
        this; the deployed robot cannot observe it directly).
      * ``self.virtual`` — onboard belief state (biased odom, EKF,
        GPS history; the controller and planner consume this).
    """
    def __init__(self, costmap, start_world, true_heading_rad,
                 goal_world, rng, roofs=(), projectors=(),
                 jammers=(), foliage=(), spoofers=(),
                 odom_yaw_bias_rate=None,
                 goal_queue=None,
                 coldstart_bias_enabled=False,
                 next_hint_enabled=None):
        self.cm = costmap
        self.start_world  = tuple(start_world)
        self.true_heading = float(true_heading_rad)
        self.goal_world   = tuple(goal_world)
        self.rng = rng
        # Chained-mission state. ``goal_queue`` is a list of remaining
        # (lat_deg, lon_deg) tuples AFTER the current ``goal_world``;
        # default ``None`` ⇒ single-goal mode (regression-safe).
        # ``leg_index`` is 1-based and ``leg_count`` is the total
        # number of legs (1 in single-goal mode). Mirrors the
        # multi-leg progression on the deployed
        # ``origin/improve/gps-waypoint-continuity`` branch where
        # each new ``NavigateToWaypoint`` goal pops the next entry
        # and the EKF / heading-fit state is *preserved* across the
        # boundary (the "preemptive next-goal cache"), while a
        # handful of per-leg baselines are reset.
        self._goal_queue_init = list(goal_queue or [])
        self.goal_queue = list(self._goal_queue_init)
        self.leg_count  = 1 + len(self._goal_queue_init)
        self.leg_index  = 1
        # Steps elapsed since the most recent leg start. Reset to 0
        # at ``_advance_to_next_leg`` so per-leg detectors that key
        # off ``self.steps == N`` (e.g. the snapshot classifier)
        # don't trip on stale counts from prior legs.
        self._leg_start_step = 0
        self._leg_start_sim_time = 0.0
        # Per-agent encoder yaw bias rate (rad of fictitious yaw
        # drift per meter of forward motion). Defaults to the
        # global calibration when ``ODOM_YAW_BIAS_ENABLE`` is
        # True; pass 0.0 to spawn a "perfect movement" twin with
        # no bias for side-by-side comparison.
        if odom_yaw_bias_rate is None:
            odom_yaw_bias_rate = (ODOM_YAW_BIAS_RAD_PER_M
                                  if ODOM_YAW_BIAS_ENABLE else 0.0)
        self._odom_yaw_bias_rate = float(odom_yaw_bias_rate)
        self.roofs       = list(roofs)        # (x_min, y_min, x_max, y_max)
        self.projectors  = list(projectors)   # ((v0, v1, v2), (bx, by))
        self.jammers     = list(jammers)      # (cx, cy, radius)
        self.foliage     = list(foliage)      # (cx, cy, radius)
        self.spoofers    = list(spoofers)     # ((cx, cy), (fx, fy))
        # Precomputed projector arrays for the per-tick bias loop. With
        # 1000 agents at 10 Hz, recomputing centroid + np.array per tick
        # is wasted work — the projectors are static once placed.
        if self.projectors:
            cxs = []; cys = []; bxs = []; bys = []
            for verts, bias in self.projectors:
                cx, cy, _ = projector_centroid_radius(verts)
                cxs.append(cx); cys.append(cy)
                bxs.append(bias[0]); bys.append(bias[1])
            self._proj_cx = np.asarray(cxs, dtype=float)
            self._proj_cy = np.asarray(cys, dtype=float)
            self._proj_bx = np.asarray(bxs, dtype=float)
            self._proj_by = np.asarray(bys, dtype=float)
        else:
            self._proj_cx = self._proj_cy = None
            self._proj_bx = self._proj_by = None
        # Roofs as a 2D array for vectorized AABB containment.
        if self.roofs:
            self._roof_arr = np.asarray(self.roofs, dtype=float)
        else:
            self._roof_arr = None
        # Jammer / foliage zone arrays. Same shape: (N, 3) of (cx, cy, r).
        self._jammer_arr = (np.asarray(self.jammers, dtype=float)
                             if self.jammers else None)
        self._foliage_arr = (np.asarray(self.foliage, dtype=float)
                              if self.foliage else None)
        # Spoofer arrays: centers and fake-target world coords.
        if self.spoofers:
            self._spoof_cx = np.asarray(
                [s[0][0] for s in self.spoofers], dtype=float)
            self._spoof_cy = np.asarray(
                [s[0][1] for s in self.spoofers], dtype=float)
            self._spoof_fx = np.asarray(
                [s[1][0] for s in self.spoofers], dtype=float)
            self._spoof_fy = np.asarray(
                [s[1][1] for s in self.spoofers], dtype=float)
        else:
            self._spoof_cx = self._spoof_cy = None
            self._spoof_fx = self._spoof_fy = None

        # ── Truth / Virtual views ─────────────────────────────────
        # Construct the proxy windows up-front so any downstream code
        # touching ``self.truth`` / ``self.virtual`` works regardless
        # of the attribute initialisation order below.
        self.truth   = Truth(self)
        self.virtual = Virtual(self)

        # Truth (the simulator knows; the robot's belief does not).
        self.true_pos = list(start_world)
        self.true_trail = [tuple(start_world)]

        # Robot's belief: perfect odom, starts at (0, 0) facing +x_odom.
        self.odom = [0.0, 0.0]
        self.odom_vel = [0.0, 0.0]       # cached; derived from
                                          # forward_vel × body_heading
        # Chaplygin sleigh state. Body heading is in *odom* frame; in
        # world it's `body_heading + true_heading`. Forward velocity is
        # scalar along the body axis, angular velocity is scalar about
        # the body center.
        #
        # `body_heading` is the REPORTED orientation — what the encoders
        # / wheel_odom_pub publish, which is what the controller and
        # EKF actually consume. `body_heading_true` is the body's
        # actual physical orientation in the (also abstract) true-odom
        # frame; the two diverge over forward distance per the
        # encoder bias model (ODOM_YAW_BIAS_RAD_PER_M). True world
        # motion follows `body_heading_true`; the EKF predict step
        # follows `body_heading` (via the reported `self.odom`
        # integral). Both are 0 at construction.
        self.body_heading = 0.0          # rad, REPORTED, in odom frame
        self.body_heading_true = 0.0     # rad, TRUE physical orientation
        # Deterministic encoder yaw bias accumulator (rad). The
        # REPORTED body angle is derived as
        #   body_heading = body_heading_true + _yaw_bias_offset
        # where _yaw_bias_offset advances by K·|Δd_truth| each tick
        # (K = self._odom_yaw_bias_rate, Δd_truth = forward_vel·dt).
        # This guarantees the reported odom is a deterministic biased
        # rotation of the truth's motion — when truth has no motion,
        # odom has no motion, and the EKF predict is identity.
        self._yaw_bias_offset = 0.0
        self.forward_vel  = 0.0          # m/s along body axis
        self.angular_vel  = 0.0          # rad/s
        # Cumulative reported forward distance (m). Stamped on every
        # gps_history entry so the closed-form fit can recover both
        # the rotation θ and the per-meter encoder yaw bias rate K
        # by linear regression of theta_i vs δD/2. Mirrored on the
        # robot in gps_handler_node._odom_distance_m.
        self.D_total = 0.0
        # Estimated encoder yaw bias rate K (rad of fictitious yaw
        # drift per meter of forward motion). Updated from the
        # joint (θ, K) closed-form fit at bootstrap and every
        # resync. Used to derotate per-tick reported odom deltas
        # by R(-K · D) before feeding to ``self.ekf.predict`` —
        # the EKF then sees a "true odom" delta and its internal θ
        # stays consistent with the θ_eff = θ - K·D model the fit
        # assumes. Mirrored on the robot in gps_handler_node.
        self.K_est = 0.0
        # Heading estimate now lives inside the EKF (created below).
        self.ekf = None

        # GPS history of (stamp_s, gps_xy, odom_xy, D_at_sample).
        # Tuple shape mirrors the robot's HistoryEntry contract so
        # the closed-form fits port across files unchanged.
        self.gps_history = []
        self.gps_scatter = []

        # Planner state
        self.path_world = None
        self.last_window = None
        self.last_pad = ASTAR_INITIAL_PAD
        # Last candidate goal A* targeted, used to trigger a replan when
        # the robot's belief about where it will land has shifted.
        self.last_planned_goal = None
        self._last_replan_time = -1e9     # force first plan
        self._best_i = 0                  # cached lookahead anchor index
        self._stuck_until = -1.0          # back off planning if no path
        # NAV2 goal-publication rate-limiter. Live candidate is
        # computed every step; the published goal (what A* / the
        # controller actually drive toward) is sampled at NAV2_GOAL_HZ.
        # Each agent's first publish is offset by a random fraction
        # of the period so a 1000-agent ensemble doesn't all snap
        # their published goals on the same wall-clock boundary
        # (which would look like a synchronized strobe in the GUI).
        self.published_goal_world = tuple(self.goal_world)
        # Real-robot parity, mirrored from
        # gps_handler_node._publish_goal (~line 1575). Once the EKF
        # position falls inside the STOP_REFINE_K · STOP_REFINE_SIGMA
        # bubble (0.6 m) around the true goal, freeze
        # `published_goal_world` so the controller / planner don't
        # chase sub-σ wobble. Latches True on first entry — does not
        # re-arm even if the EKF drifts back outside.
        self.refinement_locked = False
        # Low-pass filter on the candidate goal. Decoupled from EKF /
        # planner state so it can be tuned independently of those.
        # `None` means "not yet initialized — adopt the first raw
        # value verbatim". See `_update_candidate_smoother`.
        self._smoothed_candidate = None
        # improve/gps-waypoint-continuity — shadow EWMA for the next
        # queued GPS waypoint. Mirrors deployed
        # ``gps_handler_node._next_hint_*`` state (deployed L584-591).
        # Lives in parallel with ``_smoothed_candidate`` and is
        # consulted only on a leg switch: if the cached hint's world
        # xy matches the new leg's goal within
        # ``_hint_match_tolerance_m``, the shadow EWMA's current
        # value is promoted into ``_smoothed_candidate`` (warm
        # start) — otherwise we cold-start as before. In single-goal
        # mode (``goal_queue`` empty) the hint is never seeded and
        # ``_next_hint_enabled`` stays False, keeping the
        # single-goal headless smoke bit-identical.
        #
        # Optional CLI / kwarg override (matches deployed ROS
        # parameter ``next_hint_enabled``, default False, see
        # gps_handler_node.py L466 on
        # ``origin/improve/gps-waypoint-continuity``). When the
        # ``next_hint_enabled`` kwarg is None (the default), keep
        # the auto-on-with-queue behavior so ``--mission three-
        # waypoint`` continues to exercise the shadow EWMA without
        # the caller having to also pass ``--next-hint-enable``.
        # When the kwarg is explicitly True/False, that wins —
        # documented inline so the launcher can force-on for a
        # single-goal run, or force-off to A/B against the
        # shadow-disabled baseline.
        if next_hint_enabled is None:
            self._next_hint_enabled = bool(self._goal_queue_init)
        else:
            self._next_hint_enabled = (
                bool(next_hint_enabled)
                or bool(self._goal_queue_init))
        self._hint_match_tolerance_m = 0.5
        self._next_hint_lat_lon = None
        self._next_hint_world_xy = None
        self._next_hint_smoothed_candidate = None
        # improve/gps-waypoint-continuity — one-shot guard for the
        # cold-start θ_offset seed. Deployed handler only snaps the
        # EKF θ on the very FIRST GPS goal of the node's lifetime
        # (deployed L526); subsequent legs reuse the already-
        # bootstrapped θ. Sim doesn't currently grow a θ seed of
        # its own — this flag is here so any future seed path
        # respects the first-leg-only contract by default.
        self._coldstart_theta_seeded = False
        # CLI / kwarg toggle (matches deployed ROS parameter
        # ``coldstart_bias_enabled``, default False, see
        # gps_handler_node.py L481). Currently a wired stub: the
        # sim has no equivalent of ``_seed_coldstart_theta_if_needed``
        # because its EKF starts with theta_var0 = π² and the first
        # GPS update already produces a near-unity Kalman gain on θ.
        # TODO: if a real seed path is added (e.g. force-snap EKF θ
        # to the goal bearing on the very first leg only), gate it
        # on ``self._coldstart_bias_enabled`` and clear
        # ``self._coldstart_theta_seeded`` on cancel-without-goal so
        # the next valid leg picks up the snap. Field-parity contract
        # only ever fires once per node lifetime.
        self._coldstart_bias_enabled = bool(coldstart_bias_enabled)
        # Counter incremented every time ``_advance_to_next_leg``
        # actually promotes the shadow EWMA into the active
        # smoother (i.e. the cached hint matched the new leg within
        # tolerance). Surfaced as a sim-side observable so tests /
        # debug prints can assert on chaining behaviour.
        self._next_hint_warm_start_count = 0
        # Diagnostic counter — number of envelope-rejected candidates.
        self._cand_reject_count = 0
        # Stuck-detector state.
        # `_dist_history` is a list of (sim_time, dist_to_goal) where
        # dist comes from the EKF position estimate (or raw GPS if no
        # EKF yet). Trimmed to a `STUCK_WINDOW_S` rolling window.
        # `_stuck_recovery_until` is the sim_time before which the
        # controller should ignore the alignment gate and force a
        # forward-speed floor. -1 means inactive.
        self._dist_history = []
        self._stuck_recovery_until = -1.0
        self._stuck_event_count = 0
        # Moving-away detector → envelope-filter suspension.
        # `_envelope_suspended_until_s` is the sim_time before which
        # `_update_candidate_smoother` should bypass the 1/r envelope
        # check (so corrections to a biased heading lock can flow
        # through). -1 means inactive.
        self._envelope_suspended_until_s = -1.0
        self._moving_away_event_count = 0
        # One-shot heading-flip flag — set after N moving-away events
        # without arrival have triggered the Hail-Mary 180° flip.
        self._heading_flipped = False
        _publish_period = 1.0 / max(NAV2_GOAL_HZ, 1e-3)
        self._last_published_time = -float(
            rng.uniform(0.0, _publish_period))

        # GPS noise/disconnect state
        self._gps_bias_phase_x = rng.uniform(0, 2 * math.pi)
        self._gps_bias_phase_y = rng.uniform(0, 2 * math.pi)
        self._gps_bias_dir_a   = rng.uniform(0, 2 * math.pi)  # ellipse axis
        self._gps_bias_eccen   = 0.6 + 0.3 * rng.random()     # 0.6–0.9
        self.gps_connected     = True
        self._random_dropout_active = False
        self._random_dropout_until  = -1.0                    # sim-time
        # Blackout-zone reconnect bookkeeping. `_gps_reconnect_until`
        # is the sim-time at which the current reconnect window ends;
        # `_gps_reconnect_cooldown_until` gates how soon another
        # reconnect can fire. `_was_under_roof` tracks the prior
        # _under_roof state so we can detect the False→True edge that
        # auto-fires the reconnect.
        self._gps_reconnect_until           = -1.0
        self._gps_reconnect_cooldown_until  = -1.0
        self._was_under_roof = False
        # Lock-in recovery counter now lives inside the EKF —
        # ``self.ekf.consecutive_rejects`` is the canonical signal,
        # mirroring gps_ekf.GpsEkf. step() consumes it directly.
        # Heading-resync bookkeeping. `_heading_resync_until` gates
        # how soon the next snap can fire; `_heading_resync_count`
        # surfaces in the debug view so we can see how active this
        # recovery has been for any given agent.
        self._heading_resync_until = -1.0
        self._heading_resync_count = 0
        # Periodic heading refit (real-robot parity). Fires every
        # PERIODIC_REFIT_PERIOD_S regardless of cooldown / detector
        # state, but only snaps when the closed-form fit disagrees
        # with the EKF by more than the threshold.
        self._last_periodic_refit_s = -math.inf
        # Local-vs-world divergence detector baselines + cooldown.
        # Lazy-initialized once the EKF exists and we've seen at
        # least one GPS sample (mirrors gps_handler_node's
        # local_d_start / world_d_start lazy-init on the first odom
        # tick that has both). Reset on every divergence event so
        # the next evaluation measures from post-resync state.
        self._local_d_start = None
        self._world_d_start = None
        self._divergence_cooldown_until = -1.0
        self._divergence_event_count = 0
        # Cycle-slip state. While `_cycle_slip_until > sim_time`, every
        # GPS reading gets the persistent offset (_cycle_slip_dx,
        # _cycle_slip_dy) added on top of normal noise/bias.
        self._cycle_slip_until = -1.0
        self._cycle_slip_dx = 0.0
        self._cycle_slip_dy = 0.0
        # Noise-burst (loose ionospheric) state. While active, σ is
        # multiplied by NOISE_BURST_MULT.
        self._noise_burst_until = -1.0

        # Cache last GPS reading even when disconnected (controller fallback)
        self.last_gps_xy = tuple(start_world)

        # Belief-of-goal cloud, in WORLD frame: where the robot's current
        # plan would actually land it given its (imperfect) heading
        # estimate. Updated each GPS tick.
        self.intended_endpoint_history = []

        # Time tracking (s, since sim start)
        self.sim_time = 0.0
        self._gps_acc_time = math.inf       # forces an immediate sample

        # Bootstrap state. Unified design ported from deployed
        # ``gps_handler_node`` on ``origin/improve/gps-waypoint-continuity``
        # (L535-553): we run the EKF continuously from tick 1 with high
        # initial θ variance (theta_var0 = π²) and let the first
        # ``update_theta_measurement`` / standard GPS update produce a
        # near-unity Kalman gain that effectively snaps θ on the first
        # valid sample. The original two-state machine (explicit
        # ``_bootstrap_theta`` + ``reset_theta`` + ``_adopt_K`` for the
        # first ~5 m of motion, then graduate) caused field deadlocks:
        # NAV2 wouldn't translate without a goal, the bootstrap couldn't
        # fit without translation, robot sat stuck. Initializing
        # ``bootstrap_done = True`` from __init__ makes the explicit
        # bootstrap branch in ``step()`` (L3694-3734) unreachable —
        # matching deployed's dead L871-branch — while keeping the
        # ``_bootstrap_theta`` / ``_adopt_K`` helpers defined as harmless
        # no-ops in case any out-of-band caller still hits them.
        self.bootstrap_done = True
        self.bootstrap_min_travel = 5.0

        self.steps = 0
        self.arrived = False
        self.coasting = False               # post-arrival, friction-only
        # Robot's onboard "I succeeded" flag — fires when wheel
        # ODOM reaches the published candidate goal (odom frame).
        # That's the only termination signal the deployed
        # gps_handler_node action server has: it cannot observe
        # the real GPS goal directly, only the candidate the
        # closed-form θ fit shaped for it. We track this
        # *separately* from ``self.arrived`` (which is the sim's
        # own verdict, gated on real-GPS-vs-goal-waypoint), so
        # we can see when the robot *thought* it was done versus
        # when it *actually* was — exactly the field-test
        # discrepancy where the robot reported success while
        # sitting 40 m off the GPS goal.
        self.robot_declared_success = False
        self.robot_declared_at_step = None
        self.robot_declared_truth_xy = None
        # Predicted-convergence flags — three terminal bins so the
        # headless loop can terminate as soon as every agent is
        # classified.
        #   predicted_success: candidate within PREDICT_RADIUS_M of
        #     the real goal for PREDICT_HOLD_TICKS in a row → θ
        #     converged → controller will close the rest.
        #   predicted_failure: candidate *more than*
        #     PREDICT_FAIL_RADIUS_M from the goal for
        #     PREDICT_FAIL_HOLD_TICKS in a row while bootstrap done,
        #     OR bootstrap not done after PREDICT_BOOT_TIMEOUT_TICKS.
        self.predicted_success = False
        self.predicted_at_step = None
        self.predicted_failure = False
        self.predicted_failure_reason = None
        self.predicted_failure_at_step = None

        # Take an initial GPS sample so we have a start estimate, then
        # initialise the EKF anchored on that first reading. With a
        # huge θ variance the filter gives that prior almost no weight
        # — heading is recovered from motion, not from cold start.
        first = self._tick_gps()
        if first is None:
            first = (self.true_pos[0], self.true_pos[1])
        self.ekf = GPSEKF(first[0], first[1])

        # improve/gps-waypoint-continuity — seed the next-hint cache
        # from the head of ``goal_queue`` so the shadow EWMA can
        # converge during leg 1 and warm-start the leg-1→2 transition.
        # No-op when the queue is empty (single-goal mode), keeping
        # the headless smoke bit-identical. Mirrors what the
        # deployed dispatcher does via the
        # ``/gps_waypoint/next_hint`` topic before submitting the
        # next ``NavigateToWaypoint`` action goal (deployed
        # L457-465, L928-969).
        self._seed_next_hint_from_queue()

    def _seed_next_hint_from_queue(self):
        """Project ``goal_queue[0]`` into world meters and stash it
        as the active next-hint. Clears the shadow EWMA so it
        re-converges from scratch against the new hint. Called once
        from ``__init__`` and again at every ``_advance_to_next_leg``
        AFTER the new leg's per-leg baselines are reset.

        No-op when the queue is empty (last leg → no further hint)
        or when ``_next_hint_enabled`` is False (single-goal
        regression mode)."""
        if not self._next_hint_enabled:
            return
        if not self.goal_queue:
            # End of mission — clear any prior hint.
            self._next_hint_lat_lon = None
            self._next_hint_world_xy = None
            self._next_hint_smoothed_candidate = None
            return
        next_lat, next_lon = self.goal_queue[0]
        self._next_hint_lat_lon = (next_lat, next_lon)
        self._next_hint_world_xy = latlon_to_meters(next_lat, next_lon)
        self._next_hint_smoothed_candidate = None

    # -- GPS / belief ----------------------------------------------------
    def _gps_drift(self):
        """Slow correlated bias (multipath / ionospheric). Elliptical
        sinusoid with random phase and eccentricity per session."""
        t = self.sim_time
        omega = 2 * math.pi / GPS_BIAS_PERIOD_S
        u = math.cos(omega * t + self._gps_bias_phase_x)
        v = math.sin(omega * t + self._gps_bias_phase_y) * self._gps_bias_eccen
        a = self._gps_bias_dir_a
        bx = (math.cos(a) * u - math.sin(a) * v) * GPS_BIAS_AMPL_M
        by = (math.sin(a) * u + math.cos(a) * v) * GPS_BIAS_AMPL_M
        return bx, by

    def _under_roof(self):
        if self._roof_arr is None:
            return False
        tx, ty = self.true_pos[0], self.true_pos[1]
        a = self._roof_arr
        return bool(np.any((a[:, 0] <= tx) & (tx <= a[:, 2])
                            & (a[:, 1] <= ty) & (ty <= a[:, 3])))

    def _projector_bias(self):
        """Sum the multipath bias from every projector within
        PROJECTOR_INFLUENCE_RADIUS_M of the robot. Linear taper."""
        if self._proj_cx is None:
            return 0.0, 0.0
        R = PROJECTOR_INFLUENCE_RADIUS_M
        dx = self.true_pos[0] - self._proj_cx
        dy = self.true_pos[1] - self._proj_cy
        d = np.hypot(dx, dy)
        mask = d < R
        if not mask.any():
            return 0.0, 0.0
        w = 1.0 - d[mask] / R
        bx = float(np.dot(self._proj_bx[mask], w))
        by = float(np.dot(self._proj_by[mask], w))
        return bx, by

    def _in_jammer(self):
        """True iff the agent is inside any hex jammer. Approximated
        as a disc of the hex's circumradius — the difference is small
        and the disc check vectorizes cleanly."""
        if self._jammer_arr is None:
            return False
        a = self._jammer_arr
        dx = self.true_pos[0] - a[:, 0]
        dy = self.true_pos[1] - a[:, 1]
        return bool(np.any(dx * dx + dy * dy <= a[:, 2] ** 2))

    def _foliage_noise_mult(self):
        """Multiplier on GPS white-noise σ from foliage zones the
        agent is currently inside. Multiple overlapping foliage zones
        compound (each multiplies σ by FOLIAGE_NOISE_MULT)."""
        if self._foliage_arr is None:
            return 1.0
        a = self._foliage_arr
        dx = self.true_pos[0] - a[:, 0]
        dy = self.true_pos[1] - a[:, 1]
        inside = (dx * dx + dy * dy) <= a[:, 2] ** 2
        n = int(inside.sum())
        if n == 0:
            return 1.0
        return FOLIAGE_NOISE_MULT ** n

    def _spoofer_lock(self):
        """If the agent is inside a spoofer's influence radius, return
        the (fake_x, fake_y) world coords the receiver should report.
        Otherwise return None. Closest-spoofer wins on overlap."""
        if self._spoof_cx is None:
            return None
        dx = self.true_pos[0] - self._spoof_cx
        dy = self.true_pos[1] - self._spoof_cy
        d2 = dx * dx + dy * dy
        R2 = SPOOFER_INFLUENCE_RADIUS_M ** 2
        in_zone = d2 <= R2
        if not in_zone.any():
            return None
        # Pick the closest spoofer's fake target.
        idx = int(np.argmin(np.where(in_zone, d2, np.inf)))
        return float(self._spoof_fx[idx]), float(self._spoof_fy[idx])

    def _maybe_start_cycle_slip(self, dt):
        if self.sim_time < self._cycle_slip_until:
            return
        if self.rng.random() < CYCLE_SLIP_HZ_PER_S * dt:
            dur = self.rng.uniform(*CYCLE_SLIP_DURATION_S)
            self._cycle_slip_until = self.sim_time + dur
            # Persistent offset of σ ≈ CYCLE_SLIP_OFFSET_M, drawn once
            # at slip onset.
            self._cycle_slip_dx = self.rng.normal(0, CYCLE_SLIP_OFFSET_M)
            self._cycle_slip_dy = self.rng.normal(0, CYCLE_SLIP_OFFSET_M)

    def _maybe_start_noise_burst(self, dt):
        if self.sim_time < self._noise_burst_until:
            return
        if self.rng.random() < NOISE_BURST_HZ_PER_S * dt:
            dur = self.rng.uniform(*NOISE_BURST_DURATION_S)
            self._noise_burst_until = self.sim_time + dur

    def request_gps_reconnect(self):
        """Open a fresh-GPS window of length GPS_RECONNECT_DURATION_S.
        Returns True if the window was opened, False if still in the
        cooldown from a previous reconnect. While the window is open,
        roof blackout is bypassed and GPS readings are clean (no
        leak-skew on top of the usual bias/noise) — the agent gets
        real fixes again for the next few seconds even if it stays
        shadowed."""
        if self.sim_time < self._gps_reconnect_cooldown_until:
            return False
        self._gps_reconnect_until = (
            self.sim_time + GPS_RECONNECT_DURATION_S)
        self._gps_reconnect_cooldown_until = (
            self._gps_reconnect_until + GPS_RECONNECT_COOLDOWN_S)
        return True

    @property
    def gps_reconnect_active(self):
        return self.sim_time < self._gps_reconnect_until

    def _maybe_start_dropout(self, dt):
        if self._random_dropout_active:
            return
        # Bernoulli draw per dt — independent of GPS sample rate
        if self.rng.random() < GPS_DROPOUT_HZ_PER_S * dt:
            dur = self.rng.uniform(*GPS_DROPOUT_DURATION_S)
            self._random_dropout_until = self.sim_time + dur
            self._random_dropout_active = True

    def _tick_gps(self):
        """Take one GPS sample if currently connected. Returns the
        (noisy) reading or None if no fix this cycle (random dropout
        OR robot under a roof). In --crazy mode roofs may occasionally
        leak a heavily-skewed reflected fix instead of dropping out."""
        # Random dropout state machine
        if (self._random_dropout_active
                and self.sim_time >= self._random_dropout_until):
            self._random_dropout_active = False
        # Geometric (static) hazard membership. The reconnect on roof
        # entry must fire on the geometric edge so the agent still
        # gets its 5 s clean window — independent of whether the roof
        # is "transiently off" this tick.
        under_roof_geo = self._under_roof()
        in_jammer_geo  = self._in_jammer()
        # Transient-off rolls. With small per-tick probability, each
        # hazard has no effect this tick — gives agents inside a
        # compound zone (e.g. roof + foliage overlap) periodic clean
        # fixes that let them recover EKF / heading.
        under_roof = under_roof_geo and (
            self.rng.random() >= ROOF_TRANSIENT_OFF_PROB)
        in_jammer = in_jammer_geo and (
            self.rng.random() >= JAMMER_TRANSIENT_OFF_PROB)
        # Jammer zone: each fix is independently dropped with probability
        # JAMMER_DROPOUT_PROB while the jammer is active. Reconnect
        # window does NOT bypass the jammer — RF jamming is a different
        # physical effect than the roof-shadow → window-glance reacquire
        # trick. (When the jammer is transiently off, the dropout roll
        # is skipped entirely → fix flows through.)
        if in_jammer and self.rng.random() < JAMMER_DROPOUT_PROB:
            self.gps_connected = False
            return None
        if under_roof_geo and not self._was_under_roof:
            self.request_gps_reconnect()
        self._was_under_roof = under_roof_geo
        reconnect_active = self.sim_time < self._gps_reconnect_until
        # Roof leak: receiver locks on to a reflection while the
        # antenna is shadowed. Suppressed during a reconnect window —
        # the whole point of the window is *clean* fixes.
        leak_skew = (under_roof
                     and not reconnect_active
                     and not self._random_dropout_active
                     and ROOF_BLACKOUT_LEAK_PROB > 0.0
                     and self.rng.random() < ROOF_BLACKOUT_LEAK_PROB)
        self.gps_connected = (not self._random_dropout_active
                               and (not under_roof
                                    or leak_skew
                                    or reconnect_active))
        if not self.gps_connected:
            return None

        # Slow correlated bias + per-projector multipath shift
        dx, dy = self._gps_drift()
        px, py = self._projector_bias()
        # Spoofer lock: if inside a spoofer's influence, the receiver
        # is pinned to the spoofer's fake target rather than to truth.
        # This bypasses the normal (true_pos + bias + noise) signal.
        # When the spoofer is transiently off, the lock fails through
        # to a normal (truth-anchored) reading.
        spoof = self._spoofer_lock()
        if spoof is not None and (
                self.rng.random() < SPOOFER_TRANSIENT_OFF_PROB):
            spoof = None
        # Noise σ is multiplied by foliage and noise-burst factors.
        sigma_mult = self._foliage_noise_mult()
        if self.sim_time < self._noise_burst_until:
            sigma_mult *= NOISE_BURST_MULT
        sigma = GPS_NOISE_STD * sigma_mult
        # ── GPS antenna lever-arm: the receiver reports the
        # antenna's world position, NOT base_link's. Bowser's URDF
        # places the antenna at `GPS_ANTENNA_OFFSET_BASELINK` from
        # base_link; rotate that offset into the world frame using
        # the body's TRUE world heading (= true_heading +
        # body_heading_true — note: TRUE physical orientation, NOT
        # the encoder-biased reported one) and use the antenna
        # position as the noise / bias anchor. Spoofers still pin
        # to the fake target — they bypass the base_link-vs-antenna
        # distinction entirely.
        ax_b, ay_b = GPS_ANTENNA_OFFSET_BASELINK
        body_heading_world_true = self.true_heading + self.body_heading_true
        cT_a = math.cos(body_heading_world_true)
        sT_a = math.sin(body_heading_world_true)
        ant_dx = cT_a * ax_b - sT_a * ay_b
        ant_dy = sT_a * ax_b + cT_a * ay_b
        ant_x = self.true_pos[0] + ant_dx
        ant_y = self.true_pos[1] + ant_dy
        if spoof is not None:
            base_x, base_y = spoof
        else:
            base_x = ant_x + dx + px
            base_y = ant_y + dy + py
        nx = base_x + self.rng.normal(0, sigma)
        ny = base_y + self.rng.normal(0, sigma)
        # One-shot multipath spike
        if self.rng.random() < GPS_OUTLIER_PROB:
            nx += self.rng.normal(0, GPS_OUTLIER_STD)
            ny += self.rng.normal(0, GPS_OUTLIER_STD)
        if leak_skew:
            nx += self.rng.normal(0, ROOF_BLACKOUT_SKEW_M)
            ny += self.rng.normal(0, ROOF_BLACKOUT_SKEW_M)
        # Cycle slip: a persistent offset stays on top of every fix
        # until the slip resets a few seconds later.
        if self.sim_time < self._cycle_slip_until:
            nx += self._cycle_slip_dx
            ny += self._cycle_slip_dy

        # ── Antenna lever-arm correction (mirrors
        # gps_handler_node._gps_callback). The receiver-side code
        # subtracts an estimate of the antenna offset in world frame
        # using its current θ belief — so a wrong θ leaves a
        # heading-locked residual bias that the EKF then locks onto.
        # Done BEFORE the sample lands in gps_history / last_gps_xy
        # so the closed-form heading fits and downstream consumers
        # all see base_link estimates the same way the robot does.
        # When the EKF doesn't exist yet (very first call from
        # __init__), use θ=0 — same as the robot, whose EKF is
        # constructed at (0, 0, 0) before any /gps_fix arrives.
        if GPS_LEVER_ARM_CORRECTION_ENABLE:
            theta_est = self.ekf.theta if self.ekf is not None else 0.0
            yaw_world_est = self.body_heading + theta_est
            cR = math.cos(yaw_world_est)
            sR = math.sin(yaw_world_est)
            nx -= cR * ax_b - sR * ay_b
            ny -= sR * ax_b + cR * ay_b

        # Tuple shape (stamp_s, gps_xy, odom_xy, D) — matches the
        # robot's HistoryEntry contract in gps_ekf.py so the
        # closed-form fits ported between sim and robot work on
        # the same schema. ``D`` is the cumulative reported forward
        # distance at this sample, used by the joint (θ, K) fit
        # added for encoder-yaw-bias compensation.
        self.gps_history.append(
            (self.sim_time, (nx, ny), tuple(self.odom),
             self.D_total))
        self.gps_scatter.append((nx, ny))
        if len(self.gps_history) > GPS_HISTORY_LEN:
            self.gps_history.pop(0)
        if len(self.gps_scatter) > GPS_HISTORY_LEN:
            self.gps_scatter.pop(0)
        self.last_gps_xy = (nx, ny)
        return nx, ny

    def _joint_theta_K_fit(self, samples, min_baseline):
        """Joint estimation of heading offset θ AND encoder yaw
        bias rate K from a window of (stamp, gps, odom, D) samples.

        Each pair (anchor, sample_i) gives:
            theta_i = atan2(Δgps) − atan2(Δodom)
                    ≈ θ_world − K · D_mid_i
        where  D_mid_i = (D_anchor + D_i) / 2  is the average
        cumulative forward distance over the pair's interval.
        Linear regression of theta_i vs D_mid_i recovers (θ, K):
        intercept = θ_world,  slope = −K.

        Returns (θ, K, max_baseline). When the per-pair count is
        too small for a slope estimate, falls back to circular-mean
        θ with K = 0 — preserving the original closed-form
        behaviour.

        Mirrors gps_ekf._joint_theta_K_fit on the robot side.
        """
        if len(samples) < 4:
            return None, 0.0, 0.0
        _, g0_xy, o0_xy, D0 = samples[0]
        cos_sum = 0.0; sin_sum = 0.0; w_sum = 0.0; max_b = 0.0
        # (theta_i, D_mid_i, w_i) for the K linear fit.
        pairs = []
        for i in range(1, len(samples)):
            _, gi_xy, oi_xy, Di = samples[i]
            bdx = oi_xy[0] - o0_xy[0]; bdy = oi_xy[1] - o0_xy[1]
            bl = math.hypot(bdx, bdy)
            if bl < min_baseline:
                continue
            gdx = gi_xy[0] - g0_xy[0]; gdy = gi_xy[1] - g0_xy[1]
            gl = math.hypot(gdx, gdy)
            ratio = gl / bl if bl > 1e-9 else 0.0
            if (ratio < (1.0 / HEADING_FIT_MAGRATIO_MAX)
                    or ratio > HEADING_FIT_MAGRATIO_MAX):
                continue
            theta_i = math.atan2(gdy, gdx) - math.atan2(bdy, bdx)
            D_mid_i = 0.5 * (D0 + Di)
            cos_sum += bl * math.cos(theta_i)
            sin_sum += bl * math.sin(theta_i)
            w_sum   += bl
            if bl > max_b:
                max_b = bl
            pairs.append((theta_i, D_mid_i, bl))
        if w_sum == 0.0:
            return None, 0.0, 0.0
        theta_avg = math.atan2(sin_sum, cos_sum)
        # K via weighted linear regression of (theta_i − theta_avg)
        # against (D_mid_i − ⟨D_mid⟩). Gated on TWO conditions:
        #   (1) ≥ 3 valid pairs (statistical floor for slope).
        #   (2) D-spread across the fit window ≥
        #       BOOTSTRAP_K_MIN_D_SPREAD_M. Slope variance scales
        #       as σ²/Σ(x − x̄)², so D-spread is the dominant
        #       quality factor — without this gate, bootstrap-time
        #       fits with only ~5 m of D produce wildly noisy K
        #       estimates that the EKF's per-tick R(-K·D)
        #       derotation then compounds into multi-radian
        #       predict errors.
        #
        # When gated off, returns (theta_avg, 0.0, baseline). The
        # caller's ``self.K_est = bs_K`` line then keeps K_est at
        # 0.0 → predict step does no derotation → identical to
        # the K-disabled baseline (70.4 % arrival rate).
        if len(pairs) < 3:
            return theta_avg, 0.0, max_b
        # samples are appended monotonically with cumulative D, so
        # the spread is just last − first. Use the FIRST and LAST
        # entries of ``samples`` (anchor and most recent) directly.
        D_first = samples[0][3]
        D_last  = samples[-1][3]
        D_spread = D_last - D_first
        if D_spread < BOOTSTRAP_K_MIN_D_SPREAD_M:
            return theta_avg, 0.0, max_b
        mean_D_mid = sum(p[1] * p[2] for p in pairs) / w_sum
        sum_xy = 0.0; sum_xx = 0.0
        for theta_i, D_mid_i, w in pairs:
            residual = (theta_i - theta_avg + math.pi) % (2.0 * math.pi) - math.pi
            x = D_mid_i - mean_D_mid
            sum_xy += w * residual * x
            sum_xx += w * x * x
        if sum_xx < 1e-9:
            return theta_avg, 0.0, max_b
        slope = sum_xy / sum_xx
        K_est_raw = -slope
        # Three-tier safeguard ladder for joint-fit K adoption,
        # tested in isolation and combined; results logged here so
        # we can resume tuning without re-running exploratory
        # experiments. All measured against the K=0 baseline of
        # 704/1000 (70.4 %) arrival on the 1000-agent --real
        # sweep with the deployed encoder yaw bias active.
        #
        #   No safeguard:   150/1000 = 15.0 %  (catastrophic)
        #   (a) D-spread gate ≥ 15 m alone:
        #                   481/1000 = 48.1 %  (partial recovery)
        #   (a)+(b)+(c) D-spread + clamp + EMA α=0.2:
        #                   467/1000 = 46.7 %  (slightly worse;
        #                   EMA *slows* K toward truth, leaving
        #                   K wrong for more ticks before catching
        #                   up, compounding more error per tick)
        #
        # The variance-control knobs do not address the underlying
        # coupling failure: once a noisy K + θ pair is accepted,
        # the EKF's per-tick R(-K·D) derotation produces a
        # wrong-θ-and-K equilibrium that the resync layers
        # (cooldown-resync, periodic refit, force-resync,
        # divergence detector) cannot break. K stays disabled
        # until a non-trivial change to the coupling is made
        # (e.g., put K *into* the EKF state vector with proper
        # covariance instead of pre-derotating predict input).
        #
        # The infrastructure (BOOTSTRAP_K_MIN_D_SPREAD_M /
        # _MAX_RAD_PER_M / _EMA_ALPHA constants, 4-tuple
        # gps_history schema, joint-fit return signature,
        # ``_adopt_K`` helper, EKF derotation path, K-adoption
        # sites in resync) stays wired so the next iteration is
        # a localised tweak in this method.
        _ = K_est_raw  # computed but not yet trusted; see above
        return theta_avg, 0.0, max_b

    def _bootstrap_theta(self, min_baseline=1.5, window=None):
        """Joint (θ, K) closed-form fit on accumulated GPS+odom
        pairs. ``window=None`` anchors on the very first sample
        of the whole history; ``window=N`` slides the anchor to
        the oldest sample within the trailing N entries.

        Returns (θ, K, max_baseline). Mirrors robot's
        ``bootstrap_theta`` exactly.
        """
        if len(self.gps_history) < 4:
            return None, 0.0, 0.0
        if window is not None and len(self.gps_history) > window:
            samples = self.gps_history[-window:]
        else:
            samples = self.gps_history
        return self._joint_theta_K_fit(samples, min_baseline)

    def _closed_form_theta_window(self, n_samples, min_baseline=2.0):
        """Joint (θ, K) fit on the trailing ``n_samples`` entries,
        anchored on the oldest of them. Used by every post-
        bootstrap heading-resync mechanism. Returns (θ, K,
        max_baseline). Mirrors robot's ``closed_form_theta_window``.
        """
        if len(self.gps_history) < 4:
            return None, 0.0, 0.0
        if len(self.gps_history) > n_samples:
            samples = self.gps_history[-n_samples:]
        else:
            samples = self.gps_history
        return self._joint_theta_K_fit(samples, min_baseline)

    def _maybe_resync_heading(self):
        """Continuously check the EKF's θ against the closed-form
        GPS-vs-odom fit over a recent window. If they disagree
        significantly, snap θ to the closed-form value and re-widen
        θ_var so the EKF can keep refining. Catches the post-
        bootstrap orbit failure mode: the original closed-form fit
        ran on multipath-biased baselines, locked the EKF onto a
        wrong heading, and every world-frame command since then has
        been rotated by a fixed heading_err — producing the
        characteristic loop around the goal.

        The check itself is cheap (O(window) once per GPS tick).
        Cooldown gates how often a snap can actually fire."""
        if not self.bootstrap_done or self.ekf is None:
            return
        if self.sim_time < self._heading_resync_until:
            return
        bs_theta, bs_K, baseline = self._closed_form_theta_window(
            HEADING_RESYNC_WINDOW, min_baseline=2.0)
        if bs_theta is None:
            return
        if baseline < HEADING_RESYNC_MIN_BASELINE_M:
            return
        diff = (bs_theta - self.ekf.theta + math.pi) % (2 * math.pi) \
                - math.pi
        if abs(diff) > math.radians(HEADING_RESYNC_THRESHOLD_DEG):
            # Real-robot parity, mirrored from
            # gps_handler_node._maybe_resync_heading (lines ~1075-1079).
            # Post-bootstrap we use a confidence-weighted Kalman update
            # rather than a hard snap-replace: as P[2,2] shrinks the
            # gain falls, so a converged θ becomes increasingly
            # resistant to single noisy fits and the candidate goal in
            # map frame actually converges instead of swinging on
            # every resync. The bootstrap path retains reset_theta
            # because there is no accumulated confidence to weigh
            # against during cold start. We still update K_est here
            # so the next predict integrates the right effective
            # rotation (sim-only encoder-yaw-bias state).
            self.ekf.update_theta_measurement(
                bs_theta,
                theta_meas_std=math.radians(5.0))
            self._adopt_K(bs_K)
            self._heading_resync_until = (
                self.sim_time + HEADING_RESYNC_COOLDOWN_S)
            self._heading_resync_count += 1

    def _adopt_K(self, K_fit):
        """EMA-damped adoption of the joint-fit's encoder yaw bias
        rate K. Each resync site goes through this rather than
        slamming ``self.K_est = K_fit`` directly so a single noisy
        fit can shift the state by at most ``α · noise_span``.
        Mirrors gps_handler_node._adopt_k on the robot side."""
        a = BOOTSTRAP_K_EMA_ALPHA
        self.K_est = (1.0 - a) * self.K_est + a * float(K_fit)

    def _force_heading_resync(self):
        """Aggressive heading-resync, called from the moving-away
        branch of `_update_stuck_detector`. Uses a wider window than
        the cooldown-gated standard one (so limit-cycling agents
        eventually accumulate enough motion to fit), but still
        requires a real baseline (`MIN_BASELINE_M`) so noisy fits on
        sub-meter motion don't snap to wrong values. And only acts
        when the new fit disagrees with the current EKF θ by more
        than `DIFF_DEG` — avoids injecting noise when the agent's
        heading is already roughly correct but the limit cycle is
        physical (controller / obstacle bound). Estimator-side only;
        no controller side effects. Returns True iff EKF θ was
        updated."""
        if (self.ekf is None
                or not HEADING_FORCE_RESYNC_ENABLE
                or not GPS_HEADING_EKF_ENABLE):
            return False
        bs_theta, bs_K, baseline = self._closed_form_theta_window(
            HEADING_FORCE_RESYNC_WINDOW,
            min_baseline=HEADING_FORCE_RESYNC_MIN_BASELINE_M)
        if bs_theta is None:
            return False
        diff = ((bs_theta - self.ekf.theta + math.pi)
                % (2 * math.pi) - math.pi)
        if abs(diff) < math.radians(HEADING_FORCE_RESYNC_DIFF_DEG):
            return False
        self.ekf.reset_theta(
            bs_theta,
            theta_var=math.radians(HEADING_FORCE_RESYNC_VAR_DEG) ** 2)
        self._adopt_K(bs_K)
        # Allow the standard resync to fire on the next GPS tick too
        # (don't extend its cooldown — we want fast convergence).
        self._heading_resync_count += 1
        return True

    def _periodic_heading_refit(self):
        """Real-robot parity, mirrored from
        gps_handler_node._periodic_heading_refit. Fires every
        PERIODIC_REFIT_PERIOD_S regardless of detector state or
        cooldown. Refits θ from the standard resync window if the
        closed-form fit disagrees with the EKF by more than
        PERIODIC_REFIT_THRESHOLD_DEG. Catches small persistent
        biases (5-10°) that fall below the moving-away (1 m / 3 s)
        and local-vs-world divergence (5 m) thresholds yet still
        cause meters of cross-track error over a long goal — the
        exact regime that lets the antenna lever-arm corrupt the
        candidate-goal convergence.

        Distinct from `_maybe_resync_heading` (cooldown-gated, runs
        on every GPS tick) and `_force_heading_resync` (detector-
        driven, wider window, larger threshold).
        """
        if not PERIODIC_REFIT_ENABLE or not GPS_HEADING_EKF_ENABLE:
            return
        if not self.bootstrap_done or self.ekf is None:
            return
        if (self.sim_time - self._last_periodic_refit_s
                < PERIODIC_REFIT_PERIOD_S):
            return
        self._last_periodic_refit_s = self.sim_time
        bs_theta, bs_K, baseline = self._closed_form_theta_window(
            HEADING_RESYNC_WINDOW,
            min_baseline=PERIODIC_REFIT_MIN_BASELINE_M)
        if bs_theta is None or baseline < PERIODIC_REFIT_MIN_BASELINE_M:
            return
        diff = ((bs_theta - self.ekf.theta + math.pi)
                % (2 * math.pi) - math.pi)
        if abs(diff) <= math.radians(PERIODIC_REFIT_THRESHOLD_DEG):
            return
        # Real-robot parity, mirrored from
        # gps_handler_node._periodic_heading_refit (lines ~1118-1122).
        # Same A/B as _maybe_resync_heading: Kalman update post-
        # bootstrap so periodic refits respect accumulated confidence;
        # the early-return at the top of this method already guards
        # against the pre-bootstrap case (where reset_theta would be
        # the right call).
        self.ekf.update_theta_measurement(
            bs_theta,
            theta_meas_std=math.radians(PERIODIC_REFIT_VAR_DEG))
        self._adopt_K(bs_K)
        self._heading_resync_count += 1

    def _update_local_world_divergence(self):
        """Real-robot parity, mirrored from
        gps_handler_node._update_local_world_divergence. Cross-
        check EKF-believed distance-to-goal against raw-GPS
        distance-to-goal. Both should decrease together as the
        robot makes real progress; if EKF distance shrinks far
        ahead of GPS distance, θ is wrong and the candidate is
        converging on a phantom goal — the failure mode the
        moving-away (radial) test misses for tangential drift.

        Forces a heading resync when the divergence crosses
        LOCAL_VS_WORLD_DIVERGENCE_M and at least
        LOCAL_VS_WORLD_MIN_LOCAL_PROGRESS_M of EKF progress has
        accumulated. Cooldowns and resets the per-goal baselines
        so we don't fire repeatedly during a single divergence
        event. Independent of bootstrap state (raw GPS distance
        doesn't depend on the EKF).
        """
        if not LOCAL_VS_WORLD_DETECTOR_ENABLE or not GPS_HEADING_EKF_ENABLE:
            return
        if self.ekf is None:
            return
        if self.sim_time < self._divergence_cooldown_until:
            return
        gps = self.latest_gps()
        if gps is None:
            return
        gx, gy = self.goal_world
        ex, ey = self.ekf.pos_xy
        local_d = math.hypot(gx - ex, gy - ey)
        world_d = math.hypot(gps[0] - gx, gps[1] - gy)

        # Lazy-init the per-goal baselines on the first sample
        # that has both a valid EKF position and a fresh GPS fix —
        # mirrors gps_handler_node's `if active.local_d_start is
        # None` path.
        if self._local_d_start is None or self._world_d_start is None:
            self._local_d_start = local_d
            self._world_d_start = world_d
            return

        local_progress = self._local_d_start - local_d
        world_progress = self._world_d_start - world_d
        if local_progress < LOCAL_VS_WORLD_MIN_LOCAL_PROGRESS_M:
            return
        divergence = local_progress - world_progress
        if divergence > LOCAL_VS_WORLD_DIVERGENCE_M:
            # Same envelope-suspension side effect as moving-away
            # so the corrected candidate can flow through the 1/r
            # filter once the heading snaps.
            self._envelope_suspended_until_s = (
                self.sim_time + MOVING_AWAY_ENV_SUSPEND_S)
            self._force_heading_resync()
            self._divergence_event_count += 1
            self._divergence_cooldown_until = (
                self.sim_time + LOCAL_VS_WORLD_COOLDOWN_S)
            # Reset baselines so the next evaluation measures from
            # post-resync state, not the now-stale pre-correction
            # one — same as the robot.
            self._local_d_start = local_d
            self._world_d_start = world_d

    # -- Goal projections / display helpers ------------------------------
    def latest_gps(self):
        # gps_history entry shape is (stamp_s, gps_xy, odom_xy);
        # element 1 is the GPS reading.
        return (self.gps_history[-1][1] if self.gps_history
                else self.true_pos)

    @property
    def heading_offset_est(self):
        return self.ekf.theta if self.ekf is not None else 0.0

    @property
    def body_heading_world(self):
        """Body's TRUE orientation in world frame — true body
        heading (in true-odom) plus the constant rotation between
        odom and world. Used by the visualizer to draw the knife-
        edge arrow correctly even when the agent is stationary or
        rotating in place. With `ODOM_YAW_BIAS_ENABLE = True`
        this differs from `body_heading_world_est` by both the
        EKF's residual θ error AND the accumulated encoder drift,
        which is the whole story of the wrong-convergence."""
        return self.body_heading_true + self.true_heading

    @property
    def body_heading_world_est(self):
        """The agent's belief about its body's world-frame heading.
        Differs from `body_heading_world` by the EKF's residual
        θ error plus the encoder drift accumulated since startup
        (the latter is a real-robot-only effect that the agent's
        sensors literally cannot observe directly)."""
        return self.body_heading + self.heading_offset_est

    @property
    def heading_confidence(self):
        if self.ekf is None:
            return 0.0
        # Map theta std (rad) to a 0..1 confidence — caps at 60° → 0,
        # tightens to 1 below ~3°. Visualization sugar only.
        s = self.ekf.theta_std_rad
        s_high = math.radians(60)
        s_low  = math.radians(3)
        if s >= s_high:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - (s - s_low) / (s_high - s_low))))

    @property
    def debug(self) -> AgentDebugView:
        """Snapshot view of (what the agent knows, what we know).
        Use this to bug-fix specific stuck/oscillating agents:

            print(agents[31].debug)

        The two halves are deliberately separated so it's clear which
        signals the onboard code can react to vs. which are simulator
        ground truth. Cheap to call; allocates a couple of dataclass
        instances per invocation."""
        # Tuple shape (stamp_s, gps_xy, odom_xy): element 1 is the
        # GPS reading.
        last_gps = (self.gps_history[-1][1]
                     if self.gps_history else None)
        last_gps_latlon = (meters_to_latlon(*last_gps)
                            if last_gps is not None else None)
        ekf_x, ekf_y = (self.ekf.pos_xy if self.ekf is not None
                         else (0.0, 0.0))
        ekf_theta = self.ekf.theta if self.ekf is not None else 0.0
        ekf_theta_std = (self.ekf.theta_std_rad
                          if self.ekf is not None else 0.0)
        path_len = (len(self.path_world)
                     if self.path_world is not None else 0)
        cg = self.intermediate_goal_world()
        sv = AgentSelfView(
            odom_xy=(self.odom[0], self.odom[1]),
            odom_vel=(self.odom_vel[0], self.odom_vel[1]),
            last_gps_world_xy=tuple(last_gps) if last_gps else None,
            last_gps_latlon=last_gps_latlon,
            gps_connected=self.gps_connected,
            gps_reconnect_active=self.gps_reconnect_active,
            gps_dropout_active=self._random_dropout_active,
            ekf_pos=(ekf_x, ekf_y),
            ekf_theta_deg=math.degrees(ekf_theta),
            ekf_theta_std_deg=math.degrees(ekf_theta_std),
            bootstrap_done=self.bootstrap_done,
            candidate_goal_world=tuple(cg),
            path_len=path_len,
            best_i=self._best_i,
            last_pad_m=float(self.last_pad),
            refinement_locked=bool(self.refinement_locked),
        )
        dist = math.hypot(self.true_pos[0] - self.goal_world[0],
                          self.true_pos[1] - self.goal_world[1])
        hd_err = math.degrees(
            (self.true_heading - ekf_theta + math.pi)
            % (2 * math.pi) - math.pi)
        ekf_err = math.hypot(ekf_x - self.true_pos[0],
                              ekf_y - self.true_pos[1])
        no_path = (self.path_world is None
                    or len(self.path_world) < 2)
        in_backoff = self.sim_time < self._stuck_until
        replan_age = max(0.0, self.sim_time - self._last_replan_time)
        tv = AgentTrueView(
            pos_xy=(self.true_pos[0], self.true_pos[1]),
            heading_deg=math.degrees(self.true_heading),
            goal_xy=tuple(self.goal_world),
            dist_to_goal_m=dist,
            heading_err_deg=hd_err,
            ekf_pos_err_m=ekf_err,
            arrived=self.arrived,
            coasting=self.coasting,
            sim_time_s=self.sim_time,
            steps=self.steps,
            no_path=no_path,
            in_stuck_backoff=in_backoff,
            last_replan_age_s=replan_age,
            heading_resync_count=self._heading_resync_count,
        )
        return AgentDebugView(self_view=sv, true_view=tv)

    def goal_overlap_fraction(self):
        """Fraction of the robot's footprint area that lies inside the
        goal circle. The competition rule is "≥ 50 % counts as passing
        the waypoint", which on our geometry corresponds to the robot
        center within GOAL_RADIUS — but reporting the explicit overlap
        makes the margin visible during runs where the EKF is mid-
        convergence and noise could push the robot back over the line."""
        d = math.hypot(self.true_pos[0] - self.goal_world[0],
                       self.true_pos[1] - self.goal_world[1])
        R = GOAL_RADIUS
        r = ROBOT_RADIUS
        if d >= R + r:
            return 0.0
        if d + r <= R:
            return 1.0
        # Two-circle lens area, robust to grazing geometry.
        d2 = d * d; r2 = r * r; R2 = R * R
        a = max(-1.0, min(1.0, (d2 + r2 - R2) / (2 * d * r)))
        b = max(-1.0, min(1.0, (d2 + R2 - r2) / (2 * d * R)))
        sq = max(0.0, (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
        A = r2 * math.acos(a) + R2 * math.acos(b) - 0.5 * math.sqrt(sq)
        return A / (math.pi * r2)

    def _compute_raw_candidate(self):
        """Raw candidate-goal projection. Rotates (goal − ekf_pos) by
        the residual heading error around the EKF's current position
        estimate. The EKF anchor is intentional: in steady state the
        fixed-point of `candidate = ekf + R(ε)(goal − ekf)` is
        `ekf = goal`, so the candidate acts as a moving carrot that
        pulls the robot to the true goal regardless of any small
        residual heading bias. A spawn-fixed anchor would be
        geometrically the agent's odom-frame projection but would
        leave the robot parked at `R(ε)(goal − spawn) + spawn` — i.e.
        with `ε × lever_arm` of steady-state offset, breaking Rule 4
        for long lever arms.

        Field-parity bypass (GPS_HEADING_EKF_ENABLE=False):
          When the GPS-heading EKF isn't running, gps_handler_node
          publishes /goal_pose at ``goal_world`` in MAP frame (its
          world→odom→map projection collapses to identity at
          theta_ekf=0). The MAP frame is rotated relative to the
          GPS / ENU world frame by ``true_heading`` — map is
          aligned with the body's startup axis (= odom), and that
          axis sits at ``true_heading`` in world. So the
          candidate's WORLD-frame position (what the GUI's yellow
          X actually plots) is  R(true_heading) · goal_world . The
          agent's true world position when ``self.odom ==
          goal_world`` lands at approximately the same point, so
          in the visualization the candidate ends up overlapping
          the red robot dot — exactly like the field GUI showed.
        """
        if not GPS_HEADING_EKF_ENABLE:
            c = math.cos(self.true_heading)
            s = math.sin(self.true_heading)
            gx, gy = self.goal_world
            return (c * gx - s * gy, s * gx + c * gy)
        if self.ekf is None:
            ex, ey = self.latest_gps()
        else:
            ex, ey = self.ekf.pos_xy
        vx = self.goal_world[0] - ex
        vy = self.goal_world[1] - ey
        dtheta = self.true_heading - self.heading_offset_est
        c = math.cos(dtheta); s = math.sin(dtheta)
        return (ex + c * vx - s * vy, ey + s * vx + c * vy)

    def _compute_raw_candidate_odom_frame(self):
        """Robot-faithful candidate computation — mirrors
        gps_handler_node._compute_raw_candidate exactly.

        Returns the published goal in ODOM frame, not world. Math:
            candidate_odom = last_odom + R(-θ_ekf) · (goal_w − ekf_pos)

        The agent's NAV2 / controller-equivalent path drives
        ``self.odom`` (= last_odom on the next tick) toward this
        odom-frame target. Unlike ``_compute_raw_candidate`` (which
        uses ``self.true_heading`` for visualisation purposes), this
        method does NOT touch any truth-side knowledge — its inputs
        are the same as the deployed robot's: ``ekf.pos_xy``,
        ``ekf.theta``, ``self.odom``, ``self.goal_world``. So any
        change to this function ports to gps_handler_node line-for-
        line.

        The current sim controller still consumes
        ``intermediate_goal_world()`` (truth-aware) for the GUI,
        but algorithmic improvements that affect the published goal
        should be developed against this method first.
        """
        if self.ekf is None:
            ex, ey = (0.0, 0.0)
        else:
            ex, ey = self.ekf.pos_xy
        gx_w, gy_w = self.goal_world
        dx_w = gx_w - ex
        dy_w = gy_w - ey
        theta = self.heading_offset_est  # = ekf.theta or 0 if no EKF
        c = math.cos(-theta); s = math.sin(-theta)
        dx_o = c * dx_w - s * dy_w
        dy_o = s * dx_w + c * dy_w
        ox, oy = self.odom[0], self.odom[1]
        return (ox + dx_o, oy + dy_o)

    def intermediate_goal_world(self):
        """Filtered candidate goal in WORLD frame — what consumers
        (planner, controller, viz) should target. Returns the
        EWMA-smoothed candidate, lifted from odom frame to world frame
        via the EKF's belief about the odom↔world rotation
        (``θ_ekf`` = ``heading_offset_est``). The smoother runs once
        per ``step()`` tick (see ``_update_candidate_smoother``);
        calling this between ticks is idempotent.

        Frame plumbing (improve/gps-waypoint-continuity port — Fix 2):
        ``self._smoothed_candidate`` is stored in odom frame (matches
        deployed ``gps_handler_node._smoothed_candidate`` which is also
        odom-frame; the deployed publisher transforms odom→map via TF
        before publishing). In the sim there is no separate map frame;
        the planner / A* / GUI all operate in WORLD frame. The
        inverse of ``_compute_raw_candidate_odom_frame`` is:

            candidate_world = ekf_pos + R(θ_ekf)·(smoothed_odom − odom_xy)

        which collapses to the goal world when θ_ekf is converged AND
        the EKF position tracks truth — i.e. as the EKF refines, the
        published world-frame goal converges to ``self.goal_world``.
        With imperfect θ_ekf the candidate is offset by the residual
        heading error, exactly mirroring the deployed map-frame
        candidate."""
        if self._smoothed_candidate is None:
            # Cold start — return a world-frame raw candidate so
            # downstream code never sees odom-frame coords during
            # the first tick before the smoother latches.
            return self._compute_raw_candidate()
        sx_o, sy_o = self._smoothed_candidate
        ox, oy = self.odom[0], self.odom[1]
        dox = sx_o - ox
        doy = sy_o - oy
        if self.ekf is not None:
            theta = self.heading_offset_est  # = ekf.theta
            ex, ey = self.ekf.pos_xy
        else:
            theta = 0.0
            ex, ey = self.latest_gps()
        c = math.cos(theta); s = math.sin(theta)
        dwx = c * dox - s * doy
        dwy = s * dox + c * doy
        return (ex + dwx, ey + dwy)

    def _update_candidate_smoother(self):
        """Per-tick EWMA update with a 1/r-envelope outlier reject.
        Called from `step()` exactly once.

        Two filters compose:
          1. Envelope filter — d_raw = ‖raw − goal‖ must stay inside
             `K · max(floor, GAIN · L / r)`, where r = odom travel,
             L = robot→goal, both known to the agent. A candidate
             that escapes this envelope is treated as an outlier and
             dropped (we keep the previous smoothed value).
          2. EWMA + snap — accepted samples below `CANDIDATE_SNAP_M`
             from the smoothed state EWMA-track; samples above SNAP
             pass through verbatim (heading resync, big A* re-target).

        The envelope is dormant for r < `CANDIDATE_ENV_MIN_R_M` so
        the bootstrap → resync drop isn't gated.

        Truth-independence (improve/gps-waypoint-continuity port):
        Feeds from ``_compute_raw_candidate_odom_frame`` — the
        deployed-equivalent projection that uses only EKF state
        (``ekf.pos_xy``, ``ekf.theta``) and ``self.odom``. The
        world-frame helper ``_compute_raw_candidate`` is retained for
        GUI / viz consumers but is no longer wired into the smoother
        because it reads ``self.true_heading`` and would leak truth
        into the convergence signal. Mirrors deployed
        ``gps_handler_node._update_candidate_smoother`` (L1299-1309)
        which calls ``self._compute_raw_candidate()`` → which itself
        calls ``self._project_world_to_odom(active.goal_world_xy)``."""
        raw = self._compute_raw_candidate_odom_frame()

        # ── envelope filter ──────────────────────────────────────
        # Suspend if the moving-away detector recently fired — the
        # agent is getting *farther* from the GPS goal, which means
        # the EKF heading lock the envelope is implicitly trusting
        # is wrong. Letting raw candidates through allows reconverge.
        envelope_active = (
            CANDIDATE_ENV_ENABLE
            and self._smoothed_candidate is not None
            and self.sim_time >= self._envelope_suspended_until_s)
        if envelope_active:
            r = math.hypot(self.odom[0], self.odom[1])
            if r > CANDIDATE_ENV_MIN_R_M:
                if self.ekf is not None:
                    rx, ry = self.ekf.pos_xy
                else:
                    rx, ry = self.latest_gps()
                gx, gy = self.goal_world
                # Lever arm — frame-invariant under the rigid
                # map↔odom transform; computed in world frame
                # against the active goal (matches deployed
                # ``_update_candidate_smoother`` L1320-1328).
                lever = math.hypot(rx - gx, ry - gy)
                d_env = max(CANDIDATE_ENV_FLOOR_M,
                            CANDIDATE_ENV_GAIN_M * lever / r)
                # Compare ``raw`` (odom-frame) to the current
                # smoothed candidate (also odom-frame) so the
                # check stays in-frame. Mirrors deployed L1335-1336.
                sx_prev, sy_prev = self._smoothed_candidate
                d_raw = math.hypot(raw[0] - sx_prev, raw[1] - sy_prev)
                if d_raw > CANDIDATE_ENV_REJECT_K * d_env:
                    self._cand_reject_count += 1
                    return  # drop this sample; keep last smoothed

        # ── EWMA + snap ──────────────────────────────────────────
        if self._smoothed_candidate is None:
            self._smoothed_candidate = raw
            return
        sx, sy = self._smoothed_candidate
        dx = raw[0] - sx
        dy = raw[1] - sy
        if (dx * dx + dy * dy) > (CANDIDATE_SNAP_M * CANDIDATE_SNAP_M):
            self._smoothed_candidate = raw
        else:
            a = CANDIDATE_SMOOTH_ALPHA
            self._smoothed_candidate = (sx + a * dx, sy + a * dy)

    def _compute_raw_next_hint_candidate(self):
        """Raw candidate-goal projection targeted at
        ``self._next_hint_world_xy`` instead of ``self.goal_world``.
        Same rotation math as ``_compute_raw_candidate``; the only
        difference is the target xy. Returns ``None`` when there's
        no cached hint to project.

        Truth-aware (uses ``self.true_heading``); kept for GUI / viz
        consumers. Algorithmic consumers (the shadow EWMA in
        ``_update_next_hint_smoother``) should call
        ``_compute_raw_next_hint_candidate_odom_frame`` instead so
        the smoothed value is computed from EKF state only — matching
        deployed ``gps_handler_node._update_next_hint_smoother``
        (L1269-1297) which calls
        ``_project_world_to_odom(self._next_hint_world_xy)``.
        """
        if self._next_hint_world_xy is None:
            return None
        if not GPS_HEADING_EKF_ENABLE:
            c = math.cos(self.true_heading)
            s = math.sin(self.true_heading)
            hx, hy = self._next_hint_world_xy
            return (c * hx - s * hy, s * hx + c * hy)
        if self.ekf is None:
            ex, ey = self.latest_gps()
        else:
            ex, ey = self.ekf.pos_xy
        vx = self._next_hint_world_xy[0] - ex
        vy = self._next_hint_world_xy[1] - ey
        dtheta = self.true_heading - self.heading_offset_est
        c = math.cos(dtheta); s = math.sin(dtheta)
        return (ex + c * vx - s * vy, ey + s * vx + c * vy)

    def _compute_raw_next_hint_candidate_odom_frame(self):
        """Odom-frame variant of ``_compute_raw_next_hint_candidate``.
        Mirrors deployed
        ``gps_handler_node._update_next_hint_smoother`` (L1284):
        ``raw = self._project_world_to_odom(self._next_hint_world_xy)``
        — uses only EKF state (``ekf.pos_xy``, ``ekf.theta``) and
        ``self.odom``; no ``true_heading`` leak. Returns ``None`` if
        the hint isn't cached yet."""
        if self._next_hint_world_xy is None:
            return None
        if self.ekf is None:
            ex, ey = (0.0, 0.0)
        else:
            ex, ey = self.ekf.pos_xy
        hx_w, hy_w = self._next_hint_world_xy
        dx_w = hx_w - ex
        dy_w = hy_w - ey
        theta = self.heading_offset_est
        c = math.cos(-theta); s = math.sin(-theta)
        dx_o = c * dx_w - s * dy_w
        dy_o = s * dx_w + c * dy_w
        ox, oy = self.odom[0], self.odom[1]
        return (ox + dx_o, oy + dy_o)

    def _update_next_hint_smoother(self):
        """improve/gps-waypoint-continuity — shadow EWMA on the
        cached next-up waypoint. No envelope filter (the envelope's
        1/r gain keys off the ACTIVE goal's robot-to-goal lever
        arm, which doesn't apply to a future goal); just the same
        EWMA + SNAP shape as the active smoother. Promotion into
        ``self._smoothed_candidate`` happens in
        ``_advance_to_next_leg`` when the cached hint matches the
        new leg's goal within ``_hint_match_tolerance_m``.

        Mirrors deployed gps_handler_node._update_next_hint_smoother
        (L1269-1297)."""
        if not self._next_hint_enabled:
            return
        if self._next_hint_world_xy is None:
            return
        # improve/gps-waypoint-continuity: shadow EWMA consumes the
        # ODOM-frame projection (deployed-equivalent) so the smoothed
        # hint stays in the same frame as ``_smoothed_candidate``
        # (also odom-frame after Fix 2). Promotion at leg switch via
        # ``_advance_to_next_leg`` then transfers the smoothed value
        # frame-consistently. Mirrors deployed L1284.
        raw = self._compute_raw_next_hint_candidate_odom_frame()
        if raw is None:
            return
        if self._next_hint_smoothed_candidate is None:
            self._next_hint_smoothed_candidate = raw
            return
        sx, sy = self._next_hint_smoothed_candidate
        dx = raw[0] - sx
        dy = raw[1] - sy
        if (dx * dx + dy * dy) > (CANDIDATE_SNAP_M * CANDIDATE_SNAP_M):
            self._next_hint_smoothed_candidate = raw
        else:
            a = CANDIDATE_SMOOTH_ALPHA
            self._next_hint_smoothed_candidate = (sx + a * dx, sy + a * dy)

    def _refresh_dist_history(self):
        """Append the current raw-GPS distance-to-goal to the
        rolling history and trim out-of-window samples. Shared by
        the moving-away and stuck detectors so they consume the
        exact same time series. Returns True when the history is
        long enough for either test to fire."""
        if self.arrived:
            return False
        gps = self.latest_gps()
        if gps is None:
            return False
        gx, gy = self.goal_world
        d_goal = math.hypot(gps[0] - gx, gps[1] - gy)
        self._dist_history.append((self.sim_time, d_goal))
        cutoff = self.sim_time - max(STUCK_WINDOW_S, MOVING_AWAY_WINDOW_S)
        while self._dist_history and self._dist_history[0][0] < cutoff:
            self._dist_history.pop(0)
        return len(self._dist_history) >= MOVING_AWAY_MIN_HISTORY_TICKS

    def _oldest_within_window(self, window_s):
        """Oldest dist_history sample whose timestamp lies inside
        [t_new - window_s, t_new]. Returns (t, d) or None."""
        if not self._dist_history:
            return None
        t_new = self._dist_history[-1][0]
        target_t = t_new - window_s
        for t, d in self._dist_history:
            if t >= target_t:
                return t, d
        return None

    def _update_moving_away(self):
        """Estimator-side moving-away detector — mirrors
        gps_handler_node._update_moving_away. Pre-bootstrap trip
        wire: if the agent's RAW GPS distance to the GPS goal is
        increasing by more than ``MOVING_AWAY_THRESHOLD_M`` over
        ``MOVING_AWAY_WINDOW_S``, the heading lock is biased.
        Action: suspend the 1/r envelope filter for
        ``MOVING_AWAY_ENV_SUSPEND_S`` and force a heading resync.
        Uses raw GPS only — independent of the EKF whose belief
        we're trying to validate. Runs pre- and post-bootstrap."""
        if not MOVING_AWAY_DETECTOR_ENABLE:
            return
        if self.sim_time < self._envelope_suspended_until_s:
            return
        if not self._dist_history:
            return
        t_new, d_new = self._dist_history[-1]
        sample = self._oldest_within_window(MOVING_AWAY_WINDOW_S)
        if sample is None:
            return
        if (t_new - sample[0]) < MOVING_AWAY_WINDOW_COVERAGE * MOVING_AWAY_WINDOW_S:
            return
        delta = d_new - sample[1]
        if delta > MOVING_AWAY_THRESHOLD_M:
            self._envelope_suspended_until_s = (
                self.sim_time + MOVING_AWAY_ENV_SUSPEND_S)
            self._moving_away_event_count += 1
            # Wider-window force-resync — `_maybe_resync_heading`
            # often misses these cases because the limit-cycle
            # motion has too little net displacement for its
            # baseline floor.
            self._force_heading_resync()

    def _update_stuck(self):
        """Sim-only stuck-recovery branch — does NOT ship to the
        robot (Rule 7). Patches a pure-pursuit limit cycle in the
        Chaplygin controller: if |progress| toward the goal is
        below STUCK_PROGRESS_M over a full STUCK_WINDOW_S window,
        schedule a forward-thrust override and force replan.
        Bootstrap-gated; the moving-away detector already covers
        the wrong-θ regime that pre-bootstrap stuck would
        misdiagnose."""
        if not STUCK_DETECTOR_ENABLE:
            return
        if not self.bootstrap_done:
            return
        if len(self._dist_history) < STUCK_MIN_HISTORY_TICKS:
            return
        if self.sim_time < self._stuck_recovery_until:
            return
        if not self._dist_history:
            return
        t_new, d_new = self._dist_history[-1]
        sample = self._oldest_within_window(STUCK_WINDOW_S)
        if sample is None:
            return
        if (t_new - sample[0]) < 0.8 * STUCK_WINDOW_S:
            return
        progress = sample[1] - d_new
        if progress < STUCK_PROGRESS_M:
            self._stuck_recovery_until = (
                self.sim_time + STUCK_RECOVERY_S)
            self._stuck_event_count += 1
            self.path_world = None

    def _update_stuck_detector(self):
        """Top-level dispatcher: refresh shared history, then run
        the moving-away (estimator, ships) and stuck (sim-only,
        Rule 7) tests. Kept as a single entry point so step()
        retains its existing call site, but the two detectors are
        now in their own methods so each one can be ported / tuned
        independently. Mirrors the structural split in
        gps_handler_node — moving-away is its own method on the
        robot too."""
        if not self._refresh_dist_history():
            return
        self._update_moving_away()
        self._update_stuck()

    # -- Stepping --------------------------------------------------------
    def _advance_to_next_leg(self):
        """Pop the next ``(lat, lon)`` off ``self.goal_queue`` and
        reseat the agent on the new goal — mirrors the per-leg
        transition the deployed ``gps_handler_node`` performs when
        a new ``NavigateToWaypoint`` goal supersedes the previous
        one on branch ``origin/improve/gps-waypoint-continuity``.

        Cache (preserved across the leg boundary) — the "preemptive
        next-goal cache" + "shadow EWMA chaining" referenced in the
        deployed design:

          * ``self.ekf``           — full EKF state (x, y, θ_offset),
                                     covariance P, ``update_count``,
                                     ``rejected_count``, and
                                     ``consecutive_rejects``. The
                                     heading offset learned by leg 1
                                     applies verbatim to legs 2/3.
          * ``self.gps_history``   — the closed-form fit's sample
                                     window. Deployed handler keeps
                                     ``self._gps_history`` across
                                     legs so periodic refit /
                                     resync can fire immediately on
                                     a new leg.
          * ``self.bootstrap_done``— never re-bootstrap once True
                                     (deployed sets it True from
                                     init; sim flips on first 5 m
                                     of travel and keeps it).
          * ``self._yaw_bias_offset`` /
            ``self.K_est`` /
            ``self.D_total``       — encoder yaw bias state. The
                                     joint (θ, K) fit's calibration
                                     survives.
          * ``self._datum``-equivalents (``LAT_CENTER`` /
            ``LON_CENTER``)        — implicit (module-scoped); no
                                     re-survey on leg switch.
          * ``self._heading_resync_count`` /
            ``self._moving_away_event_count`` /
            ``self._divergence_event_count`` /
            ``self._cand_reject_count``
                                   — diagnostic counters keep
                                     accumulating across legs.

        Reset on each leg start (the "per-leg baselines") — matches
        the deployed handler's leg-switch path
        (``gps_handler_node._execute_callback`` lines ~1753-1855):

          * ``self.goal_world``           — replaced with the new leg.
          * ``self.published_goal_world`` — re-seeded to the new goal
                                            so the candidate-goal
                                            pipeline re-emits.
          * ``self.refinement_locked``    — False; new goal, new
                                            convergence bubble.
          * ``self._smoothed_candidate``  — None; deployed handler
                                            clears it on every new
                                            active goal.
          * ``self._dist_history``        — cleared so moving-away
                                            measures from
                                            post-switch state, not
                                            stale pre-switch
                                            distances.
          * ``self._envelope_suspended_until_s`` — 0 (cleared baseline).
          * ``self._local_d_start`` /
            ``self._world_d_start``       — None (lazy-init on next
                                            odom tick with both EKF
                                            pos + GPS available;
                                            mirrors deployed
                                            ``_ActiveGoal.local_d_start``
                                            / ``world_d_start``).
          * ``self._divergence_cooldown_until`` — -1 (fresh baseline
                                            window).
          * ``self._stuck_recovery_until`` — -1 (the new leg's
                                            stuck-detector starts
                                            from scratch).
          * ``self.arrived``              — False (continue running).
          * ``self.coasting``             — False (drive again).
          * ``self.robot_declared_success`` etc. — cleared so the
                                            next leg's onboard
                                            arrival fires fresh.
          * ``self.path_world`` /
            ``self.last_planned_goal``    — None; force an A* replan
                                            against the new goal on
                                            the next tick.
          * ``self._leg_start_step`` /
            ``self._leg_start_sim_time``  — stamped to now so the
                                            snapshot classifier and
                                            per-leg timing telemetry
                                            measure from leg start.

        Returns True if a new leg was popped, False if the queue
        was empty (end-of-mission)."""
        if not self.goal_queue:
            return False
        next_lat, next_lon = self.goal_queue.pop(0)
        self.goal_world = latlon_to_meters(next_lat, next_lon)
        self.leg_index += 1
        # improve/gps-waypoint-continuity — promotion check.
        # Mirrors deployed L2074-2094: if the cached hint matches
        # the new leg's goal within ``_hint_match_tolerance_m``,
        # promote the shadow EWMA's current value into the active
        # smoother (warm start). Otherwise leave the active
        # smoother as None (cold start — bit-identical to the
        # pre-port reset path). Either way, clear the hint cache
        # so a stale shadow can't accidentally satisfy the match
        # check on the next leg switch.
        warm = None
        if (self._next_hint_enabled
                and self._next_hint_world_xy is not None
                and self._next_hint_smoothed_candidate is not None):
            dx = self.goal_world[0] - self._next_hint_world_xy[0]
            dy = self.goal_world[1] - self._next_hint_world_xy[1]
            hint_d = math.hypot(dx, dy)
            if hint_d < self._hint_match_tolerance_m:
                warm = self._next_hint_smoothed_candidate
                self._next_hint_warm_start_count += 1
        # ── Per-leg baselines (RESET) ────────────────────────────
        self.arrived = False
        self.coasting = False
        self.robot_declared_success = False
        self.robot_declared_at_step = None
        self.robot_declared_truth_xy = None
        self.refinement_locked = False
        self._smoothed_candidate = warm
        # Clear the hint cache after the match check (promoted or
        # not). Mirrors deployed L2092-2094.
        self._next_hint_lat_lon = None
        self._next_hint_world_xy = None
        self._next_hint_smoothed_candidate = None
        self._dist_history = []
        self._envelope_suspended_until_s = 0.0
        self._local_d_start = None
        self._world_d_start = None
        self._divergence_cooldown_until = -1.0
        self._stuck_recovery_until = -1.0
        # Re-seed ``published_goal_world`` to the new goal so the
        # candidate-goal pipeline re-emits cleanly on the next tick
        # (deployed parlance: the first publish of a new leg goes
        # to /goal_pose; subsequent in-mission updates flow through
        # /goal_update).
        self.published_goal_world = tuple(self.goal_world)
        # Force a planner replan against the new goal.
        self.path_world = None
        self.last_planned_goal = None
        self._best_i = 0
        # Per-leg timing baselines.
        self._leg_start_step = self.steps
        self._leg_start_sim_time = self.sim_time
        # improve/gps-waypoint-continuity — seed the next-hint cache
        # from the NEW head of ``goal_queue`` so the shadow EWMA
        # starts converging on the leg-after-this immediately. On
        # the final leg the queue is empty after the pop above, and
        # ``_seed_next_hint_from_queue`` no-ops — no further hints.
        self._seed_next_hint_from_queue()
        return True

    def step(self, dt=SIM_DT):
        """One physics tick. Returns True while still navigating."""
        # ── Mission auto-advance ─────────────────────────────────
        # If we arrived on a prior leg but still have queued
        # waypoints, pop the next one and continue. The leg-switch
        # transition preserves the EKF / heading-fit cache and
        # resets only the per-leg baselines (see
        # ``_advance_to_next_leg`` for the full cache-vs-reset
        # ledger, which mirrors the deployed
        # ``gps_handler_node`` behavior on branch
        # ``origin/improve/gps-waypoint-continuity``).
        if self.arrived and self.goal_queue:
            self._advance_to_next_leg()
        if self.arrived and not self.coasting:
            return False
        self.steps += 1
        self.sim_time += dt

        # ── Sensor housekeeping ─────────────────────────────────────
        # Maybe trigger a GPS dropout / cycle slip / noise burst.
        self._maybe_start_dropout(dt)
        self._maybe_start_cycle_slip(dt)
        self._maybe_start_noise_burst(dt)
        # Sample GPS at GPS_PERIOD; the EKF coasts on prediction in between.
        self._gps_acc_time += dt
        new_gps = None
        if self._gps_acc_time >= GPS_PERIOD:
            self._gps_acc_time = 0.0
            new_gps = self._tick_gps()      # may be None during dropout

        # ── Replan only when needed. Obstacles are static, but the
        # candidate goal moves as the heading estimate refines, so we
        # also replan when that target drifts. The closest-point
        # lookahead handles smaller anchor drift without a full
        # re-search.
        plan_anchor = (self.ekf.pos_xy if self.ekf is not None
                       else self.latest_gps())
        # Live candidate (computed every step from the EKF — free).
        # First update the EWMA smoother with this tick's raw value,
        # then read the (possibly smoothed) candidate via the public
        # accessor. The smoother snaps through large steps so heading
        # resyncs propagate to the planner immediately while small
        # GPS-noise wiggles get filtered.
        self._update_candidate_smoother()
        # Shadow EWMA on the cached next-goal hint (no-op in
        # single-goal mode). Mirrors deployed L924.
        self._update_next_hint_smoother()
        # Anti-spin recovery — reads `bootstrap_done`, `arrived`, and
        # the EKF position estimate. Must run after the EKF / heading
        # bookkeeping (which `_tick_gps` did above) and before the
        # controller, so any recovery override is in effect this tick.
        self._update_stuck_detector()
        # Real-robot parity: periodic θ refit (3 s timer) + local-vs-
        # world divergence detector (catches tangential drift the
        # radial moving-away test misses). Both are estimator-side
        # and ship to the robot — running them in the same place in
        # the tick keeps detector ordering consistent with
        # gps_handler_node._odom_callback.
        self._periodic_heading_refit()
        self._update_local_world_divergence()
        live_candidate = self.intermediate_goal_world()
        # Sample the live candidate into `published_goal_world` at
        # NAV2_GOAL_HZ. Everything downstream — A*, replan trigger,
        # controller — drives toward this published goal so the sim
        # mirrors what a real NAV2 stack would actually receive.
        publish_period = 1.0 / max(NAV2_GOAL_HZ, 1e-3)
        if (self.sim_time - self._last_published_time) >= publish_period:
            # Real-robot parity, mirrored from
            # gps_handler_node._publish_goal (L1574-1577) on branch
            # ``origin/improve/gps-waypoint-continuity``.
            # ``refinement_locked`` is recomputed FRESH every publish
            # tick — not latched. A noisy GPS update that briefly dips
            # the EKF inside the STOP_REFINE bubble does NOT
            # permanently freeze the published goal; if the EKF
            # wanders back outside, publishing resumes. This matches
            # the deployed gate which is a per-tick comparison, not a
            # one-shot latch. The boolean attribute is preserved for
            # the status panel — only its previous-frame value is no
            # longer consulted as the gate.
            if self.ekf is not None:
                ex, ey = self.ekf.pos_xy
                d_goal = math.hypot(ex - self.goal_world[0],
                                    ey - self.goal_world[1])
                threshold = STOP_REFINE_K * STOP_REFINE_SIGMA_GPS_M
                self.refinement_locked = d_goal < threshold
                if not self.refinement_locked:
                    self.published_goal_world = live_candidate
            else:
                self.published_goal_world = live_candidate
            self._last_published_time = self.sim_time
        candidate_goal = self.published_goal_world
        path = self.path_world
        need_replan = (path is None or len(path) < 2)
        # Cheap drift check that re-uses the cached lookahead index so
        # we don't re-scan the entire path every step. Just looks at the
        # current anchor cell vs the cached "best" point.
        if not need_replan and not self.coasting:
            i = min(self._best_i, len(path) - 1)
            p = path[i]
            closest = math.hypot(p[0] - plan_anchor[0],
                                  p[1] - plan_anchor[1])
            if closest > REPLAN_PATH_DRIFT_M:
                need_replan = True
        if (not need_replan and not self.coasting
                and self.last_planned_goal is not None):
            # During bootstrap the candidate goal swings wildly while
            # the heading estimate firms up — using a generous threshold
            # keeps that swing from causing thrash, but we must NOT
            # gate this entirely on bootstrap_done. The first guess at
            # candidate_goal can land off-map (heading_offset_est = 0
            # rotates the goal by the full true_heading). If we don't
            # replan, the agent commits to driving toward an off-map
            # point and never recovers. Threshold scales with whether
            # the heading has converged.
            thr = (REPLAN_GOAL_DRIFT_M if self.bootstrap_done
                    else REPLAN_GOAL_DRIFT_M_BOOT)
            cg_drift = math.hypot(
                candidate_goal[0] - self.last_planned_goal[0],
                candidate_goal[1] - self.last_planned_goal[1])
            if cg_drift > thr:
                need_replan = True
        # Min-interval gate: don't replan more than once per
        # REPLAN_MIN_INTERVAL_S unless we genuinely have no path. This
        # is the throttle that makes 1000-agent runs viable.
        if (need_replan
                and self.path_world is not None and len(self.path_world) >= 2
                and (self.sim_time - self._last_replan_time)
                     < REPLAN_MIN_INTERVAL_S):
            need_replan = False
        # If a previous replan failed, back off briefly to keep failed
        # agents from hammering A* every tick.
        if need_replan and self.sim_time < self._stuck_until:
            need_replan = False
        if need_replan and not self.coasting:
            path_new, pad, win = windowed_astar(
                self.cm, plan_anchor[0], plan_anchor[1],
                candidate_goal[0], candidate_goal[1],
                pad=ASTAR_INITIAL_PAD)
            self._last_replan_time = self.sim_time
            if path_new is not None:
                self.path_world = path_new
                self.last_window = win
                self.last_pad = pad
                self.last_planned_goal = candidate_goal
                self._best_i = 0
                self._stuck_until = -1.0
            else:
                # Preserve any stale path; back off for a bit so we don't
                # spin on an unreachable goal.
                self._stuck_until = self.sim_time + 1.0

        # ── Compute desired velocity in world frame (controller) ───
        path = self.path_world
        if not self.coasting and path is not None and len(path) >= 2:
            # Walk forward from the cached anchor index, tracking the
            # best (closest) point. Once the distance starts growing
            # again we've passed the projection — stop. This is O(k)
            # per step where k is the segment count we advance, not
            # O(N_path). For 1000 agents on long paths that's the
            # difference between "snappy" and "frame-drop city".
            n = len(path)
            i = max(0, min(self._best_i, n - 1))
            p = path[i]
            best_d = math.hypot(p[0] - plan_anchor[0],
                                p[1] - plan_anchor[1])
            best_i = i
            j = i + 1
            while j < n:
                p = path[j]
                d = math.hypot(p[0] - plan_anchor[0],
                                p[1] - plan_anchor[1])
                if d < best_d:
                    best_d = d
                    best_i = j
                else:
                    break
                j += 1
            self._best_i = best_i
            target = path[-1]
            walked = 0.0
            for i in range(best_i, n - 1):
                walked += math.hypot(
                    path[i + 1][0] - path[i][0],
                    path[i + 1][1] - path[i][1])
                if walked >= LOOKAHEAD:
                    target = path[i + 1]
                    break
            # Once we've consumed the path, drive at the *live*
            # candidate goal instead of `path[-1]` (which was planned
            # against an older candidate). Without this, the robot
            # parks on a stale path-end while `dist_to_goal` is still
            # multi-metres → speed setpoint stays high but the
            # direction vector collapses to noise → limit-cycle
            # oscillation around the planned end-point.
            if best_i >= n - 1:
                target = candidate_goal
            dx_w = target[0] - plan_anchor[0]
            dy_w = target[1] - plan_anchor[1]
            d = math.hypot(dx_w, dy_w)
            # Slow down as we approach the final goal (P on distance).
            # Distance is measured against the candidate goal A* targeted
            # so the controller stays consistent with the planner.
            dist_to_goal = math.hypot(
                candidate_goal[0] - plan_anchor[0],
                candidate_goal[1] - plan_anchor[1])
            # Floor speed at MIN_SEARCH_SPEED *during bootstrap* so a
            # bad initial heading still produces enough motion to fit
            # heading_offset_est. After bootstrap, drop the floor: the
            # P-control on dist_to_goal tapers naturally near the goal,
            # which is what stops the limit-cycle oscillation we used
            # to see when EKF position bias kept dist_to_goal at zero
            # while the true robot orbited just outside the success
            # circle. Rule 5 forbids projectors near goals, so the
            # near-goal EKF-bias case the floor was protecting against
            # no longer happens.
            floor = (0.0 if (self.arrived or self.bootstrap_done)
                      else MIN_SEARCH_SPEED)
            # Anti-spin: while the stuck-detector's recovery window
            # is active, override the post-bootstrap zero floor with a
            # positive minimum. This guarantees forward translation
            # even if the alignment gate (`align = cos(heading_err)`)
            # would otherwise zero out v_des — which is exactly the
            # condition that produces the limit-cycle around the
            # candidate goal.
            if self.sim_time < self._stuck_recovery_until and not self.arrived:
                floor = max(floor, STUCK_RECOVERY_SPEED_MPS)
            # Cap speed during bootstrap so a wildly-wrong initial
            # heading can't carry the agent off the map before the
            # closed-form fit converges.
            cap = MAX_SPEED_MPS if self.bootstrap_done else 1.0
            speed_setpoint = min(cap,
                                 max(floor, dist_to_goal * SPEED_GAIN))
            # Direction to target *in odom frame* using the agent's
            # current heading-offset estimate. The body has to rotate
            # to face this direction before forward thrust is useful.
            cR = math.cos(-self.heading_offset_est)
            sR = math.sin(-self.heading_offset_est)
            dx_o = cR * dx_w - sR * dy_w
            dy_o = sR * dx_w + cR * dy_w
            target_angle_odom = math.atan2(dy_o, dx_o) if d > 1e-6 \
                else self.body_heading
        else:
            target_angle_odom = self.body_heading
            dist_to_goal = 0.0
            floor = 0.0
            cap = 0.0

        # ── Chaplygin sleigh control ──────────────────────────────
        # Heading error in odom: how far the body must rotate to point
        # at the target direction. Wrap to [-π, π].
        heading_err = (target_angle_odom - self.body_heading
                        + math.pi) % (2 * math.pi) - math.pi
        # Forward speed setpoint: scaled by alignment so the body
        # doesn't drive forward when the target is behind it. Pure
        # pursuit standard. (max(0, …) means no-reverse.)
        align = math.cos(heading_err)
        v_des = max(floor,
                     max(0.0, dist_to_goal * SPEED_GAIN * align))
        v_des = min(v_des, cap)
        # During recovery the `floor` already lifts v_des above the
        # alignment gate, but if the body is pointing roughly opposite
        # the target (align < 0) the agent will translate "backwards
        # through" the candidate. That's fine — we just need motion
        # to break the limit cycle; the path will replan anew on the
        # next tick.
        # Angular setpoint: turn toward the target direction.
        omega_des = HEADING_ERR_KP * heading_err
        if omega_des > MAX_ANGULAR_VEL:
            omega_des = MAX_ANGULAR_VEL
        elif omega_des < -MAX_ANGULAR_VEL:
            omega_des = -MAX_ANGULAR_VEL

        # P-control on forward velocity → force F (knife-edge thrust)
        F = THRUST_KP * (v_des - self.forward_vel)
        if F > MAX_THRUST:
            F = MAX_THRUST
        elif F < -MAX_THRUST:
            F = -MAX_THRUST
        # P-control on angular velocity → moment M (about center)
        M = MOMENT_KP * (omega_des - self.angular_vel)
        if M > MAX_MOMENT:
            M = MAX_MOMENT
        elif M < -MAX_MOMENT:
            M = -MAX_MOMENT

        # Equations of motion: forward and angular dynamics with
        # viscous friction on each. Body heading integrates from ω.
        f_friction = -LINEAR_DAMPING * self.forward_vel
        m_friction = -ANGULAR_DAMPING * self.angular_vel
        self.forward_vel += (F + f_friction) / ROBOT_MASS * dt
        self.angular_vel += (M + m_friction) / ROBOT_INERTIA * dt
        # Hard-clip — guards transient setpoint spikes.
        if self.forward_vel > MAX_SPEED_MPS:
            self.forward_vel = MAX_SPEED_MPS
        elif self.forward_vel < -MAX_SPEED_MPS:
            self.forward_vel = -MAX_SPEED_MPS
        ang_clip = MAX_ANGULAR_VEL * 1.5
        if self.angular_vel > ang_clip:
            self.angular_vel = ang_clip
        elif self.angular_vel < -ang_clip:
            self.angular_vel = -ang_clip

        old_odom = (self.odom[0], self.odom[1])
        # ── Body integration: TRUE physics first, then derive
        # REPORTED odom as a deterministic biased rotation of truth.
        #
        # Real-robot architecture has two cascaded EKFs:
        #   wheel /odom (biased ω) ─┐
        #                            ├─► ekf_local ──► /local_ekf/odom
        #   /multiScan/imu (clean ω) ┘                       │
        #                                                    ▼
        #                              gps_handler_node EKF ◄┘ + /gps_fix
        # The sim's invariant: the REPORTED odom IS the truth's
        # motion, just biased by accumulated encoder yaw drift —
        # never an independent integration that can random-walk
        # away from truth (which would let the EKF "ghost" wander
        # while the truth is at rest). Concretely:
        #
        #   body_heading       = body_heading_true + _yaw_bias_offset
        #   _yaw_bias_offset  += K_eff · |forward_vel·dt|
        #   odom_vel           = forward_vel · (cos, sin)(body_heading)
        #
        # K_eff = self._odom_yaw_bias_rate when LIDAR_IMU_FUSION is
        # OFF (raw wheel encoder), and K_eff = 0 when it's ON (the
        # SICK IMU + ekf_local cancel the encoder yaw bias to within
        # IMU noise — modelled as zero residual at this stage so
        # the truth-→-odom map is fully deterministic). When the
        # robot is physically at rest (forward_vel = 0), the bias
        # offset doesn't grow, body_heading freezes, and odom_vel
        # is exactly zero — guaranteeing the EKF predict can't move
        # the position estimate while truth is stopped.
        self.body_heading_true = ((self.body_heading_true
                                    + self.angular_vel * dt
                                    + math.pi) % (2 * math.pi) - math.pi)
        if LIDAR_IMU_FUSION_ENABLE:
            K_eff = 0.0
        else:
            K_eff = self._odom_yaw_bias_rate
        self._yaw_bias_offset += K_eff * abs(self.forward_vel) * dt
        self.body_heading = ((self.body_heading_true
                                + self._yaw_bias_offset
                                + math.pi) % (2 * math.pi) - math.pi)
        ch_rep = math.cos(self.body_heading)
        sh_rep = math.sin(self.body_heading)
        # Knife-edge constraint: REPORTED odom velocity is parallel
        # to the REPORTED body axis (this is what the encoders +
        # wheel_odom_pub publish; the EKF predicts on this).
        self.odom_vel[0] = self.forward_vel * ch_rep
        self.odom_vel[1] = self.forward_vel * sh_rep
        self.odom[0] += self.odom_vel[0] * dt
        self.odom[1] += self.odom_vel[1] * dt

        # ── True-world motion: integrate forward velocity along
        # the TRUE body axis, then rotate by `true_heading` (the
        # constant rotation between the abstract true-odom frame
        # and the world frame). Reported odom is decoupled from
        # this entirely — it's a sensor stream with its own drift.
        ch_true = math.cos(self.body_heading_true)
        sh_true = math.sin(self.body_heading_true)
        dxo_true = self.forward_vel * ch_true * dt
        dyo_true = self.forward_vel * sh_true * dt
        cT = math.cos(self.true_heading); sT = math.sin(self.true_heading)
        wdx = cT * dxo_true - sT * dyo_true
        wdy = sT * dxo_true + cT * dyo_true
        self.true_pos[0] += wdx
        self.true_pos[1] += wdy
        self.true_trail.append(tuple(self.true_pos))

        # Reported odom delta — what predict() and the closed-form
        # heading fit consume. Computed AFTER the reported integral
        # above so this is exactly the published wheel-odom delta.
        dxo = self.odom[0] - old_odom[0]
        dyo = self.odom[1] - old_odom[1]
        # Cumulative reported forward distance — stamped into every
        # gps_history entry so the closed-form fit can recover both
        # θ and the per-meter encoder yaw bias K (linear fit of
        # theta_i vs δD/2).
        self.D_total += math.hypot(dxo, dyo)
        # Derotate the reported odom delta by R(-K · D_total)
        # before feeding it to the EKF predict. The closed-form
        # fit estimates K such that  reported_delta =
        # R(K·D) · true_delta, so derotating recovers an implicit
        # "true odom delta" and the EKF's predict equation stays
        # the original  world_delta = R(θ) · delta . When
        # K_est == 0 (initial state or no bias), this is identity
        # and the path collapses to the original predict.
        if abs(self.K_est) > 1e-12:
            kd = self.K_est * self.D_total
            ck = math.cos(-kd); sk = math.sin(-kd)
            true_dxo = ck * dxo - sk * dyo
            true_dyo = sk * dxo + ck * dyo
        else:
            true_dxo, true_dyo = dxo, dyo

        # ── EKF predict + update.
        # The EKF is the "ghost" of truth — a biased-and-filtered
        # representation that should mirror truth's motion, not
        # acquire independent dynamics. When the body is physically
        # at rest (forward_vel ≈ 0 AND angular_vel ≈ 0) the truth
        # itself does not move, so the ghost should not move either:
        # we skip both predict (no odom delta to integrate; no
        # process-noise covariance growth) and update (no GPS-jitter
        # nudge that would visibly drift the estimate while the
        # body is stopped). The 0.01 thresholds match the coasting-
        # end deadband used elsewhere in the controller.
        body_at_rest = (
            abs(self.forward_vel) < 0.01
            and abs(self.angular_vel) < 0.01)
        if self.ekf is not None and not body_at_rest:
            self.ekf.predict(true_dxo, true_dyo, dt)
            if new_gps is not None:
                # During bootstrap the EKF linearization is unreliable —
                # don't gate (gate_chi2 huge), and replace θ in-place
                # with a closed-form estimate so the next predict/update
                # is linearized around a near-truth point.
                if not GPS_HEADING_EKF_ENABLE:
                    # GPS-heading EKF disabled — mirrors the May 9
                    # field configuration where the deployed robot
                    # was running PURE wheel odom against a goal
                    # expressed in odom frame, with no GPS feedback
                    # in the loop at all. We deliberately do NOT
                    # call ``self.ekf.update`` here: applying a GPS
                    # position update would re-pull ekf.pos toward
                    # truth every tick, which makes the candidate
                    # (= odom + (goal_world − ekf.pos)) chase a
                    # moving target and produce a spiral. The IRL
                    # signature was instead a *straight line* from
                    # start to a wrong-frame endpoint — that's what
                    # pure odom + a fixed map-↔-world rotation
                    # gives. We still pin ekf.theta = 0 (the robot
                    # had no θ estimate) and reset its covariance
                    # to a wide prior so the cascade detectors
                    # behave sensibly if they're sampled.
                    self.ekf.x[2] = 0.0
                    self.ekf.P[2, 0] = 0.0
                    self.ekf.P[0, 2] = 0.0
                    self.ekf.P[2, 1] = 0.0
                    self.ekf.P[1, 2] = 0.0
                    self.ekf.P[2, 2] = math.radians(60.0) ** 2
                elif not self.bootstrap_done:
                    # DEAD BRANCH — matches deployed L871 on
                    # ``origin/improve/gps-waypoint-continuity``.
                    # ``bootstrap_done`` is initialized True from
                    # ``__init__`` so this branch is unreachable in
                    # production; the EKF runs predict/update from
                    # tick 1 with theta_var0 = π², and the first GPS
                    # update's Kalman gain ≈ 1.0 on θ produces the same
                    # snap-on-first-sample behavior the explicit
                    # bootstrap reset used to provide. Body preserved
                    # for documentation / defensive equivalence: any
                    # future caller that does flip ``bootstrap_done``
                    # back to False will still run the pre-port
                    # ``ekf.update(..., gate_chi2=1e9)`` + closed-form
                    # ``reset_theta`` + ``_adopt_K`` sequence and
                    # graduate when ``odom_dist > 5 AND baseline > 5``.
                    self.ekf.update(new_gps[0], new_gps[1],
                                    gate_chi2=1e9)
                    # Bootstrap fit defaults to anchor-on-first
                    # (window=None). The function also supports a
                    # sliding-anchor window (set ``BOOTSTRAP_WINDOW``
                    # and pass ``window=BOOTSTRAP_WINDOW`` here) —
                    # tested at window=100, regressed convergence
                    # by ~1.5 % at 1000 agents because bootstrap
                    # graduates before the window can clip and the
                    # long tail of slow-bootstrap agents benefits
                    # from the longer baseline of the full-history
                    # fit. Kept parametric for future experiments.
                    bs_theta, bs_K, bs_baseline = self._bootstrap_theta(
                        min_baseline=1.5)
                    if bs_theta is not None:
                        # Use a baseline-dependent variance for the
                        # closed-form theta. Longer baseline → tighter
                        # fit → smaller variance, which lets the EKF
                        # actually graduate from bootstrap once enough
                        # motion has accumulated. Floor at 3° to avoid
                        # an over-confident filter early on.
                        sigma = max(math.radians(3.0),
                                     GPS_NOISE_STD / max(bs_baseline, 0.5))
                        self.ekf.reset_theta(
                            bs_theta, theta_var=sigma * sigma)
                        # Adopt the joint fit's K alongside θ so the
                        # next predict tick derotates by the right
                        # bias rate. Both states are refreshed every
                        # bootstrap tick until graduation.
                        self._adopt_K(bs_K)
                    odom_dist = math.hypot(self.odom[0], self.odom[1])
                    # Graduate from bootstrap once we've accumulated
                    # enough baseline. The earlier σ_θ < 3° check was
                    # unreachable: reset_theta was slamming theta_var
                    # to (15°)² every tick, so the criterion never
                    # fired and agents stayed speed-capped at 1 m/s
                    # forever.
                    if (odom_dist > self.bootstrap_min_travel
                            and bs_baseline > self.bootstrap_min_travel):
                        self.bootstrap_done = True
                else:
                    # Lock-in recovery: mirror gps_handler_node lines
                    # 672-679. If the EKF's own
                    # ``consecutive_rejects`` counter has crossed
                    # the streak threshold BEFORE this update, ask
                    # the filter to reinflate position covariance
                    # and force-accept this sample (gate_chi2=1e9
                    # matches the robot exactly). Otherwise run the
                    # standard gated update. The counter lives on
                    # the EKF and is incremented/cleared inside
                    # update() — sim and robot consume the same
                    # signal the same way.
                    if self.ekf.consecutive_rejects >= EKF_REJ_STREAK_RESET:
                        self.ekf.force_accept_next()
                        self.ekf.update(new_gps[0], new_gps[1],
                                        gate_chi2=1e9)
                    else:
                        self.ekf.update(new_gps[0], new_gps[1])
                    # Heading-resync runs every GPS tick (cheap). Snap
                    # only fires when the closed-form fit disagrees by
                    # more than the threshold AND the cooldown has
                    # expired, so this won't thrash on small drift.
                    self._maybe_resync_heading()

        # Record an "intended endpoint" sample on each fresh GPS tick.
        if new_gps is not None:
            self.intended_endpoint_history.append(self.intermediate_goal_world())
            if len(self.intended_endpoint_history) > INTENDED_HISTORY_LEN:
                self.intended_endpoint_history.pop(0)

        # ── Arrival / coast logic ──────────────────────────────────
        # Two distinct arrival semantics:
        #
        #   * Truth-based (default): the IGVC competition rule of
        #     ≥ 50 % footprint inside the 1 m goal ring. Fires when
        #     the robot OBJECTIVELY reaches the goal in world frame.
        #     Used for the default scripted scenario and --crazy.
        #
        #   * Local-frame (ROBOT_STRICT_ARRIVAL, set by --real):
        #     fires when the AGENT'S REPORTED ODOMETRIC POSITION
        #     reaches the published candidate goal expressed in
        #     odom frame — i.e. "the wheel encoders say I have
        #     arrived." That's how the deployed robot actually
        #     terminates: the action server publishes a candidate
        #     in map frame, NAV2 drives base_link toward it, and
        #     when local-frame distance falls below
        #     ROBOT_SUCCESS_RADIUS_M the goal succeeds. With
        #     encoder yaw bias active, the TRUE world position at
        #     this moment is far from goal_world — that's the
        #     field-test failure mode the sim must reproduce.
        #
        # Local target = R(-θ_est) · (published_goal − ekf.pos)
        # added to last_odom_xy. With GPS_HEADING_EKF_ENABLE=False
        # (θ_est = 0) and ekf.pos held at the start anchor, this
        # collapses to ``last_odom + (published_goal_world)`` —
        # i.e. the agent stops when reported odom has covered the
        # vector from start to goal, regardless of true motion.
        if not self.arrived:
            if ROBOT_STRICT_ARRIVAL:
                # Two-stage arrival (mirrors what the deployed robot
                # actually does):
                #
                #   STAGE 1 — NAV2 terminates the action goal.
                #     The action server publishes the *intended
                #     endpoint* (``self.published_goal_world``,
                #     sampled at NAV2_GOAL_HZ from the live
                #     candidate). NAV2 drives base_link toward it
                #     and the action succeeds when the EKF-
                #     estimated position lands within
                #     ROBOT_SUCCESS_RADIUS_M of that published
                #     candidate. This is the only termination
                #     signal NAV2 itself has — it cannot see the
                #     true GPS goal, only the candidate the closed-
                #     form θ fit shaped for it.
                #
                #   STAGE 2 — GPS-vs-goal validation.
                #     Once stage 1 fires the robot is *stopped at
                #     the intended endpoint*. Now it samples its
                #     real GPS and checks whether that GPS reading
                #     lies within the competition success circle
                #     (GOAL_RADIUS = 1 m) around the true goal
                #     waypoint. Only then is the run declared a
                #     success — this is what proves the closed-
                #     form θ fit + EKF actually drove the
                #     candidate onto the real GPS goal rather than
                #     pointing the robot at a phantom target.
                # STAGE 1 — robot's onboard arrival signal.
                # The deployed gps_handler_node publishes the EKF's
                # filtered position as ``/odom_filtered`` in the
                # map frame, and NAV2's BT terminates the goal
                # action when |filtered_odom − goal_world| <
                # ROBOT_SUCCESS_RADIUS_M. In our sim, the filtered
                # odom is ``self.ekf.pos_xy`` (world frame), and
                # the goal is ``self.goal_world``. With EKF off
                # (and our recent fix making GPS updates a no-op),
                # ekf.pos integrates only from biased odom →
                # ekf.pos converges to ``self.odom`` and the check
                # collapses to "wheel-odom reaches goal_world",
                # which is the field-test failure mode where the
                # robot declared success while truth was 40 m
                # off. With EKF on, ekf.pos tracks truth via GPS
                # and the check fires only when truth is at the
                # goal — the corrected behaviour.
                if self.ekf is not None:
                    rx, ry = self.ekf.pos_xy
                else:
                    rx, ry = self.odom[0], self.odom[1]
                d_robot_local = math.hypot(
                    rx - self.goal_world[0],
                    ry - self.goal_world[1])
                robot_declared_now = d_robot_local < ROBOT_SUCCESS_RADIUS_M
                if robot_declared_now and not self.robot_declared_success:
                    # First crossing — latch the robot's onboard
                    # success flag and record the truth position
                    # at that moment so downstream reporting can
                    # show *where the robot was when it thought
                    # it was done*. The robot itself stops driving
                    # at this point (NAV2 ends its action), so we
                    # also start coasting.
                    self.robot_declared_success = True
                    self.robot_declared_at_step = self.steps
                    self.robot_declared_truth_xy = (
                        self.true_pos[0], self.true_pos[1])
                    self.coasting = True

                # STAGE 2 — sim's own verdict.
                # The deployed action server can't see the real
                # GPS goal, so it stops on stage 1. The sim,
                # however, *can* see truth + GPS, so we declare
                # ``arrived = True`` only when the most recent
                # real GPS fix lies inside the competition
                # success circle (1 m around the GPS waypoint).
                # Stage 2 is gated on stage 1 because the robot
                # has to *stop* before the GPS fix is meaningful
                # as an arrival signal; while still moving, the
                # GPS just samples the trajectory.
                if robot_declared_now:
                    gps_xy = (self.gps_history[-1][1]
                              if self.gps_history else None)
                    if gps_xy is None:
                        # No fix yet — robot has declared success
                        # and is parked; wait for the next sample.
                        arrived_now = False
                    else:
                        d_gps = math.hypot(
                            gps_xy[0] - self.goal_world[0],
                            gps_xy[1] - self.goal_world[1])
                        arrived_now = d_gps < GOAL_RADIUS
                else:
                    arrived_now = False
            else:
                d_truth = math.hypot(
                    self.true_pos[0] - self.goal_world[0],
                    self.true_pos[1] - self.goal_world[1])
                arrived_now = (
                    self.goal_overlap_fraction() >= 0.5
                    or d_truth < ROBOT_SUCCESS_RADIUS_M)
            if arrived_now:
                self.arrived = True
                self.coasting = True
        # ── Snapshot-at-step-300 convergence classifier ─────────────
        # At PREDICT_SNAPSHOT_TICKS (30 s of sim), every still-
        # unclassified agent is binned by a one-shot test:
        #
        #   d_cand = |published_goal_world − goal_world|
        #              (yellow X plotted on the world axis vs the
        #               green star — exactly what the user reads off
        #               the GUI when judging convergence.)
        #   d_true = |true_pos − goal_world|
        #              (the sim's own GPS-truth distance to the
        #               goal, which the deployed robot can't see.)
        #
        # If d_cand < PREDICT_CAND_RING_M AND d_true < PREDICT_TRUE_RING_M
        # the candidate has settled near the real goal AND the
        # robot is close enough to finish the drive within a
        # short additional window — predicted_success. Otherwise
        # predicted_failure.
        if (self.steps == PREDICT_SNAPSHOT_TICKS
                and not self.arrived
                and not self.predicted_success
                and not self.predicted_failure):
            cand = self.published_goal_world
            d_cand = math.hypot(
                cand[0] - self.goal_world[0],
                cand[1] - self.goal_world[1])
            d_true = math.hypot(
                self.true_pos[0] - self.goal_world[0],
                self.true_pos[1] - self.goal_world[1])
            if (d_cand < PREDICT_CAND_RING_M
                    and d_true < PREDICT_TRUE_RING_M):
                self.predicted_success = True
                self.predicted_at_step = self.steps
            else:
                self.predicted_failure = True
                self.predicted_failure_reason = (
                    f"snapshot d_cand={d_cand:.1f}m "
                    f"d_true={d_true:.1f}m")
                self.predicted_failure_at_step = self.steps
        if self.coasting:
            v = abs(self.forward_vel) + abs(self.angular_vel) * 0.3
            if v < 0.05:                  # stopped
                self.coasting = False
                if (not self.arrived
                        and not self.predicted_success
                        and not self.predicted_failure):
                    # Robot's onboard logic terminated NAV2
                    # (robot_declared_success → coasting → stop)
                    # but the sim's GPS-vs-goal check never
                    # passed — the robot is parked at the wrong
                    # spot. Bin it as a predicted failure so the
                    # three-bin classifier covers every agent.
                    self.predicted_failure = True
                    self.predicted_failure_reason = "stopped_off_target"
                    self.predicted_failure_at_step = self.steps
                return False
        return True


# ── GUI ──────────────────────────────────────────────────────────
class GPSWaypointGUI:
    def __init__(self, sim, obstacles, args,
                 roofs=(), projectors=(),
                 jammers=(), foliage=(), spoofers=(),
                 peers=()):
        # `sim` is the primary agent (the one that gets the rich
        # follow-cam + status detail). `peers` is everyone else.
        self.sim = sim
        self.peers = list(peers)
        self.all_sims = [sim] + self.peers
        # Detect the field-parity twin pairing: exactly one peer,
        # and that peer has its encoder bias disabled. In that case
        # we want all single-agent overlays (compass, A* path,
        # intermediate marker, EKF marker, status panel) to keep
        # showing for the *primary* agent, with the twin rendered
        # alongside as a comparison body. Treat the GUI as
        # single-agent for overlay-hiding purposes.
        self._perfect_twin_pair = (
            len(self.peers) == 1
            and getattr(self.peers[0],
                        "_odom_yaw_bias_rate", None) == 0.0
            and getattr(self.sim,
                        "_odom_yaw_bias_rate", None) != 0.0)
        self.is_multi = (len(self.all_sims) > 1
                          and not self._perfect_twin_pair)
        self.obstacles = obstacles
        self.roofs = list(roofs)
        self.projectors = list(projectors)
        self.jammers = list(jammers)
        self.foliage = list(foliage)
        self.spoofers = list(spoofers)
        self.args = args

        # 16×9 figsize (was 14×8.5) — gives the right-edge "setup" column
        # of the status panel enough room so labels like "True θ -173.26° hidden"
        # don't wrap or clip past the figure boundary.
        self.fig = plt.figure(figsize=(16, 9), facecolor="#1a1a1a")
        self.fig.canvas.manager.set_window_title(
            "GPS Waypoint Sim — magnetometer-less heading discovery")
        gs = self.fig.add_gridspec(2, 3,
                                    width_ratios=[2.4, 0.6, 0.6],
                                    height_ratios=[1.0, 2.2],
                                    wspace=0.05, hspace=0.18)

        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax.set_facecolor("#0d1f12")  # darker field-green tint
        self.ax.set_xlim(-MAP_HALF, MAP_HALF)
        self.ax.set_ylim(-MAP_HALF, MAP_HALF)
        self.ax.set_aspect("equal")
        # Title doubles as the live keybinding hint. The GPS sim is
        # auto-spawn (no click-to-place), so we advertise the keys the
        # ``_on_key`` handler actually supports: P/space pause, R reset,
        # Q/Esc quit. Matches the LiDAR sim's title-as-cheatsheet idiom.
        self.ax.set_title(
            f"P=Pause | R=Reset | Q=Quit  ·  500 ft · "
            f"{LAT_CENTER:.5f} N, {abs(LON_CENTER):.5f} W",
            color="#e0e0e0", fontsize=10)
        self.ax.set_xlabel("East (m)", color="#a0a0a0")
        self.ax.set_ylabel("North (m)", color="#a0a0a0")
        self.ax.tick_params(colors="#888")
        for s in self.ax.spines.values():
            s.set_edgecolor("#444")
        self.ax.grid(True, color="#163020", linestyle=":", linewidth=0.4)

        # Scenario-specific patches live in this list so R-reset can
        # tear them down without rebuilding the figure. Populated at
        # the end of __init__ after both axes are set up.
        self._scenario_artists = []

        # 1 m success circle around the true goal. Competition rule:
        # ≥ 50 % of the robot inside passes the waypoint. We also draw
        # an inner dashed ring at GOAL_RADIUS - ROBOT_RADIUS — when
        # the robot center crosses inside this inner ring, the entire
        # footprint is in the goal circle (100 % overlap, full margin).
        self.goal_circle = Circle(sim.goal_world, GOAL_RADIUS,
                                  facecolor=(0.2, 0.9, 0.3, 0.18),
                                  edgecolor="#33ff66", linewidth=1.3,
                                  zorder=3)
        self.ax.add_patch(self.goal_circle)
        inner_r = max(0.0, GOAL_RADIUS - ROBOT_RADIUS)
        self.goal_inner_ring = Circle(sim.goal_world, inner_r,
                                       fill=False,
                                       edgecolor="#33ff66",
                                       linestyle="--", linewidth=0.9,
                                       alpha=0.5, zorder=3)
        self.ax.add_patch(self.goal_inner_ring)
        self.goal_marker, = self.ax.plot(
            [sim.goal_world[0]], [sim.goal_world[1]],
            "*", color="#33ff66", markersize=18,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=11, label="True GPS goal")

        # Robot start cross — capture the handle so R-reset can
        # reposition it; without the handle, the cross would freeze
        # at the *original* start while every subsequent R-reset
        # picks a fresh random start, leaving a stale yellow X on
        # the map.
        self.start_marker, = self.ax.plot(
            [sim.start_world[0]], [sim.start_world[1]],
            "x", color="#ffcc00", markersize=10,
            markeredgewidth=2, zorder=10, label="Start")

        # Dynamic elements ----------------------------------------------
        # A* path
        self.path_line, = self.ax.plot([], [], "-", color="#ffd24a",
                                       linewidth=1.2, alpha=0.7,
                                       zorder=6)
        # Smart-padding window — kept very subtle, mainly diagnostic
        self.window_patch = Rectangle((0, 0), 0, 0, fill=False,
                                      edgecolor="#555",
                                      linestyle=":", linewidth=0.6,
                                      alpha=0.4, zorder=4)
        self.ax.add_patch(self.window_patch)
        # GPS scatter (recent)
        self.gps_scatter = self.ax.scatter([], [], s=5,
                                           c="#6cd0ff",
                                           alpha=0.45, zorder=7)
        # Goal-belief cloud — stamps the perpendicular arc as θ refines
        self.belief_cloud = self.ax.scatter([], [], s=10,
                                            c="#ffe14a",
                                            alpha=0.15, zorder=6)
        # Running mean of the cloud
        self.belief_mean_marker, = self.ax.plot(
            [], [], "P", color="#ff66cc", markersize=11,
            markeredgecolor="black", markeredgewidth=0.6,
            zorder=10)
        # Robot trail (true)
        self.trail_line, = self.ax.plot([], [], "-", color="#ff5050",
                                        linewidth=1.5, alpha=0.85,
                                        zorder=8)
        # Robot body (true)
        self.robot_body = Circle(sim.true_pos, ROBOT_RADIUS,
                                 facecolor="#ff3030",
                                 edgecolor="white",
                                 linewidth=1.0, zorder=12)
        self.ax.add_patch(self.robot_body)
        # Heading arrow (true world heading — the body's actual
        # orientation that physics integrates against).
        self.heading_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#ff8080",
                            lw=1.3), zorder=13)
        # EKF position estimate (smoothed)
        self.ekf_marker, = self.ax.plot(
            [], [], "o", color="#9bff9b", markersize=7,
            markeredgecolor="black", markeredgewidth=0.5,
            zorder=11)
        # EKF heading arrow — agent's BELIEF about its body's
        # world-frame heading: ``body_heading + ekf.theta`` (the
        # rotation between odom and world that the EKF estimates).
        # When the local-EKF IMU fusion is healthy this overlaps
        # the truth arrow; the visible angular gap between the
        # two is the residual θ-estimate error driving any
        # convergence problem.
        self.ekf_heading_arrow = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#33ff66",
                            lw=1.3, alpha=0.85), zorder=13)
        # Intermediate goal (where the robot would actually land)
        self.intermediate_marker, = self.ax.plot(
            [], [], "X", color="#ffe14a", markersize=13,
            markeredgecolor="black", markeredgewidth=0.7,
            zorder=11)
        # GPS-disconnected indicator (red ring around robot when no fix)
        self.dropout_ring = Circle(sim.true_pos,
                                   ROBOT_RADIUS + 0.6,
                                   facecolor="none",
                                   edgecolor="#ff3030",
                                   linewidth=1.4, alpha=0.0,
                                   zorder=12)
        self.ax.add_patch(self.dropout_ring)

        # ── Compass clock-hands (centered on the map) ──────────
        # Two hands rooted at the map center:
        #   * Real north (green): always points to physical +y
        #     (the GPS / ENU "north"). Length is fixed.
        #   * Estimated north (yellow): points to where the agent
        #     thinks +y_world is. Computed as
        #       direction = R(true_heading − ekf.theta) · (0, 1)
        #     so when the EKF's θ matches truth the two hands
        #     overlap, and when θ is off (the field-parity case),
        #     the angular gap between the hands IS the heading
        #     error driving the wrong-convergence.
        self.COMPASS_CENTER = (0.0, 0.0)
        self.COMPASS_LENGTH = 12.0
        self.compass_ring = Circle(
            self.COMPASS_CENTER, 1.2, fill=False,
            edgecolor="#666", linewidth=0.8, alpha=0.6, zorder=14)
        self.ax.add_patch(self.compass_ring)
        cx0, cy0 = self.COMPASS_CENTER
        # TRUE NORTH (green): the GPS / ENU world's +y axis. Always
        # points up — that's the world's actual north reference.
        # Label is a Text artist at the arrowhead so the text
        # rides on the head dot rather than the tail.
        self.compass_real_arrow = self.ax.annotate(
            "", xy=(cx0, cy0 + self.COMPASS_LENGTH),
            xytext=(cx0, cy0),
            arrowprops=dict(arrowstyle="->",
                            color="#33ff66", lw=2.4),
            zorder=15)
        self.compass_real_dot, = self.ax.plot(
            [cx0], [cy0 + self.COMPASS_LENGTH], "o",
            color="#33ff66", markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=16)
        self.compass_real_label = self.ax.text(
            cx0, cy0 + self.COMPASS_LENGTH + 1.4,
            "TRUE N", color="#33ff66", fontsize=8,
            fontweight="bold", ha="center", va="bottom",
            zorder=16)
        # MAP NORTH (yellow): where the agent's map frame thinks
        # +y is. Rotated by the EKF's residual θ error. As the
        # EKF + ENU-projection θ-correction converge, the map-north
        # arrow rotates back onto true-north — visible proof the
        # localisation stack is aligning the agent's map frame to
        # the world.
        self.compass_est_arrow = self.ax.annotate(
            "", xy=(cx0, cy0 + self.COMPASS_LENGTH),
            xytext=(cx0, cy0),
            arrowprops=dict(arrowstyle="->",
                            color="#ffcc00", lw=2.4),
            zorder=15)
        self.compass_est_dot, = self.ax.plot(
            [cx0], [cy0 + self.COMPASS_LENGTH], "o",
            color="#ffcc00", markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=16)
        self.compass_est_label = self.ax.text(
            cx0, cy0 + self.COMPASS_LENGTH + 1.4,
            "MAP N", color="#ffcc00", fontsize=8,
            fontweight="bold", ha="center", va="bottom",
            zorder=16)

        # Per-peer body + trail. Empty when single-agent.
        # Above HEAVY_MULTI_THRESHOLD agents we replace 1000 individual
        # Circle patches and Line2D trails (which matplotlib draws one
        # at a time) with a single scatter — set_offsets() takes a
        # 1000×2 array in one call, which is what makes 1000-agent
        # rendering tractable.
        self.HEAVY_MULTI_THRESHOLD = 50
        # How many trail points to keep per peer in heavy-multi mode.
        # 50 × 0.1s tick = 5s of history — long enough to read direction,
        # short enough that a 1000-agent LineCollection stays at ≤50k
        # vertices and renders in a couple of milliseconds.
        self.PEER_TRAIL_TAIL = 50
        self.heavy_multi = len(self.peers) >= self.HEAVY_MULTI_THRESHOLD
        self.peer_bodies = []
        self.peer_trails = []
        self.peer_scatter = None
        self.peer_trail_lc = None
        if self.heavy_multi:
            # Pre-size with current peer positions; sizes are in points²
            # for scatter, so a small s≈8 keeps 1000 dots legible.
            init_xy = np.array(
                [(p.true_pos[0], p.true_pos[1]) for p in self.peers],
                dtype=float)
            self.peer_scatter = self.ax.scatter(
                init_xy[:, 0], init_xy[:, 1],
                s=8, c="#ff7777", edgecolors="none",
                alpha=0.75, zorder=11)
            # Single LineCollection for all peer trails — one draw call
            # instead of 1000.
            self.peer_trail_lc = LineCollection(
                [], colors="#ff5050", linewidths=0.5,
                alpha=0.25, zorder=7)
            self.ax.add_collection(self.peer_trail_lc)
        else:
            for p in self.peers:
                # Distinguish the field-parity perfect twin (no
                # encoder bias) with a green body and trail so the
                # comparison vs the biased red primary agent is
                # immediately readable.
                is_perfect_twin = (
                    self._perfect_twin_pair
                    and getattr(p, "_odom_yaw_bias_rate", None) == 0.0)
                if is_perfect_twin:
                    body_face = "#5fbb5f"; trail_color = "#33ff66"
                    body_alpha = 0.95; trail_alpha = 0.85
                    trail_lw = 1.2
                else:
                    body_face = "#ff7777"; trail_color = "#ff5050"
                    body_alpha = 0.85; trail_alpha = 0.30
                    trail_lw = 0.55
                body = Circle(p.true_pos, ROBOT_RADIUS,
                              facecolor=body_face, edgecolor="white",
                              linewidth=0.4, alpha=body_alpha, zorder=11)
                self.ax.add_patch(body)
                trail, = self.ax.plot([], [], "-", color=trail_color,
                                       linewidth=trail_lw, alpha=trail_alpha,
                                       zorder=7)
                self.peer_bodies.append(body)
                self.peer_trails.append(trail)
        # In multi-agent mode the rich single-agent overlays make the
        # ensemble unreadable. Hide them — the per-agent dot+trail and
        # the aggregate status panel are the real story.
        if self.is_multi:
            for art in (self.path_line, self.window_patch,
                         self.gps_scatter, self.belief_cloud,
                         self.belief_mean_marker,
                         self.intermediate_marker,
                         self.ekf_marker,
                         self.compass_ring):
                art.set_visible(False)
            self.compass_real_arrow.set_visible(False)
            self.compass_est_arrow.set_visible(False)
            self.compass_real_dot.set_visible(False)
            self.compass_est_dot.set_visible(False)
            self.compass_real_label.set_visible(False)
            self.compass_est_label.set_visible(False)
            self.ekf_heading_arrow.set_visible(False)

        # Mini-cam + status text now blit every frame too — they are
        # cheap (single ``draw_artist`` per artist, no canvas redraw),
        # so the user sees the follow-cam, goal-cam, and status text
        # update at the full 30 FPS. The old ``MINI_REFRESH_EVERY``
        # gate is gone; instead the follow-cam and goal-cam use a
        # dead-zone in ``_render_dynamic_panels`` — they only call
        # ``set_xlim/set_ylim`` (which would invalidate the bg) when
        # the robot leaves the dead zone, otherwise the markers just
        # slide within the existing window.
        self._frame_idx = 0
        self._blit_supported = True
        self._ax_bg = None
        self._mini_bg = None
        self._goal_mini_bg = None
        self._status_bg = None
        self._mini_center = None
        self._goal_center = None
        self._goal_half = None
        self._panel_recenter_needed = False

        # ── Upper-right: 5×5 m follow-cam ─────────────────────────
        self.MINI_VIEW_HALF = 2.5    # metres from robot center
        self.ax_mini = self.fig.add_subplot(gs[0, 1])
        self.ax_mini.set_facecolor("#0d1f12")
        self.ax_mini.set_aspect("equal")
        # Robot-local follow-cam: xlim/ylim fixed forever at ±MINI_VIEW_HALF
        # and every artist's data is offset by ``-robot_pos`` per frame.
        # That puts the robot at (0, 0) and slides the world around it,
        # giving a perfectly smooth follow with no bg-invalidation /
        # dead-zone snap.
        h0 = self.MINI_VIEW_HALF
        self.ax_mini.set_xlim(-h0, h0)
        self.ax_mini.set_ylim(-h0, h0)
        self.ax_mini.set_title(
            f"{2*self.MINI_VIEW_HALF:.0f}×{2*self.MINI_VIEW_HALF:.0f} m follow-cam (robot-centered)",
            color="#a0a0a0", fontsize=9)
        self.ax_mini.tick_params(colors="#666", labelsize=7)
        for s in self.ax_mini.spines.values():
            s.set_edgecolor("#444")
        self.ax_mini.grid(True, color="#163020",
                          linestyle=":", linewidth=0.4)

        # Mini-cam scenario patches are added by _build_scenario_patches
        # (called from the main map setup above), keeping reset clean.

        # Goal star + 1 m ring (same world coords; shown only when in
        # the follow-cam window).
        self.mini_goal_circle = Circle(sim.goal_world, GOAL_RADIUS,
                                       facecolor=(0.2, 0.9, 0.3, 0.18),
                                       edgecolor="#33ff66", linewidth=1.0,
                                       zorder=3)
        self.ax_mini.add_patch(self.mini_goal_circle)
        self.mini_goal_inner_ring = Circle(
            sim.goal_world, max(0.0, GOAL_RADIUS - ROBOT_RADIUS),
            fill=False, edgecolor="#33ff66",
            linestyle="--", linewidth=0.9, alpha=0.5, zorder=3)
        self.ax_mini.add_patch(self.mini_goal_inner_ring)
        self.mini_goal_marker, = self.ax_mini.plot(
            [sim.goal_world[0]], [sim.goal_world[1]],
            "*", color="#33ff66", markersize=14,
            markeredgecolor="white", markeredgewidth=0.4, zorder=11)

        # Dynamic mini-cam artists (parallel to the main map's)
        self.mini_path_line, = self.ax_mini.plot(
            [], [], "-", color="#ffd24a",
            linewidth=1.2, alpha=0.85, zorder=6)
        self.mini_gps_scatter = self.ax_mini.scatter(
            [], [], s=10, c="#6cd0ff", alpha=0.6, zorder=7)
        self.mini_trail_line, = self.ax_mini.plot(
            [], [], "-", color="#ff5050",
            linewidth=1.4, alpha=0.85, zorder=8)
        self.mini_robot = Circle(sim.true_pos, ROBOT_RADIUS,
                                  facecolor="#ff3030",
                                  edgecolor="white",
                                  linewidth=1.0, zorder=12)
        self.ax_mini.add_patch(self.mini_robot)
        self.mini_heading_arrow = self.ax_mini.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#ff8080",
                             lw=1.3), zorder=13)
        self.mini_ekf_marker, = self.ax_mini.plot(
            [], [], "o", color="#9bff9b", markersize=8,
            markeredgecolor="black", markeredgewidth=0.5, zorder=11)
        self.mini_intermediate_marker, = self.ax_mini.plot(
            [], [], "X", color="#ffe14a", markersize=11,
            markeredgecolor="black", markeredgewidth=0.6, zorder=11)
        self.mini_dropout_ring = Circle(
            sim.true_pos, ROBOT_RADIUS + 0.4,
            facecolor="none", edgecolor="#ff3030",
            linewidth=1.4, alpha=0.0, zorder=12)
        self.ax_mini.add_patch(self.mini_dropout_ring)

        # ── Middle-right: yellow-waypoint follow-cam ──────────────
        # 3×3 m view centered on `intermediate_goal_world()` (the
        # world point the robot will actually land on) when belief is
        # close to truth; expands to keep the true GPS goal in frame
        # if the candidate is currently far off. As the EKF refines
        # θ_offset, the view collapses back to the tight 3×3 minimum
        # and both points overlap.
        self.GOAL_VIEW_HALF_MIN = 1.5   # metres → 3×3 m floor
        self.GOAL_VIEW_HALF_MAX = 30.0
        self.ax_goal_mini = self.fig.add_subplot(gs[0, 2])
        self.ax_goal_mini.set_facecolor("#0d1f12")
        self.ax_goal_mini.set_aspect("equal")
        self.ax_goal_mini.set_title(
            f"{2*self.GOAL_VIEW_HALF_MIN:.0f}×{2*self.GOAL_VIEW_HALF_MIN:.0f} m goal-cam (auto-expands)",
            color="#a0a0a0", fontsize=9)
        self.ax_goal_mini.tick_params(colors="#666", labelsize=7)
        for s in self.ax_goal_mini.spines.values():
            s.set_edgecolor("#444")
        self.ax_goal_mini.grid(True, color="#163020",
                                linestyle=":", linewidth=0.4)

        # True goal star + 1 m success ring (fixed in world frame).
        self.goalcam_goal_circle = Circle(
            sim.goal_world, GOAL_RADIUS,
            facecolor=(0.2, 0.9, 0.3, 0.18),
            edgecolor="#33ff66", linewidth=1.0, zorder=3)
        self.ax_goal_mini.add_patch(self.goalcam_goal_circle)
        self.goalcam_goal_inner = Circle(
            sim.goal_world, max(0.0, GOAL_RADIUS - ROBOT_RADIUS),
            fill=False, edgecolor="#33ff66",
            linestyle="--", linewidth=0.9, alpha=0.5, zorder=3)
        self.ax_goal_mini.add_patch(self.goalcam_goal_inner)
        self.goalcam_goal_marker, = self.ax_goal_mini.plot(
            [sim.goal_world[0]], [sim.goal_world[1]],
            "*", color="#33ff66", markersize=14,
            markeredgecolor="white", markeredgewidth=0.4, zorder=11)

        # Dynamic artists. The yellow X is the camera's center so it
        # appears stationary as the world (and the truth-goal star)
        # slides around it; the EKF dot, the cloud mean, recent GPS
        # scatter, and the actual robot body all show their world
        # positions and may or may not be in frame depending on zoom.
        self.goalcam_intermediate, = self.ax_goal_mini.plot(
            [], [], "X", color="#ffe14a", markersize=14,
            markeredgecolor="black", markeredgewidth=0.7, zorder=12)
        self.goalcam_belief_cloud = self.ax_goal_mini.scatter(
            [], [], s=8, c="#ffe14a", alpha=0.20, zorder=6)
        self.goalcam_belief_mean, = self.ax_goal_mini.plot(
            [], [], "P", color="#ff66cc", markersize=10,
            markeredgecolor="black", markeredgewidth=0.5, zorder=10)
        self.goalcam_gps_scatter = self.ax_goal_mini.scatter(
            [], [], s=8, c="#6cd0ff", alpha=0.55, zorder=7)
        self.goalcam_ekf_marker, = self.ax_goal_mini.plot(
            [], [], "o", color="#9bff9b", markersize=7,
            markeredgecolor="black", markeredgewidth=0.5, zorder=11)
        self.goalcam_robot = Circle(sim.true_pos, ROBOT_RADIUS,
                                     facecolor="#ff3030",
                                     edgecolor="white",
                                     linewidth=0.9, zorder=12)
        self.ax_goal_mini.add_patch(self.goalcam_robot)

        # Hide the whole goal-cam axis in multi-agent mode — the
        # yellow waypoint is a single-agent debug aid.
        if self.is_multi:
            self.ax_goal_mini.set_visible(False)

        # ── Lower-right: status panel ─────────────────────────────
        self.ax_status = self.fig.add_subplot(gs[1, 1:])
        self.ax_status.set_facecolor("#141414")
        self.ax_status.axis("off")
        self.ax_status.set_xlim(0, 1); self.ax_status.set_ylim(0, 1)

        self.ax_status.text(
            0.04, 0.985, "GPS Waypoint Sim",
            color="#e0e0e0", fontsize=11, fontweight="bold",
            verticalalignment="top")
        self.ax_status.text(
            0.04, 0.955, "magnetometer-less, EKF-fused",
            color="#888", fontsize=8, fontstyle="italic",
            verticalalignment="top")

        # Side-by-side panels: status (live, green/white) on the left
        # half, setup (static gray) on the right half. Stacking them
        # vertically caused the multi-line status block to overlap
        # the setup block once enough lines were rendered.
        self.status_text = self.ax_status.text(
            0.02, 0.92, "", color="#f0f0f0",
            fontfamily="monospace", fontsize=9,
            verticalalignment="top", linespacing=1.32)

        self.setup_text = self.ax_status.text(
            0.52, 0.92, "", color="#888",
            fontfamily="monospace", fontsize=8,
            verticalalignment="top", linespacing=1.22)

        # Map-feature legend (the three obstacle-style glyphs the user
        # wanted called out).
        feature_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#2a2a2a", markeredgecolor="#888",
                   markersize=10, label="● low-ground obstacle"),
            Line2D([0], [0], marker="^", color="none",
                   markerfacecolor="#3a2f1f", markeredgecolor="#c8a360",
                   markersize=10, label="▲ multipath projector"),
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=(0.45, 0.55, 0.95, 0.4),
                   markeredgecolor="#7099dd",
                   markersize=10, label="■ GPS-blackout roof"),
            Line2D([0], [0], marker="h", color="none",
                   markerfacecolor=(0.85, 0.15, 0.45, 0.30),
                   markeredgecolor="#ff4080",
                   markersize=10, label="⬡ GPS jammer (sparse)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=(0.35, 0.75, 0.35, 0.30),
                   markeredgecolor="#5fbb5f",
                   markersize=10, label="● foliage (noisy)"),
            Line2D([0], [0], marker="D", color="none",
                   markerfacecolor="#cc33ff",
                   markeredgecolor="white",
                   markersize=8, label="◆ GPS spoofer"),
        ]
        # Live-overlay legend (lower-right of the map).
        live_handles = [
            Line2D([0], [0], marker="*", color="none",
                   markerfacecolor="#33ff66", markeredgecolor="white",
                   markersize=11, label="True goal + 1 m ring"),
            Line2D([0], [0], marker="o", color="#ff3030", lw=0,
                   markerfacecolor="#ff3030", markeredgecolor="white",
                   markersize=7, label="Robot (truth)"),
            Line2D([0], [0], color="#ff5050", lw=1.4,
                   label="True trajectory"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#9bff9b", markeredgecolor="black",
                   markersize=6, label="EKF position"),
            Line2D([0], [0], color="#ffd24a", lw=1.2,
                   label="A* plan"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#6cd0ff", markeredgecolor="none",
                   markersize=5, label="GPS fixes"),
            Line2D([0], [0], marker="X", color="none",
                   markerfacecolor="#ffe14a", markeredgecolor="black",
                   markersize=9, label="Intended endpoint"),
            Line2D([0], [0], marker="P", color="none",
                   markerfacecolor="#ff66cc", markeredgecolor="black",
                   markersize=9, label="Cloud mean"),
        ]
        self._features_legend = self.ax.legend(
            handles=feature_handles,
            loc="upper left", fontsize=7,
            facecolor="#0d1f12", labelcolor="#d0d0d0",
            edgecolor="#333", framealpha=0.92,
            handletextpad=0.6, borderpad=0.5, labelspacing=0.4,
            title="map features", title_fontsize=7)
        self.ax.add_artist(self._features_legend)
        self.ax.legend(
            handles=live_handles,
            loc="lower right", fontsize=7, ncol=1,
            facecolor="#0d1f12", labelcolor="#d0d0d0",
            edgecolor="#333", framealpha=0.92,
            handletextpad=0.6, borderpad=0.5,
            labelspacing=0.4)

        self.fig.subplots_adjust(left=0.05, right=0.985,
                                  top=0.95, bottom=0.07)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 30 FPS render cadence — decoupled from SIM_DT (10 Hz physics).
        # STEPS_PER_FRAME = 1 so each timer fire = one physics tick,
        # which keeps the sim:wall ratio identical to the old 10 FPS
        # × 3-step config while tripling visual smoothness.
        self._timer = self.fig.canvas.new_timer(
            interval=GUI_FRAME_DT_MS)
        self._timer.add_callback(self._tick)
        self._build_scenario_patches()
        self._refresh_static_status_header()
        self._render_dynamic()

    # ── Scenario lifecycle ───────────────────────────────────────
    def _build_scenario_patches(self):
        """Add the obstacle/roof/projector patches for the current
        scenario to both the main map and the follow-cam, and remember
        them so reset can tear them down."""
        for cx, cy, r in self.obstacles:
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                p = ax.add_patch(Circle(
                    (cx, cy), r, facecolor="#2a2a2a",
                    edgecolor="#555", linewidth=0.6, zorder=2))
                self._scenario_artists.append(p)
        for x_min, y_min, x_max, y_max in self.roofs:
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                p = ax.add_patch(Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    facecolor=(0.45, 0.55, 0.95, 0.13),
                    edgecolor="#7099dd", linewidth=0.9,
                    linestyle="--", zorder=2.4))
                self._scenario_artists.append(p)
        for verts, bias in self.projectors:
            cx, cy, _ = projector_centroid_radius(verts)
            # Orange dotted influence ring (drawn first / behind)
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                ring = ax.add_patch(Circle(
                    (cx, cy), PROJECTOR_INFLUENCE_RADIUS_M,
                    fill=False, edgecolor="#ff9933",
                    linestyle=":", linewidth=0.9, alpha=0.55,
                    zorder=2.45))
                self._scenario_artists.append(ring)
            # Triangle on top
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                p = ax.add_patch(Polygon(
                    list(verts), closed=True,
                    facecolor="#3a2f1f", edgecolor="#c8a360",
                    linewidth=1.0, zorder=2.5))
                self._scenario_artists.append(p)
            ann = self.ax.annotate(
                "", xy=(cx + bias[0], cy + bias[1]), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#c8a360",
                                 lw=0.8, alpha=0.55), zorder=2.6)
            self._scenario_artists.append(ann)
        # Foliage — soft green discs.
        for cx, cy, r in self.foliage:
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                p = ax.add_patch(Circle(
                    (cx, cy), r,
                    facecolor=(0.35, 0.75, 0.35, 0.16),
                    edgecolor="#5fbb5f", linewidth=0.8,
                    linestyle=":", zorder=2.3))
                self._scenario_artists.append(p)
        # Hex jammers — red-magenta hex outlines.
        for cx, cy, r in self.jammers:
            verts = hex_vertices(cx, cy, r)
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                p = ax.add_patch(Polygon(
                    verts, closed=True,
                    facecolor=(0.85, 0.15, 0.45, 0.10),
                    edgecolor="#ff4080", linewidth=1.0,
                    linestyle="--", zorder=2.35))
                self._scenario_artists.append(p)
        # Spoofers — magenta diamond at the spoofer, dashed magenta
        # arrow to the fake target so the lie is visible at a glance.
        for (cx, cy), (fx, fy) in self.spoofers:
            for ax in (self.ax, self.ax_mini, self.ax_goal_mini):
                ring = ax.add_patch(Circle(
                    (cx, cy), SPOOFER_INFLUENCE_RADIUS_M,
                    fill=False, edgecolor="#cc33ff",
                    linestyle=":", linewidth=0.9, alpha=0.6,
                    zorder=2.45))
                self._scenario_artists.append(ring)
                marker, = ax.plot(
                    [cx], [cy], marker="D",
                    markerfacecolor="#cc33ff",
                    markeredgecolor="white",
                    markersize=8, linestyle="None", zorder=2.6)
                self._scenario_artists.append(marker)
            ann = self.ax.annotate(
                "", xy=(fx, fy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#cc33ff",
                                 lw=0.9, alpha=0.7,
                                 linestyle="dashed"), zorder=2.6)
            self._scenario_artists.append(ann)
            fake_marker, = self.ax.plot(
                [fx], [fy], marker="x",
                color="#cc33ff", markersize=8,
                markeredgewidth=1.5, linestyle="None", zorder=2.6)
            self._scenario_artists.append(fake_marker)

    def _clear_scenario_patches(self):
        for p in self._scenario_artists:
            try:
                p.remove()
            except Exception:
                pass
        self._scenario_artists = []

    def _reload_scenario(self, sim, obstacles, roofs, projectors,
                          jammers=(), foliage=(), spoofers=(),
                          new_peers=None):
        """Hot-swap the scenario without rebuilding the figure or the
        animation timer — this is what R-reset uses to avoid leaking
        figures and timers on every press."""
        self._clear_scenario_patches()
        self.sim = sim
        if new_peers is not None:
            # Tear down stale peer artists
            for body in self.peer_bodies:
                try: body.remove()
                except Exception: pass
            for trail in self.peer_trails:
                try: trail.remove()
                except Exception: pass
            if self.peer_scatter is not None:
                try: self.peer_scatter.remove()
                except Exception: pass
                self.peer_scatter = None
            if self.peer_trail_lc is not None:
                try: self.peer_trail_lc.remove()
                except Exception: pass
                self.peer_trail_lc = None
            self.peer_bodies = []
            self.peer_trails = []
            self.peers = list(new_peers)
            self.all_sims = [sim] + self.peers
            self.is_multi = len(self.all_sims) > 1
            self.heavy_multi = len(self.peers) >= self.HEAVY_MULTI_THRESHOLD
            if self.heavy_multi:
                if self.peers:
                    init_xy = np.array(
                        [(p.true_pos[0], p.true_pos[1]) for p in self.peers],
                        dtype=float)
                else:
                    init_xy = np.empty((0, 2), dtype=float)
                self.peer_scatter = self.ax.scatter(
                    init_xy[:, 0] if len(init_xy) else [],
                    init_xy[:, 1] if len(init_xy) else [],
                    s=8, c="#ff7777", edgecolors="none",
                    alpha=0.75, zorder=11)
                self.peer_trail_lc = LineCollection(
                    [], colors="#ff5050", linewidths=0.5,
                    alpha=0.25, zorder=7)
                self.ax.add_collection(self.peer_trail_lc)
            else:
                for p in self.peers:
                    body = Circle(p.true_pos, ROBOT_RADIUS,
                                  facecolor="#ff7777", edgecolor="white",
                                  linewidth=0.4, alpha=0.85, zorder=11)
                    self.ax.add_patch(body)
                    trail, = self.ax.plot(
                        [], [], "-", color="#ff5050",
                        linewidth=0.55, alpha=0.30, zorder=7)
                    self.peer_bodies.append(body)
                    self.peer_trails.append(trail)
            # Mirror EXACTLY the artists that ``__init__`` hides when
            # ``self.is_multi`` is True. Anything left out here would
            # be visible-on-init but hidden-after-R (or vice versa)
            # across the same mode — the asymmetry the user reported.
            # ``heading_arrow`` is intentionally NOT in this list:
            # the truth heading arrow stays visible in both single
            # and multi mode (it tracks the primary agent and is
            # informative even alongside peer dots).
            multi_only_hide = (self.path_line, self.window_patch,
                                 self.gps_scatter, self.belief_cloud,
                                 self.belief_mean_marker,
                                 self.intermediate_marker,
                                 self.ekf_marker,
                                 self.compass_ring,
                                 self.compass_real_arrow,
                                 self.compass_est_arrow,
                                 self.compass_real_dot,
                                 self.compass_est_dot,
                                 self.compass_real_label,
                                 self.compass_est_label,
                                 self.ekf_heading_arrow)
            for art in multi_only_hide:
                art.set_visible(not self.is_multi)
            # Always-visible across modes — pinned in case some prior
            # state had toggled them off and the toggle never got
            # re-asserted on R.
            self.heading_arrow.set_visible(True)
            self.robot_body.set_visible(True)
            self.trail_line.set_visible(True)
        self.obstacles = list(obstacles)
        self.roofs = list(roofs)
        self.projectors = list(projectors)
        self.jammers = list(jammers)
        self.foliage = list(foliage)
        self.spoofers = list(spoofers)
        self._build_scenario_patches()
        # Reset dynamic artists' data
        self.trail_line.set_data([], [])
        self.path_line.set_data([], [])
        self.window_patch.set_xy((0, 0))
        self.window_patch.set_width(0); self.window_patch.set_height(0)
        self.gps_scatter.set_offsets(np.empty((0, 2)))
        self.belief_cloud.set_offsets(np.empty((0, 2)))
        self.belief_mean_marker.set_data([], [])
        self.intermediate_marker.set_data([], [])
        self.ekf_marker.set_data([], [])
        self.mini_path_line.set_data([], [])
        self.mini_gps_scatter.set_offsets(np.empty((0, 2)))
        self.mini_trail_line.set_data([], [])
        self.mini_intermediate_marker.set_data([], [])
        self.mini_ekf_marker.set_data([], [])
        # Goal marker / circles update positions
        self.goal_circle.center = sim.goal_world
        self.goal_inner_ring.center = sim.goal_world
        self.mini_goal_circle.center = sim.goal_world
        self.mini_goal_inner_ring.center = sim.goal_world
        self.goal_marker.set_data([sim.goal_world[0]], [sim.goal_world[1]])
        self.mini_goal_marker.set_data(
            [sim.goal_world[0]], [sim.goal_world[1]])
        # Goal-cam's green star + ring are anchored on world coords
        # too — without this, pressing R repeatedly leaves them at
        # the original goal location while the new scenario uses a
        # different goal, and the cam never finds the green star.
        self.goalcam_goal_circle.center = sim.goal_world
        self.goalcam_goal_inner.center = sim.goal_world
        self.goalcam_goal_marker.set_data(
            [sim.goal_world[0]], [sim.goal_world[1]])
        # Move the start cross to the new scenario's start.
        self.start_marker.set_data(
            [sim.start_world[0]], [sim.start_world[1]])
        self._last_motion_dir = sim.true_heading
        self._refresh_static_status_header()
        self._render_dynamic()
        # Static scenery changed — invalidate the blit cache so the
        # next ``draw_event`` callback re-snapshots a fresh background
        # for the new obstacle / roof / projector layout.
        self._ax_bg = None
        # Pre-mark the new peer_bodies / peer_trails / peer_scatter /
        # peer_trail_lc animated NOW so the next ``canvas.draw()``
        # excludes them from the static bg snapshot. Without this,
        # there's a one-frame window where the new peer artists are
        # baked into ``_ax_bg`` (since ``set_animated(True)`` happens
        # later inside ``_setup_blit``) and the subsequent blit's
        # ``draw_artist`` paints a SECOND copy on top — the classic
        # double-render. The next ``_setup_blit`` will re-set
        # animated on every artist (idempotent), so this is a safety
        # net for the gap between R and the first paint event.
        if getattr(self, "_blit_supported", True):
            for body in self.peer_bodies:
                try: body.set_animated(True)
                except Exception: pass
            for trail in self.peer_trails:
                try: trail.set_animated(True)
                except Exception: pass
            if self.peer_scatter is not None:
                try: self.peer_scatter.set_animated(True)
                except Exception: pass
            if self.peer_trail_lc is not None:
                try: self.peer_trail_lc.set_animated(True)
                except Exception: pass
        self.fig.canvas.draw_idle()

    def _refresh_static_status_header(self):
        sim = self.sim
        gx_lat, gx_lon = meters_to_latlon(*sim.goal_world)
        sx_lat, sx_lon = meters_to_latlon(*sim.start_world)
        self._setup = (
            f"── setup ──────────────\n"
            f"Map     {MAP_FT:.0f} ft / {MAP_M:.0f} m, {RES:.2f} m grid\n"
            f"Features {len(self.obstacles):>2} obs, "
            f"{len(self.projectors):>2} proj, {len(self.roofs):>2} roofs\n"
            f"GPS     {GPS_SAMPLE_HZ:.0f} Hz, σ={GPS_NOISE_STD:.2f} m\n"
            f"        bias {GPS_BIAS_AMPL_M:.1f} m / {GPS_BIAS_PERIOD_S:.0f} s\n"
            f"        outlier {GPS_OUTLIER_PROB*GPS_SAMPLE_HZ:.3f}/s, σ={GPS_OUTLIER_STD:.0f} m\n"
            f"        dropout {GPS_DROPOUT_HZ_PER_S:.2f}/s, "
            f"{GPS_DROPOUT_DURATION_S[0]:.0f}-{GPS_DROPOUT_DURATION_S[1]:.0f}s\n"
            f"Max v   {MAX_SPEED_MPH:.1f} mph "
            f"({MAX_SPEED_MPS:.2f} m/s)\n"
            f"Goal    ({gx_lat:.5f}, {gx_lon:.5f})\n"
            f"True θ  {math.degrees(sim.true_heading):+7.2f}°  hidden\n"
        )

    def _render_dynamic(self):
        """Full per-frame refresh. Kept for callers that aren't part
        of the blit fast-path (R-reset, initial draw)."""
        self._render_dynamic_main()
        self._render_dynamic_panels()

    def _render_dynamic_main(self):
        """Main-map artists only — the set the blit fast-path stamps
        every tick. No mini-cam axes-limit changes (those live in
        ``_render_dynamic_panels``) so a blit-only frame doesn't dirty
        the cached background by re-running the goal-cam autozoom."""
        sim = self.sim
        # Per-peer body + trail (no-op for single-agent runs).
        if self.heavy_multi and self.peer_scatter is not None:
            # One vectorized set_offsets per frame for bodies, plus a
            # single LineCollection.set_segments for all trails.
            xy = np.empty((len(self.peers), 2), dtype=float)
            for i, p in enumerate(self.peers):
                xy[i, 0] = p.true_pos[0]
                xy[i, 1] = p.true_pos[1]
            self.peer_scatter.set_offsets(xy)
            if self.peer_trail_lc is not None:
                tail = self.PEER_TRAIL_TAIL
                segs = []
                for p in self.peers:
                    tr = p.true_trail
                    if len(tr) >= 2:
                        segs.append(np.asarray(tr[-tail:]))
                self.peer_trail_lc.set_segments(segs)
        else:
            for body, trail, p in zip(self.peer_bodies,
                                        self.peer_trails, self.peers):
                body.center = (p.true_pos[0], p.true_pos[1])
                if p.true_trail:
                    trail.set_data([q[0] for q in p.true_trail],
                                    [q[1] for q in p.true_trail])
        # Robot body + travel-direction arrow (tangent to the path).
        # Velocity is in odom frame; rotate by true_heading for world.
        self.robot_body.center = (sim.true_pos[0], sim.true_pos[1])
        # Chaplygin sleigh — show body heading in world (knife-edge
        # axis), not motion direction. The body can be stationary or
        # rotating-in-place and still have a meaningful orientation.
        motion_dir = sim.body_heading_world
        self._last_motion_dir = motion_dir
        ah = 1.5
        self.heading_arrow.xy = (
            sim.true_pos[0] + ah * math.cos(motion_dir),
            sim.true_pos[1] + ah * math.sin(motion_dir))
        self.heading_arrow.set_position(
            (sim.true_pos[0], sim.true_pos[1]))

        # EKF heading arrow — anchored at ekf.pos, pointing along
        # the body's BELIEVED world heading (= body_heading +
        # ekf.theta). With healthy IMU fusion in the local EKF,
        # this overlaps the truth arrow; any visible gap is the
        # residual θ-estimate error feeding back through the
        # controller.
        if sim.ekf is not None:
            ex, ey = sim.ekf.pos_xy
            est_dir = sim.body_heading_world_est
            self.ekf_heading_arrow.xy = (
                ex + ah * math.cos(est_dir),
                ey + ah * math.sin(est_dir))
            self.ekf_heading_arrow.set_position((ex, ey))
        else:
            self.ekf_heading_arrow.xy = (0, 0)
            self.ekf_heading_arrow.set_position((0, 0))

        # Compass clock-hands (real vs estimated north). Real
        # north always +y; estimated north rotated by the EKF's
        # residual heading error. When the two diverge, the gap
        # between the hands IS the wrong-convergence's root cause.
        cx0, cy0 = self.COMPASS_CENTER
        L = self.COMPASS_LENGTH
        # TRUE NORTH — fixed +y in world.
        true_tip_x, true_tip_y = cx0, cy0 + L
        self.compass_real_arrow.xy = (true_tip_x, true_tip_y)
        self.compass_real_arrow.set_position((cx0, cy0))
        self.compass_real_dot.set_data(
            [true_tip_x], [true_tip_y])
        self.compass_real_label.set_position(
            (true_tip_x, true_tip_y + 1.4))
        # MAP NORTH — rotated by the EKF's residual θ error.
        theta_err = sim.true_heading - sim.heading_offset_est
        # R(theta_err) · (0, 1) = (-sin, cos)
        ex_dir = -math.sin(theta_err)
        ey_dir = math.cos(theta_err)
        est_tip_x = cx0 + L * ex_dir
        est_tip_y = cy0 + L * ey_dir
        self.compass_est_arrow.xy = (est_tip_x, est_tip_y)
        self.compass_est_arrow.set_position((cx0, cy0))
        self.compass_est_dot.set_data([est_tip_x], [est_tip_y])
        self.compass_est_label.set_position(
            (est_tip_x, est_tip_y + 1.4))

        # True trail
        if sim.true_trail:
            tx = [p[0] for p in sim.true_trail]
            ty = [p[1] for p in sim.true_trail]
            self.trail_line.set_data(tx, ty)

        # A* path (planned in world frame from ekf.pos to the
        # published candidate).
        if sim.path_world:
            px = [p[0] for p in sim.path_world]
            py = [p[1] for p in sim.path_world]
            self.path_line.set_data(px, py)
        else:
            self.path_line.set_data([], [])

        # Smart-padding window
        if sim.last_window is not None:
            x0, y0, x1, y1 = sim.last_window
            self.window_patch.set_xy((x0, y0))
            self.window_patch.set_width(x1 - x0)
            self.window_patch.set_height(y1 - y0)

        # GPS scatter
        if sim.gps_scatter:
            self.gps_scatter.set_offsets(np.array(sim.gps_scatter))
        else:
            self.gps_scatter.set_offsets(np.empty((0, 2)))

        # Goal-belief cloud + running mean
        if sim.intended_endpoint_history:
            arr = np.array(sim.intended_endpoint_history)
            self.belief_cloud.set_offsets(arr)
            mx, my = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            self.belief_mean_marker.set_data([mx], [my])
        else:
            self.belief_cloud.set_offsets(np.empty((0, 2)))
            self.belief_mean_marker.set_data([], [])

        # EKF position estimate
        if sim.ekf is not None:
            ex, ey = sim.ekf.pos_xy
            self.ekf_marker.set_data([ex], [ey])
        else:
            self.ekf_marker.set_data([], [])

        # Intermediate goal
        # Yellow X tracks the *published* goal (what NAV2 / A* are
        # actually driving toward), not the live per-tick candidate.
        # The live candidate is implicit in the cloud's volatility.
        ig = sim.published_goal_world
        self.intermediate_marker.set_data([ig[0]], [ig[1]])

        # Dropout indicator
        if not sim.gps_connected:
            self.dropout_ring.center = (sim.true_pos[0], sim.true_pos[1])
            self.dropout_ring.set_alpha(1.0)
        else:
            self.dropout_ring.set_alpha(0.0)

    def _render_dynamic_panels(self):
        """Mini follow-cam, goal-cam, and status text. Gated to every
        Nth tick by ``_tick`` so the main-map blit can run at full
        cadence — these axes share the figure canvas with the main
        map, so refreshing them forces a real ``draw_idle`` that
        invalidates the blit cache."""
        sim = self.sim
        ig = sim.published_goal_world
        # ``motion_dir`` is computed in ``_render_dynamic_main`` (the
        # body-heading-in-world the truth arrow uses) and cached on
        # ``self._last_motion_dir`` so this gated panel refresh can
        # consume it without recomputing.
        motion_dir = getattr(self, "_last_motion_dir", sim.body_heading_world)
        # ── Follow-cam (5×5 m around robot, robot-local frame) ──────
        # The cam axes' xlim/ylim are fixed at ±MINI_VIEW_HALF (set
        # once at panel construction). Each frame we offset every
        # artist's data by ``-robot_pos`` so the robot stays at (0, 0)
        # and the world slides smoothly underneath it. No xlim change
        # per frame → no bg re-snapshot → no dead-zone jump.
        rx, ry = sim.true_pos
        ah = 0.9
        # Robot stays at origin
        self.mini_robot.center = (0.0, 0.0)
        self.mini_heading_arrow.xy = (ah * math.cos(motion_dir),
                                       ah * math.sin(motion_dir))
        self.mini_heading_arrow.set_position((0.0, 0.0))

        if sim.true_trail:
            self.mini_trail_line.set_data(
                [p[0] - rx for p in sim.true_trail],
                [p[1] - ry for p in sim.true_trail])
        else:
            self.mini_trail_line.set_data([], [])

        if sim.path_world:
            self.mini_path_line.set_data(
                [p[0] - rx for p in sim.path_world],
                [p[1] - ry for p in sim.path_world])
        else:
            self.mini_path_line.set_data([], [])

        if sim.gps_scatter:
            arr = np.asarray(sim.gps_scatter, dtype=float)
            arr = arr - np.array([rx, ry])
            self.mini_gps_scatter.set_offsets(arr)
        else:
            self.mini_gps_scatter.set_offsets(np.empty((0, 2)))

        if sim.ekf is not None:
            ex, ey = sim.ekf.pos_xy
            self.mini_ekf_marker.set_data([ex - rx], [ey - ry])
        else:
            self.mini_ekf_marker.set_data([], [])

        self.mini_intermediate_marker.set_data([ig[0] - rx], [ig[1] - ry])

        # Goal star + 1 m ring (offset into robot-local frame).
        gx_w, gy_w = sim.goal_world
        self.mini_goal_circle.center = (gx_w - rx, gy_w - ry)
        self.mini_goal_inner_ring.center = (gx_w - rx, gy_w - ry)
        self.mini_goal_marker.set_data([gx_w - rx], [gy_w - ry])

        if not sim.gps_connected:
            self.mini_dropout_ring.center = (0.0, 0.0)
            self.mini_dropout_ring.set_alpha(1.0)
        else:
            self.mini_dropout_ring.set_alpha(0.0)

        # ── Goal-cam (3×3 m floor, expands so goal stays visible) ─
        # Goal-cam framing.
        #
        # Default behaviour (non-field-parity): center halfway
        # between the yellow waypoint (intended endpoint) and the
        # true goal so both stay in frame; half-extent grows to
        # contain the spread.
        #
        # Field-parity (ROBOT_STRICT_ARRIVAL=True): the candidate
        # marker is rotated by R(true_heading) and the agent's
        # *true* world position diverges from the candidate by
        # encoder drift. The user wants the goal-cam to *follow
        # the intended endpoint* — that's the action server's
        # target — and have the agent visibly approach it. So we
        # center on the candidate and grow the extent only enough
        # to keep the agent in frame.
        if not self.is_multi:
            gx, gy = ig
            tx, ty = sim.goal_world
            if ROBOT_STRICT_ARRIVAL:
                # Center on the candidate (intended endpoint).
                cx_gc, cy_gc = gx, gy
                # Keep the true robot dot in frame too, so the
                # divergence stays visible.
                rx, ry = sim.true_pos
                spread_robot = math.hypot(rx - gx, ry - gy)
                half_gc = max(self.GOAL_VIEW_HALF_MIN,
                              min(self.GOAL_VIEW_HALF_MAX,
                                  spread_robot * 0.6 + 1.0))
            else:
                cx_gc = 0.5 * (gx + tx)
                cy_gc = 0.5 * (gy + ty)
                spread = math.hypot(gx - tx, gy - ty)
                half_gc = max(self.GOAL_VIEW_HALF_MIN,
                               min(self.GOAL_VIEW_HALF_MAX,
                                    spread * 0.5 + 0.6))
            # Goal-cam recenter: dead-zone on EITHER the center or
            # the half-extent. Half-extent can grow / shrink as the
            # candidate goal moves vs the true goal, so we trip the
            # recenter when either drifts past a fraction of the
            # current cam window.
            need_recenter = self._goal_center is None or self._goal_half is None
            if not need_recenter:
                dz_c = self._goal_half * 0.4
                dz_h = self._goal_half * 0.5
                need_recenter = (
                    abs(cx_gc - self._goal_center[0]) > dz_c or
                    abs(cy_gc - self._goal_center[1]) > dz_c or
                    abs(half_gc - self._goal_half) > dz_h)
            if need_recenter:
                self.ax_goal_mini.set_xlim(cx_gc - half_gc,
                                            cx_gc + half_gc)
                self.ax_goal_mini.set_ylim(cy_gc - half_gc,
                                            cy_gc + half_gc)
                self._goal_center = (cx_gc, cy_gc)
                self._goal_half = half_gc
                self._panel_recenter_needed = True
            self.goalcam_intermediate.set_data([gx], [gy])
            self.goalcam_robot.center = (sim.true_pos[0],
                                          sim.true_pos[1])
            if sim.ekf is not None:
                ex, ey = sim.ekf.pos_xy
                self.goalcam_ekf_marker.set_data([ex], [ey])
            else:
                self.goalcam_ekf_marker.set_data([], [])
            if sim.gps_scatter:
                self.goalcam_gps_scatter.set_offsets(
                    np.array(sim.gps_scatter))
            else:
                self.goalcam_gps_scatter.set_offsets(np.empty((0, 2)))
            if sim.intended_endpoint_history:
                arr = np.array(sim.intended_endpoint_history)
                self.goalcam_belief_cloud.set_offsets(arr)
                self.goalcam_belief_mean.set_data(
                    [float(arr[:, 0].mean())],
                    [float(arr[:, 1].mean())])
            else:
                self.goalcam_belief_cloud.set_offsets(
                    np.empty((0, 2)))
                self.goalcam_belief_mean.set_data([], [])

        # ── Status text ───────────────────────────────────────
        true_dist = math.hypot(sim.true_pos[0] - sim.goal_world[0],
                               sim.true_pos[1] - sim.goal_world[1])
        heading_err = math.degrees(
            (sim.true_heading - sim.heading_offset_est + math.pi)
            % (2 * math.pi) - math.pi)
        speed_mps = math.hypot(sim.odom_vel[0], sim.odom_vel[1])
        speed_mph = speed_mps / 0.44704
        overlap = sim.goal_overlap_fraction()

        # Highlight the goal ring when the robot is overlapping it —
        # turns the visualization into a "passed?" indicator.
        if overlap >= 0.5:
            self.goal_circle.set_facecolor((0.20, 0.95, 0.35, 0.42))
            self.goal_circle.set_edgecolor("#aaffbb")
            self.mini_goal_circle.set_facecolor((0.20, 0.95, 0.35, 0.42))
            self.mini_goal_circle.set_edgecolor("#aaffbb")
        elif overlap > 0.0:
            self.goal_circle.set_facecolor((0.20, 0.90, 0.30, 0.28))
            self.goal_circle.set_edgecolor("#88ee88")
            self.mini_goal_circle.set_facecolor((0.20, 0.90, 0.30, 0.28))
            self.mini_goal_circle.set_edgecolor("#88ee88")
        else:
            self.goal_circle.set_facecolor((0.20, 0.90, 0.30, 0.18))
            self.goal_circle.set_edgecolor("#33ff66")
            self.mini_goal_circle.set_facecolor((0.20, 0.90, 0.30, 0.18))
            self.mini_goal_circle.set_edgecolor("#33ff66")

        if sim.arrived and not sim.coasting:
            state = "✓ PASSED"
            state_color = "#33ff66"
        elif sim.coasting:
            state = "✓ PASSED — coasting"
            state_color = "#33ff66"
        else:
            state = "navigating"
            state_color = "#e0e0e0"
        if sim.gps_reconnect_active:
            t_left = max(0.0, sim._gps_reconnect_until - sim.sim_time)
            gps_state = f"↻ reconnect ({t_left:.1f}s)"
        elif sim.gps_connected:
            gps_state = "● live"
        else:
            gps_state = "✕ DROPOUT"

        cloud_n = len(sim.intended_endpoint_history)
        if cloud_n:
            arr = np.array(sim.intended_endpoint_history)
            mx, my = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            mean_dist = math.hypot(mx - sim.goal_world[0],
                                   my - sim.goal_world[1])
        else:
            mean_dist = float("nan")

        ekf_pstd = sim.ekf.pos_std if sim.ekf else (0.0, 0.0)
        ekf_tstd = math.degrees(sim.ekf.theta_std_rad) if sim.ekf else 0.0
        ekf_rej = sim.ekf.rejected_count if sim.ekf else 0
        ekf_upd = sim.ekf.update_count if sim.ekf else 0
        boot = "✓" if sim.bootstrap_done else "…"
        # Heading-EKF "converged" flag for the status line: bootstrap is
        # finished AND σ_θ is tight enough that the published candidate
        # has stopped wandering. 5° σ_θ matches the resync threshold's
        # rough order so the indicator flips on the same beat the
        # operator can see the cloud freeze. Purely a display flag —
        # no algorithm path looks at this.
        if sim.ekf is None:
            converged = "n/a"
        elif sim.bootstrap_done and ekf_tstd <= 5.0:
            converged = "✓"
        else:
            converged = "…"

        path_n = len(sim.path_world) if sim.path_world else 0

        # Aggregate ensemble metrics (only meaningful with peers).
        if self.is_multi:
            arrived = sum(1 for s in self.all_sims if s.arrived)
            true_dists = [
                math.hypot(s.true_pos[0] - s.goal_world[0],
                            s.true_pos[1] - s.goal_world[1])
                for s in self.all_sims]
            mean_d = sum(true_dists) / len(true_dists)
            min_d  = min(true_dists)
            max_d  = max(true_dists)
            in_goal = sum(1 for s in self.all_sims
                           if s.goal_overlap_fraction() >= 0.5)
            ens_lines = (
                f"\n"
                f"── ensemble ───────────\n"
                f"arrived    {arrived:>3} / {len(self.all_sims):<3}\n"
                f"in goal    {in_goal:>3} / {len(self.all_sims):<3}\n"
                f"dist→goal  mean {mean_d:5.2f} m\n"
                f"           min  {min_d:5.2f} m\n"
                f"           max  {max_d:5.2f} m\n"
            )
        else:
            ens_lines = ""

        primary_label = "primary" if self.is_multi else "agent"
        # Chained-mission leg indicator. Only surfaces when the
        # agent is actually running a multi-leg mission (regression
        # gate for single-goal default behavior). Bare "leg N/M"
        # line keeps the visual cheap.
        if getattr(sim, "leg_count", 1) > 1:
            mission_lines = (
                f"mission         "
                f"leg {sim.leg_index}/{sim.leg_count}\n"
            )
        else:
            mission_lines = ""
        body = (
            f"t = {sim.sim_time:6.1f} s   {state}\n"
            f"GPS: {gps_state}\n"
            + ens_lines +
            f"\n"
            f"── {primary_label} ─────────────\n"
            + mission_lines +
            f"dist → goal     {true_dist:6.2f} m\n"
            f"in goal         {overlap*100:5.1f} %\n"
            f"speed           {speed_mph:5.2f} mph\n"
            f"heading error   {heading_err:+6.2f}°\n"
            f"\n"
            f"── EKF ────────────────\n"
            f"σ_θ             {ekf_tstd:6.2f}°\n"
            f"σ_pos       {ekf_pstd[0]:4.2f}, {ekf_pstd[1]:4.2f} m\n"
            f"updates / rej   {ekf_upd:>4} / {ekf_rej}\n"
            f"bootstrap       {boot}\n"
            f"converged       {converged}\n"
            f"\n"
            f"── cloud ──────────────\n"
            f"n={cloud_n:>4}    mean→goal {mean_dist:5.2f} m\n"
            f"\n"
            f"── A* ────────────────\n"
            f"{path_n:>4} pts    pad {sim.last_pad:4.1f} m\n"
        )
        self.status_text.set_text(body)
        self.status_text.set_color(state_color if (sim.arrived or sim.coasting) else "#f0f0f0")
        self.setup_text.set_text(self._setup)

    # ── Blit fast-path ──────────────────────────────────────────
    def _collect_mini_animated_artists(self):
        """Follow-cam (``ax_mini``) dynamic artists. The cam re-centers
        only when the robot leaves the dead zone (see ``_render_dynamic_panels``);
        between recenters every frame just blits these artists in place.
        Sorted by zorder — see ``_collect_main_animated_artists`` for
        the live-vs-static parity rationale."""
        arts = [self.mini_path_line, self.mini_gps_scatter,
                self.mini_trail_line, self.mini_robot,
                self.mini_heading_arrow, self.mini_ekf_marker,
                self.mini_intermediate_marker, self.mini_dropout_ring,
                self.mini_goal_circle, self.mini_goal_inner_ring,
                self.mini_goal_marker]
        arts.sort(key=lambda a: a.get_zorder())
        return arts

    def _collect_goal_mini_animated_artists(self):
        """Goal-cam (``ax_goal_mini``) dynamic artists. Sorted by zorder
        to match ``Axes.draw``'s per-tick stacking."""
        if self.is_multi:
            return []
        arts = [self.goalcam_intermediate, self.goalcam_belief_cloud,
                self.goalcam_belief_mean, self.goalcam_gps_scatter,
                self.goalcam_ekf_marker, self.goalcam_robot,
                self.goalcam_goal_circle, self.goalcam_goal_inner,
                self.goalcam_goal_marker]
        arts.sort(key=lambda a: a.get_zorder())
        return arts

    def _collect_status_animated_artists(self):
        """Status text panel (``ax_status``) dynamic artists."""
        return [self.status_text, self.setup_text]

    def _collect_main_animated_artists(self):
        """Artists on the main map (``self.ax``) that change per-frame.
        These get ``set_animated(True)`` so the cached background
        excludes them, and a per-frame ``draw_artist`` stamps them on
        top. Anything not in this list lives in the static snapshot.

        Sorted by zorder so the per-frame ``draw_artist`` sequence in
        ``_blit_main`` paints in the same order matplotlib uses for a
        full ``Axes.draw`` (which zorder-sorts its children). Without
        the sort, peer_bodies (z=11) appended last in the literal list
        would paint OVER compass labels (z=16) on the live screen but
        UNDER them in static savefig — producing exactly the "two
        things being rendered" mismatch between live and snapshot.
        """
        arts = [self.path_line, self.window_patch,
                self.gps_scatter, self.belief_cloud,
                self.belief_mean_marker,
                self.trail_line, self.robot_body,
                self.heading_arrow, self.ekf_marker,
                self.ekf_heading_arrow, self.intermediate_marker,
                self.dropout_ring,
                # Compass clock-hands rotate as the EKF refines θ.
                self.compass_real_arrow, self.compass_est_arrow,
                self.compass_real_dot, self.compass_est_dot,
                self.compass_real_label, self.compass_est_label,
                self.goal_circle, self.goal_inner_ring,
                self.goal_marker, self.start_marker]
        # Per-peer bodies/trails (heavy-multi uses a single
        # scatter + LineCollection — both also animated).
        if self.peer_scatter is not None:
            arts.append(self.peer_scatter)
        if self.peer_trail_lc is not None:
            arts.append(self.peer_trail_lc)
        arts.extend(self.peer_bodies)
        arts.extend(self.peer_trails)
        # Stable sort by zorder so the blit draw order matches
        # ``Axes.draw``'s zorder-sorted iteration.
        arts.sort(key=lambda a: a.get_zorder())
        return arts

    def _setup_blit(self):
        """Mark dynamic artists on all four panels animated, force a
        full canvas draw to lay down the static backgrounds, then
        snapshot ``ax.bbox`` / ``ax_mini.bbox`` / ``ax_goal_mini.bbox``
        / ``ax_status.bbox`` for per-axes blit restores. Called once
        at startup and whenever any axes' static content changes
        (R-reset, follow-cam recenter, goal-cam recenter).

        Re-entrancy guard: ``self.fig.canvas.draw()`` below fires a
        ``draw_event`` synchronously, which calls back into
        ``_on_first_draw`` → ``_setup_blit`` again. Without this guard
        each startup runs the full setup pipeline twice (once outer,
        once inner) and the inner copy_from_bbox captures the bg BEFORE
        the outer set_animated calls finish — the outer call then
        overwrites with a re-snapshot, but for one frame between the
        recursion unwinding and the next paint the screen can show
        the bg with animated artists still baked in. Skip the inner
        call cleanly."""
        if not getattr(self, "_blit_supported", True):
            return
        if getattr(self, "_in_setup_blit", False):
            return
        self._in_setup_blit = True
        all_artists = (self._collect_main_animated_artists()
                       + self._collect_mini_animated_artists()
                       + self._collect_goal_mini_animated_artists()
                       + self._collect_status_animated_artists())
        try:
            for art in all_artists:
                art.set_animated(True)
            self.fig.canvas.draw()
            cb = self.fig.canvas.copy_from_bbox
            self._ax_bg = cb(self.ax.bbox)
            self._mini_bg = cb(self.ax_mini.bbox)
            self._goal_mini_bg = (cb(self.ax_goal_mini.bbox)
                                   if not self.is_multi else None)
            self._status_bg = cb(self.ax_status.bbox)
            self._blit_supported = True
        except Exception:
            self._blit_supported = False
            for art in all_artists:
                try: art.set_animated(False)
                except Exception: pass
            self._ax_bg = None
            self._mini_bg = None
            self._goal_mini_bg = None
            self._status_bg = None
        finally:
            self._in_setup_blit = False

    def _blit_main(self):
        """Single-blit atomic redraw of all four panels.

        Issuing ``canvas.blit(axes.bbox)`` four times per frame caused
        a visible mouse-trail effect: each blit is a separate Qt-level
        partial update, and between them the user momentarily saw
        in-between canvas states. Instead, restore each panel's bg +
        draw_artist its animated artists in sequence (modifying the
        backbuffer only), then push the whole figure to screen with
        ONE ``blit(fig.bbox)`` at the end. No partial updates ever
        reach the screen.
        """
        canvas = self.fig.canvas
        try:
            # Restore + draw on the backbuffer ONLY — no blit yet.
            if self._ax_bg is not None:
                canvas.restore_region(self._ax_bg)
                for art in self._collect_main_animated_artists():
                    if art.get_visible():
                        self.ax.draw_artist(art)
            if self._mini_bg is not None:
                canvas.restore_region(self._mini_bg)
                for art in self._collect_mini_animated_artists():
                    if art.get_visible():
                        self.ax_mini.draw_artist(art)
            if self._goal_mini_bg is not None and not self.is_multi:
                canvas.restore_region(self._goal_mini_bg)
                for art in self._collect_goal_mini_animated_artists():
                    if art.get_visible():
                        self.ax_goal_mini.draw_artist(art)
            if self._status_bg is not None:
                canvas.restore_region(self._status_bg)
                for art in self._collect_status_animated_artists():
                    if art.get_visible():
                        self.ax_status.draw_artist(art)
            # Single atomic push of the whole figure to screen.
            canvas.blit(self.fig.bbox)
            # Clear the stale flag matplotlib raises when we call
            # ``set_data`` / ``set_offsets`` / ``.center =`` on the
            # animated artists. If we leave it True, the next Qt
            # paintEvent calls ``canvas.draw()`` itself, which repaints
            # the static layer over the blit'd output and produces the
            # "two things being rendered" ghost effect on the live
            # window. Setting it False here tells matplotlib the
            # canvas is up to date; the next change-of-data will set
            # it again.
            self.fig.stale = False
            for art_list in (self._collect_main_animated_artists(),
                              self._collect_mini_animated_artists(),
                              self._collect_goal_mini_animated_artists(),
                              self._collect_status_animated_artists()):
                for art in art_list:
                    try:
                        art.stale = False
                    except Exception:
                        pass
            canvas.flush_events()
        except Exception:
            canvas.draw_idle()

    # ── Event handlers ───────────────────────────────────────────
    def _tick(self):
        # Step every agent STEPS_PER_FRAME times before rendering.
        # Decoupling sim time from wallclock lets us keep agent
        # velocity low (no obstacle overshoot) while the GUI still
        # advances through the scenario at a watchable rate.
        any_running = False
        for _ in range(STEPS_PER_FRAME):
            tick_running = False
            for s in self.all_sims:
                if s.step():
                    tick_running = True
            if tick_running:
                any_running = True
            else:
                break
        # Always recompute the dynamic main-map artists (cheap
        # set_data / set_offsets / annotation moves). The mini panels
        # and status text are gated to every Nth tick to keep blit
        # frames cheap — those panels use shared-canvas axes that
        # need a full ``draw_idle`` to repaint, which is the ~50 ms
        # cost we're shedding from the per-frame budget.
        self._frame_idx = getattr(self, "_frame_idx", 0) + 1
        # Render every panel's animated artist data every frame —
        # this is cheap (set_data / set_offsets / center moves).
        self._panel_recenter_needed = False
        self._render_dynamic_main()
        self._render_dynamic_panels()
        if getattr(self, "_blit_supported", True):
            # When a follow-cam recenter happened this frame, the new
            # ``set_xlim/ylim`` invalidates the cached bgs (tick labels
            # moved, gridlines shifted). Do a synchronous full canvas
            # draw, then re-snapshot every panel's bg so the next
            # blit restores the correct static layer.
            if self._panel_recenter_needed:
                self.fig.canvas.draw()
                try:
                    cb = self.fig.canvas.copy_from_bbox
                    self._ax_bg = cb(self.ax.bbox)
                    self._mini_bg = cb(self.ax_mini.bbox)
                    self._goal_mini_bg = (cb(self.ax_goal_mini.bbox)
                                           if not self.is_multi else None)
                    self._status_bg = cb(self.ax_status.bbox)
                except Exception:
                    pass
            self._blit_main()
        else:
            # Fall-through for backends that can't blit.
            self.fig.canvas.draw_idle()
        if not any_running:
            self._timer.stop()

    def _on_key(self, event):
        if event.key == " " or event.key == "p":
            # Toggle pause
            if self._timer is None:
                return
            try:
                running = getattr(self._timer, "_running", None)
            except Exception:
                running = None
            # matplotlib timers don't expose state portably — toggle by
            # restart/stop via a simple flag.
            if getattr(self, "_paused", False):
                self._timer.start()
                self._paused = False
            else:
                self._timer.stop()
                self._paused = True
        elif event.key == "r":
            # Hot-reset: keep the current figure, rebuild scenario +
            # agents in place. Avoids leaking figures + timers (which
            # was making the sim slower geometrically per press).
            was_running = not getattr(self, "_paused", False)
            if was_running:
                self._timer.stop()
            # Truly random seed per reset — the previous formula
            # (seed + sim.steps + 1) was sequential and produced
            # visually similar scenarios on rapid presses.
            seed = secrets.randbelow(2**31)
            new_args = self.args
            scenario = build_scenario(new_args, seed=seed)
            (cm, start, goal, true_heading,
             new_obs, new_roofs, new_proj,
             new_jammers, new_foliage, new_spoofers) = scenario
            new_agents = build_agents(new_args, scenario,
                                       max(1, len(self.all_sims)))
            self._reload_scenario(
                new_agents[0], new_obs, new_roofs, new_proj,
                jammers=new_jammers, foliage=new_foliage,
                spoofers=new_spoofers,
                new_peers=new_agents[1:])
            if was_running:
                self._timer.start()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def run(self):
        self._paused = False
        # plt.show under Qt5Agg blocks until the window closes, so we
        # need to install the blit background *after* the figure has
        # been realised and laid out. ``draw_event`` fires right after
        # the first real paint — perfect hook for snapshotting the
        # static background and toggling artists to animated mode.
        # A defensive fallback also calls ``_setup_blit`` inline so
        # backends that never emit ``draw_event`` (rare) still get a
        # background captured. _tick checks ``_blit_supported`` and
        # falls back to draw_idle if the snapshot fails.
        def _on_first_draw(_evt):
            if self._ax_bg is None:
                self._setup_blit()
        self._draw_cid = self.fig.canvas.mpl_connect(
            "draw_event", _on_first_draw)
        # Resize / DPI changes invalidate the cached background.
        # Recapture on the next paint after a resize.
        def _on_resize(_evt):
            # Force a one-shot resnapshot by clearing the cache; the
            # next ``draw_event`` callback above will refill it.
            self._ax_bg = None
        self._resize_cid = self.fig.canvas.mpl_connect(
            "resize_event", _on_resize)
        self._timer.start()
        plt.show()


# ── Setup helpers ────────────────────────────────────────────────
def random_goal(rng, obstacles, exclude_start_radius=10.0,
                roofs=(), projectors=(),
                jammers=(), foliage=(), spoofers=()):
    """Pick a random goal in the map that satisfies RULES.md rule 5:
    outside circular obstacles, NOT inside any roof, NOT inside any
    projector influence, NOT inside any jammer or foliage zone, and
    NOT inside any spoofer influence. Also not too close to the
    start."""
    bound = MAP_HALF - 5.0
    for _ in range(2000):
        gx = rng.uniform(-bound, bound)
        gy = rng.uniform(-bound, bound)
        if math.hypot(gx, gy) < exclude_start_radius:
            continue
        bad = False
        for ox, oy, oR in obstacles:
            if math.hypot(gx - ox, gy - oy) < oR + GOAL_RADIUS + 1.0:
                bad = True
                break
        if not bad:
            for x_min, y_min, x_max, y_max in roofs:
                if (x_min <= gx <= x_max
                        and y_min <= gy <= y_max):
                    bad = True
                    break
        if not bad:
            for verts, _bias in projectors:
                cx, cy, _ = projector_centroid_radius(verts)
                if math.hypot(gx - cx, gy - cy) \
                        < PROJECTOR_INFLUENCE_RADIUS_M + GOAL_RADIUS:
                    bad = True
                    break
        if not bad:
            for cx, cy, r in jammers:
                if math.hypot(gx - cx, gy - cy) < r + GOAL_RADIUS:
                    bad = True
                    break
        if not bad:
            for cx, cy, r in foliage:
                if math.hypot(gx - cx, gy - cy) < r + GOAL_RADIUS:
                    bad = True
                    break
        if not bad:
            for (cx, cy), _fake in spoofers:
                if math.hypot(gx - cx, gy - cy) \
                        < SPOOFER_INFLUENCE_RADIUS_M + GOAL_RADIUS:
                    bad = True
                    break
        if not bad:
            return gx, gy
    # Fallback: edge of map
    return bound, bound


def random_start(rng, obstacles, goal_world, min_goal_dist=8.0,
                 clearance=1.5, projectors=()):
    """Pick a random spawn position satisfying RULES.md rule 5:
    outside circular obstacles, *not inside* a projector triangle's
    body (but next to one is fine), and inside a roof (square) is
    explicitly allowed. Used by --crazy to scatter agents across the
    full play area.

    `projectors` is the list of `(verts, bias)` tuples; we use the
    triangle's circumscribed disk as a conservative body proxy. The
    inflated zone around the disk is *not* excluded — agents are
    allowed to spawn next to triangles, just not on top of them."""
    bound = MAP_HALF - 2.0
    for _ in range(2000):
        sx = rng.uniform(-bound, bound)
        sy = rng.uniform(-bound, bound)
        if math.hypot(sx - goal_world[0], sy - goal_world[1]) < min_goal_dist:
            continue
        bad = False
        for ox, oy, oR in obstacles:
            if math.hypot(sx - ox, sy - oy) < oR + clearance:
                bad = True
                break
        if not bad:
            for verts, _bias in projectors:
                cx, cy, r = projector_centroid_radius(verts)
                if math.hypot(sx - cx, sy - cy) < r:
                    bad = True
                    break
        if not bad:
            return sx, sy
    # Fallback: somewhere on the map edge.
    return bound, -bound


def scripted_layout():
    """Deterministic scenario: start at origin, far-off goal (70 m
    north — well inside the 152 m map), with one circle obstacle, one
    roof, and one projector triangle laid out along the path so the
    agent's response to each is observable in turn."""
    start_world  = (0.0, 0.0)
    goal_world   = (0.0, 70.0)
    true_heading = math.radians(35.0)        # fixed but non-trivial

    # 1) Circle obstacle straddling the y-axis at y≈17 — robot must
    #    route around (it picks whichever side has more clearance).
    obstacles = [(0.0, 17.0, 4.0)]

    # 2) Roof centered on the path at y≈37 — 14 m wide × 10 m tall,
    #    forcing the robot to drive through it and lose GPS for the
    #    full vertical extent of the dropout.
    roofs = [(-7.0, 32.0, 7.0, 42.0)]

    # 3) Projector triangle a bit further along, slightly off-axis so
    #    the robot can pass it and still feel its multipath bias.
    apex   = (0.5, 54.0)
    side   = 4.5
    radius = side / math.sqrt(3)
    verts = (
        (apex[0],                       apex[1] + radius),
        (apex[0] - radius * math.sqrt(3) / 2,
         apex[1] - radius / 2),
        (apex[0] + radius * math.sqrt(3) / 2,
         apex[1] - radius / 2),
    )
    # Fixed bias: ~2.5 m east. With a 9 m influence radius the robot
    # feels the shift roughly between y = 45 and y = 63 — well clear
    # of the goal so the EKF has time to settle before arrival.
    bias = (2.5, 0.0)
    projectors = [(verts, bias)]

    return (start_world, goal_world, true_heading,
            obstacles, roofs, projectors)


def build_scenario(args, seed=None):
    """Build the shared scenario (map, start, goal, true heading)
    independent of which agent will run inside it. Used by
    `build_sim` (single agent, kept for compat) and the new multi-
    agent path."""
    rng = np.random.default_rng(args.seed if seed is None else seed)
    if not args.random:
        (start_world, goal_world, true_heading,
         obstacles, roofs, projectors) = scripted_layout()
        # Scripted layout doesn't include the additional GPS hazards.
        jammers = []; foliage = []; spoofers = []
        if args.heading_deg is not None:
            true_heading = math.radians(args.heading_deg)
    else:
        obstacles = gen_obstacles(rng, n=args.obstacles)
        projectors = gen_projectors(rng, n=args.projectors,
                                     obstacles=obstacles)
        roofs = gen_roofs(rng, n=args.roofs)
        jammers  = gen_jammers(rng, n=getattr(args, 'jammers', 0))
        foliage  = gen_foliage(rng, n=getattr(args, 'foliage', 0))
        spoofers = gen_spoofers(rng, n=getattr(args, 'spoofers', 0))
        start_world = (0.0, 0.0)
        if args.heading_deg is None:
            true_heading = rng.uniform(-math.pi, math.pi)
        else:
            true_heading = math.radians(args.heading_deg)
        if args.goal_lat is not None and args.goal_lon is not None:
            goal_world = latlon_to_meters(args.goal_lat, args.goal_lon)
        else:
            # RULES.md rule 5: goal must be outside circular obstacles
            # AND outside roofs AND outside projector influence radii.
            # Now also outside jammers, foliage, and spoofer influence.
            goal_world = random_goal(
                rng, list(obstacles),
                roofs=roofs, projectors=projectors,
                jammers=jammers, foliage=foliage, spoofers=spoofers)
    # Chained-mission goal override. Applies in BOTH the scripted
    # and random branches so a default ``--single`` smoke run with
    # ``--mission three-waypoint`` works without also passing
    # ``--random``. Overrides --goal-lat / --goal-lon (the
    # documented precedence). The remaining legs are threaded into
    # each ``GPSWaypointSim`` via ``build_agents`` so the agent
    # auto-advances on arrival.
    if getattr(args, "mission", None) == "three-waypoint":
        mission_lat, mission_lon = THREE_WAYPOINT_MISSION[0]
        goal_world = latlon_to_meters(mission_lat, mission_lon)
        # WP1 in our local-tangent plane is exactly (0, 0) because
        # LAT_CENTER / LON_CENTER are pinned to WP1. The scripted
        # start at (0, 0) would put the agent on top of leg 1's
        # goal — first arrival fires on tick 1 and leg progression
        # is uninstructive. Shift the start ~30 m north so leg 1
        # is a real ~30 m drive and the EKF / heading-fit cache
        # has motion to learn from before legs 2 and 3 kick in.
        start_world = (0.0, 30.0)
    cm = Costmap(obstacles, projectors=projectors)
    return (cm, start_world, goal_world, true_heading,
            obstacles, roofs, projectors,
            jammers, foliage, spoofers)


def build_agents(args, scenario, n_agents):
    """Spawn N independent agents on the same scenario. Each gets its
    own RNG (so GPS noise / outliers / dropouts are independent) and
    its own random initial heading (unless overridden by --heading-deg).
    With --crazy, each agent also gets a random start position scattered
    across the map instead of clustering at the origin.

    Field-parity twin (``--real``): when running in field-parity mode
    with a single primary agent, automatically pair it with a
    "perfect movement" twin that has ``odom_yaw_bias_rate = 0`` so
    the encoder-bias divergence is visible side-by-side in the GUI:
        agents[0] = field robot (with calibrated bias) — drifts.
        agents[1] = idealised twin (no bias) — converges cleanly.
    Both share the same scenario, RNG seed, and initial heading.
    """
    (cm, start, goal, true_heading, obstacles, roofs, proj,
     jammers, foliage, spoofers) = scenario
    base_seed = args.seed if args.seed is not None else 0
    spawn_perfect_twin = (
        getattr(args, "real", False) and n_agents == 1)
    # Chained-mission queue (remaining legs after the seed goal,
    # which ``build_scenario`` already projected into world meters).
    # When ``--mission three-waypoint`` is set we hand every agent
    # the same queue so a multi-agent ensemble all run the same
    # 3-leg fixture; otherwise we pass an empty list so behavior is
    # bit-identical to today.
    if getattr(args, "mission", None) == "three-waypoint":
        goal_queue = list(THREE_WAYPOINT_MISSION[1:])
    else:
        goal_queue = []
    sims = []
    for i in range(n_agents):
        rng_i = np.random.default_rng(base_seed * 7919 + i * 1000003 + 1)
        if args.heading_deg is None:
            # Each agent picks its own true heading. This is the most
            # informative ensemble — same map, different orientations.
            heading_i = rng_i.uniform(-math.pi, math.pi)
        else:
            heading_i = true_heading
        if (getattr(args, "crazy", False)
                or getattr(args, "scatter", False)):
            start_i = random_start(rng_i, obstacles, goal,
                                    projectors=proj)
        else:
            start_i = start
        sims.append(GPSWaypointSim(
            cm, start_i, heading_i, goal, rng_i,
            roofs=roofs, projectors=proj,
            jammers=jammers, foliage=foliage, spoofers=spoofers,
            goal_queue=goal_queue,
            coldstart_bias_enabled=getattr(
                args, "coldstart_bias_enable", False),
            next_hint_enabled=(
                True if getattr(args, "next_hint_enable", False)
                else None)))
    if spawn_perfect_twin:
        # Match the primary's start, heading, and noise-RNG seed so
        # the only difference is encoder bias.
        primary = sims[0]
        twin_rng = np.random.default_rng(base_seed * 7919 + 1)
        twin = GPSWaypointSim(
            cm, primary.start_world, primary.true_heading, goal, twin_rng,
            roofs=roofs, projectors=proj,
            jammers=jammers, foliage=foliage, spoofers=spoofers,
            odom_yaw_bias_rate=0.0,
            goal_queue=goal_queue,
            coldstart_bias_enabled=getattr(
                args, "coldstart_bias_enable", False),
            next_hint_enabled=(
                True if getattr(args, "next_hint_enable", False)
                else None))
        sims.append(twin)
    return sims


def build_sim(args, seed=None):
    """Single-agent compatibility shim. Returns (sim, obstacles,
    roofs, projectors) — the legacy 4-tuple."""
    scenario = build_scenario(args, seed=seed)
    (cm, start, goal, true_heading, obstacles, roofs, projectors,
     jammers, foliage, spoofers) = scenario
    rng = np.random.default_rng(args.seed if seed is None else seed)
    sim = GPSWaypointSim(cm, start, true_heading, goal, rng,
                          roofs=roofs, projectors=projectors,
                          jammers=jammers, foliage=foliage,
                          spoofers=spoofers,
                          coldstart_bias_enabled=getattr(
                              args, "coldstart_bias_enable", False),
                          next_hint_enabled=(
                              True if getattr(
                                  args, "next_hint_enable", False)
                              else None))
    return sim, obstacles, roofs, projectors


def _apply_crazy_overrides():
    """Crank scenery scale and GPS-corruption parameters for --crazy."""
    global ROOF_SIZE_RANGE_M, PROJECTOR_SIDE_RANGE_M
    global PROJECTOR_BIAS_RANGE_M, PROJECTOR_INFLUENCE_RADIUS_M
    global ROOF_BLACKOUT_LEAK_PROB, ROOF_BLACKOUT_SKEW_M
    global CYCLE_SLIP_HZ_PER_S, NOISE_BURST_HZ_PER_S
    ROOF_SIZE_RANGE_M            = (14.0, 28.0)
    PROJECTOR_SIDE_RANGE_M       = (6.0, 11.0)
    PROJECTOR_BIAS_RANGE_M       = (8.0, 18.0)
    PROJECTOR_INFLUENCE_RADIUS_M = 14.0
    ROOF_BLACKOUT_LEAK_PROB      = 0.04
    ROOF_BLACKOUT_SKEW_M         = 8.0
    # Per-agent stochastic events: cycle slips and noise bursts.
    CYCLE_SLIP_HZ_PER_S          = 0.005    # ~1 / 200 s
    NOISE_BURST_HZ_PER_S         = 0.01     # ~1 / 100 s


def _apply_real_overrides():
    """Reset GPS-hazard globals to realistic outdoor values for --real.
    Tightens the GPS noise model to match what the F9P log actually
    shows in clean sky (<10 cm stationary jitter, ~20 cm slow drift),
    and undoes any --crazy-mode bumps. Adversarial hazards (jammers,
    spoofers, cycle slips, noise bursts) are turned off by setting
    their counts to 0 in main()."""
    global ROOF_SIZE_RANGE_M, PROJECTOR_SIDE_RANGE_M
    global PROJECTOR_BIAS_RANGE_M, PROJECTOR_INFLUENCE_RADIUS_M
    global ROOF_BLACKOUT_LEAK_PROB, ROOF_BLACKOUT_SKEW_M
    global CYCLE_SLIP_HZ_PER_S, NOISE_BURST_HZ_PER_S
    global FOLIAGE_NOISE_MULT
    global GPS_NOISE_STD, GPS_BIAS_AMPL_M, GPS_OUTLIER_STD
    global ROBOT_STRICT_ARRIVAL, GPS_HEADING_EKF_ENABLE
    global ODOM_YAW_BIAS_ENABLE, LIDAR_IMU_FUSION_ENABLE
    # GPS noise model — tightened to match the recorded F9P log
    # (AutoNav-GUI-Standalone/example-playback-csv/t000_20260427_185211).
    # Stationary jitter <10 cm in the log → σ ≈ 0.05–0.10 m. Slow
    # drift ≈ 0.2 m amplitude over a minute. Outliers stay rare (one
    # ~5 m hop per 200 s).
    GPS_NOISE_STD                = 0.10
    GPS_BIAS_AMPL_M              = 0.20
    GPS_OUTLIER_STD              = 4.0
    # Scenery
    ROOF_SIZE_RANGE_M            = (6.0, 14.0)   # gazebos / pavilions
    PROJECTOR_SIDE_RANGE_M       = (3.0, 5.5)    # building-corner sized
    PROJECTOR_BIAS_RANGE_M       = (1.5, 3.0)    # realistic multipath
    PROJECTOR_INFLUENCE_RADIUS_M = 7.0           # close-range only
    ROOF_BLACKOUT_LEAK_PROB      = 0.0           # no skewed leaks
    ROOF_BLACKOUT_SKEW_M         = 0.0
    CYCLE_SLIP_HZ_PER_S          = 0.0           # not adversarial
    NOISE_BURST_HZ_PER_S         = 0.0
    FOLIAGE_NOISE_MULT           = 2.0           # ~2× σ under canopy
    # ── Field-parity flags:
    #   • ODOM_YAW_BIAS_ENABLE = True — the deployed wheel
    #     encoders have the calibrated left-encoder overcount
    #     (fix/odometry-issues NOT shipped). The bias is REAL at
    #     the wheel-encoder layer; what matters for convergence
    #     is whether ekf_local fuses it away (see below).
    #   • LIDAR_IMU_FUSION_ENABLE = True — model the deployed
    #     ekf_local stage that fuses /multiScan/imu yaw rate
    #     with the wheel-encoder yaw rate. With low IMU variance
    #     dominating the encoder bias variance, the fused yaw is
    #     clean and the gps_handler EKF receives IMU-corrected
    #     odom — exactly what /local_ekf/odom carries on the
    #     real robot once ekf_local.yaml's imu0 input is wired.
    #     Without this, the sim was pessimistically feeding raw
    #     biased wheel odom into gps_handler and reproducing an
    #     orbit pattern that doesn't represent the deployed
    #     stack.
    #   • GPS_HEADING_EKF_ENABLE = True — run the magnetometer-
    #     less θ EKF as the deployed gps_handler_node does.
    #   • ROBOT_STRICT_ARRIVAL = True — the action server's
    #     success threshold is 0.25 m, mirroring
    #     gps_handler_node SUCCESS_RADIUS_M.
    ODOM_YAW_BIAS_ENABLE         = True
    LIDAR_IMU_FUSION_ENABLE      = True
    GPS_HEADING_EKF_ENABLE       = True
    ROBOT_STRICT_ARRIVAL         = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--obstacles", type=int, default=12,
                        help="Number of random circular obstacles.")
    parser.add_argument("--roofs", type=int, default=3,
                        help="Number of GPS-blocking roof squares "
                             "(non-obstacles).")
    parser.add_argument("--projectors", type=int, default=4,
                        help="Number of multipath projector triangles "
                             "(obstacles, bias GPS).")
    parser.add_argument("--jammers", type=int, default=0,
                        help="Number of hex GPS-jammer zones "
                             "(sparse fixes inside).")
    parser.add_argument("--foliage", type=int, default=0,
                        help="Number of foliage zones (noise σ ×= "
                             f"{FOLIAGE_NOISE_MULT:.0f} inside).")
    parser.add_argument("--spoofers", type=int, default=0,
                        help="Number of GPS spoofers (pin readings "
                             "to a fixed lie).")
    parser.add_argument("--random", action="store_true",
                        help="Use a random scenario instead of the "
                             "default scripted test layout. Random "
                             "mode honours --obstacles / --roofs / "
                             "--projectors.")
    parser.add_argument("--crazy", action="store_true",
                        help="Random scenario with all GPS hazards "
                             "cranked: dense + larger obstacles, "
                             "extreme projector multipath bias, roof "
                             "blackout leaks, hex jammers, foliage "
                             "zones, GPS spoofers, cycle slips, and "
                             "noise bursts. Each agent gets its own "
                             "random start scattered across the map. "
                             "Implies --random; overrides counts.")
    parser.add_argument("--real", action="store_true",
                        help="Realistic outdoor GPS scenario: sparse "
                             "obstacles, a couple of small structures, "
                             "occasional foliage, no adversarial RF. "
                             "GPS noise / drift / outlier rates are "
                             "the calibrated u-blox-class defaults. "
                             "Implies --random; overrides counts.")
    parser.add_argument("--scatter", action="store_true",
                        help="Spawn each agent at a random valid "
                             "start position across the map instead "
                             "of clustering at the origin. Composes "
                             "with --random / --real / --crazy "
                             "(--crazy already implies --scatter).")
    parser.add_argument("--agents", type=int, default=25,
                        help="Number of agents to run in parallel "
                             "(default 25). All share the same map / "
                             "start / goal but have independent GPS "
                             "noise and random initial heading.")
    parser.add_argument("--single", action="store_true",
                        help="Shortcut for --agents 1 (full single-"
                             "agent visualization: belief cloud, EKF "
                             "marker, intended-endpoint X, A* path).")
    parser.add_argument("--heading-deg", type=float, default=None,
                        help="Force the robot's true heading in degrees "
                             "(default: random).")
    parser.add_argument("--goal-lat", type=float, default=None)
    parser.add_argument("--goal-lon", type=float, default=None)
    parser.add_argument("--mission", choices=["three-waypoint"],
                        default=None,
                        help="Run a chained multi-leg mission. "
                             "'three-waypoint' uses the canonical "
                             "GPS waypoints from the deployed "
                             "stored_waypoints.txt and exercises the "
                             "preemptive next-goal cache + per-leg "
                             "baseline reset path from "
                             "origin/improve/gps-waypoint-continuity. "
                             "Overrides --goal-lat / --goal-lon.")
    parser.add_argument(
        "--coldstart-bias-enable",
        action="store_true",
        help="Mirrors deployed ROS parameter "
             "``coldstart_bias_enabled`` (default False on the robot, "
             "see gps_handler_node.py L481). When set, fires the "
             "one-shot θ_offset snap from "
             "_seed_coldstart_theta_if_needed on the very first leg "
             "only (deployed L1179). The sim flag is currently a "
             "wired stub on the agent (see "
             "``GPSWaypointSim._coldstart_bias_enabled``) — flipping "
             "it on does NOT yet snap θ because the sim has no "
             "non-K_est seed path; left as a TODO so the launcher "
             "can toggle the field-parity behavior the moment the "
             "seed helper is implemented.")
    parser.add_argument(
        "--next-hint-enable",
        action="store_true",
        help="Mirrors deployed ROS parameter ``next_hint_enabled`` "
             "(default False on the robot, see gps_handler_node.py "
             "L466). When set, activates the shadow EWMA on the cached "
             "next-goal waypoint so the leg-transition warm start "
             "fires (``_update_next_hint_smoother`` + "
             "``_advance_to_next_leg`` promotion check). The "
             "single-goal headless smoke leaves this disabled so its "
             "numerical output stays unaffected; the "
             "``--mission three-waypoint`` smoke auto-enables the "
             "shadow EWMA when a queue is present so the mission "
             "exercise actually tests the preemptive-cache path.")
    parser.add_argument("--headless", action="store_true",
                        help="Run a few step()s without a window. "
                             "Used for smoke-testing.")
    parser.add_argument("--headless-steps", type=int, default=200)
    parser.add_argument("--full-steps", action="store_true",
                        help="Disable both early-exit conditions in "
                             "the headless loop (all-classified and "
                             "all-idle). Forces every run to advance "
                             "the full ``--headless-steps`` count for "
                             "apples-to-apples sim-time comparisons.")
    parser.add_argument("--no-ekf", action="store_true",
                        help="Disable the GPS-heading θ EKF and the "
                             "LIDAR/IMU yaw fusion. Models the field-"
                             "test configuration where the robot was "
                             "running raw biased wheel odom against a "
                             "world-frame goal, producing the orbit "
                             "signature and 40 m miss observed on May "
                             "9th. Encoder yaw bias stays on so the "
                             "drift is visible.")
    args = parser.parse_args(argv)

    if args.crazy:
        args.random = True
        args.obstacles = 30
        args.roofs     = 8
        args.projectors = 12
        args.jammers   = 5
        args.foliage   = 8
        args.spoofers  = 3
        _apply_crazy_overrides()
    elif args.real:
        # A realistic outdoor course: sparse rocks/bushes, one small
        # gazebo-style roof, one building-corner projector, a couple
        # of tree patches. No jammers / spoofers / cycle slips / noise
        # bursts — those model adversarial or pathological RF that the
        # robot will not encounter under normal operation.
        args.random = True
        args.obstacles = 6
        args.roofs     = 1
        args.projectors = 1
        args.jammers   = 0
        args.foliage   = 3
        args.spoofers  = 0
        _apply_real_overrides()
    if args.no_ekf:
        # Field-test configuration: run with raw biased wheel odom
        # and no IMU fusion. The closed-form θ EKF is bypassed
        # entirely (ekf.theta forced to 0 every update), so the
        # agent treats world-frame goals as if they were already
        # in odom — the deployed robot's pre-fix behavior.
        global GPS_HEADING_EKF_ENABLE, LIDAR_IMU_FUSION_ENABLE
        global ODOM_YAW_BIAS_ENABLE
        GPS_HEADING_EKF_ENABLE = False
        LIDAR_IMU_FUSION_ENABLE = False
        ODOM_YAW_BIAS_ENABLE = True
    if args.single:
        args.agents = 1

    scenario = build_scenario(args)
    (cm, start, goal, true_heading, obstacles, roofs, projectors,
     jammers, foliage, spoofers) = scenario
    agents = build_agents(args, scenario, max(1, args.agents))
    sim = agents[0]
    peers = agents[1:]

    if args.headless:
        import time as _time
        t0 = _time.perf_counter()
        for _ in range(args.headless_steps):
            any_running = False
            for s in agents:
                if s.step():
                    any_running = True
            # Three-bin termination: stop as soon as every agent
            # is in one of {arrived, predicted_success,
            # predicted_failure}. Falls through to all-idle as a
            # safety net in case the classifier hasn't tagged a
            # rare edge case. ``--full-steps`` suppresses both
            # early-exit conditions so apples-to-apples runs use
            # the same sim-time window.
            if not args.full_steps:
                # Chained-mission gate: even if an agent has been
                # classified for the *current* leg (arrived /
                # predicted_success / predicted_failure), don't
                # call the run "done" while it still has queued
                # waypoints. The leg-advance happens at the top of
                # the NEXT ``step()`` call, so we need to let that
                # tick run.
                n_classified = sum(
                    1 for s in agents
                    if (s.arrived or s.predicted_success
                                  or s.predicted_failure)
                    and not getattr(s, "goal_queue", [])
                )
                if n_classified == len(agents):
                    break
                if not any_running:
                    break
        wall = _time.perf_counter() - t0
        true_dist = math.hypot(sim.true_pos[0] - sim.goal_world[0],
                                sim.true_pos[1] - sim.goal_world[1])
        heading_err = math.degrees(
            (sim.true_heading - sim.heading_offset_est + math.pi)
            % (2 * math.pi) - math.pi)
        ekf = sim.ekf
        ekf_upd = ekf.update_count if ekf else 0
        ekf_rej = ekf.rejected_count if ekf else 0
        ekf_tstd = math.degrees(ekf.theta_std_rad) if ekf else 0.0
        if sim.intended_endpoint_history:
            arr = np.array(sim.intended_endpoint_history)
            mean_dist = math.hypot(arr[:, 0].mean() - sim.goal_world[0],
                                   arr[:, 1].mean() - sim.goal_world[1])
        else:
            mean_dist = float("nan")
        n_arrived = sum(1 for s in agents if s.arrived)
        n_pred_ok = sum(1 for s in agents
                         if s.predicted_success and not s.arrived)
        n_pred_fail = sum(1 for s in agents if s.predicted_failure)
        n_unclassified = (len(agents) - n_arrived
                          - n_pred_ok - n_pred_fail)
        n_stuck = sum(1 for s in agents
                       if s.path_world is None or len(s.path_world) < 2)
        will_converge = n_arrived + n_pred_ok
        # Chained-mission leg progress. Only appended when the
        # primary agent is on a multi-leg mission so the
        # single-goal smoke output stays byte-identical to today
        # (the documented regression gate).
        mission_suffix = ""
        if getattr(sim, "leg_count", 1) > 1:
            mission_suffix = (
                f" mission_leg={sim.leg_index}/{sim.leg_count}"
                f" mission_remaining={len(sim.goal_queue)}"
            )
        print(f"t={sim.sim_time:.2f}s steps={sim.steps} agents={len(agents)} "
              f"arrived={n_arrived}/{len(agents)} "
              f"pred_ok={n_pred_ok} pred_fail={n_pred_fail} "
              f"unclassified={n_unclassified} "
              f"will_converge={will_converge}/{len(agents)} "
              f"no_path={n_stuck} "
              f"wall={wall:.2f}s "
              f"per_step={1000*wall/max(1,sim.steps):.2f}ms "
              f"primary_dist={true_dist:.2f}m "
              f"heading_err={heading_err:+.2f}° "
              f"ekf_σθ={ekf_tstd:.2f}° "
              f"ekf_upd/rej={ekf_upd}/{ekf_rej} "
              f"cloud_mean_dist={mean_dist:.2f}m "
              f"pad={sim.last_pad}"
              + mission_suffix)
        return 0

    gui = GPSWaypointGUI(sim, obstacles, args,
                          roofs=roofs, projectors=projectors,
                          jammers=jammers, foliage=foliage,
                          spoofers=spoofers,
                          peers=peers)
    gui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
