# GPS Sim — Convergence Stall Scenarios

A working log of every "what if X breaks" experiment run against the sim,
with the observed outcome and what it implies for the mission greenlight
gate on the real robot. All experiments use the sim at parity with
`origin/improve/gps-waypoint-continuity` (cold-start θ snap included).

The sim lives at `GPS Sim/simulated_world/`. Reproduce any row with the
`Sim command` shown — they're all headless and finish in 1–5 s wall.

> **Design-intent correction (post-S9 revision).** The earlier
> framing of this algorithm as "EKF-death surviving" was wrong. The
> 4-heading / R-rotation scheme only fixes the *rotation* half of the
> virtual↔real transform. The *translation* half — virtual position
> drifting from real position via accumulated encoder yaw bias —
> cannot be fixed in software once it appears. The only durable fix
> is to fuse GPS into the deployed robot's EKF (global EKF /
> `navsat_transform`), so virtual ≈ real and candidate placement
> actually commands the real robot to the GPS goal. The sim now
> reproduces that failure mode honestly: with `--no-gps-ekf`, the
> controller anchors on biased odom and the real robot misses the
> goal by the translation drift — see **S9**.

---

## How to read severity

| Tier | Meaning | What it implies for the precheck |
|---|---|---|
| **Catastrophic** | Agent never arrives, heading diverges or stays wild | **Must gate** — refuse to start the mission |
| **Severe** | Agent eventually arrives but with persistent metric drift | Should monitor + warn |
| **Mild** | Agent arrives within normal tolerance, observable but recoverable | Log only |
| **Negligible** | Sim doesn't surface the effect, or it's swallowed by tolerances | No action; revisit when sim models the gap |

All severity rankings are from `--real --single --coldstart-bias-enable
--seed 11 --full-steps --headless-steps 1500`. The single-seed snapshot
isn't a substitute for a sweep, but it's reproducible and pins the
ordering.

---

## Critical processes (current findings)

These three subsystems being down causes catastrophic convergence
failure with zero recovery within 150 s sim-time:

| Subsystem | Real-robot node(s) | Sim toggle |
|---|---|---|
| **GPS receiver** | `gps_handler::gps_publisher` (serial→NavSatFix) | `--no-gps` |
| **GPS handler θ-EKF** | `gps_waypoint_handler::gps_handler_node` | `--no-gps-ekf` |
| **NAV2 planning** | `nav2_planner_server` + BT navigator | `--no-nav2` |

The `mission_precheck.py` on the branch already checks message arrival
for `/gps_fix`, `/global_ekf/odom`, and `/gps_waypoint/debug` — that
covers the first two cases at the "is the node alive" level, but **does
not check convergence quality** (σ_θ, EKF update count, GPS rate
stability) or that the `coldstart_bias_enabled` parameter is actually
true at runtime.

---

## Scenarios

### S1 — Predicted-failure classifier fires too early

**Hypothesis:** "Robot is stalling" might just be the early-failure
classifier giving up before the EKF can converge.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 800 \
    --single --real --coldstart-bias-enable --seed 7
# vs the same with --full-steps and 1500 steps:
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable --seed 7
```

**Outcome:**
- 800-step run: `pred_fail=2/2`, `primary_dist=10.89 m` — classifier
  flags failure at 10 m
- 1500-step full run: `arrived=2/2`, `primary_dist=0.12 m` — agent
  actually converged

**Severity:** **Severe** (classifier — not algorithm — issue)

**Mechanism:** `PREDICT_FAIL_RADIUS_M` / `PREDICT_FAIL_HOLD_TICKS` flips
the agent to `predicted_failure` while it's still en route. If this
classifier signal feeds the real-robot mission monitor, missions will
be aborted while the robot is still making progress.

**Recommendation:** Either raise the radius / hold-ticks, or expose the
classifier as advisory-only on the wire. The agent is converging; the
classifier is wrong.

---

### S2 — Goal behind the robot (180° heading)

**Hypothesis:** "When the goal is behind, the candidate-goal smoother
can't place the intermediate goal correctly because it's being updated
too often."

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 7 --heading-deg 180
```

**Outcome:** `arrived=0/2 pred_fail=2/2`, primary_dist=22.83 m,
heading_err=-5.49°, σθ=3.13°.

**Severity:** **Severe** but not catastrophic — the EKF converges θ
fine; the bot is making progress, just slower because it has to turn
around first. The candidate-goal smoother isn't the problem (it never
diverges); the time budget is.

**Mechanism:** Cold-start θ snap correctly fires and orients the
candidate forward. The pure-pursuit controller takes longer to drive
the U-turn than the 150 s sim cap allows. With a longer time budget
the agent arrives.

**Recommendation:** Not an algorithmic bug. If the field test bounds
the mission to 150 s, a 180° initial heading + far goal could exceed
the budget. Field robot should not be facing the wrong way when the
mission starts — covered by the **AUTO mode + autonomous-mode-engaged
gate** in `mission_precheck.py` (operator manually orients robot during
prep).

---

### S3 — Cold-start trap: GPS bias overrides the snap

**Hypothesis:** "Cold-start bias doesn't escape" — the seed is set with
high variance, but a persistent per-agent fixed bias might lock the EKF
onto the biased reading before the bootstrap can refit.

**Reproduce:**
```bash
# Coldstart OFF, 5 m installation bias:
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --seed 7 --gps-bias 5.0
# Coldstart ON, same bias:
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --seed 7 --gps-bias 5.0 \
    --coldstart-bias-enable
```

**Outcome at 800 steps:**
- Bias 5 m, coldstart OFF: primary_dist=16.02 m, σθ=7.07°,
  cloud_mean=1.11 m
- Bias 5 m, coldstart ON: primary_dist=**11.45 m**, σθ=4.02°,
  cloud_mean=**0.11 m** — snap dominates the bias

**Severity:** **Mild with coldstart ON** (bias visible but agent
converges), **Severe with coldstart OFF**.

**Mechanism:** The 45° seed variance is loose enough that the first
real GPS update overwrites the snap; subsequent updates fold the bias
into the EKF's residual error which stays bounded (~bias magnitude).
Bootstrap_theta's closed-form fit ignores the bias because it operates
on GPS *displacement*, not absolute position.

**Recommendation:** The coldstart snap does its job. If the robot's
GPS antenna has a known installation offset, this is **the** behavior
that protects against it. Verify `coldstart_bias_enabled:=true` is
actually loaded at runtime.

---

### S4 — Dense projector field (multipath traps)

**Hypothesis:** Robot starts inside a high-cost region (projector zone)
and NAV2 refuses to plan out.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --projectors 12 --random
# vs dense 20 projectors:
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 7 --projectors 20 --random
```

**Outcome:** Seed 11 with 12 projectors → `arrived=2/2,
primary_dist=0.18 m`. Seed 7 with 20 projectors → still converges
(`pred_fail` is the classifier giving up, not actual stall).

**Severity:** **Mild** — A* successfully routes around projector zones
in every scenario tested. `no_path=0` in every projector test.

**Mechanism:** The 1/r envelope filter (`CANDIDATE_ENV_ENABLE`) and
moving-away detector both work as designed — they suspend the envelope
when the agent appears stuck and force a heading resync.

**Recommendation:** No action. The "robot stuck in high-cost region"
hypothesis isn't supported by sim evidence. If this happens on the
real robot, the issue is more likely **costmap pollution from
PCADETECT/LINEDETECT obstacles** (which the sim doesn't model) than
projector-multipath.

---

### S5 — Spoofer at start position

**Hypothesis:** First GPS fix is from a spoofer, EKF locks onto the
fake target.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --spoofers 3 --random
```

**Outcome:** Agent still converges. The cold-start snap orients
toward the true goal; the spoofer's "lock to fake target" overrides
the noisy fix path but the EKF's wide variance + envelope filter
catches the divergence within a few seconds.

**Severity:** **Mild** with current parameters.

**Recommendation:** The combination of cold-start snap + 1/r envelope
+ moving-away detector handles spoofers. Worth a periodic regression
bake under `--crazy` (which packs 3 spoofers + other hazards) to make
sure tightening any of those thresholds doesn't open a hole.

---

### S6 — Severe noise (--crazy mode)

**Hypothesis:** Compound failures — many spoofers + projectors +
jammers + foliage at once.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --crazy --coldstart-bias-enable --seed 7
```

**Outcome (800-step):** primary_dist=68.39 m, heading_err=+11.67°,
cloud_mean=1.39 m, **one EKF rejection** — `ekf_upd/rej=266/1`.

**Severity:** **Catastrophic** — agent never recovers from compounded
hazards.

**Mechanism:** Multiple spoofer / projector zones along the path
overlap, the moving-away detector saturates (resync cooldown), and
the agent oscillates between transiently-clean fixes.

**Recommendation:** Don't dispatch missions in `--crazy`-grade
environments. Field implication: **GPS health monitor** should reject
mission start when GPS-rate σ or per-fix Mahalanobis is above a
threshold over a rolling window — the precheck doesn't currently
have this.

---

### S7 — PRESLAM down (LIDAR-IMU fusion failed to come up)

**Hypothesis:** Local EKF didn't start; gps_handler receives raw
encoder-biased odometry.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --no-preslam
```

**Outcome:** `arrived=2/2, primary_dist=0.12 m, heading_err=-15.83°,
cloud_mean=0.24 m` — agent converges but with **3× larger residual
heading error** than the full-stack baseline.

**Severity:** **Severe** on the sim's gentle encoder bias; could be
**catastrophic** on the field robot where the encoder bias is larger
and compounds with motor commutation.

**Mechanism:** Without IMU yaw fusion, gps_handler receives the raw
biased wheel-encoder yaw. The closed-form heading fit can compensate
for a constant offset but not a per-meter drift — encoder bias is
the latter.

**Recommendation:** **Must gate.** Precheck should verify
`/imu_inflated` is publishing AND that `ekf_local`'s yaw-rate output
matches the SICK IMU rate within tolerance. Currently only message
arrival is checked.

---

### S8 — Lever-arm correction off (URDF antenna_link missing)

**Hypothesis:** `robot_state_publisher` didn't load the URDF, antenna
offset isn't subtracted in `_gps_callback`.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --no-lever-arm
```

**Outcome:** `arrived=2/2, primary_dist=0.18 m, heading_err=+0.16°,
cloud_mean=0.20 m` — essentially identical to baseline.

**Severity:** **Negligible** in the sim, but **likely Severe on field**.

**Mechanism:** The sim's antenna offset is small (matches Bowser URDF
≈ 0.7 m magnitude); after EKF convergence the bias term is mostly
folded into the heading-offset estimate. On the field robot with
external factors (longer lever arm if antenna moved, motor swing
during turns), the residual bias is more visible.

**Recommendation:** **Should gate.** Precheck should add a lookup for
`antenna_link → base_link` TF (in addition to the existing
`map → base_link` check). Currently this isn't verified.

---

### S9 — GPS handler EKF disabled (`--no-gps-ekf`)

**Hypothesis:** `gps_waypoint_handler::gps_handler_node` died after
startup; no θ-fusion ever runs. Without GPS feeding the localization
the robot can ONLY trust biased wheel odometry, and the virtual robot
drifts from the real robot by accumulated encoder yaw-bias × distance.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --no-gps-ekf
```

**Outcome:** `arrived=0/15, arr_gps=10.14 m, goal_ratio=7.77,
goal_fire=6, ekf_upd/rej=0/0` — **EKF never updates and the new
goal-distance magnitude watchdog fires** within seconds, surfacing the
translation between virtual and real before any miss is observable
from outside.

**Severity:** **Catastrophic — and unrecoverable in software.**

**Why no algorithm can fix this:** The candidate-placement formula
(4-heading invariant) assumes the transform between virtual and real
is a PURE ROTATION R. Without GPS in the EKF, the virtual robot
translates away from the real robot, so the assumption breaks. The
controller drives the *virtual* robot to the candidate; the real
robot follows physics. After the virtual robot "arrives," the real
robot is off by the translation error. No amount of θ correction
recovers that. The sim used to claim "EKF-death survival" via a
closed-form θ fallback — that survival was an artifact of using
`latest_gps()` as the planning anchor (i.e. driving from the real
position). The deployed robot has no such anchor when GPS isn't in
the EKF, so the sim now uses `self.odom` and reproduces the real
failure.

**The proper fix is upstream on the deployed robot:** fuse GPS as a
measurement update inside the local EKF (global EKF /
`navsat_transform` style). With GPS in the EKF, the EKF's pose stays
anchored to ground truth and virtual ≈ real — at which point
candidate-on-GPS-goal genuinely commands the real robot to the goal.

**Watchdog signal:** `goal_ratio` (the new `INV_GOAL_DIST_*`
invariant) is the cleanest single number. Healthy ≈ 1.0; this
scenario reaches 7+ within the first 30 s. Pair with the existing
`ekf_upd > N` precheck so missions get refused before launch.

**Recommendation:** **Must gate.** Add `ekf_upd > N` over a rolling
window AND `goal_ratio ∈ [0.75, 1.35]` to the precheck. The latter
is what catches the case where the EKF is technically alive but
isn't being fed GPS (the situation a misconfigured `navsat_transform`
or stalled fusion topic would create).

---

### S10 — NAV2 planner down (`--no-nav2`)

**Hypothesis:** Planner_server never advertised; agent has no path.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --no-nav2
```

**Outcome:** `arrived=0/2, primary_dist=22.41 m, heading_err=-31.49°,
no_path=2`.

**Severity:** **Catastrophic**.

**Recommendation:** **Must gate.** Already partially handled by the
`/navigate_to_waypoint` action-server-ready check in mission_precheck.
Adding a "dry-run plan from current → near-goal" test would catch the
case where the action server is ready but the planner refuses the
specific goal (e.g. lethal start cell).

---

### S11 — GPS receiver completely silent (`--no-gps`)

**Hypothesis:** `/gps_fix` topic never publishes.

**Reproduce:**
```bash
./.venv/bin/python gps_sim_gui.py --headless --headless-steps 1500 \
    --full-steps --single --real --coldstart-bias-enable \
    --seed 11 --no-gps
```

**Outcome:** `arrived=0/2, primary_dist=11.97 m, heading_err=-32.76°` —
agent drifts on dead-reckoning, never converges.

**Severity:** **Catastrophic**.

**Recommendation:** **Must gate** — already checked in mission_precheck
via `/gps_fix` arrival + status. Add: minimum fix rate over 10 s
(should be ≥ 5 Hz on F9P RTK).

---

## Recommended additions to `mission_precheck.py`

In addition to the existing message-arrival checks, the sim data
suggests adding these convergence-readiness gates:

| Gate | Reason | What to check |
|---|---|---|
| **GPS rate stable** | S11 | `/gps_fix` rate ≥ 5 Hz over a 10 s window before greenlight |
| **EKF active** | S9 | `/gps_waypoint/debug` carries a non-zero `ekf_update_count` |
| **EKF σ_θ converged** | S9 | `σ_θ < 15°` for ≥ 3 seconds of stationarity before motion |
| **antenna_link TF present** | S8 | `lookupTransform("antenna_link", "base_link", t=0)` succeeds |
| **`coldstart_bias_enabled` is True** | Field parity | `ros2 param get /gps_handler_node coldstart_bias_enabled` returns True |
| **IMU yaw rate ≈ wheel yaw rate** | S7 | `|ω_imu − ω_wheel| < 0.05 rad/s` averaged over 3 s while stationary |
| **A* test plan succeeds** | S10 | One synthetic plan from current odom → first goal returns a path |

---

## What the sim still doesn't model (gaps to close)

These would let the sim diagnose more field failures:

1. **PCADETECT-driven costmap pollution.** The real Nav2 costmap is fed
   by `/scan_pca_filtered_*`. When PCADETECT misclassifies, false
   obstacles appear and A* either refuses or routes wildly. Sim uses
   static obstacles.

2. **LINEDETECT line-layer interactions.** Lines mark the costmap as
   inflated. If LINEDETECT publishes spurious lines (low-confidence
   detections), the agent may treat them as walls.

3. **TF latency.** Real TF tree has ~10 ms delays between
   `map → odom`, `odom → base_link`, and `base_link → antenna_link`.
   The sim uses instantaneous transforms.

4. **Stuck-in-LETHAL recovery.** The real robot's start cell can be
   inflated to LETHAL if line_layer happens to mark it; nav2 refuses
   to plan until the agent backs out. The sim's costmap doesn't have
   this dynamic inflation.

5. **Multi-process startup race.** The real robot launches everything
   sequentially via `TimerActions`. The sim builds the agent in one
   shot. If `slam_toolbox` hasn't published `map→odom` by the time
   gps_handler asks for the TF, the first projection silently uses
   identity and the seed snaps to a wrong θ.

---

## Reproducing the parity baseline

Every "control" run in this doc uses:

```bash
cd "GPS Sim/simulated_world"
./.venv/bin/python gps_sim_gui.py \
    --headless --headless-steps 1500 --full-steps \
    --single --real --coldstart-bias-enable \
    --seed 11
```

Result: `arrived=2/2, primary_dist≤0.2 m, heading_err≤5°, σ_θ≤5°`.

If a future change to the sim or to the deployed code breaks this,
parity is gone — investigate before merging.

---

*Last updated when `improve/gps-waypoint-continuity` was at
`adf6c6d9` ("Mission precheck and State change bug MAN to AUTO"). The
sim was at parity with `3e0370ac` ("Bigfix to the coldstart bias") as
of that date.*

---

## Appendix A — 4-heading invariant watchdog

Per `GPS Sim/DESIGN_INTENT.md`, the algorithm has two independent θ
estimators:

```
θ_motion = atan2(GPS Δ)        − atan2(odom Δ)             # (A − C)
θ_goal   = atan2(goal − GPS)   − atan2(candidate − odom)   # (B − D)
```

If both hold simultaneously, the candidate equals the true goal. The
watchdog (`_update_invariant_watchdog`, `_compute_four_heading_error`)
computes `wrap_pi(θ_motion − θ_goal)` every tick and fires a forced
heading resync + 1/r envelope suspension when |err| stays above
`INV_ERROR_THRESHOLD_DEG = 12°` for `INV_ERROR_SUSTAINED_TICKS = 30`
ticks (3 s).

**Close-to-goal gate.** `INV_ERROR_MIN_GOAL_DIST_M = 3.0` disables the
watchdog when the robot is within 3 m of the goal. Rationale:
`tan(θ_err) = bias / distance` → as `distance` shrinks toward the GPS
bias magnitude, `B`'s `atan2` becomes noise-dominated, and the
watchdog itself becomes unreliable. In this regime the deliberate
handoff per DESIGN_INTENT is to **trust the locally-converged
candidate** — θ has already converged at distance, the candidate's
odom-frame position was set with that converged θ, and by arrival
time the local-frame candidate equals the absolute GPS goal.

### Per-scenario watchdog observation (seed 11, full-steps, 1500 ticks)

| Scenario | `inv_err` final | `inv_max` | `inv_fire` | Watchdog says |
|---|---|---|---|---|
| Full stack | -8.6° | 22.7° | 0 | ✓ converged |
| `--no-preslam` | +14.0° | 22.6° | 0 | borderline; recovers |
| `--no-lever-arm` | -1.2° | 28.3° | 0 | ✓ converged |
| `--no-gps` | `--` | 0° | 0 | can't fit (no GPS) |
| **`--no-gps-ekf`** | **-111.5°** | **180°** | **7** | **🔥 fires — clearest "θ never converges" signal** |
| `--no-nav2` | `--` | 0° | 0 | can't fit (no motion) |
| `--gps-bias 1 m` | -7.2° | 26.1° | 0 | ✓ absorbed |
| `--gps-bias 3 m` | -10.9° | 24.5° | 0 | ✓ absorbed |
| `--gps-bias 5 m` | -8.3° | 22.9° | 0 | ✓ absorbed |
| `--crazy` | +61.8° | 84.2° | 0 | borderline, no sustained 3 s |
| `heading 180°` | +3.8° | 35.2° | 0 | ✓ U-turn handled |
| `heading 90°` | -1.3° | 129.2° | 0 | ✓ transient only |

10/10 healthy seeds (7, 11, 13, 21, 33, 42, 77, 100, 137, 250) all
arrive with `inv_fire = 0`. The watchdog stays quiet on healthy runs.

### What this means for the field robot

`inv_fire > 0` is the cleanest sim-observable signal that θ is
locking onto a contaminated heading. Port the watchdog into
`gps_handler_node.py` and publish `inv_err` on
`/gps_waypoint/debug`; the mission monitor can then treat
`inv_fire > 0` as a hard "do not greenlight the next leg" signal.

---

## Appendix B — Arrival validators (local + GPS + Mahalanobis χ²)

Per design intent: "use our local coordinates to validate that we are
in the GPS goal while the GPS as long as the goal is inside the
covariance ellipse." Implemented as `_arrival_validators()` returning
three independent metrics:

```
arr_local  = |odom − smoothed_candidate|         # local-frame proximity
arr_gps    = |GPS_now − goal_world|              # absolute-GPS proximity
arr_χ²     = (goal − ekf_pos)^T  P^-1  (goal − ekf_pos)
                                                  # χ² 2-DOF; < 5.99
                                                  # = goal inside 95%
                                                  # confidence ellipse
```

Arrival declaration now gates on **all three**: `arr_local < 1 m AND
arr_gps < 1 m AND arr_χ² < 5.99`.

### What the three signals reveal in stall cases

| Scenario | arr_local | arr_gps | arr_χ² | Diagnosis |
|---|---|---|---|---|
| Full stack | 0.21 m | 0.24 m | 0.05 | ✓ all three tight; arrived |
| `--gps-bias 1 m` | 0.22 m | 0.26 m | 0.05 | ✓ bias absorbed |
| `--gps-bias 5 m` | 0.21 m | 0.20 m | 0.04 | ✓ algorithm tracks despite 5 m bias |
| `--no-preslam` | 0.08 m | 0.04 m | 0.00 | ✓ arrival OK; heading drift was transient |
| **`--no-gps-ekf`** | **0.24 m** | **44.10 m** | **0.01** | **Local says "here", GPS says "no!" — declaring arrival would have been catastrophic** |
| `--no-gps` | 0.75 m | `inf` | 0.17 | No GPS to validate against |

The `--no-gps-ekf` row is the canonical case the dual gate catches.
Without `arr_gps`, the deployed arrival logic would have declared
success at 44 m off. The χ² is artificially low because the EKF
never updates — its covariance stays at the initial small value.
That's exactly why we need **all three** independent signals: each
catches a failure mode the others miss.

### Recommended addition to `/gps_waypoint/debug` payload

Publish `arr_local`, `arr_gps`, and `arr_mahalan` so the mission
monitor can see the algorithm's own arrival-readiness signals. Gate
mission-success acknowledgement on all three.

---

*Watchdog + arrival-validator implementation added on 2026-05-22.
Sim now lives "one rev ahead" of the deployed `gps_handler_node.py`
on `improve/gps-waypoint-continuity` — these are the next features
to port when sim experience says they're stable.*
