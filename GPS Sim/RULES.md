# RULES

Rules every simulated agent (and the simulator that hosts them) must
follow. These are the contract — anything in `src/gps_sim_gui.py` that
contradicts these rules is a bug.

## Rule 1 — what each agent gets to see

An agent only has access to:
- Its own local `(x, y)` position in **its own odometry frame**, where
  the spawn point is `(0, 0)`.
- GPS coordinates (longitude, latitude) coming in from its own sensors.

No agent gets ground-truth world position, no agent gets a magnetometer,
no agent reads other agents' state.

## Rule 2 — what each agent has to do

Each agent must build a path to the GPS goal *as it believes it to be*,
expressed in **its own** `(x, y)` odometry frame. The mapping from the
goal's world GPS coordinates into the agent's odom frame is the agent's
problem to solve (heading offset estimation).

## Rule 3 — odometry is the fallback

When GPS is faulty (dropout, blackout, multipath spike, projector skew),
the agent must use its local `(x, y)` odometry to keep navigating. The
EKF coasts on prediction during outages.

## Rule 4 — robustness target

The whole pipeline must survive extreme GPS misfires (multipath bias,
projector influence, roof blackout leaks, outliers, dropouts) and still
arrive within **1 m of the real goal** (the competition success
criterion: ≥ 50% of the robot footprint inside the 1 m goal circle).

## Rule 5 — placement constraints (simulator side)

Applies to scenario generation, not to agents:

- **Circular obstacles**: agents and goals must spawn **outside** them.
- **Triangle projectors**: agents may spawn **next to** them (close is
  fine), but not inside the triangle body. **Goals must not** be inside
  any triangle's GPS-influence radius.
- **Square roofs**: agents may spawn **inside** a roof. **Goals must
  not** be inside any roof.
- **Hexagon GPS jammers**: agents may drive through them. **Goals must
  not** be inside any jammer.
- **Foliage / canopy zones**: agents may drive through them. **Goals
  must not** be inside foliage.
- **GPS spoofers**: agents may pass through their influence radius.
  **Goals must not** be inside any spoofer's influence radius.

## Rule 6 — A* fallback when the goal lands in an obstacle

If a candidate goal cell is inside an obstacle (e.g. the heading
estimate has rotated the goal into a lethal cell), A* must plan to the
**nearest non-obstacle cell** rather than returning no path. The agent
must keep making progress; bailing out would let GPS noise silently
freeze it.

## Rule 7 - Algorithm application scope

The algorithm we are creating is simply to transform GPS coordinates into 
local Euclidian space by using a clever ENU projection and rotation convergence.
Anything related to the sim not working is not representative of the real robot.
We are simply placing canidate GPS goals in local Euclidain space.