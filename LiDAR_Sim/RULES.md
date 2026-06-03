# Simulation Rules

These rules define the requirements for the LiDAR terrain classification simulation. All changes must satisfy every rule before being considered complete.

## 1. PCA Reference Frame

Always use **sensor local coordinates** to calculate PCA grades. Compare PCA surface normals to the **LiDAR's own normal** (sensor up = `[0,0,1]` in sensor frame). Never use world frame, gravity vector, or IMU-derived references — these are unreliable during motion.

## 2. Pass Criteria

Agents must pass **100% of the time**. A pass means:

- All agents reach their goals
- Agents traverse ramps fully based on the grade threshold:
  - **15% threshold (8.5 deg):** pass ramp 1 (5 deg), block ramps 2-5
  - **20% threshold (11.3 deg):** pass ramps 1-2 (5, 10 deg), block ramps 3-5
  - **30% threshold (16.7 deg):** pass ramps 1-3 (5, 10, 15 deg), block ramps 4-5
- Agents must **not get stuck** in local minimums

## 3. No Map Boundary Walls

There should be **no costmap walls** at the map edges keeping agents constrained. Instead, each agent should detect the map edge on its own (e.g., no LiDAR returns = edge of navigable area). The entire map must be available for navigation.

## 4. Ramp Costmap Behavior

- **Passable ramps** (below threshold) must **not be marked as high cost** by any agent
- While going **down a ramp**, the ground immediately after the ramp must **not be marked as impassable**
- The ramp-to-flat transition must remain traversable in both directions

## 5. Spawn Safety

- Agents must **not spawn inside obstacles**
- Goals must **not be placed inside obstacles**
- Spawn positions must be validated against the costmap before the simulation begins

## 6. Equal Ramp Usage

Agents should **choose to go up ramps equally** — goals and paths should be set up so that traversable ramps are used at comparable rates, not avoided in favor of detours around them.

## 7. LiDAR-Only Perception

All calculations must be done as if the LiDAR points are coming into the system as a **3D point cloud**. No other methods should be used to gain additional information about the world. Agents must not use mesh queries, ray-casting against the terrain model, or any ground-truth data — only the LiDAR point cloud that the sensor provides.

## 8. Processing Time Budget

Each flow of calculations (from point cloud input to costmap output) must complete within **60ms**. This ensures the system runs at 15+ Hz on the real robot.
