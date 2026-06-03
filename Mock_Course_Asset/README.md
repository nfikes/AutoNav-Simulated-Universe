# Mock Course Asset

Source-of-truth Blender scene for the AutoNav mock course — the venue
geometry, ramps, GPS waypoints, obstacles, and surface tags used by the
**Fortress** (Gazebo Fortress / Ignition) simulation that boots the real
AutoNav ROS 2 stack against a virtual world.

```
Mock_Course_Asset/
├── Course.blend     ← The course scene. Source of truth.
└── README.md        ← This file.
```

`Course.blend` is tracked through Git LFS (see `.gitattributes`). Run
`git lfs install && git lfs pull` after cloning or it will be a pointer
file and Blender will refuse to open it. The Blender autosave sibling
(`Course.blend1`) is gitignored.

---

## Why this lives in the sim repo

The three Python simulators in this repo (`LiDAR_Sim`, `GPS_Sim`,
`GUI_Sim`, plus `BEHAVIOR_TREE_Sim` and `SPEED_Sim`) validate one
subsystem at a time. `Course.blend` is the bridge to a **full-stack**
simulation: drop the AutoNav robot into this scene under Gazebo
Fortress and exercise perception, navigation, GPS fusion, and the HUD
together — the same way they'll run on the Jetson — without touching
hardware.

---

## Using the course in Fortress

Fortress = [Gazebo Fortress](https://gazebosim.org/docs/fortress)
(formerly Ignition), the simulator the AutoNav stack targets when it
isn't on real hardware.

The pipeline is:

1. **Export the scene from Blender.** Save the active scene as Collada
   (`.dae`) or glTF — both are mesh formats Gazebo's `<mesh>` element
   consumes directly. Keep Z-up so the world frame lines up with
   Gazebo's convention.
2. **Wrap with an SDF world.** Reference the exported mesh from a
   `world.sdf`, attach physics + lighting, and spawn the AutoNav robot
   model. Launch with `ign gazebo world.sdf`.
3. **Bridge to ROS 2.** Use `ros_gz_bridge` to forward simulated
   camera / LiDAR / GPS / IMU / odom topics into the AutoNav node
   graph exactly the way real hardware would. The HUD, navigation
   stack, and behavior tree see the simulator the same as the Jetson
   sensor stack.

---

## Object properties drive the simulation

`Course.blend` isn't just geometry. Each course element — every ramp,
cone, waypoint marker, surface patch, GPS spoofer zone — is annotated
with **Blender custom properties** (the per-object key/value bag
accessible as `obj["key"]`). They carry the metadata the robot stack
expects:

- ramp grade in degrees
- GPS waypoint lat/lon
- surface tag (`grass`, `gravel`, `concrete`) → costmap class
- spoof / jam / multipath zones for the GPS sim

A small Blender-Python script dumps every object's custom properties
to JSON so the Fortress export step can stamp them onto the SDF world
(as `<plugin>` payloads, link metadata, or auxiliary YAML the AutoNav
launch consumes):

```python
# Run inside Blender's text editor against Course.blend.
import bpy, json

scene = {
    obj.name: {k: obj[k] for k in obj.keys() if not k.startswith("_")}
    for obj in bpy.data.objects
    if obj.keys()  # skip objects with no custom props
}
print(json.dumps(scene, indent=2, default=str))
```

Keep property names stable — downstream Fortress wrappers and the
AutoNav stack key off them.

---

## Related

- Main robot stack: [`AutoNav_2025-2026`](../../AutoNav_2025-2026) —
  the ROS 2 / Jetson workspace this scene drives in Fortress.
- Root: [`../README.md`](../README.md) — overview of the per-subsystem
  Python simulators.
