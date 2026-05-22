# Design Intent

## Robust GPS ENU direction irrelevant converging algorithm

- The robot only has access to its own GPS position and local coordinates.
- Assume there is some imaginary plane with north assumed to be where the robot is facing, anchored at some fixed datum point.
- Place a candidate goal in the frame of the imaginary plane corresponding to the ENU projection, assuming that the plane's north is True.
- Allow the robot to drive in whatever direction it would like to.
- Gather four headings, each pair in its own frame of reference.
- Take the robot's GPS heading sampled from movement and the GPS heading assuming movement towards the goal exactly. Angle in GPS space measured CCW from north.
- The heading in that imaginary frame (assisted by the robot's local coordinates) and the angle to the candidate goal measured CCW from fake north.
- According to the DOF of the system, if R is known, then if theta is equal, the position of the candidate goal and the real GPS goal are equal.
- Collapse the error in theta with successive iterations.

## Clarifications

- The imaginary plane doesn't move at all, it is just fake north gets updated which seeds how ENU projects the next candidate.
- This version of the algorithm does not require the robot to travel towards the GPS goal.
