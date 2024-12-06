# _Project_Motion_Planning
**Moral**:We have created this for better team work

**Goal**:To implement the Frenet Optimal trajectory(FOT) Methoic in diffent scenarios like Forward parking, Overtaking and Waiting for a pedestrian etc.

**Core Steps in Frenet Algorithm**:

Lateral Motion Planning- Generate lateral paths (d values) for different offsets using quintic polynomials. The paths account for the vehicle's initial lateral position, speed, and acceleration.

Longitudinal Motion Planning- Generate longitudinal paths (s values) using quartic polynomials. These paths are designed to reach a target speed or stop while minimizing jerk and acceleration.

Path Validation- Convert Frenet paths to global Cartesian coordinates using the cubic spline planner.
Check for collisions and ensure paths comply with dynamic constraints (e.g., max curvature, speed, and acceleration).

Cost Calculation-Compute a cost function for each path, combining lateral and longitudinal motion costs (jerk, time, deviation from target, etc.).

Path Selection- Among valid paths, choose the one with the minimum cost.

**Future**:Using the Basics of frenet we can implement in many complex scenarios which can be even utilized in the 3D space. But for now we just use the 2D space for depecting the complexity

**POC for ForwardParking** : Priyansh + Saee
**POC for WaitForPedestrian/Overtake** : Prem + Zain


