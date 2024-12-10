import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner
# Define constants
MAX_ROAD_WIDTH = 10.0  # Maximum road width (meters)
D_ROAD_W = 0.5  # Sampling resolution for lateral movement
MIN_T = 1.0  # Minimum time to complete parking maneuver
MAX_T = 5.0  # Maximum time to complete parking maneuver
DT = 0.1  # Time step for trajectory calculation
TARGET_SPEED = 10.0  # Target speed in m/s
D_T_S = 0.5  # Speed resolution in m/s
N_S_SAMPLE = 10  # Number of samples for speed
K_J = 1.0  # Weight for jerk
K_T = 1.0  # Weight for time
K_D = 1.0  # Weight for distance from target
K_LAT = 1.0  # Weight for lateral cost
K_LON = 1.0  # Weight for longitudinal cost
ROBOT_RADIUS = 1.5  # Radius of the robot
 
# Define the parking slot position and dimensions
parking_slot = np.array([20.0, 10.0])  # Parking slot location
parking_length = 6.0  # Length of the parking spot
parking_width = 4  # Width of the parking spot
 
# Define the car parameters
car_length = 5.0
car_width = 3.0
 
# Example of FrenetPath and QuinticPolynomial classes (simplified versions)
 
class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # Initial conditions: xs, vxs, axs, and final conditions: vxe, axe
        # time is the total time duration for the motion
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0  # a2 is half of initial acceleration
 
        # Set up the system of equations to solve for a3 and a4
        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
       
        # Solve for a3 and a4 using linear system solver
        x = np.linalg.solve(A, b)
        self.a3 = x[0]
        self.a4 = x[1]
 
    # Calculate position at time t
    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4
 
    # Calculate first derivative (velocity) at time t
    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
 
    # Calculate second derivative (acceleration) at time t
    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
 
    # Calculate third derivative (jerk) at time t
    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t
 
 
# Example: Calculate a quartic polynomial trajectory with initial and final velocities
# Convert velocities from km/h to m/s
vxs_kmh = 50  # initial velocity in km/h
vxe_kmh = 0   # final velocity in km/h
 
vxs = vxs_kmh * (1000 / 3600)  # convert km/h to m/s
vxe = vxe_kmh * (1000 / 3600)  # convert km/h to m/s
 
# Assume initial position and acceleration to be 0, and final acceleration to be 0 as well
xs = 0  # initial position in meters
axs = 0  # initial acceleration in m/s²
axe = 0  # final acceleration in m/s²
time = 10  # total time in seconds
 
# Create a QuarticPolynomial instance
qp = QuarticPolynomial(xs, vxs, axs, vxe, axe, time)
 
# Calculate the position, velocity, acceleration, and jerk at a specific time t
t = 5  # let's evaluate at time t=5 seconds
position = qp.calc_point(t)
velocity = qp.calc_first_derivative(t)
acceleration = qp.calc_second_derivative(t)
jerk = qp.calc_third_derivative(t)
 
# Print results
print(f"Position at t={t}s: {position} meters")
print(f"Velocity at t={t}s: {velocity} m/s")
print(f"Acceleration at t={t}s: {acceleration} m/s²")
print(f"Jerk at t={t}s: {jerk} m/s³")
 
class FrenetPath:
    def __init__(self):
        self.s = []  # Longitudinal path
        self.d = []  # Lateral path
        self.s_d = []  # Longitudinal velocity
        self.s_dd = []  # Longitudinal acceleration
        self.s_ddd = []  # Longitudinal jerk
        self.d_d = []  # Lateral velocity
        self.d_dd = []  # Lateral acceleration
        self.d_ddd = []  # Lateral jerk
        self.t = []  # Time steps
        self.cd = 0.0  # Cost for lateral movement
        self.cv = 0.0  # Cost for longitudinal movement
        self.cf = 0.0  # Total cost
 
def calc_frenet_paths_for_parking(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, parking_slot):
    frenet_paths = []
   
    # Generate paths with gradual transition to the parking slot
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):  # Vary lateral starting position
        for Ti in np.arange(MIN_T, MAX_T, DT):  # Time horizon for path
            fp = FrenetPath()
           
            # Lateral trajectory: Move smoothly towards the center of the parking slot
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, parking_slot[1], 0.0, 0.0, Ti)  # End at center of parking slot
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
 
            # Longitudinal trajectory: Smooth deceleration to stop in the parking slot
            lon_qp = QuinticPolynomial(s0, c_speed, c_accel, parking_slot[0], 0.0, 0.0, Ti)  # Move from start to slot
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
 
            # Calculate costs
            Jp = sum(np.power(fp.d_ddd, 2))  # Jerk penalty for lateral movement
            Js = sum(np.power(fp.s_ddd, 2))  # Jerk penalty for longitudinal movement
            ds = (TARGET_SPEED - fp.s_d[-1]) ** 2  # Speed deviation penalty at final point
 
            fp.cd = K_J * Jp + K_T * Ti + K_D * fp.d[-1] ** 2
            fp.cv = K_J * Js + K_T * Ti + K_D * ds
            fp.cf = K_LAT * fp.cd + K_LON * fp.cv
 
            frenet_paths.append(fp)
 
    return frenet_paths
 
# Function to simulate the car moving along the best Frenet path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
 
def simulate_car_movement(frenet_paths, parking_slot, car_length, car_width):
    best_path = min(frenet_paths, key=lambda path: path.cf)
   
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-5, 40)
    ax.set_ylim(-10, 20)
 
    # Parking slot position: center at (20, 10), with length 6 and width 4
    adjusted_x = parking_slot[0] - parking_slot[2] / 2
    adjusted_y = parking_slot[1] - parking_slot[3] / 2
 
    # Draw parking slot with adjusted position
    ax.add_patch(patches.Rectangle(
        (adjusted_x, adjusted_y),  # Adjusted coordinates for the parking slot
        parking_slot[2],  # Length (6 meters)
        parking_slot[3],  # Width (4 meters)
        linewidth=2,
        edgecolor='g',
        facecolor='none',
        label="Parking Slot"
    ))
 
    # Add the desired alignment line for the car's entry (dashed green line)
    ax.plot([parking_slot[0] - parking_slot[2] / 2, parking_slot[0] + parking_slot[2] / 2],
            [parking_slot[1], parking_slot[1]], 'g--', label="Desired Alignment")
 
    # Car initial position (start of the path)
    car_rect = patches.Rectangle(
        (best_path.s[0] - car_length / 2, best_path.d[0] - car_width / 2),
        car_length,
        car_width,
        linewidth=2,
        edgecolor='r',
        facecolor='none',
        label="Car"
    )
    ax.add_patch(car_rect)
 
    def update(frame):
        # Update car's position along the best path
        car_center_x = best_path.s[frame]
        car_center_y = best_path.d[frame]
        car_rect.set_xy((car_center_x - car_length / 2, car_center_y - car_width / 2))
        return car_rect,
 
    # Animate the car moving along the best path
    ani = FuncAnimation(fig, update, frames=len(best_path.s), interval=100, blit=True)
 
    ax.set_xlabel('Longitudinal Position (s) [m]')
    ax.set_ylabel('Lateral Position (d) [m]')
    ax.legend()
    ax.grid()
 
    plt.title("Car Moving Along Best Frenet Path")
    plt.show()
 
 
def visualize_frenet_paths(frenet_paths):
    plt.figure(figsize=(10, 6))
    for path in frenet_paths:
        plt.plot(path.s, path.d, label="Path", alpha=0.6)
    plt.title("Frenet Paths for Parking")
    plt.xlabel("Longitudinal Position (s) [m]")
    plt.ylabel("Lateral Position (d) [m]")
    plt.grid(True)
    plt.legend()
    plt.show()
   
 
def check_collision_parking(fp, ob, parking_slot):
    # Check if any obstacles are within the path
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for (ix, iy) in zip(fp.s, fp.d)]
        if any([di <= ROBOT_RADIUS ** 2 for di in d]):
            return False
   
    # Ensure the car is aligned and within the parking slot boundaries
    if not (min(parking_slot[0], parking_slot[0] + parking_slot[2]) <= fp.s[-1] <= max(parking_slot[0], parking_slot[0] + parking_slot[2])):
        return False
    MAX_ALIGN_ERROR = 0.1
    # Check that the car ends up centered in the parking slot
    if abs(fp.d[-1] - parking_slot[1]) > MAX_ALIGN_ERROR:  # Allow a small tolerance for misalignment
        return False
   
    return True
 
 
# Visualization of parking scenario
def visualize_parking(frenet_paths, parking_slot=[20, 10, 6, 4], car_length=4, car_width=2):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-5, 40)
    ax.set_ylim(-10, 20)
   
    # Draw parking slot with adjusted position at (20, 10)
    adjusted_x = parking_slot[0] - parking_slot[2] / 2
    adjusted_y = parking_slot[1] - parking_slot[3] / 2
   
    # Draw the parking slot
    ax.add_patch(patches.Rectangle(
        (adjusted_x, adjusted_y),  # Adjusted coordinates
        parking_slot[2],  # Slot length
        parking_slot[3],  # Slot width
        linewidth=2,
        edgecolor='g',
        facecolor='none',
        label="Parking Slot"
    ))
   
    # Visualize trajectory of all paths
    for path in frenet_paths:
        ax.plot(path.s, path.d, color='gray', alpha=0.5)
 
    # Highlight the best path
    best_path = min(frenet_paths, key=lambda path: path.cf)
    ax.plot(best_path.s, best_path.d, label="Best Path", color='b', linewidth=2)
   
    # Add the car at the final position of the best path
    car_center_x = best_path.s[-1]
    car_center_y = best_path.d[-1]
    car_rect = patches.Rectangle(
        (car_center_x - car_length / 2, car_center_y - car_width / 2),
        car_length,
        car_width,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label="Car"
    )
    ax.add_patch(car_rect)
 
    # Add labels and legend
    ax.set_xlabel('Longitudinal Position (s) [m]')
    ax.set_ylabel('Lateral Position (d) [m]')
    ax.legend()
    ax.grid()
    plt.title("Frenet Paths and Parking Slot Visualization")
    plt.show()
 
 
 
# Main simulation loop
def main():
    # Define initial conditions
    c_speed = 10.0  # Initial speed in m/s
    c_accel = 0.0  # Initial acceleration in m/s^2
    c_d = 0.0  # Initial lateral offset (in meters)
    c_d_d = 0.0  # Initial lateral velocity
    c_d_dd = 0.0  # Initial lateral acceleration
    s0 = 0.0  # Initial longitudinal position (along the road center)
 
    # Generate Frenet paths for parking
    frenet_paths = calc_frenet_paths_for_parking(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, parking_slot)
    visualize_frenet_paths(frenet_paths)
    # Check for collisions and visualize the result
    visualize_parking(frenet_paths, parking_slot=[20, 10, 6, 4], car_length=4, car_width=2)
    simulate_car_movement(frenet_paths, parking_slot=[20, 10, 6, 4], car_length=4, car_width=2)
 
if __name__ == "__main__":
    main()
