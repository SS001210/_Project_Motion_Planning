import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting functions

# Constants for road dimensions, vehicle properties, pedestrian behavior, etc.
ROAD_WIDTH = 10  # Width of the road in meters
ROAD_LENGTH = 50  # Length of the road in meters
VEHICLE_LENGTH = 5  # Length of the vehicle in meters
VEHICLE_WIDTH = 2  # Width of the vehicle in meters
PEDESTRIAN_SPEED = 1  # Pedestrian speed in m/s
MAX_VEHICLE_SPEED = 50 / 3.6  # Convert 50 km/h to m/s
BRAKING_DISTANCE = 5  # Distance at which the car applies emergency braking
SAFE_DISTANCE = 10  # Distance at which pedestrian is considered safe

# Initial positions for car and pedestrian
car_position = [5, ROAD_WIDTH / 2]  # Car starts 5 meters from the start line, centered in the lane
pedestrian_position = [45, ROAD_WIDTH - 1]  # Pedestrian starts 5 meters away from the end of the road on the zebra crossing

# Set up the plot for visualization
fig, ax = plt.subplots(figsize=(10, 5))
plt.ion()  # Enable interactive mode for visualization


class FrenetFrame:
    """
    Represents a vehicle's movement using Frenet coordinates.
    Attributes:
        s (float): Longitudinal position along the road.
        d (float): Lateral offset.
        speed (float): Speed of the vehicle.
        max_speed (float): Maximum speed.
    """

    def __init__(self, start_s, start_d, max_speed):
        self.s = start_s  # Longitudinal position along the road
        self.d = start_d  # Lateral offset
        self.speed = 0  # Initial speed
        self.max_speed = max_speed
        self.slowing_down = False
        self.emergency_braking = False

    def update_position(self, dt):
        """
        Updates the vehicle's position based on current speed and time step.
        :param dt: Time increment in seconds.
        """
        if self.emergency_braking or self.slowing_down:
            if self.speed > self.max_speed * 0.2:  # Slow to 20% of max speed
                self.speed -= 0.5  # Gradual deceleration
            else:
                self.speed = 0  # Stop the car completely
        else:
            if self.speed < self.max_speed:
                self.speed += 0.5  # Gradual acceleration
            self.s += self.speed * dt  # Update position only if not braking

    def apply_emergency_brake(self):
        """Triggers emergency braking."""
        self.emergency_braking = True
        self.slowing_down = False

    def start_slowing_down(self):
        """Starts gradual deceleration."""
        self.slowing_down = True
        self.emergency_braking = False

    def release_brake(self):
        """Releases any braking mechanism."""
        self.emergency_braking = False
        self.slowing_down = False


def plot_environment(car, pedestrian_pos):
    """
    Plots the current state of the road, car, and pedestrian.
    :param car: FrenetFrame object representing the car.
    :param pedestrian_pos: Position of the pedestrian [x, y].
    """
    ax.clear()
    
    # Draw road boundaries
    ax.plot([0, ROAD_LENGTH], [0, 0], 'k-', linewidth=2)
    ax.plot([0, ROAD_LENGTH], [ROAD_WIDTH, ROAD_WIDTH], 'k-', linewidth=2)
    ax.fill_between([0, ROAD_LENGTH], 0, ROAD_WIDTH, color='black')

    # Draw zebra crossing
    stripe_width = 0.8
    for i in range(10):  # Draw 10 alternating zebra crossing stripes
        stripe_start = 40 + i * stripe_width
        ax.fill_betweenx([0, 10], stripe_start, stripe_start + stripe_width / 2, color='white')
        ax.fill_betweenx([0, 10], stripe_start + stripe_width / 2, stripe_start + stripe_width, color='lightgrey')

    # Draw car as a green rectangle
    car_rect = plt.Rectangle((car.s - VEHICLE_LENGTH / 2, car.d - VEHICLE_WIDTH / 2),
                             VEHICLE_LENGTH, VEHICLE_WIDTH, color='green')
    ax.add_patch(car_rect)
    ax.text(car.s, car.d + 1, 'CAR', ha='center', color='white', weight='bold')

    # Draw pedestrian as a red circle
    pedestrian_circle = plt.Circle(pedestrian_pos, 0.5, color='red')
    ax.add_patch(pedestrian_circle)
    ax.text(pedestrian_pos[0], pedestrian_pos[1] + 1, 'Pedestrian', ha='center', color='black', weight='bold')

    # Set plot limits and labels
    ax.set_xlim(0, ROAD_LENGTH)
    ax.set_ylim(-2, ROAD_WIDTH + 2)
    ax.set_aspect('equal')
    ax.set_xlabel("Road Length (m)")
    ax.set_ylabel("Road Width (m)")
    plt.pause(0.1)


def simulate():
    """
    Simulates the movement of the car and pedestrian, including interactions such as braking.
    """
    car = FrenetFrame(start_s=5, start_d=ROAD_WIDTH / 2, max_speed=MAX_VEHICLE_SPEED)
    pedestrian_y = pedestrian_position[1]
    detection_range = 15  # Sensor detection range in meters

    dt = 0.1  # Time step for simulation
    
    # Data for plotting over time
    time_data = []  # Stores time points for speed and distance graphs
    speed_data = []  # Stores speed of the car
    distance_data = []  # Stores distance between car and pedestrian
    
    current_time = 0  # Initialize simulation time

    while car.s < ROAD_LENGTH or pedestrian_position[1] > 0:
        # Move pedestrian downwards on the zebra crossing
        pedestrian_position[1] -= PEDESTRIAN_SPEED * dt  

        # Calculate distance between car and pedestrian
        longitudinal_distance = pedestrian_position[0] - car.s
        lateral_distance = abs(pedestrian_position[1] - car.d)
        distance_to_pedestrian = np.sqrt(longitudinal_distance**2 + lateral_distance**2)

        # Append data for graph plotting
        time_data.append(current_time)
        speed_data.append(car.speed)
        distance_data.append(distance_to_pedestrian)

        # Print current positions for debugging purposes
        print(f"Car position (s): {car.s:.2f}, Speed: {car.speed:.2f}, Pedestrian distance: {distance_to_pedestrian:.2f}")

        # Car's behavior based on pedestrian's distance
        if distance_to_pedestrian <= detection_range:
            if longitudinal_distance > 0 and lateral_distance <= VEHICLE_WIDTH:
                print("Pedestrian detected within range. Slowing down.")
                car.start_slowing_down()

                # Apply emergency braking if pedestrian is too close
                if distance_to_pedestrian <= BRAKING_DISTANCE:
                    print("Pedestrian very close! Emergency braking.")
                    car.apply_emergency_brake()

        # Resume acceleration if pedestrian is in a safe distance
        if car.speed == 0:
            if distance_to_pedestrian > SAFE_DISTANCE:
                print("Pedestrian is in a safe range. Car can accelerate.")
                car.release_brake()
            else:
                print("Pedestrian still near the car. Car remains stopped.")

        # Resume acceleration if pedestrian is out of detection range
        if distance_to_pedestrian > detection_range:
            if not car.slowing_down and not car.emergency_braking:
                print("No pedestrian detected. Car is accelerating.")
                car.release_brake()

        # Update the car's position
        car.update_position(dt)

        # Plot the current state of the environment
        plot_environment(car, pedestrian_position)

        # Exit simulation if both car and pedestrian have crossed the road
        if pedestrian_position[1] < 0 and car.s >= ROAD_LENGTH:
            print("Simulation finished: Car and pedestrian have both crossed.")
            break

        current_time += dt  # Increment the time for the simulation

    plt.ioff()  # Disable interactive mode

    # Plot speed and distance graphs after the simulation
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, speed_data, label='Car Speed (m/s)', color='blue')
    plt.plot(time_data, distance_data, label='Distance to Pedestrian (m)', color='red')
    plt.title('Car Speed and Pedestrian Distance Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed / Distance (m)')
    plt.legend()
    plt.grid()
    plt.show()
simulate()
