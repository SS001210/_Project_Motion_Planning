import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
from matplotlib.transforms import Affine2D
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner
from matplotlib.patches import Rectangle

SIM_LOOP = 1000

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
max_rspeed = 50.0 /3.6
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
max_raccel = 1.0
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 10.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 2.0 / 3.6  # target speed [m/s]
D_T_S = 4.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 6  # sampling number of target speed
ego_length = 5 #car_length [m]
ego_width = 2 #car_width [m]
parkx = 50.0 # x coordinate for the parking 
parky = -12.0 # y coordinate for the parking 
pspot_length = 6.0
psport_breadth = 4.0
# cost weights
K_J = 0.1
K_T = 0.1
K_D = 0.1
K_LAT = 1.0
K_LON = 0.1

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


#def check_collision(fp,ob):
    #for i in range(len(ob[:, 0])):
        #d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             #for (ix, iy) in zip(fp.x, fp.y)]

        #collision = any([di <= ego_width **2 for di in d]) #because robot radius is underoot of x2 + y2 components 

        #if collision:
            #return False

    #return True


def check_paths(fplist):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        #elif not check_collision(fplist[i], ob):
            #continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path

                
def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    fx, fy, fyaw, fk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        fx.append(ix)
        fy.append(iy)
        fyaw.append(csp.calc_yaw(i_s))
        fk.append(csp.calc_curvature(i_s))

    return fx, fy, fyaw, fk, csp
#def calculateB()
def calculate_waypoints(parkx, wy_value=0.0, interval=5):
   
    # Generate wx values from 0 to parking_x + 5 with the specified interval
    wx = [i for i in range(0, int(parkx - 20.0) + 1, interval)]
    
    # Ensure the last waypoint reaches exactly parking_x + 5
    if wx[-1] != parkx -20.0:
        wx.append(parkx - 20.0)
    
    # Create wy with the same number of elements as wx
    wy = [wy_value] * len(wx)
    
    return wx, wy
#def generate_parking_target_course(tx, ty, parkx, parky):
    # Generate Parking waypoints starting from the last point of the forward path to the parking spot
    #wx = [tx[-1]]  # Start at the last point of the forward path
    #wy = [ty[-1]]  # Start at the last point of the forward path
    # Generate a path that Parkings towards the parking spot
    #while np.hypot(wx[-1] - parkx, wy[-1] - parky) > 1.0:  # Ensure we are not already at the parking spot
        #direction_x = np.sign(parkx - wx[-1])  # Determine the direction to move in x
        #direction_y = np.sign(parky - wy[-1])  # Determine the direction to move in y

        # Move step-by-step towards the parking spot with a step size (e.g., 1.0)
        #new_x = wx[-1] + direction_x * 1.0
        #new_y = wy[-1] + direction_y * 1.0

        #wx.append(new_x)
        #wy.append(new_y)

    #return wx, wy

def main():
    print(__file__ + " start!!")
    area = 20.0

    # Initial state for forward motion
    c_speed = 30.0 / 3.6  # Current speed [m/s]
    c_accel = 2.0  # Current acceleration [m/s^2]
    c_d = 0.0  # Current lateral position [m]
    c_d_d = 0.0  # Current lateral speed [m/s]
    c_d_dd = 0.0  # Current lateral acceleration [m/s^2]
    s0 = 0.0  # Current course position

    wx, wy = calculate_waypoints(parkx)
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    j =1
    while (j ==1 ):
        j= j+1
        for i in range(SIM_LOOP):
            path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd)
            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]
            c_accel = path.s_dd[1]

            if np.hypot(tx[-1] - path.x[1] , ty[-1] - path.y[1]) <= 1.0:  
                print("GOAL1")   
                break

            if show_animation:
                plt.cla()
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(tx, ty)
                road_y1, road_y2 = 10, -10
                plt.plot([min(tx), max(tx)], [road_y1, road_y1], 'black', linewidth=2)
                plt.plot([min(tx), max(tx)], [road_y2, road_y2], 'black', linewidth=2)
                px, py, yaw = path.x[1], path.y[1], path.yaw[1]
                ego_vehicle = plt.Rectangle((px - ego_length / 2, py - ego_width / 2), ego_length, ego_width, edgecolor='Black')
                parkingspot = plt.Rectangle((parkx - pspot_length / 2, parky - psport_breadth / 2), pspot_length, psport_breadth, linewidth=2, edgecolor='blue')
                plt.gca().add_patch(parkingspot)
                plt.gca().add_patch(ego_vehicle)
                plt.plot(path.x[1:], path.y[1:], "-or")
                plt.plot(path.x[1], path.y[1], "vc")
                plt.xlim(path.x[1] - area, path.x[1] + area)
                plt.ylim(path.y[1] - area, path.y[1] + area)
                plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
                plt.grid(True)
                plt.pause(0.02)
                 
    

        # Initial state for forward parking
        f_speed = 0  # Parking speed [m/s]
        f_accel = 2.0  # Parking acceleration [m/s^2]
        f_d = 0.0  # Parking lateral position [m]
        f_d_d = 0.0  # Parking lateral speed [m/s]
        f_d_dd = 0.0  # Parking lateral acceleration [m/s^2]
        fs0 = 0.0  # Current Parking course position

        #print("Switching to Parking motion")
        fvx = [tx[-1], tx[-1]+0.5,parkx-4, parkx-2, parkx]
        fvy = [ty[-1], ty[-1], parky, parky, parky]
        tx, ty, tyaw, tc, csp = generate_target_course(fvx, fvy)

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(csp, fs0, f_speed, f_accel, f_d, f_d_d, f_d_dd)
        fs0 = path.s[1]
        f_d = path.d[1]
        f_d_d = path.d_d[1]
        f_d_dd = path.d_dd[1]
        f_speed = path.s_d[1]
        f_accel = path.s_dd[1]
        
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            road_y1, road_y2 = 10, -10
            plt.plot([min(tx), max(tx)], [road_y1, road_y1], 'black', linewidth=2)
            plt.plot([min(tx), max(tx)], [road_y2, road_y2], 'black', linewidth=2)
            px, py, yaw = path.x[1], path.y[1], path.yaw[1]
            transform = Affine2D().rotate(yaw).translate(px, py) + plt.gca().transData
            ego_vehicle.set_transform(transform)
            ego_vehicle = plt.Rectangle((px - ego_length / 2, py - ego_width / 2), 
            ego_length, ego_width, edgecolor='Black')
            plt.gca().add_patch(ego_vehicle)
            parkingspot = plt.Rectangle((parkx - pspot_length / 2, parky - psport_breadth / 2),
             pspot_length, psport_breadth, linewidth=2, edgecolor='blue')
            plt.gca().add_patch(parkingspot)
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(f_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.02)
            print("GOAL2")

    print("Finish")

    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.02)
        plt.show()


if __name__ == '__main__':
    main()