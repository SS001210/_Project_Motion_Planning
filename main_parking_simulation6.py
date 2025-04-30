import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
import matplotlib.transforms as transforms
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner
from matplotlib.patches import Rectangle

SIM_LOOP = 1000

# Parameter
MAX_SPEED = 80.0 / 3.6  # mpltimum speed [m/s]
mplt_rspeed = 50.0 /3.6
MAX_ACCEL = 10.0  # mpltimum acceleration [m/ss]
mplt_raccel = 1.0
MAX_CURVATURE = 1.0  # mpltimum curvature [1/m]
MAX_ROAD_WIDTH = 10.0  # mpltimum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # mplt prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 40.0 / 3.6  # target speed of moition [m/s]
v1 = 0.0 / 3.6 # target speed of forward [m/s]
D_T_S = 4.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 6  # sampling number of target speed
road_width = 6  # Width of the road (meters)
road_length = 80  
ego_length = 4 #car_length [m]
ego_width = 2 #car_width [m]
parkx = 50.0 # x coordinate for the parking 
parky = -8.0 # y coordinate for the parking 
pspot_length = 4.0
pspot_breadth = 6.0
# cost weights
K_J = 0.1
K_T = 1
K_D = 1
K_LAT = 1.0
K_LON = 2.0 

show_animation = True

class QuarticPolynomial:

    def __init__(self, xs, vxs, plts, vxe, plte, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = plts / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      plte - 2 * self.a2])
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


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, state):
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
            current_target_speed = 30.0/3.6 if state == "motion" else v1
            for tv in np.arange(current_target_speed - D_T_S * N_S_SAMPLE,
                                current_target_speed + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (current_target_speed - tfp.s_d[-1]) ** 2

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
            if fp.yaw == 0 :
                return None
            else:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_paths(fplist):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Mplt speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Mplt accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Mplt curvature check
            continue
        #elif not check_collision(fplist[i], ob):
            #continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, state):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, state)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist)
    
    if fplist is None:
       print("No valid paths after check_paths")
       return None, None
    
    fplist.sort(key=lambda x: x.cf)
    
    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    
    if not fplist:
      print("No valid paths generated after checking constraints!")
      return None, None
     
    # sorted_fplist = fplist.sort(key=lambda x: x.cf)
    
    return best_path, fplist

                
def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    tx, ty, tyaw, tk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        tx.append(ix)
        ty.append(iy)
        tyaw.append(csp.calc_yaw(i_s))
        tk.append(csp.calc_curvature(i_s))

    return tx, ty, tyaw, tk, csp

def generate_s_trajectory(x0, y0, parkx, parky, parking_spot_length, parking_spot_breath, offset=15, m0=0, mf=0.05, k0=0, kf=0, num_points=15):
    """
    Generate an S-shaped trajectory between two points.
    
    Parameters:
        x0, y0 (float): Starting point coordinates.
        xf, yf (float): Ending point coordinates.
        m0, mf (float): Initial and final slopes (default: 0 for both).
        k0, kf (float): Initial and final curvatures (default: 0 for both).
        num_points (int): Number of points to generate along the trajectory.
        plot (bool): Whether to plot the trajectory (default: False).
    
    Returns:
        x (np.ndarray): Array of x-coordinates of the trajectory.
        y (np.ndarray): Array of y-coordinates of the trajectory.
    """
    # Construct the system of equations to solve for polynomial coefficients
    x0 = parkx - offset 
    y0 = 0
    xf = parkx + 1
    yf = parky
    A = np.array([
        [x0**5, x0**4, x0**3, x0**2, x0, 1],
        [xf**5, xf**4, xf**3, xf**2, xf, 1],
        [5*x0**4, 4*x0**3, 3*x0**2, 2*x0, 1, 0],
        [5*xf**4, 4*xf**3, 3*xf**2, 2*xf, 1, 0],
        [20*x0**3, 12*x0**2, 6*x0, 2, 0, 0],
        [20*xf**3, 12*xf**2, 6*xf, 2, 0, 0]
    ])
    b = np.array([y0, yf, m0, mf, k0, kf])

    # Solve for coefficients
    coefficients = np.linalg.solve(A, b)

    # Generate x and y points
    x = np.linspace(x0, xf, num_points)
    y = np.polyval(coefficients, x)
    return x,y   

def main():
    print(__file__ + " start!!")
    area = 40.0
   
    wx = [0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0]
    wy = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy) 
    # Initial state for forward motion
    c_speed = TARGET_SPEED  # Current speed [m/s]
    c_accel = -1.0  # Current acceleration [m/s^2]
    c_d = 0.0  # Current lateral position [m]
    c_d_d = 0.0  # Current lateral speed [m/s]
    c_d_dd = 0.0  # Current lateral acceleration [m/s^2]
    s0 = 0.0  # Current course position

    state = 'motion'
    
    # Initial state for forward parking
      # Parking speed [m/s]
    f_accel = -2.5  # Parking acceleration [m/s^2]
    f_d = 0.0  # Parking lateral position [m]
    f_d_d = 0.0  # Parking lateral speed [m/s]
    f_d_dd = 0.0  # Parking lateral acceleration [m/s^2]
    fs0 = 0.0  # Current Parking course position
    
    # Parking slot orientation check
    is_parallel = pspot_length > pspot_breadth

    for i in range(SIM_LOOP):
        if state == 'motion':
            path, fplist = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, state)
            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]
            c_accel = path.s_dd[1]

            ego_y = path.y[1]
            ego_x = path.x[1]
            f_speed = c_speed
            # Current x position of the vehicle
            if state == 'motion' and (parkx - ego_x <= 15):
                state = "forward"
                fx, fy = csp.calc_position(path.s[-1])
                if is_parallel:
                    fx, fy = csp.calc_position(path.s[-1])
                    x1, y1 = generate_s_trajectory(fx, fy, parkx, parky, pspot_length, pspot_breadth)
                    tx, ty, tyaw, tc, csp = generate_target_course(x1, y1)
                else:
                    if parky < 0 :  
                        fvx = [ego_x, ego_x + 1, parkx, parkx, parkx]
                        fvy = [ego_y, ego_y, parky + 4, parky + 2, parky]
                        tx, ty, tyaw, tc, csp = generate_target_course(fvx, fvy)
                    else :
                        fvx = [ego_x, ego_x + 1, parkx, parkx, parkx]
                        fvy = [ego_y, ego_y, parky - 4, parky - 2, parky]
                        tx, ty, tyaw, tc, csp = generate_target_course(fvx, fvy)
                continue

        elif state == 'forward':
            path, fplist = frenet_optimal_planning(csp, fs0, f_speed, f_accel, f_d, f_d_d, f_d_dd, state)
            fs0 = path.s[1]
            f_d = path.d[1]
            f_d_d = path.d_d[1]
            f_d_dd = path.d_dd[1]
            f_speed = path.s_d[1]
            f_accel = path.s_dd[1]
            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
                break

        if show_animation:
            # Plot trajectory path without clearing the figure
            plt.cla()  # Again, use plt.clear() to prevent resetting
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot([0, road_length], [road_width / 2, road_width / 2], 'k', linewidth=2)
            plt.plot([0, road_length], [-road_width / 2, -road_width / 2], 'k', linewidth=2)
            #road_y1, road_y2 = 10, -10
            #plt.plot([min(tx), mplt(tx)], [road_y2, road_y2], 'black', linewidth=2)
            px, py, yaw = path.x[1], path.y[1], path.yaw[1]
            ego_yaw = path.yaw[1] * 180 /np.pi
            print(f"px: {px}, py: {py}, yaw: {yaw}")  # Debugging info

            #Plot the Top 10 paths
            for i, fp in enumerate(fplist[:200]):  #inlky the top 10 paths
                plt.plot(fp.x, fp.y, label=f"Path {i+1} (cost: {fp.cf:.4f})", linestyle="--")
                
            ego_vehicle = Rectangle((px - ego_length / 2.0, py - ego_width / 2.0), ego_length, ego_width, facecolor = 'Red',edgecolor='Black')
            t = transforms.Affine2D().rotate_deg_around(path.x[1], path.y[1], ego_yaw) + plt.gca().transData
            parkingspot = Rectangle((parkx - pspot_length / 2, parky - pspot_breadth / 2), pspot_length, pspot_breadth, linewidth=2, edgecolor='blue')
            ego_vehicle.set_transform(t)      
            plt.gca().add_patch(parkingspot)
            plt.gca().add_patch(ego_vehicle)
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")   
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("speed[km/h]:" + str(f_speed*3.6)[0:4])
            plt.grid(True)
            plt.pause(0.04)
            #print("GOAL2")

    print("Finish")

    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.04)
        plt.show()


if __name__ == '__main__':
    main()
