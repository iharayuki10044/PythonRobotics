"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.ma.core import right_shift

import datetime

# Parameters
k = 0.1  # look forward gain
Lfc = 2.0  # [m] look-ahead distance
Lfc2 = 0.5
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle
TREAD = 0.5 # [m] 
show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.v_right = 0.0
        self.v_left = 0.0
        self.right_pos = 0.0
        self.left_pos = 0.0

    def update(self, delta, a_right, a_left, steer_right, steer_left):
        self.v_right += a_right * dt
        self.v_left += a_left * dt 
        self.v = (self.v_right + self.v_left) / 2.0
        
        self.right_pos = steer_right
        self.left_pos = steer_left
        
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []
        self.v_right = []
        self.v_left = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)
        self.v_right.append(state.v_right)
        self.v_left.append(state.v_left)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None
        self.olf_nearest_point_index2 = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf
    
    def search_target_index_steer_robot(self, state_steer_model):
    
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state_steer_model.x - icx for icx in self.cx]
            dy = [state_steer_model.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index2 = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state_steer_model.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state_steer_model.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state_steer_model.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state_steer_model.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1
     # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state_steer_model.x - icx for icx in self.cx]
            dy = [state_steer_model.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind2 = np.argmin(d)
            self.old_nearest_point_index2 = ind2
        else:
            ind2 = self.old_nearest_point_index2
            distance_this_index = state_steer_model.calc_distance(self.cx[ind2],
                                                      self.cy[ind2])
            while True:
                distance_next_index = state_steer_model.calc_distance(self.cx[ind2 + 1],
                                                          self.cy[ind2 + 1])
                if distance_this_index < distance_next_index:
                    break
                ind2 = ind2 + 1 if (ind2 + 1) < len(self.cx) else ind2
                distance_this_index = distance_next_index
            self.old_nearest_point_index2 = ind2

        Lf2 = k * state_steer_model.v + Lfc2  # update look ahead distance

        # search look ahead target point index
        while Lf2 > state_steer_model.calc_distance(self.cx[ind2], self.cy[ind2]):
            if (ind2 + 1) >= len(self.cx):
                break  # not exceed goal
            ind2 += 1

        return ind, Lf, ind2, Lf2

def pure_pursuit_robot_control(state, trajectory, pind, target_speed):
    ind, Lf, ind2, Lf2 = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
        tx2 = trajectory.cx[ind2]
        ty2 = trajectory.cy[ind2]
        
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    delta = math.atan2(ty2 - state.y, tx2 - state.x) - state.yaw
    omega = state.v * alpha / Lfc

    radius = state.v / omega
    
    di = math.atan2(radius * math.sin(delta), radius * math.cos(delta) - WB*0.5)
    do = math.atan2(radius * math.sin(delta), radius * math.cos(delta) + WB*0.5)

    if omega > 0:
        steer_right = di
        steer_left = do
    else:
        steer_right = do
        steer_left = di
 
    vr = target_speed + 0.5 * omega * Lf
    vl = target_speed - 0.5 * omega * Lf

    return delta, alpha, omega, vr, vl, steer_right, steer_left, ind, ind2

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    #  target course
    cx = np.arange(0, 50, 0.5)
    
    # default course
    # cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    cy = [2.0*math.sin(ix /1.5) + 3.0*math.sin(ix/5.0) + 5.0*math.sin(ix/10.0) for ix in cx]

    # hitoigomi course
    # cy = [math.sin(ix / 5.0) * ix / 2.0  + math.cos(ix / 50) *ix /5 for ix in cx]
    # cy = [math.sin(ix / 2.0) for ix in cx]

    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=-0.0, y=-3.0, yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)

    target_ind, fake, target_ind2, fake_ = target_course.search_target_index_steer_robot(state)


    while T >= time and lastIndex > target_ind:

        # Calc control input
        # ai = proportional_control(target_speed, state.v)

        # simulate augumented dynamic of bicycle model
        di, alpha, omega, vr, vl, steer_right, steer_left, target_ind, target_ind2= pure_pursuit_robot_control(
            state, target_course, target_ind, target_speed)
        
        a_right = proportional_control(vr , states.v_right[-1])
        a_left = proportional_control(vl , states.v_left[-1])
        
        state.update(di, a_right, a_left, steer_right, steer_left)  # Control vehicle
        time += dt
        states.append(time, state)
    
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        
        date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%y_%m%d_%H%M")
        result_dir= "./result/"
        
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)
        # plt.savefig(result_dir +  "trajectory_" + date + ".png")

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        plt.plot(states.t, [ir * 3.6 for ir in states.v_right], "-g")
        plt.plot(states.t, [il * 3.6 for il in states.v_left], "-b")
        # plt.savefig(result_dir + "steer_" + date + ".png")
        plt.show()

if __name__ == '__main__':
    print("===========================================")
    print("Pure pursuit path tracking simulation start")
    print("===========================================")
    main()
