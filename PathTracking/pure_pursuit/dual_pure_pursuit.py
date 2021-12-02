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
Lfc = 3.0  # [m] look-ahead distance
Lfc_carrot2 = 1.0
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 2.9  # [m] wheel base of vehicle
TREAD = 0.5 # [m] 
show_animation = True

class StateSteerModel:
    def __init__(self, x, y, yaw, v, steer_right_pos, steer_left_pos):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.target_ind = 0
        self.target_ind_carrot2 = 0
        self.steer_right_pos = steer_right_pos
        self.steer_left_pos = steer_left_pos
        self.steer_right_x = x + (TREAD/2) * math.cos(yaw + steer_right_pos)
        self.steer_right_y = y + (TREAD/2) * math.sin(yaw + steer_right_pos)
        self.steer_left_x = x + (TREAD/2) * math.cos(yaw - steer_left_pos)
        self.steer_left_y = y + (TREAD/2) * math.sin(yaw - steer_left_pos) 

    def update(self, a, omega, right_steer, left_steer):
        self.yaw += omega * dt
        self.v += a * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.steer_right_pos = right_steer 
        self.steer_left_pos = left_steer

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
        self.steer_right_pos = []
        self.steer_left_pos = []

    def append_steer(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)
        self.steer_right_pos.append(state.steer_right_pos)
        self.steer_left_pos.append(state.steer_left_pos) 

def proportional_control(target, current):
    a = Kp * (target - current)

    return a

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None
        self.old_nearest_point_carrot2_index = None
    
    def search_target_index_steer_robot(self, state_steer_model):
    
        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state_steer_model.x - icx for icx in self.cx]
            dy = [state_steer_model.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
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

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_carrot2_index is None:
            # search nearest point index
            dx = [state_steer_model.x - icx for icx in self.cx]
            dy = [state_steer_model.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind_carrot2 = np.argmin(d)
            self.old_nearest_point_carrot2_index = ind_carrot2
        else:
            ind_carrot2 = self.old_nearest_point_carrot2_index
            distance_this_index = state_steer_model.calc_distance(self.cx[ind_carrot2],
                                                      self.cy[ind_carrot2])
            while True:
                distance_next_index = state_steer_model.calc_distance(self.cx[ind_carrot2 + 1],
                                                          self.cy[ind_carrot2 + 1])
                if distance_this_index < distance_next_index:
                    break
                ind_carrot2 = ind_carrot2 + 1 if (ind_carrot2 + 1) < len(self.cx) else ind_carrot2
                distance_this_index = distance_next_index
            self.old_nearest_point_carrot2_index = ind_carrot2

        Lf_carrot2 = k * state_steer_model.v + Lfc_carrot2  # update look ahead distance

        # while Lf > state_steer_model.calc_distance(self.cx[ind], self.cy[ind]):
        #     if (ind + 1) >= len(self.cx):
        #         break  # not exceed goal
        #     ind += 1
        
        # ind_carrot2 = ind
        # while Lfc_carrot2 < state_steer_model.calc_distance(self.cx[ind_carrot2], self.cy[ind_carrot2]):
        #     ind_carrot2 -= 1
        #     if ind_carrot2 < 0:
        #         ind_carrot2 = 0
        #         break
            
        # Lf_carrot2 = k * state_steer_model.v + Lfc_carrot2

        return ind, Lf, ind_carrot2, Lf_carrot2

def pure_pursuit_robot_steer_control(state, trajectory, pind, target_speed):
    ind, Lf, ind_2, lf2= trajectory.search_target_index_steer_robot(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
        
        tx2 = trajectory.cx[ind_2]
        ty2 = trajectory.cy[ind_2]
        
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    omega = target_speed * alpha / Lf
    r = math.fabs(target_speed / omega)
    
    # delta = math.atan2(alpha * target_speed, Lf)
    delta = math.atan2(ty2 - state.y, tx2 - state.x) - state.yaw    
    
    omega = target_speed * alpha / Lf
    right_steer = math.atan2(r * math.sin(delta), r * math.cos(delta) - TREAD/2.0)
    left_steer = math.atan2(r * math.sin(delta), r * math.cos(delta) + TREAD/2.0)

    return delta, alpha, omega, right_steer, left_steer, ind, ind_2

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
    cy = [-1 *math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    # hitoigomi course
    # cy = [math.sin(ix / 5.0) * ix / 2.0  + math.cos(ix / 50) *ix /5 for ix in cx]
    # cy = [math.sin(ix / 2.0) for ix in cx]

    target_speed = 10.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = StateSteerModel(x=-0.0, y=-3.0, yaw=0.0, v=0.0, steer_right_pos=0.0, steer_left_pos=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append_steer(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, fake, target_ind_carrot2, fake_ = target_course.search_target_index_steer_robot(state)

    while T >= time and lastIndex > target_ind:

        # Calc control input
        ai = proportional_control(target_speed, state.v)

        #simulate augumented steer robot model
        di, alpha, omega, right_steer, left_steer, target_ind, target_ind_carrot2= pure_pursuit_robot_steer_control(
            state, target_course, target_ind, target_speed)
        state.update(ai, omega, right_steer, left_steer)        
        time += dt
        states.append_steer(time, state)

        print(target_ind, target_ind_carrot2)

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
            plt.plot(cx[target_ind_carrot2], cy[target_ind_carrot2], "xy", label="target")
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
        # plt.savefig(result_dir + "speed_" + date + ".png")

        plt.subplots(1)
        plt.plot(states.t, [ir  * 180 / 3.14 for ir in states.steer_right_pos], "-r")
        plt.plot(states.t, [il * 180 / 3.14 for il in states.steer_left_pos], "-b")
        plt.xlabel("Time[s]")
        plt.ylabel("Steer[rad]")
        plt.grid(True)
        # plt.savefig(result_dir + "steer_" + date + ".png")
        plt.show()

if __name__ == '__main__':
    print("===========================================")
    print("Pure pursuit path tracking simulation start")
    print("===========================================")
    main()
