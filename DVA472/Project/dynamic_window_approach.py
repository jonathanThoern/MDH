"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg, dot, sum, tile
import math

import sys
sys.path.append("../../")


show_animation = True


class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = 0  # [m/s]
        self.max_yawrate = 30.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_dyawrate = 30.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 1.0
        self.robot_radius = 0.3  # [m]


def motion(x, u, dt):
    # motion model

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin,vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj


def calc_final_input(x, u, dw, config, goal, ob):

    xinit = x[:]
    min_cost = 10000.0
    min_u = u
    min_u[0] = 0.0
    best_traj = np.array([x])

    # evalucate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(xinit, v, y, config)

            # calc cost
            to_goal_cost = calc_to_goal_cost(traj, goal, config,x)
            speed_cost = config.speed_cost_gain * \
                (config.max_speed - traj[-1, 3])
            ob_cost = calc_obstacle_cost(traj, ob, config)
            #print(ob_cost)

            final_cost = to_goal_cost + speed_cost + ob_cost 

            #print (final_cost)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                min_u = [v, y]
                best_traj = traj

    return min_u, best_traj


def calc_obstacle_cost(traj, ob, config):
    # calc obstacle cost inf: collistion, 0:free

    skip_n = 2
    minr = float("inf")

    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in range(len(ob[:, 0])):
            
            ox = ob[i, 0]
            oy = ob[i, 1]      
            dx = traj[ii, 0] - ox 
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)
            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0 / minr  # OK


def calc_to_goal_cost(traj, goal, config,x):
    # calc to goal cost. It is 2D norm.
    goal_magnitude = math.sqrt(goal[0]**2 + goal[1]**2)
    traj_magnitude = math.sqrt(traj[-1, 0]**2 + traj[-1, 1]**2)
    dot_product = (goal[0] * traj[-1, 0]) + (goal[1] * traj[-1, 1])
    error = dot_product / (goal_magnitude * traj_magnitude)
    
    error_angle = math.acos(error)
    cost = config.to_goal_cost_gain * error_angle

    return cost


def dwa_control(x, u, config, goal, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u, traj = calc_final_input(x, u, dw, config, goal, ob)

    return u, traj


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def main(gx=0.0, gy=5.0):
    y_increment = 5.0
    x_temp_inc = 0.0
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # obstacles [x(m) y(m), ....]
    ob_lineL = []
    ob_lineR = []
    ob_lineRx = []
    ob_lineLx = []
    y_lineL = 0
    x_lineL = -5
    y_lineR = 0
    x_lineR = 5
    while y_lineL<=20:
        ob_lineL.append([x_lineL,y_lineL])
        y_lineL = y_lineL + 0.1
        y_lineL = round(y_lineL,1)
        
    while y_lineR<=25:
        ob_lineR.append([x_lineR,y_lineR])
        y_lineR = y_lineR + 0.1
        y_lineR = round(y_lineR,1)

    while x_lineR>=-20:
        ob_lineRx.append([x_lineR,y_lineR])
        x_lineR = x_lineR - 0.1
        x_lineR = round(x_lineR,1)

    while x_lineL>=-25:
        ob_lineLx.append([x_lineL,y_lineL])
        x_lineL = x_lineL - 0.1
        x_lineL = round(x_lineL,1)
        
        
    ob_lineL = np.array(ob_lineL)
    ob_lineR = np.array(ob_lineR)
    ob_lineLx = np.array(ob_lineLx)
    ob_lineRx = np.array(ob_lineRx)

    #Line_left = np.concatenate((ob_lineL,ob_lineLx), axis =0)
    #Line_right = np.concatenate((ob_lineR,ob_lineRx), axis =0)
    Line_left = ob_lineL
    Line_right = ob_lineR
    #Line_left = np.split(Line_left,10)
    #Line_right = np.split(Line_right,10)
    
    #print(Line_left.shape)
    #print(Line_right.shape)

    #ob = np.array([[0, 2],[5.0, 4.0],[-5.0, -5.0],[-5.0, -6.0],[-5.0, -9.0],[-8.0, -9.0],[-7.0, -9.0],[-12.0, -12.0]])
    ob = np.array([[-0.5, 15.0]])

    u = np.array([0.0, 0.0])
    config = Config()
    traj = np.array(x)

    left = 0
    right = 0
    ink_y = True
    
    #while(1):
    for i in range(1000):
        u, ltraj = dwa_control(x, u, config, goal, ob)

        x = motion(x, u, config.dt)
        traj = np.vstack((traj, x))  # store state history
        print("x:",x)

        # print(traj)

        if show_animation:
            plt.cla()
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plt.plot(ob_lineL[:, 0], ob_lineL[:, 1], "ok")
            plt.plot(ob_lineR[:, 0], ob_lineR[:, 1], "ok")
            #plt.plot(ob_lineRx[:, 0], ob_lineRx[:, 1], "ok")
            #plt.plot(ob_lineLx[:, 0], ob_lineLx[:, 1], "ok")
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
            
        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.robot_radius:
            print("Goal!!")
            break
        if(ink_y == True):
            goal[1] = x[1] + y_increment
            goal[1] =round(goal[1],1)
            
        k=len(Line_left)
        j=len(Line_right)
        h=0
        for i in Line_left:
            
            #print("i",i[1])
            if i[1] == goal[1]:
                left = i[0]
                ink_y = True
                break
            elif(h == k-1):
                ink_y = False
                x_temp_inc = x_temp_inc + 0.1
                x_temp_inc = round(x_temp_inc,1)
                #left = -20
            h=h+1
                
        h=0
        for i in Line_right:
            if i[1] == goal[1]:
                right = i[0]
                break
            #elif(h == j-1):
                #right = 20
            h=h+1
        if(x_temp_inc > y_increment):
            x_temp_inc = y_increment

        if(ink_y == True):
            goal[0] = left + (right - left)/2
            goal[0] = round(goal[0],1)
        else:
            goal[0] = x[0] - x_temp_inc
            goal[0] = round(goal[0],1)

    print("Done")
    if show_animation:
        plt.plot(traj[:, 0], traj[:, 1], "-r")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
