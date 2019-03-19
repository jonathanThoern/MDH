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
        self.max_speed = 20.0  # [m/s] 0.02
        self.min_speed = 0   # [m/s]
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s] -good value
        self.max_accel = 2.0  # [m/ss]
        self.max_dyawrate = 1500.0 * math.pi / 180.0  # [rad/ss]
        self.v_reso = 0.01  # [m/s]
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s] 
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 1.0
        self.speed_cost_gain = 1.0
        self.robot_radius = 0.3  # car radius(?) + obstacle radius(0.10) [m] = 0.3
        self.x_comp=1
        self.y_invert=416
        self.y=1
def extract_obs(j,boxes,config):
    x = boxes[j]
    y = boxes[j+1]
    X1,Y1 = np.meshgrid(x,y)
    ob = np.array([X1.flatten(), Y1.flatten()]).T
    return ob
#ymin, ymax, xmin, xmax
def extract_lines(j,boxes,config):

    retur=2
    # Extract corners of detected lines and obstacles
    ymin=config.y_invert-boxes[j]
    ymax=config.y_invert-boxes[j+1]
    xmin_comp=boxes[j+2]-208
    xmax_comp=boxes[j+3]-208

    xmin=boxes[j+2]
    xmax=boxes[j+3]

    y = [192,205,248]
    y = np.sort(y)
    dist = [1,2,3]
    ymin = np.interp(ymin, y, dist)
    ymax = np.interp(ymax, y, dist)

    # Calculate middle
    y = (ymax + ymin)/2

    # Calculate middle
    x = (xmax + xmin)/2
    

    # Check if line can be determined with confidence 
    if x>228:
        retur=0 # right
    elif x<188:
        retur=1 # left

    x = (xmax_comp + xmin_comp)/2

    pixelW = y * 0.75
    PixelPm= pixelW / 416
    x = PixelPm * x

        

        

    X1,Y1 = np.meshgrid(x,y)
    ob = np.array([X1.flatten(), Y1.flatten()]).T
        
    return ob, retur
    

def motion(x, u, dt):
    # motion model
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]

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
    #print("y:",y)
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


def main(gx, gy, d_lines, boxes,prev_angle): #gx=0.0, gy=5.0
    config = Config()
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0, 0, 90*(math.pi / 180.0), 0.0, 0.0])
    # goal position [x(m), y(m)]
    ref = np.array([0,config.x_comp])
    goal = np.array([gx, gy])
    u = np.array([0.0, 0.0])
    
    traj = np.array(x)

    

    ob = []
    ob = np.array(ob)
    j = 1
    retur = 3
    for i in range(0, boxes[0]):
        temp_ob=extract_obs(j,boxes,config)
        j = j+2
        if i !=0:
            ob = np.concatenate((ob,temp_ob), axis = 0)
        else:
            ob = temp_ob

    retur = 3
    right_L = []
    right_L = np.array(right_L)
    left_L = []
    left_L = np.array(left_L)
    j = 1
    add_l=0
    add_r=0
    for i in range(0, d_lines[0]):

        temp_line, retur = extract_lines(j,d_lines,config)
        j = j+4
        if retur == 0:
            if add_r != 0:
                right_L = np.concatenate((right_L,temp_line), axis = 0)
            else:
                right_L = temp_line
            add_r+= 1
        elif retur == 1:
            if add_l !=0:
                left_L = np.concatenate((left_L,temp_line), axis = 0) # CHANGE CONCATENATE, i!=0 galler ej
            else:
                left_L = temp_line
            add_l+= 1
        elif retur == 2:
            if prev_angle < 0:
                temp_line[add_l][0] = -0.45
                if add_l !=0:
                    left_L = np.concatenate((left_L,temp_line), axis = 0) 
                else:
                    left_L = temp_line
            else:
                temp_line[add_r][0] = 0.45
                if add_r != 0:
                    right_L = np.concatenate((right_L,temp_line), axis = 0)
                else:
                    right_L = temp_line
                add_r+= 1
        
    for i in range(1000):# change to a lower loop-value
        u, ltraj = dwa_control(x, u, config, goal, ob)

        x = motion(x, u, config.dt)
        traj = np.vstack((traj, x))  # store state history

        left_L_sum = [[0.0,0.0]]
        left_L_sum = np.array(left_L_sum)

        right_L_sum = [[0.0,0.0]]
        right_L_sum = np.array(right_L_sum)

        if show_animation:
            plt.cla()
            plt.plot(ltraj[:, 0], ltraj[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            if len(right_L) != 0:
                plt.plot(right_L[0][0], right_L[0][1], "or")
            if len(left_L) != 0:
                plt.plot(left_L[0][0], left_L[0][1], "om")
            #plt.plot(-0.75, 3, "om")
            plt.plot(ref[0], ref[1], "og")
            plt.plot(ob[:, 0], ob[:, 1], "ob")
            plot_arrow(x[0], x[1], x[2])
            plt.axis([-3.75,3.75,-4,5])
            plt.grid(True)
            plt.pause(0.0001)

        # check goal
        if math.sqrt((x[0] - goal[0])**2 + (x[1] - goal[1])**2) <= config.robot_radius:
            print("Goal!!")
            break

        #Magnitude right line
        mag_right = 0
        mag_r_temp = 0
        
        if len(right_L)!= 0 and add_r!=0:
            for i in range(0,add_r):
                mag_r_temp= math.sqrt((right_L[i][0] - ref[0])**2 + (right_L[i][1]*(0.9/5) - ref[1]*(0.9/5))**2)
                mag_right = mag_right + mag_r_temp
                right_L_sum[0][0] = right_L[i][0] + right_L_sum[0][0]
                right_L_sum[0][1] = right_L[i][1] + right_L_sum[0][1]
            #mag_right = mag_right/len(right_L)
            #right_L_sum = right_L_sum/len(right_L)
            mag_right = mag_right/add_r
            right_L_sum = right_L_sum/add_r
            
        else:
            temp_r_l = [[1, 5]]
            temp_r_l = np.array(temp_r_l)
            right_L = temp_r_l
            mag_right = math.sqrt((right_L[0][0] - ref[0])**2 + (right_L[0][1]*(0.9/5) - ref[1]*(0.9/5))**2)
            right_L_sum = right_L
            
        mag_left = 0
        mag_l_temp = 0
        
        #Magnitude left line
        if len(left_L) != 0 and add_l!=0:
            for i in range(0,add_l):
                mag_l_temp= math.sqrt((left_L[i][0] - ref[0])**2 + (left_L[i][1]*(0.9/5) - ref[1]*(0.9/5))**2)
                mag_left = mag_left + mag_l_temp
                left_L_sum[0][0] = left_L[i][0] + left_L_sum[0][0]
                left_L_sum[0][1] = left_L[i][1] + left_L_sum[0][1]
            mag_left=mag_left/len(left_L)
            left_L_sum = left_L_sum/len(left_L)
        else:
            temp_l_l = [[-1, 5]]
            temp_l_l = np.array(temp_l_l)
            left_L = temp_l_l
            mag_left = math.sqrt((temp_l_l[0][0] - ref[0])**2 + (temp_l_l[0][1]*(0.9/5) - ref[1]*(0.9/5))**2)
            #mag_left = math.sqrt((left_L[0][0] - ref[0])**2 + (left_L[0][1] - ref[1])**2)
            #left_L_sum = left_L
            left_L_sum = temp_l_l

        #Calculate goal based on magnitude
        mag_tot=0
        mag_tot = (mag_right + mag_left) /2
        x_cord_r=0
        x_cord_l=0
        x_cord_r = right_L_sum[0][0] + math.sqrt(mag_tot**2 - (right_L_sum[0][1]*(0.9/5) - (ref[1]+config.y)*(0.9/5))**2)
        x_cord_l = left_L_sum[0][0] - math.sqrt(mag_tot**2 - (left_L_sum[0][1]*(0.9/5)- (ref[1]+config.y)*(0.9/5))**2)
        goal[0] = (x_cord_r  + x_cord_l)/2
        goal[1] = ref[1] + config.y 

        # Previous implementation of reallocation - do we still need it?
        """
        angle = math.degrees(math.acos(math.cos(x[2]))) #- 90 # postive angle -> left turn
        print("calc angle:", angle)
        if boxes[0] == 0:
            goal[0] = round(((x_cord_r+config.x_comp) + (x_cord_l+config.x_comp))/2,1)
        else:
            m=0
            for l in range(0,boxes[0]):
                if angle == math.degrees(math.atan2(ob[m][m+1]-x[1],ob[m][m]-x[0])):
                    if goal[0] > ob[m][m]:
                        goal[0] = round(((x_cord_r+config.x_comp) + (x_cord_l+config.x_comp))/2,1) + 2
                        m+2
                    else:
                        goal[0] = round(((x_cord_r+config.x_comp) + (x_cord_l+config.x_comp))/2,1) - 2
                        m+2

        goal[1] = ref[1] + config.y

        if angle >= 30:
            angle = 30
        elif angle <= -30:
            angle = -30
        print("angle:", math.degrees(x[2]))
    
        """
        # Only test - placing goal on obstacle 
        #ob[0][0] = goal[0] 
        #ob[0][1] = goal[1] 

        # Calculate steering angle based on trajectory
        line_angle = math.degrees(ltraj[len(ltraj)-1][2])
        angle_temp=line_angle - math.degrees(x[2])

    if show_animation:
        plt.plot(traj[:, 0], traj[:, 1], "-r")
        plt.pause(0.0001)
    plt.show()     
    return angle_temp


#if __name__ == '__main__':
 #   main()
