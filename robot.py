# -*- coding: utf-8 -*-
"""
Robot module

Created on Fri Nov  3 11:58:48 2017

@author: jesse
"""

import math
import sys

import numpy as np

import expset

link_dict = {'ROS': 'robot_ros', 'VREP': 'robot_vrep'}
try:
    link_module = link_dict[expset.ENVIRONMENT_TYPE]
except KeyError:
    sys.exit("ENVIRONMENT_TYPE" + expset.ENVIRONMENT_TYPE + "undefined \n")
link = __import__(link_module)


""" Relavent data of the robot and environment"""
mobilebase_pose2d = np.full(3, -1, dtype=np.float64) # Robot pose [x(m), y(m), theta]
goal_pose2d = np.full(2,-1, dtype=np.float64) # Goal position [x(m), y(m)]
goal_relpose2d = np.full(2,-1, dtype=np.float64) # Goal relative position in robot frame [x(m), y(m)]
dist_obstacle = np.full(link.N_Ultrasonic, -1, dtype=np.float64) # Distance (m) to obstacles measured by ultrasonic sensors
dist_goal = -1 # Distance (m) from robot to goal
last_dist_goal = -1 # Distance (m) from robot to goal of last timestep


""" Initialize the robot and env. info """
def setup():
    global mobilebase_pose2d, goal_pose2d, goal_relpose2d
    global dist_obstacle, dist_goal, last_dist_goal    
    
    mobilebase_pose2d = np.full(3, -1, dtype=np.float64)
    goal_pose2d = np.full(2, -1, dtype=np.float64)
    goal_relpose2d = np.full(2, -1, dtype=np.float64)
    dist_obstacle = np.full(link.N_Ultrasonic, -1, dtype=np.float64) 
    dist_goal = -1
    last_dist_goal = -1
    # get the first data of the robot and environment
    update()
    return


""" Connect to the robot """
def connect():
    link.connect()
    return
    

""" Start the robot """
def start():
    link.start()
    return
    

""" Update relavent data of the robot and environment """
def update():
    global mobilebase_pose2d, goal_pose2d, goal_relpose2d
    global dist_obstacle, dist_goal, last_dist_goal
    
    mobilebase_pose2d = get_mobilebase_pose2d()
    goal_pose2d = get_goal_pose_2d()
    goal_relpose2d = get_goal_relpose_2d()
    dist_obstacle = get_distance_obstacle()
    last_dist_goal = dist_goal
    dist_goal = distance2d(np.array([mobilebase_pose2d[0], mobilebase_pose2d[1]]), goal_pose2d) 
#    print("Robot position: " + "[" + str(mobilebase_pose2d[0]), str(mobilebase_pose2d[1]), "]")
#    print("Goal position: " + "[" + str(goal_pose2d[0]), str(goal_pose2d[1]), "]")
#    print("Goal relative position: " + "[" + str(goal_relpose2d[0]), str(goal_relpose2d[1]), "]")
#    print("Distance to obstacle: %s" %(min(dist_obstacle)))
#    print("Distance to goal: " + str(dist_goal))
#    print("Last distance to goal: " + str(last_dist_goal))

""" Calculate 2D distance between two pose """
def distance2d(pose_a, pose_b):
    delta_dist = abs(pose_a - pose_b)
    dist = math.sqrt(delta_dist[0] ** 2 + delta_dist[1] ** 2)
    return dist
    


""" Get visual ovservation from sensor """    
def get_observation():
    if expset.SENSOR_TYPE == 'KINECT':
        de = link.get_image_depth()
    elif expset.SENSOR_TYPE == 'LASER':
        pass
    return de
    

""" Return 2D pose of the mobilebase (x,y,theta) """
def get_mobilebase_pose2d():
    pos = link.get_mobilebase_pose2d()
    return pos


""" Return an array of distances measured by ultrasonic sensors (m) """
def get_distance_obstacle():
    dist = link.get_distance_obstacle()
    return dist


""" Returns the position of the goal object:  [ x(m), y(m) ] """
def get_goal_pose_2d():
    pos = link.get_goal_pose_2d()
    return pos
    
    
""" Returns the relative position of the goal object to the robot:  [ x(m), y(m) ] """
def get_goal_relpose_2d():
    pos = link.get_goal_relpose_2d()
    return pos
    
    
""" Move robot wheels """
def move_wheels(left_wheel, right_wheel):
    # input equivalent to discretized angular velocity
    link.move_wheels(left_wheel, right_wheel)
    return