# -*- coding: utf-8 -*-
"""
Task configuration: Hallway

Created on Fri Nov  3 11:43:38 2017

@author: jesse
"""

import numpy as np

import robot


# Task parameters:
NAME = "navigation_hallway"
ROBOT = "Pioneer 3DX with laser or Kinect"

STEP_TIME = 1 # (s)
MAX_SPEED = 0.5 # maximum motor speed (m/s)
RANGE_COLL = 0.08 # minimum distance to obstacles (m)
RANGE_GOAL = 0.1 # minimum distance for which goal is considered being reached


# Task goal (rewards):
REWARD = np.array([10, 2.5, -10])
def get_reward():
    dist_to_goal = robot.dist_goal
    dist_to_obs = min(robot.dist_obstacle)
    
    if dist_to_goal < RANGE_GOAL:
        r1 = max(REWARD)
    else:
        r1 = dist_to_goal * REWARD[1]
        
    if dist_to_obs < RANGE_COLL:
        r2 = min(REWARD)
    else:
        r2 = 0
        
    r = r1 + r2
    
    return r
    


def setup():
    pass    