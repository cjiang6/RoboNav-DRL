# -*- coding: utf-8 -*-
"""
Experiment parameters setup

Created on Fri Nov  3 11:13:50 2017

@author: jesse
"""

import time
import numpy as np

# Basic parameters -------------------------------------------------------
TASK_ID = "hallway" # Select a scenario
ENVIRONMENT_TYPE = "VREP" # Select an experiment platform: VREP or ROS
SENSOR_TYPE = "KINECT" # Select a sensory input: KINECT or LASER
SPEED_RATE = 3.0 # Recommended: VREP: 3.0 (x3); REAL ROBOT: 1.0 (x1)

N_REPETITIONS = 1  # Number of repetitions of the experiment
N_EPISODES = 100  # >1 for episodic experiments: Uses arrays from previous epi
N_STEPS = 60 * 60  # Number of steps in each epi: 1 step ~ 1 second (Sets LE.N_STEPS)

CONTINUE_PREVIOUS_EXP = False
PREVIOUS_EXP_FILE = ""

DISPLAY_STEP = 1800  # Policy will be printed each DISPLAY_STEP step


# Learning parameters -----------------------------------------------------
ALGORITHM = "DQN" # Deep Q-learning 


# Other parameters -------------------------------------------------------
EPISODIC = False


""" Check experiment parameters """
def check():
    
    global N_STEPS, EPISODIC, DISPLAY_STEP

    EPISODIC = False

    if N_EPISODES > 1:
        EPISODIC = True

    N_STEPS = int(N_STEPS)

    if ENVIRONMENT_TYPE == "VREP" and SPEED_RATE == 1:
        print("\n\n\n\n WARNING: VREP WITH SPEED_RATE = 1 \n\n")
        time.sleep(10)

    if ENVIRONMENT_TYPE == "ROS" and SPEED_RATE > 1:
        print("\n\n\n\n WARNING: ROS WITH SPEED_RATE: ", SPEED_RATE, "\n\n")
        time.sleep(10)

    if not DISPLAY_STEP:
        DISPLAY_STEP = int(1e6)

