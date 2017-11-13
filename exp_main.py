# -*- coding: utf-8 -*-
"""
Start an experiment

Created on Fri Nov  3 12:53:38 2017

@author: jesse
"""

import expset
import robonav_drl

expset.ENVIRONMENT_TYPE = "VREP"
expset.TASK_ID = "hallway"
expset.SENSOR_TYPE = "KINECT" 

expset.N_REPETITIONS = 1
expset.N_EPISODES = 10000
expset.N_STEPS = 200 
expset.SPEED_RATE = 3.0

expset.DISPLAY_STEP = 10

expset.ALGORITHM = "DQN"

robonav_drl.run()