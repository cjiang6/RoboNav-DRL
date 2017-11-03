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

expset.N_REPETITIONS = 1
expset.N_EPISODES = 1
expset.N_STEPS = 60 * 60 

expset.DISPLAY_STEP = 10

robonav_drl.run()