# -*- coding: utf-8 -*-
"""
Learning Process

Created on Wed Nov  8 10:33:38 2017

@author: jesse
"""

import time

import numpy as np

import expset
import show
import task
import robot

learning_module = "algorithm_" + expset.ALGORITHM
learning_algorithm = __import__(learning_module)

def setup():
    global s, rAll
    
    rAll = 0 # total rewards in each episode
    s = robot.get_observation() # first sensory observation
    #print(s)
    
    #learning_algorithm.setup()

#def run():
    