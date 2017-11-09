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


""" Connect to the robot """
def connect():
    link.connect()
    return
    

""" Start the robot """
def start():
    link.start()
    return
    

""" Get visual ovservation from sensor """    
def get_observation():
    de = link.get_image_depth()
    return de