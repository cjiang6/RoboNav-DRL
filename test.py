# -*- coding: utf-8 -*-
"""
Test script

Created on Fri Nov  3 19:56:31 2017

@author: cjiang
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import vrep

WAIT = vrep.simx_opmode_oneshot_wait
ONESHOT = vrep.simx_opmode_oneshot
STREAMING = vrep.simx_opmode_streaming
BUFFER = vrep.simx_opmode_buffer

ip = '127.0.0.1'
port = 19997
vrep.simxFinish(-1)  # just in case, close all opened connections
global clientID
clientID = vrep.simxStart(ip, port, True, True, 3000, 5)
# Connect to V-REP
if clientID == -1:
    import sys
    sys.exit('\nV-REP remote API server connection failed (' + ip +
             ':' + str(port) + '). Is V-REP running?')
print('Connected to Robot')

vrep.simxStopSimulation(clientID, ONESHOT)
vrep.simxStartSimulation(clientID, ONESHOT)

rc, kinect_rgb_ID = vrep.simxGetObjectHandle(clientID, 'kinect_rgb', WAIT)
rc, kinect_depth_ID = vrep.simxGetObjectHandle(clientID, 'kinect_depth', WAIT)

rc, resolution, image = vrep.simxGetVisionSensorImage(clientID, kinect_depth_ID, 0, STREAMING)
time.sleep(0.5)
rc, resolution, image = vrep.simxGetVisionSensorImage(clientID, kinect_depth_ID, 0, STREAMING)

im = np.array(image, dtype=np.uint8)
im.resize([resolution[1], resolution[0], 3])
plt.imshow(im, origin='lower')