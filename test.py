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

N_Ultrasonic = 16
ultrasonicID = [-1]*N_Ultrasonic
distance=np.full(N_Ultrasonic, -1, dtype=np.float64)
detect_state=np.full(N_Ultrasonic, -1, dtype=np.float64)

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

# setup robot components
rc, left_motorID = vrep.simxGetObjectHandle(clientID, 'leftMotor', WAIT)
rc, right_motorID = vrep.simxGetObjectHandle(clientID, 'rightMotor', WAIT)
    
for idx in range(0, N_Ultrasonic):
    item = 'ultrasonicSensor' + str(idx+1)
    rc, ultrasonicID[idx] = vrep.simxGetObjectHandle(clientID, item, WAIT)
    
rc, kinect_rgb_ID = vrep.simxGetObjectHandle(clientID, 'kinect_rgb', WAIT)
rc, kinect_depth_ID = vrep.simxGetObjectHandle(clientID, 'kinect_depth', WAIT)

#time.sleep(10)
# send and receive components data
vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, STREAMING)
vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0, STREAMING)
    
for i in range(0, N_Ultrasonic):
    rc, ds, detected_point, doh, dsn = vrep.simxReadProximitySensor(
        clientID, ultrasonicID[i], STREAMING)
    distance[i] = detected_point[2]
    detect_state[i] = ds
print(distance)
time.sleep(0.5)
for i in range(0, N_Ultrasonic):
    rc, ds, detected_point, doh, dsn = vrep.simxReadProximitySensor(
        clientID, ultrasonicID[i], STREAMING)
    if ds == 1:
        distance[i] = detected_point[2]
    else:
        distance[i] = float('nan')
    detect_state[i] = ds
print(distance)

#vrep.simxStopSimulation(clientID, ONESHOT)
    
rc, resolution, image = vrep.simxGetVisionSensorImage(clientID, kinect_depth_ID, 0, STREAMING)
time.sleep(0.5)
rc, resolution, image = vrep.simxGetVisionSensorImage(clientID, kinect_depth_ID, 0, STREAMING)

im = np.array(image, dtype=np.uint8)
im.resize([resolution[1], resolution[0], 3])
plt.imshow(im, origin='lower')