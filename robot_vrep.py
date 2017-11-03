# -*- coding: utf-8 -*-
"""
Link to VREP simulator

Created on Fri Nov  3 12:00:57 2017

@author: jesse
"""
import time

import numpy as np

import vrep

WAIT_RESPONSE = False  # True: Synchronous response (too much delay)

LASER_DISTRIBUTION = ('sensor_front', 'sensor_front_left', 'sensor_left',
                      'sensor_rear_left', 'sensor_rear', 'sensor_rear_right',
                      'sensor_right', 'sensor_front_right')

HAS_LASER = False
HAS_KINECT = True


# V-REP data transmission modes:
WAIT = vrep.simx_opmode_oneshot_wait
ONESHOT = vrep.simx_opmode_oneshot
STREAMING = vrep.simx_opmode_streaming
BUFFER = vrep.simx_opmode_buffer

if WAIT_RESPONSE:
    MODE_INI = WAIT
    MODE = WAIT
else:
    MODE_INI = STREAMING
    MODE = BUFFER

# Robot components 
N_LASERS = 8  # 1 point laser each

robotID = -1
laserID = [-1] * N_LASERS
left_motorID = -1
right_motorID = -1
clientID = -1

kinect_rgb_ID = -1  # not used so far
kinect_depth_ID = -1  # not used so far

distance = np.full(N_LASERS, -1, dtype=np.float64)  # distances from lasers (m)
pose = np.full(3, -1, dtype=np.float64)  # Pose 2d base: x(m), y(m), theta(rad)


""" send a message for printing in V-REP """
def show_msg(message):
    vrep.simxAddStatusbarMessage(clientID, message, WAIT)
    return


""" Connect to the simulator"""
def connect():
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
    show_msg('Python: Hello')
    time.sleep(0.5)
    return
    

""" Disconnect from the simulator"""
def disconnect():
    # Make sure that the last command sent has arrived
    vrep.simxGetPingTime(clientID)
    show_msg('RoboNav-DRL: Bye')
    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
    time.sleep(0.5)
    return
    

""" Start the simulation (force stop and setup)"""
def start():    
    stop()
    setup_devices()
    vrep.simxStartSimulation(clientID, ONESHOT)
    time.sleep(0.5)
    # Solve a rare bug in the simulator by repeating:
    setup_devices()
    vrep.simxStartSimulation(clientID, ONESHOT)
    time.sleep(0.5)
    return
    

""" Stop the simulation """
def stop():
    vrep.simxStopSimulation(clientID, ONESHOT)
    time.sleep(0.5)
    

""" Assign the devices from the simulator to specific IDs """
def setup_devices():
    global robotID, left_motorID, right_motorID, laserID
    global kinect_rgb_ID, kinect_depth_ID
    # rc: return_code (not used)
    # robot
    rc, robotID = vrep.simxGetObjectHandle(clientID, 'robot', WAIT)
    # motors
    rc, left_motorID = vrep.simxGetObjectHandle(clientID, 'leftMotor', WAIT)
    rc, right_motorID = vrep.simxGetObjectHandle(clientID, 'rightMotor', WAIT)    
    # lasers
    for idx, item in enumerate(LASER_DISTRIBUTION):
        ec, laserID[idx] = vrep.simxGetObjectHandle(clientID, item, WAIT)    
    # Kinect
    if HAS_KINECT:
        rc, kinect_rgb_ID = vrep.simxGetObjectHandle(
            clientID, 'kinect_rgb', WAIT)
        rc, kinect_depth_ID = vrep.simxGetObjectHandle(
            clientID, 'kinect_depth', WAIT)

    # start up devices

    # wheels
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0, STREAMING)
    # pose
    vrep.simxGetObjectPosition(clientID, robotID, -1, MODE_INI)
    vrep.simxGetObjectOrientation(clientID, robotID, -1, MODE_INI)
    # distances from lasers
    for i in laserID:
        vrep.simxReadProximitySensor(clientID, i, MODE_INI)

    if HAS_KINECT:
        rc, resolution, image = vrep.simxGetVisionSensorImage(
            clientID, kinect_rgb_ID, 0, MODE_INI)
        im = np.array(image, dtype=np.uint8)
        im.resize([resolution[1], resolution[0], 3])
        # plt.imshow(im, origin='lower')
        # return_code, resolution, depth = vrep.simxGetVisionSensorImage(
        #     clientID, kinect_depth_ID, 0, MODE_INI)
        # de = np.array(depth)
        time.sleep(0.5)
    return
    

""" Get RGB image from a Kinect """
def get_image_rgb():
    rc, resolution, image = vrep.simxGetVisionSensorImage(
        clientID, kinect_rgb_ID, 0, MODE)

    im = np.array(image, dtype=np.uint8)
    im.resize([resolution[1], resolution[0], 3])
    # im.shape
    # plt.imshow(im,origin='lower')
    return im


""" get image Depth from a Kinect """
def get_image_depth():
    rc, resolution, depth = vrep.simxGetVisionSensorImage(
        clientID, kinect_depth_ID, 0, MODE)
    de = np.array(depth)
    return de


""" return the pose of the robot:  [ x(m), y(m), Theta(rad) ] """
def get_mobilebase_pose2d():
    rc, pos = vrep.simxGetObjectPosition(clientID, robotID, -1, MODE)
    rc, ori = vrep.simxGetObjectOrientation(clientID, robotID, -1, MODE)
    pos = np.array([pos[0], pos[1], ori[2]])
    return pos
    

def get_distance_obstacle():
    """ return an array of distances measured by lasers (m) """
    for i in range(0, N_LASERS):
        rc, ds, detected_point, doh, dsn = vrep.simxReadProximitySensor(
            clientID, laserID[i], MODE)
        distance[i] = detected_point[2]
    return distance


""" move the wheels. Input: Angular velocities in rad/s """
def move_wheels(v_left, v_right):
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, v_left, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, v_right, STREAMING)
    return


""" stop the base wheels """
def stop_motion():
    vrep.simxSetJointTargetVelocity(clientID, left_motorID, 0, STREAMING)
    vrep.simxSetJointTargetVelocity(clientID, right_motorID, 0, STREAMING)
    return
    

#""" Returns the position of the goal object:  [ x(m), y(m), z(m) ] """
#def get_goal_pose_3d():   
#    rc, pos = vrep.simxGetObjectPosition(clientID, ballID, -1, MODE)
#    return np.array(pos)