# -*- coding: utf-8 -*-
"""
Deep Q-Learning algorithm

Created on Wed Nov  8 11:16:05 2017

@author: jesse
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import random
from skimage import transform

#import expset
#import lp

class DeepQNetwork():
    def __init__(self, obs_dim1, obs_dim2, action_size):        
#        # Old version -----------------------------------------------------        
#        # observation input
#        self.observation = tf.placeholder(tf.float32, shape = (None, obs_dim1, obs_dim2))
#                
#        # Network Parameters
#        conv1_num = 16
#        conv2_num = 32
#        fc1_num = 256
#
#        # convolutions acti:relu
#        self.conv1 = tf.contrib.layers.convolution2d(
#            inputs=tf.expand_dims(self.observation, 3),
#            num_outputs = conv1_num,
#            kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
#        
#        self.conv2 = tf.contrib.layers.convolution2d(
#            inputs=self.conv1,
#            num_outputs = conv2_num,
#            kernel_size=[4,4], stride=[2,2], padding='VALID', biases_initializer=None)
#        
#        self.convout = tf.contrib.layers.flatten(self.conv2)
#        # End of Old version ----------------------------------------------
        
        # Modified version -------------------------------------------------
        # observation input
        self.scalarInput = tf.placeholder(shape=[None,obs_dim1*obs_dim2*3], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,obs_dim1,obs_dim2,3])
        # Network Parameters
        conv1_num = 16
        conv2_num = 32
        fc1_num = 256
        # convolutions acti:relu
        self.conv1 = slim.conv2d(
            inputs=self.imageIn,num_outputs=conv1_num,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
        self.conv2 = slim.conv2d(
            inputs=self.conv1,num_outputs=conv2_num,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
        self.convout = tf.contrib.layers.flatten(self.conv2)
        # End of Modified version ------------------------------------------
        
        # Dueling
        # Do it later

        # fully-connected
        self.W1 = tf.Variable(tf.random_normal([self.convout.get_shape().as_list()[1], fc1_num]))
        self.b1 = tf.Variable(tf.random_normal([fc1_num]))
        self.fc1 = tf.nn.relu( tf.matmul(self.convout, self.W1) + self.b1 ) 
        
        self.W2 = tf.Variable(tf.random_normal([fc1_num, action_size]))
        self.b2 = tf.Variable(tf.random_normal([action_size]))
        # Out
        self.Qout = tf.matmul(self.fc1, self.W2) + self.b2
        self.predict = tf.argmax(self.Qout, 1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        

class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
        

def processState(states, dim1, dim2):
    states = transform.resize(states, (dim1, dim2, 3))
    return np.reshape(states,[dim1*dim2*3])
    
    
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    