# -*- coding: utf-8 -*-
""" 
RoboNav-DRL main script

Created on Fri Nov  3 11:08:09 2017

@author: jesse

"""
import signal
import sys
import time
from shutil import copyfile

import numpy as np

import tensorflow as tf

import expset
import algorithm_DQN
from algorithm_DQN import DeepQNetwork, experience_buffer

tasks_path = "tasks/"
results_path = "results/"

""" Perform experiments/simulations, save and show results """
def run():
    
    import sys
    if sys.version_info[0] < 3:
        sys.exit("Python 3 required")
        
    expset.check()
    
    # Copy the selected taskfile to speed up the execution:
    try:
        copyfile("tasks/" + expset.TASK_ID + ".py", "task.py")
    except IOError:
        sys.exit("Task " + expset.TASK_ID + " not found. Please check exp.TASK_ID")
    import task
    import robot
    #import lp
    import show
    import save
    
    task.setup()
    
    caption = (expset.TASK_ID)
#    if expset.SUFFIX:
#        caption += "_" + expset.SUFFIX

    path = save.new_dir(results_path, caption)  # Create result directory

    epi = 0
    # Average Reward per step (aveR):
    ave_r = np.zeros((expset.N_REPETITIONS, expset.N_STEPS))
    # Mean(aveR) of all tests per step
    mean_ave_r = np.zeros(expset.N_STEPS)
    # AveR per episode
    epi_ave_r = np.zeros([expset.N_REPETITIONS, expset.N_EPISODES])
    # actual step time per episode (for computational cost only)
    actual_step_time = np.zeros(expset.N_REPETITIONS)
    
    robot.connect()  # Connect to V-REP / ROS

    if expset.CONTINUE_PREVIOUS_EXP:
        prev_exp = __import__(expset.PREVIOUS_EXP_FILE)
        print("NOTE: Continue experiments from: " + expset.PREVIOUS_EXP_FILE)
        time.sleep(3)
        
    # Experiment repetition loop ------------------------------------------
    for rep in range(expset.N_REPETITIONS):
        
        # Training parameters
        action_dic = {'0': 'FORWARD', 
                      '1': 'FULL_RIGHT', 
                      '2': 'FULL_LEFT', 
                      '3': 'HALF_RIGHT',
                      '4': 'HALF_LEFT'}
        batch_size = 32 # How many experiences to use for each training step.
        update_freq = 4 # How often to perform a training step.
        y = .99 # Discount factor on the target Q-values
        startE = 1 # Starting chance of random action
        endE = 0.1 # Final chance of random action
        anneling_steps = 100000 # How many steps of training to reduce startE to endE.
        num_episodes = 500000 # How many episodes of game environment to train network with.
        
        pre_train_steps = 50000 # 10000 #How many steps of random actions before training begins.
        simulation_time = 400 # 200 
        max_epLength = 400 # 200 # the same as simulation time
        tau = 0.001 # Rate to update target network toward primary network
        ob_len = 480 # Size of the image of the environment
        action_size = len(action_dic)
        num_sim_steps = 30000
        load_model = False
        
        # Learning algorithm initialization
        tf.reset_default_graph()
        mainQN = DeepQNetwork(ob_len, action_size)
        targetQN = DeepQNetwork(ob_len, action_size)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        targetOps = algorithm_DQN.updateTargetGraph(trainables, tau)
        copyOps = algorithm_DQN.updateTargetGraph(trainables, 1.0)
        myBuffer = experience_buffer()
        
        #Set the rate of random action decrease
        e = startE
        stepDrop = (startE - endE)/anneling_steps
                
        # Create lists for counting         
        stepList = [] # List of total steps of each epi
        rList = [] # list of total rewards of each epi
        total_steps = 0 # Count total steps of each repetition
        
        
        
        with tf.Session() as sess:
            sess.run(init)
            if load_model == True:
                print("Loading model ... ")
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            # Episode loop ----------------------------------------------------
            for epi in range(expset.N_EPISODES):                
                   
                robot.start()               
                show.process_count(caption, rep, epi)
                robot.setup()
                
                episodeBuffer = experience_buffer()            
                s = robot.get_observation()
                                
                rAll = 0.0 # total reward per each episode
                d = False # if reach the destination
                
                for step in range(0, expset.N_STEPS):
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        a = np.random.randint(0,len(action_dic))
                    else:
                        a = sess.run(mainQN.predict,feed_dict={mainQN.observation:np.expand_dims(s, axis=0)})[0]
                               
                    print("Action is " + str(a) + "at timestep: " + str(step))
                            
                    # Update robot motion
                    move_direction = action_dic[str(a)]
                    if move_direction == 'FORWARD':
                        robot.move_wheels(1*task.MAX_SPEED, 1*task.MAX_SPEED)
                    elif move_direction == 'FULL_RIGHT':
                        robot.move_wheels(1*task.MAX_SPEED, -1*task.MAX_SPEED)
                    elif move_direction == 'FULL_LEFT':
                        robot.move_wheels(-1*task.MAX_SPEED, 1*task.MAX_SPEED)
                    elif move_direction == 'HALF_RIGHT':
                        robot.move_wheels(1.5*task.MAX_SPEED, 0.5*task.MAX_SPEED)
                    elif move_direction == 'HALF_LEFT':
                        robot.move_wheels(0.5*task.MAX_SPEED, 1.5*task.MAX_SPEED)
                        
                    robot.update()        
                    # Get new observation and reward
                    s1 = robot.get_observation() 
                    r = task.get_reward()
                    #d = task.reach_dest()                
                    
                    total_steps += 1                    
                    # Save to experience buffer
                    episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
                    
                    # Update Deep Q-Network
                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop
                        if total_steps % (update_freq) == 0:
                            trainBatch = myBuffer.sample(batch_size) # Get a random batch of experiences
                            # Perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.observation:np.reshage(np.vstack(trainBatch[:,3]), [batch_size, ob_len, 640])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.observation:np.reshape(np.vstack(trainBatch[:,3]), [batch_size, ob_len, 640])})
                            end_multiplier =- (trainBatch[:,4] - 1)
                            doubleQ = Q2[range(batch_size), Q1]
                            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                            # Update the network with our target values
                            _ = sess.run(mainQN.updateModel, feed_dict={ mainQN.observation:np.reshape(np.vstack(trainBatch[:,0]), [batch_size, ob_len, 640]),
                                                                         mainQN.targetQ:targetQ,
                                                                         mainQN.actions:trainBatch[:,1]})
                            # Update the target network toward the primary network
                            algorithm_DQN.updateTarget(targetOps, sess)
                            
                    rAll += r
                    s = s1
                    
                    if d == True: # End the episode if destination is reached
                        break
                    
                    print("Finish timestep: " + str(step))
                # End of one episode ---------------------------------------
             
                myBuffer.add(episodeBuffer.buffer)
                stepList.append(step)
                rList.append(rAll)
                
                # Periodically save the model
                if epi % 1000 == 0:
                    saver.save(sess, path + '/model-' + str(epi) + '.ckpt')
                    print("Model saved")
            
            saver.save(sess, path + '/model-' + str(epi) + '.ckpt')
            
            # End of one repetition
                
                
                
                
#
#            lp.setup()  # Learning process setup
#
#            if (expset.EPISODIC and epi > 0) or expset.CONTINUE_PREVIOUS_EXP:
#                lp.q, lp.v = last_q, last_v
#                lp.policy, lp.count = last_policy, last_q_count
#
#            lp.run()  # Execute the learning process
#
#            ave_r[rep] = lp.ave_r_step
#            actual_step_time[rep] = lp.actual_step_time
#
#            if expset.EPISODIC:
#                last_q, last_v = lp.q, lp.v
#                last_policy, last_q_count = lp.policy, lp.q_count
#
#                epi_ave_r[rep, epi] = lp.ave_r_step[lp.step]
#
#        # end of episode -----------------------------------------------
#
#        show.process_remaining(rep, epi)
#
#        mean_ave_r = np.mean(ave_r, axis=0)

        # End of experiment repetition loop ----------------------------

    # Mean of AveR per step (last episode)

#    save.plot_mean(mean_ave_r, epi)
#
#    save.simple(ave_r, "aveR")
#    #   If EPISODIC: Save ave_r of last episode
#
#    if expset.EPISODIC:
#        # Mean of AveR reached (last step) per episode
#        mean_epi_ave_r = np.mean(epi_ave_r, axis=0)
#        save.plot_mean(mean_epi_ave_r, "ALL")
#        save.simple(epi_ave_r, "EPI")
#
#    final_r = mean_ave_r[lp.step]
#    final_actual_step_time = np.mean(actual_step_time)
#
#    save.log(final_r, final_actual_step_time)
#    save.arrays()
#    print("Mean average Reward = %0.2f" % final_r, "\n")
#    print("Mean actual step time (s): %0.6f" % final_actual_step_time, "\n")

    
