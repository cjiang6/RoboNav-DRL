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

import expset

tasks_path = "tasks/"
results_path = "results/"

""" Perform experiments/simulations, save and show results """
def run():
    
    import sys
    if sys.version_info[0] < 3:
        sys.exit("Python 3 required")
        
    expset.check()
    
    # copy the selected taskfile to speed up the execution:
    try:
        copyfile("tasks/" + expset.TASK_ID + ".py", "task.py")
    except IOError:
        sys.exit("Task " + expset.TASK_ID + " not found. Please check exp.TASK_ID")
    import task
    import robot
#    import lp
#    import show
    import save
    
    task.setup()
    
    caption = (expset.TASK_ID)
#    if expset.SUFFIX:
#        caption += "_" + expset.SUFFIX

    save.new_dir(results_path, caption)  # create result directory

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
        if expset.CONTINUE_PREVIOUS_EXP:
            last_q, last_v = prev_exp.q, prev_exp.v
            last_policy, last_q_count = prev_exp.policy, prev_exp.q_count
        else:
            last_q = last_v = last_policy = last_q_count = None

        # Episode loop ----------------------------------------------------
        for epi in range(expset.N_EPISODES):

            robot.start()

#            show.process_count(caption, rep, epi, expset.EPISODIC)
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

    
