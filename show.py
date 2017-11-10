# -*- coding: utf-8 -*-
"""
Message Display

Created on Wed Nov  8 10:25:25 2017

@author: jesse
"""

import expset

""" Show repetition and episode info """
def process_count(caption, rep, epi):
    print("-"*21, "START", "-"*21)
    print(caption)
    
    if expset.N_REPETITIONS > 1:
        print('repetition \t{0} of {1}'.format(rep+1, expset.N_REPETITIONS))
        
    print('episode \t{0} of {1}'.format(epi+1, expset.N_EPISODES))
    print("-"*50)
    

""" Show episode summary """
def epi_summary(caption, rep, epi, total_steps, all_r_epi, ave_r_epi, act_step_time):
    print("-"*22, "END", "-"*23)
    print(caption)
    
    if expset.N_REPETITIONS > 1:
        print('repetition \t{0} of {1}'.format(rep+1, expset.N_REPETITIONS))
        
    print('episode \t{0} of {1}'.format(epi+1, expset.N_EPISODES))
    print("total steps: " + str(total_steps))
    print("total rewards: " + str(all_r_epi))
    print("average reward: " + str(ave_r_epi))
    print("actual step time: " + str(act_step_time))
    print("-"*50)
    