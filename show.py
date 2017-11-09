# -*- coding: utf-8 -*-
"""
Message Display

Created on Wed Nov  8 10:25:25 2017

@author: jesse
"""

import expset

def process_count(caption, rep, epi):
    print("-"*50)
    print(caption)
    
    if expset.N_REPETITIONS > 1:
        print('repetition \t{0} of {1}'.format(rep+1, expset.N_REPETITIONS))
        
    print('episode \t{0} of {1}'.format(epi+1, expset.N_EPISODES))
    print("-"*50)