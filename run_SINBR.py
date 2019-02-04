# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:23:37 2018

@author: shb
"""

from SINBR import SINBR_main
import numpy as np
from warnings import warn

def run_SINBR(N_train,X_train,ind_cond,output_file,lr_N=0.001,lr_V=0.000001,size_NN=[240,300,240],tilde_M=50,M=50,num_post_samp=1000,opt_iters=6000):
    #N_train: A numpy array containing the counts, rows corresponding to genes and columns corresponding to samples.
    #X_train: A numpy array containing the design matrix, columns corresponding to samples.
    #ind_cond: A list containing the index (indices) corresponding to the condition(s) of interest.
    #output_file: The name for the file to save the results. Saves in json and mat format.
    #lr_N: The base learning rate for mixing distribution parameters (NN parameters, \phi).
    #lr_V: The base learning rate for variational parameters (\varepsilon).
    #size_NN: The list containing the hidden layer sizes for the NN used in the mixing distribution.
    #num_post_samp: Number of samples to generate and use for calculating posterior means and KL divergence.
    #opt_iters: Number of optimization iterations.
    
    N_train = N_train.astype(np.float32)
    X_train = X_train.astype(np.float32)   
    
    exit_code=SINBR_main(N_train,X_train,ind_cond,lr_N,lr_V,size_NN,tilde_M,M,num_post_samp,output_file,opt_iters)

    if exit_code==1:
        warn('The learning rate is too high. Trying a lower learning rate...')
        lr_N = min(lr_N,0.001)
        lr_V = min(lr_V*0.1,0.0000001)
        opt_iters = max(6000,opt_iters)
        exit_code=SINBR_main(N_train,X_train,ind_cond,lr_N,lr_V,size_NN,tilde_M,M,num_post_samp,output_file,opt_iters)
        if exit_code == 0:
            print('Done')
        elif exit_code != 0:
            raise Exception('Not successful.')

