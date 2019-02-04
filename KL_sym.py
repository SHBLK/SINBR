# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:23:37 2018

@author: shb
"""

import numpy as np

def KL_sym(p_1,p_2):
    #This is a function to calculate KL distance based on samples
    #Input is samples*genes
    ct=1.5
    regu=1e-10
    nbins=100
    p_all=np.concatenate((p_1,p_2),axis=0)
    quant_1 = np.percentile(p_all,25,axis=0,keepdims=True)
    quant_3 = np.percentile(p_all,75,axis=0,keepdims=True)
    IQD = quant_3 - quant_1
    x_min = np.maximum(quant_1 - ct*IQD,0)
    x_max = quant_3 + ct*IQD
    it=p_1.shape[1]
    out_ = np.zeros((it,1))
    for i in range(it):
        idd=np.where(np.logical_and(p_1[:,i]>=x_min[0,i],p_1[:,i]<=x_max[0,i]))
        f_1=np.histogram(p_1[idd,i],bins=np.linspace(x_min[0,i],x_max[0,i],num=nbins))[0]/float(len(idd)) + regu
        idd=np.where(np.logical_and(p_2[:,i]>=x_min[0,i],p_2[:,i]<=x_max[0,i]))
        f_2=np.histogram(p_2[idd,i],bins=np.linspace(x_min[0,i],x_max[0,i],num=nbins))[0]/float(len(idd)) + regu
        out_[i,0]=np.sum((f_1-f_2)*np.log(f_1/f_2))
    return out_
    
    