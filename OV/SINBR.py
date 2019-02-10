# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:23:37 2018

@author: shb
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os


from scipy.io import loadmat,savemat
import tensorflow as tf
from tensorflow.contrib.distributions import fill_triangular

import json
from timeit import default_timer as timer
from KL_sym import KL_sym

from distribution_u import log_lognormal, sample_ln, sample_n






def SINBR_main(N_train,X_train,ind_cond,lr_N,lr_V,size_NN,K,J_val,num_post_samp,file_name,opt_iters):
    
    slim=tf.contrib.slim

    def sample_hyper(noise_dim,K,z_dim,reuse=False): 
        with tf.variable_scope("hyper_q") as scope:
            if reuse:
                scope.reuse_variables()
            e2 = tf.random_normal(shape=[K,noise_dim])
        
        
            h2 = slim.stack(e2,slim.fully_connected,size_NN)

            mu = tf.reshape(slim.fully_connected(h2,z_dim,activation_fn=None,scope='implicit_hyper_mu'),[-1,z_dim])
        return mu

    ind_all_zeros = np.where(np.sum(N_train,axis=1)==0)[0]
    ind_non_zeros = np.where(np.sum(N_train,axis=1)!=0)[0]
    N_train = N_train[np.sum(N_train,axis=1)!=0,:]
    regul_=0.0




    tf.reset_default_graph();

    P,N = np.shape(N_train) #P corresponds to K, N corresponds to J
    V,N = np.shape(X_train)

    a0 = b0 = c0 = d0 = g0 = 0.01 #fixed gamma prior parameters


    noise_dim = P*V + N + V +  1
    psi_dim = P*V + N + V +  1
    psi_dim_beta = P*V
    psi_dim_r = N
    psi_dim_alph = V
    psi_dim_h = 1

    J = tf.placeholder(tf.int32, shape=())


    fff = tf.get_variable("KK", shape=[P,V], dtype=tf.float32, initializer=tf.initializers.truncated_normal(mean=0.2,stddev=0.1,seed=12)) # For diagonal covariance
    chol_cov = tf.matrix_diag(fff)#chol_cov is P * V * V For diagonal covariance
    covariance = tf.matmul(chol_cov, tf.matrix_transpose(chol_cov))



    inv_cov = tf.matrix_inverse(covariance) # P * V * V
    inv_cov_1 = tf.expand_dims(inv_cov,axis=0)
    inv_cov_2 = tf.tile(inv_cov_1,[J+1,1,1,1]) # J+1*P*V*V

    log_cov_det = tf.reshape(tf.log(tf.matrix_determinant(covariance)+regul_),shape=[P,1]) # P

    sigh_ = tf.get_variable("sig_h",shape=[1], dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-5.0,stddev=0.1,seed=12))
    sigh = tf.exp(sigh_)

    sigr_ = tf.get_variable("sig_r",shape=[1,N],dtype=tf.float32,initializer=tf.initializers.random_normal(mean=0.40,stddev=0.1,seed=12))
    sigr = tf.exp(sigr_)

    sigalph_ = tf.get_variable("sig_alph",shape=[1,V],dtype=tf.float32,initializer=tf.initializers.random_normal(mean=-6.9,stddev=0.1,seed=12))
    sigalph = tf.exp(sigalph_)
    
      
    scale = tf.placeholder(tf.float32, shape=())

    n_dta = tf.placeholder(tf.float32,[P,N],name='data_n')
    x_dta = tf.placeholder(tf.float32,[V,N],name='data_x')

    psi_sample = tf.squeeze(sample_hyper(noise_dim,K,psi_dim))
    mu_beta = tf.transpose(tf.reshape(tf.slice(psi_sample,[0,0],[-1,psi_dim_beta]),shape=[K,P,V]),[1,0,2]) #P*K*V  
    mu_r = tf.slice(psi_sample,[0,psi_dim_beta],[-1,psi_dim_r])
    mu_alph = tf.slice(psi_sample,[0,psi_dim_beta+psi_dim_r],[-1,psi_dim_alph])
    mu_h = tf.slice(psi_sample,[0,psi_dim_beta+psi_dim_r+psi_dim_alph],[-1,psi_dim_h])


    
    sigr_11=tf.tile(sigr,[K,1])
    sigalph_11=tf.tile(sigalph,[K,1])

    beta_sample=sample_n(mu_beta,tf.transpose(chol_cov,perm=[0,2,1]))#P*K*V
    r_sample = sample_ln(mu_r,sigr_11) #K*N
    alph_sample = sample_ln(mu_alph,sigalph_11) #K*V
    h_sample = sample_ln(mu_h,sigh)# K*1

    psi_star_0 = sample_hyper(noise_dim,J,psi_dim,reuse=True)
    mu_beta_0 = tf.reshape(tf.slice(psi_star_0,[0,0],[-1,psi_dim_beta]),shape=[J,P,V])   
    mu_r_0 = tf.slice(psi_star_0,[0,psi_dim_beta],[-1,psi_dim_r])
    mu_alph_0 = tf.slice(psi_star_0,[0,psi_dim_beta+psi_dim_r],[-1,psi_dim_alph])
    mu_h_0 = tf.slice(psi_star_0,[0,psi_dim_beta+psi_dim_r+psi_dim_alph],[-1,psi_dim_h])


    mu_beta_1 = tf.expand_dims(mu_beta_0,axis=2) 
    mu_beta_2 = tf.tile(mu_beta_1,[1,1,K,1])
    mu_r_1 = tf.expand_dims(mu_r_0,axis=1) 
    mu_r_2 = tf.tile(mu_r_1,[1,K,1])
    mu_alph_1 = tf.expand_dims(mu_alph_0,axis=1) 
    mu_alph_2 = tf.tile(mu_alph_1,[1,K,1])
    mu_h_1 = tf.expand_dims(mu_h_0,axis=1) 
    mu_h_2 = tf.tile(mu_h_1,[1,K,1])

    merge = tf.placeholder(tf.int32, shape=[])
    mu_beta_star = tf.cond(merge>0,lambda:tf.concat([mu_beta_2, tf.expand_dims(mu_beta,axis=0)],0),lambda:mu_beta_2) #J+1*P*K*V 
    mu_r_star = tf.cond(merge>0,lambda:tf.concat([mu_r_2, tf.expand_dims(mu_r,axis=0)],0),lambda:mu_r_2) #J+1*K*N
    mu_alph_star = tf.cond(merge>0,lambda:tf.concat([mu_alph_2, tf.expand_dims(mu_alph,axis=0)],0),lambda:mu_alph_2) #J+1*K*V
    mu_h_star = tf.cond(merge>0,lambda:tf.concat([mu_h_2, tf.expand_dims(mu_h,axis=0)],0),lambda:mu_h_2) #J+1*K*1


    beta_sample_0 = tf.expand_dims(beta_sample,axis=0)#P*K*V
    beta_sample_1 = tf.cond(merge>0,lambda:tf.tile(beta_sample_0,[J+1,1,1,1]),lambda:tf.tile(beta_sample_0,[J,1,1,1])) #J+1*P*K*V
    r_sample_0 = tf.expand_dims(r_sample,axis=0)
    r_sample_1 = tf.cond(merge>0,lambda:tf.tile(r_sample_0,[J+1,1,1]),lambda:tf.tile(r_sample_0,[J,1,1])) #J+1*K*N
    alph_sample_0 = tf.expand_dims(alph_sample,axis=0)
    alph_sample_1 = tf.cond(merge>0,lambda:tf.tile(alph_sample_0,[J+1,1,1]),lambda:tf.tile(alph_sample_0,[J,1,1])) #J+1*K*V
    h_sample_0 = tf.expand_dims(h_sample,axis=0)
    h_sample_1 = tf.cond(merge>0,lambda:tf.tile(h_sample_0,[J+1,1,1]),lambda:tf.tile(h_sample_0,[J,1,1])) #J+1*K*1

    sigalph_0=tf.expand_dims(sigalph_11,axis=0)#1*K*V
    sigalph_1=tf.cond(merge>0,lambda:tf.tile(sigalph_0,[J+1,1,1]),lambda:tf.tile(sigalph_0,[J,1,1]))
    sigr_0=tf.expand_dims(sigr_11,axis=0)#1*K*N
    sigr_1=tf.cond(merge>0,lambda:tf.tile(sigr_0,[J+1,1,1]),lambda:tf.tile(sigr_0,[J,1,1]))

    xvx = tf.matmul(beta_sample_1-mu_beta_star,inv_cov_2)*(beta_sample_1-mu_beta_star)
    ker = tf.reduce_sum(-0.5*tf.reduce_sum(xvx,3),axis=1) 


    term2 = tf.reduce_sum(log_lognormal(r_sample_1,mu_r_star,sigr_1),axis=2)
    term3 = tf.reduce_sum(log_lognormal(alph_sample_1,mu_alph_star,sigalph_1),axis=2)
    term4 = tf.reduce_sum(log_lognormal(h_sample_1,mu_h_star,sigh),axis=2)

    log_H = tf.transpose(tf.reduce_logsumexp(ker+term2+term3+term4,axis=0,keepdims=True))-\
        tf.log(tf.cast(J,tf.float32)+1.0)-\
        0.5*tf.reduce_sum(log_cov_det,keepdims=True) 


    r_sample_P = tf.expand_dims(r_sample,axis=1)#K*1*N
    N_train_P = tf.expand_dims(n_dta,axis=0)#1*P*N
    X_train_P_ = tf.expand_dims(x_dta,axis=0)#1*V*N
    X_train_P = tf.tile(X_train_P_,[K,1,1])
    beta_sample_P=tf.transpose(beta_sample,perm=[1,0,2])#K*P*V

    log_P = tf.reduce_sum( ((c0 - 1 + (P/2))*tf.log(alph_sample+regul_)) - ((d0 + (tf.reduce_sum(tf.square(beta_sample_P),axis=1)/2) )*alph_sample),axis=1 ,keepdims=True)+\
        ((a0*N + b0 - 1)*tf.log(h_sample+regul_)) - (g0*h_sample) - (tf.reduce_sum(r_sample,axis=1,keepdims=True)*h_sample) + (tf.reduce_sum(tf.log(r_sample+regul_),axis=1,keepdims=True)*(a0-1))+\
        tf.reduce_sum(tf.reduce_sum(tf.lgamma(tf.add(N_train_P,r_sample_P)),axis=2),axis=1,keepdims=True)-\
        P*tf.reduce_sum(tf.lgamma(r_sample),axis=1,keepdims=True)+\
        tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.matmul(beta_sample_P,X_train_P),N_train_P)-tf.multiply(tf.log(1+tf.exp(tf.matmul(beta_sample_P,X_train_P))+regul_),tf.add(N_train_P,r_sample_P)),axis=2),axis=1,keepdims=True)
        


    loss = tf.reduce_mean(scale*(log_H - log_P))



    nn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hyper_q')
    lr=tf.constant(lr_N)


    train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=nn_var)

    lr2=tf.constant(lr_V)

    train_op2 = tf.train.GradientDescentOptimizer(learning_rate=lr2).minimize(loss,var_list=[fff,sigr_,sigalph_,sigh_])

    init_op=tf.global_variables_initializer()

#%%

    sess=tf.InteractiveSession()

    start_time_train = timer()

    sess.run(init_op)

    record = []
    
    opt_iters_V=np.floor(opt_iters*0.9)
    opt_iters_red=np.floor(opt_iters*0.33)
    try:
      for i in range(opt_iters):
        _,cost=sess.run([train_op1,loss],{n_dta:N_train,x_dta:X_train,lr:lr_N*(0.7**(i/opt_iters_red)),J:J_val,merge:1,scale:min(1.0e-0,1.0)})
        if i<opt_iters_V:
            if (lr_V/lr_N) < 0.01:
                for dum_cnt in range(5):
                    _,cost_n,=sess.run([train_op2,loss],{n_dta:N_train,x_dta:X_train,lr2:lr_V*(0.9**(i/opt_iters_red)),J:J_val,merge:1,scale:min(1.0e-0,1.0)})
            else:
                _,cost_n,=sess.run([train_op2,loss],{n_dta:N_train,x_dta:X_train,lr2:lr_V*(0.9**(i/opt_iters_red)),J:J_val,merge:1,scale:min(1.0e-0,1.0)})
   
        record.append(cost)
        if i%1000 == 0:
            print("iter:", '%04d' % (i+1), "cost=", np.mean(record),',', np.std(record),"cost_N=",cost,"cost_V=",cost_n)
            record = []
    except tf.errors.InvalidArgumentError:
        sess.close()
        end_time_train = timer()
        return 1

    end_time_train = timer()

    #Generate from posteriors
    start_time_test = timer()
    beta_hive=np.zeros([num_post_samp,P,V])
    r_hive=np.zeros([num_post_samp,N])
    alph_hive=np.zeros([num_post_samp,V])
    h_hive=np.zeros([num_post_samp,1])
    for i in range(num_post_samp):    
        beta_tmp,r_tmp,alph_tmp,h_tmp = sess.run([beta_sample,r_sample,alph_sample,h_sample],{n_dta:N_train,x_dta:X_train})
        beta_hive[i,:,:] = np.squeeze(beta_tmp[:,0,:])
        r_hive[i,:] = np.squeeze(r_tmp[0,:])
        alph_hive[i,:] = np.squeeze(alph_tmp[0,:])
        h_hive[i,:] = np.squeeze(h_tmp[0,:])
    end_time_test = timer()

    bb1=np.squeeze(beta_hive[:,:,ind_cond])
    bb0=np.squeeze(beta_hive[:,:,0])
    if len(ind_cond)>1:
        KL_sivi = KL_sym(np.exp(bb0),np.exp(bb0+sum(bb1,axis=2)))
    else:
        KL_sivi = KL_sym(np.exp(bb0),np.exp(bb0+bb1))
    KL_sivi_all = np.zeros((len(ind_all_zeros)+len(ind_non_zeros),1))
    KL_sivi_all[ind_non_zeros,0] = KL_sivi[:,0] 
    data_dic = {"beta_hive" : beta_hive.tolist(),"r_hive" : r_hive.tolist(), "h_hive": h_hive.tolist(), 
            "alph_hive": h_hive.tolist(),"train_time": end_time_train-start_time_train,"test_time": end_time_test-start_time_test, "KL_SIVI": KL_sivi_all.tolist(), "KL_SIVI_SUB": KL_sivi.tolist()}
    
    f_name_1 = file_name + '.mat'
    f_name_2 = file_name + '.json'
    savemat(f_name_1,data_dic)
    with open(f_name_2, 'w') as fp:
        json.dump(data_dic, fp)
    return 0
