# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:23:37 2018

@author: shb
"""

import tensorflow as tf


regul_=0.0

def lognormal(z,mu,sigma):
    pdf = 1/(sigma*z)*tf.exp(-0.5*tf.square(tf.log(z)-mu)/tf.square(sigma))
    return pdf


def log_lognormal(z,mu,sigma):
    pdf = (-tf.log(sigma+regul_)-tf.log(z+regul_))+(-0.5*tf.square(tf.log(z+ regul_)-mu)/tf.square(sigma)) 
    return pdf


def sample_ln(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z=tf.exp(mu+eps*sigma)
    return z

def logitnormal(z,mu,sigma):
    logit = tf.log(z/(1-z))
    term1 = 1/(z*(1-z))
    term2 = 1/(sigma)*tf.exp(-0.5*tf.square(logit-mu)/tf.square(sigma))
    pdf = term1*term2
    return pdf
    
def sample_logitn(mu,sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+eps*sigma
    return tf.exp(z)/(1+tf.exp(z))

def sample_n(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+tf.matmul(eps,sigma)   
    return z

def sample_n_e(mu,sigma):

    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu+tf.einsum('kpv,pmv->kpm',eps,sigma)   
    return z
