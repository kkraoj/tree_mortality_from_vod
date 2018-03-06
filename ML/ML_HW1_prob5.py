# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:57:22 2017

@author: kkrao
"""
import os 
import numpy as np
from numpy import transpose as t, dot as dot
import matplotlib.pyplot as plt

os.chdir('D:/Krishna/Acads/Q4/ML/HW')
train=np.loadtxt('quasar_train.csv')
x=np.concatenate((np.ones(x.shape[0])[:, np.newaxis], x), axis=1)
test=np.loadtxt('quasar_test.csv')
theta_in=np.zeros(x.shape[1])
