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
x=np.loadtxt('logistic_x.txt')
x=np.concatenate((np.ones(x.shape[0])[:, np.newaxis], x), axis=1)
y=np.loadtxt('logistic_y.txt')
theta_in=np.zeros(x.shape[1])

def sigmoid(yi,xi, theta): # sigmoid function                                                 
    z = yi*np.dot(t(theta),xi).astype("float_")     
    return 1.0 / (1.0 + np.exp(-z))    

def grad_cost(y,x, theta): ## gradient of cost function
    H=0
    for i in range(len(y)):
        H+=y[i]*x[i]\
                 *(1-sigmoid(y[i],x[i,],theta))                                                   
    H/=-len(y)                                       
    return H                       

def hessian(x, y, theta): #constructing hessian matrix
    H=0
    for i in range(len(y)):
        H+=np.exp(-y[i]*np.dot(t(theta),x[i,]))*y[i]**2\
                 *np.outer(x[i],t(x[i]))*sigmoid(y[i],x[i,],theta)**2                                                      
    H/=len(y)                                       
    return H

def newtons_method(x, y, theta, max_iterations=1000, delta = 1e-7, ):                                                                                                                               
    deltal = np.Infinity                                                                
    i = 0                                                                           
    while abs(deltal) > delta and i < max_iterations:                                       
        i += 1
        ##update rule
        theta_new=theta-np.dot(np.linalg.inv(hessian(x, y, theta)),\
                            grad_cost(y,x,theta))                                                                
        deltal = np.linalg.norm(theta_new-theta)                                                           
        theta = theta_new                                                                
    return theta

theta=newtons_method(x, y, theta_in)

res=100#resolution of line to plot
x1=np.linspace(0,8,res)
x2=(-theta[0]-x1*theta[1])/theta[2]#boundary line for p = 0.5
marker=['x' if l>0 else 'o' for l in y]
fig, ax = plt.subplots(1,1,figsize=(3,3))
for _x1,_x2,_m in zip(x[:,1],x[:,2],marker):
    ax.scatter(_x1,_x2, marker=_m,c='k')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.plot(x1,x2,'k--')
ax.set_title('Logistic Regression classifier')
