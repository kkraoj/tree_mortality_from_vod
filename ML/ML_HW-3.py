# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:03:11 2017

@author: kkrao
"""

import os
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
os.chdir('D:/Krishna/Acads/Q4/ML/HW/')

A = imread('mandrill-small.tiff')
#plt.imshow(A)R
K=16

def initialize_centroids(A, k=16):
    """returns k centroids from the initial A"""
    col=np.random.randint(0,np.shape(A)[0],k)
    row=np.random.randint(0,np.shape(A)[1],k)
    centroids=A[row,col,:]
    return centroids

def closest_centroid(A, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    closest=np.copy(A[:,:,0])
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            distances = np.linalg.norm(A[i,j].astype(int)-centroids.astype(int),axis=1)
            closest[i,j]=np.argmin(distances)
    return closest

def move_centroids(A, closest, centroids0):
    """returns the new centroids assigned from the A closest to them"""
    centroids=centroids0.copy()
    for k in range(np.shape(centroids)[0]):
        centroids[k]=np.mean(A[np.where(closest==k)],axis=0)
    return centroids

def K_means(A,maxiter=10,tol=1e-1):
    centroids1 = initialize_centroids(A)
    centroids0 = 10*centroids1.copy()
    niter=0
    while np.linalg.norm(centroids1-centroids0)>tol:
        centroids0=centroids1.copy()
        closest=closest_centroid(A, centroids0)
        centroids1=move_centroids(A, closest, centroids0)
        if niter >=maxiter:
            break
        niter+=1
    return centroids1
## part (b) 
centroids=K_means(A)
## part (c)
B = imread('mandrill-large.tiff')
def compress(B,centroids):
    closest=closest_centroid(B, centroids)
    comp=np.copy(B)
    for i in range(np.shape(B)[0]):
        for j in range(np.shape(B)[1]):
            comp[i,j]=centroids[closest[i,j]]
    return comp

comp=compress(B,centroids)
plt.imshow(comp)
plt.axis('off')
plt.tight_layout()
#plt.imshow(B)
#
##part(d)
