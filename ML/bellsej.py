### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###
import os
import sounddevice as sd
import numpy as np
from tabulate import tabulate
os.chdir('D:/Krishna/Acads/Q4/ML/HW/')
Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)
    
def g(x): #sigmoid function
    return 1/(1+np.exp(-x)) 

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:
        for i in range(M):  
            vector=1-2*g(np.inner(W,X[i,:]))
            W+=alpha*(np.outer(vector,X[i,:])+np.linalg.inv(W.transpose()))
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S=np.inner(X,W)
    ##################################
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])
    print(tabulate(W, tablefmt="latex", floatfmt=".2f"))

if __name__ == '__main__':
    main()
