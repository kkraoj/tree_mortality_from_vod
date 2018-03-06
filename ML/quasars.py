from __future__ import division
import os
import sys
import numpy as np
from numpy import transpose as t, dot as dot
from numpy.linalg import inv as inv
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_data():
    os.chdir('D:/Krishna/Acads/Q4/ML/HW')
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X_):
    X = None
    X=t(np.vstack((np.ones(X_.shape[0]), X_)))
    return X

def smooth_data(raw, wavelengths, tau):
    smooth = np.copy(raw)
    for i in range(raw.shape[0]):
        sys.stdout.write('\r'+'Smoothing data for row %d...'%i)
        sys.stdout.flush()
        smooth[i,:]=LWR_smooth(raw[i,:], wavelengths, tau) #apply LWR smooth t0 all rows
    return smooth

def LWR_smooth(Y, X, tau):#locally weighted regression
    Y_hat = np.zeros(Y.shape) #initialize
    for i in range(len(Y)):
        W=np.diag(np.exp(-(X[i]-X)**2/2/tau**2)) ## construct weight matrix
        Y_hat[i]=X[i]*reduce(dot, [t(X),W,X])**(-1)*reduce(dot, [t(X),W,Y])
    ##reduce is for triple dot product
    ##return prediction
    return Y_hat 

def LR_smooth(Y, X_):
    X = add_intercept(X_)
    yhat = np.zeros(Y.shape)
    theta = np.zeros(2)
    theta=dot(np.linalg.inv(dot(t(X),X)),dot(t(X),Y))
    yhat=dot(X,theta)[:,np.newaxis]
    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(X,raw_Y,alpha=0.5,color='k',edgecolor="None")
    i=0
    for Ys_,desc_ in zip(Ys,desc):
        i+=1
        ax.plot(X,Ys_,lw=1.5,label=desc_)
    ax.set_xlabel('Wavelength $(\AA)$')
    ax.set_ylabel('Flux measurements')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_c(Yhat, Y, X, filename):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(X,Y,'k--',lw=1.5,label='smoothed')
    ax.plot(X[:len(Yhat)],Yhat,'r-',lw=1.5,label='fitted')
    ax.set_xlabel('Wavelength $(\AA)$')
    ax.set_ylabel('Flux measurements')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def split(full, wavelengths):
    ind_left=np.where(wavelengths==1200)[0][0]## splitting below 1200
    ind_right=np.where(wavelengths==1300)[0][0] ## split above 1300
    left,right=full[:,:ind_left],full[:,ind_right:]
    return left, right
#
def dist(a, b,axis=None):
    if len(b.shape)*len(a.shape)>1: # check if input arrays are 1d or 2d
        axis=1 
    distance=np.sum((a-b)**2,axis=axis)
    return distance
    
def ker(t):
    return max(1-t,0)

def func_reg(left_train, right_train, f_right):
    m, n = left_train.shape
    lefthat = np.zeros(n)
    dist_j=dist(right_train,f_right) # distance with all other spectra
    neigh=dist_j.argsort()[:3] # index of least 3 distances
    h=max(dist_j) # max distance
    nr,dr=0,0
    for i in neigh:
        d=dist(right_train[i,:],f_right)
        nr+=ker(d/h)*left_train[i,:] ##numerator
        dr+=ker(d/h) ##denominator
    lefthat=nr/dr
    return lefthat

def main():
    raw_train, raw_test, wavelengths = load_data()

    ## Part b.i
    lr_est, theta = LR_smooth(raw_train[0], wavelengths)
    print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
    plot_b(wavelengths, raw_train[0], [lr_est], ['Regression line'], 'ps1q5b1.png')

#    ## Part b.ii
    lwr_est_5 = LWR_smooth(raw_train[0], wavelengths, 5)
    plot_b(wavelengths, raw_train[0], [lwr_est_5], ['tau = 5'], 'ps1q5b2.png')

    ## Part b.iii
    lwr_est_1 = LWR_smooth(raw_train[0], wavelengths, 1)
    lwr_est_10 = LWR_smooth(raw_train[0], wavelengths, 10)
    lwr_est_100 = LWR_smooth(raw_train[0], wavelengths, 100)
    lwr_est_1000 = LWR_smooth(raw_train[0], wavelengths, 1000)
    plot_b(wavelengths, raw_train[0],
             [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
             ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
             'ps1q5b3.png')

    ### Part c.i
    smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5) for raw in [raw_train, raw_test]]

    #### Part c.ii
    left_train, right_train = split(smooth_train, wavelengths)
    left_test, right_test = split(smooth_test, wavelengths)
#
    train_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_train, right_train)]
    print('\n Part c.ii) Training error: %.4f' % np.mean(train_errors))

    ### Part c.iii
    test_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_test, right_test)]
    print('\n Part c.iii) Test error: %.4f' % np.mean(test_errors))

    left_1 = func_reg(left_train, right_train, right_test[0])
    plot_c(left_1, smooth_test[0], wavelengths, 'ps1q5c3_1.png')
    left_6 = func_reg(left_train, right_train, right_test[5])
    plot_c(left_6, smooth_test[5], wavelengths, 'ps1q5c3_6.png')
    pass

if __name__ == '__main__':
    main()
