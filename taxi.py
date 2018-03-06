# -*- coding: utf-8 -*-
"""
Created on Mon May 15 04:25:43 2017

@author: kkrao
Figure opening in new window must be enabled. 
Go to tools->preferences->iPython->graphics->
select backend = Automatic for Windows or QT for UNIX
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
from matplotlib import pyplot as plt
import random
import matplotlib.ticker as plticker
from random import randint
import time 

# variables of model. Feel free to change them!
num_cars=4 # number of cars in city
size=10 # size of city in units
pause=0.05  # rate of refreshing frames in seconds

# initialize
y,x = np.mgrid[size:-size-1:-1, -size:size+1:1]
sim_time=500
time_now=np.arange(sim_time)
plt.close()
random.seed(10)
fs=15
start=0
dest=0
##### define required functions
def dist(start,car):
    d=[]
    for i in range(len(car)):
        c=car[i]
        d.append(abs(c[0]-start[0])+abs(c[1]-start[1]))
    d=np.array(d)
    return d  

def dist1(start,my_car):
    d=[]
    c=my_car
    d.append(abs(c[0]-start[0])+abs(c[1]-start[1]))
    d=np.array(d)
    return d  

def move(car,size):
    from random import randint
    for i in range(len(car)):
        c=car[i]
        j=randint(0,1)
        if c[j]==size:
            c[j]=c[j]-1
        elif c[j]==-size:
            c[j]=c[j]+1
        else:               
            r=randint(0,1)
            if r>0.5:
                c[j]=c[j]+1
            else:
                c[j]=c[j]-1
        car[i]=c
    return car
def move_closer(point,car):
    if point[0]!=car[0]:
        car[0]=car[0]+(point[0]-car[0])/abs(point[0]-car[0])
    else:
        car[1]=car[1]+(point[1]-car[1])/abs(point[1]-car[1])
    return car


def onclick(event):
    click=np.array([event.xdata, event.ydata])
    click=click.astype(int)
    global start
    start=click
def onclick2(event):
    click2=np.array([event.xdata, event.ydata])
    click2=click2.astype(int)
    global dest
    dest=click2
    

def plot(car,pause):
    fig.clear()
    ax=fig.gca()
    plt.scatter(*zip(*car),marker='o',c='b',s=50,label='Available cars')
    ax.set_xlim([-size,size])
    ax.set_ylim([-size,size])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Ride Hailing Simulation',fontsize=1.2*fs)
    intervals = 1
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='major', axis='both', linestyle='-')
    plt.text(0,-size-2,'Choose your pickup point with mouse',fontsize=fs,ha='center')
    plt.draw()
    plt.legend(loc='upper right')
    plt.pause(pause)  
    
def plot1(start,my_car,car,pause,dec):
    fig.clear()
    ax=fig.gca()
    plt.scatter(*zip(*car),marker='o',c='b',s=50,label="Available cars")
    plt.scatter(my_car[0],my_car[1],marker='o',c='y',s=70,label='Your car')
    if dec==0:
        string='Pickup point'
    else:
        string='Drop point'
    plt.scatter(start[0],start[1],marker='s',c='y',s=70,label=string)
    ax.set_xlim([-size,size])
    ax.set_ylim([-size,size])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Ride Hailing Simulation',fontsize=fs)
    intervals = 1
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='major', axis='both', linestyle='-')
    if dec==1:
        plt.text(0,-size-2,'Choose your destination with mouse',fontsize=fs,ha='center')
    elif dec==0:
        plt.text(0,-size-2,'I will be right there',fontsize=fs,ha='center')        
    elif dec==2:
        plt.text(0,-size-2,'Thank you for riding with us. Have a nice day!',fontsize=fs,ha='center')
    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(pause)  
############################################################################
# Begin Code
## Place cars randomly
car=[]
for i in range(num_cars):    
    car.append([randint(-size,size),randint(-size,size)])                     
            
fig=plt.figure()

plt.ion()
#start simluation
for t in time_now:
    plot(car,pause)                 
    move(car,size)
    fig.canvas.mpl_connect('button_press_event', onclick)
    if type(start) is np.ndarray: # user identified
        break
d=dist(start,car)
c=np.argmin(d)
my_car=np.array(car[c])
del car[c]
## go to pick up passenger
while dist1(start,my_car)>0:
    move_closer(start,my_car)
    move(car,size)
    plot1(start,my_car,car,pause,0)
## wait for user to decide destination
for t in time_now:
    plot1(start,my_car,car,pause,1)                
    move(car,size)
    fig.canvas.mpl_connect('button_press_event', onclick2)
    if type(dest) is np.ndarray: ## destination identified
        break
## Go to destination
while dist1(dest,my_car)>0:
    move_closer(dest,my_car)
    move(car,size)
    plot1(dest,my_car,car,pause,3)          
plot1(dest,my_car,car,pause,2)        