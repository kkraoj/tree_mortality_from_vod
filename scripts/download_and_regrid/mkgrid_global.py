# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:21:30 2017

@author: Krishna Rao
"""
'''
function [gridout, EASE_r, EASE_s] = 
%MKGRID_GLOBAL(x)  Creates a 586x1383 global EASE grid matrix for mapping   
% gridout = mkgrid_global(x) uses the 2090887 element array (x) and returns
% a 586x1383 global EASE grid matrix.

% The user should change the paths to the EASE grid row and column data (below)
% to their specific system locations.

 Lucas A. Jones <lucas@ntsg.umt.edu>
'''
import numpy as np
import matplotlib.pyplot as plt


def mkgrid_global(x):
    #Load ancillary EASE grid row and column data, where <MyDir> is the path to 
    #wherever the globland_r and globland_c files are located on your machine.
    MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
    fid = open(MyDir+'/'+'anci/globland_r','rb');
    EASE_r = np.fromfile(fid,dtype=np.int16)
    fid.close()
    
    fid = open(MyDir+'/'+'anci/globland_c','rb');
    EASE_s = np.fromfile(fid,dtype=np.int16)
    fid.close()
#    plt.latlon = True
    #Initialize the global EASE grid 
    gridout = np.empty([586,1383]);
    gridout[:]=np.NaN                  
    
    #Loop through the elment array
    for i in list(range(209091)):
        '''  
        %Distribute each element to the appropriate location in the output
        %matrix (EASE grid base address is referenced to (0,0), but MATLAB is
        %(1,1)
        '''
        gridout[EASE_r[i],EASE_s[i]] = x[i];
               
    return(gridout)
              
