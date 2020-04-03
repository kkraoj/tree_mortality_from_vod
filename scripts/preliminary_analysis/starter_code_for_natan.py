

# -*- coding: utf-8 -*-

"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from __future__ import division

import os
import sys
import time
import numpy as np
import pandas as pd
from mkgrid_global import mkgrid_global
from dirs import MyDir, Dir_fig, Dir_CA, Dir_mort
import matplotlib.pyplot as plt
 


start_time = time.clock()

year_range=range(2007,2008)
date_range=range(1,367)

pass_type = 'D';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter
map_factor = 1e4

fid = open(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\anci\MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\anci\MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
y=np.asarray(late)*1e-5 ###convert using scale factor
x=np.asarray(lone)*1e-5 ###convert using scale factor
grid_y, grid_x = np.mgrid[90:-90.25:-0.25, -180:180.25:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_x),np.min(grid_y),np.max(grid_x),np.max(grid_y)]    
os.chdir(Dir_fig) 
Df = np.empty([367, y.shape[0], y.shape[1]]);
Df[:]=np.NaN     
for year in year_range:
    for date in date_range:               
        fname=MyDir+'/%s/%s/AMSRU_Mland_%s%03d%s.%s'%(param,year,year,date,pass_type,param)
        if  os.path.isfile(fname):
            sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
            sys.stdout.flush()   
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()          
            data[data<0.0] = np.nan
            data = -np.log(data)  
            Df[date] = mkgrid_global(data) 
end_time=time.clock()
time_taken=end_time-start_time
print('Program executed in %s seconds'%time_taken)
np.save(r"D:\Krishna\Project\data\RS_data\tc10\vod_2007.npy", Df)
np.save(r"D:\Krishna\Project\data\RS_data\tc10\longitude.npy", x)           
np.save(r"D:\Krishna\Project\data\RS_data\tc10\latitude.npy", y)

datam = np.nan_to_num(data)

fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(x.flatten().tolist(), y.flatten().tolist(), c = datam.flatten().tolist(), cmap = "magma",s = 1)
