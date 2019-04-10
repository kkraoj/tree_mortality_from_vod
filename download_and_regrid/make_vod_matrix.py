
"""
Created on Sun Apr 30 00:35:32 2017

files takes binary files from ntsg site as input and delivers matrix of vod as output
input files must be in EASE-GRID projection

@author: kkrao
"""
from __future__ import division

import os
import sys
import time
import numpy as np
import pandas as pd
from dirs import MyDir, Dir_fig, Dir_CA, Dir_mort
from scipy.interpolate import griddata
from mkgrid_global import mkgrid_global        

start_time = time.clock()
year_range=range(2002,2018)
date_range=range(1,367,1)

pass_type = 'D';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter

fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
y,x =np.asarray(late)*1e-5, np.asarray(lone)*1e-5
y,x =y.flatten(), x.flatten()
grid_y, grid_x = np.mgrid[90:-90:-0.25, -180:180:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_x),np.min(grid_y),np.max(grid_x),np.max(grid_y)]    
os.chdir(MyDir+'/LPDR_v2/VOD_matrix') 
store=pd.HDFStore('VOD_LPDR_v2.h5')
for year in year_range:
    for date in date_range:               
        fname=MyDir+'/LPDR_v2/%s/AMSRU_Mland_%s%03d%s.%s'%(year,year,date,pass_type,param)
        if  os.path.isfile(fname):
            sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
            sys.stdout.flush()   
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()             
            data[data<0.0] = np.nan
            data = -np.log(data)            
            datagrid = mkgrid_global(data)           
            z=np.asarray(datagrid)
            z=z.flatten()                                
            grid_z = griddata((x,y), z, (grid_x, grid_y), method='linear')
            grid_z = pd.DataFrame(grid_z, index=grid_y[:,0],columns=grid_x[0,:])
            grid_z.index.name = 'Lat'
            grid_z.columns.name = 'Lon'
            grid_z.name = 'VOD_%s_%s_%03d'%(pass_type,year,date)
            store[grid_z.name]=grid_z
store.close()

## check work
#plt.scatter(grid_x,grid_y, c=grid_z,s=3)
end_time=time.clock()
time_taken=round((end_time-start_time)/60/60, 1)
print('Program executed in %s hours'%time_taken)
           