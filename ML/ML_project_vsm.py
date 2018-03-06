# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:55:55 2017

@author: kkrao
"""

from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dirs import MyDir
from mkgrid_global import mkgrid_global  
from scipy.interpolate import griddata


year_range=range(2003,2016)
date_range=range(1,366,1)
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'vsm';           #Type the parameter
fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone]
latcorners=[331, 387]
loncorners=[1097, 1200]
x,y=np.asarray(lone),np.asarray(late)
x,y=x.flatten(),y.flatten()
new_lat = np.load('fire_lat.npy')
new_lon = np.load('fire_lon.npy')
new_x, new_y = np.meshgrid(new_lon, new_lat)
Df=pd.DataFrame()
for year in year_range:
    for date in date_range:      
        sys.stdout.write('\r'+'Processing data for %s %03d...'%(year,date))
        sys.stdout.flush()
        fname=MyDir+'/'+param+'/'+str(year)+'/AMSRU_Mland_'+str(year)+'%03d'%date+pass_type+'.'+param
        if  os.path.isfile(fname):
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()     
            data[data<0]=np.nan                    
#            data = -np.log(data)            
            z = mkgrid_global(data)                                           
            z=np.asarray(z)
            z=z.flatten() 
            new_z = griddata((y,x), z, (new_y, new_x), method='linear')
            time=pd.to_datetime(str(year)+'%03d'%date,format='%Y%j')
            index=[(t,l) for (t,l) in zip(np.repeat(time,len(new_lat)),new_lat)]
            index=pd.MultiIndex.from_tuples(index,names=('Time','lat'))
            new_z = pd.DataFrame(new_z,index=index,columns=new_lon)
            new_z=new_z.iloc[latcorners[0]:latcorners[1],loncorners[0]:loncorners[1]]
            Df=Df.append(new_z)
Df.columns.name='lon'
store=pd.HDFStore(MyDir+'/ML_Project_Data.h5')
store['vsm']=Df
store.close()

store=pd.HDFStore(MyDir+'/VOD_VSM_Data.h5')
Df=store['vsm']
store.close()


level_values = Df.index.get_level_values
gg=Df.groupby([pd.Grouper(freq='M', level=0)]+[level_values(i) for i in [1]]).mean()


store=pd.HDFStore(MyDir+'/ML_Project_Data.h5')
store['vsm']=gg
store.close()
