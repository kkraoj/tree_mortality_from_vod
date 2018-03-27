# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 12:55:35 2018

@author: kkrao
Objectives: 
    1. make vod world numpy array
    2. make latlon array
"""
import os
import sys
import arcpy
import pandas as pd
import numpy as np
import seaborn as sns
from dirs import MyDir, Dir_mort,Dir_CA,supply_lat_lon, remove_vod_affected, RWC,\
    subset_forest_cov
from scipy.interpolate import griddata

os.chdir(MyDir+'/LPDR_v2/VOD_matrix') # location of VOD matrix
store=pd.HDFStore('VOD_LPDR_v2.h5') #this is your store or marketplace of vod files for all dates

param='VOD'
pass_type='A'

### make lats and lons numpy arrays
year=2011 #any year
date=343 # any day
filename='%s_%s_%s_%03d'%(param,pass_type,year,date)
Df=store[filename]


latcorners=np.array([32,43])
loncorners=np.array([-125,-116])
my_lat,my_lon=supply_lat_lon('GC') # all 370 grids  
gridID=range(370)                          
master_Df=pd.DataFrame(columns=gridID)
master_Df.index.name = 'vod_pm_lpdr_v2'                                                 
                            
year_range=range(2002, 2018)
date_range=range(367)
all_files=store.keys()
#for year in year_range:
#    for date in date_range:
#        filename='/%s_%s_%s_%03d'%(param,pass_type,year,date)
#        if filename in all_files:
#            sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
#            sys.stdout.flush() 
#            
#            Df=store[filename]
#            Df=Df.loc[(Df.index>=latcorners[0]) & (Df.index<=latcorners[1]), \
#                   (Df.columns>=loncorners[0]) & (Df.columns<=loncorners[1])]     
#            std_lat, std_lon = np.meshgrid(Df.index,Df.columns,indexing='ij')
#            data = griddata((std_lat.flatten(),std_lon.flatten()),Df.values.flatten(),\
#                         (my_lat,my_lon), method='linear')
#            data=pd.DataFrame([data],index=[pd.to_datetime('%s%s'%(year,date),format='%Y%j')],columns=gridID)
#            master_Df=master_Df.append(data)

master_Df=remove_vod_affected(master_Df)
os.chdir(Dir_CA)
new_store=pd.HDFStore('data_subset_GC.h5')
master_Df=subset_forest_cov(master_Df)
new_store['vod_v2']=master_Df
rwc=RWC(master_Df)
new_store['RWC_v2']=rwc
