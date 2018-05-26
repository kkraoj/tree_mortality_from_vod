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
import pandas as pd
import numpy as np
import seaborn as sns
from dirs import MyDir # home directory

os.chdir(MyDir+'/LPDR_v2/VOD_matrix') # location of VOD matrix
store=pd.HDFStore('VOD_LPDR_v2.h5') #this is your store or marketplace of vod files for all dates

param='VOD'
pass_type='D'

### make lats and lons numpy arrays
year=2011 #any year
date=343 # any day
filename='%s_%s_%s_%03d'%(param,pass_type,year,date)
Df=store[filename]
lats, lons = np.meshgrid(Df.index,Df.columns,indexing='ij')


## make vod matrix in vod_world
year_range=range(2002, 2018)
date_range=range(367)
vod_world=np.full([len(year_range)*len(date_range),lats.shape[0],lats.shape[1]], np.nan)
all_files=store.keys()
counter=0
for year in year_range:
    for date in date_range:
        sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
        sys.stdout.flush() 
        filename='/%s_%s_%s_%03d'%(param,pass_type,year,date)
        if filename in all_files:
            vod_world[counter,:,:]=store[filename].values #vod file for year, date
        counter+=1

np.save('LPDRv2_D.npy', vod_world)
