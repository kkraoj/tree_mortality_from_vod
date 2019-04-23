# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""

import os
import numpy as np
import pandas as pd
from dirs import MyDir
import sys
year_range=range(2002,2016)
date_range=range(1,366,1)
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter
os.chdir("D:/Krishna/Project/data/Mort_Data/Misc_data")
filename = 'vod_world_2.h5'
store=pd.HDFStore(filename)
#data_source1='vod'
vod_world=pd.DataFrame(columns = range(209091))          
for year in year_range:
    for date in date_range:               
        fname=MyDir+'/'+param+'/'+str(year)+'/AMSRU_Mland_'+\
                                     str(year)+"%03d"%date+pass_type+'.'+param
        if  os.path.isfile(fname):
            sys.stdout.write('\r'+"Processing: \t Year: %s\t Date: %03d"%(year, date))
            sys.stdout.flush()
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()            
            data[data<0.0] = np.nan                  
            data = -np.log(data)
            index = pd.to_datetime('%s-%03d'%(year, date),format = "%Y-%j")
            vod_world.loc[index,:] = data       
vod_world.sort_index(ascending=True, inplace = True)
store['vod'] = vod_world
store.close()   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
