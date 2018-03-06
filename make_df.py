# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
## df from stats table

from __future__ import division
import os
import sys
import arcpy
import pandas as pd
import numpy as np
from dirs import Dir_mort, Dir_CA
year_range=range(2016,2018)
date_range=range(1,366,1)
map_factor=1e4
os.chdir(Dir_CA)
store=pd.HDFStore('data_subset.h5')
old_Df=store['vod_pm']
nos=old_Df.shape[1] # number of grid cells
Df=pd.DataFrame()
Df.index.name='gridID'
pass_type = 'A';             #Type the overpass: 'A' or 'D'
arcpy.env.workspace=Dir_mort+'/'+'CA_proc.gdb'
for year in year_range:
    for date in date_range:             
        sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
        sys.stdout.flush()   
        data=pd.DataFrame(np.full(nos,np.nan).reshape(1, nos), \
                          columns=old_Df.columns,\
                          index = [pd.to_datetime('%s%03d'%(year,date),format='%Y%j')])
        fname="VOD_stats_%s_%03d_%s" %(year,date,pass_type)
        if  arcpy.Exists(fname):
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                data.loc[pd.to_datetime('%s%03d'%(year,date),format='%Y%j')\
                         ,row.getValue('gridID')]=\
                         row.getValue('MEAN')/map_factor# grid ID is 1 to 370
            Df=pd.concat([Df,data],axis=0)
Df=pd.concat([old_Df,Df],axis=0)
store['vod_pm']=Df
store.close()
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
