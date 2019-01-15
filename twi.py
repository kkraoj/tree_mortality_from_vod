# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 03:00:39 2018

@author: kkrao
"""
import arcpy
import os
import pandas as pd
import numpy as np
import gdal
from dirs import Dir_CA, build_df_from_arcpy, Dir_mort, MyDir

#def get_value(filename, mx, my):
#    ds = gdal.Open(filename)
#    gt = ds.GetGeoTransform()  
#    data = ds.GetRasterBand(1).ReadAsArray()
#    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
#    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
#    return data[py,px]
#os.chdir(MyDir)
#table=Dir_mort+'/grid.gdb/grid_subset_GC'
#latlon=build_df_from_arcpy(table, columns = ['gridID','x','y'])
#latlon.index = latlon.gridID.astype(int)
#latlon.drop('gridID',axis = 1, inplace = True)
#latlon.rename(columns = {'x':'longitude','y':'latitude'}, inplace = True)
#for col in ['twi']:   
#    latlon['%s'%col] =\
#      get_value(os.path.join('twi/na_cti_prj'), \
#        latlon.longitude.values, latlon.latitude.values)    
##print(latlon.twi.unique())
#### fill into df
#os.chdir(Dir_CA)
#store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#for feature in ['twi']:
#    df = store['location']
#    for index in df.index:
#        df.loc[index,:] = latlon[feature]
#    df.index.name = feature
#    store[df.index.name] = df
#store.close()
####### twi mean and std from arcmap zonal stats table#########################
os.chdir(MyDir)
df = build_df_from_arcpy('twi/twi_stats', columns=['GRIDID','MEAN','STD'],dtype=None, index_col = 'GRIDID')
df.index.name = 'gridID'
df.index = df.index.astype(int)
df.columns = df.columns.str.lower()
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
for feature in df.columns:
    d = store['location']
    for index in d.index:
        d.loc[index,:] = df[feature]
    d.index.name = 'twi_'+feature
    store[d.index.name] = d
store.close()
