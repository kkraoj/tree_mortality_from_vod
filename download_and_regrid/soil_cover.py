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
from dirs import Dir_CA, build_df_from_arcpy, Dir_mort

def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    return data[py,px]
#table=Dir_mort+'/grid.gdb/grid_subset_GC'
#latlon=build_df_from_arcpy(table, columns = ['gridID','x','y'])
#latlon.index = latlon.gridID.astype(int)
#latlon.drop('gridID',axis = 1, inplace = True)
#latlon.rename(columns = {'x':'longitude','y':'latitude'}, inplace = True)
#for col in ['silt','sand','clay']:   
#    latlon['%s'%col] =\
#      get_value(os.path.join('D:\Krishna\Project\data\RS_data\soil\NACP_MSTMIP_UNIFIED_NA_SOIL_MA_1242\data',\
#      'Unified_NA_Soil_Map_Topsoil_%s_Fraction.tif'%col.capitalize()), \
#        latlon.longitude.values, latlon.latitude.values)
#latlon['soil_type']=1
#latlon.loc[(latlon.sand>=latlon.silt)&(latlon.sand>=latlon.clay),\
#           'soil_type']=2
#latlon.loc[(latlon.clay>=latlon.silt)&(latlon.clay>=latlon.sand),\
#           'soil_type']=3
#latlon.replace(-999.0, 33.3,inplace = True)
#latlon.rename(columns = {'silt':'silt_fraction','sand':'sand_fraction'}, inplace = True)
#### fill into df
#os.chdir(Dir_CA)
#store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#for feature in ['silt_fraction','sand_fraction']:
#    df = store['location']
#    for index in df.index:
#        df.loc[index,:] = latlon[feature]
#    df.index.name = feature
#    store[df.index.name] = df
#store.close()

###=================convertin % to fraction====================================
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
for feature in ['silt_fraction','sand_fraction']:
    store[feature]/=100.
store.close()