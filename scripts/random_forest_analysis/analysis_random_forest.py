# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:32:07 2017

@author: kkrao
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from dirs import Dir_CA, select_north_south_grids
### commented above line because dirs also import osgeo and I am having a problem fixing it
Dir_CA='D:/Krishna/projects/vod_from_mortality/codes/data/Mort_Data/CA'
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
def rf_assemble(year_range,input_sources):
    dfs=range(len(input_sources))
    for i in range(len(input_sources)):
        dfs[i]=store[input_sources[i]]
    if len(dfs)==1:
        dfs=[dfs]
    out=pd.DataFrame(index=range(dfs[0].shape[1]*(year_range[-1]-year_range[0]+1)))
    for df in dfs:                    
        df=df[(df.index.year>=year_range[0]) &\
                  (df.index.year<=year_range[-1])].T
        array=pd.Series(df.values.flatten(),name=df.columns.name)
        out[array.name]=array
    return(out)

def rf_assemble_north_south(year_range,input_sources):
    dfs=range(len(input_sources))
    for i in range(len(input_sources)):
        dfs[i]=store[input_sources[i]]
    if len(dfs)==1:
        dfs=[dfs]
    out_n=pd.DataFrame(index=range(dfs[0].shape[1]*(year_range[-1]-year_range[0]+1)))
    out_s=pd.DataFrame(index=range(dfs[0].shape[1]*(year_range[-1]-year_range[0]+1)))
    for df in dfs:                    
        df=df[(df.index.year>=year_range[0]) &\
                  (df.index.year<=year_range[-1])]
        df_n, df_s = select_north_south_grids(df)
        df_n, df_s = df_n.T, df_s.T
        array_n=pd.Series(df_n.values.flatten(),name=df_n.columns.name)
        array_s=pd.Series(df_s.values.flatten(),name=df_s.columns.name)
        out_n[array_n.name]=array_n
        out_s[array_s.name]=array_s
    return(out_n, out_s)

def rf_fill_nan(df):
    df.fillna(method='bfill',inplace=True)
    df.fillna(method='ffill',inplace=True)

    return df

def rf_remove_nan(df):
    df.dropna(inplace=True)
#    df.index=range(df.shape[0])
    return df

#base model
base_model_sources=['mortality_025_grid','BPH_025_grid','LAI_025_grid_sum',\
'LAI_025_grid_win','RWC_matched', 'aspect_mean', 'aspect_std', 'canopy_height',\
 'cwd','elevation_mean','elevation_std',\
 'forest_cover','ppt_sum','ppt_win','tmax_sum','tmax_win',\
 'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
'PEVAP_sum','EVP_win','PEVAP_win','vsm_sum','vsm_win','location', 'silt_fraction',\
 'sand_fraction','twi_mean','twi_std', 'ndwi_sum','ndwi_win']
year_range=range(2009,2016)
###### static sources only
#input_sources=['mortality_025_grid','aspect_mean', 'aspect_std', 'canopy_height',\
# 'elevation_mean','elevation_std',\
# 'forest_cover','location', 'silt_fraction',\
# 'sand_fraction','twi']

### sources till 2016 available
#input_sources=['mortality_025_grid','LAI_025_grid_sum',\
#'LAI_025_grid_win','RWC_v2', 'aspect_mean', 'aspect_std', 'canopy_height',\
# 'elevation_mean','elevation_std',\
# 'forest_cover','ppt_sum','ppt_win','tmax_sum','tmax_win',\
# 'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
#'PEVAP_sum','EVP_win','PEVAP_win','location']
#year_range=range(2009,2017)

##uncorrelated sources |r| < 0.50
lai_model_sources = base_model_sources+['rwc_lai']

trimmed_model_sources=['mortality_025_grid',"RWC_matched","cwd","elevation_std",\
                       "elevation_mean","ppt_sum","location",\
                       "aspect_mean","aspect_std","vsm_sum","canopy_height",\
                       'EVP_win', 'twi_mean']

### base model + lagged RWC
#####changed to new after peer review
lagged_model_sources = base_model_sources+['RWC_lag_1_new', 'RWC_lag_2_new']
os.chdir(Dir_CA)
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
###----------filling RWC values for 2009
#Df = store['RWC_lag_1']
#Df = Df.interpolate(method = 'spline', order = 3, axis = 1).bfill()
#from dirs import RWC
#Df = RWC(store['vod_pm'], start_year = 2008).shift(1)
#Df.index.name = 'RWC_lag_1_fill'
#store[Df.index.name]=Df
####=====================================

save = False
if save:

    for counter, input_sources in enumerate([base_model_sources, \
                             trimmed_model_sources, lagged_model_sources,\
                             lai_model_sources]):
        Df=rf_assemble(year_range,input_sources)
        Df=rf_remove_nan(Df)
        if counter==0:
            Df.to_csv('D:/Krishna/projects/vod_from_mortality/codes/data/rf_data_base_model_with_year.csv')
        elif counter==1:
            Df.to_csv('D:/Krishna/Project/data/rf_data_trimmed_model.csv')
        elif counter ==2:
            Df.to_csv('D:/Krishna/Project/data/rf_data_lagged_model.csv')
        else:
            Df.to_csv('D:/Krishna/Project/data/rf_data_lai_model.csv')
    
    
    Df_n, Df_s=rf_assemble_north_south(year_range,base_model_sources)
    Df_n=rf_remove_nan(Df_n)
    Df_s=rf_remove_nan(Df_s)
    Df_n.to_csv('D:/Krishna/Project/data/rf_data_base_model_north.csv')
    Df_s.to_csv('D:/Krishna/Project/data/rf_data_base_model_south.csv')
    
#%%

## need to add year column to rf dataframe for Jean Peierre's student
#
#year = dfs[0].copy()
#
#for row in range(year.shape[0]):
#    year.iloc[row,:] = int(year.index.year[row])
#    
#year = year.astype(int)
#store['year'] = year
#     
#base_model_sources=['year','mortality_025_grid','BPH_025_grid','LAI_025_grid_sum',\
#'LAI_025_grid_win','RWC_matched', 'aspect_mean', 'aspect_std', 'canopy_height',\
# 'cwd','elevation_mean','elevation_std',\
# 'forest_cover','ppt_sum','ppt_win','tmax_sum','tmax_win',\
# 'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
#'PEVAP_sum','EVP_win','PEVAP_win','vsm_sum','vsm_win','location', 'silt_fraction',\
# 'sand_fraction','twi_mean','twi_std', 'ndwi_sum','ndwi_win']
