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
from dirs import Dir_CA
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
 'sand_fraction','twi_mean','twi_std']
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
trimmed_model_sources=['mortality_025_grid',"RWC_matched","cwd","elevation_std",\
                       "elevation_mean","ppt_sum","location",\
                       "aspect_mean","aspect_std","vsm_sum","canopy_height",\
                       'EVP_win', 'twi_mean']

### base model + lagged RWC
#
lagged_model_sources = base_model_sources+['RWC_lag_1', 'RWC_lag_2']
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

for counter, input_sources in enumerate([base_model_sources, \
                         trimmed_model_sources, lagged_model_sources]):
    Df=rf_assemble(year_range,input_sources)
    Df=rf_remove_nan(Df)
    if counter==0:
        Df.to_csv('D:/Krishna/Project/data/rf_data_base_model.csv')
    elif counter==1:
        Df.to_csv('D:/Krishna/Project/data/rf_data_trimmed_model.csv')
    else:
        Df.to_csv('D:/Krishna/Project/data/rf_data_lagged_model.csv')

#### lagged model [DONT USE]: predictors: 2009 - 2015, FAM: 2010 - 2016
#Df=rf_assemble(range(2009,2015),*inputs)
#Df_lead = rf_assemble(range(2010,2016),*inputs)
#Df['FAM'] = Df_lead['FAM']
##----------------------------------------------------------
### Misc,---------------------------------------------------------------
#subprocess.call("/usr/bin/Rscript --vanilla /D:/Krishna/Project/codes/rf_model.rmd", shell=True)
#Null analysis-----------------------------------------------------------------
#Null=Df[['RWC','cwd','vsm_sum','vsm_win']]
#Null=Null.isnull()
#Null.replace(True,np.nan,inplace=True)
#for column in Null.columns:
#    Null[column].replace(0.0,Null.columns.get_loc(column),inplace=True)
#Null['intersection']=Null.notnull().T.sum()
#Null.intersection[Null.intersection!=4]=np.nan
#Null.intersection[Null.intersection==4]=4
#fig,ax=plt.subplots(figsize=(6,3))
#Null.plot(linestyle='',marker='|',mew='0.1',markersize=10,ax=ax,legend=False,color='darkblue')
##plt.tick_params(axis='y', which='major', labelsize=7)
##ax.set_xlabel('Fraction of Nulls')
#ax.set_ylabel('Features')
#ax.set_title('Data Availability')
#plt.yticks(range(len(Null.columns)),Null.columns)
#labels=range(2009,2016)
#plt.xticks(np.linspace(0,len(Null.index),len(labels),endpoint=False),labels)
#
#
#Null.intersection.replace(False,np.nan,inplace=True)
#Null_frac=(Null.intersection==0).astype(int).mean()
#print('intersection of Nulls = %0.2f'%Null_frac)
#-----------------------------------------------------------------------------
#Df=rf_fill_nan(Df)
#Df['valid_RWC_no_vsm']=0
#Df.loc[(Df.RWC.notnull() & (Df.vsm_sum.isnull() | Df.vsm_win.isnull())),'valid_RWC_no_vsm' ]  = 1
#Df['valid_RWC_no_vsm'].sum()/Df.shape[0]*100

###================================================================
