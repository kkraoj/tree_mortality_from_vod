# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:10:39 2020

@author: kkrao
"""

import numpy as np
import pandas as pd
import sys
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable



dir_data = r"D:\Krishna\projects\vod_from_mortality\codes\data"


def select_south_america():
#    (lat -60 to 12, lon = -180 to -29, is what I used)
    lat_range = [-60,12]
    lon_range = [-180,-29]
    EASE_r, EASE_s = supply_EASE()
    lat,lon = supply_lat_lon()
    select = (lat>=lat_range[0])&(lat<=lat_range[1])
    lat_range = [np.where(select==True)[0].min(),np.where(select==True)[0].max()] 
    select = (lon>=lon_range[0])&(lon<=lon_range[1])
    lon_range = [np.where(select==True)[1].min(),np.where(select==True)[0].max()] 
            
    select = (EASE_r>=lat_range[0])&(EASE_r<=lat_range[1])&(EASE_s>=lon_range[0])&(EASE_s<=lon_range[1])
    select= np.where(select==True)[0]

    df = pd.read_pickle(os.path.join(dir_data, r"Mort_Data\Misc_data\vod_world_3")).astype(float).round(decimals = 2)
    df = df.loc[:,select]
    df = df.clip(0.0,3.0)
    return df

def plot_cdf(ref, data, ref_label = 'AMSRE', data_label = "AMSR2"):
        fig, ax = plt.subplots(figsize =  (3,3))
        n_ref,bins_ref, _ = ax.hist(ref, np.linspace(0,3.01,302), normed=1, histtype='step',linewidth = 1.5, color = 'k',
                                   cumulative=True, label=ref_label)
        n_data, bins_data, _ = ax.hist(data, np.linspace(0,3.01,302), normed=1, histtype='step',linewidth = 1.5, color = 'darkgreen',
                                   cumulative=True, label=data_label)
        ax.legend(loc = 'lower right')
        ax.set_xlabel('VOD')
        ax.set_ylabel('Likelihood of occurence')
        ax.axvline(0.57, color = "grey", linestyle = '--')
        ax.axvline(2.24, color = "grey", linestyle = '--')
        ax.annotate('CA', xy=(0.3, 0.8), xycoords='axes fraction',color='grey')
        ax.set_xticks(range(4))
        return n_ref,bins_ref, n_data, bins_data

def create_cdfs(df):
    
    ref = df.loc[(df.index.year<=2010),:].values.flatten()
    ref = ref[~np.isnan(ref)]
    data = df.loc[(df.index.year>=2013),:].values.flatten()
    data = data[~np.isnan(data)]
    
    n_ref,bins_ref, n_data, bins_data = \
        plot_cdf(ref, data, '2003-2010','2013-2015')
        
    ref_fun = pd.Series(n_ref, index = bins_ref[:-1])
    ref_fun.name = "AMSRE"
    data_fun =pd.Series(n_data, index = bins_data[:-1])
    data_fun.name = "AMSR2"
    store = pd.HDFStore(os.path.join(dir_data, "Mort_data\CA\cdf_match_south_america.h5"))
    store[ref_fun.name] = ref_fun
    store[data_fun.name] = data_fun
    store.close()
   


def supply_EASE():
    fid = open(os.path.join(dir_data,r'RS_data\anci\globland_r'),'rb');
    EASE_r = np.fromfile(fid,dtype=np.int16)
    fid.close()
    
    fid = open(os.path.join(dir_data,r'RS_data\anci\globland_c'),'rb');
    EASE_s = np.fromfile(fid,dtype=np.int16)
    fid.close()
    
    return EASE_r, EASE_s

def supply_lat_lon():
    fid = open(os.path.join(dir_data,r'RS_data\anci\MLLATLSB'),'rb');
    late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
    fid.close()
    fid = open(os.path.join(dir_data,r'RS_data\anci\MLLONLSB'),'rb');
    lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
    fid.close()
    
    late=np.asarray(late)*1e-5 ###convert using scale factor
    lone=np.asarray(lone)*1e-5 ###convert using scale factor
    return late, lone

def mkgrid_global(x):
    #Load ancillary EASE grid row and column data, where <MyDir> is the path to 
    #wherever the globland_r and globland_c files are located on your machine.
    EASE_r, EASE_s = supply_EASE()
    
#    plt.latlon = True
    #Initialize the global EASE grid 
    gridout = np.empty([586,1383]);
    gridout[:]=np.NaN                  
    
    #Loop through the elment array
    for i in list(range(209091)):
        '''  
        %Distribute each element to the appropriate location in the output
        %matrix (EASE grid base address is referenced to (0,0), but MATLAB is
        %(1,1)
        '''
        gridout[EASE_r[i],EASE_s[i]] = x[i];
               
    return(gridout)
              
def match(val):
    if np.isnan(val) or val==3.00:
        return val
    cdf_data = data.iloc[data.index.get_loc(val, method ="nearest")]
    new_val = ref.iloc[ref.searchsorted(cdf_data)].index[0]
    return new_val
#    except IndexError:
#        print('[INFO] Original value = %0.2f'%val)
#        print('[INFO] CDF value = %0.2f'%cdf_data)
#        print('[INFO] New index value = %d'%ref.searchsorted(cdf_data))
#        return np.nan

def plot_ts(raw,matched):
    sns.set(font_scale=2, style ='ticks')
    fig, ax = plt.subplots()
    matched.mean(axis =1).plot(ax = ax, marker = 's', label = 'SA-cdf-matched', color = 'm')
    raw.mean(axis =1).plot(ax = ax, marker = 's', label = 'raw', color = 'k')
    ax.set_ylabel('VOD')
    plt.legend(loc = 'lower left')



df = select_south_america()
create_cdfs(df)
ref = pd.HDFStore(os.path.join(dir_data, "Mort_Data\CA\cdf_match_south_america.h5"))["AMSRE"]
data = pd.HDFStore(os.path.join(dir_data, "Mort_Data\CA\cdf_match_south_america.h5"))["AMSR2"]

#gg = pd.DataFrame([ref,data])

df_amsr2= df.loc[df.index.year>=2013]
df_amsr2 = df_amsr2.applymap(match)
df_amsr2.to_pickle(os.path.join(dir_data,'vod-biomass/matched-vod-south_america'))
#df_amsr2 = pd.read_pickle(os.path.join(dir_data,'vod-biomass/matched-vod-global'))

df_annual = df.groupby(df.index.year).mean()
df_annual_matched = df_annual.copy()
df_annual_matched.update(df_amsr2.groupby(df_amsr2.index.year).mean())




#%% select amazon and plot TS

lat_range = [293, 344]
lon_range = [422,462]

EASE_r, EASE_s = supply_EASE()
select = (EASE_r>=lat_range[0])&(EASE_r<=lat_range[1])&(EASE_s>=lon_range[0])&(EASE_s<=lon_range[1])
select= np.where(select==True)[0]
df_annual = df_annual.loc[:,select]
df_annual_matched = df_annual_matched.loc[:,select]

plot_ts(df_annual,df_annual_matched)
