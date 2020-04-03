# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:10:39 2020

@author: kkrao
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def match(df = None):
#    os.chdir(Dir_CA)
    ref = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSRE"]
    data = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSR2"]
#    if df==None:
#        store = pd.HDFStore('data_subset_GC.h5')
#        df = store['vod_pm']
#        df = df.loc[df.index.year>=2012,:]
    for i in df.index:
        for j in df.columns:
            sys.stdout.write('\r'+'Processing data for i:%s, j:%s ...'%(i,j))
            sys.stdout.flush() 
            val = df.loc[i,j]
            if np.isnan(val):
                continue
            cdf_data = data.iloc[data.index.get_loc(val, method ="nearest")]
            new_val = ref.iloc[ref.searchsorted(cdf_data)].index[0]
#            if new_val - val > 0.1:
#                print('[INFO] Shift > 0.1')
            df.loc[i,j] = new_val
#    old_df = store['vod_pm']
#    old_df = old_df.loc[old_df.index.year<=2011,:]
#    new_df = pd.concat([old_df,df], axis =0)
#    store['vod_pm_matched'] = new_df
    return df


fid = open(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\anci\MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(r'D:\Krishna\projects\vod_from_mortality\codes\data\RS_data\anci\MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()

late=np.asarray(late)*1e-5 ###convert using scale factor
lone=np.asarray(lone)*1e-5 ###convert using scale factor

lat_range = [293, 344]
lon_range = [422,462]

MyDir = 'D:/Krishna/projects/vod_from_mortality/codes/data/RS_data'  #Type the path to your data
fid = open(MyDir+'/'+'anci/globland_r','rb');
EASE_r = np.fromfile(fid,dtype=np.int16)
fid.close()

fid = open(MyDir+'/'+'anci/globland_c','rb');
EASE_s = np.fromfile(fid,dtype=np.int16)
fid.close()

select = (EASE_r>=lat_range[0])&(EASE_r<=lat_range[1])&(EASE_s>=lon_range[0])&(EASE_s<=lon_range[1])
select= np.where(select==True)[0]

df = pd.read_pickle(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\Misc_data\vod_world_3").astype(float)
mean = df.groupby(df.index.year).mean()
#df = None
df = df.loc[:,select]
mean = mean.loc[:,select]
df_amsr2= df.loc[df.index.year>=2013]
mean_amsr2 = mean.loc[mean.index>=2013]
#amazon_mean = mean.mean(axis = 1)
df_amsr2 = match(df_amsr2)
mean_amsr2 = match(mean_amsr2)
matched = mean.copy()
matched.update(mean_amsr2)

sns.set(font_scale=2, style ='ticks')
fig, ax = plt.subplots()
matched.mean(axis =1).plot(ax = ax, marker = 's', label = 'cdf-matched', color = 'm')
mean.mean(axis =1).plot(ax = ax, marker = 's', label = 'raw', color = 'k')
ax.set_ylabel('VOD')
plt.legend(loc = 'lower left')