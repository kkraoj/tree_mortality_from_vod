# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:10:39 2020

@author: kkrao
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from mkgrid_global import mkgrid_global
from mpl_toolkits.axes_grid1 import make_axes_locatable



dir_data = r"D:\Krishna\projects\vod_from_mortality\codes\data"

ref = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSRE"]
data = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSR2"]

ref = ref.reindex(np.linspace(0,3,1000),method = 'nearest')
ref = ref.clip(0.0,1.0)
data = data.reindex(np.linspace(0,3,1000),method = 'nearest')
data = data.clip(0.0,1.0)

def match(val):
    if np.isnan(val):
        return val
    cdf_data = data.iloc[data.index.get_loc(val, method ="nearest")]
    try:
        new_val = ref.iloc[ref.searchsorted(cdf_data)].index[0]
#                           np.sum([ref<0.5])
#        print('[INFO] Success \t Original value = %0.2f'%val)
        return new_val
    except IndexError:
        print('[INFO] Original value = %0.2f'%val)
        print('[INFO] CDF value = %0.2f'%cdf_data)
        print('[INFO] New index value = %d'%ref.searchsorted(cdf_data))
        return np.nan

df = pd.read_pickle(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\Misc_data\vod_world_3").astype(float)
##df = None
#df = df.clip(0.0,3.0)
#df_amsr2= df.loc[df.index.year>=2013]
##amazon_mean = mean.mean(axis = 1)
#df_amsr2 = df_amsr2.applymap(match)
#df_amsr2.to_pickle(os.path.join(dir_data,'vod-biomass/matched-vod-global'))
df_amsr2 = pd.read_pickle(os.path.join(dir_data,'vod-biomass/matched-vod-global'))
##gg.applymap(match)
#matched = df.copy()
#matched.update(df_amsr2)

df_annual = df.groupby(df.index.year).mean()
df_annual.update(df_amsr2.groupby(df_amsr2.index.year).mean())

#%% plotting
sns.set(font_scale=2, style ='ticks')

fid = open(os.path.join(dir_data,r'RS_data\anci\MLLATLSB'),'rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(os.path.join(dir_data,r'RS_data\anci\MLLONLSB'),'rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
y=np.asarray(late)*1e-5 ###convert using scale factor
x=np.asarray(lone)*1e-5 ###convert using scale factor

diff = df_annual.loc[df_annual.index>=2013].mean() - df_annual.loc[df_annual.index<=2012].mean()

diffm = mkgrid_global(diff)
            
fig, ax = plt.subplots(figsize = (8,4),dpi=300)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.annotate('$\Delta$ VOD', xy=(-0.3, 1.05), xycoords='axes fraction',ha = 'left',va = 'bottom')

plot = ax.scatter(x.flatten().tolist(), y.flatten().tolist(), c = diffm.flatten().tolist(), cmap = "RdYlGn",s = 1, vmin = -0.2, vmax = 0.2)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(plot, cax= cax, orientation='vertical')


#%%

diff = df_annual.loc[df_annual.index>=2013].mean() - df_annual.loc[df_annual.index<=2012].mean()

#%% plotting

diff = df_annual.loc[df_annual.index>=2013].mean() - df_annual.loc[df_annual.index<=2012].mean()
diff = diff/df_annual.loc[df_annual.index<=2012].mean()*100
diffm = mkgrid_global(diff)
                         
           
fig, ax = plt.subplots(figsize = (8,4),dpi=300)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.annotate('$\Delta$ VOD (%)', xy=(-0.3, 1.05), xycoords='axes fraction',ha = 'left',va = 'bottom')

plot = ax.scatter(x.flatten().tolist(), y.flatten().tolist(), c = diffm.flatten().tolist(), cmap = "RdYlGn",s = 1, vmin = -15, vmax = 15)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(plot, cax= cax, orientation='vertical')

#%% average VOD map

fig, ax = plt.subplots(figsize = (8,4),dpi=300)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.annotate('VOD', xy=(-0.3, 1.05), xycoords='axes fraction',ha = 'left',va = 'bottom')

plot = ax.scatter(x.flatten().tolist(), y.flatten().tolist(), c = diffm.flatten().tolist(), cmap = "YlGnBu",s = 1, vmin = 0, vmax = 2)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(plot, cax= cax, orientation='vertical')

