# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 09:04:26 2018

@author: kkrao
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import numpy as np
#from dirs import Dir_CA, Dir_ms_fig
import seaborn as sns
os.chdir(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\Misc_data")
#filename = 'vod_world.h5'
#store=pd.HDFStore(filename)
#data_source1='/vod'
#df=store[data_source1]
df = pd.read_pickle('vod_world_3')
#df = df.astype(float)
#data = df.loc[:,173066].rolling(30,min_periods=1).mean()
#data.dropna(inplace = True)
#ref = data.loc[data.index < '2011-10-01']
#data = data.loc[data.index > '2012-07-31']
ref = df.loc[(df.index.year<=2010),:].values.flatten()
ref = ref[~np.isnan(ref)]
data = df.loc[(df.index.year>=2013),:].values.flatten()
data = data[~np.isnan(data)]
ref = ref[ref>0];data = data[data>0]
## plot time series
def plot_ts(ref, data):
    fig, ax = plt.subplots(figsize =  (6,1.5))
    ref.plot(linestyle = '-',color = 'k', ax = ax, label = 'AMSRE')
    data.plot(linestyle = '-',color = 'darkgreen', ax = ax, label = 'AMSR2')
    ax.set_ylabel('VOD')
    ax.legend()
    ax.set_ylim(0.9,2.1)
## plot cdf
def plot_cdf(ref, data, ref_label = 'AMSRE', data_label = "AMSR2"):
    fig, ax = plt.subplots(figsize =  (3,3))
    n_ref,bins_ref, _ = ax.hist(ref, 10000, normed=1, histtype='step',linewidth = 1.5, color = 'k',
                               cumulative=True, label=ref_label)
    n_data, bins_data, _ = ax.hist(data, 10000, normed=1, histtype='step',linewidth = 1.5, color = 'darkgreen',
                               cumulative=True, label=data_label)
    ax.legend(loc = 'lower right')
    ax.set_xlabel('VOD')
    ax.set_ylabel('Likelihood of occurence')
    ax.axvline(0.57, color = "grey", linestyle = '--')
    ax.axvline(2.24, color = "grey", linestyle = '--')
    ax.annotate('CA', xy=(0.3, 0.8), xycoords='axes fraction',color='grey')
    ax.set_xticks(range(4))
    return n_ref,bins_ref, n_data, bins_data

def match(df = None):
#    os.chdir(Dir_CA)
    ref = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSRE"]
    data = pd.HDFStore(r"D:\Krishna\projects\vod_from_mortality\codes\data\Mort_Data\CA\cdf_match.h5")["AMSR2"]
    if df==None:
        store = pd.HDFStore('data_subset_GC.h5')
        df = store['vod_pm']
        df = df.loc[df.index.year>=2012,:]
    for i in df.index:
        for j in df.columns:
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
    
    
def cdf_match(ref, data):
    ref_dist = stats.gamma(*stats.gamma.fit(ref))
    data_dist = stats.gamma(*stats.gamma.fit(data))
    data_corrected = ref_dist.ppf(data_dist.cdf(data))
    return data_corrected   
#matched = cdf_match(ref,data)
#plot_cdf(ref, matched)
#n_ref,bins_ref, n_data, bins_data = \
#    plot_cdf(ref, data, '2003-2010','2013-2015')
    
    

#ref_fun = pd.Series(n_ref, index = bins_ref[:-1])
#ref_fun.name = "AMSRE"
#data_fun =pd.Series(n_data, index = bins_data[:-1])
#data_fun.name = "AMSR2"
#store = pd.HDFStore("cdf_match.h5")
#store[ref_fun.name] = ref_fun
#store[data_fun.name] = data_fun
#store.close()
#plot_ts(ref,data)
#plot_ts(ref,matched)
#store = pd.HDFStore('data_subset_GC.h5')
#old = store['vod_pm']
#new.loc[:,333].rolling(30,min_periods=1).mean().plot()
#new = store['vod_pm_matched']
#old = old.values.flatten(); new = new.values.flatten()
#old = old[~np.isnan(old)]; new = new[~np.isnan(new)]
#plot_cdf(old, new, 'before matching','after matching')
#
#sns.set_style('ticks')
#bins = 1000
#lw = 1
#fig, ax = plt.subplots(figsize =  (3,3))
#for year in [2003, 2006, 2009, 2013]:
#    start=year
#    end = year+2
#    if year==2009:
#        end = year+1
#    label = '%s-%s'%(start,end)
#    gg = df.loc[(df.index.year>=start)&(df.index.year<=end),:].values.flatten()
#    gg = gg[~np.isnan(gg)]
#    y,x, _ = ax.hist(gg, bins = bins, alpha = 1, histtype = "step", normed = 1,\
#            label = label, linewidth = lw, cumulative = True)
#    data = pd.Series(y, index = x[:-1], name = label)
#    data.to_pickle(label)
#gg = df.loc[(df.index.year>=2003)&(df.index.year<=2010),:].values.flatten()
#gg = gg[~np.isnan(gg)]
#label = '2003-2010'
#y,x,_ = ax.hist(gg, bins, normed=1, histtype='step',linewidth = lw,
#        cumulative=True, label=label)
#data = pd.Series(y, index = x[:-1], name = label)
#data.to_pickle(label)
#ax.legend(loc = 'lower right')
#ax.set_xlabel('VOD')
#ax.set_ylabel('Likelihood of occurence')
#ax.set_xticks(range(4))
#ax.set_xlim(0,3)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.show()

#### plotting histograms as lines
#years = {'2003-2005':'s', \
#         '2006-2008':'o',\
#         '2009-2010':'^',\
#         '2013-2015':'v',\
#         '2003-2010':'p'}
#fig, ax = plt.subplots(figsize =  (3,3))
#counter = 0
#for year in sorted(years.keys()):
#    data = pd.read_pickle(year)
#    sensor = " (AMSR-E)"
#    if year == '2013-2015':
#        sensor = " (AMSR-2)"
#    data.plot(ax=ax,label = year+sensor, marker = years[year], \
#              markevery = (counter,300), markersize = 5, lw = 1)
#    counter+=60
#ax.set_xlim(0,3)
#plt.legend()
#plt.show()
##plt.savefig(Dir_ms_fig+'/Figure_S8.tiff', dpi = 300, bbox_inches="tight")
#    
