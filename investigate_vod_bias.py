# -*- coding: utf-8 -*-
"""
Created on Sun Aug 05 19:22:07 2018

@author: kkrao
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

from mpl_toolkits.basemap import Basemap
from scipy import optimize
from matplotlib.patches import Rectangle, Patch
from dirs import Dir_CA, MyDir, supply_lat_lon, get_marker_size, plot_map
from mkgrid_global import mkgrid_global
import seaborn as sns
from scipy import stats


def decrease_res(array, size = 10000):
    array = np.sort(array)
    sample = np.random.choice(array, size = size, replace = False)
    return sample
    
    
    
    
    
sns.set_style('ticks')

os.chdir("D:/Krishna/Project/data/Mort_Data/Misc_data")
df = pd.read_pickle('vod_world_3').astype(np.float)
#### ignore satellite changeover time
#df = df.loc[df.index.year>=2009,:]
df.loc[(df.index >= '2011-10-01')&(df.index <= '2012-07-31'),:] = np.nan
#factor = 1e-5
#fid = open(MyDir+'/anci/MLLATLSB','rb');
#lat= np.fromfile(fid,dtype=np.int32).reshape((586,1383))*factor
#fid = open(MyDir+'/anci/MLLONLSB','rb');
#lon= np.fromfile(fid,dtype=np.int32).reshape((586,1383))*factor
#fid.close()
#bias = df.loc[df.index.year>=2012,:].mean()-\
#             df.loc[df.index.year<=2011,:].mean()
#bias = mkgrid_global(bias)
#fig,ax,m,plot = plot_map(lat,lon, bias,proj = 'cyl',latcorners = [-60,75], \
#                         vmin = -0.4, vmax = 0.4, cmap = 'bwr_r', enlarge = 2)
#cb=fig.colorbar(plot,ax=ax,fraction=0.018,aspect=20,pad=0.02)
######---------------------------------- CA map
#latcorners=np.array([33,42.5])
#loncorners=np.array([-124.5,-117])
#fig,ax,m,plot = plot_map(lat,lon, bias,proj = 'cyl',latcorners = latcorners,\
#                         loncorners = loncorners,drawstates = True,\
#                         vmin = -0.4, vmax = 0.4, cmap = 'bwr_r', enlarge = 2)
#cb=fig.colorbar(plot,ax=ax,pad=0.02)
#########------------------------------------ hist
#sns.set_style('whitegrid')
#fig, ax = plt.subplots(figsize = (2,2))
#for year in range(2005,2010)+[2013]:
#    start=year
#    end = year+2
##    gg = bias.flatten()
#    gg = df.loc[(df.index.year>=start)&(df.index.year<=end),:].values.flatten()
#    gg = gg[~np.isnan(gg)]
#    ax.hist(gg, bins = 1000, alpha = 1, histtype = "step", \
#            label = '%s - %s'%(start,end), linewidth = 1.5)
#    ax.set_xlim(0,2.5)
##    ax.set_ylim(0,3.5e5)
#    ax.set_ylabel('Frequency (grid cells)')
#    ax.set_xlabel('VOD')
##    ax.set_title('%s - %s'%(start,end))
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.show()
####---------------------------------------- ks test stats
for year in [2003, 2006, 2009]:
    y1=year;y2=year+2
    y3=year+3;y4=year+5
    sample1 = df.loc[(df.index.year>=y1)&(df.index.year<=y2),:].values.flatten()
    sample1= sample1[~np.isnan(sample1)]
    if year==2009:
        y3=year+4;y4=year+6
    sample2 = df.loc[(df.index.year>=year+1)&(df.index.year<=year+3),:].values.flatten()
    sample2= sample2[~np.isnan(sample2)]
    sample1 = decrease_res(sample1)
    sample2 = decrease_res(sample2)
    kstat,pvalue = stats.ks_2samp(sample1, sample2)
    print('(%s-%s) & (%s-%s) are different. kstat = %0.4f, pvalue = %0.4f)'\
         %(y1,y2,y3,y4, kstat, pvalue))
#    stats.ks_2samp(np.random.randn(100000), np.random.randn(100000))
#    print('Mean = %0.4f'%gg.mean())
#    print(stats.ttest_1samp(gg,0.0, axis =None ))
#    print(stats.ttest_1samp(np.random.randn(100000),0.0, axis =None ))
#    (gg<0).sum()/float(len(gg))

######---------------------------------------------- africa timeseries
#lat_in = -11.69
#lon_in = 24.13
#np.where(np.abs(lat-lat_in)<=1e-2)
#lat[352,:]
#np.where(np.abs(lon-lon_in)<=1e-1)
#lon[:,784]
#for i in range(209091):
#    if (EASE_r[i] == 352)&(EASE_s[i] == 784):
#        print(i)##173066

#fig, ax = plt.subplots(figsize = (6,2))
#df.loc[:,173066].rolling(30,min_periods=1).mean().plot(linestyle = '-',color = 'k')

