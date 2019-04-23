# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:01:12 2019

@author: kkrao
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from dirs import Dir_CA, remove_vod_affected, select_high_mort_grids, RWC
import numpy as np
import seaborn as sns
np.set_printoptions(threshold=np.nan)

store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
df = store["rwc_lai"]
df2 = store["RWC_matched"]

####################plots
############ scatter plot rwc and rwc_lai
#fig, ax = plt.subplots(figsize = (3,3))
#ax.scatter(df2, df, s = 5, alpha = 0.5)
#ax.set_xlabel("RWC")
#ax.set_ylabel(r"$\frac{RWC}{LAI}$")
#R2 = df.stack().corr(df2.stack())
#ax.annotate("$R$ = %0.2f"%R2, xy=(0.1, 0.8), color = "darkred",\
#            xycoords='axes fraction',ha = "left")

######lai time series 
#gridcell =333
#lai = store['LAI_025_grid_sum']
#lai.index+= timedelta(days=227)
#fig, ax = plt.subplots(figsize = (6,2))
#store["LAI_025_grid"].loc[:,gridcell].plot(legend = False, ax = ax)
#lai.loc[:,gridcell].plot(legend = False, ax = ax, marker = 'o',\
#       markersize = 6, linestyle = "", color = 'b')
#ax2 = ax.twin
#store["vod_pm_matched"].loc[:,gridcell].plot(legend = False, ax = ax2)
#ax.set_ylabel('LAI')
#ax.set_xlabel("")
#plt.show()

############# time series VOD/LAI
#vod = store["vod_pm_matched"]
#lai = store["LAI_025_grid"]
#lai = lai.resample('1d').asfreq().interpolate()
#vod = vod.resample('1d').asfreq().interpolate()
#df = vod/lai
#df.index.name = 'vod_lai'
#store[df.index.name] = df
#df = df.loc[(df.index.year>=2009)&(df.index.year<=2015),:]
#
#fig, ax = plt.subplots(figsize = (6,2))
#df.loc[:,333].plot(legend = False, ax = ax)
#ax.set_ylabel(r'$\frac{VOD}{LAI}$')
#ax.set_xlabel("")
#plt.show()

##rwc = vod/lai scatter plot with mort
####### mort and ndwi sum win scatter plot
mort = store['mortality_025_grid']
mort = mort.loc[(mort.index.year>=2009)&(mort.index.year<=2015),:]
rwc= store['rwc_vod_lai']
rwc = rwc.loc[(rwc.index.year>=2009)&(rwc.index.year<=2015),:]
fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(rwc,mort, color = 'k', s = 6, alpha = 0.5)
ax.set_xlabel(r"RWC = $\frac{f(VOD)}{LAI}$")
ax.set_ylabel("FAM")
rwc.index.name = ""
mort.index.name = ""
R2 = rwc.stack().corr(mort.stack())
ax.annotate("$R$ = %0.2f"%R2, xy=(0.1, 0.8), color = "darkred",\
        xycoords='axes fraction',ha = "left")

###############
#df = store['vod_pm_matched']
#df = remove_vod_affected(df)
#df = select_high_mort_grids(df)
#start_month=7
#months_window=3
#df = df.loc[(df.index.year>=2009)]
#df=df.loc[(df.index.month>=start_month) & (df.index.month<start_month+months_window)]
#df = df.groupby(df.index.year).mean()
#print((df.std()/df.mean()).mean())
#
#df2 = store[ '/LAI_025_grid_sum']
#df2 = df2.loc[(df2.index.year>=2009)]
#df2 = select_high_mort_grids(df2)
#print((df2.std()/df2.mean()).mean())

######VOD/LAI composite time series with time shifting
vod = store['vod_pm_matched']
lai = store['/LAI_025_grid']

df = vod/lai

grid_cell = 333
alpha1 = 0.2
alpha2 = 0.5
color = '#BD2031'

sns.set_style('ticks')
fig, ax = plt.subplots(figsize = (6,2))
df.loc[:,grid_cell].rolling(60,min_periods=1).mean().plot(ax = ax, label = 'vod/lai')
vod.loc[:,grid_cell].rolling(30,min_periods=1).mean().plot(ax = ax, label = 'vod', color = 'k', alpha = alpha2)
lai.loc[:,grid_cell].rolling(30,min_periods=1).mean().plot(ax = ax, label = 'lai', color = 'g', alpha = alpha2)

for year in np.unique(df.index.year):
    ax.axvspan(*pd.to_datetime(['%d-01-01'%year,'%d-03-30'%year]), alpha=alpha1, facecolor=color)
plt.legend()


##store the new df with RWC of synthetic time series

df.head()
df.index.name = "rwc_vod_lai"
df = RWC(df, start_year = 2005, start_month=1)
store[df.index.name] = df ### rwc = anomaly(VOD/LAI)


