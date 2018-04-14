# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 13:19:13 2018

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dirs import Dir_CA

start_year=2009
end_year=2016
cardinal='#BD2031'

os.chdir(Dir_CA)
store=pd.HDFStore('data_subset_GC.h5')
data=store['vod_v2']
data=data[(data.index.year>=start_year) &\
          (data.index.year<=end_year)]  
data=data.rolling(30,min_periods=1).mean()
#data=data.loc[:,[139,83,129,117,128]] # high mort pixels
choose=273
data=data.loc[:,choose]
#data=data.loc[:,[196,197,219,258,273]] # low mort pixels

data2=store['vod_pm']
data2=data2[(data2.index.year>=start_year)]  
data2=data2.rolling(30,min_periods=1).mean()
data2=data2.loc[:,choose] # high mort pixels




sns.set_style('ticks')
fig, ax = plt.subplots(figsize=(8,2))
#    ax.grid(axis='x')
ax.set_ylabel('VOD')



cmap=sns.dark_palette("palegreen", as_cmap=True)
#plt.rc('axes', prop_cycle=([cmap(i) for i in np.linspace(0, 0.9, 5)]))
#ax.set_color_cycle([cmap(i) for i in np.linspace(0, 0.9, 5)])
ax.plot(data,'-',label='LPDR v2')  
ax.plot(data2,'-',label='LPDR v1')
for year in np.unique(data.index.year):
    ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=0.5, facecolor=cardinal)

#ax.set_yticks(np.arange(1.2,1.7,0.1))
sns.despine(offset=10, trim=True)
plt.legend()
