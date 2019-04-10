# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 00:16:20 2017

@author: kkrao
"""


import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dirs import Dir_CA,RWC
os.chdir(Dir_CA)
np.set_printoptions(threshold='nan')
store=pd.HDFStore('data.h5')
Df=store['vod_pm']
start='2011-09'
end='2012-09'
colors=['grey','indianred']
labels=['Unaffected','Affected']
thresh=2.02
aff_start='2011-10-01'
aff_end='2012-08-01'



fig,ax = plt.subplots(figsize=(4,4))
Df=Df[(Df.index>=start) &\
              (Df.index<=end)]
Df.plot(legend=False,ax=ax,color=colors[0],alpha=0.7,label=labels[0])

gg=Df.T[(Df >= thresh).any(axis=0)].T
gg=gg[(gg.index>=aff_start) &\
              (gg.index<=aff_end)]
gg.plot(legend=False,ax=ax,color=colors[1],alpha=0.7)
handles=range(2)
for i,col,label in zip(handles,colors,labels):
    handles[i]=matplotlib.lines.Line2D([], [], color=col, markersize=100, label=label)
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels) 
ax.set_ylim([0,3.05])
ax.set_ylabel('VOD')
ax.set_title('VOD Timeseries, Raw')
ax.annotate(2011, xy=(-0.05, -0.08), xycoords='axes fraction',\
                ha='left',va='top',size=10)
plt.show()
#gg.to_csv('D:/Krishna/Project/data/affected_grids.csv')
Df.loc[aff_start:aff_end,gg.columns.values]=np.nan

fig,ax = plt.subplots(figsize=(4,4))
Df.plot(legend=False,ax=ax,color=colors[0],alpha=0.7,label=labels[0])
ax.legend(handles=[handles[0]], labels=[labels[0]],loc='lower left') 
ax.set_ylim([0,3.05])
ax.set_ylabel('VOD')
ax.set_title('VOD Timeseries, filtered')
ax.annotate(2011, xy=(-0.05, -0.08), xycoords='axes fraction',\
                ha='left',va='top',size=10)

Df=store['vod_pm']
Df.loc[aff_start:aff_end,gg.columns.values]=np.nan
store['vod_pm_filtered']=Df
rwc=RWC(Df)
rwc.index.name='RWC_filtered'
store[rwc.index.name]=rwc

Df=store['vod_pm']
fig,ax = plt.subplots(figsize=(4,4))
Df.loc[aff_start:aff_end,Df.columns.values]=np.NaN
Df=Df.loc[Df.index.year>=2009]
Df.plot(legend=False,ax=ax,color=colors[0],alpha=0.7,label=labels[0],lw=0.5)
#Df[Df.index.year>=2009].rolling(5,min_periods=0).mean().mean(1).plot(legend=False,ax=ax,color='m',label='mean',lw=1)
ax.set_ylim([0,3.05])
ax.set_ylabel('VOD')
ax.set_title('VOD Timeseries, Filtered')
ax.legend(handles=[handles[0]], labels=[labels[0]],loc='lower left') 

store['vod_pm_filtered']=Df
rwc=RWC(Df)
rwc.index.name='RWC_filtered'
store[rwc.index.name]=rwc