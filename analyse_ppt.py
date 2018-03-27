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

season='sum'
data=store['ppt_%s'%season]
data=data[(data.index.year>=start_year) &\
          (data.index.year<=end_year)]  
data=data.loc[:,[73,83,97,104,118]] 
data=data.mean(axis=1)




sns.set_style('ticks')
fig, ax = plt.subplots(figsize=(3,3))
#ax.bar(data.mean(axis=1))  

data.plot.bar(legend=False,ax=ax,rot=45,color=cardinal)
ax.set_xticklabels(data.index.year)
ax.set_ylabel('Precipitation %s (mm)'%season)
ax.set_xlabel('Calendar Year')
ax.set_ylim([0,500])
sns.despine()