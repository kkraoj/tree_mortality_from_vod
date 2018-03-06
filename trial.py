
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 23:26:40 2017

@author: kkrao
"""
import os
import pandas as pd
from dirs import Dir_CA
os.chdir(Dir_CA)
store=pd.HDFStore(Dir_CA+'/data_subset.h5')
Df=store['vod_pm']
Df=Df.loc[Df.index.year>=2015]
Df.plot(legend=False)
