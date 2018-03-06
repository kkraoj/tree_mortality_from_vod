# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 17:45:32 2017

@author: kkrao
"""
#import arcpy
#import os
import pandas as pd
import numpy as np
from dirs import Dir_CA

store = pd.HDFStore(Dir_CA+'/CWD_Df.h5') 
store_major=   pd.HDFStore(Dir_CA+'/data.h5')           
fields=['EVP','PEVAP']
for variable in fields:
    Df=store[variable]
    Df.index=Df['gridID']-1
    Df.drop('gridID',axis='columns',inplace=True)
    Df=Df.T
    Df.index=pd.to_datetime(Df.index,format='%Y%m')
    Df.index.name=variable
    store[Df.index.name] = Df

for variable in fields:
    store_major[variable]=store[variable]