# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 12:55:35 2018

@author: kkrao
Objectives: 
    1. Understand how LPDR_v2 VOD data is stored and presented

"""
import os
import pandas as pd
from dirs import MyDir # home directory

os.chdir(MyDir+'/LPDR_v2/VOD_matrix') # location of VOD matrix
store=pd.HDFStore('VOD_LPDR_v2.h5') #this is your store or marketplace of vod files for all dates

param='VOD'
year=2008
date=43
pass_type='A'

filename='%s_%s_%s_%03d'%(param,pass_type,year,date)
Df=store[filename] #vod file for year, date
print(Df.head())# display first few rows of vod

