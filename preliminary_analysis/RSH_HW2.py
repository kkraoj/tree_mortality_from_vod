# -*- coding: utf-8 -*-
"""
Created on Thu May 17 03:03:07 2018

@author: kkrao
"""

import os
import arcpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Dir='D:/Krishna/Acads/18spr/RSH/HW2'
os.chdir(Dir)
filenames = os.listdir(Dir)
dates = [pd.to_datetime(filename[2:10], format = '%Y%m%d') for filename in filenames if 'swe' in filename]
pixel = [400, 612]

Df = pd.DataFrame(index = dates, columns = ['SWE','albedo'])

for filename in filenames:    
    inRas = arcpy.Raster(filename)
    # Convert Raster to numpy array
    arr = arcpy.RasterToNumPyArray(inRas,nodata_to_value=0)
    value = arr[pixel[0],pixel[1]]
    if 'swe' in filename:
        Df.loc[pd.to_datetime(filename[2:10], format = '%Y%m%d'), 'SWE'] = value
    else:
        Df.loc[pd.to_datetime(filename[2:10], format = '%Y%m%d'), 'albedo'] = value

plt, ax1 = plt.subplots(figsize = (6,3))
Df['SWE'].plot(ax = ax1, color='darkorange')
ax1.set_ylabel('SWE', color='darkorange')
ax1.tick_params('y', colors='darkorange')

ax2 = ax1.twinx()
Df['albedo'].plot(ax = ax2, color='m' )
ax2.set_ylabel('albedo', color='m')
ax2.tick_params('y', colors='m')
Df.astype(float)
print(np.corrcoef(Df['SWE'],Df['albedo']))
