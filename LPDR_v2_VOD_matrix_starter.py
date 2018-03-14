# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 12:55:35 2018

@author: kkrao
Objectives: 
    1. Understand how LPDR_v2 data is stored and retrieved
    2. access data for example year and day
    3. plot data
"""
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from dirs import MyDir # home directory
from mpl_toolkits.basemap import Basemap

os.chdir(MyDir+'/LPDR_v2/VOD_matrix') # location of VOD matrix
store=pd.HDFStore('VOD_LPDR_v2.h5') #this is your store or marketplace of vod files for all dates

param='VOD'
year=2014
date=245
pass_type='A'

filename='%s_%s_%s_%03d'%(param,pass_type,year,date)
Df=store[filename] #vod file for year, date
#print(Df.head())# display first few rows of vod

# plot vod-------------------------------------------------------------------

def get_marker_size(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
    return marker_size

enlarge=1.5
font = {'family' : 'normal',
        'size'   : 22}
#mpl.rc('font', **font)
sns.set(font_scale=2)
#mpl.rcParams.update({'font.size': 22})
latcorners=np.array([-70,70])
loncorners=np.array([-180,180])
#lats, lons = np.mgrid[90:-90:-0.25, -180:180:0.25]
lats, lons = np.meshgrid(Df.index,Df.columns,indexing='ij')
cmap = 'YlGnBu'
sns.set_style('ticks')

fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
marker_size=get_marker_size(ax,fig,loncorners)
m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                ax=ax)
m.drawcoastlines()
m.drawmapboundary(fill_color='lightcyan')
m.fillcontinents(color='papayawhip',zorder=0)
plot=m.scatter(lons, lats, s=marker_size,c=Df,cmap=cmap,\
                    marker='s', vmin=0.0,vmax=3.0)
ax.set_title('LPDR_v2 %s, year: %d, day: %d'%(param, year, date))
cbaxes = fig.add_axes([0.17, 0.3, 0.03, 0.15])
cb=fig.colorbar(plot,ax=ax,cax=cbaxes,ticks=range(4))
cbaxes.annotate('%s'%param,xy=(0,1.1), xycoords='axes fraction',\
            ha='left')
#

