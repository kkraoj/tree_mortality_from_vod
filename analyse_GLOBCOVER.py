# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:35:03 2018

@author: kkrao
"""

from __future__ import division
import os
from dirs import MyDir, Dir_CA, make_lat_lon, bar_label, supply_lat_lon, \
    get_marker_size_v2, select_years, adjust_spines, make_axes_locatable
#from pyhdf.SD import SD, SDC

import gdal
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from textwrap import wrap
from collections import Counter

### histogram at 250 m resolution==============================================
#os.chdir(MyDir+'/Forest')
#file_name = 'gc_2009_clip'
#
#gdal_dataset = gdal.Open(file_name)
#gdal_dataset.GetSubDatasets()
##lc_data = gdal.Open('HDF4_EOS:EOS_GRID:"MCD12C1.A2001001.051.2014274170618.hdf":MOD12C1:Land_Cover_Type_1_Percent')
#lc = gdal_dataset.ReadAsArray()
#np.save('gc_2009_lc.npy',lc)
#gdal_dataset.GetGeoTransform()
##
##
#
#def subset_lc(lc,classes):
#    mask=~np.isin(lc,classes)
##    data=np.ma.masked_array(lc, mask=mask)
#    lc[mask]=255
#    return lc
#    
#classes=[50,60,70,90,100]
#
#data=subset_lc(lc,classes)
#unique, counts = np.unique(data, return_counts=True)
#lc_sum=dict(zip(unique, counts))
#del(lc_sum[255])
#
#### fraction of mixed forest in proportion to total forest
###forest is defined as classes 50,60,70,90,100
#mixed_fraction = lc_sum[100]/sum(lc_sum.values())
#lc_frac=dict((k, v/sum(lc_sum.values())) for k,v in lc_sum.items())
#lc_labels = pd.read_excel(MyDir+'/Forest'+'/GLOBCOVER/Globcover2009_Legend.xls',header=0,index_col=0)
#lc_labels.Label=lc_labels.Label.str.wrap(40)
#
#
#lc_frac=pd.DataFrame(lc_frac.values(),index = lc_labels.loc[lc_frac.keys(),'Label'],columns = ['fraction coverage'])
#
#mpl.style.use('seaborn-darkgrid')
#fig, ax = plt.subplots(figsize=(3,3))
#plot=lc_frac.plot.barh(legend=False,ax=ax)
#ax.set_xlim([0,1])
#bar_label(lc_frac['fraction coverage'],ax, x_offset=0.02)


### histogram at 25 Km resolution==============================================
#
#lc=pd.read_excel('D:/Krishna/Project/working_tables.xlsx',\
#                 sheetname='gc_ever_deci',index_col=1) 
#lc=lc.loc[:,['evergreen','deciduous']]
#lc.loc[:,'evergreen']=lc.evergreen>lc.deciduous
#lc.deciduous=~lc.evergreen
#lc=lc.sum()/lc.sum().sum()
#
#fig, ax = plt.subplots(figsize=(3,3))
#plot=lc.plot.barh(ax=ax)
#bar_label(lc, ax)
#ax.set_xlim([0,1])
#ax.set_title('Dominant forest cover type, 25 km$^2$')

### histogram at 25 Km resolution with mixed landcover type====================

#lc=pd.read_excel('D:/Krishna/Project/working_tables.xlsx',\
#                 sheetname='gc_ever_deci',index_col=1) 
#lc=lc.loc[:,['evergreen','deciduous','mixed']]
#lc=lc.idxmax(axis=1)
#lc=pd.DataFrame.from_dict(Counter(lc),orient='index')
#lc.columns=['Dominant forest cover frequency']
#lc=lc.append(pd.DataFrame(0,index=['mixed'],columns=['Dominant forest cover frequency']))
#lc/=lc.sum()
#
#fig, ax = plt.subplots(figsize=(3,3))
#plot=lc.plot.barh(ax=ax,legend=False)
#bar_label(lc['Dominant forest cover frequency'], ax)
#ax.set_xlim([0,1])
#ax.set_title('Dominant forest cover type, 25 km$^2$')

### map of woody cover and evergreen cover to see spatial gradients============

#lc=pd.read_excel('D:/Krishna/Project/working_tables.xlsx',\
#                 sheetname='gc_ever_deci',index_col=1) 
#lc=lc.loc[:,['evergreen','deciduous','mixed','woody']]
#param='evergreen'
#
#lats,lons = supply_lat_lon(landcover = 'GC_subset')
#
#latcorners=np.array([-70,70])
#loncorners=np.array([-180,180])
#latcorners=np.array([33,42.5])
#loncorners=np.array([-124.5,-117]) 
#
#enlarge=1
#cmap = 'Greens'
#sns.set_style('ticks')
#
#fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
#marker_size=get_marker_size_v2(ax,fig,loncorners, marker_factor=2.2)
#m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#                ax=ax)
#m.drawcoastlines()
#m.drawstates()
#m.drawmapboundary(fill_color='lightcyan',zorder=-1)
#m.fillcontinents(color='papayawhip',zorder=0)
#plot=m.scatter(lons, lats, s=marker_size,c=lc[param],cmap=cmap,\
#                    marker='s',vmin=0,vmax=1)
#ax.set_title('%s fraction cover'%param)
#cb=fig.colorbar(plot,ax=ax,ticks=[0,.25,.50,.75,1.00],pad=0.02) #
##cb.ax.annotate('fraction cover',xy=(0,1.03), xycoords='axes fraction',ha='left')


### Check to see if FAM is ocurring only in pure pixels========================

os.chdir(Dir_CA)
store=pd.HDFStore('data_subset_GC.h5')
start_year = 2009
end_year = 2017
x_param='deciduous_cover_fraction'
color_param='woody_cover_fraction'

x=store[x_param]
y=store['mortality_025_grid']
z=store[color_param]

x,y,z = select_years(start_year, end_year, x,y,z)
cmap='YlOrBr'
fig, ax = plt.subplots(figsize=(3,3))
plot=ax.scatter(x,y, c=z,cmap=cmap, marker='s', vmin=0, vmax=1, s=10)
ax.set_xlabel(x_param)
ax.set_ylabel('FAM')
ax.set_xlim([-0.05,1])

adjust_spines(ax,['left','bottom'])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(plot, cax=cax,ticks=[0,1])
cax.annotate('%s'%color_param,xy=(1.4,0.5), xycoords='axes fraction',\
            ha='left',rotation=90, va='center')
#    cax.set_yticklabels(['Low', 'High'])
#    cax.tick_params(axis='y', right='off',pad=0)



