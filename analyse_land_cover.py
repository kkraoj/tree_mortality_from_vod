# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:35:03 2018

@author: kkrao
"""

from __future__ import division
import os
from dirs import MyDir, make_lat_lon
#from pyhdf.SD import SD, SDC
import h5py
import tables
import gdal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


os.chdir(MyDir+'/Forest')
file_name = 'MCD12C1.A2012001.051.2013178154403.hdf'
file_name = 'MCD12C1.A2001001.051.2014274170618.hdf'
ds = gdal.Open('HDF4_SDS:UNKNOWN:"MCD12C1.A2001001.051.2014274170618.hdf":6')
gdal_dataset = gdal.Open(file_name)
gdal_dataset.GetSubDatasets()
lc_data = gdal.Open('HDF4_EOS:EOS_GRID:"MCD12C1.A2001001.051.2014274170618.hdf":MOD12C1:Land_Cover_Type_1_Percent')
#lc = lc_data.ReadAsArray()
#np.save('2001_lc_pc',lc)
#gdal_dataset.GetGeoTransform()


def get_marker_size(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    width *= fig.dpi
    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
    return marker_size

lc=np.load('2001_lc_pc.npy')
sum_frac=np.sum(lc,axis=0)


lc_dict={0:'Water',\
         1:'Evergreen Needleleaf',\
         2:'Evergreen Broadleaf forest',\
         3:'Deciduous Needleleaf forest',\
         4:'Deciduous Broadleaf forest',\
         5:'Mixed forest',\
         6:'Closed Shrublands',\
         7:'Open Shrublands',\
         8:'Woody savannas',\
         9:'Savannas'}
lc_class=9
data=lc[lc_class, :,:]

lats,lons = make_lat_lon(lc_data.GetGeoTransform())
Df=pd.DataFrame(data,index=lats[:,0],columns=lons[1,:])
#lats, lons = np.meshgrid(lats,lons,indexing='ij')

param='%s \n %% cover'%lc_dict[lc_class]
latcorners=np.array([-70,70])
loncorners=np.array([-180,180])
latcorners=np.array([33,42.5])
loncorners=np.array([-124.5,-117]) 

Df=Df.loc[(Df.index>=latcorners[0]) & (Df.index <=latcorners[1])]
Df=Df.loc[:,(Df.columns >=loncorners[0]) & (Df.columns <=loncorners[1])]

enlarge=1
lats, lons = np.meshgrid(Df.index,Df.columns,indexing='ij')
cmap = 'Greens'
sns.set_style('ticks')

fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
marker_size=get_marker_size(ax,fig,loncorners)
m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                ax=ax)
m.drawcoastlines()
m.drawstates()
m.drawmapboundary(fill_color='lightcyan',zorder=-1)
m.fillcontinents(color='papayawhip',zorder=0)
plot=m.scatter(lons, lats, s=marker_size,c=Df,cmap=cmap,\
                    marker='s',vmin=0,vmax=100)
ax.set_title('%s'%lc_dict[lc_class])
cb=fig.colorbar(plot,ax=ax,ticks=[0,25,50,75,100],pad=0.02) #
cb.ax.annotate('% cover',xy=(0,1.03), xycoords='axes fraction',ha='left')

#plt.imshow ( data, interpolation='nearest', vmin=0, cmap='Greys')
#plt.colorbar()
#h5file = tables.open_file(file_name)
#h5py.File(file_name, 'r')
#file = SD(file_name, SDC.READ)
#print file.info()