# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:35:03 2018

@author: kkrao
"""

from __future__ import division
import os
from dirs import MyDir, make_lat_lon, bar_label
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


os.chdir(MyDir+'/Forest')
file_name = 'gc_2009_clip'

gdal_dataset = gdal.Open(file_name)
gdal_dataset.GetSubDatasets()
#lc_data = gdal.Open('HDF4_EOS:EOS_GRID:"MCD12C1.A2001001.051.2014274170618.hdf":MOD12C1:Land_Cover_Type_1_Percent')
lc = gdal_dataset.ReadAsArray()
np.save('gc_2009_lc.npy',lc)
gdal_dataset.GetGeoTransform()
#
#

def subset_lc(lc,classes):
    mask=~np.isin(lc,classes)
#    data=np.ma.masked_array(lc, mask=mask)
    lc[mask]=255
    return lc
    
classes=[50,60,70,90,100]

data=subset_lc(lc,classes)
unique, counts = np.unique(data, return_counts=True)
lc_sum=dict(zip(unique, counts))
del(lc_sum[255])

### fraction of mixed forest in proportion to total forest
##forest is defined as classes 50,60,70,90,100
mixed_fraction = lc_sum[100]/sum(lc_sum.values())
lc_frac=dict((k, v/sum(lc_sum.values())) for k,v in lc_sum.items())
lc_labels = pd.read_excel(MyDir+'/Forest'+'/GLOBCOVER/Globcover2009_Legend.xls',header=0,index_col=0)
lc_labels.Label=lc_labels.Label.str.wrap(40)


lc_frac=pd.DataFrame(lc_frac.values(),index = lc_labels.loc[lc_frac.keys(),'Label'],columns = ['fraction coverage'])

mpl.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(3,3))
plot=lc_frac.plot.barh(legend=False,ax=ax)


bar_label(lc_frac['fraction coverage'],ax, x_offset=0.02)
#labels=ax.get_yticklabels()
#labels = [ '\n'.join(wrap(l, 20)) for l in labels ]
#ax.set_xlabels(labels)













#lc[data]

    
#def get_marker_size(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
#    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#    width = bbox.width
#    width *= fig.dpi
#    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
#    return marker_size
#
#lc=np.load('gc_2009_lc.npy')
#sum_frac=np.sum(lc,axis=0)
#
#
#lc_dict={0:'Water',\
#         1:'Evergreen Needleleaf',\
#         2:'Evergreen Broadleaf forest',\
#         3:'Deciduous Needleleaf forest',\
#         4:'Deciduous Broadleaf forest',\
#         5:'Mixed forest',\
#         6:'Closed Shrublands',\
#         7:'Open Shrublands',\
#         8:'Woody savannas',\
#         9:'Savannas'}
#lc_class=9
#data=lc[lc_class, :,:]
#
lats,lons = make_lat_lon(gdal_dataset.GetGeoTransform())
Df=pd.DataFrame(data,index=lats[:,0],columns=lons[1,:])
##lats, lons = np.meshgrid(lats,lons,indexing='ij')
#
#param='%s \n %% cover'%lc_dict[lc_class]
#latcorners=np.array([-70,70])
#loncorners=np.array([-180,180])
#latcorners=np.array([33,42.5])
#loncorners=np.array([-124.5,-117]) 
#
#Df=Df.loc[(Df.index>=latcorners[0]) & (Df.index <=latcorners[1])]
#Df=Df.loc[:,(Df.columns >=loncorners[0]) & (Df.columns <=loncorners[1])]
#
#enlarge=1
#lats, lons = np.meshgrid(Df.index,Df.columns,indexing='ij')
#cmap = 'Greens'
#sns.set_style('ticks')
#
#fig, ax = plt.subplots(figsize=(8*enlarge,5*enlarge))
#marker_size=get_marker_size(ax,fig,loncorners)
#m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#                ax=ax)
#m.drawcoastlines()
#m.drawstates()
#m.drawmapboundary(fill_color='lightcyan',zorder=-1)
#m.fillcontinents(color='papayawhip',zorder=0)
#plot=m.scatter(lons, lats, s=marker_size,c=Df,cmap=cmap,\
#                    marker='s',vmin=0,vmax=100)
#ax.set_title('%s'%lc_dict[lc_class])
#cb=fig.colorbar(plot,ax=ax,ticks=[0,25,50,75,100],pad=0.02) #
#cb.ax.annotate('% cover',xy=(0,1.03), xycoords='axes fraction',ha='left')
#
##plt.imshow ( data, interpolation='nearest', vmin=0, cmap='Greys')
##plt.colorbar()
##h5file = tables.open_file(file_name)
##h5py.File(file_name, 'r')
##file = SD(file_name, SDC.READ)
##print file.info()