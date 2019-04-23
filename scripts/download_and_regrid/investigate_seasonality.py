# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 16:58:19 2018

@author: kkrao
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.basemap import Basemap
from scipy import optimize
from matplotlib.patches import Rectangle, Patch
from dirs import Dir_CA, supply_lat_lon, get_marker_size
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

start_year=2009
end_year=2015
os.chdir(Dir_CA)
filename = 'data_subset_GC.h5'
data_source1='vod_pm'
store=pd.HDFStore(filename)
df=store[data_source1]
    
df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
df=df.rolling(30,min_periods=1).mean()
mean = df.groupby(df.index.dayofyear).mean() 

###=------------------- figures
zoom = 1.
year_range = range(start_year, end_year+1)
cols=len(year_range)
rows=1
latcorners=np.array([33,38.5])
loncorners=np.array([-120.5,-117])     
marker_factor=7*zoom
grid_size = 25

sns.set_style("ticks")

fig, axs = plt.subplots(nrows=rows,ncols=cols   ,\
                        sharey='row', figsize = (9*zoom,1.5*zoom))
plt.subplots_adjust(wspace=0.04,hspace=0.04)
marker_size=get_marker_size(axs[0],fig,loncorners,grid_size,marker_factor)
marker_size = 15
proj='cyl'
cmap = "magma_r"
image_path = os.path.join("D:\Krishna\Project\codes", "map-pin-red.png")
for year in year_range:
    d = df.loc[df.index.year==year,:]
    d.index = d.index.dayofyear
    R = d.corrwith(mean, axis = 0)
#    print('%0.2f'%R.loc[104])
    ax=axs[year-year_range[0]]
#        ax.set_title(str(year))
    ax.annotate(str(year), xy=(0.5, 1.03), xycoords='axes fraction',\
            ha='center',va='bottom')
    m = Basemap(projection=proj,lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    imscatter(lons[27], lats[27]+0.25, image_path, zoom=2e-2, ax=ax)
    plot_mort=m.scatter(lons, lats,s=marker_size,c=R,cmap=cmap,\
                        marker='s',\
                        vmin=0,vmax=1)
    
    
plt.show()
#    m.scatter(lons[27], lats[27],s=4,c='r',\
#                        marker='o')
    
cb0=fig.colorbar(plot_mort,ax=axs.ravel().tolist(), fraction=0.03,\
                 aspect=20,pad=0.02)
cb0.ax.annotate('R(VOD(t), S(t))',xy=(-5,1.06),xycoords='axes fraction',\
                    ha='left',va='bottom',size=10)