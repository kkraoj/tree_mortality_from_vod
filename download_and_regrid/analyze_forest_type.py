# -*- coding: utf-8 -*-
"""
Created on Sun Apr 08 19:30:25 2018

@author: kkrao
"""
import sys
import os
import arcpy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dirs import Dir_mort, Dir_CA, build_df_from_arcpy, subset_forest_cov,\
                select_years, initiliaze_plot, scatter_threshold, select_forest_type_grids,\
                supply_lat_lon
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.basemap import Basemap


os.chdir(Dir_mort)
arcpy.env.workspace=Dir_mort+'/grid.gdb'

lookup = build_df_from_arcpy(Dir_mort+'/ADS2016.gdb/host_typ', ['CODE','FOREST_TYPE'], dtype='str', index_col = 'CODE')
lookup.index =lookup.index.astype('int16')
cardinal='#BD2031'
classes = 3
fs=14
publishable = initiliaze_plot(fs = 14)
store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
fortype = store['mortality_025_forest_type']


#### get mortality forest type of all grids for each year =====================
#
#year_range = range(2005, 2017)
#Df_master=pd.DataFrame(index = pd.to_datetime(year_range, format = '%Y'), columns = range(370),)
#Df_master.loc[:,:] = 0
#Df_master.columns.name = 'gridID'
#Df_master.index.name = 'Forest Type' 
#Df_master.head()
#
#for year in year_range:
#    sys.stdout.write('\r'+'Processing data for year:%s ...'%(year))
#    sys.stdout.flush()   
#    
#    table='ADS%s_i_j'%str(year)[-2:]
#    Df=build_df_from_arcpy(table, ['gridID', 'Shape_Area', 'FOR_TYPE1', 'SURVEY_ID2'], dtype=str)
#    Df.gridID -= 1
#    Df=Df.groupby(['gridID','FOR_TYPE1']).Shape_Area.sum().reset_index()
#    Df.index = Df.FOR_TYPE1
#    for_type=Df.groupby('gridID').Shape_Area.idxmax()
#    for_type=pd.DataFrame([for_type], index = [pd.to_datetime(year, format = '%Y')])
#    Df_master.loc[Df_master.index.year==year]+=for_type
#
#Df_master=subset_forest_cov(Df_master)
#store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
##store['mortality_025_forest_type'] = Df_master
##store.close()
##
### plot unique forest type mortality oberservations for major type============

Df_master = store['mortality_025_forest_type']
mort = store['mortality_025_grid']
Df_master=Df_master[mort>=0.1]
Df_master = select_years(2009, 2015, Df_master)

#
unique = pd.value_counts(Df_master.values.ravel())
unique.index=lookup.loc[unique.index].FOREST_TYPE
                       
plt.style.use('seaborn-ticks')
fig, ax = plt.subplots(figsize = (4,4))
unique.plot(legend = False, ax=ax, kind = 'bar', color = 'grey')
ax.set_ylabel('Frequency')
ax.set_xlabel('Majority mortality forest type')


## plot FAM Vs RWC for different categories==================================
#
#x=store['cwd']
#y=store['mortality_025_grid']

#x,y = select_years(2009, 2015, x, y)
#FAM_thresh = 0.1
##y = y[y>=FAM_thresh]
#
#zoom = 1
#alpha = 1
#cmap = 'viridis'

#sns.set_style("ticks")
#publishable.set_figsize(1*zoom, classes*zoom, aspect_ratio = 1)
#plt.subplots_adjust(hspace=0.25)
#
#fig, axs = plt.subplots(classes, 1 , sharex = True)
#
#for (ax, forest) in zip(axs, [2122, 9001, 3015, 1202, 6631, 3020]):
#    
#    xs, ys = select_forest_type_grids(forest, fortype, x,y)
#    plot_data, z = scatter_threshold(xs,ys, ax, lookup.loc[forest].FOREST_TYPE, FAM_thresh = 0.,\
#                                     guess = (600,0.05,1e-1,1e-1), x_range = [200, 1200])
#
#    #### 4, 5, 6 rank are 1202, 6631, 3020
#ax.set_xlabel('CWD (mm)')
#cbaxes = fig.add_axes([0.95, 0.123, 0.06, 0.76])
#cb=fig.colorbar(plot_data,ax=axs[1],\
#                ticks=[min(z), max(z)],cax=cbaxes)
#cb.ax.set_yticklabels(['Low', 'High'])
#cb.ax.tick_params(axis='y', right='off',pad=4)
#cbaxes.annotate('Point\ndensity',xy=(0,1.02), xycoords='axes fraction',\
#            ha='left')

#publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
#                                             prefix = '', suffix = '.', fontweight = 'bold')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax.annotate('Dry', xy=(0.9, -0.28), xycoords='axes fraction',color=cardinal)
#ax.annotate('Wet', xy=(0., -0.28), xycoords='axes fraction',color='dodgerblue')

## plot spatial grids of forest types==========================================

#forest_range = [2122, 9001, 3015, np.nan]
#year_range = range(2009, 2016)
#fortype=select_years(2009,2015,fortype)
#rows, cols = 1, fortype.shape[0]
#latcorners=np.array([33,42.5])
#loncorners=np.array([-124.5,-117]) 
#lats,lons=supply_lat_lon()
#
#sns.set_style("ticks")
#zoom =1.5
#publishable.set_figsize(2*zoom, 0.5*zoom, aspect_ratio = 1)
#green = '#1b9e77'
#brown = '#d95f02'
#blue = '#7570b3'
#white = 'white'
#colors=[green, brown, blue, white]
#
#proj = 'cyl'
#fig, axs = plt.subplots(nrows=rows,ncols=cols   ,\
#                        sharey='row')
#plt.subplots_adjust(wspace=0.04,hspace=0.04,top=0.83)
##
#for year in year_range:   
#    ax=axs[year-year_range[0]]
#    ax.annotate(str(year), xy=(0.5, 1.03), xycoords='axes fraction',\
#            ha='center',va='bottom')
#    m = Basemap(projection=proj,lat_0=45,lon_0=0,resolution='l',\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#            ax=ax)
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#    z=fortype[fortype.index.year==year]  
#    z= z[fortype.isin(forest_range)]
#    z.replace(forest_range, colors, inplace = True)
#    m.scatter(lons, lats,s=8,c=z.values[0],marker='s')
#i=0
#patches = forest_range[:3]
#for (forest, color) in zip(forest_range[:3], colors):
#    patches[i] = mpatches.Patch(color=color, label=lookup.loc[forest].FOREST_TYPE)
#    i+=1
#plt.legend(handles =patches, bbox_to_anchor = [0.7,0], ncol = 3, handletextpad = 1)
##
#ax.annotate('California', xy=(0.1, 0), xycoords='axes fraction',\
#                ha='left',va='bottom',size=fs*0.7)
#plt.show()