# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
sns.set(font_scale=1.2)
sns.set_style("white")
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2009,2017)    
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
mort.index=pd.to_datetime(mort.index,format='%Y')
mort_09_15=mort[(mort.index.year>=np.min(year_range))&(mort.index.year<=np.max(year_range))]
store = pd.HDFStore(Dir_CA+'/sigma0.h5')
sigma0=store['sigma0']
sigma0.index=pd.to_datetime(sigma0.index,format='%Y%j')
sigma0_09_15=sigma0[(sigma0.index.year>=np.min(year_range))&(sigma0.index.year<=np.max(year_range))]
store.close()

thresh=0.8
VOD_anomaly=min_anomaly(sigma0_09_15)
grids=Dir_mort+'/CA_proc.gdb/smallgrid'
grid_size=0.05
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
### map plots
latcorners=[33,42.5]
loncorners=[-124.5,-117]    
#latcorners=[37,38]
#loncorners=[-120,-119] 

cols=mort_09_15.shape[0]
fig_width=zoom*cols
fig_height=1.5*zoom*rows
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height))
plt.subplots_adjust(wspace=0.04,hspace=0.001)
marker_factor=1
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
for year in year_range:   
    mort_data_plot=mort_09_15[mort_09_15.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_data_plot,cmap='Reds',marker='s',vmin=0,vmax=1)
    
    VOD_data_plot=VOD_anomaly[VOD_anomaly.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    
    plot_vod=m.scatter(lats, lons,s=marker_size,c=VOD_data_plot,cmap='RdPu_r',marker='s',\
                       vmin=-2.5,vmax=-1)
cb1=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.0074,aspect=20,pad=0.02)
cb2=fig.colorbar(plot_vod,ax=axs[1,:].ravel().tolist(), fraction=0.0074,aspect=20,pad=0.02)
cb2.ax.invert_yaxis()
fig.text(0.1,0.67,'Fractional area \n Mortality',horizontalalignment='center',verticalalignment='center',rotation=90)
fig.text(0.1,0.33,'Min. $VOD_{ASCAT}$ \n anomaly',horizontalalignment='center',verticalalignment='center',rotation=90)
plt.show()

## all in one kde plots
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
plt.subplots_adjust(wspace=0.3,hspace=0.2)
mort_09_15.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[0])
axs[0].set_ylabel('Kernel Density \n of FAM')
axs[0].legend(mort_09_15.index.year)
VOD_anomaly[VOD_anomaly>=-100].T.plot(kind='kde',legend=True,cmap='jet',ax=axs[1])
axs[1].set_ylabel('Kernel Density \n of $VOD_{ASCAT}$')
axs[1].legend(VOD_anomaly.index.year)
axs[1].invert_xaxis()







#
#
###trial
#h = 0.05   # assuming equally spaced data-points
## you can use the colormap like this in your case:
#cmap = plt.cm.Blues_r
#plt.figure()
#ax=plt.gca()
#m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#        llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#        llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#        )
#m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
##m.readshapefile(Dir_CA+'/smallgrid','smallgrid',drawbounds=True, color='black')
#for x, y, c in zip(lats, lons, VOD_data_plot):
#    plot=ax.add_artist(Rectangle(xy=(x-h/2., y-h/2.), 
#                  color=cmap(c),             # or, in your case: color=cmap(c)                  
#                  
#                  width=h, height=h))  # Gives a square of area h*h
#plt.colorbar(plot)
#plt.show()
