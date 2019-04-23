# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
sns.set_style("white")
### map plots
latcorners=[33,42.5]
loncorners=[-124.5,-117]    
#latcorners=[37,38]
#loncorners=[-120,-119] 
zoom=2
rows=2


##cwd
store = pd.HDFStore(Dir_CA+'/Young_Df.h5') 
cwd=store['cwd_acc']
cwd.index.name='gridID'
cwd=cwd.T
cwd.drop('gridID',inplace=True)
cwd.index=pd.to_datetime(cwd.index,format='%Y')
store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort.index.name='gridID'
mort=mort.T
mort.drop('gridID',inplace=True)
mort.index=[x[-4:] for x in mort.index] 
mort.index=pd.to_datetime(mort.index)
mort_09_15=mort[(mort.index.year>=2009)&(mort.index.year<=2015)]
store.close()
year_range=mort_09_15.index.year
cols=mort_09_15.shape[0]
zoom=2
rows=2
fig_width=zoom*cols
fig_height=1.5*zoom*rows

grids=Dir_mort+'/CA_proc.gdb/grid'
grid_size=0.25
marker_factor=2
cwd_max=np.nanmax(cwd.iloc[:, :].values)
cwd_min=np.nanmin(cwd.iloc[:, :].values)
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height))
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
plt.subplots_adjust(wspace=0.04,hspace=0.001)
for year in year_range:   
    mort_data_plot=mort_09_15[mort_09_15.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=1.5*marker_size,c=mort_data_plot,cmap='Reds'\
                        ,marker='s',vmin=0,vmax=0.4)
    
    cwd_data_plot=cwd[cwd.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_cwd=m.scatter(lats, lons,s=1.5*marker_size,c=cwd_data_plot,cmap='RdPu'\
                       ,marker='s',vmin=400,vmax=cwd_max)
cb1=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.008,aspect=20,pad=0.02)
cb2=fig.colorbar(plot_cwd,ax=axs[1,:].ravel().tolist(), fraction=0.008,aspect=20,pad=0.02)
#cb2.ax.invert_yaxis()
fig.text(0.1,0.67,'Fractional area \n Mortality',horizontalalignment='center',verticalalignment='center',rotation=90)
fig.text(0.1,0.33,'CWD annual\n accumulated',horizontalalignment='center',verticalalignment='center',rotation=90)
plt.show()

#KDE
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
                        sharey='row')
plt.subplots_adjust(wspace=0.5,hspace=0.2)
for year in year_range:   
    mort_data_plot=mort_09_15[mort_09_15.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    plot_mort=mort_data_plot.T.plot(kind='kde',ax=ax,legend=False,color='r')  
    ax.set_ylabel('Kernel Density \n of FAM')
    ax.set_xlim([-0.1,0.4])
    cwd_data_plot=cwd[cwd.index.year==year]
    ax=axs[1,year-year_range[0]]
    plot_cwd=cwd_data_plot.T.plot(kind='kde',ax=ax,legend=False,color='g')
    ax.set_ylabel('Kernel Density \n of CWD')
    ax.set_xlim([0,cwd_max*1.3])
plt.show()
#np.nanmin(mort_09_15.iloc[:, :].values)

## all in one kde plots
sns.set_style("darkgrid")
mort=mort_09_15
data=cwd
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
plt.subplots_adjust(wspace=0.45,hspace=0.2)
mort.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[0])
axs[0].set_ylabel('Kernel Density \n of FAM')
axs[0].legend(mort.index.year)
data.T.plot(kind='kde',legend=True,cmap='jet',ax=axs[1])
axs[1].set_ylabel('Kernel Density \n of CWD')
axs[1].legend(data.index.year)
#axs[1].invert_xaxis()


