# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:41:22 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
os.chdir(Dir_CA)
sns.set(font_scale=1.2)
mpl.rcParams['font.size'] = 15
store=pd.HDFStore('data.h5')

#inputs---------------------------------------------------------
data=(store['vod_pm']-store['vod_am'])/store['vod_am']
data=store['vod_005_grid']/store['LAI_005_grid']
grid_size=5
thresh=0.9
start_year=2009
start_month=7
data_label=r'$\frac{VOD_{ASCAT}}{LAI}$'
#-------------------------------------------------------------

mort=store['mortality_%03d_grid'%grid_size]
mort_main=mort.copy()
#data=store['vod_pm']
data=data.loc[(data.index.month>=start_month) & (data.index.month<=start_month+3)]
#data.loc[:,ind(thresh,mort)].plot(legend=False)
#data=store['vod_am']/store['LAI_025_grid']
data_anomaly=median_anomaly(data)
data_yearly=data.resample("A").mean()
#data_yearly.loc[:,ind(thresh,mort_main)].plot(legend=False)
#data_anomaly.loc[:,ind(thresh,mort)].plot(legend=False)

end_year=min(max(mort.index.year),max(data.index.year))
data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
                          (data_anomaly.index.year<=end_year)]
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]

year_range=mort.index.year
cols=mort.shape[0]
zoom=2
rows=2
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
data_min=np.nanmin(data_anomaly.iloc[:, :].values)
data_max=np.nanmax(data_anomaly.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
if grid_size==25:
    grids=Dir_mort+'/CA_proc.gdb/grid'
    marker_factor=8
elif grid_size==5:
    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
    marker_factor=8
marker_factor=2
lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
sns.set_style("white")
fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
                        sharey='row')
marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
plt.subplots_adjust(wspace=0.04,hspace=0.04)
for year in year_range:   
    mort_plot=mort[mort.index.year==year]
    ax=axs[0,year-year_range[0]]
    ax.set_title(str(year))
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_mort=m.scatter(lats, lons,s=1.5*marker_size,c=mort_plot,cmap='Reds'\
                        ,marker='s',vmin=0,vmax=1)
    data_plot=data_anomaly[data_anomaly.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_data=m.scatter(lats, lons,s=1.5*marker_size,c=data_plot,cmap='PuRd'\
                       ,marker='s',vmin=-1,vmax=data_max)
cb1=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb2=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)

axs[0,0].set_ylabel('Fractional area \n Mortality')
axs[1,0].set_ylabel(data_label+' \n anomaly')
fig.suptitle('Timeseries maps of mortality and '+data_label)

plt.show()

## all in one kde plots
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
plt.subplots_adjust(wspace=0.2)
mort.T.plot(kind='kde',legend=False,cmap='jet',ax=axs[0])
axs[0].set_ylabel('Density')
axs[0].set_xlabel('FAM')
l2=data_anomaly.loc[:,ind(thresh,mort)].T.plot(kind='kde',legend=True,cmap='jet',ax=axs[1],label=year_range)
axs[1].set_xlabel(data_label+' anomaly')
axs[1].set_ylabel('')
#axs[1].legend(data_anomaly.index.year)
plt.legend(labels=data_anomaly.index.year,bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
#axs[1].invert_xaxis()
fig.suptitle('Kernel density of high mortality regions')

### scatter plot
x=data_anomaly.loc[:,ind(thresh,mort_main)].values.flatten()
y=mort.loc[:,ind(thresh,mort_main)].stack().values.flatten()
non_nan_ind=np.argwhere(~np.isnan(x))
x=x[non_nan_ind]
y=y[non_nan_ind]

lm = linear_model.LinearRegression(fit_intercept=True)
lm.fit(x,y)

sns.set(font_scale=1.4)
fig, ax = plt.subplots(1,1,figsize=(4,4))
plt.scatter(x,y,color='b',alpha=0.6)
plt.plot(x,lm.predict(x),color='r',linewidth=2)
#ax.set_xlim([0,2.5])
ax.set_ylim([0,1.1])
ax.set_xlabel(data_label+' anomaly')
ax.set_ylabel('Fractional area mortality')
ax.set_title('FMA Vs '+data_label+' anomaly' )
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), \
            xycoords='axes fraction',fontsize=10)
ax.annotate('$R^2 = %.2f$'%lm.score(x,y), xy=(0.05, 0.8), \
            xycoords='axes fraction',color='r')
