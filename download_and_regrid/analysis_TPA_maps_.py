# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:41:22 2017

@author: kkrao
"""
###response is number of dead trees
from dirs import *
os.chdir(Dir_CA)
sns.set(font_scale=1.2)
mpl.rcParams['font.size'] = 15
store=pd.HDFStore('data.h5')

#inputs
data2=(store['cwd'])
data=(store['vod_pm']) 
#data2=data
grid_size=25
thresh=0.3
start_year=2009
start_month=7
months_window=3
data2_label='CWD annually \naccumulated'
data_label="VOD \nanomaly"
mort_label='Dead trees\nper acre'

#----------------------------------------------------------------------


mort=store['TPA_%03d_grid'%grid_size]
mort=mort[mort>0]
#mort=mort.mask(mort==np.nan)
mort_main=mort.copy()
data2_anomaly=data2

data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
data_anomaly=year_anomaly_mean(data)



end_year=min(max(mort.index.year),max(data.index.year))
data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
                          (data_anomaly.index.year<=end_year)]
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]

year_range=mort.index.year
cols=mort.shape[0]
zoom=1.1
rows=3
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
data_min=np.nanmin(data_anomaly.iloc[:, :].values)
data_max=np.nanmax(data_anomaly.iloc[:, :].values)
data2_min=np.nanmin(data2_anomaly.iloc[:, :].values)
data2_max=np.nanmax(data2_anomaly.iloc[:, :].values)
tree_min=np.nanmin(mort.iloc[:, :].values)
tree_max=np.nanmax(mort.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
if grid_size==25:
    grids=Dir_mort+'/CA_proc.gdb/grid'
    marker_factor=7
elif grid_size==5:
    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
    marker_factor=4

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
    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap='PuRd',\
                        marker='s',\
                        vmin=0.9*tree_min,vmax=0.9*tree_max,\
                        norm=mpl.colors.LogNorm()\
                                               )
    #---------------------------------------------------------------
    data_plot=data_anomaly[data_anomaly.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap='PuRd_r'\
                       ,marker='s',vmin=0.9*data_min,vmax=0.9*data_max)
    #-------------------------------------------------------------------
    data2_plot=data2_anomaly[data2_anomaly.index.year==year]
    ax=axs[2,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot2_data=m.scatter(lats, lons,s=marker_size,c=data2_plot,cmap='PuRd'\
                       ,marker='s',vmin=0.9*data2_min,vmax=0.9*data2_max)
cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb1=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb2=fig.colorbar(plot2_data,ax=axs[2,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
#cb2.ax.set_ylabel('(mm)',rotation = 0)
cb2.ax.text(1,0.9,'  (mm)',horizontalalignment='left',fontsize=12)
cb1.ax.invert_yaxis()
axs[0,0].set_ylabel(mort_label,rotation = 0,labelpad=50)
axs[1,0].set_ylabel(data_label,rotation = 0,labelpad=30)
axs[2,0].set_ylabel(data2_label,rotation = 0,labelpad=50)
fig.suptitle('Timeseries maps of mortality and indicators')
plt.show()

## all in one kde plots
thresh=5
sns.set_style("darkgrid")
fig= plt.figure(figsize=(8,4))
ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=1)
ax2 = plt.subplot2grid((1, 5), (0, 1), colspan=2)
ax3 = plt.subplot2grid((1, 5), (0, 3), colspan=2)
plt.subplots_adjust(wspace=0.4)
ax=ax3
data2_anomaly.loc[:,ind(thresh,mort_main)].T.plot(kind='kde',legend=False,cmap='jet',ax=ax)
ax.set_xlabel(data2_label)
ax.set_ylabel('')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
ax=ax2
data_anomaly.loc[:,ind(thresh,mort_main)].T.plot(kind='kde',legend=False,cmap='jet',ax=ax)
ax.set_xlabel(data_label)
ax.invert_xaxis()
ax.set_ylabel('')
plt.legend(labels=data_anomaly.index.year,bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
ax=ax1
mort.loc[:,ind(thresh,mort_main)].T.plot(kind='kde',legend=False,cmap='jet',ax=ax)
ax.set_xlabel(mort_label)
ax.set_xlim([-0.1*thresh,5*thresh])
ax.set_ylim([0,1])
ax.set_ylabel('Density',labelpad=20,rotation = 0)
#ax.set_yscale('log')
fig.suptitle('Distribution of indicators in Southern Sierra')

### scatter plot log scale
thresh=-np.inf
rep_times=1
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
plt.subplots_adjust(wspace=0.15)
ax=axs[0]
ax.set_yscale('log')
x=data_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,np.log10(y),rep_times,thresh)
plot_data=ax.scatter(x,10**y,c=z,edgecolor='',cmap='jet',alpha=0.3)
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
ax.set_xlim([-3,3])
ax.invert_xaxis()

ax=axs[1]
ax.set_yscale('log')
x=data2_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,np.log10(y),rep_times,thresh)
plot2_data=ax.scatter(x,10**y,c=z,edgecolor='',cmap='jet',alpha=0.3)
ax.set_xlabel(data2_label)
cbaxes = fig.add_axes([0.92, 0.6, 0.03, 0.2])
cb=fig.colorbar(plot2_data,ax=axs[1],\
                ticks=[min(z), max(z)],cax=cbaxes)
cb.ax.set_yticklabels(['Low', 'High'])
fig.suptitle('Scatter plot relating mortality with indicators')
cbaxes.text(0,1.2,'Density')

### scatter plot linear scale scale
thresh=0.0
rep_times=1
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
plt.subplots_adjust(wspace=0.15)
ax=axs[0]
x=data_anomaly.loc[:,ind(thresh,mort_main)].values.flatten()
y=mort.loc[:,ind(thresh,mort_main)].values.flatten()
x,y,z=clean_xy(x,y,rep_times,thresh)
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
ax.set_xlim([-3,3])
ax.invert_xaxis()


ax=axs[1]
x=data2_anomaly.loc[:,ind(thresh,mort_main)].values.flatten()
y=mort.loc[:,ind(thresh,mort_main)].values.flatten()
x,y,z=clean_xy(x,y,rep_times,thresh)
plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap='jet',alpha=0.3)
ax.set_xlabel(data2_label)
cbaxes = fig.add_axes([0.92, 0.6, 0.03, 0.2])
cb=fig.colorbar(plot2_data,ax=axs[1],\
                ticks=[min(z), max(z)],cax=cbaxes)
cb.ax.set_yticklabels(['Low', 'High'])
fig.suptitle('Scatter plot relating mortality with indicators')
cbaxes.text(0,1.2,'Density')
### scatter plot
#thresh=0.03
#sns.set_style("darkgrid")
#fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
#plt.subplots_adjust(wspace=0.15,top=0.85)
#ax=axs[0]
#x=data_anomaly.values.flatten()
#y=mort.values.flatten()
#high_mort_ind=np.where(y>=thresh)
#x=x[high_mort_ind]; y=y[high_mort_ind]
#non_nan_ind=np.where(~np.isnan(x))
#x=x[non_nan_ind];   y=y[non_nan_ind]
#x.shape=(len(x),1); y.shape=(len(y),1)
##x=np.repeat(x,2);   y=np.repeat(y,2)
#lm.fit(x,y)
#plot_data=ax.scatter(x,y,color='b',alpha=0.5)
#ax.plot(x, lm.predict(x), color='r',linewidth=2)
#ax.set_xlabel(data_label)
#ax.set_ylabel(mort_label)
#ax.invert_xaxis()
#ax.annotate('$R^2 = %.2f$'%lm.score(x,y), xy=(0.05, 0.8), \
#            xycoords='axes fraction',color='r')
#-------------------------------------------------------------------
#ax=axs[1]
#x=data2_anomaly.values.flatten()
#y=mort.values.flatten()
#high_mort_ind=np.where(y>=thresh)
#x=x[high_mort_ind]; y=y[high_mort_ind]
#non_nan_ind=np.where(~np.isnan(x))
#x=x[non_nan_ind];   y=y[non_nan_ind]
#x.shape=(len(x),1); y.shape=(len(y),1)
##x=np.repeat(x,2);   y=np.repeat(y,2)
#lm.fit(x,y)
#plot2_data=ax.scatter(x,y,color='b',alpha=0.5)
#ax.plot(x, lm.predict(x), color='r',linewidth=2)
#ax.set_xlabel(data2_label)
#ax.annotate('$R^2 = %.2f$'%lm.score(x,y), xy=(0.05, 0.8), \
#            xycoords='axes fraction',color='r')
#
#fig.suptitle('Relationship of mortality with indicators \nfor high mortality regions')
#

#### box plot
thresh=0.7
boxes=16
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
plt.subplots_adjust(wspace=0.15,top=0.85)
ax=axs[0]
yb=box_equal_nos(data_anomaly,mort,boxes,thresh)
yb.boxplot(ax=ax,fontsize=9,rot=90)
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
#ax.xticks(rotation='vertical')
ax.invert_xaxis()
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#-------------------------------------------------------------------
ax=axs[1]
yb=box_equal_nos(data2_anomaly,mort,boxes,thresh)
yb.boxplot(ax=ax,fontsize=9,rot=90)
ax.set_xlabel(data2_label)
#ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
fig.suptitle('Relationship of mortality with indicators \nfor high mortality regions')

