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
#data2=data.copy()
grid_size=25
start_year=2009
start_month=7
months_window=3
data2_label='CWD annually \naccumulated'
data_label="VOD log\nanomaly"
#data2_label=data_label
cmap='viridis'
alpha=0.7
#species='evergreen'
#species='deciduous'
#mort_label='Dead trees\nper acre'
mort_label='Fractional area\nof mortality'

#----------------------------------------------------------------------
mort=store['mortality_%03d_grid'%(grid_size)]
mort=mort[mort>0]
mort_main=mort.copy()
data2_anomaly=data2
data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
data_anomaly=log_anomaly(data,start_month,months_window)
end_year=min(max(mort.index.year),max(data.index.year))
data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
                          (data_anomaly.index.year<=end_year)]
#data2_anomaly=data_anomaly
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
#(mort,data_anomaly,data2_anomaly)=mask_columns(ind_small_species(species),\
#                                 mort,data_anomaly,data2_anomaly)
year_range=mort.index.year
cols=mort.shape[0]
zoom=1.1
rows=3
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
data_min= np.nanmin(data_anomaly.iloc[:, :].values)
data_max= np.nanmax(data_anomaly.iloc[:, :].values)
data_max=1.2
data_min=0.9
data2_min=np.nanmin(data2_anomaly.iloc[:, :].values)
data2_max=np.nanmax(data2_anomaly.iloc[:, :].values)
tree_min=np.nanmin(mort.iloc[:, :].values)
tree_max=np.nanmax(mort.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
if grid_size==25:
    grids=Dir_mort+'/CA_proc.gdb/grid'
    marker_factor=7
    scatter_size=20
elif grid_size==5:
    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
    marker_factor=2
    scatter_size=4

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
    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap=cmap,\
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
    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap+'_r'\
                       ,marker='s',vmin=0.9*data_min,vmax=0.9*data_max,\
#                        norm=mpl.colors.LogNorm()\
                       )
    #-------------------------------------------------------------------
    data2_plot=data2_anomaly[data2_anomaly.index.year==year]
    ax=axs[2,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot2_data=m.scatter(lats, lons,s=marker_size,c=data2_plot,cmap=cmap\
                       ,marker='s',vmin=0.9*data2_min,vmax=0.9*data2_max)
cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb1=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb2=fig.colorbar(plot2_data,ax=axs[2,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
cb2.ax.text(1,0.9,'  (mm)',horizontalalignment='left',fontsize=12)
cb1.ax.invert_yaxis()
#cb1.set_ticks(np.linspace(0.2,0.8,4))
#cb1.set_ticklabels(np.linspace(0.2,0.8,4))
axs[0,0].set_ylabel(mort_label,rotation = 0,labelpad=50)
axs[1,0].set_ylabel(data_label,rotation = 0,labelpad=40)
axs[2,0].set_ylabel(data2_label,rotation = 0,labelpad=50)
fig.suptitle('Timeseries maps of mortality and indicators')
plt.show()

### scatter plot linear scale 
thresh=0.0
rep_times=1
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
plt.subplots_adjust(wspace=0.15)
ax=axs[0]
x=data_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y,rep_times,thresh)
plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
#ax.set_xlim([-3,3])
ax.invert_xaxis()
guess=(0.25,0.01,1e-1,1e-1)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y,guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                color='r',alpha=0.6)
ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
residuals = y- piecewise_linear(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
ax.annotate('$R^2 = $%.2f'%r_squared, xy=(0.05, 0.9), xycoords='axes fraction',\
            ha='left',color='r')
ax=axs[1]
x=data2_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y,rep_times,thresh)
plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
ax.set_xlabel(data2_label)
cbaxes = fig.add_axes([0.92, 0.6, 0.03, 0.2])
cb=fig.colorbar(plot2_data,ax=axs[1],\
                ticks=[min(z), max(z)],cax=cbaxes)
cb.ax.set_yticklabels(['Low', 'High'])
guess=(700,0.01,1e-4,1e-4)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                color='r',alpha=0.6)

ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
fig.suptitle('Scatter plot relating mortality with indicators')
cbaxes.text(0,1.2,'Density')
#ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left')