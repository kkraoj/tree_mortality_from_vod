# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:41:22 2017

@author: kkrao
"""
###response is number of dead trees
from __future__ import division
from dirs import *
os.chdir(Dir_CA)
sns.set(font_scale=1.2)
mpl.rcParams['font.size'] = 15
#store=pd.HDFStore('data.h5')
#
##inputs
#data2=(store['cwd'])
#data=(store['vod_pm']) 
##data2=data.copy()
#grid_size=25
#start_year=2009
#start_month=7
#months_window=3
#data2_label='CWD annually \naccumulated'
#data_label="Relative water\ncontent"
##data2_label=data_label
#cmap='viridis'
#alpha=0.7
##species='evergreen'
##species='deciduous'
##mort_label='Dead trees\nper acre'
#mort_label='Fractional area\nof mortality'
#
##----------------------------------------------------------------------
#mort=store['mortality_%03d_grid'%(grid_size)]
#mort=mort[mort>0]
#mort_main=mort.copy()
#data2_anomaly=data2
#data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
#data_anomaly=RWC(data)
#end_year=min(max(mort.index.year),max(data.index.year))
#data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
#                          (data_anomaly.index.year<=end_year)]
##data2_anomaly=data_anomaly
#mort=mort[(mort.index.year>=start_year) &\
#          (mort.index.year<=end_year)]
##(mort,data_anomaly,data2_anomaly)=mask_columns(ind_small_species(species),\
##                                 mort,data_anomaly,data2_anomaly)
#year_range=mort.index.year
#cols=mort.shape[0]
#zoom=1.1
#rows=3
#latcorners=[33,42.5]
#loncorners=[-124.5,-117] 
#data_min= np.nanmin(data_anomaly.iloc[:, :].values)
#data_max= np.nanmax(data_anomaly.iloc[:, :].values)
#data2_min=np.nanmin(data2_anomaly.iloc[:, :].values)
#data2_max=np.nanmax(data2_anomaly.iloc[:, :].values)
#tree_min=np.nanmin(mort.iloc[:, :].values)
#tree_max=np.nanmax(mort.iloc[:, :].values)
#fig_width=zoom*cols
#fig_height=1.5*zoom*rows
#if grid_size==25:
#    grids=Dir_mort+'/CA_proc.gdb/grid'
#    marker_factor=7
#    scatter_size=20
#elif grid_size==5:
#    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
#    marker_factor=2
#    scatter_size=4
#
#lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
#lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
#sns.set_style("white")
#fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
#                        sharey='row')
#marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
#plt.subplots_adjust(wspace=0.04,hspace=0.04)
#for year in year_range:   
#    mort_plot=mort[mort.index.year==year]
#    ax=axs[0,year-year_range[0]]
#    ax.set_title(str(year))
#    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#            ax=ax)
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#    plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap=cmap,\
#                        marker='s',\
#                        vmin=0.9*tree_min,vmax=0.9*tree_max,\
#                        norm=mpl.colors.LogNorm()\
#                                               )
#    #---------------------------------------------------------------
#    data_plot=data_anomaly[data_anomaly.index.year==year]
#    ax=axs[1,year-year_range[0]]
#    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#            ax=ax)
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap\
#                       ,marker='s',vmin=0.9*data_min,vmax=0.9*data_max)
#    #-------------------------------------------------------------------
#    data2_plot=data2_anomaly[data2_anomaly.index.year==year]
#    ax=axs[2,year-year_range[0]]
#    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#            ax=ax)
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#    plot2_data=m.scatter(lats, lons,s=marker_size,c=data2_plot,cmap=cmap\
#                       ,marker='s',vmin=0.9*data2_min,vmax=0.9*data2_max)
#cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
#                 aspect=20,pad=0.02)
#cb1=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
#                 aspect=20,pad=0.02)
#cb2=fig.colorbar(plot2_data,ax=axs[2,:].ravel().tolist(), fraction=0.01,\
#                 aspect=20,pad=0.02)
#cb2.ax.text(1,0.9,'  (mm)',horizontalalignment='left',fontsize=12)
#cb1.set_ticks(np.linspace(0.2,0.8,4))
#cb1.set_ticklabels(np.linspace(0.2,0.8,4))
#axs[0,0].set_ylabel(mort_label,rotation = 0,labelpad=50)
#axs[1,0].set_ylabel(data_label,rotation = 0,labelpad=50)
#axs[2,0].set_ylabel(data2_label,rotation = 0,labelpad=50)
#fig.suptitle('Timeseries maps of mortality and indicators')
#plt.show()
#
#### scatter plot linear scale 
#thresh=0.0
#rep_times=1
#sns.set_style("darkgrid")
#fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
#plt.subplots_adjust(wspace=0.15)
#ax=axs[0]
#x=data_anomaly.values.flatten()
#y=mort.values.flatten()
#x,y,z=clean_xy(x,y,rep_times,thresh)
#plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
#ax.set_xlabel(data_label)
#ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
##ax.set_xlim([-3,3])
#ax.invert_xaxis()
#guess=(0.25,0.01,1e-1,1e-1)
#popt , pcov = optimize.curve_fit(piecewise_linear, x, y,guess)
#perr = np.sqrt(np.diag(pcov))
#xd = np.linspace(min(x), max(x), 1000)
#ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
#ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
#                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
#                                color='r',alpha=0.6)
#ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
#ymin,ymax=ax.get_ylim()
#ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
#                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
#ax.set_ylim([ymin,ymax])
##ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
##            ha='left')
#residuals = y- piecewise_linear(x, *popt)
#ss_res = np.sum(residuals**2)
#ss_tot = np.sum((y-np.mean(y))**2)
#r_squared = 1 - (ss_res / ss_tot)
#ax.annotate('$R^2 = $%.2f'%r_squared, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left',color='r')
#ax=axs[1]
#x=data2_anomaly.values.flatten()
#y=mort.values.flatten()
#x,y,z=clean_xy(x,y,rep_times,thresh)
#plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
#ax.set_xlabel(data2_label)
#cbaxes = fig.add_axes([0.92, 0.6, 0.03, 0.2])
#cb=fig.colorbar(plot2_data,ax=axs[1],\
#                ticks=[min(z), max(z)],cax=cbaxes)
#cb.ax.set_yticklabels(['Low', 'High'])
#guess=(700,0.01,1e-4,1e-4)
#popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
#perr = np.sqrt(np.diag(pcov))
#xd = np.linspace(min(x), max(x), 1000)
#ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
#ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
#                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
#                                color='r',alpha=0.6)
#
#ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
#ymin,ymax=ax.get_ylim()
#ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
#                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
#ax.set_ylim([ymin,ymax])
#fig.suptitle('Scatter plot relating mortality with indicators')
#cbaxes.text(0,1.2,'Density')
#ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left')

##---------------------------------------------------------------------------
#
#"""
#RWC detail plot.
#plotting only 2015 RWC for different percentiles of RWC definition
#"""
#rows=1
#cols=6
#fig_width=zoom*cols
#fig_height=1.5*zoom*rows
#fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
#                        sharey='row')
#marker_size=get_marker_size(axs[0],fig,loncorners,grid_size,marker_factor)
#plt.subplots_adjust(wspace=0.04,hspace=0.04,top=0.8,bottom=0)
#for col in range(cols):   
#    #---------------------------------------------------------------
#    upper_quantile=0.95+col/100
#    data_anomaly=RWC_detail(data,upper_quantile)
#    data_plot=data_anomaly[data_anomaly.index.year==2015]
#    ax=axs[col]
#    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#            ax=ax)
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#    plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap\
#                       ,marker='s',vmin=0,vmax=1)
#    ax.set_xlabel('%1.2f'%upper_quantile,fontsize=10)
#    #-------------------------------------------------------------------
#cb1=fig.colorbar(plot_data,ax=axs.tolist(), fraction=0.01,\
#                 aspect=20,pad=0.02)
##cb1.ax.invert_yaxis()
#cb1.set_ticks(np.linspace(0.2,0.8,4))
#cb1.set_ticklabels(np.linspace(0.2,0.8,4))
#axs[0].set_ylabel(data_label,rotation = 0,labelpad=50)
#fig.suptitle('2015 maps of RWC')
#axs[0].annotate('upper cutoff percentile =', xy=(-0.1, -0.13), xycoords='axes fraction',\
#            ha='right',fontsize=10)
#plt.show()
##------------------------------------------------------------------
#"""
#script to output values of RWC details for nan cells when upper limit 
#set to 0.95
#"""
#VOD_median=data[data.index.year==2015].quantile(0.5).loc[360]
#cols=5
#Df=pd.DataFrame()
#for col in range(cols): 
#    upper_quantile=0.99-col/100
#    upper_cutoff=data.quantile(upper_quantile).loc[360]
#    lower_cutoff=data.quantile(1-upper_quantile).loc[360]
#    RWC_value=RWC_detail(data,upper_quantile)[data_anomaly.index.year==2015].\
#                        loc['2015-01-01',360]
#    output_values=[VOD_median,upper_cutoff,lower_cutoff,RWC_value]
#    output_values=pd.DataFrame(data=output_values,\
#                   columns=['%1.2f'%upper_quantile],\
#                   index=['Season median','upper cutoff','lower cutoff','RWC'])
#    Df=pd.concat([Df,output_values],axis='columns') 
#Df.columns.name='upper cutoff percentile' 
#display(HTML(data.head().to_html()))
#sns.set_style('darkgrid')  
#alpha=0.5
#fig, ax = plt.subplots(1,1,figsize=(3,3))                 
#data.loc[:,360].hist(normed=True,ax=ax,alpha=alpha,label='all years',bins=40)
#data[data.index.year==2015].loc[:,360].hist(normed=True,ax=ax,alpha=alpha,label='2015')
#ax.set_xlabel('VOD')
#ax.set_ylabel('Normalized\nfrequency')
#ax.set_title('Distribution of VOD for sample grid cell')
#plt.legend()
#
#
##------------------------------------------------------------------
#alpha=0.5
#rows=1
#cols=mort.shape[0]
#fig_width=zoom*cols
#fig_height=1.5*zoom*rows
#fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
#                        sharey='row')
#marker_size=get_marker_size(axs[0],fig,loncorners,grid_size,marker_factor)
#plt.subplots_adjust(wspace=0.04,hspace=0.04,top=0.7,bottom=-0.1)
#for year in year_range:   
#    #---------------------------------------------------------------
#    ax=axs[year-year_range[0]]
#    p1=data.quantile(0.95).hist(normed=True,ax=ax,alpha=alpha,\
#                    color='lightcoral',label='$upper\:cutoff_{all\:years}$')
#    p2=data.quantile(0.05).hist(normed=True,ax=ax,alpha=alpha,\
#                    color='seagreen',label='$lower\:cutoff_{all\:years}$')
#    p3=data[data.index.year==year].quantile(0.5).hist(normed=True,\
#           ax=ax,alpha=alpha,color='royalblue',\
#           label='$median_{season}$')
#    ax.set_title(str(year))
#    #-------------------------------------------------------------------
#axs[3].set_xlabel('VOD')
#axs[0].set_ylabel('Normalized\nfrequency',rotation = 0,labelpad=50,va='center')
#fig.suptitle('Distribution of VOD')
#handles, labels = ax.get_legend_handles_labels()
#plt.legend(handles,labels,fontsize=10,loc='lower right',bbox_to_anchor=[1,-0.8])
#plt.show()
#


### maps with only variables
#inputs
store=pd.HDFStore('data_subset_GC.h5')
data=(store['RWC_v2']) 
grid_size=25
start_year=2009
data_label="RWC"
cmap='viridis'
alpha=0.7
mort_label='FAM'

#----------------------------------------------------------------------
mort=store['mortality_%03d_grid'%(grid_size)]
mort=mort[mort>0]
end_year=min(max(mort.index.year),max(data.index.year))
data=data[(data.index.year>=start_year) &\
                          (data.index.year<=end_year)]
#data2=data
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
year_range=mort.index.year
cols=len(year_range)
zoom=1.1
rows=2
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
data_min= np.nanmin(data.iloc[:, :].values)
data_max= np.nanmax(data.iloc[:, :].values)
tree_min=np.nanmin(mort.iloc[:, :].values)
tree_max=np.nanmax(mort.iloc[:, :].values)
fig_width=zoom*cols
fig_height=1.5*zoom*rows
if grid_size==25:
    marker_factor=7
    scatter_size=20
elif grid_size==5:
    marker_factor=2
    scatter_size=4
lats,lons=supply_lat_lon('GC_subset')

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
    plot_mort=m.scatter(lons,lats,s=marker_size,c=mort_plot,cmap='inferno_r',\
                        marker='s',\
                        vmin=tree_min,vmax=tree_max,\
                        norm=mpl.colors.PowerNorm(gamma=1./2.)\
                                               )
    #---------------------------------------------------------------
    data_plot=data[data.index.year==year]
    ax=axs[1,year-year_range[0]]
    m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
            ax=ax)
    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
    plot_data=m.scatter(lons,lats,s=marker_size,c=data_plot,cmap=cmap\
                       ,marker='s',vmin=0.0,vmax=1.0)
    #-------------------------------------------------------------------
    
cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)
#tick_locator = ticker.MaxNLocator(nbins=7)
#cb0.locator = tick_locator
#cb0.update_ticks()
cb0.set_ticks(np.linspace(0,0.6 ,7))
cb1=fig.colorbar(plot_data,ax=axs[1,:].ravel().tolist(), fraction=0.01,\
                 aspect=20,pad=0.02)


cb1.set_ticks(np.linspace(0.2,0.8,4))
axs[0,0].set_ylabel(mort_label)
axs[1,0].set_ylabel(data_label)
#fig.suptitle('Timeseries maps of mortality and indicators')
plt.show()
