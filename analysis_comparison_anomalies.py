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
end_year=2015
start_month=8
start_day=212
months_window=3
data2_label='CWD annually \naccumulated'
data_label="VOD \nanomaly"
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
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
mort_main=mort.copy()
    
data=data.loc[(data.index.month>=start_month) & \
              (data.index.month<start_month+months_window)]
data=data[(data.index.year>=start_year) &\
                          (data.index.year<=end_year)]
anomaly_to_plot=[year_anomaly_mean(data),RWC(data),log_anomaly(data),\
                 median_div_max(data),min_div_max(data)]
data_label=['Anomaly of\nmean','RWC',r'$log(\frac{median}{95^{th}\ \%tile})$',\
r'$\frac{median}{95^{th}\ \%tile}$',r'$\frac{5^{th}\ \%tile}{95^{th}\ \%tile}$']
data_min=[-3,0,-0.1,0.75,0.7]
data_max=[2.8,1,0.0,1.05,1.0]
year_range=mort.index.year
cols=mort.shape[0]
zoom=1.1
latcorners=[33,42.5]
loncorners=[-124.5,-117] 
rows=len(anomaly_to_plot)+1


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
#fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(fig_width,fig_height),\
#                        sharey='row')
#marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
#plt.subplots_adjust(wspace=0.04,hspace=0.1)
#i=0
#for anomaly in anomaly_to_plot:
#    i+=1
#    data_anomaly=anomaly
##    data_min= np.nanmin(data_anomaly.iloc[:, :].values)
##    data_max= np.nanmax(data_anomaly.iloc[:, :].values)
#    for year in year_range:   
#        mort_plot=mort[mort.index.year==year]
#        ax=axs[0,year-year_range[0]]
#        ax.set_title(str(year))
#        m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#                ax=ax)
#        m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#        plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap=cmap,\
#                            marker='s',\
##                            vmin=0.9*tree_min,vmax=0.9*tree_max,\
#                            norm=mpl.colors.LogNorm()\
#                                                   )
#
#        #---------------------------------------------------------------
#        data_plot=data_anomaly[data_anomaly.index.year==year]
#        ax=axs[i,year-year_range[0]]
#        m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
#                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
#                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
#                ax=ax)
#        m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
#        plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap\
#                           ,marker='s',vmin=data_min[i-1],vmax=data_max[i-1])
#    cb1=fig.colorbar(plot_data,ax=axs[i,:].ravel().tolist(), fraction=0.01,\
#                     aspect=20,pad=0.02)
#
#        #-------------------------------------------------------------------
#    
#
##    cb1.ax.invert_yaxis()
#    #cb1.set_ticks(np.linspace(0.2,0.8,4))
#    #cb1.set_ticklabels(np.linspace(0.2,0.8,4))
#    axs[i,0].set_ylabel(data_label[i-1],rotation = 0,labelpad=50)
#axs[0,0].set_ylabel(mort_label,rotation = 0,labelpad=50)
#cb0=fig.colorbar(plot_mort,ax=axs[0,:].ravel().tolist(), fraction=0.01,\
#             aspect=20,pad=0.02)
#fig.suptitle('Timeseries maps of mortality and indicators')
#plt.show()

### scatter plot linear scale 
i=0
thresh=0.0
rep_times=1
sns.set_style("darkgrid")
guess_x=[0.1,0.25,-0.1,0.9,0.9]
for anomaly in anomaly_to_plot:
    i+=1
    data_anomaly=anomaly
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4))
    plt.subplots_adjust(wspace=0.15)
    ax=ax
    x=data_anomaly.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y,rep_times,thresh)
    plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_xlabel(data_label[i-1])
    ax.set_ylabel(mort_label,labelpad=50,rotation = 0)
    #ax.set_xlim([-3,3])
#    ax.invert_xaxis()
    guess=(guess_x[i-1],0.01,1e-1,1e-1)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y,guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1,label='Breakpoint\nlinear')
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=ax.get_ylim()
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim([ymin,ymax])
    #ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
    #            ha='left')
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    ax.annotate('$R^2 = $%.2f'%r_squared, xy=(0.05, 0.9), xycoords='axes fraction',\
                ha='left',color='r')
    x,xd= np.reshape(x,(len(x),1)),np.reshape(xd,(len(xd),1))
    fit = SVR(epsilon=1e-2).fit(x, y)
    ax.plot(xd, fit.predict(xd),'--',color='fuchsia', label='SVR')
    ax.annotate('$R^2 = $%.2f'%fit.score(x,y), xy=(0.05, 0.8), xycoords='axes fraction',\
                ha='left',color='fuchsia')
