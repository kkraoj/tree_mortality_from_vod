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
data2_label='CWD accumulated\n(2011-2015)'
data_label="VOD anomaly\n(2015)"
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
data2_anomaly=cwd_accumulate(data2,2011,2015)
data=data.loc[((data.index.month>=start_month) & \
              (data.index.month<start_month+months_window))]
data_anomaly=year_anomaly_mean(data)
end_year=min(max(mort.index.year),max(data.index.year))
data_anomaly=data_anomaly[(data_anomaly.index.year>=end_year) &\
                          (data_anomaly.index.year<=end_year)]
#data2_anomaly=data_anomaly
mort=mort[(mort.index.year>=end_year) &\
          (mort.index.year<=end_year)]
#(mort,data_anomaly,data2_anomaly)=mask_columns(ind_small_species(species),\
#                                 mort,data_anomaly,data2_anomaly)
year_range=mort.index.year

if grid_size==25:
    grids=Dir_mort+'/CA_proc.gdb/grid'
    marker_factor=7
    scatter_size=20
elif grid_size==5:
    grids=Dir_mort+'/CA_proc.gdb/smallgrid'
    marker_factor=2
    scatter_size=4

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
guess=(1,0.01,1e-1,1e-1)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y,guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
#ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
#ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
#                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
#                                color='r',alpha=0.6)
#ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
#ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
#                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
#ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left')
residuals = y- piecewise_linear(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
#ax.annotate('$R^2 = $%.2f'%r_squared, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left',color='r')
guess=(0.25,0.01,1e-1,1e-1)
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
                      alpha=0.3, color='k', lw=0, zorder=10))
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
guess=(1000,0.01,1e-4,1e-4)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
#ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
#ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
#                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
#                                color='r',alpha=0.6)

#ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
#ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
#                      hatch='//////', color='k', lw=0, fill=False,zorder=10))
ax.set_ylim([ymin,ymax])
guess=(2500,0.01,1e-4,1e-4)
popt , pcov = optimize.curve_fit(piecewise_linear, x, y,guess)
perr = np.sqrt(np.diag(pcov))
xd = np.linspace(min(x), max(x), 1000)
ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1,label='Linear')
ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                color='r',alpha=0.6)
ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
ymin,ymax=ax.get_ylim()
ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                      color='k', lw=0, alpha=0.3,zorder=10))
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
fit = SVR(epsilon=1e-2,gamma=1e-6).fit(x, y)
ax.plot(xd, fit.predict(xd),'--',color='fuchsia', label='SVR',lw=1)
ax.annotate('$R^2 = $%.2f'%fit.score(x,y), xy=(0.05, 0.8), xycoords='axes fraction',\
            ha='left',color='fuchsia')
fig.suptitle('Scatter plot relating mortality with indicators')
cbaxes.text(0,1.2,'Density')
#ax.annotate('%s trees'%species, xy=(0.05, 0.9), xycoords='axes fraction',\
#            ha='left')
ax.legend(loc='upper right',bbox_to_anchor=[1.38,0.6])