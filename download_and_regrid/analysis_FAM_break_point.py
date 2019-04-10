# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 12:28:31 2017

@author: kkrao
"""
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
start_year=2009
start_month=7
months_window=3
data2_label='CWD annually \naccumulated (mm)'
data_label="VOD \nanomaly"
mort_label='Fractional area\n of mortality'
cmap='viridis'
alpha=0.7
rep_times=1
#----------------------------------------------------------------------
mort=store['mortality_%03d_grid'%grid_size]
mort=mort[mort>0]
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
#----------------------------------------------------------------------

### scatter plot linear scale scale
sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey='row')
plt.subplots_adjust(wspace=0.15)
ax=axs[0]
x=data_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y,rep_times,0)
plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s')
ax.set_xlabel(data_label)
ax.set_ylabel(mort_label,labelpad=40,rotation = 0)
ax.set_xlim([-3,3])
ax.invert_xaxis()
popt , pcov = optimize.curve_fit(piecewise_linear, x, y)
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
ax=axs[1]
x=data2_anomaly.values.flatten()
y=mort.values.flatten()
x,y,z=clean_xy(x,y,rep_times,0)
plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s')
ax.set_xlabel(data2_label)
fig.suptitle('Scatter plot relating mortality with indicators')
cbaxes = fig.add_axes([0.92, 0.6, 0.03, 0.2])
cb=fig.colorbar(plot2_data,ax=axs[1],\
                ticks=[min(z), max(z)],cax=cbaxes)
cbaxes.text(0,1.2,'Density')
cb.ax.set_yticklabels(['Low', 'High'])
guess=(500,0.01,1e-4,1e-4)
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

