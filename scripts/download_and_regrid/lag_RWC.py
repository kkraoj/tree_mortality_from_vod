# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:25:13 2017

@author: kkrao
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot as acf_plot
from dirs import Dir_CA, clean_xy


def plot_lag_RWC(data1='RWC',data2='cwd',data1_label="RWC (-)",\
                 data2_label='CWD (mm)',
                    mort_label='FAM (-)',\
                    mort='mortality_%03d_grid',\
                    grid_size=25,cmap='viridis', start_year=2011,\
                    end_year = 2015,mort_range=[0,0.42],\
                    journal='EcolLett',alpha=0.7,lag_range=range(3)):
    if grid_size==25:
        scatter_size=20
    elif grid_size==5:
        scatter_size=4
    os.chdir(Dir_CA)
    store=pd.HDFStore('data.h5')
    mort=store[mort%(grid_size)]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    sns.set(font_scale=1.1,style='ticks')
    fig, axs = plt.subplots(1,len(lag_range),sharey='row',figsize=(3*len(lag_range),3))
    plt.subplots_adjust(wspace=0.08)
    for lag in lag_range:
        ax=axs[lag]
        Df=store[data1]
        Df=Df[(Df.index.year>=(start_year-lag)) &\
              (Df.index.year<=(end_year-lag))] 
        x=Df.values.flatten()
        y=mort.values.flatten()
        x,y,z=clean_xy(x,y)
        plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
        ax.set_xlabel(data1_label)
        ax.annotate('Lag = %1d year'%lag, xy=(0.5, 1.03), xycoords='axes fraction',\
                ha='center',va='bottom',color='k')
    axs[0].set_ylabel(mort_label)
    cbaxes = fig.add_axes([0.25, 0.55, 0.02, 0.15])
    cb=fig.colorbar(plot_data,ax=ax,\
                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off')
    cbaxes.annotate('Scatter plot\ndensity',xy=(0,1.2), xycoords='axes fraction',\
                ha='left')
    cb.outline.set_visible(False)

def main():
    pass
#    plot_lag_RWC()

if __name__ == '__main__':
    main()
os.chdir(Dir_CA)
store=pd.HDFStore('data_subset_GC.h5')
mort=store['mortality_025_grid']
#==============================================================================
thresh=0.1
cond=mort>=thresh
mort=mort.loc[:,cond.any(axis=0)]
sns.set(style='ticks')
fig, ax = plt.subplots(figsize=(6,3))
for col in mort.columns:
    acf_plot(mort[col],ax=ax,alpha=0.2,color='grey')
acf_plot(mort.mean(axis=1),ax=ax,color='darkviolet',label='mean')
ax.set_ylim([-1,1])
ax.set_xlim([0,6])
ax.set_title('Autocorrelation function for mortality')
ax.set_xlabel('Lag (years)')
ax.legend(bbox_to_anchor=(0.95, 0.3))
plt.grid('off')
plt.show()

#==============================================================================
mort=store['mortality_025_grid']
thresh=0.3
cond=mort>=thresh
mort=mort.loc[:,cond.any(axis=0)]
sns.set(style='ticks')
fig, ax = plt.subplots(figsize=(6,3))
mort.plot(color='grey',alpha=0.4,legend=False,ax=ax)
mort.mean(1).plot(color='darkviolet',label='mean',legend=True,ax=ax)
ax.set_ylabel('FAM (-)')
ax.set_title('Mortality trend for high mortality regions')
ax.set_xlabel('Year')
#ax.legend(handles=plot_mean)
plt.grid('off')
plt.show()