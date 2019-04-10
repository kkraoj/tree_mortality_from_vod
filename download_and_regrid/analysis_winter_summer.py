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
start_month=7
start_day=212
months_window=3
data2_label='CWD annually \naccumulated'
data_label="VOD \nanomaly"
#data2_label=data_label
cmap='viridis'
alpha=0.5
y_label='VOD'
plot_title='VOD (1:30 pm)'
#species='evergreen'
#species='deciduous'
#mort_label='Dead trees\nper acre'
mort_label='Fractional area\nof mortality'

#----------------------------------------------------------------------
mort=store['mortality_%03d_grid'%(grid_size)]
mort=mort[mort>0]
mort_main=mort.copy()
data2_anomaly=data2
data_summer=data.loc[(data.index.month>=start_month) & \
              (data.index.month<start_month+months_window)]
data_winter=data.loc[((data.index.month>=1) & \
              (data.index.month<1+months_window))]
data=data_summer.append(data_winter)
data.sort_index(inplace=True)
data_anomaly=year_anomaly_mean(data)
#end_year=min(max(mort.index.year),max(data.index.year))
data_anomaly=data_anomaly[(data_anomaly.index.year>=start_year) &\
                          (data_anomaly.index.year<=end_year)]
#data2_anomaly=data_anomaly
mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
#(mort,data_anomaly,data2_anomaly)=mask_columns(ind_small_species(species),\
#                                 mort,data_anomaly,data2_anomaly)
data_summer=data_summer.loc[(data_summer.index.year>=start_year)&\
                          (data_summer.index.year<=end_year)]
data_winter=data_winter.loc[(data_winter.index.year>=start_year)&\
                          (data_winter.index.year<=end_year)]


fig, ax = plt.subplots(1,1,figsize=(4,4))
data_summer.plot(legend=False,color='lightcoral',alpha=alpha,ax=ax,\
                 lw=0.5,linestyle='',marker='.',markersize=0.5,label='Summer')
data_winter.plot(legend=False,color='royalblue',alpha=alpha,ax=ax,\
                 lw=0.5,linestyle='',marker='.',markersize=0.5,label='Winter')
data_summer[0].plot(color='lightcoral',alpha=alpha,ax=ax,\
                 lw=1,linestyle='-',label='Summer')


l1=handles[-1]
data_winter[0].plot(color='royalblue',alpha=alpha,ax=ax,\
                 lw=1,label='Winter')
handles, labels = ax.get_legend_handles_labels()
l2=handles[-1]
l3=ax.axvline(data_summer.groupby(data_summer.index.year).tail(1).index[-1],\
                      0.08,0.6,linestyle='--',linewidth=1,\
                      color='k',label='Start of\nwater year',zorder=10)
[ax.axvline(at_x,0.08,0.6,linestyle='--',linewidth=1,color='k',label='_nolegend_')\
            for at_x in \
            data_summer.groupby(data_summer.index.year).tail(1).index]
ax.set_ylabel(y_label)
ax.set_title(plot_title)
ax.set_ylim([0.3,3.2])
ax.legend(handles=[l1,l2,l3],loc='upper right')