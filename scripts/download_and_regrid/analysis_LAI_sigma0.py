# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
sns.set(font_scale=1.5)
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2009,2017)
    
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
mort.index=pd.to_datetime(mort.index,format='%Y')
store = pd.HDFStore(Dir_CA+'/LAI.h5')
LAI=store['LAI_smallgrid']
LAI=LAI[(LAI.index.year>=2009)]
store = pd.HDFStore(Dir_CA+'/sigma0.h5')
sigma0=store['sigma0']
sigma0.index=pd.to_datetime(sigma0.index,format='%Y%j')
VOD=sigma0
store.close()

##plot LAI
fs=15
thresh=0.8
fig=plt.figure()
ax=LAI[ind(thresh,mort)].plot(legend=False,alpha=0.7,color='k',linewidth=0.8)
ax.set_ylim([0,9])
ax.set_ylabel('LAI')
ax.set_title('Variation of LAI')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')

## calculate RWC
VOD_monthly=pd.DataFrame()
LAI_monthly=pd.DataFrame()
for year in year_range:
    a=VOD[VOD.index.year==year]
    a=a.groupby(a.index.month).mean()
    a.index=pd.to_datetime({'year':year,'month':a.index,'day':1})
    VOD_monthly=pd.concat([VOD_monthly,a])
    a=LAI[LAI.index.year==year]
    a=a.groupby(a.index.month).mean()
    a.index=pd.to_datetime({'year':year,'month':a.index,'day':1})
    LAI_monthly=pd.concat([LAI_monthly,a])
RWC=VOD_monthly/LAI_monthly
fig=plt.figure()
ax=RWC[ind(thresh,mort)].plot(legend=False,alpha=0.7,color='k',linewidth=0.8,fontsize=10)
ax.set_ylabel('RWC')
ax.set_title('Variation of RWC=${VOD_{ASCAT}}/{LAI}$')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')

RWC_min_anomaly=min_anomaly(RWC)
colors = cm.rainbow(np.linspace(0, 1, len(year_range)))
plt.figure()
for year,c in zip(year_range,colors):
#    if year in {2011 ,2012}:
#        continue
    x=RWC_min_anomaly[ind(thresh,mort)][RWC_min_anomaly.index.year==year]
    y=mort[ind(thresh,mort)][mort[ind(thresh,mort)].index.year==year]
    ax=plt.plot(x,y,'o',alpha=0.5,color=c)
    ax=plt.plot(x.iloc[0,0],y.iloc[0,0],'o',alpha=0.5,color=c,label='%s'%year)
ax=plt.gca()
ax.invert_xaxis()
ax.set_ylabel('FAM')
ax.set_xlabel('Min. MoY RWC anomaly')
ax.set_title('FAM Vs. RWC anomaly')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.2, 0.9), xycoords='axes fraction')
plt.legend(fontsize=10)
plt.show()

## make box plots with equal number
boxes=15
yb=box_equal_nos(RWC_min_anomaly[ind(thresh,mort)],mort[ind(thresh,mort)],boxes)
plt.figure()
ax=yb.boxplot(grid=False,showfliers=False,showmeans=True)
ax=plt.gca()
ax.invert_xaxis()
ax.set_ylabel('FAM')
ax.set_xlabel('Min. MoY RWC anomaly')
ax.set_title('FAM Vs. RWC anomaly')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')
ax.annotate('Equal samples binning', xy=(0.05, 0.8), color='r',xycoords='axes fraction',fontsize=fs)
ax.set_xticklabels(yb.columns,rotation=90)

#store = pd.HDFStore(Dir_CA+'/LAI.h5')
#store['RWC_min_anomaly_smallgrid']=RWC_min_anomaly
#store.close()