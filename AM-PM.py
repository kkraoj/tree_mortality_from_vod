# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2005,2016)
day_range=range(1,366,1)
    
store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort.index.name='gridID'
mort=mort.T
mort.drop('gridID',inplace=True)
mort.index=[x[-4:] for x in mort.index] 
mort.index=pd.to_datetime(mort.index)
mort_05_15=mort[mort.index.year!=2016]
store = pd.HDFStore(Dir_CA+'/vodDf.h5')#ascending is 1:30 PM
VOD_PM=store['vodDf']
VOD_PM.index.name='gridID'
VOD_PM=VOD_PM.T
VOD_PM.drop('gridID',inplace=True)
VOD_PM.index=[x[:-1] for x in VOD_PM.index] 
VOD_PM.index=pd.to_datetime(VOD_PM.index,format='%Y%j')
VOD_PM=VOD_PM[VOD_PM.index.dayofyear!=366]
store = pd.HDFStore(Dir_CA+'/vod_D_Df.h5')
VOD_AM=store['vod_D_Df']
VOD_AM=VOD_AM[VOD_AM.index.dayofyear!=366]
store.close()

## calculate RWC
#RWC=(VOD_PM-VOD_AM)/VOD_PM
RWC=VOD_PM/VOD_AM
plt.figure()
ax=RWC[ind(thresh,mort)].rolling(1).mean().plot(legend=False,alpha=0.7,color='k',linewidth=0.8)
ax.set_ylabel('RWL')
ax.set_title('Variation of RWL')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')







mean=RWC.groupby(RWC.index.dayofyear).mean()
sd=RWC.groupby(RWC.index.dayofyear).std()
RWC_min_anomaly=pd.DataFrame()
for year in year_range:
    a=RWC[RWC.index.year==year]
    a.index=a.index.dayofyear
    min_anomaly=((a-mean)/sd).min()
    min_anomaly.name=pd.Timestamp(year,1,1)
    RWC_min_anomaly=pd.concat([RWC_min_anomaly,min_anomaly],1)
RWC_min_anomaly=RWC_min_anomaly.T
thresh=0.2
colors = cm.rainbow(np.linspace(0, 1, len(year_range)))
plt.figure()
for year,c in zip(year_range,colors):
#    if year in {2011 ,2012}:
#        continue
    x=RWC_min_anomaly[ind(thresh,mort)][RWC_min_anomaly.index.year==year]
    y=mort_05_15[ind(thresh,mort)][mort_05_15[ind(thresh,mort)].index.year==year]
    ax=plt.plot(x,y,'o',color=c,alpha=0.5)
    ax=plt.plot(x.iloc[0,0],y.iloc[0,0],'o',alpha=0.5,color=c,label='%s'%year)
ax=plt.gca()
ax.invert_xaxis()
ax.set_ylabel('FAM')
ax.set_xlabel('Min. DoY RWL anomaly')
ax.set_title('FAM Vs. RWL anomaly')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.2, 0.9), xycoords='axes fraction')
plt.legend(fontsize=10)
plt.show()

## make box plots with equal number
boxes=15
yb=box_equal_nos(RWC_min_anomaly[ind(thresh,mort)],mort_05_15[ind(thresh,mort)],boxes)
plt.figure()
ax=yb.boxplot(grid=False,showfliers=False,showmeans=True)
ax=plt.gca()
ax.invert_xaxis()
ax.set_ylabel('FAM')
ax.set_xlabel('Min. DoY RWL anomaly')
ax.set_title('FAM Vs. RWL anomaly')
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')
ax.annotate('Equal samples binning', xy=(0.05, 0.8), color='r',xycoords='axes fraction',fontsize=fs)
ax.set_xticklabels(yb.columns,rotation=90)
