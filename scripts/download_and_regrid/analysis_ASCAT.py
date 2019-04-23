# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *

arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2009,2016)
day_range=range(1,362,2)
    
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
#store['mort']=m
store = pd.HDFStore(Dir_CA+'/sigma0.h5')
sigma0=store['sigma0']
#store['sigma0']=a
store.close()
nos=5938
ind0=[l for l in mort.columns if mort.loc[2016,l] >=0.0]
ind4=[l for l in mort.columns if mort.loc[2016,l] >=0.4]
ind6=[l for l in mort.columns if mort.loc[2016,l] >=0.6]
ind8=[l for l in mort.columns if mort.loc[2016,l] >=0.8]
ind=dict([('0.0',ind0),('0.4',ind4),('0.6',ind6),('0.8',ind8)])

thresh='0.8'
a=sigma0[ind[thresh]]
m=mort.loc[year_range,ind[thresh]]

## plot VOD
fs=15
a.index=pd.to_datetime(a.index,yearfirst=True,format='%Y%j')
plt.figure()
a.rolling(window=5).mean().plot(legend=False,alpha=0.4)
#a.mean(1).plot(legend=True,color='m',linewidth=2,label='mean')
ax=plt.gca()
#ax.set_xlabel(r'Date (yyyyddd)',fontsize=fs)
ax.set_title(r"Variation of $VOD_{ASCAT}$ with time", fontsize=fs)
ax.set_ylabel(r'$VOD_{ASCAT}$',fontsize=fs)
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)
a=sigma0[ind[thresh]]






### day of the year analysis for xfun=min(x-mew)/std
mean=pd.DataFrame(np.full((len(day_range),len(a.columns)),np.NaN), columns=a.columns,index=day_range)
std=mean.copy()
for j in day_range:
    day='%03d'%j
    row=[row for row in a.index if day in str(row)[-3:]]
    mean.ix[j]=a.ix[row].mean()
    std.ix[j]=a.ix[row].std()

## plot mean and std
fs=15
plt.figure()
mean.plot(legend=False,color='grey',alpha=0.3)
mean.mean(1).plot(legend=True,color='m',fontsize=fs,linewidth=3,label='mean')
ax=plt.gca()
ax.set_title(r'Day of Year Mean of $VOD_{ASCAT}$',fontsize=fs)
ax.set_xlabel("Day of Year", fontsize=fs)
ax.set_ylabel(r'$\mu(VOD_{ASCAT})$',fontsize=fs)
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)

plt.figure()
std.plot(legend=False,color='grey',alpha=0.3)
std.mean(1).plot(legend=True,color='m',fontsize=fs,linewidth=3,label='mean')
ax=plt.gca()
ax.set_title(r'Day of Year $\sigma(VOD_{ASCAT})$',fontsize=fs)
ax.set_xlabel("Day of Year", fontsize=fs)
ax.set_ylabel(r'$\sigma(VOD_{ASCAT})$',fontsize=fs)
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)

#### calculate x function
VOD_anomaly=pd.DataFrame()
for k in year_range:
    year=str(k)
    row=[row for row in a.index if year in str(row)[:4]]
    day=[day%1000 for day in row]
    year_vod=a.ix[row]
    year_vod.index=day
    year_mean=mean.ix[day]
    year_std=std.ix[day]
    func=(year_vod-year_mean)/year_std
    func_min=pd.DataFrame(func.min()).T
    func_min.index=[k]
    VOD_anomaly=pd.concat([VOD_anomaly,func_min],axis=0)
### plot the anomaly in scatter type
plt.figure()
plt.plot(VOD_anomaly, m,'or',mfc='None')
ax=plt.gca()
ax.invert_xaxis()
ax.set_xlabel(r'Minimum DoY $VOD_{ASCAT}$ anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title(r'Mortality Vs $VOD_{ASCAT}$ anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)

colors = cm.rainbow(np.linspace(0, 1, len(year_range)))
plt.figure()
for year,c in zip(year_range,colors):
    x=VOD_anomaly[VOD_anomaly.index==year]
    y=m[m.index==year]
    ax=plt.plot(x,y,'o',alpha=0.5,color=c)
    ax=plt.plot(x.iloc[0,0],y.iloc[0,0],'o',alpha=0.5,color=c,label='%s'%year)
ax=plt.gca()
ax.invert_xaxis()
ax.set_xlabel(r'Minimum DoY $VOD_{ASCAT}$ anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title(r'Mortality Vs $VOD_{ASCAT}$ anomaly',fontsize=fs) 
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.2, 0.9), xycoords='axes fraction')
plt.legend(fontsize=10)
plt.show()
    
# make box plot with equal width
boxes=15
y=m.values.flatten()
x=VOD_anomaly.values.flatten()
min=(np.nanmin(x))
max=(np.nanmax(x))
binwidth=(max-min)/boxes
yb=np.empty((len(x),boxes,))
yb[:] = np.nan
for i in range(boxes):
    row=np.where((x>=min+i*binwidth) & (x<min+(i+1)*binwidth))
    yb[row,i]=y[row]
name=(np.arange(min+binwidth/2,max-binwidth/2+binwidth/10,binwidth))
name=np.round(name,2)
yb=pd.DataFrame(yb,columns=name)
yb=pd.DataFrame(yb,columns=name)
plt.figure()
fs=15
ax=yb.boxplot(grid=False,showfliers=False,showmeans=True)
ax=plt.gca()
ax.invert_xaxis()
ax.set_xlabel(r'Minimum DoY $VOD_{ASCAT}$ anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title(r'Mortality Vs $VOD_{ASCAT}$ anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)
ax.annotate('Equal width binning', xy=(0.05, 0.8), color='r',xycoords='axes fraction',fontsize=fs)
ax.set_xticklabels(name,rotation=90)
plt.plot([0],'^',alpha=0,label='mean')
plt.legend()
## make box plots with equal number
y=m.values.flatten()
x=VOD_anomaly.values.flatten()
inds=x.argsort()
x=x[inds]
y=y[inds]
inds=np.argwhere(~np.isnan(x))
x=x[inds]
y=y[inds]
boxes=15
count=len(x)/boxes
count=np.ceil(count).astype(int)
yb=pd.DataFrame()
for i in range(boxes):
    data=y[i*count:(i+1)*count]
    name=np.mean(x[i*count:(i+1)*count]).round(2)
    data=pd.DataFrame(data,columns=[name])
    yb=pd.concat([yb,data],axis=1)
plt.figure()
fs=15
ax=yb.boxplot(grid=False,showfliers=False,showmeans=True)
ax.set_xlim([0,22])
ax=plt.gca()
ax.invert_xaxis()
ax.set_xlabel(r'Minimum DoY $VOD_{ASCAT}$ anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title(r'Mortality Vs $VOD_{ASCAT}$ anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %s'%thresh, xy=(0.05, 0.9), xycoords='axes fraction',fontsize=fs)
ax.annotate('Equal samples binning', xy=(0.05, 0.8), color='r',xycoords='axes fraction',fontsize=fs)
ax.set_xticklabels(yb.columns,rotation=90)



