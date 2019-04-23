# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
mpl.rc('font', size=15)


arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2009,2016)
day_range=range(1,362,2)
    
store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
mort=store['mort']
mort.index=pd.to_datetime(mort.index,format='%Y')
mort_09_15=mort[(mort.index.year>=np.min(year_range))&(mort.index.year<=np.max(year_range))]
store = pd.HDFStore(Dir_CA+'/sigma0.h5')
sigma0=store['sigma0']
sigma0.index=pd.to_datetime(sigma0.index,format='%Y%j')
sigma0_09_15=sigma0[(sigma0.index.year>=np.min(year_range))&(sigma0.index.year<=np.max(year_range))]
store.close()

thresh=0.8
## plot VOD
plt.figure()
ax=sigma0_09_15[ind(thresh,mort)].rolling(window=5).mean().plot(legend=False,alpha=0.4)
ax.set_title(r"Variation of $VOD_{ASCAT}$ with time")
ax.set_ylabel(r'$VOD_{ASCAT}$')
ax.grid()
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')

## plot raw sigma0 in db
sigma0_dB_09_15=10*np.log10((np.exp(1))**(-2*sigma0_09_15))
plt.figure()
ax=sigma0_dB_09_15[ind(thresh,mort)].rolling(window=7).mean().plot(legend=False,alpha=0.4)
ax.set_title(r"Variation of $\sigma_0(dB)$ with time")
ax.set_ylabel(r'$\sigma_0(dB)$')
ax.grid()
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.9), xycoords='axes fraction')



VOD_anomaly=min_anomaly(sigma0_09_15)
### plot the anomaly in scatter type
thresh=0.0
colors = cm.rainbow(np.linspace(0, 1, len(year_range)))
plt.figure()
for year,c in zip(year_range,colors):
    x=VOD_anomaly[ind(thresh,mort)][VOD_anomaly.index.year==year]
    y=mort_09_15[ind(thresh,mort)][mort_09_15.index.year==year]
    ax=plt.plot(x,y,'o',alpha=0.5,color=c)
    ax=plt.plot(x.iloc[0,0],y.iloc[0,0],'o',alpha=0.5,color=c,label='%s'%year)
ax=plt.gca()
ax.invert_xaxis()
ax.set_xlabel(r'Minimum DoY $VOD_{ASCAT}$ anomaly')
ax.set_ylabel('FAM')
ax.set_title(r'Mortality Vs $VOD_{ASCAT}$ anomaly') 
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.2, 0.9), xycoords='axes fraction')
plt.legend(fontsize=10)
plt.show()


### map plots
latcorners=[33,42.5]
loncorners=[-124.5,-117]    

# A good LCC projection for USA plots
#m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
#            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
zoom=2
rows=2
cols=mort_09_15.shape[0]
width=zoom*cols
height=1.5*zoom*rows



fig, axs = plt.subplots(nrows=rows,ncols=cols,figsize=(width,height))

m = Basemap(projection='cyl',lat_0=45,lon_0=0,resolution='l',\
        llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
        llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
        ax=axs[0,1]                    \
                            )

m.readshapefile(Dir_CA+'/smallgrid','smallgrid', color='orange')
m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')

  