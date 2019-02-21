# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 23:54:55 2019

@author: kkrao
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dirs import MyDir, Dir_CA, RWC, seasonal_transform


##### make ndwi DF
#os.chdir(MyDir)
#Df = pd.DataFrame()
#
#for file in os.listdir('ndwi/raw'):
#    df =pd.read_csv('ndwi/raw/'+file)
#    df['gridID'] = int(file.split('_')[0])
#    Df = Df.append(df, ignore_index = True)
#
## 2-5/2+5
#Df['ndwi'] = (Df["Nadir_Reflectance_Band2"]-Df["Nadir_Reflectance_Band5"])/\
#          (Df["Nadir_Reflectance_Band2"]+Df["Nadir_Reflectance_Band5"])
#Df.index = Df.gridID
#Df = Df.loc[:,['date','ndwi', 'gridID']]
#
#Df = Df.pivot(index = 'date',columns = 'gridID', values = 'ndwi')
#print(Df.head())
#Df.index.name = 'ndwi'
#Df.index = pd.to_datetime(Df.index)
#store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#store[Df.index.name] = Df

#for season in ['sum','win']:
#    df=seasonal_transform(Df,season)
#    df=df.groupby(df.index.year).mean()
#    df.index=pd.to_datetime(df.index,format='%Y')
#    df.index.name=Df.index.name+'_'+season
#    store[df.index.name]=df
##### plotting
sns.set_style('ticks')
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
Df = store['ndwi']
Df = Df.loc[(Df.index.year>=2009)&(Df.index.year<=2015),:]



###### time series of NDWI
fig, ax = plt.subplots(figsize = (6,2))
Df.loc[:,333].plot(ax = ax)
ax.annotate("FAM = 0.04", xy=(0.7, 0.8), color = "darkred",\
            xycoords='axes fraction',ha = "left")
        
#Df.plot(ax = ax, legend = False)
ax.set_ylabel('ndwi')
ax.set_xlabel("")

#### scatter plot between ndwi and rwc
#rwc = store['RWC_matched']
#rwc_opt = RWC(Df, start_year = 2009)
#fig, ax = plt.subplots(figsize = (3,3))
#ax.scatter(rwc, rwc_opt, s = 5)
#ax.set_ylabel("$RWC_{NDWI}$")
#ax.set_xlabel("$RWC_{VOD}$")
#R2 = rwc.corrwith(rwc_opt).mean()
#ax.annotate("$R$ = %0.2f"%R2, xy=(0.1, 0.8), color = "darkred",\
#            xycoords='axes fraction',ha = "left")
        
######### RWC and ndwi sum win scatter plot
#rwc = store['RWC_matched']
#for season in ['sum','win']:
#    df = store['ndwi_%s'%season]
#    df = df.loc[(df.index.year>=2009)&(df.index.year<=2015),:]
#
#    fig, ax = plt.subplots(figsize = (3,3))
#    ax.scatter(rwc, df, s = 5)
#    ax.set_ylabel("$NDWI_{%s}$"%season)
#    ax.set_xlabel("RWC")
#    if season == 'win':
#        ax.set_ylim(bottom = -0.15)
#    else:
#        ax.set_ylim([-0.2,0.1])
######### mort and ndwi sum win scatter plot
#mort = store['mortality_025_grid']
#mort = mort.loc[(mort.index.year>=2009)&(mort.index.year<=2015),:]
#for season in ['sum','win']:
#    df = store['ndwi_%s'%season]
#    df = df.loc[(df.index.year>=2009)&(df.index.year<=2015),:]
#
#    fig, ax = plt.subplots(figsize = (3,3))
#    ax.scatter(df,mort, color = 'k', s = 6, alpha = 0.5)
#    ax.set_xlabel("$NDWI_{%s}$"%season)
#    ax.set_ylabel("FAM")
#    if season == 'win':
#        ax.set_xlim(left = -0.15)
#    else:
#        ax.set_xlim([-0.2,0.1])
#    R2 = mort.corrwith(df).mean()
#    ax.annotate("$R$ = %0.2f"%R2, xy=(0.1, 0.8), color = "darkred",\
#            xycoords='axes fraction',ha = "left")


#print(Df.shape)

#df.to_pickle('vwc_10-16-2018')
#df.drop_duplicates(subset = 'site').drop(['fuel','percent','date'], axis = 1).to_csv('fuel_moisture/site_info_query_10-16-2018.csv')

