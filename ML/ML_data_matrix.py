# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:55:55 2017

@author: kkrao
"""

from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dirs import MyDir
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
"""
features:
    et, nl, pet, precip, q, temp, sm, vod
response:
    fire
"""
os.chdir('C:/Users/kkrao/Dropbox/fire_proj_data')
data_types=['et', 'nl', 'pet', 'precip', 'q', 'temp', 'sm','vod','fire']
data_types_names=['Actual Evapotranspiration','Night Lights','Potential Evapotranspiration',\
                  'Precipitation','Specific Humidity','Temperature','Soil Moisture','Vegetation Optical Depth','Fire']
data_type_dict=dict(zip(data_types,data_types_names))
data_type_dates=[source+'dates' for source in data_types]

start_date, end_date=pd.to_datetime('1900-01-01'),pd.to_datetime('2100-01-01')
## choose start and end dates of intersection of all data types
for source in data_type_dates:
    dates=np.load(source+'.npy')
    if pd.to_datetime(min(dates))>start_date:
        start_date=pd.to_datetime(min(dates))
    if pd.to_datetime(max(dates))<end_date:
        end_date=pd.to_datetime(max(dates))

#make m by n matrix in pandas data frame
## first just put lat lons
lat = np.load('fire_lat.npy')
lon = np.load('fire_lon.npy')
lon,lat=np.meshgrid(lon,lat)
latcorners=[331, 387]
loncorners=[1097, 1200]
lon,lat=lon[latcorners[0]:latcorners[1],loncorners[0]:loncorners[1]],\
           lat[latcorners[0]:latcorners[1],loncorners[0]:loncorners[1]]
           
Df=pd.DataFrame(lat.flatten(),columns=['lat'])
Df['lon']=lon.flatten()
number_of_months=(end_date.year - start_date.year)*12 +\
                 (end_date.month - start_date.month)+1
Df=Df.append([Df]*(number_of_months-1), ignore_index=True)
##Now put all other features and response at the end
for source in data_types:    
    df=np.load(source+'.npy')
    np.shape(df)
    dates=np.load(source+'dates.npy')
    df=df[(pd.to_datetime(dates)>=start_date)&\
          (pd.to_datetime(dates)<=end_date),:,:]
    df=pd.DataFrame(df.flatten(),columns=[source])
    Df=pd.concat([Df,df],axis=1)
## ignore ocean and unavilable data
Df.dropna(inplace=True)
Df['fire']/=62500
Df=Df.loc[Df['fire']>0.001,:] # for regression
X=Df.drop(['fire','lat','lon'],axis=1) # for regression
X = (X - X.mean(axis=0)) /X.std(axis=0) # for regression
Y=Df['fire'] # for regression
#Df.loc[Df['fire']>=1000,'fire']='Yes' # for classification
#Df.loc[Df['fire']<1000,'fire']='No' # for classification
#Df.drop(['lat','lon'],axis=1,inplace=True) # for classification
#Df.drop(['nl','temp',],axis=1,inplace=True) # for classification
#Df[Df.columns.difference(['fire'])] = (Df[Df.columns.difference(['fire'])]\
#  - Df[Df.columns.difference(['fire'])].mean(axis=0)) /\
#       Df[Df.columns.difference(['fire'])].std(axis=0) # for classification
#X=Df.drop(['fire'],axis=1) # for class
#Y=Df['fire'] # for class

# linear regression============================================================
#lm = linear_model.LinearRegression()
#lm.fit(X,Y)
#coefs=pd.DataFrame(lm.coef_,columns=['Coefs'],index=X.columns)
#coefs.index.name='Features'
#print(coefs)
#fig, axs =plt.subplots(3,3,sharey=True,figsize=(6,6))
#plt.subplots_adjust(hspace=0.5,top=0.93)
#axs=axs.ravel()
#i=0
#for source in X.columns:
#    ax=axs[i]
#    Df.plot.scatter(source,'fire',5,'k',ax=ax,alpha=0.3)
#    xd=np.linspace(min(X[source]),max(X[source]),100)
#    ax.plot(xd,xd*coefs.loc[source][0]+lm.intercept_,'r-',lw=2)
#    i+=1
#for j in range(i, 9):
#    axs[j].axis("off")
#plt.suptitle('Ordinary Linear Regression')

# Local regression ============================================================
neigh_range=np.arange(10,50,5)
LWR_rsq=pd.DataFrame(np.zeros(len(neigh_range)), columns=['r_squared'],index=neigh_range)
LWR_rsq.index.name='No. of neighbors'
for neigh_iter in neigh_range:
    neigh = KNeighborsRegressor(n_neighbors=neigh_iter)
    X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=42)
    neigh.fit(X_train, y_train) 
    Y_hat=neigh.predict(X_test)
    SS_total=sum((y_test - np.mean(y_train))**2)
    SS_residual=sum((y_test - Y_hat)**2)
    SS_regression=sum((Y_hat - np.mean(y_train))**2)
    rsq=1 - SS_residual/SS_total
    LWR_rsq.loc[neigh_iter]=rsq

sns.set_style('ticks')
fig, ax = plt.subplots(figsize=(2,2))
LWR_rsq.plot(style='--ko',ax=ax,legend=False)
ax.set_ylabel("CV set $R^2$")
ax.set_title('K-Neighbors selection')
ax.set_xlim([0,50])
ax.set_ylim([0.2,0.4])
ax.axvline(15,linewidth=1, color='r')
ax.annotate('K=15'%rsq, xy=(0.48, 0.25), xycoords='axes fraction',\
            ha='right',va='top',size=8,color='r')

neigh = KNeighborsRegressor(n_neighbors=60)
X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size=0.2, random_state=42)
neigh.fit(X_train, y_train) 
Y_hat=neigh.predict(X_test)
SS_total=sum((y_test - np.mean(y_train))**2)
SS_residual=sum((y_test - Y_hat)**2)
SS_regression=sum((Y_hat - np.mean(y_train))**2)
rsq=1 - SS_residual/SS_total
           
# plots
fig, axs =plt.subplots(2,4,sharey=True,figsize=(8,4))
plt.subplots_adjust(hspace=0.5,top=0.91)
axs=axs.ravel()
i=0
for source in X.columns:
    ax=axs[i]
    ax.scatter(X_test[source],y_test,5,'k',alpha=0.4)
    ax.set_xlabel(data_type_dict[source])
    inds=np.argsort(X_test[source])
    X_test.sort_values(source,inplace=True)
    plot_line=ax.plot(X_test[source],Y_hat[inds],'ro',alpha=0.5,markersize=1,lw=1,label='Predictions')
    if i==0 or i==4:
        ax.set_ylabel('fire ($m^2$)')
    i+=1
for j in range(i, 8):
    axs[j].axis("off")
lgnd =ax.legend(handles=plot_line,)
lgnd.legendHandles[0]._legmarker.set_markersize(6)
#ax.plot(-1.2,27000,'ro',markersize=5)
plt.suptitle('Locally-weighted Linear Regression')

##neural net===================================================================
#mlp = MLPRegressor()
#mlp.fit(X,Y)
#Y_hat=mlp.predict(X)

## Logistic regression==========================================================
#logreg = linear_model.LogisticRegression()
#logreg.fit(X, Y)
#
## classification plots
#sns.set_style('ticks')
#sns.pairplot(Df,hue='fire',size=1.5,plot_kws={'s':3,'alpha':0.1,'edgecolor':'none'},\
#             palette=sns.color_palette("seismic", 2))


# Model Performance============================================================

sns.set_style('ticks')
fig,ax=plt.subplots(figsize=(2,2))
ax.scatter(y_test,Y_hat,5,'k',alpha=0.3)
ax_range=[0,0.5]
ax.set_xlim(ax_range)
ax.set_ylim(ax_range)
ax.plot(ax_range,ax_range,color='grey',lw=0.6)
ax.set_xlabel('Actual Fire ($m^2$)')
ax.set_ylabel('Predicted Fire ($m^2$)')
ax.annotate('1:1 line', xy=(0.9, 0.95), xycoords='axes fraction',\
            ha='right',va='top',size=9)
ax.annotate('$R^2$ = %0.2f'%rsq, xy=(0.1, 0.75), xycoords='axes fraction',\
            ha='left',va='top',size=11,color='r')
ax.set_title('Performance, test set')

#sns.set_style('ticks')
#fig,ax=plt.subplots(figsize=(4,4))
#z=Df['RWC']
#plot=ax.scatter(Df[var1],Df[var2],marker='s',c=z,cmap=cmap)
#ax.set_xlim(var1_range)
#ax.set_ylim(var2_range)
#ax.set_xlabel(var1_label)
#ax.set_ylabel(var2_label)
#ax.plot(var1_range,var2_range,color='grey',lw=0.6)
#cbaxes = fig.add_axes([0.2, 0.60, 0.03, 0.1])
#cb=fig.colorbar(plot,ax=ax,\
#                ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
#cb.ax.set_yticklabels(['Low', 'High'])
#cb.ax.tick_params(axis='y', right='off')
#cbaxes.annotate('RWC',xy=(0,1.2), xycoords='axes fraction',\
#            ha='left')
#cb.outline.set_visible(False)
#ax.annotate('1:1 line', xy=(0.9, 0.95), xycoords='axes fraction',\
#            ha='right',va='top')
    
####==================== extend peat dates till 2000
data=np.load('peat.npy')
data=data[:12,:,:]
np.shape(data)
datadates=np.load('firedates.npy')
n=len(datadates)/12
data= data.repeat(n,axis=0)
np.save('peat_l.npy', data)
np.save('peat_ldates.npy',datadates)














