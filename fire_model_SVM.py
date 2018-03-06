
from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dirs import MyDir
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


#from scipy.stats import gaussian_kde
#import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
#from scipy import signal
                                                                                                                                      #thresh_range=[90]


#======================================================
# PREPARE TRAINING SET
#features: enso, iod, landcover1, landcover2, MTDCA, mySCA2, nl, precip, temp, vod
#response: fire

os.chdir('C:/Users/kkrao/Dropbox/fire_proj_data')
np.random.seed(2)
#list variables
data_types=['et', 'peat', 'pet', 'precip', 'q', 'temp', 'sm','vod','fire']
#data_types.remove('vod') ## just tried to remove vod so that we have data til 2016. but did not help
data_types_names=['Actual Evapotranspiration','Peat Lands','Potential Evapotranspiration',\
              'Precipitation','Specific Humidity','Temperature','Soil Moisture','Vegetation Optical Depth','Fire']
data_type_dict=dict(zip(data_types,data_types_names))
data_type_dates=[source+'dates' for source in data_types]
### which dataset extends only till 2013?

for source in data_type_dates:
    dates = np.load(source+'.npy')
    maxdate = pd.to_datetime(max(dates))
#    print('%s max date %s'%(source, maxdate))

#initialize general start and end date time
start_date, end_date = pd.to_datetime('1900-01-01'), pd.to_datetime('2100-01-01')

#search through dates and find start and end dates that are at intersection of all data types
for source in data_type_dates:
    dates = np.load(source+'.npy')
    mindate = pd.to_datetime(min(dates))
    maxdate = pd.to_datetime(max(dates))
    if mindate > start_date:
        start_date = mindate
    if maxdate < end_date:
        end_date = maxdate
#    print('%s min date is: %s'%(source, mindate))
#create lat lon vector vector
lat = np.load('fire_lat.npy')
lon = np.load('fire_lon.npy')
lon,lat=np.meshgrid(lon,lat)
latcorners=[331, 387]
loncorners=[1097, 1200]
lon,lat=lon[latcorners[0]:latcorners[1],loncorners[0]:loncorners[1]],\
       lat[latcorners[0]:latcorners[1],loncorners[0]:loncorners[1]]

start_date=pd.to_datetime('2015-01-01')     
#initialize dataframe by adding lat, lon, and months
df = pd.DataFrame(lat.flatten(), columns=['lat'])
df['lon'] = lon.flatten()
num_months = (end_date.year - start_date.year)*12 + (end_date.month)
df = df.append([df]*(num_months-1), ignore_index=True)



#add all features into m x n matrix
for source in data_types:

    #load one dataset
    data = np.load(source+'.npy')
    dates = np.load(source+'dates.npy')
    
    #keep only those dates between the start and end date
    data = data[(pd.to_datetime(dates)>=start_date) & \
                (pd.to_datetime(dates)<=end_date),:,:]
    
    #add to dataframe
    data_df = pd.DataFrame(data.flatten(),columns=[source])
    df = pd.concat([df,data_df],axis=1)

#remove examples where any data is missing
df.dropna(inplace=True)
#df.fillna(method='bfill',inplace=True)
#df.fillna(method='ffill',inplace=True)
df.fire/=67500
#for classification
#of nonzero values, 25 percntile = 29.6, 50th=66, 75th = 171.4
    
### subset rows you dont want-===================================================
#df=df.loc[df.fire>0,:]
df=df.loc[df.peat==1,:]


###===========================================================================
thresh1=0
thresh_range= np.arange(1e-3,1e-1,1e-3)
thresh_range=[0.0960]
thresh_range=[1e-2]
for thresh2 in thresh_range:
    
    dfsub=df.copy()
    dfsub.loc[df.fire<=thresh1,'fire']='no fire'
    dfsub.loc[(df.fire>thresh1) & (df.fire<=thresh2),'fire']='small fire'
    dfsub.loc[df.fire>thresh2,'fire']='large fire'
    #df.loc[(df.fire<=30) & (df.fire>5),'fire']= 1
    #df.loc[(df.fire<=66) & (df.fire>30),'fire']= 2
    #df.loc[(df.fire<=172) & (df.fire>66),'fire']= 3
    #df.loc[df['fire']>172,'fire'] = 4 # for classification
    
    
    #split into features and output
    X = dfsub.drop(['fire','lat','lon'], axis=1)
    y = dfsub['fire']
    
    ##PLOT info on data
    #info = df.drop(['lat','lon','date'], axis=1)
    #ax=info.hist(figsize=(10,8),xlabelsize=12,ylabelsize=12)
    #ax = info.boxplot(return_type='axes')
    #ax.set_ylim((-2,4))
    
    
    #MORE PREPROCESSING OF DATA
    #CATEGORICAL DATA: get dummies and add them as columns to the features
#    cat_data = ['peat','landcover1','landcover2']
#    peat_dum = pd.get_dummies(X['peat'])
#    lc1_dum = pd.get_dummies(X['landcover1'])
#    lc2_dum = pd.get_dummies(X['landcover2'])
#    #remove the features
#    subset = X.drop(cat_data,axis=1)
#    X = pd.concat([subset,peat_dum,lc1_dum,lc2_dum], axis=1)
#    #combine the overlapping columns in the 2 majority landcover classes
#    X = X.groupby(lambda x:x, axis=1).sum()
       
    
    
    #split into training and test
    
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    #Standardize train and test data
    
    X_train_scale=scale(X_train)
    X_test_scale=scale(X_test)
    
    
    
    
    
    # Fitting a logistic regression model
    log_model = LogisticRegression(penalty='l2',C=1)
    log_model.fit(X_train_scale,y_train)
    
    # Checking the model's accuracy
    y_pred=log_model.predict(X_test_scale)
    score = accuracy_score(y_test,y_pred)
    print('For thresh = %0.4f, accuracy = %0.2f'%(thresh2, score))
    fig, ax= plt.subplots(figsize=(2,3))
    sns.heatmap(confusion_matrix(y_test,y_pred,labels=\
             ['no fire', 'small fire','large fire']), \
    annot=True, fmt='d',square=True,ax=ax)
#    from pandas.plotting import scatter_matrix
    #scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')



#======================================================
#
## REGRESSIONS
#
#neigh = KNeighborsRegressor(n_neighbors=100)
#neigh.fit(X, Y) 
#Y_hat=neigh.predict(X)
#
##model = LogisticRegression()
##model.fit(X=X,y=Y)
##Y_hat = model.predict(X)
#
#
#
### plots
#fig, axs = plt.subplots(4,4,sharey=True,figsize=(6,6))
#plt.subplots_adjust(hspace=0.5,top=0.93)
#axs=axs.ravel()
#i=0
#for source in X.columns:
#    ax=axs[i]
#    df.plot.scatter(source,'fire',5,'k',ax=ax,alpha=0.3)
#    inds=np.argsort(X[source])
#    ax.plot(X[source][inds],Y_hat[inds],'r-',lw=1)
#    i+=1
##for j in range(i, 8):
##    axs[j].axis("off")
#plt.suptitle('Locally-weighted Linear Regression')
#
#
#
#
#
#    

#for plotting
#df['fire'].plot(kind='hist')
