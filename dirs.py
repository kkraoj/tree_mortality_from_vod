# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:50:05 2017

@author: kkrao
"""

from __future__ import division
#from IPython import get_ipython
#get_ipython().magic('reset -sf') 
import plotsettings
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.io
import os
import arcpy
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from arcpy.sa import *
import pylab
import h5py
import urllib
import urllib2
from ftplib import FTP 
from subprocess import Popen, PIPE
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab
from matplotlib.patches import Rectangle  
import seaborn as sns
from sklearn import datasets, linear_model
from scipy.stats import gaussian_kde
from sklearn import datasets, linear_model
lm = linear_model.LinearRegression(fit_intercept=True)
from matplotlib.ticker import FormatStrFormatter
from scipy import optimize
import pickle
from matplotlib import ticker
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from IPython.display import display, HTML
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar

    
MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'
Dir_NLDAS=MyDir+'/NLDAS'

def box_equal_nos(x,y,boxes,thresh):
#    x=data_anomaly # for debugging only
#    y=mort
    x=x.values.flatten(); y=y.values.flatten()
    inds=x.argsort()
    x=x.take(inds)  ;y=y.take(inds)
#    x=x[inds];  y=y[inds]
    inds=np.where(~np.isnan(x))[0]
    x=x.take(inds)  ;y=y.take(inds)
#    x=x[inds];  y=y[inds]
    inds=np.where(y>=thresh)[0]
    x=x.take(inds)  ;y=y.take(inds)
#    x=x[inds];  y=y[inds]
    x_range=x.max()-x.min()
    if x_range/boxes < 0.1:
        round_digits=3
    elif x_range/boxes < 1:
        round_digits=2
    else: 
        round_digits=0
    count=len(x)/boxes
    count=np.ceil(count).astype(int)
    yb=pd.DataFrame()
    for i in range(boxes):
        data=y[i*count:(i+1)*count]
        name=np.mean(x[i*count:(i+1)*count]).round(round_digits)
        data=pd.DataFrame(data,columns=[name])
        yb=pd.concat([yb,data],axis=1)
    return yb



def add_squares(axes, x_array, y_array, size=0.5, **kwargs):
    size = float(size)
    for x, y in zip(x_array, y_array):
        square = pylab.Rectangle((x-size/2,y-size/2), size, size, **kwargs)
        axes.add_patch(square)
    return True

def get_marker_size(ax,fig,loncorners,grid_size,marker_factor):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    marker_size=width*grid_size/100/np.diff(loncorners)[0]/4*marker_factor
    return marker_size


def median_anomaly(Df):
    mean=Df.groupby(Df.index.dayofyear).mean()
    sd=Df.groupby(Df.index.dayofyear).std()
    Df_anomaly=pd.DataFrame()
    for year in np.unique(Df.index.year):
        a=Df[Df.index.year==year]
        a.index=a.index.dayofyear
        anomaly=((a-mean)/sd)
        anomaly=anomaly.median()
        anomaly.name=pd.Timestamp(year,1,1)
        Df_anomaly=pd.concat([Df_anomaly,anomaly],1)
    Df_anomaly=Df_anomaly.T
    return Df_anomaly    
def min_anomaly(Df):
    mean=Df.groupby(Df.index.dayofyear).mean()
    sd=Df.groupby(Df.index.dayofyear).std()
    Df_min_anomaly=pd.DataFrame()
    for year in np.unique(Df.index.year):
        a=Df[Df.index.year==year]
        a.index=a.index.dayofyear
        min_anomaly=((a-mean)/sd).min()
        min_anomaly.name=pd.Timestamp(year,1,1)
        Df_min_anomaly=pd.concat([Df_min_anomaly,min_anomaly],1)
    Df_min_anomaly=Df_min_anomaly.T
    return Df_min_anomaly

  

def clean_xy(x,y,rep_times=1,thresh=0.0):
    from scipy.stats import gaussian_kde
    # for testing ONLY
#    x=data_anomaly.values.flatten()
#    y=np.log10(mort.values.flatten())
    non_nan_ind=np.where(~np.isnan(x))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)
    non_nan_ind=np.where(~np.isnan(y))[0]
    x=x.take(non_nan_ind);y=y.take(non_nan_ind)
    inds=np.where(y>=thresh)[0]
    x=x.take(inds)  ;y=y.take(inds)
    x=np.repeat(x,rep_times);y=np.repeat(y,rep_times)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
#    x, y, z = np.reshape(x,(len(x),1)), np.reshape(y,(len(y),1)),\
#                        np.reshape(z,(len(z),1))
    return x,y,z

#year='2015'
#grid='grid'
#table=Dir_mort+"/"+grid+".gdb/ADS"+year[-2:]+"_i_j" 
#columns=['gridID','TPA1','Shape_Area','HOST1','HOST2','FOR_TYPE1']
def build_df_from_arcpy(table, columns='all'):
    if columns=='all':
        columns=[f.name for f in arcpy.ListFields(table)]
    cursor = arcpy.SearchCursor(table)
    Df=pd.DataFrame(columns=columns)
    for row in cursor:
        data=pd.DataFrame([row.getValue(x) for x in columns],index=columns,dtype='str').T
        Df=Df.append(data)
    arcpy.Compact_management(Dir_mort+'/species.gdb')
    return Df

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def ind(thresh,mort):
    ind=[l for l in mort.columns if mort.loc['2016-01-01',l] >=thresh]
    return ind

def ind_species(species):
    #input 'c' or 'd' and get indices
    os.chdir(Dir_CA)
    with open('grid_to_species.txt', 'rb') as fp:
        for_type = pickle.load(fp)
    out=[i[0] for i in for_type if i[1]==species]
    return out

def ind_small_species(species):
    #input 'c' or 'd' and get indices
    os.chdir(Dir_CA)
    with open('small_grid_%s.txt'%species, 'rb') as fp:
        out = pickle.load(fp)
    return out

def mask_columns(columns=None,*dataframes):
#    df=mort
#    columns=ind_small_species(species)
    i=0
    out=range(len(dataframes))
    for df in dataframes:                    
        mask = ((df == df) | df.isnull()) & (df.columns.isin(columns))
        df=df.mask(~mask)
#        df.fillna(0,inplace=True)
        out[i]=df
        i+=1
    if i==1:
        out=out[0]
    return(out)

def year_anomaly_mean(Df): #anomaly of mean
    mean=Df.mean()
    sd=Df.std()
    Df_anomaly=pd.DataFrame()
    for year in np.unique(Df.index.year):
        a=Df[Df.index.year==year].mean()
#        a.index=a.index.dayofyear
        anomaly=((a-mean)/sd)
#        anomaly=anomaly.median()
        anomaly.name=pd.Timestamp(year,1,1)
        anomaly.replace([np.inf, -np.inf], 0,inplace=True)
        Df_anomaly=pd.concat([Df_anomaly,anomaly],1)
    Df_anomaly=Df_anomaly.T
    return Df_anomaly   

def mean_anomaly(Df): #mean of anomaly
    mean=Df.groupby(Df.index.dayofyear).mean()
    sd=Df.groupby(Df.index.dayofyear).std()
    Df_anomaly=pd.DataFrame()
    for year in np.unique(Df.index.year):
        a=Df[Df.index.year==year]
        a.index=a.index.dayofyear
        anomaly=((a-mean)/sd)
        anomaly=anomaly.mean()
        anomaly.replace([np.inf, -np.inf], 0,inplace=True)
        anomaly.name=pd.Timestamp(year,1,1)
        Df_anomaly=pd.concat([Df_anomaly,anomaly],1)
    Df_anomaly=Df_anomaly.T
    return Df_anomaly


def RWC(Df,upper_quantile=0.95,start_month=7, months_window=3,start_year=2009):
    Df=Df[Df.index.year>=start_year]
    Df=Df.loc[(Df.index.month>=start_month) & (Df.index.month<start_month+months_window)]
    out=(Df.groupby(Df.index.year).quantile(0.5)-Df.quantile(1-upper_quantile))/\
        (Df.quantile(upper_quantile)-Df.quantile(1-upper_quantile))
    out[(out>1.0)]=np.nan
    out[(out<0.0)]=np.nan   
    out.index=pd.to_datetime(out.index,format='%Y')
    return out

def log_anomaly(Df):
    out=np.log10((Df.groupby(Df.index.year).median()/Df.quantile(0.95)))
#    out[(out>1.0)]=np.nan
#    out[(out<0.0)]=np.nan        
    out.index=pd.to_datetime(out.index,format='%Y')
    return out

def median_div_max(Df):
    out=((Df.groupby(Df.index.year).median()/Df.quantile(0.95)))
#    out[(out>1.0)]=np.nan
#    out[(out<0.0)]=np.nan        
    out.index=pd.to_datetime(out.index,format='%Y')
    return out

def min_div_max(Df):
    out=((Df.groupby(Df.index.year).quantile(0.05)/Df.quantile(0.95)))
#    out[(out>1.0)]=np.nan
#    out[(out<0.0)]=np.nan        
    out.index=pd.to_datetime(out.index,format='%Y')
    return out

def cwd_accumulate(df,start_year,end_year):
    df=df.loc[(df.index.year<=end_year) & (df.index.year>=start_year)]
    return df.sum()

def append_prediction(name='rf_predicted'):
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
    df=pd.read_csv('D:/Krishna/Project/data/%s.csv'%name,index_col=0)
    df=df['predicted_FAM']
    length=store['RWC'].shape[1]
    df=df.reindex(range(length*7))
    df=pd.DataFrame(df.values.reshape((int(len(df)/length),length),order='F'),\
                    columns=store['RWC'].columns)
    df.index=pd.to_datetime(df.index+2009,format='%Y')
    df.index.name='predicted_FAM'
    store[df.index.name]=df
    return df
         
def import_mort_leaf_habit(species,grid_size=25,start_year=2009,end_year=2015):
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
    mort=store['mortality_%s_%03d_grid'%(species,grid_size)]
    mort=mort[mort>0]
    mort=mort[(mort.index.year>=start_year) &\
          (mort.index.year<=end_year)]
    return mort

def subset_forest_cov(Df):
    fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx')
    Df=Df[fc.gridID]
    return Df