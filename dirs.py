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

def get_marker_size_v2(ax,fig,loncorners,grid_size=0.25,marker_factor=1.):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    width *= fig.dpi
    marker_size=width*grid_size/np.diff(loncorners)[0]*marker_factor
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
def build_df_from_arcpy(table, columns='all',dtype=None, index_col = None):
    if columns=='all':
        columns=[f.name for f in arcpy.ListFields(table)]
    cursor = arcpy.SearchCursor(table)
    Df=pd.DataFrame(columns=columns)
    for row in cursor:
        data=pd.DataFrame([row.getValue(x) for x in columns],index=columns).T # removed dtype
        Df=Df.append(data)
    Df=Df.astype(dtype)
    if index_col != None:
        Df.index = Df[index_col]
        Df.drop(index_col, axis = 1, inplace = True)
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
    out.index.name='RWC'
    out.columns.name='gridID'
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

def append_prediction(name='rf_predicted',filename='data_subset_GC.h5'):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
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

def subset_forest_cov(Df,landcover = 'GC_subset'):
    fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',sheetname=landcover)
    Df=Df[fc.gridID]
    return Df

def append_color_importance(Df):
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    climate=['CWD','ppt_sum','ppt_win','tmax_sum','tmax_win',\
             'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
            'PEVAP_sum','EVP_win','PEVAP_win','vsm_sum','vsm_win']
    veg=['mortality_025_grid','live_basal_area','LAI_sum',\
            'LAI_win','RWC','canopy_height','forest_cover',]
    topo=['aspect_mean', 'aspect_std','elevation_mean','elevation_std','location']
    Df['color']=None
    Df.loc[Df.index.isin(climate),'color']=blue
    Df.loc[Df.index.isin(veg),'color']=green
    Df.loc[Df.index.isin(topo),'color']=brown
    return Df

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
#    else:
        # no yaxis ticks
#        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
#    else:
        # no xaxis ticks
#        ax.xaxis.set_ticks([])
#checking

def make_lat_lon(gt):
    lats=np.arange(start=gt[3]+gt[5]/2,stop=-gt[3]+gt[5]/2, step=gt[5])
    lons=np.arange(start=gt[0]+gt[1]/2,stop=-gt[0]+gt[1]/2, step=gt[1])
    lats, lons = np.meshgrid(lats,lons,indexing='ij')
    return lats,lons

def bar_label(heights,ax,color='k',x_offset=0,y_offset=0):
    for i, v in enumerate(heights):
        ax.text(v + x_offset, i + y_offset, '%0.2f'%v, \
                color=color, fontweight='bold',va='center',ha='left')


def remove_vod_affected(Df):
    aff_start='2011-10-01'
    aff_end='2012-08-01'
    Df.loc[aff_start:aff_end]=np.nan
    return Df

def supply_lat_lon(landcover = 'GC_subset'):
    fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',sheetname=landcover)
    lat=fc.y
    lon=fc.x
    return lat,lon

def select_high_mort_grids(Df,remove_nans=False):
    high_mort = [ 33,  73,  83,  84,  91,  97, 104, 105, 117, 118, 128, 129, 130,
                137, 138, 139, 141, 149, 150, 151, 160, 161, 170, 171, 172, 173,
                181, 182, 183, 184, 195, 196, 197, 198, 207, 208, 209, 218, 219,
                220, 221, 227, 230, 231, 232, 233, 240, 258, 259, 260, 261, 268,
                272, 273, 274, 277, 281, 287, 288, 289, 290, 296, 304, 308, 311,
                312, 317, 320, 326, 328, 334, 335, 336, 343, 348, 349, 350]
    Df=Df.loc[:,high_mort]
    if remove_nans:
        Df.dropna(axis=1, inplace=True)
    return Df

def select_bounding_box_grids(Df, thresh=0.25):
    ## Thresh equals the distance from each side of unit square
    ## to bounding box
    
    low_thresh=0.+thresh
    high_thresh=1.-thresh
    lc=pd.read_excel('D:/Krishna/Project/working_tables.xlsx',\
                 sheetname='gc_ever_deci',index_col=1)  
    index=lc.loc[(lc.evergreen>=low_thresh)&
           (lc.evergreen<=high_thresh)&
           (lc.deciduous>=low_thresh)&
           (lc.deciduous<=high_thresh)].index
    Df=Df.loc[:,index]
    return Df

def select_years(start_year =2009, end_year = 2015, *Dfs):
    '''
    Function will take unknown number of timeseries indexed pandas and 
    output dataframes within start_year and end_year
    '''
    if len(Dfs)<=1:
        Df=Dfs[0]
        Df=Df[(Df.index.year>=start_year) &\
              (Df.index.year<=end_year)]
        out=Df
    else:   
        out=range(len(Dfs))
        i=0
        for Df in Dfs:
            Df=Df[(Df.index.year>=start_year) &\
                      (Df.index.year<=end_year)]
            out[i]=Df
            i+=1
    return out

def select_north_south_grids(Df):
    north_grids = [ 137, 138, 139, 141, 145, 149, 150, 151, 160, 161, 166, 170, 171, 172, 173, \
                   174, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, \
                   192, 193, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, \
                   209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, \
                   226, 227, 228, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 242, 243, \
                   244, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, \
                   260, 261, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, \
                   278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, \
                   294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 308, 309, 310, \
                   311, 312, 313, 314, 315, 316, 317, 318, 320, 323, 324, 325, 326, 327, 328, \
                   329, 330, 331, 332, 333, 334, 335, 336, 340, 341, 342, 343, 344, 345, 348, \
                   349, 350, 357, 358, 359, 360, 361, 362, 363, 365]
    
    north_grids = [188, 189, 190, 191, 192, 193, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 320, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 340, 341, 342, 343, 344, 345, 348, 349, 350, 357, 358, 359, 360, 361, 362, 363, 365, 328, 329, 330, 331, 332, 333, 334, 335, 336, 340, 341, 342, 343, 344, 345, 348, 349, 350, 357, 358, 359, 360, 361, 362, 363, 365]
#    north_grids=[177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 320, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 340, 341, 342, 343, 344, 345, 348, 349, 350, 357, 358, 359, 360, 361, 362, 363, 365 ]
    ### ADMP regions
    north_grids = [183, 184, 185, 186, 188, 189, 195, 196, 197, 198, 199, 201, 202, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 320, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 340, 341, 342, 343, 344, 345, 348, 349, 350, 357, 358, 359, 360, 361, 362, 363, 365]
    Df_north = Df.loc[:, Df.columns.isin(north_grids)]
    Df_south = Df.loc[:, ~Df.columns.isin(north_grids)]

    return Df_north, Df_south

def initiliaze_plot(journal = 'EcolLett', fs = 12):
    fontsizes=['font.size','lines.markersize',
     'legend.fontsize',
     'axes.labelsize','xtick.labelsize','ytick.labelsize']
    publishable = plotsettings.Set(journal)
    for x in fontsizes:
        plotsettings.journals.journals['EcolLett']['rcParams'][x]=fs
    return publishable
                                
def scatter_threshold(x, y, ax,  panel_label, cmap = 'viridis', alpha = 1, scatter_size = 10, \
                      mort_label = 'FAM (-)', y_range = [0,0.5], x_range = [0,1], FAM_thresh = 0.,\
                        guess = (.3,0.05,1e-1,1e-1)):
    if '%s'%type(x)=="<class 'pandas.core.frame.DataFrame'>":
        x=x.values.flatten()
    if '%s'%type(y)=="<class 'pandas.core.frame.DataFrame'>":
        y=y.values.flatten()
    x,y,z=clean_xy(x,y, thresh = FAM_thresh)
    plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_ylabel(mort_label)
    guess=guess
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=y_range[0], y_range[1]
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f; Threshold = %0.2f; Error = %0.2f'%(panel_label, r_squared, popt[0], perr[0]))
    
    ax.annotate('%s'%panel_label, xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
#    ax.set_aspect('equal')
    return plot_data, z
    
def select_forest_type_grids(forest, fortype, *Dfs):
    if len(Dfs)<=1:
        Df=Dfs
        Df = Df[fortype==forest].values.flatten()
        out=Df
    else:   
        out=range(len(Dfs))
        i=0
        for Df in Dfs:
            Df = Df[fortype==forest].values.flatten()
            out[i]=Df
            i+=1
    return out
    
    