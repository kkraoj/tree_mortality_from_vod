# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:11:14 2017

@author: kkrao
"""
from __future__ import division
import os
import plotsettings
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from scipy import stats
from matplotlib_scalebar.scalebar import ScaleBar
from dirs import Dir_CA, Dir_mort, get_marker_size, import_mort_leaf_habit, clean_xy,\
                    piecewise_linear,append_prediction, \
                    append_color_importance, adjust_spines,supply_lat_lon,\
                    select_high_mort_grids, select_bounding_box_grids,\
                    select_north_south_grids, select_years, scatter_threshold, \
                    Dir_ms_fig, plot_map, breakpoint, variable_type_color,\
                    var_dict, select_non_ocean_pixels
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap
from scipy import optimize
from matplotlib.patches import Rectangle, Patch
from datetime import timedelta
import types
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

#mpl.rcParams.update({'figure.autolayout': True})
#
cardinal='#BD2031'
zoom=1.0
fs=14
journal='GlobChangeBio'


#mpl.rcParams['font.size'] = 32
ppt = 0
dpi = 300
start_year=2009
end_year=2015
os.chdir(Dir_CA)
filename = 'data_subset_GC.h5'
append_prediction(filename=filename)
def unit_of(key):
    keys = ['RWC',
             'ppt_win',
             'elevation_mean',
             'EVP_win',
             'vpdmax_win',
             'elevation_std',
             'CWD',
             'tmax_win',
             'tmax_sum',
             'tmean_win',
             'tmean_sum',
             'live_basal_area',
             'vpdmax_sum',
             'aspect_mean',
             'twi_std',
             'ppt_sum',
             'LAI_win',
             'PEVAP_win',
             'twi_mean',
             'EVP_sum',
             'aspect_std',
             'location',
             'LAI_sum',
             'PEVAP_sum',
             'vsm_win',
             'silt_fraction',
             'canopy_height',
             'vsm_sum',
             'sand_fraction',
             'forest_cover',\
             'ndwi_sum','ndwi_win']
    values = ['(-)','(mm)','(m)','(mm)','(hPa)','(m)','(mm)','($^o$C)',\
              '($^o$C)','($^o$C)','($^o$C)','(m$^2$/ha)','(hPa)','($^o$N)',\
              '(m)','(mm)','(-)','(mm)','(m)','(mm)','(m)','(-)','(-)','(mm)',\
               '(-)','(-)','(m)','(-)','(-)','(-)','(-)','(-)']
    dictionary = dict(zip(keys, values))
    return dictionary[key]

def change_font_size(fs = 14):
    fontsizes=['font.size','lines.markersize',
               'legend.fontsize',
               'axes.labelsize','xtick.labelsize','ytick.labelsize']
    publishable = plotsettings.Set(journal)
    for x in fontsizes:
        plotsettings.journals.journals[journal]['rcParams'][x]=fs
        publishable = plotsettings.Set(journal)
    return publishable
publishable = change_font_size(fs)    
    
    
def flatten_diagonally(npA, diagonals = None):
    diagonals = diagonals or xrange(-npA.shape[0] + 1, npA.shape[1])
    return np.concatenate(map(lambda x: \
          np.diagonal(np.rot90(npA), offset = x), diagonals))

def plot_timeseries_maps(var1='mortality_%03d_grid', var1_range=[1e-5, 0.40],\
                         var1_label="Observed",\
                    var2='predicted_FAM', var2_range=[1e-5, 0.40],\
                     var2_label="Modeled",\
                    grid_size=25,cmap='inferno_r',cmap2='inferno_r', \
                    ticks=5,\
                    title='Timeseries of observed and predicted mortality',proj='cyl'):
    os.chdir(Dir_CA)
#    sns.set(font_scale=1.2)
#    mpl.rcParams['font.size'] = 15
    store=pd.HDFStore(filename)
    data_label=var2_label
    alpha=0.7
    mort_label=var1_label
    mort=store[var1%(grid_size)]
#    mort=mort[mort>0]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    pred_mort=append_prediction(filename=filename)
    year_range=mort.index.year
    cols=mort.shape[0]
    rows=2
    latcorners=np.array([33,42.5])
    loncorners=np.array([-124.5,-117]) 
    ###plottiing onle non ocean pixels
    lats,lons=supply_lat_lon('GC_subset_no_ocean')
    mort = select_non_ocean_pixels(mort)
    pred_mort = select_non_ocean_pixels(pred_mort)
    ####################
    sns.set_style("ticks")
    publishable.set_figsize(2*zoom, 1*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=rows,ncols=cols   ,\
                            sharey='row')
#    marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
    marker_size = 4
    plt.subplots_adjust(wspace=0.04,hspace=0.04,top=0.83)
    
    

    for year in year_range:   
        mort_plot=mort[mort.index.year==year]
        ax=axs[0,year-year_range[0]]
#        ax.set_title(str(year))
        ax.annotate(str(year), xy=(0.5, 1.03), xycoords='axes fraction',\
                ha='center',va='bottom')
        m = Basemap(projection=proj,lat_0=45,lon_0=0,resolution='l',\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                ax=ax)
        m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
        plot_mort=m.scatter(lons, lats,s=marker_size,c=mort_plot,cmap=cmap,\
                            marker='s',\
                            vmin=var1_range[0],vmax=var1_range[1],\
                            norm=mpl.colors.PowerNorm(gamma=1./2.)\
                                                   )
#        m.drawparallels(parallels,labels=[1,0,0,0], dashes=[2,900])
#        m.drawmeridians(meridians,labels=[0,0,1,0], dashes=[2,900])
        #---------------------------------------------------------------
        data_plot=pred_mort[pred_mort.index.year==year]
        ax=axs[1,year-year_range[0]]
#        ax.annotate(str(year), xy=(0.96, 0.95), xycoords='axes fraction',\
#                ha='right',va='top')
        m = Basemap(projection=proj,lat_0=45,lon_0=0,resolution='l',\
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[1],\
                ax=ax)
        m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')
        plot_data=m.scatter(lons, lats,s=marker_size,c=data_plot,cmap=cmap2\
                           ,marker='s',vmin=var2_range[0],vmax=var2_range[1],\
                           norm=mpl.colors.PowerNorm(gamma=1./2.)\
                                                  )
#        m.drawparallels(parallels,labels=[1,0,0,0], dashes=[1.5,900])
#        m.drawmeridians(meridians,labels=[0,0,0,1], dashes=[1.5,900])
        #-------------------------------------------------------------------
    cb0=fig.colorbar(plot_mort,ax=axs.ravel().tolist(), fraction=0.03,\
                     aspect=30,pad=0.02)
#    cb0.ax.tick_params(labelsize=fs) 
    tick_locator = ticker.MaxNLocator(nbins=ticks)
    cb0.locator = tick_locator
    cb0.update_ticks()
    cb0.set_ticks(np.linspace(var1_range[0],var1_range[1] ,ticks))
    cb0.ax.annotate('FAM (-)',xy=(0,1.03),xycoords='axes fraction',\
                    ha='left',va='bottom',size=fs)
    axs[0,0].set_ylabel(mort_label)
    axs[1,0].set_ylabel(data_label)
#    fig.suptitle(title)
#    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
#                                                 prefix = '', suffix = '.', fontweight = 'bold')
    scalebar = ScaleBar(100*1e3*1.05,box_alpha=0,sep=2,location='lower left') # 1 pixel = 0.2 meter
#    ax.add_artist(scalebar)
#    ax.annotate('California', xy=(0.1, 0), xycoords='axes fraction',\
#                    ha='left',va='bottom',size=fs*0.7)
    plt.savefig(Dir_ms_fig+'/Figure_3.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    return cb0
#==============================================================================
def plot_leaf_habit_thresh(data='RWC',data_label="RWC (-)",\
                    mort_label='FAM (-)',\
                    mort_key='mortality_%03d_grid_spatial_overlap', data_range=[0,1],mort_range=[0,0.7],\
                    grid_size=25,cmap='viridis',\
                    ticks=5,\
                    alpha=0.7):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    
    sns.set_style("whitegrid")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    
    mort=store[mort%(grid_size)]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    data=store[data]
    data=data[(data.index.year>=start_year) &\
              (data.index.year<=end_year)]    
    
    
    publishable.set_figsize(0.76*zoom, 1.7*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col')
    plt.subplots_adjust(hspace=0.25)
    ax=axs[0]
    species='evergreen'
    mort=store[mort_key+'_%s'%species]
    
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    
    mort=import_mort_leaf_habit(species=species)
    x=data.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y)
    plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_ylabel(mort_label)
    guess=(0.3,0.05,1e-1,1e-1)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    plot_data=ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=ax.get_ylim()
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(mort_range)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f'%(species, r_squared))
#    ax.set_title('%s trees'%species.title(),loc='left')
    ax.annotate('%s trees'%species.title(), xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
    
    ax=axs[1]
    species='deciduous'
    ax.annotate('%s trees'%species.title(), xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
    
    mort=import_mort_leaf_habit(species=species)
    x=data.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y)
    plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_xlabel(data_label)
    ax.set_ylabel(mort_label)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=ax.get_ylim()
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(mort_range)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f'%(species, r_squared))
    cbaxes = fig.add_axes([0.65, 0.7, 0.1, 0.1])
    cb=fig.colorbar(plot2_data,ax=axs[1],\
                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off',pad=4)
    cbaxes.annotate('Point\ndensity',xy=(0,1.2), xycoords='axes fraction',\
                ha='left')
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
    cb.outline.set_visible(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.annotate('Dry', xy=(0, -0.18), xycoords='axes fraction',color=cardinal)
    ax.annotate('Wet', xy=(0.9, -0.18), xycoords='axes fraction',color='dodgerblue')
    
def plot_boxplot(data_source1='mortality_%03d_grid',data_source2='RWC',data_source3='cwd',\
                            data_label1='FAM (-)',\
                            data_label2='RWC (-)',\
                            data_label3='CWD (mm)',\
                            grid_size=25,\
                            start_month=7,months_window=3):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    
#    data=store[data_source1]
#    data=data[(data.index.year>=start_year) &\
#      (data.index.year<=end_year)]  
#    data=data.rolling(30,min_periods=1).mean()
#    #           mask=(data.index.month>=start_month) & (data.index.month<start_month+months_window)
#    #           data[~mask]=np.nan
    publishable.set_figsize(0.6*zoom, 1.35*zoom, aspect_ratio =1)
    fig, axs = plt.subplots(3,1,sharex=True)
    plt.subplots_adjust(hspace=0.4)
#    ax=axs[0]
#    ax.grid(axis='x')
#    ax.set_ylabel(data_label1)
#    ax.plot(data.median(axis=1),'-',color='w',lw=1)
#    #           ax.set_ylim(1.2,1.3)
##    ax.fill_between(data.index,data.median(axis=1)-data.std(axis=1),\
##    data.median(axis=1)+data.std(axis=1),alpha=0.6,color='midnightblue')
#    ax.fill_between(data.index,data.quantile(0.95,axis=1),data.quantile(0.05,axis=1)\
#    ,alpha=0.6,color='midnightblue')  
#    ax.tick_params(axis='x', bottom='off')
#    for year in np.unique(data.index.year):
#        ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=0.3, color='tomato')
    ax=axs[0]
    data=store[data_source2]
    data=data[(data.index.year>=start_year) &\
             (data.index.year<=end_year)]
    ax.set_ylabel(data_label2)
    data.T.plot(kind='box',color='k',ax=ax)
    ax.grid(axis='x')
    ax.tick_params(axis='x',which='both', bottom='off')
    ax=axs[1]
    data=store[data_source3]
    data=data[(data.index.year>=start_year) &\
             (data.index.year<=end_year)]
    ax.tick_params(axis='x',which='both', bottom='off')
    ax.set_ylabel(data_label3)
    data.T.plot(kind='box',color='k',ax=ax)
    mean_shift=(data[data.index.year==2015].mean(axis=1)[0]-data[data.index.year==2014].mean(axis=1)[0])/\
    data[data.index.year==2015].mean(axis=1)[0]*100
    print('mean change in %s from 2014 to 2015 = %0.2f %%'%(data_label3,mean_shift))
    ax.grid(axis='x')    
    ax=axs[2]
    data=store[data_source1%grid_size]
    data=data[(data.index.year>=start_year) &\
             (data.index.year<=end_year)]
    ax.set_ylabel(data_label1)
    data.T.plot(kind='box',color='k',ax=ax,rot=45)
    ax.grid(axis='x')
    xtl=[item.get_text()[:4] for item in ax.get_xticklabels()]
    ax.set_xticklabels(xtl)
    ax.set_ylim(-0.01,0.15)
    mean_shift=(data[data.index.year==2015].mean(axis=1)[0]-data[data.index.year==2014].mean(axis=1)[0])/\
    data[data.index.year==2015].mean(axis=1)[0]*100
    print('mean change in %s from 2014 to 2015 = %0.2f %%'%(data_label1,mean_shift))
#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
            prefix = '', suffix = '.', fontweight = 'bold')        
    plt.show()  
        
def plot_pdf(data_source1='vod_pm',data_source2='RWC',data_source3='cwd',\
                            \
                            start_month=7,months_window=3):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_data.csv',index_col=0)
    input_sources=['FAM','live_basal_area','LAI_sum',\
     'aspect_mean', 'canopy_height',\
     'cwd','elevation_mean',\
     'forest_cover','ppt_sum','tmax_sum',\
     'tmean_sum','vpdmax_sum','EVP_sum',\
    'PEVAP_sum','vsm_sum','RWC']
    Df.drop('dominant_leaf_habit',axis=1,inplace=True)
#    Df=Df.loc[:,Df.columns.isin(input_sources)]
    ordered_sources=((Df.quantile(0.75)-Df.quantile(0.25))/Df.max()).\
                    sort_values(ascending=False).index
    publishable.set_figsize(1*zoom, len(input_sources)/14*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    fig, axs = plt.subplots(len(ordered_sources),1)
    i=0
    for source in ordered_sources:
        ax=axs[i]
        sns.violinplot(x=source,data=Df,ax=ax,palette=['orchid'],cut=0,linewidth=1,alpha=1)
        ax.set_ylabel(source,rotation=0,labelpad=0,ha='right',va='center')
        ax.tick_params(axis='x', bottom='off',labelbottom='off')
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_bounds(-0.525, 0.525)
        ax.spines['right'].set_bounds(-0.525, 0.525)
        i+=1
    axs[0].spines['top'].set_visible(True)
    axs[-1].spines['bottom'].set_visible(True)
    ax.tick_params(axis='x', bottom='on',labelbottom='on')
    ax.set_xticks([0,0.4])
    ax.set_xticklabels(['0.0','1.0'])
    ax.annotate('Normalized range',xy=(0.5,-1.3), xycoords='axes fraction',\
                ha='center',va='top')
    
def plot_regression(var1='FAM', var_range=[-0.02, 0.45],\
                         var1_label="Observed FAM (-)",\
                    var2='predicted_FAM',\
                     var2_label="Modeled FAM (-)",\
                    grid_size=25,cmap='PuOr', \
                    ticks=5,\
                    title='Regression of observed and predicted mortality',proj='cyl',\
                    dataset='test_data'):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s.csv'%dataset,index_col=0)   
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    test_rsq=pd.read_csv('D:/Krishna/Project/data/rf_test_rsq.csv',index_col=0).loc[1,'x']  
    fig, ax = plt.subplots()
    z=Df['RWC']
    plot=ax.scatter(Df[var1],Df[var2],marker='s',c=z,cmap=cmap,s=scatter_size,\
                    vmin=0,vmax=1)

    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
    ax.set_xticks(np.linspace(0,0.4,5))
    ax.set_yticks(np.linspace(0,0.4,5))
    plt.axis('equal')
    ax.plot(var_range,var_range,color='grey',lw=0.6)
    ax.set_xlim(var_range)
    ax.set_ylim(var_range)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot, cax=cax)
#    cax.set_yticklabels(['Low', 'High'])
#    cax.tick_params(axis='y', right='off')
    cax.annotate('RWC (-)',xy=(-0.6,1.07), xycoords='axes fraction',\
                ha='left')
#    cb.outline.set_visible(False)
#    slope, intercept, r_value, p_value, std_err = stats.linregress(Df[var1],Df[var2])
    ax.annotate('$R^2_{test}=%0.2f$'%test_rsq, xy=(0.85, 0.95), xycoords='axes fraction',\
                ha='right',va='top')
#    ax.annotate('1:1 line', xy=(0.9, 0.97), xycoords='axes fraction',\
#                ha='right',va='top',color='grey')
    rms = sqrt(mean_squared_error(Df[var1], Df[var2]))
    bias = np.mean(Df[var1]- Df[var2])
    print('RMSE = %0.4f | Bias = %0.4f'%(rms, bias))
    plt.savefig(Dir_ms_fig+'/Figure_4.tiff', dpi = dpi, bbox_inches="tight")

def plot_importance(filename =None, width = 0.8, height = 2, save = None,\
                    savename = None):
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/%s.csv'%filename,index_col=0)   
    publishable = change_font_size(13) 
    publishable.set_figsize(width*zoom, height*zoom, aspect_ratio =1)
    sns.set_style('whitegrid')
    Df=Df.sort_values('mean')
    Df = Df/Df['mean'].sum()
    Df=append_color_importance(Df)
    Df.index = Df.index.str.lower()
    Df.index = [var_dict[x] for x in Df.index]
    fig, ax = plt.subplots()
    Df['mean'].plot.barh(width=0.8,color=Df.color,xerr=Df['sd'],\
           error_kw=dict(ecolor='k', lw=1, capsize=2, capthick=1),ax=ax)
#    ax.tick_params(axis='y', left='off',pad=-1)
    ax.set_xlabel('Relative Importance (-)')
    ax.yaxis.grid(False)
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    legend_elements = [Patch(facecolor=green, edgecolor=None,label='Vegetation'),\
                       Patch(facecolor=brown, edgecolor=None,label='Topography'),\
                        Patch(facecolor=blue, edgecolor=None,label='Climate')]
    ax.legend(handles=legend_elements,frameon=True, title='Variable Type')
#    plt.tight_layout()
    
    if r'$\rm RWC_{1\ year\ lag}$' in Df.index:
        for ticklabel, index in zip(ax.get_yticklabels(), Df.index):
            if index in [r'$\rm RWC_{1\ year\ lag}$',r'$\rm RWC_{2\ years\ lag}$']:
                ticklabel.set_color(green)
    if save:
        plt.savefig(Dir_ms_fig+'/%s.tiff'%savename, dpi = dpi, bbox_inches="tight")
    print(Df)
    return Df
    
    
def plot_FAM_TPA_corr(var1='mortality_%03d_grid', var1_range=[-0.02, 0.42],\
                         var1_label="Observed FAM (-)",\
                    var2='TPA_%03d_grid', var2_range=[-0.6, 13],\
                     var2_label="Dead trees (acre"+"$^{-1}$"+")",\
                    grid_size=25,cmap=sns.dark_palette("seagreen",as_cmap=True,reverse=True), \
                    ticks=5,\
                    title='Regression of observed and predicted mortality',proj='cyl',\
                    alpha=1):
    os.chdir(Dir_CA)  
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    store=pd.HDFStore(filename)
    var1=store[var1%grid_size]
    var2=store[var2%grid_size]
    var1=var1[(var1.index.year>=start_year) &\
      (var1.index.year<=end_year)]  
    var2=var2[(var2.index.year>=start_year) &\
      (var2.index.year<=end_year)]
    fig, ax = plt.subplots(1,1)
    x,y,z=clean_xy(var1.values.flatten(),var2.values.flatten())
    xd=np.linspace(min(x),max(x),100)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    plot=ax.scatter(x,y,marker='s',c=z,cmap=cmap,edgecolor='',alpha=alpha, s=20)
    ax.plot(xd,intercept + xd*slope,'r--',lw=1)
    ax.fill_between(xd,intercept + xd*(slope-std_err),intercept + xd*(slope+std_err),\
                    color='red',alpha=alpha)
    ax.set_xlim(var1_range)
    ax.set_ylim(var2_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot, cax=cax,ticks=[min(z),max(z)])
    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
                ha='left')
    cax.set_yticklabels(['Low', 'High'])
    cax.tick_params(axis='y', right='off',pad=0)


    return r_value**2
    
def plot_importance_rank():
    Df=pd.read_csv('D:/Krishna/Project/data/rf_sensitivity_rank.csv',index_col=0)   
    publishable.set_figsize(1, 0.5, aspect_ratio =1)
    sns.set_style('ticks')
    Df=Df.sort_values('Freq')
    
    fig, ax = plt.subplots(1,1)
    plot=Df.plot.barh(width=0.8,color='grey',ax=ax,legend=False)
    ax.tick_params(axis='y', left='off',pad=-1)
    ax.set_xlabel('Normalized frequency of first rank occurences')
    ax.set_xlim(0,1)
    
def plot_PET_AET(data_source1='vod_pm',data_source2='RWC',data_source3='cwd',\
                            data_label1='VOD',\
                            data_label2='RWC',\
                            data_label3='CWD',\
                            \
                            start_month=7,months_window=3):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    data=store[data_source1]
    data=data[(data.index.year>=start_year) &\
      (data.index.year<=end_year)]  
    data=data.rolling(30,min_periods=1).mean()
    #           mask=(data.index.month>=start_month) & (data.index.month<start_month+months_window)
    #           data[~mask]=np.nan
    publishable.set_figsize(1, 1, aspect_ratio =1)
    sns.set_style('ticks')
    fig, axs = plt.subplots(3,1,sharex=True)
    plt.subplots_adjust(hspace=0.23)
    ax=axs[0]
    ax.grid(axis='x')
    ax.set_ylabel(data_label1)
    ax.plot(data.median(axis=1),'-',color='w',lw=1)
    #           ax.set_ylim(1.2,1.3)
#    ax.fill_between(data.index,data.median(axis=1)-data.std(axis=1),\
#    data.median(axis=1)+data.std(axis=1),alpha=0.6,color='midnightblue')
    ax.fill_between(data.index,data.quantile(0.95,axis=1),data.quantile(0.05,axis=1)\
    ,alpha=0.6,color='midnightblue')    
    data=store[data_source2]
    data=data[(data.index.year>=start_year) &\
             (data.index.year<=end_year)]
    ax.tick_params(axis='x', bottom='off')
    for year in data.index.year:
        ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=0.3, color='red')
    ax=axs[1]
    ax.grid(axis='x')
    ax.set_ylabel(data_label2)
    ax.errorbar(data.index+timedelta(days=start_month*30+16),data.mean(axis=1),yerr=data.std(axis=1),\
    color='lightsalmon',fmt='s',ms=6,capsize=4,capthick=1)
    ax.tick_params(axis='x', bottom='off')
    data=store[data_source3]
    data=data[(data.index.year>=start_year) &\
             (data.index.year<=end_year)]
    ax=axs[2]
    ax.grid(axis='x')
    ax.set_ylabel(data_label3)
    y1,y2=store['PEVAP'].mean(1),store['EVP'].mean(1)
    y1=y1[(y1.index.year>=start_year) &\
      (y1.index.year<=end_year)]
    y2=y2[(y2.index.year>=start_year) &\
      (y2.index.year<=end_year)]
    t1,t2=(y1-y2)/2,(y2-y1)/2
    ax.plot(y1,color='crimson',lw=1)
    ax.plot(y2,color='navy',lw=1)
    ax.fill_between(y1.index,y1,y2,alpha=0.3,color='darkgreen')
#    ax.errorbar(data.index+timedelta(days=start_month*30+16),data.mean(axis=1),yerr=data.std(axis=1),\
#    color='darkgreen',fmt='o',ms=6,capsize=4,capthick=1)
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
            prefix = '', suffix = '.', fontweight = 'bold')
    ax.annotate('PET', xy=(0.98, 0.95), xycoords='axes fraction',\
                ha='right',va='top',color='crimson',fontweight='bold')
    ax.annotate('AET', xy=(0.39, 0.12), xycoords='axes fraction',\
                ha='right',va='top',color='navy',fontweight='bold')
    ax.annotate('CWD', xy=(0.67, 0.38), xycoords='axes fraction',\
                ha='right',va='top',color='darkgreen',fontweight='bold')
    
def idxquantile(s, q=0.50, *args, **kwargs):
    qv = s.quantile(q, *args, **kwargs)
    return (s.sort_values()[::-1] <= qv).idxmax()

def plot_RWC_definition(data_source1='vod_pm',data_source2='RWC',\
                            data_label1='VOD (-)',\
                            data_label2='RWC (-)',\
                            grid_cell = 333,\
                            start_month=7,months_window=3\
                            ,alpha1=0.2,color='#BD2031',alpha2=0.7):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    df=store[data_source1]
    df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
    df=df.rolling(30,min_periods=1).mean()
    publishable.set_figsize(1.8*zoom, 0.5*zoom, aspect_ratio =1)
#    for col in df.columns:
    sns.set_style('ticks')
    fig, ax = plt.subplots(1,1,sharex=True)
#    ax.grid(axis='x')
    ax.set_ylabel(data_label1)
    
    data = df.loc[:,grid_cell]
#        data=df.loc[:,333 #.loc 104 columns is pretty
    ax.plot(data,'-',color='k')  
    for year in np.unique(data.index.year):
        ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=alpha1, facecolor=color)
    mask=(data.index.month>=start_month) & (data.index.month<start_month+months_window)
    data[~mask]=np.nan
    u,l=data.quantile(0.95),data.quantile(0.05)
    ax.scatter(idxquantile(data,q=0.05),data.quantile(0.05),s=45,lw=1,c='sienna',\
               edgecolor='w',marker='o', zorder = 100)
    ax.scatter(idxquantile(data,q=0.95),data.quantile(0.95),s=45,lw=1,c='indigo',\
               edgecolor='w',marker='o', zorder = 100)
    ax2 = ax.twinx()
    for year in np.unique(data.index.year):
        subset=data.loc[data.index.year==year]
        ### vertical lines
#        ax.plot([idxquantile(subset),idxquantile(subset)],[l,subset.quantile(0.5)],\
#                 ls='-',color=color,lw=3,alpha=1,solid_capstyle='butt')
        ###
        ax.plot(pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]),[subset.quantile(0.5),subset.quantile(0.5)],\
                 ls='-',color=color,lw=2,alpha=0.8,solid_capstyle='butt')
    ax.set_xlim([data.index.min(),data.index.max()])
    ax2.set_ylabel('$\quad\quad\quad$'+data_label2,color=color)
    ax2.tick_params(colors=color)
    ax.set_ylim(1.1,1.8)
    y1range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y2min = (l - ax.get_ylim()[0])/y1range
    y2max = (u - ax.get_ylim()[0])/y1range
    ax2.set_yticks([y2min,y2max])
    ax2.set_yticklabels([0.0,1.0])
    ax.set_yticks(np.arange(1.2,1.9,0.2))
#    ax2.grid(axis='y',alpha=0.2,color=color)
    ax2.axhline(y2max, color = 'indigo', lw = 1, alpha = 0.4)
    ax2.axhline(y2min, color = 'sienna', lw = 1, alpha = 0.4)
#        ax.set_title('Gridcell %s'%col)
#    ax.annotate(r'95$^{th}$ percentile summer VOD', xy=(0.5, 0.92), va = 'top',ha = 'right',\
#                color='indigo',xycoords='axes fraction', fontsize = 9)
#    ax.annotate(r'5$^{th}$ percentile summer VOD', xy=(0.99, 0.42), va = 'center',ha = 'right',\
#                color='sienna',xycoords='axes fraction', fontsize = 9)
    plt.savefig(Dir_ms_fig+'/Figure_1.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    

def plot_RWC_definition_v2(data_source1='vod_pm',data_source2='RWC',\
                            data_label1='VOD (-)',\
                            data_label2='RWC (-)',\
                            grid_cell = 333, save = False,\
                            start_month=7,months_window=3\
                            ,alpha1=0.2,color='#BD2031',alpha2=0.7):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    df=store[data_source1]
    df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
#    df=df.rolling(30,min_periods=1).mean()
    publishable.set_figsize(1.8*zoom, 0.5*zoom, aspect_ratio =1)
#    for col in df.columns:
    sns.set_style('ticks')
    fig, ax = plt.subplots(1,1,sharex=True)
#    ax.grid(axis='x')
    ax.set_ylabel(data_label1)
    
    data2 = df.loc[:,grid_cell]
#        data=df.loc[:,333 #.loc 104 columns is pretty
#    ax.plot(data,'-',color='k')  
    data = pd.read_pickle("D:/Krishna/Project/data/Mort_Data/Misc_data/gridID333_AMSRE_VOD")
    data = data.loc[data.index.year<2009]
    data = data.append(data2)
    data = data.rolling(30,min_periods=1).mean()
    ax.plot(data,'-',color='k', alpha = 0.7)  
    for year in np.unique(data.index.year):
        ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=alpha1, facecolor=color)
    mask=(data.index.month>=start_month) & (data.index.month<(start_month+months_window))
    data[~mask]=np.nan
#    u,l=data.quantile(0.95),data.quantile(0.05)
    min_year, max_year = idxquantile(data,q=0.05), idxquantile(data,q=0.95)
#    ax.scatter(year,l,s=45,lw=1,c='sienna',\
#               edgecolor='w',marker='o', zorder = 100)
#    ax.scatter(idxquantile(data,q=0.95),u,s=45,lw=1,c='indigo',\
#               edgecolor='w',marker='o', zorder = 100)
    ax2 = ax.twinx()
    for year in np.unique(data.index.year):
        subset=data.loc[data.index.year==year]
        ### vertical lines
#        ax.plot([idxquantile(subset),idxquantile(subset)],[l,subset.quantile(0.5)],\
#                 ls='-',color=color,lw=3,alpha=1,solid_capstyle='butt')
        ###
        median = subset.quantile(0.5)
        ax.plot(pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]),[median,median],\
                 ls='-',color=color,lw=2,alpha=1,solid_capstyle= 'butt')
        if year==min_year.year:
            l=median
#            ax.scatter(min_year,l,s=25,lw=1,c='sienna',\
#               edgecolor='w',marker='o', zorder = 100)
        if year==max_year.year:
            u = median
#            ax.scatter(max_year,u,s=25,lw=1,c='indigo',\
#               edgecolor='w',marker='o', zorder = 100)
    
    ax.set_xlim([data.index.min(),data.index.max()])
    ax2.set_ylabel('$\quad\quad\quad$'+data_label2,color=color)
    ax2.tick_params(colors=color)
    ax.set_ylim(1.1,1.8)
    y1range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y2min = (l - ax.get_ylim()[0])/y1range
    y2max = (u - ax.get_ylim()[0])/y1range
    ax2.set_yticks([y2min,y2max])
    ax2.set_yticklabels([0.0,1.0])
    ax.set_yticks(np.arange(1.2,1.9,0.2))
    ax.set_xticks(pd.date_range('2003-01-01', periods=7, freq='2AS')
)
    
#    ax2.grid(axis='y',alpha=0.2,color=color)
    ax2.axhline(y2max, color = 'indigo', lw = 1, alpha = 0.4)
    ax2.axhline(y2min, color = 'sienna', lw = 1, alpha = 0.4)
#        ax.set_title('Gridcell %s'%col)
#    ax.annotate(r'95$^{th}$ percentile summer VOD', xy=(0.5, 0.92), va = 'top',ha = 'right',\
#                color='indigo',xycoords='axes fraction', fontsize = 9)
#    ax.annotate(r'5$^{th}$ percentile summer VOD', xy=(0.99, 0.42), va = 'center',ha = 'right',\
#                color='sienna',xycoords='axes fraction', fontsize = 9)
    if save:
        plt.savefig(Dir_ms_fig+'/Figure_1_extend.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    
def plot_RWC_definition_with_vpd(data_source1='vod_pm',data_source2='RWC',\
                            data_label1='VOD (-)',\
                            data_label2='RWC (-)',\
                            grid_cell = 333,\
                            start_month=7,months_window=3\
                            ,alpha1=0.2,color='#BD2031',alpha2=0.7):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    df=store[data_source1]
    df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
#    df=df.rolling(30,min_periods=1).mean()
    publishable.set_figsize(1.8*zoom, 1.5*zoom, aspect_ratio =1)
    
#    for col in df.columns:
    sns.set_style('ticks')
    fig, axs = plt.subplots(3,1,sharex=True)
#    ax.grid(axis='x')
    ax = axs[0]
    ax.set_ylabel(data_label1)
    
    data2 = df.loc[:,grid_cell]
#        data=df.loc[:,333 #.loc 104 columns is pretty
#    ax.plot(data,'-',color='k')  
    data = pd.read_pickle("D:/Krishna/Project/data/Mort_Data/Misc_data/gridID333_AMSRE_VOD")
    data = data.loc[data.index.year<2009]
    data = data.append(data2)
    data = data.rolling(30,min_periods=1).mean()
    ax.plot(data,'-',color='k')  
    for year in np.unique(data.index.year):
        for axis in axs:
            axis.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=alpha1, facecolor=color)
    ax = axs[0]
    mask=(data.index.month>=start_month) & (data.index.month<(start_month+months_window))
    data[~mask]=np.nan
#    u,l=data.quantile(0.95),data.quantile(0.05)
    min_year, max_year = idxquantile(data,q=0.05), idxquantile(data,q=0.95)
#    ax.scatter(year,l,s=45,lw=1,c='sienna',\
#               edgecolor='w',marker='o', zorder = 100)
#    ax.scatter(idxquantile(data,q=0.95),u,s=45,lw=1,c='indigo',\
#               edgecolor='w',marker='o', zorder = 100)
    ax2 = ax.twinx()
    #### more variables on request of reviewer

    var = '/LAI_025_grid'
    col = 'g'
    ax3 = axs[1]
    d = store[var]
    d.loc[:,grid_cell].rolling(30,min_periods=1).mean().plot(ax = ax3, alpha = 1, color = col, zorder = -100)
    ax3.set_ylabel('LAI (-)')
    
    var = 'vpdmax'
    col = 'steelblue'
    ax4 = axs[2]
    d = store[var]
    d.index = pd.to_datetime(d.index)
    ax4.plot(d.index, d.loc[:,grid_cell].rolling(30,min_periods=1).mean(), alpha = 1, zorder = -100)
    ax4.set_ylabel(r'$\rm VPD_{max}$ (hPa)')
    ax4.set_xticks(pd.date_range('2003-01-01', periods=7, freq='2AS'))
    ax4.set_xlim([data.index.min(),data.index.max()])
    
    for year in np.unique(data.index.year):
        subset=data.loc[data.index.year==year]
        ### vertical lines
#        ax.plot([idxquantile(subset),idxquantile(subset)],[l,subset.quantile(0.5)],\
#                 ls='-',color=color,lw=3,alpha=1,solid_capstyle='butt')
        ###
        median = subset.quantile(0.5)
        ax.plot(pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]),[median,median],\
                 ls='-',color=color,lw=2,alpha=1,solid_capstyle= 'butt', zorder = np.inf)
        if year==min_year.year:
            l=median
#            ax.scatter(min_year,l,s=25,lw=1,c='sienna',\
#               edgecolor='w',marker='o', zorder = 100)
        if year==max_year.year:
            u = median
#            ax.scatter(max_year,u,s=25,lw=1,c='indigo',\
#               edgecolor='w',marker='o', zorder = 100)

    ax.set_xlim([data.index.min(),data.index.max()])
    ax2.set_ylabel('$\quad\quad\quad$'+data_label2,color=color)
    ax2.tick_params(colors=color)
    ax.set_ylim(1.1,1.8)
    y1range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y2min = (l - ax.get_ylim()[0])/y1range
    y2max = (u - ax.get_ylim()[0])/y1range
    ax2.set_yticks([y2min,y2max])
    ax2.set_yticklabels([0.0,1.0])  
    ax.set_yticks(np.arange(1.2,1.9,0.2))
    ax.set_xticks(pd.date_range('2003-01-01', periods=7, freq='2AS'))
    
#    ax2.grid(axis='y',alpha=0.2,color=color)
    ax2.axhline(y2max, color = 'indigo', lw = 1, alpha = 0.4)
    ax2.axhline(y2min, color = 'sienna', lw = 1, alpha = 0.4)
#        ax.set_title('Gridcell %s'%col)
#    ax.annotate(r'95$^{th}$ percentile summer VOD', xy=(0.5, 0.92), va = 'top',ha = 'right',\
#                color='indigo',xycoords='axes fraction', fontsize = 9)
#    ax.annotate(r'5$^{th}$ percentile summer VOD', xy=(0.99, 0.42), va = 'center',ha = 'right',\
#                color='sienna',xycoords='axes fraction', fontsize = 9)

    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 
    ax.patch.set_visible(False) # hide the 'canvas' 
    
    for (caption, axis) in zip(['a.','b.','c.'],axs):
        axis.annotate(caption, xy=(-0.22, 1), xycoords='axes fraction', weight = 'bold')
        



    plt.savefig(Dir_ms_fig+'/Figure_1_vpd.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    
def plot_VOD_by_LAI(data_source1='vod_pm',data_source2='RWC',\
                            data_label1='VOD (-)',\
                            data_label2='RWC (-)',\
                            grid_cell = 333,\
                            start_month=7,months_window=3\
                            ,alpha1=0.2,color='#BD2031',alpha2=0.7):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    df=store[data_source1]
    df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
#    df=df.rolling(30,min_periods=1).mean()
    publishable.set_figsize(1.8*zoom, 1.5*zoom, aspect_ratio =1)
#    for col in df.columns:
    sns.set_style('ticks')
    fig, axs = plt.subplots(3,1,sharex=True)
#    ax.grid(axis='x')
    ax = axs[0]
    ax.set_ylabel(data_label1)
    
    data2 = df.loc[:,grid_cell]
#        data=df.loc[:,333 #.loc 104 columns is pretty
#    ax.plot(data,'-',color='k')  
    data = pd.read_pickle("D:/Krishna/Project/data/Mort_Data/Misc_data/gridID333_AMSRE_VOD")
    data = data.loc[data.index.year<2009]
    data = data.append(data2)
    data = data.rolling(30,min_periods=1).mean()
    ax.plot(data,'-',color='k')  
    

    var = '/LAI_025_grid'
    col = 'g'
    ax3 = axs[1]
    d = store[var].loc[:,grid_cell].rolling(30,min_periods=1).mean()
    d.plot(ax = ax3, alpha = 1, color = col, zorder = -100)
    ax3.set_ylabel('LAI (-)', color=col)
    
    var = 'VOD/LAI'
    col = 'steelblue'
    dd = data/d
#    d.index = pd.to_datetime(d.index)
    ax4 = axs[2]
    dd.rolling(30,min_periods=1).mean().plot(ax = ax4, alpha = 1, color = col, zorder = -100)
#    ax4.plot(dd.index, dd, alpha = 1, zorder = -100)
    ax4.set_ylabel(r'VOD/LAI', color=col)
    ax4.set_xticks(pd.date_range('2003-01-01', periods=7, freq='2AS'))
    ax4.set_xlim([data.index.min(),data.index.max()])
    

    ax.set_yticks(np.arange(1.2,1.9,0.2))
    ax.set_xticks(pd.date_range('2003-01-01', periods=7, freq='2AS'))
    
    ax.set_zorder(axs[1].get_zorder()+1) # put ax in front of ax2 
    ax.patch.set_visible(False) # hide the 'canvas' 
    
                        
                        
    for year in np.unique(data.index.year):
        for axis in axs:
            axis.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=alpha1, facecolor=color)
   
    plt.show()
    
def plot_vod_matched(grid_cell = 333):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    df=store['vod_pm']
    df=df[(df.index.year>=start_year) &\
      (df.index.year<=end_year)]  
    df=df.rolling(30,min_periods=1).mean()
    
    df2=store['vod_pm_matched']
    df2=df2[(df2.index.year>=2012) &\
      (df2.index.year<=end_year)]  
    df2=df2.rolling(30,min_periods=1).mean()    
    publishable.set_figsize(1.8*zoom, 0.5*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    fig, ax = plt.subplots(1,1,sharex=True)
#    ax.grid(axis='x')
    ax.set_ylabel('VOD (-)')
    data = df.loc[:,grid_cell]; data2 = df2.loc[:,grid_cell]

    ax.plot(data,'--',color='k', label = "original")  
    ax.plot(data2,'-',color='k', label = "cdf-matched")
    

    ax.set_ylim(1.1,1.8)
    
    ax.set_yticks(np.arange(1.1,1.9,0.1))
    plt.legend(loc = "lower right", bbox_to_anchor = [1.03,-0.14])
    plt.savefig(Dir_ms_fig+'/Figure_S2.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    
def plot_rwc_cwd(data1='RWC',data2='cwd',data1_label="RWC (-)",\
                 data2_label='CWD (mm)',
                    mort_label=(1-ppt)*'FAM (-)'+ppt*'Fractional Area of Mortality (-)',\
                    mort='mortality_%03d_grid',\
                    grid_size=25,cmap='viridis', \
                    \
                    alpha=1):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    mort=store[mort%(grid_size)]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    mort_range=[0,mort.values.max()*1.1]
    data1=store[data1]
    data1=data1[(data1.index.year>=start_year) &\
              (data1.index.year<=end_year)] 
    data2=store[data2]
    data2=data2[(data2.index.year>=start_year) &\
              (data2.index.year<=end_year)]     
    
    publishable.set_figsize(2*zoom, 1*zoom, aspect_ratio = 1)
    #_-------------------------------------------------------------------------------
    fig, axs = plt.subplots(nrows=1,ncols=2,sharey='row')
    plt.subplots_adjust(wspace=0.10)
    ax=axs[0]
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax.set_yscale('log')
#    ax.set_ylim([1e-4,0.4])
    x=data1.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y)
    plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_ylabel(mort_label)
    guess=(0.08,0.02,1e-4,1e-2)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    plot_data=ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=ax.get_ylim()
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(mort_range)  
    ax.set_xlabel(data1_label)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f; Threshold = %0.2f'%(data1.index.name, r_squared, popt[0]))
    ax.annotate('Dry', xy=(0, -0.3), xycoords='axes fraction',color='sienna')
    ax.annotate('Wet', xy=(0.9, -0.3), xycoords='axes fraction',color='dodgerblue')
    adjust_spines(ax,['left','bottom'])
    #_-------------------------------------------------------------------------------
    ax=axs[1]
    ax.annotate('Dry', xy=(0.9, -0.3), xycoords='axes fraction',color='sienna')
    ax.annotate('Wet', xy=(0,-0.3), xycoords='axes fraction',color='dodgerblue')
    x=data2.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y)
    plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size,\
                          vmin=min(z),vmax=max(z))
#    print('min = %0.2f, max = %0.2f'%(min(z),max(z)))
    ax.set_xlabel(data2_label)
#    ax.set_ylabel(mort_label)
    guess=(600,0.05,1e-4,1e-2)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=ax.get_ylim()
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
#    ax.set_ylim(mort_range)
    ax.tick_params(axis='y', left='off')
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f; Threshold = %0.2f'%(data2.index.name, r_squared, popt[0]))
#    cbaxes = fig.add_axes([0.38, 0.5, 0.04, 0.18])
#    cb=fig.colorbar(plot2_data,ax=axs[1],\
#                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot2_data, cax=cax,ticks=[min(z),max(z)])
    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
                ha='left')
    cax.set_yticklabels(['Low', 'High'])
    cax.tick_params(axis='y', right='off',pad=0)
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
#    cb.outline.set_visible(False)
    adjust_spines(ax,['bottom'])
    
def plot_grid(data='data',color='#BD2031'):
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s_base_model.csv'%data,index_col=0)  
    
    input_sources=Df.columns.tolist()
    input_sources.remove('FAM')
    input_sources.sort( key=lambda x: var_dict[x.lower()])
#    input_sources.remove('location') ## remove location to make predictor 25
    sns.set_style('ticks') 
    publishable = change_font_size(9) 
    publishable.set_figsize(3.3*zoom, 4*zoom, aspect_ratio =1)
    fig, axs = plt.subplots(6,6,sharey='row')
#    fig.delaxes(axs[5,4])
    plt.subplots_adjust(hspace=0.3)
#    axs=flatten_diagonally(axs, diagonals = None)
    
    niter=0
    for xcol, ax in zip(input_sources, axs.ravel()):
        Df.plot(kind='scatter', x=xcol, y='FAM', ax=ax, \
                s=10,alpha=0.5, color=variable_type_color(xcol),edgecolor='None')
        
        ax.set_xlabel('',labelpad=-5)
        ax.tick_params(axis='both', which='major', pad=1)
        ax.set_aspect('auto',adjustable='box-forced')
        ax.set_ylim([-0.02,0.45])
        ax.set_ylabel('')
#        if niter ==6:
#            ax.set_ylabel('FAM')
        niter+=1
        if xcol=='twi_mean':
            ax.set_xlim(250,650)
        elif xcol=='twi_std':
            ax.set_xlim(120, 350)
        #####
        r_squared, rmse = breakpoint(Df, xcol=xcol,ax=ax)        
        ax.annotate(var_dict[xcol.lower()]+' '+unit_of(xcol), xycoords='axes fraction',\
                ha='center',va='bottom', xy=(0.5, 1.01), color = 'k') 
        ax.annotate('$R^2 = %0.2f$'%r_squared+'\n$RMSE = %0.2f$'%rmse,\
                    xycoords='axes fraction', xy=(0.5, 0.98),\
                ha='center',va='top', color = 'k', size = 8) 
    for ax in axs.ravel()[-4:]:
        ax.axis('off')
    fig.text(0.085,0.5,'FAM (-)',rotation=90)
    
    green = '#1b9e77'
    brown = '#d95f02'
    blue = '#7570b3'
    legend_elements = [Patch(facecolor=green, edgecolor=None,label='Vegetation'),\
                       Patch(facecolor=brown, edgecolor=None,label='Topography'),\
                        Patch(facecolor=blue, edgecolor=None,label='Climate')]
    plt.legend(handles=legend_elements,frameon=True, title='Variable Type')
    
    
    plt.savefig(Dir_ms_fig+'/Figure_S5.tiff', dpi = dpi, bbox_inches="tight")
    
def shift_labels(ax,SHIFT=0.5):
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType( lambda self, x: mpl.text.Text.set_x(self, x-self.customShiftValue ), 
                                    label, mpl.text.Text )
    for label in ax.yaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_y = types.MethodType( lambda self, y: mpl.text.Text.set_y(self, y-self.customShiftValue ), 
                                    label, mpl.text.Text )
        
def plot_heatmap(data='data',cmap='PRGn_r'):
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s_base_model.csv'%data,index_col=0) 
    Df.drop('FAM',axis = 1, inplace = True)
    corr=Df.corr()  
    trimmed_model_sources=['mortality_025_grid',"RWC","CWD","elevation_std",\
                       "elevation_mean","ppt_sum","location",\
                       "aspect_mean","aspect_std","vsm_sum","canopy_height",\
                       'EVP_win', 'twi_mean']
    keep=trimmed_model_sources
    #=========================================================================

    ##=======================================================================
    sns.set_style('white')
    publishable = change_font_size(11) 
    publishable.set_figsize(2*zoom, 2*zoom, aspect_ratio =1)
    inds=sns.clustermap(corr,figsize=(1e-10,1e-10)\
                        ).dendrogram_row.reordered_ind     
    corr=corr[inds]
    corr=corr.reindex(corr.columns)
    mask = np.zeros_like(corr)

    hide_inds=np.array([corr.columns.get_loc(c) for c in corr.columns if c not in keep])
    mask[:,hide_inds]=True
    mask[hide_inds,:]=True   
    mask[np.tril_indices_from(mask)] = False
    corr.index = [var_dict[x] for x in corr.index.str.lower()]
    corr.columns = [var_dict[x] for x in corr.columns.str.lower()]
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((0,0),corr.shape[1],corr.shape[0],color='darkgrey',zorder=-1,alpha=0.5))
    cbar_ax = fig.add_axes([.9, .121, .03, .76])
    sns.heatmap(corr,vmin=-1,vmax=1,mask=mask,ax=ax, annot=True, fmt='.1f', \
                annot_kws={"size": 6}, cbar_ax=cbar_ax,\
                square=True,cbar_kws={'ticks':np.linspace(-1,1,11)},cmap=cmap,\
                                      linewidths=.2)
    plt.savefig(Dir_ms_fig+'/Figure_S6.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
#    print(corr.shape)
    print(corr.loc['RWC'])
#    ### investigate============================================================
    cutoff = 0.1870
    calc_keep = (corr**2).quantile(0.75).sort_values()[:11]
    calc_keep = calc_keep.index    
    diff = list(set(keep).symmetric_difference(set(calc_keep)))
#    print(calc_keep)
#    print(diff)
    ####### output for feeding into RF weights in R studio
#    order_required = ["live_basal_area", "LAI_sum",         "LAI_win",         "RWC",             "aspect_mean",     "aspect_std",     
#    "canopy_height",   "CWD" ,            "elevation_mean",  "elevation_std",   "forest_cover" ,   "ppt_sum" ,        "ppt_win",        
#    "tmax_sum" ,       "tmax_win"  ,      "tmean_sum"   ,    "tmean_win",       "vpdmax_sum" ,     "vpdmax_win",      "EVP_sum" ,       
#    "PEVAP_sum",       "EVP_win"  ,       "PEVAP_win" ,      "vsm_sum",         "vsm_win" ,        "location"  ,      "silt_fraction" , 
#    "sand_fraction"  , "twi_mean"  ,      "twi_std"] 
#    order_required = [x.lower() for x in order_required]
#    var_weights = np.array(1-corr.abs().mean().loc[order_required].values )
#    return var_weights
    
    
def plot_rsq_subset():
    sns.set_style('whitegrid')
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    labels=['None'	,'tmean_sum',	'PEVAP_win',	\
            'EVP_sum',	'tmax_sum',	'tmax_win',	'vpdmax_win',	'vpdmax_sum',	\
            'PEVAP_sum',	'ppt win',	'tmean_win',	'EVP_win']
    values=np.array([0.41,	0.41,	0.41,	0.41	,0.4,	0.4,	0.4,	0.39,	0.39,	0.38,	0.38,	0.35])
    x=np.arange(len(values))
    fig, ax = plt.subplots()
    ax.plot(x,values,'--ko')
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels,rotation=45)
    ax.set_ylabel("Test $R^2$")
    ax.set_xlabel('Feature removed')
    ax.set_title('Backward selection')
    
def plot_LPDR2(cmap='plasma',scatter_size=10,var1_label='VOD, LPDRv1',var2_label='VOD, LPDRv2',\
               var_range=[0.5,2.2]):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    Df1=store['vod_pm']
    Df1=Df1.loc[Df1.index.year == 2015]
    Df1=Df1.mask(Df1==np.nan)
    Df2=store['vod_pm_LPDR2']
    publishable.set_figsize(0.5*zoom, 0.5*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    x=Df1.values.flatten()
    y=Df2.values.flatten()
    inds=~np.isnan(x)
    x,y=x[inds],y[inds]
    inds=~np.isnan(y)
    x,y=x[inds],y[inds]
#    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)    
    fig, ax = plt.subplots()
    sns.kdeplot(x,y, cmap='PuRd', n_levels=60, shade=True,ax=ax,clip=var_range)

##    x,y,z=clean_xy(Df1,Df2)
#    ax.scatter(Df1,Df2,marker='o',c='k',cmap=cmap,s=scatter_size,alpha=0.5)
    ax.set_xlim(var_range)
    ax.set_ylim(var_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
    ax.plot(var_range,var_range,color='grey',lw=0.6)
#    cbaxes = fig.add_axes([0.2, 0.50, 0.1, 0.24])
#    cb=fig.colorbar(plot,ax=ax,\
#                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
#    cb.ax.set_yticklabels(['Low', 'High'])
#    cb.ax.tick_params(axis='y', right='off')
#    cbaxes.annotate('RWC',xy=(0,1.2), xycoords='axes fraction',\
#                ha='left')
#    cb.outline.set_visible(False)
#    ax.annotate('$R^2_{test}=$0.45', xy=(0.98, 0.57), xycoords='axes fraction',\
#                ha='right',va='top')
#    ax.annotate('1:1 line', xy=(0.9, 0.97), xycoords='axes fraction',\
#                ha='right',va='top',color='grey')
    
def plot_leaf_habit_mort(xlabel='evergreen', ylabel='deciduous',cmap='inferno',alpha=1,scatter_size=10,\
                         var1_label = 'Evergreen FAM', var2_label='Deciduous FAM',var_range=[-0.03,0.7],\
                        FAM_thresh=0.0):
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    
#    x=import_mort_leaf_habit(species=xlabel).values.flatten()
#    y=import_mort_leaf_habit(species=ylabel).values.flatten()

    Df1=store['mortality_%s_025_grid_spatial_overlap'%xlabel]
    Df2=store['mortality_%s_025_grid_spatial_overlap'%ylabel]
    
    Df1=Df1[(Df1.index.year>=start_year) &\
              (Df1.index.year<=end_year)]
    Df2=Df2[(Df2.index.year>=start_year) &\
              (Df2.index.year<=end_year)]
    
    bound_thresh = 0.1
    Df1=select_bounding_box_grids(Df1,thresh=bound_thresh)
    Df2=select_bounding_box_grids(Df2,thresh=bound_thresh)
    
    FAM_thresh=0.1
    x,y,z=clean_xy(Df1.values.flatten(),Df2.values.flatten())
    inds=np.where(y>=FAM_thresh)[0]
    x=x.take(inds)  ;y=y.take(inds)
    inds=np.where(x>=FAM_thresh)[0]
    x=x.take(inds)  ;y=y.take(inds)
    x,y,z=clean_xy(x,y)
    sns.set_style('ticks')   
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
#    cmap=sns.cubehelix_palette(light=1, reverse=True, as_cmap=True)
    
    r2=np.corrcoef(x,y)[0,1]
    print(r2)
    fig, ax = plt.subplots()
    ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_xlim(var_range)
    ax.set_ylim(var_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
    ax.plot(var_range,var_range,color='grey',lw=0.6)
    ax.annotate('$R^2=$%0.2f'%r2, xy=(0.1, 0.85), xycoords='axes fraction',\
                ha='left',va='top')
#    sns.kdeplot(x,y, cmap='PuRd', n_levels=60, shade=True,ax=ax)
#    ax.annotate('Bounding box = %0.2f wide'%(1.-2*bound_thresh), xy=(0.1, 0.95),\
#                xycoords='axes fraction',\
#                ha='left',va='top')
    ax.annotate('$FAM_{species} \geq$ %0.2f'%FAM_thresh, xy=(0.1, 0.95),\
                xycoords='axes fraction',\
                ha='left',va='top')

def plot_ever_deci(cmap=sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=False)):
    os.chdir('D:/Krishna/Project')
    Df=pd.read_excel('working_tables.xlsx',sheetname='gc_ever_deci',index_col=1)  
    high_mort = [ 33,  73,  83,  84,  91,  97, 104, 105, 117, 118, 128, 129, 130,
                137, 138, 139, 141, 149, 150, 151, 160, 161, 170, 171, 172, 173,
                181, 182, 183, 184, 195, 196, 197, 198, 207, 208, 209, 218, 219,
                220, 221, 227, 230, 231, 232, 233, 240, 258, 259, 260, 261, 268,
                272, 273, 274, 277, 281, 287, 288, 289, 290, 296, 304, 308, 311,
                312, 317, 320, 326, 328, 334, 335, 336, 343, 348, 349, 350]
    Df_subset=Df.loc[high_mort]
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    cmap='BuGn'
    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_yticks([0,0.25,0.5,0.75,1])
    sns.kdeplot(Df['evergreen'],Df['deciduous'], cmap=cmap, n_levels=60, shade=True,ax=ax)
    ax.scatter(Df_subset['evergreen'],Df_subset['deciduous'],c=cardinal, s=30, \
               linewidth=0.5, marker="x",label='FAM $\geq$ 0.1')
    ax.collections[0].set_alpha(0)
    ax.set_xlabel('Evergreen woody cover')
    ax.set_ylabel('Deciduous woody cover')
    Df.loc[abs(Df['evergreen']-Df['deciduous'])<=0.25].index # get grid IDS where difference less than 0.25
    ax.add_patch(Rectangle((0.20,0.20), 0.6, 0.6,alpha=0.4,color='grey',label='Subset'))
    # add fill between code
    x = np.arange(0.0, 2, 0.01)
    y1 = x-0.25
    y2 = x+0.25
#    ax.fill_between(x, y1, y2,alpha=0.4,color='grey')
    legend = ax.legend()
    legend.get_frame().set_facecolor('blue')
    plt.show()

def plot_north_south_thresh(data='RWC',data_label="RWC (-)",\
                    mort_label=(1-ppt)*'FAM (-)'+ppt*'Fractional Area of Mortality (-)',\
                    mort_key='mortality_%03d_grid', mort_range=[0,0.5],\
                    grid_size=25,cmap='viridis',\
                    ticks=5,\
                    alpha=1):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    
    mort=store[mort_key%(grid_size)]
    data=store[data]

    mort, data = select_years(2009, 2015, mort,data)
    
    publishable.set_figsize(0.9*zoom, 2.8*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col')
    plt.subplots_adjust(hspace=0.25)
    ax=axs[0]
    location='Northern California'

    x, _=select_north_south_grids(data)
    y, _=select_north_south_grids(mort)
    x,y = x.values.flatten(), y.values.flatten()
    

    x,y,z=clean_xy(x,y)
    plot_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_ylabel(mort_label)
    guess=(0.3,0.05,1e-1,1e-1)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    plot_data=ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ymin,ymax=mort_range[0], mort_range[1]
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(mort_range)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f'%(location, r_squared))
    print(popt[0])
#    ax.set_title('%s trees'%species.title(),loc='left')
    ax.annotate('%s'%location, xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
    
    ax=axs[1]
    guess=(0.3,0.05,1e-1,1e-1)
    location='Southern California'
    ax.annotate('%s'%location, xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
    _, x=select_north_south_grids(data)
    _, y=select_north_south_grids(mort)
   
    x,y = x.values.flatten(), y.values.flatten()
    x,y,z=clean_xy(x,y)
    plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_xlabel(data_label)
    ax.set_ylabel(mort_label)
    popt , pcov = optimize.curve_fit(piecewise_linear, x, y, guess)
    perr = np.sqrt(np.diag(pcov))
    xd = np.linspace(min(x), max(x), 1000)
    ax.plot(xd, piecewise_linear(xd, *popt),'r--',linewidth=1)
    ax.fill_between(xd, piecewise_linear(xd, popt[0],popt[1],popt[2]-perr[2],popt[3]-perr[3]),\
                    piecewise_linear(xd, popt[0],popt[1],popt[2]+perr[2],popt[3]+perr[3]), \
                                    color='r',alpha=0.6)
    
    ax.axvline(popt[0],linestyle='--',linewidth=2,color='k')
    ax.add_patch(Rectangle([popt[0]-perr[0],ymin],2*perr[0],ymax-ymin,\
                          hatch='//////', color='k', lw=0, fill=False,zorder=10))
    ax.set_ylim(mort_range)
    residuals = y- piecewise_linear(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print('R-squared for %s = %0.2f'%(location, r_squared))
    print(popt[0])
    
    cbaxes = fig.add_axes([0.95, 0.123, 0.06, 0.76])
    cb=fig.colorbar(plot2_data,ax=axs[1],\
                    ticks=[min(z), max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off',pad=4)
    cbaxes.annotate('Point\ndensity',xy=(0,1.05), xycoords='axes fraction',\
                ha='left')
#    cb.outline.set_visible(False)

    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')

#    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.annotate('Dry', xy=(0., -0.28), xycoords='axes fraction',color=cardinal)
    ax.annotate('Wet', xy=(0.9, -0.28), xycoords='axes fraction',color='dodgerblue')
    ax.set_xticks([0,0.25,0.50,0.75,1])
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(plot2_data, cax=cax,ticks=[min(z),max(z)])
#    cax.annotate('Point\ndensity',xy=(0,1.04), xycoords='axes fraction',\
#                ha='left')
#    cax.set_yticklabels(['Low', 'High'])
#    cax.tick_params(axis='y', right='off',pad=0)
def plot_rwc_cwd_all_forest_type(data1='RWC_matched',data2='cwd',data1_label="RWC (-)",\
                     data2_label='CWD (mm)',
                    mort_label=(1-ppt)*'FAM (-)'+ppt*'Fractional Area of Mortality (-)',\
                    mort='mortality_025_grid',cmap='summer'):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    zoom = 1
    data1, data2, mort=store[data1], store[data2], store[mort]
    data1, data2, mort=select_years(2009, 2015, data1, data2, mort)
    data1_north, data1_south=select_north_south_grids(data1)
    data2_north, data2_south=select_north_south_grids(data2)
    mort_north, mort_south=select_north_south_grids(mort)
    
    grid_mapper = pd.read_pickle('grid_to_3_groups')
    group = 'hardwood'
    grids = grid_mapper.loc[grid_mapper==group].index
    data1, data2, mort=data1.loc[:,grids], data2.loc[:,grids], mort.loc[:,grids]          
                                               
    publishable.set_figsize(1.6*zoom, 2.35*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=3,ncols=2)
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
###===========================================================================   
    ax = axs[0,0]
    plot_data, z = scatter_threshold(data1,mort, ax, 'All regions', FAM_thresh = 0.,\
                                     guess=(1e-1,1e-2,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.set_xlabel('')
    ax.set_xticklabels([])
###===========================================================================       
    ax=axs[1,0]
    plot_data, z = scatter_threshold(data1_north,mort_north, ax, 'Northern domain', FAM_thresh = 0.,\
                                     guess=(0.08,0.02,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    adjust_spines(ax,['left','bottom'])
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
###===========================================================================       
    ax=axs[2,0]
    plot_data, z = scatter_threshold(data1_south,mort_south, ax, 'Southern domain', FAM_thresh = 0.,\
                                     guess=(0.08,0.02,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    adjust_spines(ax,['left','bottom'])
#    ax.set_xlabel('')
#    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.set_xlabel(data1_label)
###===========================================================================   
    ax = axs[0,1]
    plot_data, z = scatter_threshold(data2,mort, ax, 'All regions', FAM_thresh = 0.,\
                                     guess = (600,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])    
###===========================================================================       
    ax = axs[1,1]
    plot_data, z = scatter_threshold(data2_north,mort_north, ax, 'Northern domain', FAM_thresh = 0.,\
                                     guess = (600,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
###===========================================================================       
    ax = axs[2,1]
    plot_data, z = scatter_threshold(data2_south,mort_south, ax, 'Southern domain', FAM_thresh = 0.,\
                                     guess = (700,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    
    ax.set_yticklabels([])
    ax.set_xlabel(data2_label)
#    axs[2,1].annotate('Dry', xy=(0.9, -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,1].annotate('Wet', xy=(0., -0.4), xycoords='axes fraction',color='dodgerblue')
#    axs[2,0].annotate('Dry', xy=(0., -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,0].annotate('Wet', xy=(0.9, -0.4), xycoords='axes fraction',color='dodgerblue')
###===========================================================================       
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(plot_data, cax=cax,ticks=[min(z),max(z)], cmap = cmap)
    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
                ha='left')
    cax.set_yticklabels(['Low', 'High'])
    cax.tick_params(axis='y', right='off',pad=0)
###===========================================================================   
    shift = -0.53
    axs[2,0].annotate('', xy=(0, shift), xycoords='axes fraction', xytext=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,0].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,1].annotate('', xytext=(0, shift), xycoords='axes fraction', xy=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,1].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,0].annotate('|', xy=(0.99, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'left')
    axs[2,1].annotate('|', xy=(0.01, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'right')
    

###===========================================================================   
#    if save_fig:
#        plt.savefig(Dir_ms_fig+'/Figure_2.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()
    
    
def plot_rwc_cwd_all(data1='RWC_matched',data2='cwd',data1_label="RWC (-)",\
                     data2_label='CWD (mm)',
                    mort_label=(1-ppt)*'FAM (-)'+ppt*'Fractional Area of Mortality (-)',\
                    mort='mortality_025_grid',cmap='summer'):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    zoom = 1
    data1, data2, mort=store[data1], store[data2], store[mort]
    data1, data2, mort=select_years(2009, 2015, data1, data2, mort)
    data1_north, data1_south=select_north_south_grids(data1)
    data2_north, data2_south=select_north_south_grids(data2)
    mort_north, mort_south=select_north_south_grids(mort)
    
    
    publishable.set_figsize(1.6*zoom, 2.35*zoom, aspect_ratio = 1)
    
    fig, axs = plt.subplots(nrows=3,ncols=2)
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
###===========================================================================   
    ax = axs[0,0]
    plot_data, z = scatter_threshold(data1,mort, ax, 'All regions', FAM_thresh = 0.,\
                                     guess=(1e-1,1e-2,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.set_xlabel('')
    ax.set_xticklabels([])
###===========================================================================       
    ax=axs[1,0]
    plot_data, z = scatter_threshold(data1_north,mort_north, ax, 'Northern domain', FAM_thresh = 0.,\
                                     guess=(0.08,0.02,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    adjust_spines(ax,['left','bottom'])
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
###===========================================================================       
    ax=axs[2,0]
    plot_data, z = scatter_threshold(data1_south,mort_south, ax, 'Southern domain', FAM_thresh = 0.,\
                                     guess=(0.08,0.02,1e-4,1e-2), x_range = [-.05,1],\
                                           cmap = cmap)
    ax.set_xticks([0,0.25,0.50,0.75,1])
    adjust_spines(ax,['left','bottom'])
#    ax.set_xlabel('')
#    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.set_xlabel(data1_label)
###===========================================================================   
    ax = axs[0,1]
    plot_data, z = scatter_threshold(data2,mort, ax, 'All regions', FAM_thresh = 0.,\
                                     guess = (600,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])    
###===========================================================================       
    ax = axs[1,1]
    plot_data, z = scatter_threshold(data2_north,mort_north, ax, 'Northern domain', FAM_thresh = 0.,\
                                     guess = (600,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
###===========================================================================       
    ax = axs[2,1]
    plot_data, z = scatter_threshold(data2_south,mort_south, ax, 'Southern domain', FAM_thresh = 0.,\
                                     guess = (700,0.05,1e-3,1e-3), x_range = [200,1200],\
                                           cmap = cmap)
    ax.set_xticks([200,450,700, 950, 1200])
    ax.set_yticks(np.arange(0,0.6,0.1))
    adjust_spines(ax,['left','bottom'])
    ax.axes.get_yaxis().get_label().set_visible(False)
    
    ax.set_yticklabels([])
    ax.set_xlabel(data2_label)
#    axs[2,1].annotate('Dry', xy=(0.9, -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,1].annotate('Wet', xy=(0., -0.4), xycoords='axes fraction',color='dodgerblue')
#    axs[2,0].annotate('Dry', xy=(0., -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,0].annotate('Wet', xy=(0.9, -0.4), xycoords='axes fraction',color='dodgerblue')
###===========================================================================       
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(plot_data, cax=cax,ticks=[min(z),max(z)], cmap = cmap)
    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
                ha='left')
    cax.set_yticklabels(['Low', 'High'])
    cax.tick_params(axis='y', right='off',pad=0)
###===========================================================================   
    shift = -0.53
    axs[2,0].annotate('', xy=(0, shift), xycoords='axes fraction', xytext=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,0].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,1].annotate('', xytext=(0, shift), xycoords='axes fraction', xy=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,1].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,0].annotate('|', xy=(0.99, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'left')
    axs[2,1].annotate('|', xy=(0.01, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'right')
    

###===========================================================================   
    if save_fig:
        plt.savefig(Dir_ms_fig+'/Figure_2.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()

def binned_plot(data, mort, ax, panel_label):
    bins = np.linspace(0.0,1, 11)
    labels = np.linspace(0.1,1, 10)
    if data.index.name=='CWD':
        bins = np.linspace(200,1200, 11)
        labels = np.linspace(300,1200, 10).astype(int)
    df = pd.DataFrame({data.index.name:data.values.flatten(), \
                       mort.index.name:mort.values.flatten()})
    df.dropna(inplace = True)
    df['binned'] = pd.cut(df[data.index.name], bins=bins, labels=labels)
    sns.boxplot(x="binned", y="FAM",data=df, linewidth=1.5,ax = ax, color = 'teal', fliersize = 0)
    ax.annotate('%s'%panel_label, xy=(0.01, 1.03), xycoords='axes fraction',\
                ha='left',va='bottom')
    ax.set_xlabel("")
    
def plot_rwc_cwd_all_v2(data1='RWC_matched',data2='cwd',data1_label="RWC",\
                     data2_label='CWD (mm)',
                    mort_label=(1-ppt)*'FAM (-)'+ppt*'Fractional Area of Mortality (-)',\
                    mort='mortality_025_grid',cmap='viridis', high_thresh = 0.05):
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore(filename)
    zoom = 1
    data1, data2, mort=store[data1], store[data2], store[mort]
    data1, data2, mort=select_years(2009, 2015, data1, data2, mort)
    mort[mort<high_thresh] = np.nan
    data1_north, data1_south=select_north_south_grids(data1)
    data2_north, data2_south=select_north_south_grids(data2)
    mort_north, mort_south=select_north_south_grids(mort)
    publishable = change_font_size(9) 
    publishable.set_figsize(1.6*zoom, 2.35*zoom, aspect_ratio = 1)
    
    fig, axs = plt.subplots(nrows=3,ncols=2)
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
###===========================================================================   
    ax = axs[0,0]
    binned_plot(data1, mort, ax, "All regions")
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.set_xlabel('')
    ax.set_xticklabels([])
###===========================================================================       
    ax=axs[1,0]
    binned_plot(data1_north, mort_north, ax, "Northern domain")
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
###===========================================================================       
    ax=axs[2,0]
    binned_plot(data1_south, mort_south, ax, "Southern domain")
#    ax.set_xlabel('')
#    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.set_xlabel(data1_label)
###===========================================================================   
    ax = axs[0,1]
    binned_plot(data2, mort, ax, "All regions")
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])    
###===========================================================================       
    ax = axs[1,1]
    binned_plot(data2_north, mort_north, ax, "Northern domain")
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.axes.get_yaxis().get_label().set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
###===========================================================================       
    ax = axs[2,1]
    binned_plot(data2_south, mort_south, ax, "Southern domain")
    ax.set_yticks(np.arange(0,0.6,0.1))
    ax.axes.get_yaxis().get_label().set_visible(False)
    
    ax.set_yticklabels([])
    ax.set_xlabel(data2_label)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)
#    axs[2,1].annotate('Dry', xy=(0.9, -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,1].annotate('Wet', xy=(0., -0.4), xycoords='axes fraction',color='dodgerblue')
#    axs[2,0].annotate('Dry', xy=(0., -0.4), xycoords='axes fraction',color=cardinal)
#    axs[2,0].annotate('Wet', xy=(0.9, -0.4), xycoords='axes fraction',color='dodgerblue')
###===========================================================================       
#    divider = make_axes_locatable(axs[0,1])
#    cax = divider.append_axes("right", size="5%", pad=0.1)
#    fig.colorbar(plot_data, cax=cax,ticks=[min(z),max(z)])
#    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
#                ha='left')
#    cax.set_yticklabels(['Low', 'High'])
#    cax.tick_params(axis='y', right='off',pad=0)
###===========================================================================   
    shift = -0.41
    axs[2,0].annotate('', xy=(0, shift), xycoords='axes fraction', xytext=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,0].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,1].annotate('', xytext=(0, shift), xycoords='axes fraction', xy=(1, shift), 
            arrowprops=dict( color='grey', width=0.4, headwidth=8,headlength=20))
    axs[2,1].annotate('Drier', color = 'grey',xy=(0.5, shift), xycoords='axes fraction',\
                va = 'bottom', ha = 'center')
    axs[2,0].annotate('|', xy=(0.99, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'left')
    axs[2,1].annotate('|', xy=(0.01, shift), fontsize = fs - 2, xycoords='axes fraction',\
       color = 'grey', va = 'center', ha = 'right')
#    for ax in axs.ravel():
#        ax.set_ylim(0,0.1)

##===========================================================================   
    plt.savefig(Dir_ms_fig+'/high_mort_rwc_cwd_box_plot.tiff', dpi = dpi, bbox_inches="tight")
    plt.show()




def plot_inc_cumm_corr(cmap=sns.dark_palette("seagreen",as_cmap=True,reverse=True),\
                       alpha = 1,\
                       var1_range = [-0.02,0.5],var1_label = 'Incremental FAM (-)',\
                       var2_range = [-0.02,1.0],var2_label = 'Cumulative FAM (-)'):
    os.chdir(Dir_CA)  
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    store=pd.HDFStore(filename)
    var1=store['mortality_025_grid']
    var2=store['mortality_025_grid']
    var2=var2.cumsum(0)
    var1=var1[(var1.index.year>=start_year) &\
      (var1.index.year<=end_year)]  
    var2=var2[(var2.index.year>=start_year) &\
      (var2.index.year<=end_year)]
    
    fig, ax = plt.subplots(1,1)
    x,y,z1=clean_xy(var1.values.flatten(),var2.values.flatten(), thresh = 0.0)
    plot=ax.scatter(x,y,marker='o',c=z1,cmap=cmap,edgecolor='',alpha=alpha, s=20)
    xd=np.linspace(min(x),max(x),100)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.plot(xd,intercept + xd*slope,'b--',lw=1, label = 'All')
    ax.fill_between(xd,intercept + xd*(slope-std_err),intercept + xd*(slope+std_err),\
                    color='b',alpha=0.5)
    
    x,y,z=clean_xy(var1.values.flatten(),var2.values.flatten(), thresh = 0.1)
    xd=np.linspace(min(x),max(x),100)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.plot(xd,intercept + xd*slope,'m--',lw=1, label = 'FAM>0.1')
    ax.fill_between(xd,intercept + xd*(slope-std_err),intercept + xd*(slope+std_err),\
                    color='m',alpha=0.5)
    
    ax.legend()
    ax.set_xlim(var1_range)
    ax.set_ylim(var2_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
#    ax.annotate('$R^2=%0.2f$'%r_value**2, xy=(0.10, 0.90), xycoords='axes fraction',\
#                ha='left',va='top')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot, cax=cax,ticks=[min(z1),max(z1)])
    cax.annotate('Point\ndensity',xy=(0,1.06), xycoords='axes fraction',\
                ha='left')
    cax.set_yticklabels(['Low', 'High'])
    cax.tick_params(axis='y', right='off',pad=0)
    plt.savefig(Dir_ms_fig+'/Figure_S4.tiff', dpi = dpi, bbox_inches="tight")
    return r_value**2

def plot_north_south_map():
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    Df=pd.HDFStore(filename)['mortality_025_grid']
    Df_north, Df_south=select_north_south_grids(Df)
    Df.loc[:,Df_north.columns] = 'orange'
    Df.loc[:,Df_south.columns] = 'green'
    
    ### plotting
    latcorners=np.array([32.5,42])
    loncorners=np.array([-124.4,-116.4]) 
    lats,lons, gridID =supply_lat_lon('GC_subset_no_ocean', return_gridID = True)
#    colors = Df.loc['2005-01-01',:].copy()
    #### trying to color by host forest
    grid_mapper = pd.read_pickle('grid_to_3_groups')
    grid_mapper = grid_mapper.loc[gridID]
    grid_mapper.replace('fir', 'lightgreen', inplace = True)
    grid_mapper.replace('pine', 'darkolivegreen', inplace = True)
    grid_mapper.replace('hardwood', 'saddlebrown', inplace = True)
    colors = grid_mapper.copy()
    
    ### experiment RGB plot. Colors = RGB
    fractions = pd.read_pickle('grid_to_3_groups_fraction')
    fractions = fractions.loc[gridID]
    fractions = fractions.loc[:,['hardwood','fir','pine']]
    colors = fractions.copy()
            
    zoom = 2
    publishable.set_figsize(1*zoom, 2*zoom, aspect_ratio =1)
    fig, ax = plt.subplots()
    fig, ax, m = plot_map(lats,lons, var =colors.values,\
             latcorners = latcorners, loncorners = loncorners,\
             enlarge = 1, marker_factor = 1.8, \
             cmap = 'YlGnBu', markercolor = 'r',\
             fill = 'papayawhip', background = 'lightcyan',\
             height = 1, width = 1,\
             drawcoast = False, drawcountries = False,\
             drawstates = False, drawcounties = False,\
             resolution = 'l', fig = fig, ax = ax,\
             shapefilepath = '%s/CA'%Dir_CA,\
             shapefilename = "CA")
    plt.savefig(Dir_ms_fig+'/Figure_S3_rgb.tiff', dpi = dpi, bbox_inches="tight")
#    m.readshapefile(Dir_CA+'/CA','CA',drawbounds=True, color='black')

def plot_cdf_match():
    years = OrderedDict()
    years['2003-2005']='s'
    years['2006-2008']='o'
    years['2009-2010']='^'
    years['2003-2010']='v'
    years['2013-2015']='p'
    publishable.set_figsize(2*zoom, 1*zoom, aspect_ratio =1)
    fig, ax = plt.subplots()
    counter = 0
    os.chdir("D:/Krishna/Project/data/Mort_Data/Misc_data")
    
    for year in years.keys():
        data = pd.read_pickle(year)
        sensor = " (AMSR-E)"
        if year == '2013-2015':
            sensor = " (AMSR-2)"
        data.plot(ax=ax,label = year+sensor, marker = years[year], \
                  markevery = (counter,250), markersize = 6, lw = 1)
        counter+=50
    ax.set_xlim(0,3)
    plt.legend()
    ax.set_xlabel('VOD')
    ax.set_ylabel('Likelihood of occurence')
    plt.savefig(Dir_ms_fig+'/Figure_S1.tiff', dpi = 300, bbox_inches="tight")
    plt.show()
save_fig = True
def main():
#    plot_RWC_definition_v2(data_source1='vod_pm_matched',grid_cell = 182, save = False) #Figure 1
#    plot_rwc_cwd_all()  #Figure 2
    plot_timeseries_maps() #Figure 3
#    plot_regression() #Figure 4
#    Df = plot_importance(filename ='rf_sensitivity_importance_base_model', \
#                    width = 0.9, height = 2.4, save = True, savename = 'Figure_5') #Figure 5
#    Df =plot_importance(filename ='rf_sensitivity_importance_lagged_model', \
#                    width = 0.9, height = 2.5, save = False, \
#                    savename = 'Figure_6') #Figure1 6
#    plot_north_south_map()

################################################################################
#    plot_importance(filename ='rf_sensitivity_importance_trimmed_model',\
#                    width = 1.1, height = 0.8, save = True, savename = 'Figure_S7')
#    Df = plot_importance(filename ='rf_sensitivity_importance_lai_model', \
#                    width = 0.9, height = 2.4, save = True, savename = 'Figure_S8') #Figure 5
    plot_grid() #Figure S5
#    plot_heatmap(cmap = 'PRGn') #Figure S6
#    plot_inc_cumm_corr() #Figure S4
#    plot_vod_matched() #Figure S2
#    plot_cdf_match() # Figure cdf match S1

###############################################################################    
#    plot_RWC_definition_with_vpd(data_source1='vod_pm_matched') #Figure 1
#    plot_VOD_by_LAI(data_source1='vod_pm_matched') #Figure 1

#    plot_RWC_definition(data_source1='vod_pm_matched') #old Figure 1
#     plot_rwc_cwd_all_forest_type()  #Figure 2
#    plot_rwc_cwd() 
#    plot_rwc_cwd_all_v2()
#    plot_boxplot() 
#    plot_LPDR2()
#    plot_ever_deci()
#    plot_pdf()
#    plot_leaf_habit_thresh()
#    plot_leaf_habit_mort()
#    plot_FAM_TPA_corr()
#    plot_north_south_thresh(data='RWC',data_label="RWC (-)")

if __name__ == '__main__':
    main()