# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:11:14 2017

@author: kkrao
"""
from __future__ import division
import arcpy
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
                    piecewise_linear,append_prediction, subset_forest_cov
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap
from scipy import optimize
from matplotlib.patches import Rectangle
from datetime import timedelta
import types
from matplotlib.ticker import FormatStrFormatter
#
cardinal='#BD2031'
zoom=2.0
fs=24
fontsizes=['font.size','lines.markersize',
 'legend.fontsize',
 'axes.labelsize','xtick.labelsize','ytick.labelsize']
plotsettings.Set('EcolLett')
for x in fontsizes:
    plotsettings.journals.journals['EcolLett']['rcParams'][x]=fs
#mpl.rcParams['font.size'] = 32
def plot_settings(zoom=zoom):
#    sns.set(font_scale=zoom*1.2/1.5)
#    mpl.rcParams['font.size'] = 32
#    zoom = zoom
    return(zoom)
#    publishable = plotsettings.Set(journal)
#    publishable.set_figsize(2, 1, aspect_ratio = 1)    

def flatten_diagonally(npA, diagonals = None):
    diagonals = diagonals or xrange(-npA.shape[0] + 1, npA.shape[1])
    return np.concatenate(map(lambda x: \
          np.diagonal(np.rot90(npA), offset = x), diagonals))

def plot_timeseries_maps(var1='mortality_%03d_grid', var1_range=[1e-5, 0.4],\
                         var1_label="Observed FAM",\
                    var2='predicted_FAM', var2_range=[1e-5, 0.4],\
                     var2_label="Predicted FAM",\
                    grid_size=25,cmap='inferno_r',cmap2='inferno_r', \
                    start_month=7,months_window=3,ticks=5,\
                    title='Timeseries of observed and predicted mortality',proj='cyl',\
                    journal='EcolLett'):
    os.chdir(Dir_CA)
#    sns.set(font_scale=1.2)
    mpl.rcParams['font.size'] = 15
    store=pd.HDFStore('data_subset.h5')
    data_label=var2_label
    alpha=0.7
    mort_label=var1_label
    mort=store[var1%(grid_size)]
#    mort=mort[mort>0]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    pred_mort=append_prediction()
    year_range=mort.index.year
    cols=mort.shape[0]
    rows=2
    latcorners=np.array([33,42.5])
    loncorners=np.array([-124.5,-117]) 
    zoom=plot_settings()
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid_subset'
        marker_factor=7*zoom
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
    lats = [row[0] for row in arcpy.da.SearchCursor(grids, 'x')]
    lons = [row[0] for row in arcpy.da.SearchCursor(grids, 'y')]
    zoom=plot_settings()
    sns.set_style("white")
    publishable = plotsettings.Set(journal)
    
    publishable.set_figsize(2*zoom, 1*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=rows,ncols=cols   ,\
                            sharey='row')
    marker_size=get_marker_size(axs[0,0],fig,loncorners,grid_size,marker_factor)
    plt.subplots_adjust(wspace=0.04,hspace=0.04,top=0.83)
    parallels = np.arange(*latcorners+2,step=5)
    meridians = np.arange(*loncorners+1.5,step=5)
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
        plot_mort=m.scatter(lats, lons,s=marker_size,c=mort_plot,cmap=cmap,\
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
        plot_data=m.scatter(lats, lons,s=marker_size,c=data_plot,cmap=cmap2\
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
    axs[0,0].set_ylabel(mort_label)
    axs[1,0].set_ylabel(data_label)
#    fig.suptitle(title)
#    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
#                                                 prefix = '', suffix = '.', fontweight = 'bold')
    scalebar = ScaleBar(100*1e3*1.05,box_alpha=0,sep=2,location='lower left') # 1 pixel = 0.2 meter
#    ax.add_artist(scalebar)
    ax.annotate('CA State', xy=(0.1, 0), xycoords='axes fraction',\
                    ha='left',va='bottom',size=16)
    plt.show()
    return cb0
#==============================================================================
def plot_leaf_habit(data='RWC',data_label="RWC (-)",\
                    mort_label='FAM (-)',\
                    mort='mortality_%03d_grid', data_range=[0,1],mort_range=[0,0.7],\
                    grid_size=25,cmap='viridis',\
                    ticks=5,\
                    journal='EcolLett',alpha=0.7):
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
    store=pd.HDFStore('data_subset.h5')
    mort=store[mort%(grid_size)]
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    data=store[data]
    data=data[(data.index.year>=start_year) &\
              (data.index.year<=end_year)]    
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()
    publishable.set_figsize(0.76*zoom, 1.7*zoom, aspect_ratio = 1)
    fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col')
    plt.subplots_adjust(hspace=0.25)
    ax=axs[0]
    species='evergreen'
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
                            start_month=7,months_window=3,journal='EcolLett'):
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
    
#    data=store[data_source1]
#    data=data[(data.index.year>=start_year) &\
#      (data.index.year<=end_year)]  
#    data=data.rolling(30,min_periods=1).mean()
#    #           mask=(data.index.month>=start_month) & (data.index.month<start_month+months_window)
#    #           data[~mask]=np.nan
    publishable.set_figsize(0.5*zoom, 1.35*zoom, aspect_ratio =1)
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
#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
            prefix = '', suffix = '.', fontweight = 'bold')        
    plt.show()  
        
def plot_pdf(data_source1='vod_pm',data_source2='RWC',data_source3='cwd',\
                            \
                            start_month=7,months_window=3,journal='EcolLett'):
    publishable = plotsettings.Set(journal)
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_data.csv',index_col=0)
    input_sources=['FAM','live_basal_area','LAI_sum',\
     'aspect_mean', 'canopy_height',\
     'cwd','elevation_mean',\
     'forest_cover','ppt_sum','tmax_sum',\
     'tmean_sum','vpdmax_sum','EVP_sum',\
    'PEVAP_sum','vsm_sum','RWC']
    zoom=plot_settings()
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
    
def plot_regression(var1='FAM', var1_range=[-0.02, 0.42],\
                         var1_label="Observed FAM (-)",\
                    var2='predicted_FAM', var2_range=[-0.02, 0.42],\
                     var2_label="Predicted FAM (-)",\
                    grid_size=25,cmap='PuOr', \
                    ticks=5,\
                    title='Regression of observed and predicted mortality',proj='cyl',\
                    journal='EcolLett',dataset='test_data'):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s.csv'%dataset,index_col=0)   
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    z=Df['RWC']
    plot=ax.scatter(Df[var1],Df[var2],marker='s',c=z,cmap=cmap,s=scatter_size)
    ax.set_xlim(var1_range)
    ax.set_ylim(var2_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
    ax.plot(var1_range,var2_range,color='grey',lw=0.6)
    cbaxes = fig.add_axes([0.2, 0.50, 0.1, 0.24])
    cb=fig.colorbar(plot,ax=ax,\
                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off')
    cbaxes.annotate('RWC',xy=(0,1.2), xycoords='axes fraction',\
                ha='left')
    cb.outline.set_visible(False)
    ax.annotate('$R^2_{test}=$0.45', xy=(0.98, 0.57), xycoords='axes fraction',\
                ha='right',va='top')
    ax.annotate('1:1 line', xy=(0.9, 0.97), xycoords='axes fraction',\
                ha='right',va='top',color='grey')

def plot_importance(journal='EcolLett'):
    publishable = plotsettings.Set(journal)
    os.chdir(Dir_CA)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_sensitivity_importance.csv',index_col=0)   
    zoom=plot_settings()
    publishable.set_figsize(1.6*zoom, 1.5*zoom, aspect_ratio =1)
    sns.set_style('whitegrid')
    Df=Df.sort_values('mean')
    
    fig, ax = plt.subplots()
    plot=Df['mean'].plot.barh(width=0.8,color='grey',xerr=Df['sd'],\
           error_kw=dict(ecolor='k', lw=1, capsize=2, capthick=1),ax=ax)
#    ax.tick_params(axis='y', left='off',pad=-1)
    ax.set_xlabel('Importance (-)')
    ax.yaxis.grid(False)

def plot_FAM_TPA_corr(var1='mortality_%03d_grid', var1_range=[-0.02, 0.42],\
                         var1_label="Observed FAM (-)",\
                    var2='TPA_%03d_grid', var2_range=[-0.6, 13],\
                     var2_label="Dead trees (acre"+"$^{-1}$"+")",\
                    grid_size=25,cmap=sns.dark_palette("seagreen",as_cmap=True,reverse=True), \
                    ticks=5,\
                    title='Regression of observed and predicted mortality',proj='cyl',\
                    journal='EcolLett',alpha=0.6):
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()
    os.chdir(Dir_CA)  
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    store=pd.HDFStore('data_subset.h5')
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
    plot=ax.scatter(x,y,marker='s',c=z,cmap=cmap,edgecolor='',alpha=alpha)
    ax.plot(xd,intercept + xd*slope,'r--',lw=1)
    ax.fill_between(xd,intercept + xd*(slope-std_err),intercept + xd*(slope+std_err),\
                    color='red',alpha=alpha)
#    ax.axis('equal')
    ax.set_xlim(var1_range)
    ax.set_ylim(var2_range)
    ax.set_xlabel(var1_label)
    ax.set_ylabel(var2_label)
#    ax.plot(var1_range,var2_range,color='grey',lw=0.6)
    cbaxes = fig.add_axes([0.2, 0.65, 0.03, 0.1])
    cb=fig.colorbar(plot,ax=ax,\
                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off',pad=-4)
    cbaxes.annotate('Scatter Point\ndensity',xy=(0,1.2), xycoords='axes fraction',\
                ha='left')
    cb.outline.set_visible(False)
#    ax.annotate('1:1 line', xy=(0.9, 0.95), xycoords='axes fraction',\
#                ha='right',va='top')
    return r_value**2
    
def plot_importance_rank(journal='EcolLett'):
    publishable = plotsettings.Set(journal)
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
                            start_month=7,months_window=3,journal='EcolLett'):
    publishable = plotsettings.Set(journal)
    sns.set_style("ticks")
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
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
                            \
                            start_month=7,months_window=3,journal='EcolLett'\
                            ,alpha1=0.2,color='#BD2031',alpha2=0.7):
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()*1.5
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
    data=store[data_source1]
    data=data[(data.index.year>=start_year) &\
      (data.index.year<=end_year)]  
    data=data.rolling(30,min_periods=1).mean()
    publishable.set_figsize(1.2*zoom, 0.3*zoom, aspect_ratio =1)
    sns.set_style('ticks')
    fig, ax = plt.subplots(1,1,sharex=True)
#    ax.grid(axis='x')
    ax.set_ylabel(data_label1)
    data=data.loc[:,104]
    ax.plot(data,'-',color='k')  
    for year in np.unique(data.index.year):
        ax.axvspan(*pd.to_datetime(['%d-07-01'%year,'%d-09-30'%year]), alpha=alpha1, facecolor=color)
    mask=(data.index.month>=start_month) & (data.index.month<start_month+months_window)
    data[~mask]=np.nan
    u,l=data.quantile(0.95),data.quantile(0.05)
#    ax.axhline(u,xmin=0.07,ls='--',color=color,lw=1,dashes=(3.45, 10.7))  
#    ax.axhline(l,xmin=0.07,ls='--',color=color,lw=1,dashes=(3.45, 10.7))  
#    ax.axhline(u,ls='dotted',color=color,lw=1)  
#    ax.axhline(l,ls='dotted',color=color,lw=1)
    ax.scatter(idxquantile(data,q=0.05),data.quantile(0.05),s=30,lw=2,c='None',edgecolor=color,marker='o')
    ax.scatter(idxquantile(data,q=0.95),data.quantile(0.95),s=30,lw=2,c='None',edgecolor=color,marker='s')
    ax2 = ax.twinx()
    for year in np.unique(data.index.year):
        subset=data[data.index.year==year]
        ax.plot([idxquantile(subset),idxquantile(subset)],[l,subset.quantile(0.5)],\
                 ls='-',color=color,lw=8,alpha=alpha2,solid_capstyle='butt')
#        ax.plot([idxquantile(subset),idxquantile(subset)],[subset.quantile(0.5),u],\
#                 ls='-',color=color,lw=2,alpha=0.3)
#        ax.plot(idxquantile(subset),subset.quantile(0.5),'s',color=color,markersize=4)
    ax.set_xlim([data.index.min(),data.index.max()])
    ax2.set_ylabel('$\ \ $'+data_label2,color=color)
    ax2.tick_params(colors=color)
    ax2.set_yticks([0.32,(0.32+0.67)/2,0.67])
    ax2.set_yticklabels([0.0,0.5,1.0])
    ax.set_yticks(np.arange(1.2,1.6,0.1))
    ax2.grid(axis='y',alpha=0.2,color=color)
def plot_rwc_cwd(data1='RWC',data2='cwd',data1_label="RWC (-)",\
                 data2_label='CWD (mm)',
                    mort_label='FAM (-)',\
                    mort='mortality_%03d_grid',\
                    grid_size=25,cmap='viridis', \
                    \
                    alpha=0.7):
    if grid_size==25:
        grids=Dir_mort+'/CA_proc.gdb/grid'
        marker_factor=7
        scatter_size=20
    elif grid_size==5:
        grids=Dir_mort+'/CA_proc.gdb/smallgrid'
        marker_factor=2
        scatter_size=4
    publishable = plotsettings.Set(journal)
    zoom=plot_settings()
    sns.set_style("whitegrid")
    os.chdir(Dir_CA)
    store=pd.HDFStore('data_subset.h5')
    mort=store[mort%(grid_size)]
    mort=subset_forest_cov(mort)
    mort=mort[(mort.index.year>=start_year) &\
              (mort.index.year<=end_year)]
    mort_range=[0,mort.values.max()*1.1]
    data1=store[data1]
    data1=subset_forest_cov(data1)
    data1=data1[(data1.index.year>=start_year) &\
              (data1.index.year<=end_year)] 
    data2=store[data2]
    data2=subset_forest_cov(data2)
    data2=data2[(data2.index.year>=start_year) &\
              (data2.index.year<=end_year)]     
    
    publishable.set_figsize(2*zoom, 1*zoom, aspect_ratio = 1)
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
    guess=(0.01,0.05,1e-4,1e-2)
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
    print('R-squared for RWC = %0.2f'%(r_squared))
    ax.annotate('Dry', xy=(0, -0.14), xycoords='axes fraction',color=cardinal)
    ax.annotate('Wet', xy=(0.9, -0.14), xycoords='axes fraction',color='dodgerblue')
    ax=axs[1]
    ax.annotate('Dry', xy=(0.9, -0.14), xycoords='axes fraction',color=cardinal)
    ax.annotate('Wet', xy=(0, -0.14), xycoords='axes fraction',color='dodgerblue')
    x=data2.values.flatten()
    y=mort.values.flatten()
    x,y,z=clean_xy(x,y)
    plot2_data=ax.scatter(x,y,c=z,edgecolor='',cmap=cmap,alpha=alpha,marker='s',s=scatter_size)
    ax.set_xlabel(data2_label)
#    ax.set_ylabel(mort_label)
    guess=(500,0.05,1e-4,1e-2)
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
    print('R-squared for CWD = %0.2f'%(r_squared))
    cbaxes = fig.add_axes([0.38, 0.5, 0.04, 0.18])
    cb=fig.colorbar(plot2_data,ax=axs[1],\
                    ticks=[min(z)+0.1*max(z), 0.9*max(z)],cax=cbaxes)
    cb.ax.set_yticklabels(['Low', 'High'])
    cb.ax.tick_params(axis='y', right='off')
    cbaxes.annotate('Point\ndensity',xy=(0,1.2), xycoords='axes fraction',\
                ha='left')
    publishable.panel_labels(fig = fig, position = 'outside', case = 'lower',
                                                 prefix = '', suffix = '.', fontweight = 'bold')
    cb.outline.set_visible(False)
    
def plot_grid(data='data',journal='EcolLett',color='#BD2031'):
    publishable = plotsettings.Set(journal)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s.csv'%data,index_col=0)  
    Df.drop('dominant_leaf_habit',axis=1,inplace=True)
    importance=pd.read_csv('D:/Krishna/Project/data/rf_sensitivity_importance.csv',index_col=0)[:-1]
    input_sources=importance.index.tolist()
#    zoom=plot_settings()
    sns.set_style('ticks') 
    publishable.set_figsize(2*zoom, 2.05*zoom, aspect_ratio =1)
    fig, axs = plt.subplots(5,5,sharey='row')
    plt.subplots_adjust(hspace=0.4)
    axs=flatten_diagonally(axs, diagonals = None)
    niter=0
    for xcol, ax in zip(input_sources, axs):
        Df.plot(kind='scatter', x=xcol, y='FAM', ax=ax, \
                s=10,alpha=0.5, color='k',edgecolor='None')
        ax.set_xlabel('',labelpad=-5)
        ax.tick_params(axis='both', which='major', pad=1)
        ax.set_aspect('auto',adjustable='box-forced')
        ax.set_ylim([-0.02,0.42])
        ax.set_ylabel('')
#        if niter ==6:
#            ax.set_ylabel('FAM')
        niter+=1
        ax.annotate(xcol, xy=(0.5, 0.95), xycoords='axes fraction',\
                ha='center',va='top') 
    fig.text(0.04,0.5,'FAM',rotation=90)
def shift_labels(ax,SHIFT=0.5):
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_x = types.MethodType( lambda self, x: mpl.text.Text.set_x(self, x-self.customShiftValue ), 
                                    label, mpl.text.Text )
    for label in ax.yaxis.get_majorticklabels():
        label.customShiftValue = SHIFT
        label.set_y = types.MethodType( lambda self, y: mpl.text.Text.set_y(self, y-self.customShiftValue ), 
                                    label, mpl.text.Text )
        
def plot_heatmap(data='data',journal='EcolLett',cmap="YlGnBu"):
    publishable = plotsettings.Set(journal)
    Df=pd.read_csv('D:/Krishna/Project/data/rf_%s.csv'%data,index_col=0)   
    corr=Df.corr()  
    blank=["ppt_win","tmax_sum","tmax_win", "tmean_sum",\
           "tmean_win","vpdmax_sum","vpdmax_win","EVP_sum",\
           "PEVAP_sum","PEVAP_win","EVP_win"]
#    corr.loc[:3,blank]=np.NaN
    zoom=plot_settings(1.0)
    sns.set_style('white')
    publishable.set_figsize(1*zoom, 1*zoom, aspect_ratio =1)
    inds=sns.clustermap(corr,figsize=(1e-10,1e-10)\
                        ).dendrogram_row.reordered_ind     
    corr=corr[inds]
    corr=corr.reindex(corr.columns)
    mask = np.zeros_like(corr)
    show_inds=np.array([corr.columns.get_loc(c) for c in corr.columns if c not in blank])
    hide_inds=np.array([corr.columns.get_loc(c) for c in corr.columns if c in blank])
#    show_inds=np.arange(np.shape(mask)[1])[~blank_inds]
    mask[:,hide_inds]=True
    mask[hide_inds,:]=True   
    mask[np.tril_indices_from(mask)] = False
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((0,0),corr.shape[1],corr.shape[0],color='darkgrey',zorder=-1,alpha=0.5))
    cbar_ax = fig.add_axes([.9, .121, .03, .76])
    sns.heatmap(corr,vmin=-1,vmax=1,mask=mask,ax=ax,cbar_ax=cbar_ax,\
                square=True,cbar_kws={'ticks':np.linspace(-1,1,11)},cmap=cmap,\
                                      linewidths=.2)
#    plt.show()

#    plt.show()
#    ax.tick_params(axis='both', pad=-1,color='lightgrey')
    
def plot_rsq_subset(journal='EcolLett'):
    publishable = plotsettings.Set(journal)
    zoom=plot_settings(1.0)
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
    

def main():
    plot_RWC_definition()
#    plot_rwc_cwd()
#    plot_boxplot()
#    plot_timeseries_maps()
#    plot_importance()
#    plot_leaf_habit()
##    plot_FAM_TPA_corr()
#    plot_regression()
##    plot_pdf()
#    plot_grid()
#    plot_heatmap()

if __name__ == '__main__':
    main()