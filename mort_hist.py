from __future__ import division
from IPython import get_ipython
get_ipython().magic('reset -sf') 


'''

Code run order
1. tree_intersection
2. add_threshold
3. raster_proc
4. tree_stats


'''
import numpy as np
import pandas as pd
import matplotlib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.io
import os
import arcpy
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import copy
import matplotlib.mlab as mlab
from scipy.stats import norm

MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'

arcpy.env.workspace = Dir_mort+'/Mortality_intersect.gdb'
# arcpy.Statistics_analysis("futrds", "C:/output/output.gdb/stats", [["Shape_Length", "SUM"]], "NM")
start=5
end=16


sev=1 #severity index to be recorded
sev2=2
mort_summary=pd.DataFrame()

for i in range(start,end+1):
    area=[]
    severity=[]
    gridID=[]
    if i<10:       
        year="0%s" %i
    else: 
        year="%s" %i   
    file="ADS"+year+"_intersect"
    field = "frac_area_mort"
    cursor = arcpy.SearchCursor(file)
    for row in cursor:
        area.append(row.getValue('frac_area_mort'))
        severity.append(row.getValue("SEVERITY1"))
        gridID.append(row.getValue("gridID"))
    area=pd.DataFrame(area, columns=['fam_%s'%year])
    severity=pd.DataFrame(severity, columns=['sev_%s'%year])
    gridID=pd.DataFrame(gridID, columns=['gridID_%s'%year])
    mort_summary=pd.concat([mort_summary, area, severity,gridID],axis=1)

store = pd.HDFStore(Dir_CA+'/mort_summary.h5')
store['mort_summary'] = mort_summary    
fig=plt.figure()    

for i in list(reversed(range(start,end+1))):
    area=[]
    severity=[]
    if i<10:       
        year="0%s" %i
    else: 
        year="%s" %i
    data=mort_summary['fam_%s'%year]
    data=data[~np.isnan(data)]
    plt.hist(data,bins=20,alpha=0.5,edgecolor='black',label='Year = %s'%year)

ax=plt.gca()
fs=15
ax.set_xlabel('Fractional area of all mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,450])
ax.set_title('Histogram of fractional area of mortality (all kind)',fontsize=fs)
ax.legend()
plt.show()


for i in list(reversed(range(start,end+1))):
    area=[]
    severity=[]
    if i<10:       
        year="0%s" %i
    else: 
        year="%s" %i
    data=mort_summary.loc[mort_summary['sev_%s'%year] == 2, 'fam_%s'%year]
    data=data[~np.isnan(data)]
    if data.empty:
        continue
    plt.hist(data,bins=20,alpha=0.5,edgecolor='black',label='Year = %s'%year)

ax=plt.gca()
fs=15
ax.set_xlabel('Fractional area of mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,450])
ax.set_title('Histogram of fractional area of mortality for severity =2',fontsize=fs)
ax.legend()
plt.show()

for i in list(reversed(range(start,end+1))):
    area=[]
    severity=[]
    if i<10:       
        year="0%s" %i
    else: 
        year="%s" %i
    data=mort_summary.loc[mort_summary['sev_%s'%year] == 1, 'fam_%s'%year]
    data=data[~np.isnan(data)]
    if data.empty:
        continue
    plt.hist(data,bins=20,alpha=0.5,edgecolor='black',label='Year = %s'%year)

ax=plt.gca()
fs=15
ax.set_xlabel('Fractional area of mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,450])
ax.set_title('Histogram of fractional area of mortality for severity =1',fontsize=fs)
ax.legend()
plt.show()


     
        

#
#mort=np.zeros((end-start+1,7))
#mort2=copy.copy(mort)
#c1=-1
#for thresh in np.arange(0,0.06,0.01):
#    c1=c1+1
#    for i in range(start,end+1):
#
#        sum=0
#        count=0
#        sum2=0
#        count2=0
#        for row in cursor:
#            if row.getValue(field)>=thresh:
#                if row.getValue("SEVERITY1")==sev:                
#                    sum=sum+row.getValue(field)
#                    count=count+1
#                if row.getValue("SEVERITY1")==sev2:                
#                    sum2=sum2+row.getValue(field)
#                    count2=count2+1
#        mort[i-5,0]=2000+i           
#        if count==0:
#            count=1
#        if count2==0:
#            count2=1
#        sum=sum/count
#        sum2=sum2/count2
#        mort[i-5,c1+1]=sum
#        mort2[i-5,c1+1]=sum2
#
#
#arcpy.env.workspace=Dir_mort+'/CA_proc.gdb'
##os.chdir(Dir_fig)
#
#year_range=range(2005,2016)
#date_range=range(1,367,50)
##year_range=2005
##date_range=1
#
#pass_type='A'
#vod=np.zeros((len(year_range)*len(date_range),7))
#def interpolate_gaps(values, limit=None):
#    """
#    Fill gaps using linear interpolation, optionally only fill gaps up to a
#    size of `limit`.
#    """
#    values = np.asarray(values)
#    i = np.arange(values.size)
#    valid = np.isfinite(values)
#    filled = np.interp(i, i[valid], values[valid])
#
#    if limit is not None:
#        invalid = ~valid
#        for n in range(1, limit+1):
#            invalid[:-n] &= invalid[n:]
#        filled[invalid] = np.nan
#
#    return filled
#
#c1=-1
#for k in year_range:
#    for j in date_range: 
#        c1=c1+1
#        year = '%s' %k          #Type the year
#        if j>=100:
#            date='%s'%j
#        elif j >=10:
#            date='0%s' %j
#        else:
#            date='00%s'%j 
#        vod[c1,0]=(k+j/365)
#        fname="stats_%s_%s_%s"%(year,date,pass_type)
#        field = "mean"
#        cursor = arcpy.SearchCursor(fname)
#        c2=0
#        for row in cursor:
#            c2=c2+1
#            vod[c1,c2]=row.getValue(field)/10000    
#vod[ vod==0 ] = np.nan
#
####### plotting
#c1=-1
#for thresh in np.arange(0,0.06,0.01):
#    c1=c1+1    
#    x=mort[:,0]
#    y=mort[:,c1+1]
#    y2=mort2[:,c1+1]
#    fig=plt.figure()
#    fs=15
#    ax=plt.gca()
#    ax.set_xlabel('Year',fontsize=fs)
#    ax.set_ylabel('Fractional area of Mortality',fontsize=fs,color='b')
#    ax.tick_params('y', colors='b')
#    plt.title('Variation of fractional area of Tree Mortality',fontsize=fs)   
#    ax.set_ylim([0,0.15])
#    ax.set_xlim([2005,2015])
#    ax.grid(color='grey', linestyle='-', linewidth=0.5)
#    l1=ax.plot(x,y,'-c',label='Severity=%s'%sev,linewidth=3)
#    l2=ax.plot(x,y2,'-b',label='Severity=%s'%sev2,linewidth=3)
#    l3=ax.axhline(y=thresh,c='black',linestyle='dashed',linewidth=1,label='Threshold')
#    ax.plot(np.nan, '-r', label = 'VOD')
#    x=vod[:,0]
#    y=vod[:,c1+1]
#    ax2 = ax.twinx()
#    ax2.set_xlim([2005,2015])
#    ax2.set_ylabel('VOD',color='r')
#    ax2.tick_params('y', colors='r')
#    ax2.set_ylim([0.5,2.5])
#    y=interpolate_gaps(y,limit=7)
#    l4=ax2.plot(x,y,c='r',label='VOD',linewidth=1)
#    ax.legend()
#    plt.show()