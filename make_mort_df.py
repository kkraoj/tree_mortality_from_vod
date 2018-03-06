# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""




## Makes mort df from mort_summary df

from __future__ import division
from IPython import get_ipython
get_ipython().magic('reset -sf') 

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
from arcpy.sa import *
import h5py


arcpy.env.overwriteOutput=True

MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'
year_range=range(2005,2016)
date_range=range(1,367,1)

arcpy.env.workspace = Dir_mort+'/Mortality_intersect.gdb'
# arcpy.Statistics_analysis("futrds", "C:/output/output.gdb/stats", [["Shape_Length", "SUM"]], "NM")
start=5
end=16


sev=1 #severity index to be recorded
sev2=2
mort_summary=pd.DataFrame()
nos=370 # number of grid cells

#store['mort_summary'] = mort_summary
store = pd.HDFStore(Dir_CA+'/mort_summary.h5')     
mort_summary = store['mort_summary']


mort=pd.DataFrame(np.arange(1,nos+1),columns=['gridID'])
mort1=pd.DataFrame(np.arange(1,nos+1),columns=['gridID'])
mort2=pd.DataFrame(np.arange(1,nos+1),columns=['gridID'])
for i in range(start,end+1):
    if i<10:       
        year="0%s" %i
    else: 
        year="%s" %i
    fam1=pd.DataFrame(np.full(nos,np.nan), columns=['fam_20'+year+'_1'])
    fam2=pd.DataFrame(np.full(nos,np.nan), columns=['fam_20'+year+'_2'])
    fam=pd.DataFrame(np.full(nos,np.nan), columns=['fam_20'+year])
    colname='gridID_'+year
    mort_summary['gridID_'+year]
    for j in range(len(mort_summary[colname])):
        if np.isnan(mort_summary.at[j,'gridID_'+year]):
            break
        else:            
            if mort_summary.at[j,'sev_'+year]==1:
                fam1.iloc[mort_summary.at[j,'gridID_'+year].astype(int)-1]=mort_summary.at[j,'fam_'+year]
            elif mort_summary.at[j,'sev_'+year]==2:
                fam2.iloc[mort_summary.at[j,'gridID_'+year].astype(int)-1]=mort_summary.at[j,'fam_'+year]
            else:
                fam.iloc[mort_summary.at[j,'gridID_'+year].astype(int)-1]=mort_summary.at[j,'fam_'+year]
    mort=pd.concat([mort, fam],axis=1)
    mort1=pd.concat([mort1, fam1],axis=1)
    mort2=pd.concat([mort2, fam2],axis=1)
    
    
                

store = pd.HDFStore(Dir_CA+'/mort.h5')        
store['mort'] = mort 

#store = pd.HDFStore(Dir_CA+'/mort1.h5')        
store['mort1'] = mort1

#store = pd.HDFStore(Dir_CA+'/mort2.h5')        
store['mort2'] = mort2      
            
# data=mort_summary.loc[mort_summary['sev_%s'%year] == 2, 'fam_%s'%year]            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
