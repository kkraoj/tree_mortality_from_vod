# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
## Makes data df
import arcpy
import os
import pandas as pd
import numpy as np
from dirs import Dir_NLDAS
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
month_range=range(1,13)
param=dict([('PEVAP', 8), ('EVBS', 36), ('EVCW', 34), ('EVP', 10), ('TRANS', 35),('SBSNO',37)])
param=dict([('EVP',10)])
param=dict([('LAI',39)])
factor=1e3
arcpy.env.workspace = Dir_NLDAS+'/Proc/CWD.gdb'
os.chdir(Dir_NLDAS+'/Proc/CWD.gdb')
nos=370 # number of grid cells
store = pd.HDFStore(Dir_CA+'/LAI_Df.h5')    
for p in param:    
    Df=pd.DataFrame() 
    Df.index.name='gridID'
    for k in year_range:
        year = '%s' %k          #Type the year 
        print('Processing '+p+' data for year '+year+' ...')
        for j in month_range:
            month='%02d'%j
            data=pd.DataFrame(np.full(nos,np.NaN), columns=[year+month])
            fname=p+'_stats_'+year+'_'+month
            if  arcpy.Exists(fname):
                cursor = arcpy.SearchCursor(fname)
                for row in cursor:
                    data.iloc[row.getValue('gridID')-1]=row.getValue('MEAN')/factor
                Df=pd.concat([Df,data],axis=1)  
    store[p] = Df
store.close()         
##change units
#store=pd.HDFStore(Dir_CA+'/MOS_Df.h5') 
#store2 = pd.HDFStore(Dir_CA+'/MOSu_Df.h5')
#for p in param:
#    if (p=='EVBS' or p=='EVCW' or p=='TRANS'):
#        Df=store[p]*86400/2.5e6*30
#    elif p=='SBSNO':
#        Df=store[p]*86400/0.334e6*30
#    else:
#        Df=store[p]
#    store2[p] = Df            
#store.close()            
#store2.close()            
            
            
 ##replace Nans by zeros
#store=pd.HDFStore(Dir_CA+'/CWD_Df.h5') 
#store2 = pd.HDFStore(Dir_CA+'/CWDu_Df.h5')
#for p in param:
#    Df=store[p]
#    Df[np.isnan(Df)]=0
#    store[p]=Df
#    Df=store2[p]
#    Df[np.isnan(Df)]=0
#    store2[p]=Df                 
#store.close()            
#store2.close()             
#            
#            
#            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
