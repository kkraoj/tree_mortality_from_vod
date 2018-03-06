# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""




## Makes mort df from mort_summary df

from __future__ import division
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2005,2016)
date_range=range(1,367,1)
factor=1e4
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
nos=370 # number of grid cells
vodDf=pd.DataFrame()
vodDf.index.name='gridID'
pass_type = 'D';             #Type the overpass: 'A' or 'D'
for k in year_range:
    year = '%s' %k          #Type the year
    for j in date_range:        
        date='%03d'%j      
        sys.stdout.write('\r'+'Processing data for '+year+' '+date+' ...')
        sys.stdout.flush()
        vod=pd.DataFrame(np.full(nos,np.nan), columns=[year+date+pass_type])
        fname="VOD_stats_%s_%s_%s" %(year,date,pass_type)
        if  arcpy.Exists(fname):
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                vod.iloc[row.getValue('gridID')-1]=row.getValue('MEAN')/factor# grid ID is 1 to 370
            vodDf=pd.concat([vodDf,vod],axis=1)
store = pd.HDFStore(Dir_CA+'/vod_D_Df.h5')          
store['vod_D_Df'] = vodDf
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
