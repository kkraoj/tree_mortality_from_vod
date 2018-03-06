# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
   
# section for making Df
from __future__ import division
from dirs import*
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
day_range=range(1,362,2)
scale = 1e3;          #Type the multiplier associated with the factor
offset = -33
nos=5938
arcpy.env.workspace=dir+'Proc/sigma0.gdb/'
Df=pd.DataFrame()
for k in year_range:    
    year = '%s' %k          #Type the year 
    print('Processing data for year '+year+' ...') 
    year='%s'%k
    for j in day_range:
        day='%03d'%j 
        data=pd.DataFrame(np.full(nos,np.NaN), columns=[year+day])
        fname='stats_'+year+day
        if  arcpy.Exists(fname):
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                data.iloc[row.getValue('gridID')-1]=row.getValue('MEAN')      
            Df=pd.concat([Df,data],axis=1)            
Df.index.name='gridID'

#            
Df=-0.5*np.log(10**((Df/scale+offset)/10)) 
store = pd.HDFStore(Dir_CA+'/sigma0.h5')        
store['sigma0'] = Df 
store.close()          
            
    
            
            
#sameple commands           
            
