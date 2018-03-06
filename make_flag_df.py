# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
## Makes flag df
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2005,2016)
date_range=range(1,366,1)
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
start=5
end=16
sev=1 #severity index to be recorded
sev2=2
nos=370 # number of grid cells
flagDf=pd.DataFrame(np.arange(1,nos+1),columns=['gridID'])
pass_type = 'A';             #Type the overpass: 'A' or 'D'
for k in year_range:
    for j in date_range:        
        year = '%s' %k          #Type the year
        if j>=100:
            date='%s'%j
        elif j >=10:
            date='0%s' %j
        else:
            date='00%s'%j        
        flag=pd.DataFrame(np.full(nos,np.nan), columns=[year+date+pass_type])
        fname="flag_stats_%s_%s_%s" %(year,date,pass_type)
        if  arcpy.Exists(fname):
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                flag.iloc[row.getValue('gridID')-1]=row.getValue('MEAN')
            flagDf=pd.concat([flagDf,flag],axis=1)
store = pd.HDFStore(Dir_CA+'/flagDf3.h5')       
store['flag'] = flagDf         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
