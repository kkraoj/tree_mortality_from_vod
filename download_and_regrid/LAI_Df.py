# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
   
# section for making Df
from __future__ import division
from dirs import*
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
day_range=range(1,362,8)
scale = 0.1;          #Type the multiplier associated with the factor
nos=5938
#nos=370
col_names=range(nos)
arcpy.env.workspace=Dir_mort+'/'+'CA_proc.gdb'
Df=pd.DataFrame()
for k in year_range:    
    year = '%s' %k          #Type the year 
    for j in day_range:
        day='%03d'%j 
        sys.stdout.write('\r'+'Processing data for '+year+' '+day+' ...')
        sys.stdout.flush()
        data=pd.DataFrame([np.full(nos,np.NaN)], \
                           index=[pd.to_datetime(year+day,format='%Y%j')],\
                                 columns=col_names)
        fname='LAI_stats_smallgrid'+year+day
        if  arcpy.Exists(fname):
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                data.iloc[:,row.getValue('gridID')-1]=row.getValue('MEAN')*scale      
            Df=pd.concat([Df,data])            
Df=Df.rename_axis('gridID',axis='columns')

store = pd.HDFStore(Dir_CA+'/LAI.h5')        
store['LAI_smallgrid'] = Df 
store.close()          
        
            
#np.max(Df.loc[:,:].values)