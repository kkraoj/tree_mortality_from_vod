# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""




## sectin for making attribute tables

from __future__ import division
from dirs import*
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
box_area=5000**2
for k in year_range:
    year='%s'%k
    Y1='%02d'%(k-2000)
    arcpy.env.workspace = Dir_mort+'/ADS'+year+'.gdb'
    inFeature='ADS'+Y1
    outFeature= 'd_'+inFeature
    arcpy.Dissolve_management(inFeature, outFeature)
    joinFeature=Dir_mort+'/CA_proc.gdb/smallgrid'
    inFeature=[outFeature,joinFeature]
    outFeature='i_'+outFeature
    arcpy.Intersect_analysis(inFeature, outFeature)
    inFeature=outFeature
    fieldName='FAM'
    arcpy.AddField_management(inFeature, fieldName, "FLOAT")   
    arcpy.CalculateField_management(inFeature, fieldName, "!Shape_Area!/%s"%box_area,"PYTHON_9.3")
    dropField='gridID'
    arcpy.DeleteField_management(inFeature, dropField)
    outFeature='j_'+inFeature
    targetFeature=inFeature     
    matchoption='HAVE_THEIR_CENTER_IN'
    fieldmappings = arcpy.FieldMappings()
    fieldmappings.addTable(targetFeature)
    fieldmappings.addTable(joinFeature)
    arcpy.SpatialJoin_analysis(targetFeature, joinFeature, outFeature, "#", "#", '#',matchoption)
            
            
            
# section for making Df
from __future__ import division
from dirs import*
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
nos=5938
mortDf=pd.DataFrame()
for k in year_range:
    year='%s'%k
    Y1='%02d'%(k-2000)
    arcpy.env.workspace = Dir_mort+'/ADS'+year+'.gdb'  
    mort=pd.DataFrame(np.full(nos,0), columns=[year])
    fname='j_i_d_ADS'+Y1
    if  arcpy.Exists(fname):
        cursor = arcpy.SearchCursor(fname)
        for row in cursor:
            mort.iloc[row.getValue('gridID')-1]=row.getValue('FAM')
        mortDf=pd.concat([mortDf,mort],axis=1)            
mortDf.index.name='gridID'

            

store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')        
store['mort'] = mortDf 
store.close()          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
