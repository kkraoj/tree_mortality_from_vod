# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 00:17:08 2017

@author: kkrao
"""
from __future__ import division
import arcpy
import os
import sys
import glob
import re
import gc
from dirs import MyDir, Dir_CA,Dir_mort
import pandas as pd
import numpy as np
from arcpy.sa import ZonalStatisticsAsTable, Raster
#gc.enable()
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
month_range=range(1,13)
day_range=range(1,32)
####----------------------------------------------------------------------------
#arcpy.CheckOutExtension('Spatial')
#for field in ['vpdmax']:
#    os.chdir(MyDir+'/PRISM/%s'%field)
#    arcpy.env.workspace=MyDir+'/PRISM/%s'%field
#    for inRaster in glob.glob("PRISM*.bil"):
#        sys.stdout.write('\r'+'Processing data for %s ...'%inRaster)
#        sys.stdout.flush()
#        outRaster='c_'+inRaster
#        arcpy.CopyRaster_management(inRaster,outRaster , pixel_type='32_BIT_SIGNED')
#        ##make raster table
#        inRaster=outRaster
#        arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")
#        outRaster= 'p_'+inRaster
#        arcpy.ProjectRaster_management(inRaster,outRaster, 4326) ## project
#        inRaster=outRaster
#print('\nProcessing complete')
#arcpy.CheckInExtension('Spatial')
####---------------------------------------------------------------------------
#inZoneData = Dir_CA+"/grid.shp"
#zoneField = "gridID"
#arcpy.CheckOutExtension('Spatial')
#counter=0
#field ='vpdmax'
#os.chdir(MyDir+'/PRISM/%s_1'%field)
#arcpy.env.workspace=MyDir+'/PRISM/%s_1'%field
#for inRaster in glob.glob("p_c_*.bil"):
#    sys.stdout.write('\r'+'Processing data for %s ...'%inRaster)
#    sys.stdout.flush()
#    year=re.findall(r"[0-9]{8,8}", inRaster)[0]
#    day=int(year[-2:])
#    month=int(year[-4:-2])
#    year=int(year[:4])
#    outTable=Dir_mort+'/CA_proc.gdb/%s_%04d_%02d_%02d_stats'%(field,year,month,day)
#    if not(arcpy.Exists(outTable)):
#        outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster,outTable,"DATA","MEAN")
#print('\nProcessing complete')
#arcpy.CheckInExtension('Spatial')
#####---------------------------------------------------------------------------
# make Df
arcpy.env.workspace=MyDir+'/PRISM/'
nos=370
scale=1.0
store = pd.HDFStore(Dir_CA+'/data.h5') 
for field in ['vpdmax']:
    Df=pd.DataFrame()
    for year in year_range:
        for month in month_range:
            for day in day_range:
                fname=Dir_mort+'/CA_proc.gdb/%s_%04d_%02d_%02d_stats'%(field,year,month,day)                
                if  arcpy.Exists(fname):
                    sys.stdout.write('\r'+'Processing data for %s ...'%fname)
                    sys.stdout.flush()                    
                    data=pd.DataFrame([np.full(nos,np.NaN)], \
                             columns=range(nos))
                    cursor = arcpy.SearchCursor(fname)
                    for row in cursor:
                        data.iloc[:,row.getValue('gridID')]=row.getValue('MEAN')/scale
                    data.index=[pd.to_datetime('%04d%02d%02d'%(year,month,day),format='%Y%m%d')]
                    Df=Df.append(data)
    Df.columns.name='gridID'
    Df.index.name=field
    store[Df.index.name] = Df
#store.close()     
