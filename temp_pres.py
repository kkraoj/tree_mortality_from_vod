# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 00:17:08 2017

@author: kkrao
"""
from __future__ import division
import arcpy
import os
import glob
import re
from dirs import MyDir, Dir_CA,Dir_mort
import pandas as pd
import numpy as np
from arcpy.sa import ZonalStatisticsAsTable, Raster
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
month_range=range(1,13)
###----------------------------------------------------------------------------
arcpy.CheckOutExtension('Spatial')
for field in ['ppt','tmax','tmean']:
    os.chdir(MyDir+'/PRISM/%s'%field)
    arcpy.env.workspace=MyDir+'/PRISM/%s'%field
    sys.stdout.write('\r'+'Processing data for %s ...'%field)
    sys.stdout.flush()
    for inRaster in glob.glob("PRISM*.bil"):
        arcpy.CopyRaster_management(inRaster, 'c_'+inRaster, pixel_type='32_BIT_SIGNED')
        ##make raster table
        inRaster='c_'+inRaster
        arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")
    for inRaster in glob.glob("c_PRISM*.bil"):
        if arcpy.Exists('p_'+inRaster):
            continue
        arcpy.ProjectRaster_management(inRaster, 'p_'+inRaster, 4326) ## project
        inRaster='p_'+inRaster
arcpy.CheckInExtension('Spatial')
####---------------------------------------------------------------------------
inZoneData = Dir_CA+"/grid.shp"
zoneField = "gridID"
arcpy.CheckOutExtension('Spatial')
for field in ['ppt','tmax','tmean']:
    os.chdir(MyDir+'/PRISM/%s'%field)
    arcpy.env.workspace=MyDir+'/PRISM/%s'%field
    sys.stdout.write('\r'+'Processing data for %s ...'%field)
    sys.stdout.flush()
    for inRaster in glob.glob("p_c_*.bil"):
        year=re.findall(r"[0-9]{6,6}", inRaster)[0]
        month=int(year[-2:])
        year=int(year[:4])
        outTable=Dir_mort+'/CA_proc.gdb/%s_%d_%d_stats'%(field,year,month)
        outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster,outTable,"DATA","MEAN")
arcpy.CheckInExtension('Spatial')
####---------------------------------------------------------------------------
## make Df
arcpy.env.workspace=MyDir+'/PRISM/'
nos=370
scale=1.0
store = pd.HDFStore(Dir_CA+'/data.h5') 
for field in ['ppt','tmax','tmean']:
    Df=pd.DataFrame()
    for year in year_range:
        for month in month_range:      
            fname=Dir_mort+'/CA_proc.gdb/%s_%d_%d_stats'%(field,year,month)
            if  arcpy.Exists(fname):
                data=pd.DataFrame([np.full(nos,np.NaN)], \
                         columns=range(nos))
                cursor = arcpy.SearchCursor(fname)
                for row in cursor:
                    data.iloc[:,row.getValue('gridID')]=row.getValue('MEAN')*scale
                    data.index=[pd.to_datetime('%d%02d'%(year,month),format='%Y%m')]
            Df=Df.append(data)
    Df.columns.name='gridID'
    Df.index.name=field
    store[Df.index.name] = Df
####---------------------------------------------------------------------------



store.close()     

