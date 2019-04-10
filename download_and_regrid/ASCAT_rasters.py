# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
"""
shift raster
project raster
clip raster
build attr table
"""
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
day_range=range(1,362,2)
shiftby=1150*4450
dir=MyDir+'/ASCAT/sigma-0/'
for k in year_range:    
    year = '%s' %k          #Type the year 
    print('Processing data for year '+year+' ...')
    arcpy.env.workspace=dir+year
    os.chdir(dir+year)
    for j in day_range:
        day='%03d'%j
        inRaster='sigma0_'+year+day+'.tif'
        if os.path.isfile(inRaster): 
            arcpy.Shift_management(inRaster, 's_'+inRaster, "0","%s"%shiftby)
            inRaster='s_'+inRaster
            arcpy.ProjectRaster_management(inRaster, 'p_'+inRaster, 4326) ## project
            inRaster='p_'+inRaster
            arcpy.Clip_management(inRaster,'#','cl_'+inRaster,Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
            inRaster='cl_'+inRaster
            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
