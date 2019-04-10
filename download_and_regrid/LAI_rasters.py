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
year_range=range(2005,2017)
day_range=range(1,362,8)
dir=MyDir+'/LAI/'
arcpy.env.workspace=dir
os.chdir(dir)
for k in year_range:    
    year = '%s'%k          #Type the year 
    for j in day_range:
        day='%03d'%j
        sys.stdout.write('\r'+'Processing data for '+year+' '+day+' ...')
        sys.stdout.flush()
        inRaster1='MOD15A2.A'+year+day+'.h08v04.tsf.sat.hdf'
        inRaster2='MOD15A2.A'+year+day+'.h08v05.tsf.sat.hdf'
        inRaster3='MOD15A2.A'+year+day+'.h09v04.tsf.sat.hdf'
        if os.path.isfile(inRaster1)\
        & os.path.isfile(inRaster2)\
        & os.path.isfile(inRaster3): 
            outRaster='LAI'+year+day+'.tif'
            if not(arcpy.Exists(outRaster)):
                arcpy.MosaicToNewRaster_management([inRaster1,inRaster2,inRaster3],\
                                               dir, outRaster,number_of_bands= 1)
                inRaster=outRaster
                arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
