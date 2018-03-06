# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
month_range=range(1,13)
factor = 1e3;          #Type the multiplier associated with the factor
nodata=1e30
param=dict([('PEVAP', 8), ('EVBS', 36), ('EVCW', 34), ('EVP', 10), ('TRANS', 35)])
param=dict([('EVP',10)])
param=dict([('LAI',39)])
for p in param:
    if p=='PEVAP':
        model='FORA'
    else:
        model='VIC'   
    arcpy.CheckOutExtension("Spatial")         
    for k in year_range:       
        year = '%s' %k          #Type the year 
        print('Processing '+p+' data for year '+year+' ...')
        for j in month_range:
            arcpy.env.workspace=Dir_NLDAS+'/'+year
            month='%02d'%j
            inRaster=p+'_'+year+'_'+month+'.tif'
            arcpy.ExtractSubDataset_management('NLDAS_'+model+'0125_M.A'+year+month+'.002.grb',Dir_NLDAS+'/Proc/'+inRaster, '%d'%param[p])
            arcpy.env.workspace=Dir_NLDAS+'/Proc'        
            arcpy.ProjectRaster_management(inRaster, 'p_'+inRaster, 4326) ## project
            inRaster='p_'+inRaster
            arcpy.Clip_management(inRaster,'#','cl_'+inRaster,Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
            inRaster='cl_'+inRaster
            outRaster = Raster(inRaster)*factor # mapping
            outRaster.save('m_'+inRaster)
            #copy raster
            inRaster='m_'+inRaster
            arcpy.CopyRaster_management(inRaster, 'c_'+inRaster, pixel_type='32_BIT_SIGNED',nodata_value='0')
            ##make raster table
            inRaster='c_'+inRaster
            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")   
    arcpy.CheckInExtension("Spatial")  



            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
