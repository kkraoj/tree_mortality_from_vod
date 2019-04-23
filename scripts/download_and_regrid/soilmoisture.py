# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from __future__ import division
import arcpy
import os
import sys
import numpy as np
import pandas as pd
import numpy.ma as ma
from osgeo import gdal
from osgeo import osr
from dirs import MyDir, Dir_CA,Dir_mort
from mkgrid_global import mkgrid_global 
from scipy.interpolate import griddata
from arcpy.sa import Raster, ZonalStatisticsAsTable

arcpy.env.overwriteOutput=True
year_range=range(2005,2016)
date_range=range(1,366,1)
scale_factor=1e4
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'vsm';           #Type the parameter
factor = 1e-0;          #Type the multiplier associated with the factor
fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone]
nrows=int(np.shape(late)[0])
ncols=int(np.shape(late)[1])
y=np.asarray(lone)
y=y.flatten()
x=np.asarray(late)
x=x.flatten()
grid_x, grid_y = np.mgrid[90:-90.25:-0.25, -180:180.25:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_y),np.min(grid_x),np.max(grid_y),np.max(grid_x)]
srs = osr.SpatialReference()                 # Establish its coordinate encoding
srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
xres,yres=0.25,0.25
nrows,ncols=721,1441
geotransform=(xmin,xres,0,ymax,0, -yres) 
#for year in year_range:
#    arcpy.CheckOutExtension("Spatial") 
#    arcpy.env.workspace=MyDir+'/'+param+'/%s'%year
#    os.chdir(MyDir+'/'+param+'/%s'%year)
#    for date in date_range: 
#        sys.stdout.write('\r'+'Processing data for %s %03d...'%(year,date))
#        sys.stdout.flush()
#        fname='AMSRU_Mland_%s%03d%s.%s'%(year,date,pass_type,param)
#        if  os.path.isfile(fname):
#            fid = open(fname,'rb');
#            data=np.fromfile(fid)
#            fid.close()            
#            for i in list(range(len(data))):
#                if data[i]<=0.0:
#                    data[i]=np.nan                     
#            data = [i*factor for i in data]; 
#            datagrid = mkgrid_global(data)                                           
#            datagridm = ma.masked_invalid(datagrid)
#            z=np.asarray(datagridm)
#            z=z.flatten()                                
#            grid_z = griddata((y,x), z, (grid_y, grid_x), method='linear')
#            grid_z = ma.masked_invalid(grid_z)           
#            outRaster='vsm_%s_%03d_%s.tif' %(year,date,pass_type)
#            output_raster = gdal.GetDriverByName('GTiff').Create(outRaster,ncols, nrows, 1 ,gdal.GDT_Float32,)  # Open the file
#            output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
#            output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
#            output_raster.GetRasterBand(1).WriteArray(grid_z)   # Writes my array to the raster
#            output_raster.FlushCache()
#            output_raster = None
#            inRaster=outRaster
#            outRaster='cl_'+inRaster
#            arcpy.Clip_management(inRaster,'#',outRaster,Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
#            inRaster=outRaster
#            outRaster = Raster(inRaster)*scale_factor # mapping
#            outRaster.save('m_'+inRaster)
#            inRaster='m_'+inRaster
#            outRaster='c_'+inRaster
#            arcpy.CopyRaster_management(inRaster,outRaster, pixel_type='32_BIT_SIGNED',nodata_value='0')
#            inRaster=outRaster
#            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")
#    arcpy.CheckInExtension("Spatial")              
###----------------------------------------------------------------------------

#inZoneData = Dir_CA+"/grid.shp"
#zoneField = "gridID"
#for year in year_range:
#    arcpy.CheckOutExtension("Spatial") 
#    arcpy.env.workspace=MyDir+'/'+param+'/%s'%year
#    os.chdir(MyDir+'/'+param+'/%s'%year)
#    for date in date_range: 
#        sys.stdout.write('\r'+'Processing data for %s %03d...'%(year,date))
#        sys.stdout.flush()
#        inRaster='c_m_cl_%s_%04d_%03d_%s.tif'%(param,year,date,pass_type)
#        if os.path.isfile(inRaster):
#            outTable=Dir_mort+'/CA_proc.gdb/%s_%04d_%03d_%s_stats'%(param,year,date,pass_type)
#            if not(arcpy.Exists(outTable)):
#                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster,outTable,"DATA","MEAN")
#    arcpy.CheckInExtension('Spatial')            
#print('\nProcessing complete')

#####---------------------------------------------------------------------------
### make Df
arcpy.env.workspace=MyDir+'/PRISM/'
nos=370
store = pd.HDFStore(Dir_CA+'/data.h5') 
Df=pd.DataFrame()
for year in year_range:
    for date in date_range:      
        fname=Dir_mort+'/CA_proc.gdb/%s_%04d_%03d_%s_stats'%(param,year,date,pass_type)
        sys.stdout.write('\r'+'Processing data for %04d %03d ...'%(year,date))
        sys.stdout.flush()
        if  arcpy.Exists(fname):
            data=pd.DataFrame([np.full(nos,np.NaN)], \
                     columns=range(nos))
            cursor = arcpy.SearchCursor(fname)
            for row in cursor:
                data.iloc[:,row.getValue('gridID')]=row.getValue('MEAN')/scale_factor
            data.index=[pd.to_datetime('%04d%03d'%(year,date),format='%Y%j')]
            Df=Df.append(data)
Df.columns.name='gridID'
Df=Df[Df.index.dayofyear!=366] 
Df.index.name=param
store[Df.index.name] = Df

            
            
            
#for year in year_range:
#    for date in date_range:
#        gg=pd.to_datetime('%04d%03d'%(year,date),format='%Y%j')                         
#        print(gg)    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
