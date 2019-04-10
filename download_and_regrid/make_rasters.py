
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from __future__ import division

import os
import sys
import arcpy
import time
import numpy as np
import pandas as pd
from dirs import MyDir, Dir_fig, Dir_CA, Dir_mort
from scipy.interpolate import griddata
from mkgrid_global import mkgrid_global   
from arcpy.sa import Raster, ZonalStatisticsAsTable      
from osgeo import osr, gdal

start_time = time.clock()
arcpy.env.overwriteOutput=True

year_range=range(2015,2016)
date_range=range(1,367,1)

pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter
map_factor = 1e4

fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
y=np.asarray(late)*1e-5
y=y.flatten()
x=np.asarray(lone)*1e-5
x=x.flatten()
grid_y, grid_x = np.mgrid[90:-90.25:-0.25, -180:180.25:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_x),np.min(grid_y),np.max(grid_x),np.max(grid_y)]    
arcpy.env.workspace=Dir_fig
os.chdir(Dir_fig) 
arcpy.CheckOutExtension("Spatial")
for year in year_range:
    for date in date_range:               
        fname=MyDir+'/%s/%s_LPDR2/AMSRU_Mland_%s%03d%s.%s'%(param,year,year,date,pass_type,param)
        if  os.path.isfile(fname):
            sys.stdout.write('\r'+'Processing data for year:%s, day:%03d ...'%(year,date))
            sys.stdout.flush()   
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()            
#            for i in list(range(len(data))):
#                if data[i]<0.0:
#                    data[i]=np.nan                     
#            data = [i*factor for i in data]; 
            data[data<0.0] = np.nan
            data = -np.log(data)            
            datagrid = mkgrid_global(data) 
#            datagrid=datagrid.astype(int)                                          
            datagridm = np.ma.masked_invalid(datagrid)
            
            z=np.asarray(datagridm)
            z=z.flatten()                                
            grid_z = griddata((x,y), z, (grid_x, grid_y), method='linear')
            grid_z = np.ma.masked_invalid(grid_z)
            
            nrows,ncols = np.shape(grid_z)
            nrows=np.int(nrows)
            ncols=np.int(ncols)            
            xres = (xmax-xmin)/(ncols-1)
            yres = (ymax-ymin)/(nrows-1)
            geotransform=(xmin,xres,0,ymax,0, -yres) 

            output_raster = gdal.GetDriverByName('GTiff').Create('VOD_LPDR2_%s_%03d_%s.tif' %(year,date,pass_type),ncols, nrows, 1 ,gdal.GDT_Float32,)  # Open the file
            output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
            srs = osr.SpatialReference()                 # Establish its coordinate encoding
            srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
            output_raster.GetRasterBand(1).WriteArray(grid_z)   # Writes my array to the raster
            output_raster.FlushCache()
            output_raster = None
            
            ## clipping            
            arcpy.Clip_management('VOD_LPDR2_%s_%03d_%s.tif' %(year,date,pass_type), \
                                  "#",'VOD_LPDR2_%s_%03d_%s_clip.tif' %(year,date,pass_type),\
                                   Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
#            ##mapping algebra * 1
            inRaster = 'VOD_LPDR2_%s_%03d_%s_clip.tif'%(year,date,pass_type)
            
            outRaster = Raster(inRaster)*map_factor
            outRaster.save(Dir_fig+'/'+'VOD_LPDR2_%s_%03d_%s_clip_map.tif'%(year,date,pass_type))
            #copy raster
            inRaster=outRaster
            
            arcpy.CopyRaster_management(inRaster, 'VOD_LPDR2_%s_%03d_%s_clip_map_copy.tif'%(year,date,pass_type),\
                                        pixel_type='16_BIT_UNSIGNED',nodata_value='0')
            ##make raster table
            inRaster='VOD_LPDR2_%s_%03d_%s_clip_map_copy.tif'%(year,date,pass_type)
            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")
arcpy.CheckInExtension("Spatial")  
end_time=time.clock()
time_taken=end_time-start_time
print('Program executed in %s seconds'%time_taken)
           