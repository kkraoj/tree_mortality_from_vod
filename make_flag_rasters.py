# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017
@author: kkrao
"""
from dirs import *
arcpy.env.overwriteOutput=True


year_range=range(2005,2016)
date_range=range(1,366,1)
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'flag';           #Type the parameter
factor = 1e-0;          #Type the multiplier associated with the factor

fid = open(MyDir+'/anci/MLLATLSB','rb');
late= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
fid = open(MyDir+'/anci/MLLONLSB','rb');
lone= np.fromfile(fid,dtype=np.int32).reshape((586,1383))
fid.close()
late = [i*1e-5 for i in late];
lone = [i*1e-5 for i in lone]
latcorners=[32,42.5]
loncorners=[-125,-113.5]
nrows=int(np.shape(late)[0])
ncols=int(np.shape(late)[1])
y=np.asarray(late)
y=y.flatten()
x=np.asarray(lone)
x=x.flatten()
grid_x, grid_y = np.mgrid[90:-90.25:-0.25, -180:180.25:0.25]
xmin,ymin,xmax,ymax = [np.min(grid_y),np.min(grid_x),np.max(grid_y),np.max(grid_x)]                
for k in year_range:
    for j in date_range:        
        year = '%s' %k          #Type the year
        if j>=100:
            date='%s'%j
        elif j >=10:
            date='0%s' %j
        else:
            date='00%s'%j        
        fname=MyDir+'/'+param+'/'+'/AMSRU_Mland_'+year+date+pass_type+'.'+param
        if  os.path.isfile(fname):
            fid = open(fname,'rb');
            data=np.fromfile(fid,dtype='byte')
            fid.close()            
#            for i in list(range(len(data))):
#                if data[i]<0.0:
#                    data[i]=np.nan                     
#            data = [i*factor for i in data]; 
#            data = -np.log(data)            
            from mkgrid_global import mkgrid_global              
            datagrid = mkgrid_global(data) 
#            datagrid=datagrid.astype(int)                                          
            datagridm = ma.masked_invalid(datagrid)
            from scipy.interpolate import griddata
            z=np.asarray(datagridm)
            z=z.flatten()                                
            grid_z = griddata((y,x), z, (grid_y, grid_x), method='linear')
            grid_z = ma.masked_invalid(grid_z)
            nrows,ncols = np.shape(grid_z)
            nrows=np.int(nrows)
            ncols=np.int(ncols)            
            xres = (xmax-xmin)/(ncols-1)
            yres = (ymax-ymin)/(nrows-1)
            geotransform=(xmin,xres,0,ymax,0, -yres)   
            arcpy.env.workspace=Dir_fig
            os.chdir(Dir_fig)
            output_raster = gdal.GetDriverByName('GTiff').Create('flag_%s_%s_%s.tif' %(year,date,pass_type),ncols, nrows, 1 ,gdal.GDT_Float32,)  # Open the file
            output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
            srs = osr.SpatialReference()                 # Establish its coordinate encoding
            srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                         # Anyone know how to specify the 
                                                         # IAU2000:49900 Mars encoding?
            output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system 
                                                               # to the file
            output_raster.GetRasterBand(1).WriteArray(grid_z)   # Writes my array to the raster
            output_raster.FlushCache()
            output_raster = None
            arcpy.Clip_management('flag_%s_%s_%s.tif' %(year,date,pass_type), "#",'flag_%s_%s_%s_clip.tif' %(year,date,pass_type),Dir_CA+'/'+"CA.shp", "0", "ClippingGeometry")
#            ##mapping algebra * 1
            inRaster = 'flag_%s_%s_%s_clip.tif'%(year,date,pass_type)
            arcpy.CheckOutExtension("Spatial")
            outRaster = Raster(inRaster)*1
            outRaster.save(Dir_fig+'/'+'flag_%s_%s_%s_clip_map.tif'%(year,date,pass_type))
            #copy raster
            inRaster=outRaster
            arcpy.CopyRaster_management(inRaster, 'flag_%s_%s_%s_clip_map_copy.tif'%(year,date,pass_type), pixel_type='4_BIT',nodata_value='0')
            ##make raster table
            inRaster='flag_%s_%s_%s_clip_map_copy.tif'%(year,date,pass_type)
            arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")
            ### zonal stats
#            inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"CA_grid"
#            zoneField = "gridID"
#            outTable = Dir_mort+'/'+'CA_proc.gdb'+'/'+"stats_flag_%s_%s_%s" %(year,date,pass_type)
#            outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
arcpy.CheckInExtension("Spatial")  
            
            
            
            
            
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
