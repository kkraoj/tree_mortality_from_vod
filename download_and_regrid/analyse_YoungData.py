# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2009,2016)
factor = 1e3;          #Type the multiplier associated with the factor
Dir_Young=MyDir+'/Young' 
arcpy.CheckOutExtension("Spatial")         
for k in year_range:       
    year = '%s' %k          #Type the year 
    print('Processing data for year '+year+' ...')
    inRaster='Young_'+year+'.tif'
    arcpy.env.workspace=Dir_Young        
    outRaster = Raster(inRaster)*factor # mapping
    outRaster.save('m_'+inRaster)
    #copy raster
    inRaster='m_'+inRaster
    arcpy.CopyRaster_management(inRaster, 'c_'+inRaster, pixel_type='32_BIT_SIGNED',nodata_value='0')
    ##make raster table
    inRaster='c_'+inRaster
    arcpy.BuildRasterAttributeTable_management(inRaster, "Overwrite")   
arcpy.CheckInExtension("Spatial")  


#####s stats table

os.chdir(Dir_Young+'/Proc')
arcpy.env.workspace=Dir_Young+'/Proc'
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"CA_grid"
zoneField = "gridID"
arcpy.CheckOutExtension("Spatial")         
for k in year_range:
    year = '%s' %k          #Type the year 
    print('Processing '+p+' data for year '+year+' ...')
    fname=Dir_Young+'/Proc'+'/c_m_Young_'+year+'.tif'
    if  os.path.isfile(fname):
        inRaster=fname
        outTable = Dir_Young+'/Young.gdb/'+'stats_'+year
        outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
arcpy.CheckInExtension('Spatial')


#### make dataframe
arcpy.env.workspace = Dir_Young+'/Young.gdb'
nos=370 # number of grid cells
store = pd.HDFStore(Dir_CA+'/Young_Df.h5')       
Df=pd.DataFrame(np.arange(1,nos+1),columns=['gridID'])  
for k in year_range:
    year = '%s' %k          #Type the year 
    print('Processing data for year '+year+' ...')    
    data=pd.DataFrame(np.full(nos,np.nan), columns=[year])
    fname='stats_'+year
    if arcpy.Exists(fname):
        cursor = arcpy.SearchCursor(fname)
        for row in cursor:
            data.iloc[row.getValue('gridID')-1]=row.getValue('MEAN')/factor
        Df=pd.concat([Df,data],axis=1)  
store['cwd_acc'] = Df
store.close()



            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
