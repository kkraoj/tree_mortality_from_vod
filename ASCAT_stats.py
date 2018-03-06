from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
day_range=range(1,362,2)
dir=MyDir+'/ASCAT/sigma-0/'  
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"smallgrid"
zoneField = "gridID"   
for k in year_range:
    arcpy.CheckOutExtension("Spatial")     
    year = '%s' %k          #Type the year 
    arcpy.env.workspace=dir+year
    os.chdir(dir+year)
    print('Processing data for year '+year+' ...')
    for j in day_range:
        day='%03d'%j
        fname='cl_p_s_sigma0_'+year+day+'.tif'
        if  os.path.isfile(fname):
            inRaster=fname
            outTable = dir+'Proc/sigma0.gdb/'+'stats_'+year+day
            if not(arcpy.Exists(outTable)):
                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
    arcpy.CheckInExtension('Spatial')
