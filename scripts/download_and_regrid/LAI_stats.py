from dirs import *
arcpy.env.overwriteOutput=True
dir=MyDir+'/LAI/'
os.chdir(dir)
arcpy.env.workspace=dir
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"grid"
zoneField = "gridID"
year_range=range(2015,2017)
#year_range=[2007,2010,2013]
day_range=range(1,362,8)
#date_range=range(286,366,1)
for k in year_range:
    arcpy.CheckOutExtension('Spatial')
    year = '%s' %k          #Type the year 
    for j in day_range:        
        day='%03d'%j
        sys.stdout.write('\r'+'Processing data for '+year+' '+day+' ...')
        sys.stdout.flush()
        fname='LAI'+year+day+'.tif'
        if  os.path.isfile(fname):
            inRaster=fname
            outTable = Dir_mort+'/'+'CA_proc.gdb'+'/LAI_stats_grid'+year+day
            if not(arcpy.Exists(outTable)):
                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
    arcpy.CheckInExtension('Spatial')
