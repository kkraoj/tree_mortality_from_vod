from arcpy.sa import*
import os
arcpy.env.overwriteOutput=True
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'
os.chdir(Dir_fig)
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"CA_grid"
zoneField = "gridID"
year_range=range(2005,2016)
#year_range=[2007,2010,2013]
date_range=range(1,366)
#date_range=range(286,366,1)
pass_type='D'
for k in year_range:
    arcpy.CheckOutExtension('Spatial')
    year = '%s' %k          #Type the year 
    for j in date_range:        
        date='%03d'%j
        sys.stdout.write('\r'+'Processing data for '+year+' '+date+' ...')
        sys.stdout.flush()
        fname=Dir_fig+'/'+'VOD_%s_%s_%s_clip_map_copy.tif'%(year,date,pass_type)
        if  os.path.isfile(fname):
            inRaster=fname
            outTable = Dir_mort+'/'+'CA_proc.gdb'+'/'+'VOD_stats_%s_%s_%s' %(year,date,pass_type)
            if not(arcpy.Exists(outTable)):
                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
    arcpy.CheckInExtension('Spatial')
