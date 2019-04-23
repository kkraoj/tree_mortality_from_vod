import arcpy
import sys
from arcpy.sa import ZonalStatisticsAsTable
import os
arcpy.env.overwriteOutput=True
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'
os.chdir(Dir_fig)
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"grid_subset" # grids with 70% forest only
zoneField = "gridID"
year_range=range(2015,2016)
#year_range=[2007,2010,2013]
date_range=range(1,366)
#date_range=range(286,366,1)
pass_type='A'
for year in year_range:
    arcpy.CheckOutExtension('Spatial')
    for date in date_range:        
        sys.stdout.write('\r'+'Processing data for %s, %03d ...'%(year,date))
        sys.stdout.flush()
        fname=Dir_fig+'/'+'VOD_LPDR2_%s_%03d_%s_clip_map_copy.tif'%(year,date,pass_type)
        if  os.path.isfile(fname):
            inRaster=fname
            outTable = Dir_mort+'/'+'CA_proc.gdb'+'/'+'VOD_LPDR2_stats_%s_%03d_%s' %(year,date,pass_type)
            if not(arcpy.Exists(outTable)):
                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
    arcpy.CheckInExtension('Spatial')
