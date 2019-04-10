from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2005,2017)
month_range=range(1,13)
param=dict([('PEVAP', 8), ('EVBS', 36), ('EVCW', 34), ('EVP', 10), ('TRANS', 35)])
param=dict([('EVP',10)])
param=dict([('LAI',39)])
os.chdir(Dir_NLDAS+'/Proc')
arcpy.env.workspace=Dir_NLDAS+'/Proc'   
inZoneData = Dir_mort+'/'+'CA_proc.gdb'+'/'+"CA_grid"
zoneField = "gridID"
for p in param:
    arcpy.CheckOutExtension("Spatial")         
    for k in year_range:
        year = '%s' %k          #Type the year 
        print('Processing '+p+' data for year '+year+' ...')
        for j in month_range:
            month='%02d'%j
            fname=Dir_NLDAS+'/Proc/'+'c_m_cl_p_'+p+'_'+year+'_'+month+'.tif'
            if  os.path.isfile(fname):
                inRaster=fname
                outTable = Dir_NLDAS+'/Proc/CWD.gdb/'+p+'_stats_'+year+'_'+month
                outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
    arcpy.CheckInExtension('Spatial')
