# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 08:49:45 2018

@author: kkrao
"""

from __future__ import division
import arcpy
import sys
import pandas as pd
import numpy as np
from arcpy.sa import ZonalStatisticsAsTable
from dirs import Dir_mort, MyDir, Dir_CA,build_df_from_arcpy

year_range=range(2005,2018)

#### section 1: make xonal tables of area of ever and deci mortality
###============================================================================

#arcpy.env.overwriteOutput=True

#arcpy.CheckOutExtension('Spatial')
#for k in year_range:
#    sys.stdout.write('\r'+'Processing data for %s ...'%k)
#    sys.stdout.flush()
#    year='%s'%k
#    Y1='%02d'%(k-2000)
#    arcpy.env.workspace = Dir_mort+'/ADS'+year+'.gdb'
#    inFeature='ADS'+Y1
#    outFeature= 'd_GC_'+inFeature
#    arcpy.Dissolve_management(inFeature, outFeature)
##    joinFeature=Dir_mort+'/CA_proc.gdb/smallgrid'
#    joinFeature=Dir_mort+'/grid.gdb/grid_subset_GC'
#    inFeature=[outFeature,joinFeature]
#    outFeature='i_'+outFeature
#    arcpy.Intersect_analysis(inFeature, outFeature)
#    inFeature=outFeature
##    fieldName='FAM'
##    arcpy.AddField_management(inFeature, fieldName, "FLOAT")   
##    arcpy.CalculateField_management(inFeature, fieldName, "!Shape_Area!/%s"%box_area,"PYTHON_9.3")
#    dropField='gridID'
#    arcpy.DeleteField_management(inFeature, dropField)
#    outFeature='j_'+inFeature
#    targetFeature=inFeature     
#    matchoption='HAVE_THEIR_CENTER_IN'
#    fieldmappings = arcpy.FieldMappings()
#    fieldmappings.addTable(targetFeature)
#    fieldmappings.addTable(joinFeature)
#    arcpy.SpatialJoin_analysis(targetFeature, joinFeature, outFeature, "#", "#", '#',matchoption)
#    
#    inZoneData = outFeature
#    zoneField = "gridID"
#    for species in ['ever','deci']:
#        inRaster=MyDir+'/Forest/GLOBCOVER/gc_%s'%species
#        outTable=Dir_mort+'/species.gdb/gc_mort_stats_%s_%s'%(year, species)
#        outZSaT = ZonalStatisticsAsTable(inZoneData, zoneField, inRaster, outTable,"DATA","MEAN")
#
#arcpy.CheckInExtension('Spatial')

### Section 2: read the stats tables and make Dfs
### ===========================================================================
## FAM evergreen = mortality shape area*fraction of evergreen landcover in mortality
### shape/(area of 0.25 degree box * fraction of evergreen in 0.25 box)

box_area=772768725.0 ## area of 0.25 degree grid in square meters
lc=pd.read_excel('D:/Krishna/Project/working_tables.xlsx',\
                 sheetname='gc_ever_deci',index_col=1)  
store=pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
species_full={'ever':'evergreen','deci':'deciduous'}

for species in ['ever','deci']:
    Df=pd.DataFrame(index=[pd.to_datetime(year_range,format='%Y')],\
                       columns=store['mortality_025_grid'].columns)
    Df.fillna(0,inplace=True)
    for year in year_range:
        sys.stdout.write('\r'+'Processing data for %s %s ...'%(species_full[species],year))
        sys.stdout.flush()
        #get area of mortality
        arcpy.env.workspace = Dir_mort+'/ADS%s.gdb'%year
        table='j_i_d_GC_ADS%02d'%(year-2000)
        mort_area=build_df_from_arcpy(table, columns=['gridID', 'Shape_Area'])
        mort_area.index=mort_area.gridID.astype(int)
        mort_area.drop('gridID',axis='columns',inplace=True)
        #get area of mortality for just evergreen or deciduous
        arcpy.env.workspace = Dir_mort+'/species.gdb'
        mort_species_cover=build_df_from_arcpy('gc_mort_stats_%s_%s'%(year, species), columns=['gridID', 'MEAN'])
        mort_species_cover.index=mort_species_cover.gridID.astype(int)
        mort_species_cover.drop('gridID',axis='columns',inplace=True)
        data=pd.DataFrame([mort_area['Shape_Area']*mort_species_cover['MEAN']],\
                          index=[pd.to_datetime(year,format='%Y')])
        Df.loc[Df.index.year==year]+=data
    Df/=lc[species_full[species]] 
    Df/=box_area   
    store['mortality_%s_025_grid_spatial_overlap'%species_full[species]]=Df
        