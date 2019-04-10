# Name: Dissolve_Example2.py
# Description: Dissolve features based on common attributes

 
# Import system modules
from __future__ import division
import arcpy
import math

arcpy.env.overwriteOutput=True

MyDir = 'D:/Krishna/Project/data/RS_data'  #Type the path to your data
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
Dir_fig='D:/Krishna/Project/figures'
Dir_mort='D:/Krishna/Project/data/Mort_Data/CA_Mortality_Data'

box_area=772768725
dissolveField="SEVERITY1"
# Set local variables
for i in range(5,17):
    if i<10:       
        year="0%s" %i
    else:
        year="%s" %i
    arcpy.env.workspace = Dir_mort+"/ADS20%s.gdb" %year
    inFeature = "ADS%s" %year
    outFeature = "ADS%s_dissolved" %year
    arcpy.Dissolve_management(inFeature, outFeature,dissolveField)
        #################################
    inFeature = [outFeature,Dir_mort+"/CA_proc.gdb/CA_grid"]
    outFeature = Dir_mort+"/Mortality_intersect.gdb/ADS"+year+"_intersect"
    arcpy.Intersect_analysis(inFeature, outFeature,join_attributes="NO_FID")
    ###########################
    inFeature = outFeature
    fieldName = "frac_area_mort"
    fieldPrecision = 4
    fieldScale = 2
    arcpy.AddField_management(inFeature, fieldName, "float",field_precision=fieldPrecision,field_scale=fieldScale)
    expression="math.ceil(!SHAPE.AREA!/%s*10000)/10000" %box_area
    arcpy.CalculateField_management(inFeature, fieldName, expression,"PYTHON_9.3")
    