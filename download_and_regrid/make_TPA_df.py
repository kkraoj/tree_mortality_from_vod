# Name: Dissolve_Example2.py
# Description: Dissolve features based on common attributes

 
# Import system modules
from __future__ import division
from dirs import *
year_range=range(2005,2017)
arcpy.env.overwriteOutput=True
grid='grid'
#grid='smallgrid'
arcpy.CheckOutExtension('Spatial')

for y in year_range:
    year=str(y)[-2:]
    arcpy.env.workspace = Dir_mort+"/ADS20%s.gdb" %year
    inFeature = ["ADS%s" %year,Dir_mort+"/CA_proc.gdb/"+grid]
    outFeature = Dir_mort+"/"+grid+".gdb/ADS"+year+"_i"
    arcpy.Intersect_analysis(inFeature, outFeature,join_attributes="NO_FID")
    ###########################
    targetFeature=outFeature
    outFeature=outFeature+'_j'
    joinFeature=inFeature[1]     
    matchoption='HAVE_THEIR_CENTER_IN'
    joinoperation='JOIN_ONE_TO_MANY'
    arcpy.SpatialJoin_analysis(targetFeature, joinFeature, outFeature, joinoperation, "#", '#',matchoption)
arcpy.CheckInExtension('Spatial')

#----------------------------------------------------------make Df 
m2_per_acre=4046.86
grid='grid'
nos=370
grid='smallgrid'
nos=5938
col_names=range(nos)
arcpy.env.workspace=Dir_mort+"/"+grid+".gdb/"
Df=pd.DataFrame()
for k in year_range:    
    year = '%s' %k          #Type the year 
    sys.stdout.write('\r'+'Processing data for '+year+' ...')
    sys.stdout.flush()
    data=pd.DataFrame([np.full(nos,0)], \
                       index=[pd.to_datetime(year,format='%Y')],\
                             columns=col_names)
    fname=Dir_mort+"/"+grid+".gdb/ADS"+year[-2:]+"_i_j"
    if  arcpy.Exists(fname):
        cursor = arcpy.SearchCursor(fname)
        for row in cursor:
            gridID=row.getValue('gridID')
            TPA = row.getValue('TPA1')
            if TPA < 0: # overwrite no TPA value by average TPA values
                TPA = 5
            no_trees=TPA*row.getValue('Shape_Area')/m2_per_acre
            data.iloc[:,gridID-1]=data.iloc[:,gridID-1]+int(no_trees)            
        Df=pd.concat([Df,data])            
Df=Df.rename_axis('gridID',axis='columns')
Df=Df.astype(int)

#------------------------------------------------store data
os.chdir(Dir_CA)
store_major=pd.HDFStore('data.h5')
store_major['no_trees_005_grid']=Df
store_major.close()


## adjusting to TPA from number of trees
#store_major=pd.HDFStore('data.h5')
#area=4491**2/m2_per_acre
#area=22398**2/m2_per_acre
#Df=store_major['no_trees_025_grid']/area
#store_major['TPA_025_grid']=Df
#store_major.close()             




