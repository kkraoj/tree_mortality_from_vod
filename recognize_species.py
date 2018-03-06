# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 20:50:23 2017

@author: kkrao
"""
from __future__ import division
from dirs import*
m2_per_acre=4046.86
grid='grid'
year='2015'
nos=370
area=22398**2/m2_per_acre
table=Dir_mort+"/"+grid+".gdb/ADS"+year[-2:]+"_i_j" 
columns=['gridID','TPA1','Shape_Area','HOST1','FOR_TYPE1']
Df_main=build_df_from_arcpy(table, columns)
Df=Df_main
Df=Df.astype(np.float)
Df.TPA1[(Df.TPA1<0)]=5
no_trees=pd.DataFrame((Df.TPA1*Df.Shape_Area/m2_per_acre),columns=['no_trees'])
Df=pd.concat([Df,no_trees],axis=1)
Df=Df.astype({'gridID':np.int16,\
              'HOST1':np.int16,\
              'FOR_TYPE1':np.int16,\
#              'no_trees':np.int16,\ # flooring drastically underestimates TPA
              })
TPA_Df=Df.groupby('gridID').no_trees.sum()
TPA_Df=TPA_Df/area
TPA_Df.name='TPA'
data=Df.groupby(['gridID','FOR_TYPE1']).no_trees.sum()
max_species=data.groupby(level=0).idxmax()
TPA_Df.index=pd.MultiIndex.from_tuples(max_species,names=('gridID','FOR_TYPE'))
TPA_Df[TPA_Df>4].groupby(level=1).mean().index
    
table=Dir_mort+'/species.gdb/all_2015'
species_Df=build_df_from_arcpy(table)
species_Df=species_Df.astype({'CODE':np.int16})
gridID=[max_species[i][0]-1 for i in max_species.index]
for_type=[max_species[i][1] for i in max_species.index]
for i in range(len(for_type)):
    for_type[i]=species_Df[species_Df.CODE==for_type[i]].FOREST_CAT[0]
for_type=zip(gridID,for_type)
for j in (set(gridID) ^ set(range(nos))):
    for_type.append((j,'d'))
for_type=sorted(for_type)      
os.chdir(Dir_CA)      
with open("grid_to_species.txt", "wb") as fp:   #Pickling
    pickle.dump(for_type, fp)      


#### recognize species on small grid:
from dirs import*
table=Dir_mort+'/species.gdb/small_grid_2001'
species_Df=build_df_from_arcpy(table)
species_Df=species_Df.astype(np.float)
species_Df=species_Df.astype(np.int16)
species_Df.index=species_Df.gridID-1
species_Df.drop('gridID',axis=1,inplace=True)
evergreen=species_Df[species_Df.MEAN.isin([1,7])].index
deciduous= species_Df[species_Df.MEAN.isin([6,8])].index
os.chdir(Dir_CA) 
#Pickling # both evergreen and decisuous need to be stored separately     
with open("small_grid_deciduous.txt", "wb") as fp:   
    pickle.dump(deciduous, fp) 