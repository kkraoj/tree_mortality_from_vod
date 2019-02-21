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
data=Df.groupby(['gridID','FOR_TYPE1']).Shape_Area.sum()


max_species=data.groupby(level=0).idxmax()
max_species.index = max_species.index.astype(int)
for entry in max_species:
    max_species.loc[entry[0]] = entry[1]
max_species.index-=1

table=Dir_mort+'/species.gdb/all_2015_3_groups'
species_Df=build_df_from_arcpy(table, dtype = np.int16)
species_Df=species_Df.astype({'CODE':np.int16})
species_Df.index = species_Df.CODE

#max_species = max_species.apply(lambda x: species_Df.loc[x,'FOREST_CAT'])
#max_species.to_pickle(Dir_CA+'/grid_to_3_groups')   


##### percentage of all three groups for each grid
data_df = data.reset_index()
#not_found = []
#for index, row in data_df.iterrows():
#    if row.loc['FOR_TYPE1'] not in species_Df.index:
#        if row.loc['FOR_TYPE1'] not in not_found:
#            not_found.append(row.loc['FOR_TYPE1'])
data_df.loc[:,'FOR_TYPE1'] = data_df.loc[:,'FOR_TYPE1'].apply(lambda x: species_Df.loc[x,'FOREST_CAT'])
data_df = data_df.groupby(['gridID','FOR_TYPE1']).Shape_Area.sum().reset_index()
data_df = data_df.pivot(index = 'gridID',columns = 'FOR_TYPE1',values = 'Shape_Area')
data_df = data_df.divide(data_df.sum(axis = 1), axis = 0)
data_df.index-=1
#data_df.to_pickle(Dir_CA+'/grid_to_3_groups_fraction')
#