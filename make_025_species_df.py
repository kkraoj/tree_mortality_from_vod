# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 21:41:26 2017

@author: kkrao
"""

from dirs import*
nos=5938
table=Dir_mort+'/species.gdb/mapping_small_grid_to_grid'
mapping=build_df_from_arcpy(table,columns=['gridID','gridID_1'])
mapping=mapping.astype(np.float)
mapping=mapping.astype(np.int16)
mapping.index=mapping.gridID-1
mapping.drop('gridID',axis=1,inplace=True)
mapping.columns=['smallgridID']
mapping.smallgridID-=1

nos=370
species='evergreen'
species='deciduous'
os.chdir(Dir_CA)
store=pd.HDFStore('data.h5')
Df=store['TPA_005_grid']
Df=mask_columns(ind_small_species(species),Df)  
#Df.dropna(axis=1, how='all',inplace=True)
Df_new=pd.DataFrame(np.full((Df.shape[0],nos),0),index=Df.index)

for gridID in Df_new.columns:
    if gridID in mapping.index:
        if len(mapping.loc[gridID])>1:
            Df_new[gridID]=Df[mapping.loc[gridID].T.values[0]].mean(axis='columns')
        else:
            Df_new[gridID]=Df[mapping.loc[gridID].T.values[0]]
store['TPA_%s_025_grid'%species]=Df_new
store.close()

