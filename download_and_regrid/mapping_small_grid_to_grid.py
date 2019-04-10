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

os.chdir(Dir_CA)
store=pd.HDFStore('data.h5')
Df=store['cwd']
Df_new=pd.DataFrame(np.full((Df.shape[0],nos),np.nan),index=Df.index)
for index in Df.columns:
    sys.stdout.write('\r'+'Processing data for %s'%index+' ...')
    sys.stdout.flush()
    if index in mapping.index: 
        data=[Df[index]]*len(mapping.loc[index])
        data=pd.DataFrame(data).T
        if len(mapping.loc[index].values.tolist())>1:                 
            data.columns=[item for sublist in mapping.loc[index].values.tolist() \
                      for item in sublist]
        else:
            data.columns= mapping.loc[index].values.tolist()
        Df_new.update(data)

store['cwd_005_grid']=Df_new
store.close()