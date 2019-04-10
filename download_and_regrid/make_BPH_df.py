# Name: Dissolve_Example2.py
# Description: Dissolve features based on common attributes

 
# Import system modules
from __future__ import division
from dirs import *
year_range=range(2009,2016)
table=MyDir+'/Young/Young.gdb/Young_gridded_025' 
nos=370
col_names=range(nos)
columns=['gridID','year','live_bah']
Df_main=build_df_from_arcpy(table,columns)
Df_main=Df_main.astype(np.float)
Df_main=Df_main.astype({'gridID':np.int16,'year':np.int16})
Df_main.gridID=Df_main.gridID-1
Df=pd.DataFrame(np.full((len(year_range),nos),0), \
                       index=[pd.to_datetime(year_range,format='%Y')],\
                             columns=col_names)
Df.columns.name='gridID'
for year in year_range:
    data=Df_main[Df_main.year==year]
    Df.loc[pd.to_datetime(year,format='%Y')]=data.groupby('gridID').live_bah.mean()
Df.fillna(0,inplace=True)
#------------------------------------------------store data
os.chdir(Dir_CA)
store_major=pd.HDFStore('data.h5')
store_major['BPH_025_grid']=Df
store_major.close()



