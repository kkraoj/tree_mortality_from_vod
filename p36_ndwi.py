# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:58:48 2019

@author: kkrao
"""

import pandas as pd

df = pd.read_excel(r"D:\Krishna\Project\working_tables.xlsx", \
                   sheet_name = "all_lat_lon",
                   index_col = 3)


gridIDs = pd.HDFStore('D:/Krishna/Project/data/Mort_Data/CA/data_subset_GC.h5')['RWC_matched'].columns
df = df.loc[gridIDs,['x','y']]
df.rename(columns = {'x':'longitude','y':'latitude'}, inplace = True)
df.index.name = "site"
df = df.loc[:,['latitude','longitude']]

print(df.head())
#df.to_csv(r"D:\Krishna\Project\data\RS_data\ndwi\queried_lat_lon.csv")

