
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 23:26:40 2017

@author: kkrao
"""
import os
import pandas as pd
import numpy as np
from dirs import MyDir
store=pd.HDFStore(MyDir+'/VOD_VSM_Monthly.h5') ## address of data
Df=store['vod'] ## for soil moisture do 'vsm'
store.close()

#data now stored in a pandas dataframe
## view sample data

Df.head()

## access data for some samplt month 2005 Jun
Df.shape
Df.loc['2005-06'].shape
voddates=pd.Series(Df.index.get_level_values('Time').unique())
vod=np.repeat(np.zeros((1, 56,103)),len(voddates),axis=0)
i=0
for dates in voddates:
   vod[i,:,:]=Df.loc[dates]
   i+=1
voddates=np.array(voddates.apply(lambda dt: dt.replace(day=1)))
os.chdir('C:/Users/kkrao/Dropbox/fire_proj_data')
np.save('vod',vod)    
np.save('voddates',voddates) 
nl=np.load('nl.npy')
