# -*- coding: utf-8 -*-
"""
Created on Fri May 18 20:26:38 2018

@author: kkrao
"""

import pandas as pd
Dir_CA='D:/Krishna/Project/data/Mort_Data/CA'
dataset='test_data'
Df=pd.read_csv('D:/Krishna/Project/data/rf_%s.csv'%dataset,index_col=0)   
Df = Df.loc[:,['FAM', 'predicted_FAM']]
corr = Df.corr(method = 'pearson')**2
print('r times r = %0.2f'%corr.iloc[0,1])



R2 = 1- ((Df.FAM - Df.predicted_FAM)**2).sum()/((Df.FAM - Df.FAM.mean())**2).sum()
print('R2 = %0.2f'%R2)