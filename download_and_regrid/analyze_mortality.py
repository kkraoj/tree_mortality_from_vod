# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 13:19:13 2018

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dirs import Dir_CA

#start_year=2009
#end_year=2016
cardinal='#BD2031'

os.chdir(Dir_CA)
store=pd.HDFStore('data_subset_GC.h5')
Df=store['mortality_025_grid']
fam_thresh=0.1
print(Df.columns[(Df>=fam_thresh).any()==True])

'''

runfile('D:/Krishna/Project/codes/analyze_mortality.py', wdir='D:/Krishna/Project/codes')
Int64Index([ 33,  73,  83,  84,  91,  97, 104, 105, 117, 118, 128, 129, 130,
            137, 138, 139, 141, 149, 150, 151, 160, 161, 170, 171, 172, 173,
            181, 182, 183, 184, 195, 196, 197, 198, 207, 208, 209, 218, 219,
            220, 221, 227, 230, 231, 232, 233, 240, 258, 259, 260, 261, 268,
            272, 273, 274, 277, 281, 287, 288, 289, 290, 296, 304, 308, 311,
            312, 317, 320, 326, 328, 334, 335, 336, 343, 348, 349, 350],
           dtype='int64', name=u'gridID')

'''