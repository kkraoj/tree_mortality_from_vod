# -*- coding: utf-8 -*-
"""
Created on Fri Nov 09 14:23:11 2018

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
from dirs import Dir_CA, select_north_south_grids
from scipy import stats
os.chdir(Dir_CA)

store = pd.HDFStore('data_subset_GC.h5')
climate_sources = input_sources=[
             'cwd','ppt_sum','ppt_win','tmax_sum','tmax_win',\
             'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
            'PEVAP_sum','EVP_win','PEVAP_win','vsm_sum','vsm_win']
for source in climate_sources:
    Df = store[source]
    Df_north, Df_south = select_north_south_grids(Df)
    kstat,pvalue = stats.ks_2samp(Df_north.values.flatten(), Df_south.values.flatten())
    print('{source}\t {kstat:.2f}\t {pvalue:.3f}'.format(source = source, kstat = kstat, pvalue = pvalue))