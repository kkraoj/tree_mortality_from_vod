# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 20:50:23 2017

@author: kkrao
"""

from dirs import*
table=Dir_mort+'/species.gdb/all_2015'
Df_main=build_df_from_arcpy(table)
