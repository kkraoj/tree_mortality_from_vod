# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:41:22 2017

@author: kkrao
"""
###response is number of dead trees
import os
import arcpy
import matplotlib as mpl
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from dirs import Dir_CA, Dir_mort, get_marker_size, RWC,clean_xy, piecewise_linear,\
append_prediction
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle  
from plot_functions import plot_timeseries_maps,plot_leaf_habit,plot_RWC_timeseries,\
    plot_regression,plot_importance, plot_correlation,plot_importance_rank,plot_pdf
os.chdir(Dir_CA)

#timeseries_maps(var2='RWC',var2_range=[1e-5,1], var2_label='Relatve\nwater content',cmap2='inferno',\
#                title='Timeseries of observed mortality and RWC')
cb0=plot_timeseries_maps()
#------------------------------------------------------------------------
plot_leaf_habit(cmap='plasma')

plot_RWC_timeseries()

plot_regression()

plot_importance()

plot_correlation()

plot_pdf()
