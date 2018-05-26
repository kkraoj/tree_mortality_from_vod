# -*- coding: utf-8 -*-
"""
Created on Fri May 18 20:26:38 2018

@author: kkrao
"""

import pandas as pd
from simpledbf import Dbf5

dbf = Dbf5('c:/Users/kkrao/Desktop/species_map.dbf')

Df = dbf.to_dataframe()
