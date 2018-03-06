# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from dirs import *
arcpy.env.overwriteOutput=True
year_range=range(2009,2017)
day_range=range(1,362,2)
dir=MyDir+'/ASCAT/sigma-0/'
utilpath='D:/Krishna/Project/codes/SIR/'
for k in year_range:  
    year = '%s' %k          #Type the year 
    Y1='%02d'%(k-2000)
    print('Processing data for year '+year+' ...')
    for j in day_range:
        DY1='%03d'%j
        DY2='%03d'%(j+4)
        infile=dir+year+'/msfa-a-NAm'+Y1+'-'+DY1+'-'+DY2+'.sir'
        if os.path.exists(infile):
            outfile=dir+year+'/sigma0_'+year+DY1+'.tif'
            p = Popen(utilpath+'sir_util2', stdin=PIPE) #NOTE: no shell=True here
            #input file name, convert to what, geotiff, min saturation value, max sat value, show text, output file, quit
            p.communicate(os.linesep.join([infile,'6','3','-32','0','0',outfile,'0']))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
