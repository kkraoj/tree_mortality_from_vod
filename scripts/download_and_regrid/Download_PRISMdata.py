import urllib
import os
import sys
from dirs import MyDir

#data=['ppt','tmax','tmean']
#year_range=range(2005,2017)
#month_range=range(1,13)
#link='http://services.nacse.org/prism/data/public/4km/'
#for file in data:
#    os.chdir(MyDir+'/PRISM')
#    if not(os.path.isdir(MyDir+'/PRISM/'+file)):        
#        os.mkdir(file)
#    os.chdir(MyDir+'/PRISM/'+file)
#    for year in year_range:
#        for month in month_range:
#            linkname=link+file+'/%d%02d'%(year,month)
#            filename='PRISM_%s_stable_4kmM2_%d%02d_bil.zip'%(file,year,month)
#            if not(os.path.isfile(filename)):
#                urllib.urlretrieve(linkname,filename)
                
##-----------------------------------------------------------------------------

data=['vpdmax']
year_range=range(2005,2017)
month_range=range(1,13)
day_range=range(1,32)
link='http://services.nacse.org/prism/data/public/4km/'
for variable in data:
    os.chdir(MyDir+'/PRISM')
    if not(os.path.isdir(MyDir+'/PRISM/'+variable)):        
        os.mkdir(variable)
    os.chdir(MyDir+'/PRISM/'+variable)
    for year in year_range:
        for month in month_range:
            for day in day_range:
                sys.stdout.write('\r'+'Processing data for %d %02d %02d...'%(year,month,day))
                sys.stdout.flush()
                linkname=link+variable+'/%d%02d%02d'%(year,month,day)
                filename='PRISM_%s_stable_4kmM2_%d%02d%02d_bil.zip'%(variable,year,month,day)
                if not(os.path.isfile(filename)):
                    urllib.urlretrieve(linkname,filename)