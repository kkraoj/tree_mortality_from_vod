# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from dirs import*
year_range=range(2005,2016)
date_range=range(1,366,1)
pass_type = 'A';             #Type the overpass: 'A' or 'D'
param = 'tc10';           #Type the parameter
factor = 1e-0;          #Type the multiplier associated with the factor
bnds = [0.0,3.0];       #Type the lower and upper bounds of the parameter      
vod_world=pd.DataFrame()           
for k in year_range:
    for j in date_range:        
        year = '%s' %k          #Type the year
        if j>=100:
            date='%s'%j
        elif j >=10:
            date='0%s' %j
        else:
            date='00%s'%j        
        fname=MyDir+'/'+param+'/'+year+'/AMSRU_Mland_'+year+date+pass_type+'.'+param
        if  os.path.isfile(fname):
            fid = open(fname,'rb');
            data=np.fromfile(fid)
            fid.close()            
            for i in list(range(len(data))):
                if data[i]<=0.0:
                    data[i]=np.nan                     
            data = [i*factor for i in data]; 
            data = -np.log(data)
            data=pd.DataFrame(data,columns=[year+date+pass_type])
            vod_world=pd.concat([vod_world,data],axis=1)            
store = pd.HDFStore(Dir_CA+'/vod_world.h5')       
store['vod'] = vod_world

y=np.asarray(late)
y=y.flatten()
x=np.asarray(lone)
x=x.flatten(1)
top = 49.3457868 # north lat
left = -124.7844079 # west long
right = -66.9513812 # east long
bottom =  24.7433195 # south lat
index= (y>=bottom) & (y<=top)
y=y[index]
x=x[index]       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
