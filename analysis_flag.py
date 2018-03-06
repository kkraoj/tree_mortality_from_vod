# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""

from dirs import *
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
date_range=range(1,366)
year_range=range(2005,2016)
store = pd.HDFStore(Dir_CA+'/flagDf3.h5')          
flag=store['flag']
store = pd.HDFStore(Dir_CA+'/vod_world.h5')          
vodW=store['vod']
#store = pd.HDFStore(Dir_CA+'/mort.h5')          
#mort=store['mort']
#mort1=store['mort1']
#mort2=store['mort2']

plt.figure()
fs=15
b=flag.drop('gridID',axis=1)
ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
thresh=0.2
#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
#thresh=0.4
#ind=range(0,370)
#thresh=0
b=b.iloc[ind,:]
b=b.rolling(window=40,axis=1).mean()
rows, columns = b.shape

ax=b.T.plot(legend=False,xticks=[],alpha=0.7)
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('flag',fontsize=fs)
ax.set_ylim([0,8])
ax.set_xlim([columns*-0.05,columns*1.05])
ax.set_title('Variation of flag',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/11))
ax.set_xticklabels(year_range)
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.text(2005,2.5,'FAM Threshold = %.1f'%thresh,fontsize=fs)  
#plt.legend()    
plt.show()
plt.close()   

start = time.clock()
fs=15
b=vodW
jump=1000
rows, columns = b.shape
jump=range(0,rows,jump)
b=b.iloc[jump]
m=b.mean(0)
m=m.rolling(window=10).mean()
#b=b.rolling(window=40,axis=1).mean()
plt.figure()
ax=b.T.plot(legend=False,xticks=[],alpha=0.5, linewidth=1)
m.plot(linewidth=2,label='Mean',color='m',legend=True,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('VOD',fontsize=fs)
ax.set_ylim([0,3])
ax.set_xlim([columns*-0.05,columns*1.05])
ax.set_title('Variation of World VOD',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/11))
ax.set_xticklabels(year_range)
ax.grid(color='grey', linestyle='-', linewidth=0.2)
#ax.text(2005,2.5,'FAM Threshold = %.1f'%thresh,fontsize=fs)     
plt.show()
#plt.close() 
elapsed = (time.clock() - start)
print(elapsed)

