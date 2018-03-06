# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:35:32 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
date_range=range(1,367)
year_range=range(2005,2016)
store = pd.HDFStore(Dir_CA+'/vodDf.h5')          
vod=store['vodDf']
store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort1=store['mort1']
mort2=store['mort2']

######## plotting begins
b=vod.iloc[:,2183:2583]
b=b>2.2
b=b.sum(1)
b=b[b>0]
b=b[b>10]
b.index=b.index+1
print(b.index)


b=mort.iloc[:,1:]
b=b>0.4
b=b.sum(1)
b=b[b>0]
b.index=b.index+1
print(b.index)



### plot whole data
#
plt.figure()
fs=15
b=mort.drop('gridID',axis=1)
ax=b.T.plot(legend=False,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('Fractional area of mortality',fontsize=fs)
ax.set_title('Variation of fractional area of ALL mortality',fontsize=fs)     
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(np.arange(2005,2017))        
           
            
            
plt.figure()
fs=15
b=mort1.drop('gridID',axis=1)
ax=b.T.plot(legend=False,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('Fractional area of mortality',fontsize=fs)
ax.set_title('Variation of fractional area of mortality,Sev=1',fontsize=fs)     
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(np.arange(year_range)                     
            
            
plt.figure()
fs=15
b=mort2.drop('gridID',axis=1)
ax=b.T.plot(legend=False,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('Fractional area of mortality',fontsize=fs)
ax.set_title('Variation of fractional area of mortality,Sev=2',fontsize=fs)     
ax.set_xticks(np.arange(0,12))
ax.set_xticklabels(year_range)        
              
            
            
plt.figure()
fs=15
b=vod.drop('gridID',axis=1)
ax=b.T.plot(legend=False,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('VOD',fontsize=fs)
ax.set_title('Variation of VOD',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/11))
ax.set_xticklabels(year_range)
ax.grid(color='grey', linestyle='-', linewidth=0.2)   
ax.text(500,2.5,'FAM Threshold=0',fontsize=fs)      
   


plt.figure()
fs=15
b=vod.drop('gridID',axis=1)
ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
b=vod.iloc[ind,:]
ax=b.T.plot(legend=False,xticks=[])
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('VOD',fontsize=fs)
ax.set_ylim([0.5,3.0])
ax.set_title('Variation of VOD',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/11))
ax.set_xticklabels(year_range)
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.text(2005,2.5,'Threshold=0.2',fontsize=fs)      
 


          
            
            
            
plt.figure()
fs=15
b=mort.drop('gridID',axis=1)
ax=b.plot.hist(alpha=0.5,bins=20)
ax.set_xlabel('Fractional area of mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_title('Histogram of fractional area of ALL mortality',fontsize=fs)  
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,350])          
   

plt.figure()
fs=15
b=mort1.drop('gridID',axis=1)
ax=b.plot.hist(alpha=0.5,bins=20)
ax.set_xlabel('Fractional area of mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_title('Histogram of fractional area of mortality, Sev=1',fontsize=fs)  
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,350])          


plt.figure()
fs=15
b=mort2.drop('gridID',axis=1)
ax=b.plot.hist(alpha=0.5,bins=20)
ax.set_xlabel('Fractional area of mortality',fontsize=fs)
ax.set_ylabel('Frequency',fontsize=fs)
ax.set_title('Histogram of fractional area of mortality, Sev=2',fontsize=fs)  
ax.set_xlim([-0.02,0.6])
ax.set_ylim([0,350])          







ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
ind=range(0,370)

plt.figure()
for i in range(0,11):
    b=vod.drop('gridID',axis=1)
    b=b.iloc[ind,:]
    b=b.iloc[:,i*365:(i+1)*365]
    s=b.std(1)
    year=2005+i
    year='%s'%year
    m=mort['fam_'+year]
    m=m.iloc[ind]   
    plt.plot(s,m,'ro',mfc='none')
fs=15
ax=plt.gca()
ax.set_xlabel('s.d.(VOD)',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Scatter Plot of FAM Vs s.d.(VOD) ',fontsize=fs) 
plt.text(0.6,0.3,'Threshold = 0',fontsize=fs)          




ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2

#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
ind=range(0,370)

plt.figure()
for i in range(0,11):
    b=vod.drop('gridID',axis=1)
    b=b.iloc[ind,:]
    b=b.iloc[:,i*365:(i+1)*365]
    mu=b.mean(1)
    s=b.std(1)
    b=b.sub(mu,0)
    b=b.abs()
    b=b.divide(s,0)
    b=b.mean(1)
    year=2005+i
    year='%s'%year
    m=mort['fam_'+year]
    m=m.iloc[ind]   
    plt.plot(b,m,'ro',mfc='none')
fs=15
ax=plt.gca()
ax.set_xlim([0.55,1])   
ax.set_ylim([0,0.4])
ax.set_xlabel('seasonal spread(VOD)',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Scatter Plot of FAM Vs seasonal spread(VOD) ',fontsize=fs)
plt.text(0.6,0.3,'FAM Threshold = 0',fontsize=fs)   
plt.plot([0.55,0.9],[0,0.4],'-k')        



ind=range(0,370)
plt.figure()
for i in range(0,11):
    b=vod.drop('gridID',axis=1)
    b=b.iloc[ind,:]
    b=b.iloc[:,i*365:(i+1)*365]
    mu=b.mean(1)
    s=b.std(1)
    b=b.sub(mu,0)
    b=b.abs()
    b=b.divide(s,0)
    b=b.mean(1)
    year=2005+i
    year='%s'%year
    m=mort1['fam_'+year+'_1']
    m=m.iloc[ind]   
    plt.plot(b,m,'ro',mfc='none')
fs=15
ax=plt.gca()
ax.set_xlabel('seasonal spread(VOD)',fontsize=fs)
ax.set_xlim([0.55,1])
ax.set_ylim([0,0.4])
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Scatter Plot of FAM Vs seasonal spread(VOD) ',fontsize=fs)
plt.text(0.6,0.3,'Severity = 1',fontsize=fs)           






#ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
#            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
#            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
#            296] # greater than 0.2

#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4

ind=range(0,370)
plt.figure()
for k in year_range:
    b=vod.drop('gridID',axis=1)
#    for j in date_range:
    year = '%s' %k          #Type the year
    if j>=100:
        date='%s'%j
    elif j >=10:
        date='0%s' %j
    else:
        date='00%s'%j 
    col_name=year+date+'A'
    print(b.at[0,col_name]),


def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]





ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2

b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
b=~np.isnan(b)
b=b*1
b=b.mean(0)
#b=b.rolling(window=20).mean()
b=smooth(b,window_len=20,window='bartlett')
plt.plot((np.array(b)),'-c', label='FAM Threshold=0.2')

ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
b=~np.isnan(b)
b=b*1
b=b.mean(0)
#b=b.rolling(window=20).mean()
b=smooth(b,window_len=20,window='bartlett')
plt.plot((np.array(b)),'-b', label='FAM Threshold=0.4')

ind=range(0,370)
b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
b=~np.isnan(b)
b=b*1
b=b.mean(0)
b=smooth(b,window_len=20,window='bartlett')
plt.plot((np.array(b)),'-m', label='FAM Threshold=0')
fs=15

ax=plt.gca()
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylim([0.0,1.0])
ax.set_ylabel('Fraction of VOD Data available',fontsize=fs)
ax.set_title('VOD Data availability',fontsize=fs)
ax.set_xticks(np.arange(0,len(b),len(b)/11))
ax.set_xticklabels(year_range) 
ax.grid(color='grey', linestyle='-', linewidth=0.2) 
ax.legend(fontsize=fs-2) 


############# 2005-2015
ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
thresh=0.2
#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
#ind=range(0,370)
num=len(ind)
b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
for i in range(0,52*11-2):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)
week=np.array(range(0,11*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,50):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)
colnames=[]
for i in range(0,50):
    colnames.append('%d'%i)
m1.columns=colnames 
s1.columns=colnames    
m2=m1.mean(0)
s2=s1.mean(0)

############################ for 2005 - 2010 only
ind=range(0,370)
num=len(ind)
b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
for i in range(0,52*6):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)


week=np.array(range(0,6*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,52):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)

colnames=[]
for i in range(0,52):
    colnames.append('%d'%i)

m1.columns=colnames 
s1.columns=colnames    
m2=m1.mean(0)
s2=s1.mean(0)

########################

############################ for 2010 - 2015 only
ind=range(0,370)
num=len(ind)
b=vod.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
for i in range(52*6,52*11-2):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)


week=np.array(range(0,5*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,50):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)

colnames=[]
for i in range(0,50):
    colnames.append('%d'%i)

m1.columns=colnames 
s1.columns=colnames    

m2=m1.mean(0)
s2=s1.mean(0)

########################
plt.figure()
s1.T.plot(legend=False,alpha=0.1,label='_nolegend_')
s2.plot(legend=True, linewidth=3,color='m',label='Mean')
fs=15
ax=plt.gca()
ax.set_xlabel('Week of Year',fontsize=fs)
ax.set_ylim([0.0,0.15])
#ax.set_xlim(1)
ax.set_ylabel(r'$\sigma(VOD)$',fontsize=fs)
ax.set_title(r'$\sigma(VOD)$ Vs Week of Year for 2005 - 2015',fontsize=fs)
#ax.set_xticks(np.arange(0,len(s2)+1,len(s2)/5))
#ax.set_xticklabels(range(0,51,10)) 
plt.text(10,0.02,'FAM Threshold = %.1f'%thresh,fontsize=fs)  
ax.grid(color='grey', linestyle='-', linewidth=0.2) 

plt.close()


plt.figure()
m1.T.plot(legend=False,alpha=0.5)
m2.plot(legend=True,linewidth=3,color='m',label='Mean')
fs=15
ax=plt.gca()
ax.set_xlabel('Week of Year',fontsize=fs)
ax.set_ylim([0.4,2.1])
ax.set_ylabel(r'$\mu(VOD)$',fontsize=fs)
ax.set_title(r'$\mu(VOD)$ Vs Week of Year (2005 - 2015)',fontsize=fs)
plt.text(10,0.6,'FAM Threshold = %.1f'%thresh,fontsize=fs)  
ax.grid(color='grey', linestyle='-', linewidth=0.2) 


            
################### x function = (x - mu) / sigma

ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
thresh=0.2
#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
#thresh=0.4
#ind=range(0,370)
#thresh=0
num=len(ind)
vod1=pd.concat([vod,vod.iloc[:,3995-10:]],axis=1)
b=vod1.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
colnames=[]
for i in range(0,52):
    colnames.append('%d'%i)
for i in range(0,52*11):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)
week=np.array(range(0,11*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,52):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)
m1.columns=colnames 
s1.columns=colnames    
m2=m1.mean(0)
s2=s1.mean(0)
mew.columns=colnames*11
std.columns=colnames*11
m1=pd.concat([m1]*11,axis='columns')
s1=pd.concat([s1]*11,axis='columns')
numr=mew.sub(m1)
x=numr.divide(s1)

minx=pd.DataFrame()
for i in range(len(year_range)):
    c=x.iloc[:,i*52:(i+1)*52]
    c=c.min(axis='columns')
    minx=pd.concat([minx,c],axis=1)
minx.colnames=year_range
y=mort.drop(['gridID','fam_2016'],axis=1)
y=y.iloc[ind,:]

plt.figure()
plt.plot(minx,y,'or',mfc='none')
fs=15
plt.gca().invert_xaxis()
ax=plt.gca()
ax.set_xlim([1,-3])   
ax.set_ylim([0,0.45])
ax.set_xlabel('Min weekly VOD anomaly',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Scatter Plot of FAM Vs VOD anomaly ',fontsize=fs)
plt.text(0.8,0.4,'FAM Threshold = %.1f'%thresh,fontsize=fs)   
#plt.text(0.8,0.4,'Severity =1',fontsize=fs)   
plt.plot([0.6,-1.5],[0,0.45],'-k')        


y=y.values.flatten()
x=minx.values.flatten()
binwidth=0.5
yb=np.empty((len(x),8,))
yb[:] = np.nan
for i in range(-6,2):
    index=np.where((x>=i/2) & (x<i/2+binwidth))
    yb[index,i+6]=y[index]
yb=pd.DataFrame(yb)
name=(np.arange(-2.75,1.25,0.5))
yb.columns=name
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
plt.gca().invert_xaxis()
#ax.set_xlim([1,-3])   
ax.set_ylim([0,0.08])
ax.set_xlabel('Min weekly VOD anomaly',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Box Plot of FAM Vs VOD anomaly ',fontsize=fs)
plt.text(8,0.06,'FAM Threshold = %.1f'%thresh,fontsize=fs)   
#plt.text(0.8,0.4,'Severity =1',fontsize=fs)          


### sev =1 and 2 
#ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
#            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
#            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
#            296] # greater than 0.2
#thresh=0.2
#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
#thresh=0.4
ind=range(0,370)
thresh=0
num=len(ind)
vod1=pd.concat([vod,vod.iloc[:,3995-10:]],axis=1)
b=vod1.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
colnames=[]
for i in range(0,52):
    colnames.append('%d'%i)
for i in range(0,52*11):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)
week=np.array(range(0,11*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,52):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)
m1.columns=colnames 
s1.columns=colnames    
m2=m1.mean(0)
s2=s1.mean(0)
mew.columns=colnames*11
std.columns=colnames*11
m1=pd.concat([m1]*11,axis='columns')
s1=pd.concat([s1]*11,axis='columns')
numr=mew.sub(m1)
x=numr.divide(s1)

minx=pd.DataFrame()
for i in range(len(year_range)):
    c=x.iloc[:,i*52:(i+1)*52]
    c=c.min(axis='columns')
    minx=pd.concat([minx,c],axis=1)
minx.colnames=year_range
y=mort.drop(['gridID','fam_2016'],axis=1)
y=y.iloc[ind,:]

sev=2
y=mort2.drop(['gridID','fam_2016_%d'%sev],axis=1)
y=y.iloc[ind,:]  
plt.figure()
plt.plot(minx,y,'or',mfc='none')
fs=15
plt.gca().invert_xaxis()
ax=plt.gca()
ax.set_xlim([1,-3])   
ax.set_ylim([0,0.45])
ax.set_xlabel('Min weekly VOD anomaly',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Scatter Plot of FAM Vs VOD anomaly ',fontsize=fs)
#plt.text(0.8,0.4,'FAM Threshold = %.1f'%thresh,fontsize=fs)   
plt.text(0.8,0.4,'Severity = %d'%sev,fontsize=fs)   
plt.plot([0.6,-1.5],[0,0.45],'-k')        
  

y=y.values.flatten()
x=minx.values.flatten()
binwidth=0.5
yb=np.empty((len(x),8,))
yb[:] = np.nan
for i in range(-6,2):
    index=np.where((x>=i/2) & (x<i/2+binwidth))
    yb[index,i+6]=y[index]
yb=pd.DataFrame(yb)
name=(np.arange(-2.75,1.25,0.5))
yb.columns=name

                                 
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
plt.gca().invert_xaxis()
#ax.set_xlim([1,-3])   
ax.set_ylim([-0.01,0.05])
ax.set_xlabel('Min weekly VOD anomaly',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Box Plot of FAM Vs VOD anomaly ',fontsize=fs)
#plt.text(8,0.20,'FAM Threshold = %.1f'%thresh,fontsize=fs)  
plt.text(8,0.2,'Severity = %d'%sev,fontsize=fs)      
            
            
#### make box plot with equal width

ind=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
thresh=0.2
#ind=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
#thresh=0.4
#ind=range(0,370)
#thresh=0
num=len(ind)
vod1=pd.concat([vod,vod.iloc[:,3995-10:]],axis=1)
b=vod1.drop('gridID',axis=1)
b=b.iloc[ind,:]
mew=pd.DataFrame()
std=pd.DataFrame()
colnames=[]
for i in range(0,52):
    colnames.append('%d'%i)
for i in range(0,52*11):
    c=b.iloc[:,range(i*7,(i+1)*7)]
    m=c.mean(1)
    s=c.std(1)
    mew=pd.concat([mew,m],axis=1)
    std=pd.concat([std,s],axis=1)
week=np.array(range(0,11*52,52))
m1=pd.DataFrame()
s1=pd.DataFrame()
for i in range(0,52):
    m=mew.iloc[:,week+i].mean(1)
    s=mew.iloc[:,week+i].std(1)
    s.rename(columns='%f'%i)
    m1=pd.concat([m1,m],axis=1)
    s1=pd.concat([s1,s],axis=1)
m1.columns=colnames 
s1.columns=colnames    
m2=m1.mean(0)
s2=s1.mean(0)
mew.columns=colnames*11
std.columns=colnames*11
m1=pd.concat([m1]*11,axis='columns')
s1=pd.concat([s1]*11,axis='columns')
numr=mew.sub(m1)
x=numr.divide(s1)

minx=pd.DataFrame()
for i in range(len(year_range)):
    c=x.iloc[:,i*52:(i+1)*52]
    c=c.min(axis='columns')
    minx=pd.concat([minx,c],axis=1)
minx.colnames=year_range
y=mort.drop(['gridID','fam_2016'],axis=1)
y=y.iloc[ind,:]
y=y.values.flatten()
x=minx.values.flatten()
inds=x.argsort()
x=x[inds]
y=y[inds]
boxes=8
count=len(x)/boxes
count=np.ceil(count).astype(int)
yb=pd.DataFrame()
for i in range(boxes):
    data=y[i*count:(i+1)*count]
    name=np.mean(x[i*count:(i+1)*count]).round(2)
    data=pd.DataFrame(data,columns=[name])
    yb=pd.concat([yb,data],axis=1)
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=0.7*fs,grid=False,showfliers=False)
plt.gca().invert_xaxis()
#ax.set_xlim([1,-3])   
ax.set_ylim([0,0.08])
ax.set_xlabel('Min weekly VOD anomaly',fontsize=fs)
ax.set_ylabel('Fractional area of mortality (FAM)',fontsize=fs)
ax.set_title('Box Plot of FAM Vs VOD anomaly ',fontsize=fs)
plt.text(8,0.06,'FAM Threshold = %.1f'%thresh,fontsize=fs)   
#plt.text(0.8,0.4,'Severity =1',fontsize=fs)          

