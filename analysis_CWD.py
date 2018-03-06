# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:57:47 2017

@author: kkrao
"""
from __future__ import division
from dirs import *
import matplotlib.lines as mlines

arcpy.env.workspace = Dir_mort+'/CA_proc.gdb'
os.chdir(Dir_mort+'/CA_proc.gdb')
year_range=range(2005,2017)
month_range=range(12)

param=dict([('PEVAP', 'Potential Evaporation'), \
            ('EVBS', 'Direct evaporation from bare soil'), \
            ('EVCW', 'Canopy water evaporation'), \
            ('EVP', 'Evaporation'), \
            ('TRANS', 'Transpiration'),\
            ('SBSNO','Sublimation')])

 
        
store = pd.HDFStore(Dir_CA+'/mort.h5')          
mort=store['mort']
mort1=store['mort1']
mort2=store['mort2']
store = pd.HDFStore(Dir_CA+'/CWDu_Df.h5')
storeMOS = pd.HDFStore(Dir_CA+'/MOS_Df.h5')
storeY = pd.HDFStore(Dir_CA+'/Young_Df.h5') 
storeVOD=pd.HDFStore(Dir_CA+'/vodDf.h5') 

ind1=[ 32,  33,  57,  63,  73,  74,  75,  83,  84,  91,  97,  98, 104,
            105, 116, 117, 118, 128, 129, 130, 138, 139, 149, 150, 160, 161,
            171, 172, 183, 196, 197, 218, 219, 220, 221, 258, 261, 273, 289,
            296] # greater than 0.2
ind2=[34, 84, 98, 118, 129, 130, 140] # greater than 0.4
ind3=range(0,370)
index=dict([('0.2',ind1),('0.4',ind2),('0.0',ind3)])


fs=15
year=2016
year=year-2005
thresh=0.2
for p in param:
    plt.figure(p)
    b=store[p]
    b=b.drop('gridID',axis=1)
    b=b.iloc[index['%.1f'%thresh],:]
    b=b.iloc[:,12*year:(year+1)*12]
    ax=b.T.plot(legend=False,xticks=[],alpha=0.7,color='r')
    ax.set_xlabel('Month',fontsize=fs)
    ax.set_ylabel(param[p]+' (mm)',fontsize=fs)
    ax.set_ylim([0,410])
    #ax.set_xlim([columns*-0.05,columns*1.05])
    ax.set_title('Variation of '+param[p],fontsize=fs)     
    ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/12))
    ax.set_xticklabels(month_range)
    ax.grid(color='grey', linestyle='-', linewidth=0.2)
    ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)
#    ax.text(10,0,'FAM Threshold = %.1f'%thresh,fontsize=fs)  
    #plt.legend()    
    plt.show()
#    plt.close()   
## plot of summation of evaporation
col=dict([('EVBS', 'y'), \
            ('EVCW', 'g'), \
            ('EVP', 'r'), \
            ('TRANS', '0.3'),\
            ('SBSNO','b')])
fs=15
year=2015
year=year-2005
thresh=0.2
count=0
for p in param:
    if not(p=='PEVAP'):
        count=count+1        
        b=store[p]
        b=b.drop('gridID',axis=1)
        b=b.iloc[index['%.1f'%thresh],:]
        b=b.iloc[:,12*year:(year+1)*12]
        if count==1:          
            ax=b.T.plot(legend=False,xticks=[],alpha=0.4,color=col[p])
            ax1=b[0:1].T.plot(legend=False,xticks=[],color=col[p],label='%s'%p,ax=ax)
            ax=plt.gca()
            ax.set_ylim([0,410])
        else:
            b.T.plot(legend=False,xticks=[],alpha=0.4,color=col[p],ax=ax)
            ax1=b[0:1].T.plot(xticks=[],color=col[p],ax=ax,label='%s'%p,legend=False)
ax.grid(color='grey', linestyle='-', linewidth=0.2)            
ax.set_title('Components of Total Evapotranspiration',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/12))
ax.set_xticklabels(month_range)
ax.set_xlabel('Month',fontsize=fs)
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.5), xycoords='axes fraction',fontsize=fs)
ax.annotate('Year = %d'%(year+2005), xy=(0.05, 0.4), xycoords='axes fraction',fontsize=fs)
ax.set_ylabel('Quantity (mm)',fontsize=fs)
handles=range(len(param)-1)
i=0
for p in param:
    if not(p=='PEVAP'):
        handles[i]=matplotlib.lines.Line2D([], [], color=col[p], markersize=100, label=param[p])
        i+=1
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels) 
plt.show()


#### calc CWD

#cwd=store['PEVAP']-store['EVP']
cwd=store['PEVAP']-store['TRANS']-store['SBSNO']-store['EVCW']-store['EVBS']
fs=15
thresh=0.2
#plt.figure()

year=2016
year=year-2005
b=cwd
b=b.drop('gridID',axis=1)
b=b.iloc[index['%.1f'%thresh],:]
b=b.iloc[:,12*year:(year+1)*12]
ax=b.T.plot(legend=False,xticks=[],alpha=0.7,color='k')
ax.set_xlabel('Month',fontsize=fs)
ax.set_ylabel('CWD (mm)',fontsize=fs)
ax.set_ylim([0,410])
#ax.set_xlim([columns*-0.05,columns*1.05])
ax.set_title('Variation of CWD',fontsize=fs)     
ax.set_xticks(np.arange(0,len(b.columns),len(b.columns)/12))
ax.set_xticklabels(month_range)
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)
#plt.legend()    
#plt.show()
    
    
cwd_acc=pd.DataFrame()
for i in year_range:
    year=i-2005
    b=cwd
    b=b.drop('gridID',axis=1) 
    b=b.iloc[:,12*year:(year+1)*12]
    data=b.sum(1).rename('%d'%i)
    cwd_acc=pd.concat([cwd_acc,data],axis=1)

## plotting cwd _acc
plt.figure()
fs=15
thresh=0.2
b=cwd_acc.iloc[index['%.1f'%thresh],:]
m=b.mean(0)
ax=b.T.plot(legend=False,alpha=0.7,color='k')
m.plot(legend=True, linewidth=5,color='m',label='Mean')
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('CWD annually accumulated (mm)',fontsize=fs)
ax.set_title('Annual Accumulation of CWD over time',fontsize=fs)     
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)


#### plot both data Young Data
yn=range(2005,2017)
yp=range(2009,2016)
cwd_Y=storeY['cwd_acc']
plt.figure()
fs=15
thresh=0.2
mult=1
bp=cwd_Y.iloc[index['%.1f'%thresh],:]
bp=bp.drop('gridID',axis=1) 
mp=bp.mean(0)
bn=mult*cwd_acc.iloc[index['%.1f'%thresh],:]
mn=bn.mean(0)
plt.plot(yn,bn.T,alpha=0.1,color='b')
plt.plot(yn,mn.T,color='b',linewidth=4,label='NLDAS')
plt.plot(yp,bp.T,alpha=0.1,color='m')
plt.plot(yp,mp.T,color='m',linewidth=4,label='PRISM')
ax=plt.gca()
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('CWD annually accumulated (mm)',fontsize=fs)
ax.set_title('Comparison of NLDAS and PRISM CWD',fontsize=fs)     
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)
plt.legend(loc='upper right')
plt.show()

### mort vs cwd_acc scatter plot
thresh=0.2
y=mort.drop('gridID',axis=1)
x=cwd_acc
x=storeY['cwd_acc'].drop('gridID',axis=1)
x=x.iloc[index['%.1f'%thresh],:-1] # dropped 2016
y=y.iloc[index['%.1f'%thresh],:-1]
plt.figure()
plt.plot(x,y,'ro',mfc='None')
ax=plt.gca()
ax.set_xlabel('CWD (mm)',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.45])
ax.set_title('Mortality Vs CWD',fontsize=fs)     
ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)
#plt.legend()

### make box plot for NLDAS
y=mort.drop('gridID',axis=1)
x=cwd_acc
x=x.iloc[index['%.1f'%thresh],:-1] # dropped 2016
y=y.iloc[index['%.1f'%thresh],:-1]
y=y.values.flatten()
x=x.values.flatten()
boxes=7
min=(np.min(x))
max=(np.max(x))
binwidth=(max-min)/boxes
yb=np.empty((len(x),boxes+1,))
yb[:] = np.nan
for i in range(boxes+1):
    row=np.where((x>=min+i*binwidth) & (x<min+(i+1)*binwidth))
    yb[row,i]=y[row]
yb=pd.DataFrame(yb)
name=(np.arange(min,max+binwidth/2,binwidth))
name=name.astype(int)
yb.columns=name
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
ax.set_xlabel('Annual CWD (mm)',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.35])
ax.set_title('Box plot of Mortality Vs CWD',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)

### make box plot for PRISM
thresh=0.2
y=mort.drop('gridID',axis=1)
x=storeY['cwd_acc'].drop('gridID',axis=1)
x=x.iloc[index['%.1f'%thresh],:] 
y=y.iloc[index['%.1f'%thresh],4:11] # 2009 - 2015
y=y.values.flatten()
x=x.values.flatten()
boxes=7
min=(np.min(x))
max=(np.max(x))
binwidth=(max-min)/boxes
yb=np.empty((len(x),boxes+1,))
yb[:] = np.nan
for i in range(boxes+1):
    row=np.where((x>=min+i*binwidth) & (x<min+(i+1)*binwidth))
    yb[row,i]=y[row]
yb=pd.DataFrame(yb)
name=(np.arange(min,max+binwidth/2,binwidth))
name=name.astype(int)
yb.columns=name
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
ax.set_xlabel('Annual CWD (mm)',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.08])
ax.set_title('Box plot of Mortality Vs CWD',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)

#### CWD anomaly and plot for NLDAS
y=mort.drop('gridID',axis=1)
x=cwd_acc
x=(x-x.mean(0))/x.std(0)
x=x.iloc[index['%.1f'%thresh],:-1] # dropped 2016
y=y.iloc[index['%.1f'%thresh],:-1]
m=x.mean(1)
s=x.std(1)
x=x.sub(m,0)
x=x.div(s,0)
y=y.values.flatten()
x=x.values.flatten()
boxes=7
min=(np.min(x))
max=(np.max(x))
binwidth=(max-min)/boxes
yb=np.empty((len(x),boxes+1,))
yb[:] = np.nan
for i in range(boxes+1):
    row=np.where((x>=min+i*binwidth) & (x<min+(i+1)*binwidth))
    yb[row,i]=y[row]
yb=pd.DataFrame(yb)
name=(np.arange(min+binwidth/2,max+binwidth/2+binwidth/10,binwidth))

name=np.round(name,2)
yb.columns=name
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.35])
ax.set_title('Box plot of Mortality Vs CWD Anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)

##scatter plot
plt.figure()
plt.plot(x,y,'or',mfc='None')
ax=plt.gca()
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title('Mortality Vs CWD Anomaly',fontsize=fs)
ax.set_ylim([0,0.35])






### make box plot for PRISM
thresh=0.2
y=mort.drop('gridID',axis=1)
x=storeY['cwd_acc'].drop('gridID',axis=1)
x=x.iloc[index['%.1f'%thresh],:] 
y=y.iloc[index['%.1f'%thresh],4:11] # 2009 - 2015
m=x.mean(1)
s=x.std(1)
x=x.sub(m,0)
x=x.div(s,0)
y=y.values.flatten()
x=x.values.flatten()
boxes=7
min=(np.min(x))
max=(np.max(x))
binwidth=(max-min)/boxes
yb=np.empty((len(x),boxes+1,))
yb[:] = np.nan
for i in range(boxes+1):
    row=np.where((x>=min+i*binwidth) & (x<min+(i+1)*binwidth))
    yb[row,i]=y[row]
yb=pd.DataFrame(yb)
name=(np.arange(min+binwidth/2,max+binwidth/2+binwidth/10,binwidth))
name=np.round(name,2)
yb.columns=name
### box plots
plt.figure()
fs=15
ax=yb.boxplot(fontsize=fs,grid=False,showfliers=False)
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.35
#ax.set_xlim([-2,1.5])
ax.set_title('Box plot of Mortality Vs CWD Anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)

plt.figure()
plt.plot(x,y,'or',mfc='None')
ax=plt.gca()
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_title('Mortality Vs CWD Anomaly',fontsize=fs)
ax.set_ylim([0,0.35])

##### PRISM box plots with equal number in bins
thresh=0.2
y=mort.drop('gridID',axis=1)
x=storeY['cwd_acc'].drop('gridID',axis=1)
x=x.iloc[index['%.1f'%thresh],:] 
y=y.iloc[index['%.1f'%thresh],4:11] # 2009 - 2015
m=x.mean(1)
s=x.std(1)
x=x.sub(m,0)
x=x.div(s,0)
y=y.values.flatten()
x=x.values.flatten()
inds=x.argsort()
x=x[inds]
y=y[inds]
boxes=10
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
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.08])
#ax.set_xlim([-2,1.5])
ax.set_title('Box plot of Mortality Vs CWD Anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)


#### VOD box plots with equal number in bins
thresh=0.2
y=mort.drop('gridID',axis=1)
x=storeVOD['vodDf'].drop('gridID',axis=1)

x=x.iloc[index['%.1f'%thresh],:] 
y=y.iloc[index['%.1f'%thresh],-1] # 2005 - 2015
m=x.mean(1)
s=x.std(1)
x=x.sub(m,0)
x=x.div(s,0)
y=y.values.flatten()
x=x.values.flatten()
inds=x.argsort()
x=x[inds]
y=y[inds]
boxes=9
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
ax.set_xlabel('Annual CWD Anomaly',fontsize=fs)
ax.set_ylabel('FAM',fontsize=fs)
ax.set_ylim([0,0.08])
#ax.set_xlim([-2,1.5])
ax.set_title('Box plot of Mortality Vs CWD Anomaly',fontsize=fs)     
ax.annotate('FAM Threshold = %.1f'%thresh, xy=(0.05, 0.85), xycoords='axes fraction',fontsize=fs)
