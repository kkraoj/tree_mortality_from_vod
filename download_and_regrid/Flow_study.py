# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:58:24 2017

@author: kkrao
"""
## import data
from Flow_study__init__ import * 
os.chdir(Dir_data)
American_water_data=pd.read_csv('custom sql query (american_water).csv')
San_Gabriel_data=pd.read_csv('custom sql query (san_gabriel_valley).csv')
##inputs
number_of_bins=10
uncertainty_factor=1
meter_limits_standard=[20,30,50,100,160,350,600,1350]
meter_limits_adjusted=pd.DataFrame([x*60*uncertainty_factor for x in meter_limits_standard]\
                                   ,columns=['adjusted max gph']\
                                   ,index=[0.625,0.75,1,1.5,2,3,4,6])
meter_limits_adjusted.index.name='Meter sizes'
## cleanup
df=American_water_data.copy()
df=df[df.volume>=0]
df = df[df['Meter Size'] != 0]
Meter_sizes=sorted(df['Meter Size'].unique())
Meter_sizes=[1.5,2,3,6]
for size in Meter_sizes:
    df[df['Meter Size']==size]=df[df['Meter Size']==size][df[df['Meter Size']==size]['volume']<=meter_limits_adjusted.ix[size][0]]
#plotting
fig, axs = plt.subplots(2,2, figsize=(6, 4))
fig.subplots_adjust(hspace = .5, wspace=.5)
axs=axs.ravel()
count=0
for size in Meter_sizes:
    data = df[df['Meter Size']==size]
    bins = np.linspace(data['volume'].min(), data['volume'].max(), number_of_bins)
    groups = data.groupby(pd.cut(data['volume'], bins))
    height=groups['count'].sum()
    axs[count].bar(bins[:-1],height,width=bins[1],align='edge')
    axs[count].set_title('%.1f" meter'%size)
    count+=1
    print(data['volume'].max())
fig.text(0.5,0.04, "Flow rate (gph)", ha="center", va="center")
fig.text(0.05,0.5, "Volume used (gal)", ha="center", va="center", rotation=90)
fig.show();\

        
#experimental area
#for size in Meter_sizes:o
#    data = df[df['Meter Size']==size]
#    max=data['volume'].max()
#    print(max);\

                 