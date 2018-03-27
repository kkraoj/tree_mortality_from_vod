# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:08:35 2017

@author: kkrao
"""

from dirs import *
os.chdir(Dir_CA)

#store_major=pd.HDFStore('data.h5')
#store = pd.HDFStore(Dir_CA+'/ASCAT_mort.h5')          
#mort=store['mort']
#mort.index=pd.to_datetime(mort.index,format='%Y')
#store_major['mortality_005_grid']=mort
#
#store = pd.HDFStore(Dir_CA+'/mort.h5')          
#mort=store['mort']
#mort.index.name='gridID'
#mort=mort.T
#mort.drop('gridID',inplace=True)
#mort.index=[x[-4:] for x in mort.index] 
#mort.index=pd.to_datetime(mort.index,format='%Y')
#mort.fillna(0,inplace=True)
#mort.index.name = 'FAM_gross'
#store['mort']=mort
#store.close()
#store_major['mortality_025_grid']=mort
#
#store = pd.HDFStore(Dir_CA+'/vodDf.h5')#ascending is 1:30 PM
#VOD_PM=store['vodDf']
#VOD_PM.index.name='gridID'
#VOD_PM=VOD_PM.T
#VOD_PM.drop('gridID',inplace=True)
#VOD_PM.index=[x[:-1] for x in VOD_PM.index] 
#VOD_PM.index=pd.to_datetime(VOD_PM.index,format='%Y%j')
#VOD_PM=VOD_PM[VOD_PM.index.dayofyear!=366]           
#store_major['vod_pm']=VOD_PM
#           
#store = pd.HDFStore(Dir_CA+'/vod_D_Df.h5')
#VOD_AM=store['vod_D_Df']
#VOD_AM=VOD_AM[VOD_AM.index.dayofyear!=366]           
#store_major['vod_am']=VOD_AM
#
#store = pd.HDFStore(Dir_CA+'/LAI.h5')        
#store_major['LAI_005_grid'] = store['LAI_smallgrid']
#store_major['LAI_025_grid'] = store['LAI_grid']
#          
#store = pd.HDFStore(Dir_CA+'/sigma0.h5')
#sigma0=store['sigma0']
#sigma0.index=pd.to_datetime(sigma0.index,format='%Y%j')           
#store_major['vod_005_grid']=sigma0
#
#store = pd.HDFStore(Dir_CA+'/Young_Df.h5') 
#cwd=store['cwd_acc']
#cwd.index.name='gridID'
#cwd=cwd.T
#cwd.drop('gridID',inplace=True)
#cwd.index=pd.to_datetime(cwd.index,format='%Y')
#store_major['cwd']=cwd
#    
#store_major.close()
#store.close()
#
####----------------------------------------------------------------------------
#def seasonal_transform(data,season):
#    start_month=7
#    months_window=3
#    if season=='win':
#        start_month=1
#    data=data.loc[(data.index.month>=start_month) & (data.index.month<start_month+months_window)]
#    return data
#
#store['RWC']=data_anomaly 
#cwd=store['cwd']
#cwd.index.name='cwd'
#store['cwd']=cwd
#     
#mort=store['mortality_025_grid']
#mort.index.name='FAM'
#store['mortality_025_grid']=mort
#
#LAI=store['LAI_025_grid']
#for season in ['sum','win']:
#    df=seasonal_transform(LAI,season)
#    df=df.groupby(df.index.year).mean()
#    df.index=pd.to_datetime(df.index,format='%Y')
#    df.index.name='LAI_%s'%season
#    store['LAI_025_grid_%s'%season]=df
#
####----------------------------------------------------------------------------
#df=store['BPH_025_grid']
#df.index.name='live_tree_density'
#store['BPH_025_grid']=df
#
####----------------------------------------------------------------------------
#import pickle
#os.chdir(Dir_CA)
#nos=370
#with open('grid_to_species.txt', 'rb') as fp:
#    for_type = pickle.load(fp)
#Df=pd.DataFrame(np.full((11,nos),np.NaN), \
#                           index=[pd.to_datetime(year_range,format='%Y')],\
#                                 columns=range(nos))
#for species in ['c','d']:
#    index=[i[0] for i in for_type if i[1]==species]
#    Df.loc[:,Df.columns.intersection(index)]=species
#Df.replace(['c','d'],['evergreen','deciduous'],inplace=True)
#Df.index.name='dominant_leaf_habit'
#store[Df.index.name]=Df
#store.close()
####----------------------------------------------------------------------------
#store=pd.HDFStore(Dir_CA+'/data.h5')
#for field in ['ppt']:
#    for season in ['sum','win']:
#        df=store[field]
#        df=seasonal_transform(df,season)
#        df=df.groupby(df.index.year).sum()
#        df.index=pd.to_datetime(df.index,format='%Y')
#        df.index.name='%s_%s'%(field,season)
#        store[df.index.name]=df
####----------------------------------------------------------------------------
#store=pd.HDFStore(Dir_CA+'/data.h5')
#for field in ['PEVAP','EVP']:
#    for season in ['sum','win']:
#        df=store[field]
#        df=seasonal_transform(df,season)
#        df=df.groupby(df.index.year).sum()
#        df.index=pd.to_datetime(df.index,format='%Y')
#        df.index.name='%s_%s'%(field,season)
#        store[df.index.name]=df
#
####----------------------------------------------------------------------------
#store=pd.HDFStore(Dir_CA+'/data.h5')
#for field in ['vpdmax']:
#    for season in ['sum','win']:
#        df=store[field]
#        df=seasonal_transform(df,season)
#        df=df.groupby(df.index.year).mean()
#        df.index=pd.to_datetime(df.index,format='%Y')
#        df.index.name='%s_%s'%(field,season)
#        store[df.index.name]=df
#store=pd.HDFStore(Dir_CA+'/data.h5')
####----------------------------------------------------------------------------
#for field in ['tmax']:
#    for season in ['sum','win']:
#        df=store[field]
#        df=seasonal_transform(df,season)
#        df=df.groupby(df.index.year).mean()
#        df.index=pd.to_datetime(df.index,format='%Y')
#        df.index.name='%s_%s'%(field,season)
#        store[df.index.name]=df             
#
####----------------------------------------------------------------------------
#subset_store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#to_remove=[
# '/LAI_005_grid',
# '/RWC_005_grid',
# '/RWC_global_norm',
# '/TPA_005_grid',
# '/cwd_005_grid',
# '/mortality_005_grid',
# '/vod_005_grid',
# '/vod_am',
# '/vod_pm',
# '/vod_pm_005_grid',
#]
#for key in subset_store.keys():
#    subset_store[key]=subset_forest_cov(subset_store[key])
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',index_col=3)    
#Df=subset_store['mortality_025_grid']
#Df/=fc.forest_cov
#subset_store['mortality_025_grid']=Df
#            
#subset_store.get_node('mortality_025_grid')._f_rename('mortality_025_grid_tree_cover')
#Df=subset_store['mortality_025_grid_tree_cover']
#Df=Df.loc[Df.index.year>=2009]
#Df_acc=Df.copy()
#for time in Df.index:
#    df=Df.loc[:time]
#    Df_acc.loc[time]=df.sum()
#subset_store['mortality_025_grid']=Df_acc # storing accumulated FAM. Remove after testing
#
#subset_store['mortality_025_grid_accumulated']=Df_acc            
#subset_store['mortality_025_grid']=subset_store['mortality_025_grid_tree_cover']
#

##--------------------------------------------------------
#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#store.remove('mortality_025_grid_tree_cover')


##--------------------------------------------------------
#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#for lag in [1,2]: 
#    Df=store['RWC']
#    Df_shift=Df.shift(lag)
#    Df_shift.index.name=Df_shift.index.name+'_lag_%d'%lag
#    store[Df_shift.index.name]=Df_shift  
#store.remove('/RWC_lag_1_lag_2')
##--------------------------------------------------------
###changing denominator to live - dead
#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#Df=store['mortality_025_grid']
#Df=subset_forest_cov(Df) # removing cells with <0.7 tree cover
#dr=Df.shift(1)
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',index_col=3)    
#Df=Df/(fc.forest_cov-dr)
#subset_store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#Df.index.name='FAM'
#subset_store['mortality_025_grid_live_minus_dead']=Df

             #store=pd.HDFStore(Dir_CA+'/data.h5')
#Df=store['mortality_025_grid']
#Df=subset_forest_cov(Df) # removing cells with <0.7 tree cover
#dr=Df.shift(1)
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',index_col=3)    
#Df=Df/(fc.forest_cov-dr)
#subset_store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#Df.index.name='FAM'
#subset_store['mortality_025_grid_live_minus_dead']=Df
            
### shifting mortality by -1 years. so that all predictor variables
### appear to be lagged by one year with respect to it. 

#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#Df = store['mortality_025_grid']
#Df=Df.shift(-1)
#store['mortality_025_grid_lead']=Df
     
###----------------------------------------------------- 
## updated RWC with 2016 and 2017 years from LPDR v2
#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#Df = store['vod_pm']
#RWC=RWC(Df)
#RWC.index.name='RWC'
#store['RWC_extended']=RWC

####---------------------------------------------------
#extending all other dfs to 2016
#store=pd.HDFStore(Dir_CA+'/data_subset.h5')
#input_sources=['mortality_025_grid','LAI_025_grid_sum',\
#'LAI_025_grid_win','RWC_extended', 'aspect_mean', 'aspect_std', 'canopy_height',\
# 'elevation_mean','elevation_std',\
# 'forest_cover','ppt_sum','ppt_win','tmax_sum','tmax_win',\
# 'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
#'PEVAP_sum','EVP_win','PEVAP_win']
##
#for source in input_sources:
#    max_year=max(store[source].index.year)
#    print('for %s, max year = %d'%(source,max_year))
            

#to_extend_sources=['aspect_mean', 'aspect_std', 'canopy_height','elevation_mean',\
#'elevation_std','forest_cover']
#for source in to_extend_sources:
#    Df=store[source]
#    Df.loc[pd.to_datetime('2016-01-01')]=Df.loc[pd.to_datetime('2015-01-01')]
#    store[source]=Df
         
####rather than extending to_extend_sources, I mistakenly extended input sources \
#making all dfs in inpur sources to have 2015 and 2016 rows equal. 
## fixing the issue here
#super_store=pd.HDFStore(Dir_CA+'/data.h5')
##copying from original h5 file
#source_with_missing_2016=['mortality_025_grid','LAI_025_grid_sum',\
#'LAI_025_grid_win','ppt_sum','ppt_win','tmax_sum','tmax_win',\
# 'tmean_sum','tmean_win','vpdmax_sum','vpdmax_win','EVP_sum',\
#'PEVAP_sum','EVP_win','PEVAP_win']
#
#for source in source_with_missing_2016:
#    store[source]=super_store[source]

### subsetting to grids with forest cover > 70%

#for source in source_with_missing_2016:
#    store[source]=subset_forest_cov(store[source])
## make FAM = dead tree area / area of tree cover rather than area of grid cell
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',index_col=3)    
#Df=store['mortality_025_grid']
#Df/=fc.forest_cov

#store['mortality_025_grid']=Df

#for source in source_with_missing_2016:
#    print('%s file index name: %s'%(source, store[source].index.name))

#Df=store['mortality_025_grid']
#Df.index.name='FAM'
#store['mortality_025_grid']=Df
## ^fixed all peoblems with data_subset.h5 3/5/3018

####---------------------------------------------------
### subsetting data based GlobCover 2009 into data_subset_GC.h5

#new_store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
### clean up new_store
#to_keep=['/BPH_025_grid',
# '/EVP',
# '/EVP_sum',
# '/EVP_win',
# '/LAI_025_grid',
# '/LAI_025_grid_sum',
# '/LAI_025_grid_win',
# '/PEVAP',
# '/PEVAP_sum',
# '/PEVAP_win',
# '/RWC',
# '/TPA_025_grid',
# '/aspect_mean',
# '/aspect_std',
# '/canopy_height',
# '/cwd',
# '/dominant_leaf_habit',
# '/elevation_mean',
# '/elevation_std',
# '/forest_cover', # to be changed for gc. Type yes after changing. Changed? Y
# '/mortality_025_grid',
# '/mortality_deciduous_025_grid', # to be changed for gc. Type yes after changing. Changed? No
# '/mortality_evergreen_025_grid', # to be changed for gc. Type yes after changing. Changed? No
# '/ppt',
# '/ppt_sum',
# '/ppt_win',
# '/tmax',
# '/tmax_sum',
# '/tmax_win',
# '/tmean',
# '/tmean_sum',
# '/tmean_win',
# '/vpdmax',
# '/vpdmax_sum',
# '/vpdmax_win',
# '/vsm',
# '/vsm_sum',
# '/vsm_win']
#
#for dataset in new_store.keys():
#    if not(dataset in to_keep):
#        new_store.remove(dataset)

#new_store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#old_store = pd.HDFStore(Dir_CA+'/mort.h5')
#
#Df = old_store['mort']
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',sheetname='GC',index_col=3)    
#Df/=fc.gc_fc
#Df=subset_forest_cov(Df,landcover = 'GC_subset')
#Df.index.name='FAM'
#new_store['mortality_025_grid']=Df
##sds
#
#### extend time independent variables till 2016
#to_extend_sources=['aspect_mean', 'aspect_std', 'canopy_height','elevation_mean',\
#'elevation_std','forest_cover']
#for source in to_extend_sources:
#    Df=new_store[source]
#    Df.loc[pd.to_datetime('2016-01-01')]=Df.loc[pd.to_datetime('2015-01-01')]
#    new_store[source]=Df
#
#### subset gridIDs to forestcover >=0.70 as per GLOBCOVER2009 data
#
#for source in new_store.keys():
#    new_store[source]=subset_forest_cov(new_store[source],landcover = 'GC_subset')
#
###update to new forest cover based on globcover
#Df=new_store['forest_cover']
#fc=pd.read_excel(MyDir+'/Forest/forest_cover.xlsx',sheetname='GC_subset',index_col=3)  
#for index in Df.index:
#    Df.loc[index]=fc.gc_fc
         
## add vod_pm into data_dubset_Gc because it is required to plot RWC definition
#new_store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#old_store = pd.HDFStore(Dir_CA+'/data.h5')
#
#Df = old_store['vod_pm']
#Df=remove_vod_affected(Df)
#Df=subset_forest_cov(Df,landcover = 'GC_subset')
#new_store['vod_pm']=Df
#=====================================================
## adding gridID as a dataframe in data_seubset_GC only
#new_store = pd.HDFStore(Dir_CA+'/data_subset_GC.h5')
#Df=new_store['mortality_025_grid']
#Df=Df.astype(int)
#Df.loc[:,:]=Df.columns
#Df.index.name='location'
#new_store[Df.index.name]=Df
#=====================================================