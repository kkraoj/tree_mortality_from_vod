import urllib
from dirs import*


data=['sph','pr','tmmn','tmmx','pdsi','pet']
year_range=range(2005,2017)
link='http://northwestknowledge.net/metdata/data/'
for j in range(len(data)):
    file=data[j]
    os.chdir(MyDir+'/PET')
    if not(os.path.isdir(MyDir+'/PET/'+file)):        
        os.mkdir(file)
    os.chdir(MyDir+'/PET/'+file)
    for i in year_range:
        filename=file+'_%d.nc'%i
        if not(os.path.isfile(filename)):
            urllib.urlretrieve(link+filename,filename)

                   