from dirs import*


year_range=range(2009,2017)
day_range=range(1,362,2)
baselink='ftp://ftp.scp.byu.edu/data/ascat/'
dir=MyDir+'/ASCAT/sigma-0/'

for year in year_range:
    Y1='%02d'%(year-2000)
    ftp = FTP('ftp.scp.byu.edu')
    ftp.login()
    print('Processing data for year '+'%s'%year+' ...')
    folderName = 'data/ascat/'+'%s'%year+'/sir/msfa/NAm/'
    ftp.cwd(folderName)
    DY1=ftp.nlst()      
    DY2=map(int, DY1)
    DY2=[x+4 for x in DY2]   
    DY2=["%03d"%x for x in DY2]
    os.chdir(dir)
    if not(os.path.isdir('%s'%year)):        
        os.mkdir('%s'%year)
    os.chdir('%s'%year)     
    for i in range(len(DY1)):    
        filename='msfa-a-NAm'+Y1+'-'+DY1[i]+'-'+DY2[i]+'.sir.gz'
        link=baselink+'%s'%year+'/sir/msfa/NAm/'+DY1[i]+'/a/'
        if not(os.path.isfile(filename)):
            urllib.urlretrieve(link+filename,filename)
