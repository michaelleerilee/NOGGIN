
import Krige
# from Krige import core
from Krige.DataField import DataField

# /Users/mrilee/Dropbox/data/VIIRS/VNP03IMG.A2021182.0000.001.2021182063427.nc

datafile="VNP02IMG.A2021182.0000.001.2021182064359.nc"
geofile ="VNP03IMG.A2021182.0000.001.2021182063427.nc"
srcdir="/Users/mrilee/Dropbox/data/VIIRS/"


if __name__ == "__main__":
    print('hello world')
    df = DataField(
        datafilename=datafile
        ,datafieldname=None
        ,geofile=geofile
        ,srcdirname=srcdir
    )



    
