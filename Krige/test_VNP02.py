
import numpy

import Krige
# from Krige import core
from Krige.DataField import DataField
from Krige import fig_generator

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

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

    mn = numpy.amin(df.data)
    mx = numpy.amax(df.data)
    print('mnmx: ',mn,mx)

    # Display results
    # fig_gen = fig_generator(1,1)
    df.scatterplot(title='vnp02',colorbar=True,marker_size=7,vmin=mn,vmax=mx,sample=0.01)
    plt.show()

    