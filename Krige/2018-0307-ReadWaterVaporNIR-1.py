#!/opt/local/bin/python

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *
from noggin import *

FILE_NAMES=[
('MYD05_L2.A2015304.2125.006.2015305175459.hdf','MYD03.A2015304.2125.006.2015305161909.hdf')
]

MODIS_DIR="MODIS/"

f0=0; f1=len(FILE_NAMES)

FILE_NAMES_1=FILE_NAMES[f0:f1]
firstFlag=True

n=len(FILE_NAMES_1)

FILE_TUPLE=FILE_NAMES[0]

GEOLOC_NAME = MODIS_DIR+FILE_TUPLE[1]
geo = SD(GEOLOC_NAME, SDC.READ)
lat = geo.select('Latitude')
latitude = lat[:,:]
lon = geo.select('Longitude')
longitude = lon[:,:]

FILE_NAME = MODIS_DIR+FILE_TUPLE[0]
print(' loading ',FILE_NAME )
hdf = SD(FILE_NAME, SDC.READ)
DATAFIELD_NAME = 'Water_Vapor_Near_Infrared'
# Need to be careful with georeferencing
#    DATAFIELD_NAME = 'Water_Vapor_Infrared'
#    key_units="units"

ds = hdf.select(DATAFIELD_NAME)
# print(type(ds))
data = ds[:,:].astype(np.double)
ds_dims = [ds.dimensions()[i] for i in ds.dimensions().keys()]
print('ds_dims: ',ds_dims)

attrs        = ds.attributes(full=1)
lna          = attrs["long_name"]
long_name    = lna[0]
aoa          = attrs["add_offset"]
add_offset   = aoa[0]
fva          = attrs["add_offset"]
_FillValue   = fva[0]
sfa          = attrs["scale_factor"]
scale_factor = sfa[0]
vra          = attrs["valid_range"]
valid_min    = vra[0][0]
valid_max    = vra[0][1]
ua           = attrs["unit"]
units        = ua[0]

invalid = np.logical_or(data > valid_max, data < valid_min)
invalid = np.logical_or(invalid,data == _FillValue)

data[invalid] = np.nan
data = (data - add_offset) * scale_factor
data = np.ma.masked_array(data, np.isnan(data))

lat_m = np.nanmean(latitude)
lon_m = np.nanmean(longitude)
# Render the plot in a lambert equal area projection.
m = Basemap(projection='laea', resolution='l', lat_ts=65,
            lat_0=lat_m, lon_0=lon_m,
            width=3000000,height=2500000)
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(50., 91., 10.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180, 181., 30), labels=[0, 0, 0, 1])
# m.scatter(longitude, latitude, c=data, latlon=True)
m.pcolormesh(longitude, latitude, data, latlon=True)
# m.pcolor(longitude, latitude, data, latlon=True)
cb=m.colorbar()
cb.set_label(units, fontsize=8)

basename = os.path.basename(FILE_NAME)
plt.title('{0}\n{1}'.format(basename, long_name))
fig = plt.gcf()
# pngfile = "{0}.py.png".format(basename)
# fig.savefig(pngfile)
plt.show()
print('done')

