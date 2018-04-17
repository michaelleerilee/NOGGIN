#!/opt/local/bin/python
"""

sketch_read_1.py

Sketch for reading, registering, and visualizing 05_L2 and 09GA data.

Started out as example code (for 2B31) from the HDF Group.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/NOGGIN.

"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC

# FILE_NAME = '2B31.20140108.91989.7.HDF'
# FILE_NAME = '2B31.20091203.68644.7.HDF'
# DATAFIELD_NAME = 'rrSurf'
# DATAFIELD_NAME = 'prSurf'

FILE_NAMES=[
'MOD05_L2.A2015304.1455.006.2015308193119.hdf',
'MOD05_L2.A2015304.1500.006.2015308193124.hdf',
'MOD05_L2.A2015304.1505.006.2015308195111.hdf',
'MOD05_L2.A2015304.1635.006.2015308154932.hdf',
'MOD05_L2.A2015304.1640.006.2015308155029.hdf',
'MOD05_L2.A2015304.1645.006.2015308154952.hdf',
'MYD05_L2.A2015304.1800.006.2015305180731.hdf',
'MYD05_L2.A2015304.1805.006.2015305175319.hdf',
'MYD05_L2.A2015304.1810.006.2015305180241.hdf',
'MYD05_L2.A2015304.1940.006.2015305181743.hdf',
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.1950.006.2015305175415.hdf',
'MOD09GA.A2015304.h10v06.006.2015318014330.hdf'
]

# FILE_NAME = 'MOD05_L2.A2015304.1645.006.2015308154952.hdf'
# FILE_NAME = FILE_NAMES[11]
# DATAFIELD_NAME = 'Water_Vapor_Infrared'

FILE_NAME = FILE_NAMES[12]
DATAFIELD_NAME = 'sur_refl_b05_1'
# DATAFIELD_NAME = '500m Surface Reflectance Band 5 - first layer'

print('filename: ',FILE_NAME)

hdf = SD(FILE_NAME, SDC.READ)
ds_dic = hdf.datasets()
for idx,sds in enumerate(ds_dic.keys()):
    print idx,sds
# quit()
print 100
ds = hdf.select(DATAFIELD_NAME)
print 200
# print(type(ds))

# data = ds[:,:].astype(np.double)
ds_dims = [ds.dimensions()[i] for i in ds.dimensions().keys()]
print("data size: ",ds_dims)
# ilo=4000; ihi=5000
ilo0=0;
ihi0=ds_dims[0]-1;

data = ds[ilo0:ihi0,:].astype(np.double)

print("loaded: ",FILE_NAME," data: ",data.shape)

# No _FillValue attribute is defined.
# The value is -9999.9.
_FillValue = np.min(data)
data[data == _FillValue] = np.nan

print 1000
# Handle attributes.
attrs = ds.attributes(full=1)
ua=attrs["units"]
units = ua[0]
print 1100

# Retrieve the geolocation data.        
lat = hdf.select('Latitude')
print 1200
# latitude = lat[:,:]
latitude = lat[ilo0:ihi0,:]
lon = hdf.select('Longitude')
print 1300
# longitude = lon[:,:]
longitude = lon[ilo0:ihi0,:]

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
m = Basemap(projection='cyl', resolution='h')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data, s=0.1, cmap=plt.cm.jet,
m.scatter(longitude, latitude, c=data, s=1, cmap=plt.cm.jet,
          edgecolors=None, linewidth=0)
cb = m.colorbar()
cb.set_label(units)

basename = os.path.basename(FILE_NAME)
plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
fig = plt.gcf()
plt.show()
    
pngfile = "{0}.py.png".format(basename)
fig.savefig(pngfile)

"""
Note
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mpl_toolkits/basemap/__init__.py:3222: MatplotlibDeprecationWarning: The ishold function was deprecated in version 2.0.
  b = ax.ishold()
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mpl_toolkits/basemap/__init__.py:3231: MatplotlibDeprecationWarning: axes.hold is deprecated.
    See the API Changes document (http://matplotlib.org/api/api_changes.html)
    for more details.
  ax.hold(b)
"""

