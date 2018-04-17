#!/opt/local/bin/python
"""

sketch_read_2.py

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
'MYD05_L2.A2015304.1950.006.2015305175415.hdf'
]
# f0=3; f1=5+1
# f0=0; f1=f0+3
f0=6; f1=f0+3+3
    
FILE_NAMES=[
'MOD05_L2.A2015304.1635.006.2015308154932.hdf',
'MYD05_L2.A2015304.1800.006.2015305180731.hdf',
'MYD05_L2.A2015304.1805.006.2015305175319.hdf',
'MYD05_L2.A2015304.1810.006.2015305180241.hdf',
'MYD05_L2.A2015304.1940.006.2015305181743.hdf',
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.1950.006.2015305175415.hdf'
]
f0=0; f1=len(FILE_NAMES)

FILE_NAMES_1=FILE_NAMES[f0:f1]
firstFlag=True

n=len(FILE_NAMES_1)
data=np.zeros((406,270))
data_=np.zeros((n*406,270))
latitude_=np.zeros((n*406,270))
longitude_=np.zeros((n*406,270))
j=0
# for i in range(12):    
for i in range(f0,f1):
    # FILE_NAME = 'MOD05_L2.A2015304.1645.006.2015308154952.hdf'
    FILE_NAME = FILE_NAMES[i]
    DATAFIELD_NAME = 'Water_Vapor_Infrared'

    hdf = SD(FILE_NAME, SDC.READ)
    # ds_dic = hdf.datasets()
    # for idx,sds in enumerate(ds_dic.keys()):
    #     print idx,sds
    # quit()
    ds = hdf.select(DATAFIELD_NAME)
    # print(type(ds))
    
    # data = ds[:,:].astype(np.double)
    ds_dims = [ds.dimensions()[i] for i in ds.dimensions().keys()]
    print("data size:  ",ds_dims)
    # ilo=4000; ihi=5000
    ilo0=0;
    ihi0=406;
    # ihi0=ds_dims[0];
    ilo1=0;
    ihi1=270
    # ihi1=ds_dims[1];
    print("ilo,ihi:0,1: ",ilo0,ihi0,ilo1,ihi1)
    # data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    
    print("loaded: ",FILE_NAME," data: ",data.shape)
    
    # No _FillValue attribute is defined.
    # The value is -9999.9.
    _FillValue = np.min(data)
    data[data == _FillValue] = np.nan
    
    # Handle attributes.
    attrs = ds.attributes(full=1)
    ua=attrs["units"]
    units = ua[0]
    
    # Retrieve the geolocation data.        
    lat = hdf.select('Latitude')
    # latitude = lat[:,:]
    latitude = lat[ilo0:ihi0,ilo1:ihi1]
    lon = hdf.select('Longitude')
    # longitude = lon[:,:]
    longitude = lon[ilo0:ihi0,ilo1:ihi1]

    if(firstFlag):
        units_     = units
        firstFlag  = False
        
    data_      [j*406:(j+1)*406,:] = data[:,:]
    latitude_  [j*406:(j+1)*406,:] = latitude[:,:]
    longitude_ [j*406:(j+1)*406,:] = longitude[:,:]
    j = j+1
        
print("data_: ",data_.shape)
print("lat_ : ",latitude_.shape)
print("lon_ : ",longitude_.shape)
    
# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
m = Basemap(projection='cyl', resolution='h')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
m.scatter(longitude_, latitude_, c=data_, s=1, cmap=plt.cm.jet,
          edgecolors=None, linewidth=0, norm=mpl.colors.LogNorm(),
          latlon=True)
cb = m.colorbar()
cb.set_label(units_)

# basename = os.path.basename(FILE_NAME)
basename = 'outfile'
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

