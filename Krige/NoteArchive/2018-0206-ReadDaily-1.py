#!/opt/local/bin/python

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *

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
'MOD05_L2.A2015304.1815.006.2015308155414.hdf',
'MYD05_L2.A2015304.1800.006.2015305180731.hdf',
'MYD05_L2.A2015304.1805.006.2015305175319.hdf',
'MYD05_L2.A2015304.1810.006.2015305180241.hdf',
'MYD05_L2.A2015304.1940.006.2015305181743.hdf',
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.1950.006.2015305175415.hdf',
'MYD05_L2.A2015304.2125.006.2015305175459.hdf'
]
    
FILE_NAMES=[
'MOD05_L2.A2015304.1815.006.2015308155414.hdf',
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.1950.006.2015305175415.hdf',
'MYD05_L2.A2015304.2125.006.2015305175459.hdf'
]

FILE_NAMES=[
'MYD05_L2.A2015304.2125.006.2015305175459.hdf'
]
    
FILE_NAMES=[    
'MYD08_D3.A2015304.006.2015305223906.hdf'
]

f0=0; f1=len(FILE_NAMES)

FILE_NAMES_1=FILE_NAMES[f0:f1]
firstFlag=True

nlat=180
latIndex=np.linspace(-90,90,nlat)

nlon=360
lonIndex=np.linspace(-180,180,nlon)

lons, lats = np.meshgrid(lonIndex, latIndex, sparse=False)
print('lats.shape: ',lats.shape)
print('lons.shape: ',lons.shape)

n=len(FILE_NAMES_1)
data=np.zeros((nlat,nlon))
dataFull=np.zeros((nlat,nlon))
annulus=np.zeros((nlat,nlon))

dataFull_=np.zeros((n*nlat,nlon))
data_=np.zeros((n*nlat,nlon))
annulus_=np.zeros((n*nlat,nlon))

mock_fill_value = 0.0000111

latitude_=np.zeros((n*nlat,nlon))
longitude_=np.zeros((n*nlat,nlon))
j=0
# for i in range(12):    
for i in range(f0,f1):
    print('i: ',i)
    # FILE_NAME = 'MOD05_L2.A2015304.1645.006.2015308154952.hdf'
    FILE_NAME = FILE_NAMES[i]
    DATAFIELD_NAME = 'Atmospheric_Water_Vapor_Mean'

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
    ihi0=nlat;
    # ihi0=ds_dims[0];
    ilo1=0;
    ihi1=nlon
    # ihi1=ds_dims[1];
    print("ilo,ihi:0,1: ",ilo0,ihi0,ilo1,ihi1)
    # data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    
    print("loaded: ",FILE_NAME," data: ",data.shape)
    
    # Handle attributes.
    attrs = ds.attributes(full=1)
    ua=attrs["units"]
    units = ua[0]

    # No _FillValue attribute is defined.
    # The value is -9999.9.
    _FillValue = np.min(data)

    scale_factor = attrs["scale_factor"][0]
    add_offset   = attrs["add_offset"][0]
    # Scale the data.
    data[data != _FillValue] = scale_factor * ( data[data != _FillValue] - add_offset )
    
    # the right thing to do
    # data[data != _FillValue] = 10.0*float(j+1)
    data[data == _FillValue] = mock_fill_value
    
    # data[data == _FillValue] = np.nan
    # data[data == _FillValue] = 0


    
##    # Retrieve the geolocation data.        
##    lat = hdf.select('Latitude')
##    # latitude = lat[:,:]
##    latitude = lat[ilo0:ihi0,ilo1:ihi1]
##    lon = hdf.select('Longitude')
##    # longitude = lon[:,:]
##    longitude = lon[ilo0:ihi0,ilo1:ihi1]

    latitude  = lats[ilo0:ihi0,ilo1:ihi1]
    longitude = lons[ilo0:ihi0,ilo1:ihi1]

    rad_from_deg = pi/180.0
    
    # lat0 =  25.0 * rad_from_deg
    # lon0 =-120.0 * rad_from_deg

    lat0 =  17.75 * rad_from_deg
    lon0 =-112.75 * rad_from_deg
    
    # rad0 =   0.75 * rad_from_deg
    rad0 =   1.5 * rad_from_deg
    # rad0 =   2 * rad_from_deg
    # rad0 =   0.7 * rad_from_deg
    # rad0 =   0.25 * rad_from_deg

    # radOuter = 4*rad0
    radOuter = 2*rad0
    # radOuter = 1.25*rad0

    # # High contrast case
    # lat0 =  17.75 * rad_from_deg
    # lon0 =-112.75 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # West of Baja
    lat0 =  28.0 * rad_from_deg
    lon0 =-120 * rad_from_deg
    rad0 =   1.5 * rad_from_deg
    radOuter = 2*rad0
    
    x0 = cos( lat0 )
    y0 = x0 * sin ( lon0 )
    x0 = x0 * cos ( lon0 )
    z0 = sin( lat0 )

    # In data coords, not data_.
    latr = rad_from_deg * latitude
    lonr = rad_from_deg * longitude
    x1 = np.cos( latr )
    y1 = x1 * np.sin ( lonr )
    x1 = x1 * np.cos ( lonr )
    z1 = np.sin( latr )
    dot = x1*x0 + y1*y0 + z1*z0
    print("dot mnmx: ",np.nanmin(dot),np.nanmax(dot))

    # data[np.where(longitude  < -120.0)] = 0.01
    # data[np.where(longitude >= -120.0)] = 1.0
    dataFull = np.copy(data)
    data[ dot > cos(rad0) ] = np.nan

    # annulus by default 0.
    annulus[np.where((dot >= cos(radOuter)) & (dot < cos(rad0))) ] = 1
    annulus[np.where((dot < cos(radOuter))) ] = 2

    if(firstFlag):
        units_     = units
        firstFlag  = False
        
    data_      [j*nlat:(j+1)*nlat,:] = data[:,:]
    dataFull_      [j*nlat:(j+1)*nlat,:] = dataFull[:,:]
    annulus_   [j*nlat:(j+1)*nlat,:] = annulus[:,:]
    # data_      [j*nlat:(j+1)*nlat,:] = float(j+1)*np.ones(data.shape)
    longitude_ [j*nlat:(j+1)*nlat,:] = longitude[:,:]
    latitude_  [j*nlat:(j+1)*nlat,:] =latitude[:,:] 
    j = j+1


print("data_ mnmx: ",np.nanmin(data_),np.nanmax(data_))
print("data_.shape: ",data_.shape)
print("lat_.shape : ",latitude_.shape)
print("lon_.shape : ",longitude_.shape)
print("latitude_.shape : ",latitude_.shape)
print("longitude_.shape : ",longitude_.shape)
print("latitude_ mnmx: ",np.nanmin(latitude),np.nanmax(latitude))
print("longitude_ mnmx: ",np.nanmin(longitude_),np.nanmax(longitude))


# marker_size = 4
marker_size = 16
# m_alpha = 1.0
m_alpha = 0.5

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
m = Basemap(projection='cyl', resolution='h')
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          latlon=True, alpha=m_alpha)
#  norm=mpl.colors.LogNorm(),
cb = m.colorbar()
cb.set_label(units_)
cb.set_alpha(1)
cb.draw_all()

# basename = os.path.basename(FILE_NAME)
basename = 'outfile'
plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
fig = plt.gcf()
plt.show()
    
# pngfile = "{0}.py.png".format(basename)
# fig.savefig(pngfile)

"""
Note
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mpl_toolkits/basemap/__init__.py:3222: MatplotlibDeprecationWarning: The ishold function was deprecated in version 2.0.
  b = ax.ishold()
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mpl_toolkits/basemap/__init__.py:3231: MatplotlibDeprecationWarning: axes.hold is deprecated.
    See the API Changes document (http://matplotlib.org/api/api_changes.html)
    for more details.
  ax.hold(b)
"""

# data_src=np.zeros((data_len,3))
# data_src[:,0] = ...

print("Starting kriging...")
# data_x = np.ravel(longitude_[data_ != np.nan ])
# data_y = np.ravel(latitude_[data_ != np.nan ] )
# data_z = np.ravel(data_[data_!= np.nan])

data_x = np.ravel(longitude_[ annulus_ == 1 ])
data_y = np.ravel(latitude_ [ annulus_ == 1 ])
data_z = np.ravel(data_     [ annulus_ == 1 ])

print("data_z.shape: ",data_z.shape)
print("data_z.mnmx:  ",np.min(data_z),np.max(data_z))

from scipy.spatial import ConvexHull

xy0 = np.zeros((data_z.shape[0],2))
xy0[:,0]=data_x
xy0[:,1]=data_y
hull=ConvexHull(xy0)

from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt

data_x1 = data_x
# data_x1 = np.remainder(data_x+360,360)

OK = OrdinaryKriging( data_x1, data_y, data_z, variogram_model='linear',
                         verbose=False, enable_plotting=True
                          )

#? gridx = np.ravel(longitude_[data_ == np.nan ])
#? gridy = np.ravel(latitude_[data_ == np.nan ] )
gridx = np.ravel(longitude_[ annulus_ == 0 ])
gridy = np.ravel(latitude_[  annulus_ == 0 ])

xy1 = np.zeros((gridx.shape[0],2))
xy1[:,0]=gridx
xy1[:,1]=gridy
grid_hull=ConvexHull(xy1)

print('gridx.shape: ',gridx.shape)
print('gridy.shape: ',gridx.shape)
print('data_ isnan: ',np.count_nonzero(~np.isnan(data_)))

gridx1 = gridx
# gridx1 = np.remainder(gridx+360,360)

z, ss = OK.execute('points',gridx1,gridy)
print('z.shape:     ',z.shape)
print("z.mnmx:      ",np.min(z),np.max(z))

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
m = Basemap(projection='cyl', resolution='h')
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,

vmin0=np.nanmin(data_); vmax0=np.nanmax(data_)
# vmin1=0.5*np.nanmin(z); vmax1=1.5*np.nanmax(z)
vmin1=0.9*np.nanmin(z); vmax1=1.1*np.nanmax(z)
vmin=vmin1
vmax=vmax1

# m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=plt.cm.jet,
#           edgecolors=None, linewidth=0,
#           latlon=True, alpha=m_alpha, norm=mpl.colors.Normalize(vmin,vmax),
#           vmin=vmin, vmax=vmax
#               )

# vmin=np.nanmin(data_); vmax=np.nanmax(data_)
#
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          latlon=True, alpha=m_alpha, norm=mpl.colors.Normalize(vmin,vmax),
          vmin=vmin, vmax=vmax
          )
## norm=mpl.colors.LogNorm(),

# m.scatter(gridx,gridy,c=z, s=marker_size, cmap=plt.cm.jet,
m.scatter(gridx,gridy,c=z, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          norm=mpl.colors.Normalize(vmin,vmax),
          vmin=vmin, vmax=vmax
          )

#2          latlon=True, alpha=m_alpha
#2              , norm=mpl.colors.Normalize(vmin,vmax),
#2          vmin=vmin, vmax=vmax
#norm=mpl.colors.LogNorm(),

# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
from matplotlib.patches import Polygon
def draw_screen_poly( lons, lats, m ):
    x, y = m( lons, lats )
    plt.gca().add_patch(Polygon( zip(x,y), facecolor='black', alpha=0.8, fill=False ))

# def draw_simplex(lons, lats, hull, 

draw_screen_poly( data_x[hull.vertices], data_y[hull.vertices], m )
draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

cb = m.colorbar()
cb.set_label(units_)
cb.set_alpha(1)
cb.draw_all()

# basename = os.path.basename(FILE_NAME)
basename = 'outfile'
plt.title('{0}\n{1}-OrdKrig-'.format(basename, DATAFIELD_NAME))
fig = plt.gcf()
plt.show()


#########

m = Basemap(projection='cyl', resolution='h')
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])

# vmin=np.nanmin(data_); vmax=np.nanmax(data_)
#
# m.scatter(longitude_, latitude_, c=dataFull_, s=marker_size, cmap=plt.cm.jet,
m.scatter(longitude_, latitude_, c=dataFull_, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          latlon=True, alpha=m_alpha, norm=mpl.colors.Normalize(vmin,vmax),
          vmin=vmin, vmax=vmax
          )
## norm=mpl.colors.LogNorm(),

draw_screen_poly( data_x[hull.vertices], data_y[hull.vertices], m )
draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

cb = m.colorbar()
cb.set_label(units_)
cb.set_alpha(1)
cb.draw_all()

# basename = os.path.basename(FILE_NAME)
basename = 'outfile'
plt.title('{0}\n{1}-FULL-'.format(basename, DATAFIELD_NAME))
fig = plt.gcf()
plt.show()
