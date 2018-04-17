#!/opt/local/bin/python

# (find-file-other-frame "2018-0222-AdaptiveSketchNotes-1.org")

import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
from math import *
from noggin import *

from MODIS_DataField import MODIS_DataField

# recalculate = False

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
'MYD05_L2.A2015304.1945.006.2015305180304.hdf',
'MYD05_L2.A2015304.2125.006.2015305175459.hdf'
]
MODIS_DIR="MODIS/"

FILE_NAMES=[
    'MYD05_L2.A2015304.1945.061.2018054025242.hdf',
    'MYD05_L2.A2015304.2125.061.2018054031117.hdf'
]
MODIS_DIR="MODIS-61/"

f0=0; f1=len(FILE_NAMES)

FILE_NAMES_1=FILE_NAMES[f0:f1]
firstFlag=True

n=len(FILE_NAMES_1)

# Consider Water_Vapor_Infrared
# ~ 406 or 408
n_along_swath  = 408
# ~ 270
n_across_swath = 270

# Consider Water_Vapor_Near_Infrared
# n_along_swath  = 406*5
# n_across_swath = 1354

data=np.zeros((n_along_swath,n_across_swath))
dataFull=np.zeros((n_along_swath,n_across_swath))
annulus=np.zeros((n_along_swath,n_across_swath))

dataFull_=np.zeros((n*n_along_swath,n_across_swath))
data_=np.zeros((n*n_along_swath,n_across_swath))
annulus_=np.zeros((n*n_along_swath,n_across_swath))

mock_fill_value = 0.0000111

latitude_=np.zeros((n*n_along_swath,n_across_swath))
longitude_=np.zeros((n*n_along_swath,n_across_swath))
j=0
# for i in range(12):    
for i in range(f0,f1):
    print('i: ',i)
    # FILE_NAME = 'MOD05_L2.A2015304.1645.006.2015308154952.hdf'
    FILE_NAME = MODIS_DIR+FILE_NAMES[i]
    print(' loading ',FILE_NAME )

#    DATAFIELD_NAME = 'Water_Vapor_Near_Infrared'
#    key_units="unit"
# Need to be careful with georeferencing
    
    DATAFIELD_NAME = 'Water_Vapor_Infrared'
    key_units="units"

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
    ihi0=n_along_swath;
    # ihi0=ds_dims[0];
    ilo1=0;
    ihi1=n_across_swath
    # ihi1=ds_dims[1];
    print("ilo,ihi:0,1: ",ilo0,ihi0,ilo1,ihi1)
    # data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    data = ds[ilo0:ihi0,ilo1:ihi1].astype(np.double)
    
    print("loaded: ",FILE_NAME," data: ",data.shape)
    
    # No _FillValue attribute is defined.
    # The value is -9999.9.
    _FillValue = np.min(data)

    # Handle attributes.
    attrs = ds.attributes(full=1)
    ua=attrs[key_units]
    units = ua[0]

    scale_factor = attrs["scale_factor"][0]
    add_offset   = attrs["add_offset"][0]
    # Scale the data.
    data[data != _FillValue] = scale_factor * ( data[data != _FillValue] - add_offset )
    
    # the right thing to do
    # data[data != _FillValue] = 10.0*float(j+1)
    # data[data == _FillValue] = mock_fill_value
    
    # data[data == _FillValue] = np.nan
    # data[data == _FillValue] = 0


    # For Water_Vapor_Infrared
    # Retrieve the geolocation data.        
    lat = hdf.select('Latitude')
    # latitude = lat[:,:]
    latitude = lat[ilo0:ihi0,ilo1:ihi1]
    lon = hdf.select('Longitude')
    # longitude = lon[:,:]
    longitude = lon[ilo0:ihi0,ilo1:ihi1]
    
    rad_from_deg = pi/180.0

    # Initial testing down in the corner
    # lat0 =  25.0 * rad_from_deg
    # lon0 =-120.0 * rad_from_deg
    lat0 =  17.75 * rad_from_deg
    lon0 =-112.75 * rad_from_deg
    # rad0 =   0.75 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    rad0 =   2 * rad_from_deg
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

    # West of Baja 1.5 degrees
    lat0 =  28.0 * rad_from_deg
    lon0 =-120 * rad_from_deg
    rad0 =   1.5 * rad_from_deg
    radOuter = 2*rad0

    # West of Baja 0.25 degrees
    # lat0 =  28.0 * rad_from_deg
    # lon0 =-120 * rad_from_deg
    # rad0 =   0.25 * rad_from_deg
    # radOuter = 2*rad0

    #++ West of Baja ## Very good
    # lat0 =  28.0 * rad_from_deg
    # lon0 =-120 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # nice
    # lat0 =  20.0 * rad_from_deg
    # lon0 =-115 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # nice - A bite out of the edge
    # lat0 =  20.0 * rad_from_deg
    # lon0 =-110 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # Nice cut in the south
    # lat0 =  18.0 * rad_from_deg
    # lon0 =-113 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # 
    # lat0 =  20.0 * rad_from_deg
    # lon0 =-114 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0

    # 
    # lat0 =  20.0 * rad_from_deg
    # lon0 =-115 * rad_from_deg
    # rad0 =   1.5 * rad_from_deg
    # radOuter = 2*rad0
    
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

    # Cut the hole
    # data[ dot > cos(rad0) ] = np.nan

    # annulus by default 0.
    annulus[np.where((dot >= cos(radOuter)) & (dot < cos(rad0))) ] = 1
    annulus[np.where((dot < cos(radOuter))) ] = 2
    # Accept all data.
    annulus[:,:]=1 

    if(firstFlag):
        units_     = units
        firstFlag  = False
        
    data_      [j*n_along_swath:(j+1)*n_along_swath,:] = data[:,:]
    dataFull_      [j*n_along_swath:(j+1)*n_along_swath,:] = dataFull[:,:]
    annulus_   [j*n_along_swath:(j+1)*n_along_swath,:] = annulus[:,:]
    # data_      [j*n_along_swath:(j+1)*n_along_swath,:] = float(j+1)*np.ones(data.shape)
    latitude_  [j*n_along_swath:(j+1)*n_along_swath,:] = latitude[:,:]
    longitude_ [j*n_along_swath:(j+1)*n_along_swath,:] = longitude[:,:]
    j = j+1


print("data_ mnmx: ",np.nanmin(data_),np.nanmax(data_))
print("data_.shape: ",data_.shape)
print("lat_.shape : ",latitude_.shape)
print("lon_.shape : ",longitude_.shape)
print("latitude_.shape : ",latitude_.shape)
print("longitude_.shape : ",longitude_.shape)
print("latitude_ mnmx: ",np.nanmin(latitude),np.nanmax(latitude))
print("longitude_ mnmx: ",np.nanmin(longitude_),np.nanmax(longitude))

##### KRIGING #####
from scipy.spatial import ConvexHull

data_orig = np.copy(data_)
# data_=np.log(data_)
#data_=np.log10(data_)

idx_fill = data_ != _FillValue

# The source grid
src_x = np.ravel(longitude_[ (annulus_ != 0) & idx_fill ])
src_y = np.ravel(latitude_ [ (annulus_ != 0) & idx_fill ])
src_z = np.ravel(data_     [ (annulus_ != 0) & idx_fill ]);
# print("data_z.shape: ",data_z.shape)
# print("data_z.mnmx:  ",np.min(data_z),np.max(data_z))

# The target grid -- the cut
gridx  = np.ravel(longitude_[ annulus_ == 0 ])
gridy  = np.ravel(latitude_ [ annulus_ == 0 ])

# The target grid -- rectangle
gridxy = np.meshgrid(np.arange(-120.0,-105.0,0.25),np.arange(  20.0,  25.0,0.25))
gridx = np.ravel(gridxy[0])
gridy = np.ravel(gridxy[1])

# Target results
gridz  = np.zeros(gridx.shape)
gridss = np.zeros(gridx.shape)

# nlags=16
nlags=12

# custom_args = [ \
#               0.04,3.3,0.0,\
#               0.04/pow(6.0,8.0)\
#               ]
# custom_args = [0.35,5.0,0.00,1000.0]
# def custom_vg(params,dist):
#         return \
#           (float(params[0]) - float(params[2]))\
#           *(1 - np.exp(-dist/(float(params[1])/3.0)))\
#           *(1 - np.exp(-(float(params[3])/3.0)/(dist+0.000001)))\
#           +float(params[2])

# custom_args = [0.35,5.0,0.00,1000.0]
# def custom_vg(params,dist):
#         return \
#           (float(params[0]) - float(params[2]))\
#           *(1 - np.exp(-dist/(float(params[1])/3.0)))\
#           *(1 - np.exp(-(float(params[3])/3.0)/(dist+0.000001)))\
#           +float(params[2])

# custom_args = [exp(2.6), 1.0/5.0]
# def custom_vg(params,dist):
#     sill    = np.float(params[0])
#     falloff = np.float(params[1])
#     # peak    = 1.0/falloff
#     d       = dist
#     return \
#       np.log(falloff*sill*d*np.exp(1.0-falloff*d))

# custom_args = [2.6, 1.0/5.0, 0.75]
custom_args = None
# A gamma-rayleigh distribution
def custom_vg(params,dist):
    sill    = np.float(params[0])
    falloff = np.float(params[1])
    beta    = np.float(params[2])
    fd      = falloff*dist
    omfd    = 1.0-falloff*dist
    bfd2    = beta*omfd*omfd
    return \
      sill*fd*np.exp(omfd-bfd2)

dg=gridx.size
# dg=gridx.size/5
# dg=gridx.size/10
# dg=1

dx = span_array(gridx)
dy = span_array(gridy)
dr = sqrt(dx*dx+dy*dy)

# smaller beta => tighter sample distribution
#+ beta0=1.5*(dr)
# beta0=0.5*(dr)
beta0=0.75*(dr)
# beta0=1.0*(dr)
l=1.2*(dx/2)
w=1.2*(dy/2)

gridz, data_x, data_y, data_z = drive_OKrige(\
                         grid_stride=dg\
                         ,random_permute=True\
                         ,x=gridx,y=gridy\
                         ,src_x=src_x,src_y=src_y,src_z=src_z\
                         ,variogram_model='custom'\
                         ,variogram_parameters=custom_args\
                         ,variogram_function=custom_vg\
                         ,enable_plotting=True
                         ,npts=1000
                         ,beta0=beta0
                         ,frac=0.0
                         ,l=l,w=w
                         )

xy1 = np.zeros((gridz.shape[0],2))
xy1[:,0] = gridx
xy1[:,1] = gridy
grid_hull = ConvexHull(xy1)
               
# print('gridz: ',gridz)

# gridz[gridz<0] = 0
# data_=data_orig
# data_z=np.exp(data_z)
# gridz=np.exp(gridz)
    
##### PLOTTING #####

data_x_basemap = longitude_ [idx_fill]
data_y_basemap = latitude_  [idx_fill]
data_z_basemap = data_      [idx_fill]

# vmin0=np.nanmin(data_); vmax0=np.nanmax(data_)
# vmin0=np.nanmin(data_); vmax0=np.nanmax(data_)
vmin0=np.nanmin(data_z_basemap); vmax0=np.nanmax(data_z_basemap)
# vmin1=0.5*np.nanmin(z); vmax1=1.5*np.nanmax(z)
vmin1=0.9*np.nanmin(gridz); vmax1=1.1*np.nanmax(gridz)
vmin=vmin1
vmax=vmax1

iSubplot = -1
ncolSubplots = 1
nrowSubplots = 1
nTotalSubplots = ncolSubplots*nrowSubplots
fig, axes = plt.subplots(ncolSubplots,nrowSubplots)

if nTotalSubplots == 1:
    axes = [axes]

def iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots):
    if nrowSubplots == 1 and ncolSubplots == 1 :
        iSub = 0
    elif nrowSubplots == 1 :
        iSub = icolSubplot
    elif ncolSubplots == 1 :
        iSub = irowSubplot
    else:
        iSub = (icolSubplot,irowSubplot)
    return iSub

marker_size = 4
# marker_size = 16
# marker_size = 32
m_alpha = 1.0
# m_alpha = 0.5

m_lat0   = lat0 / rad_from_deg
m_lon0   = lon0 / rad_from_deg
m_dlat   = 8*3
m_dlon   = 8*3
llcrnrlon = m_lon0 - m_dlon/2
llcrnrlat = m_lat0 - m_dlat/2
urcrnrlon = m_lon0 + m_dlon/2
urcrnrlat = m_lat0 + m_dlat/2

print('plot box: ', llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)

##### PLOTTING ##### 1. Sampled data

colormap_0 = plt.cm.rainbow
colormap_1 = plt.cm.gist_yarg
colormap_2 = plt.cm.plasma
colormap_x = colormap_0

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])

m.scatter(data_x, data_y, c=data_z
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True
              ,alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )

basename = 'Sample data'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING ##### 2. Sampled data + krige

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=plt.cm.rainbow,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
m.scatter(data_x, data_y, c=data_z
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True
              ,alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

m.scatter(gridx,gridy,c=gridz
          ,cmap=colormap_x
          ,linewidth=0
          ,alpha=m_alpha
          ,latlon=True
          ,vmin=vmin, vmax=vmax
          ,edgecolors=None
          ,s=marker_size*2
          ,marker='s'
          )

# m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
          edgecolors=(0.0,1.0,0.0),
          linewidth=1,
          alpha=m_alpha*0.5,
          latlon=True,
          vmin=vmin, vmax=vmax,
          facecolors='none'
          )

#           norm=mpl.colors.Normalize(vmin,vmax),

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

# basename = os.path.basename(FILE_NAME)
basename = 'Sample data + krige'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING ##### 2.1 Krige

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=plt.cm.rainbow,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
# m.scatter(data_x, data_y, c=data_z
#               ,s=marker_size ,cmap=colormap_x
#               ,edgecolors=None, linewidth=0
#               ,latlon=True
#               ,alpha=m_alpha
#               ,vmin=vmin, vmax=vmax
#                 )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

m.scatter(gridx,gridy,c=gridz
          ,cmap=colormap_x
          ,linewidth=0
          ,alpha=m_alpha
          ,latlon=True
          ,vmin=vmin, vmax=vmax
          ,edgecolors=None
          ,s=marker_size*2
          ,marker='s'
          )

# m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
# m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
#           edgecolors=(0.0,1.0,0.0),
#           linewidth=1,
#           alpha=m_alpha*0.5,
#           latlon=True,
#           vmin=vmin, vmax=vmax,
#           facecolors='none'
#           )
# 
# #           norm=mpl.colors.Normalize(vmin,vmax),

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

# basename = os.path.basename(FILE_NAME)
basename = 'Krige'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()


##### PLOTTING ##### 3. Full data

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=plt.cm.rainbow,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True
              ,alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

m.scatter(gridx,gridy,c=gridz
          ,cmap=colormap_x
          ,linewidth=0
          ,alpha=m_alpha
          ,latlon=True
          ,vmin=vmin, vmax=vmax
          ,edgecolors=None
          ,s=marker_size*2
          ,marker='s'
          )

# m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
          edgecolors=(0.0,1.0,0.0),
          linewidth=1,
          alpha=m_alpha*0.5,
          latlon=True,
          vmin=vmin, vmax=vmax,
          facecolors='none'
          )

#           norm=mpl.colors.Normalize(vmin,vmax),

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

# basename = os.path.basename(FILE_NAME)
basename = 'Full, sample data + krige'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING ##### 4. Full data, sample unmarked

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=plt.cm.rainbow,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True
              ,alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

## m.scatter(gridx,gridy,c=gridz
##           ,cmap=colormap_x
##           ,linewidth=0
##           ,alpha=m_alpha
##           ,latlon=True
##           ,vmin=vmin, vmax=vmax
##           ,edgecolors=None
##           ,s=marker_size*2
##           ,marker='s'
##           )

## # m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
## m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##           edgecolors=(0.0,1.0,0.0),
##           linewidth=1,
##           alpha=m_alpha*0.5,
##           latlon=True,
##           vmin=vmin, vmax=vmax,
##           facecolors='none'
##           )

#           norm=mpl.colors.Normalize(vmin,vmax),

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

# basename = os.path.basename(FILE_NAME)
basename = 'Full data'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING #####

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=colormap_x,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True, alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

m.scatter(gridx,gridy,c=gridz
              ,cmap=colormap_x
              ,edgecolors=None
              ,linewidth=0
              ,alpha=m_alpha
              ,latlon=True
              ,vmin=vmin, vmax=vmax
              ,marker='s'
              ,s=marker_size*3
          )

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

## # m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
## m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##           edgecolors=(0.0,1.0,0.0),
##           linewidth=1,
##           alpha=m_alpha*0.5,
##           vmin=vmin, vmax=vmax,
##           facecolors='none'
##           )
## 
## #           norm=mpl.colors.Normalize(vmin,vmax),


# basename = os.path.basename(FILE_NAME)
basename = 'Full data + krige'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING #####

# m_alpha=0.5
m_alpha=1.0

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub] )

#                 llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
#                 )

m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=colormap_x,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
              ,s=marker_size ,cmap=colormap_x
              ,edgecolors=None, linewidth=0
              ,latlon=True, alpha=m_alpha
              ,vmin=vmin, vmax=vmax
                )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

m.scatter(    gridx,gridy,c=gridz
              ,cmap=colormap_x
              ,linewidth=0
              ,alpha=m_alpha
              ,latlon=True
              ,vmin=vmin, vmax=vmax
              ,marker='s'
              ,edgecolors=None
              ,s=marker_size*2
              )

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

##++ m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##++           edgecolors=(0.0,1.0,0.0),
##++           linewidth=1,
##++           alpha=m_alpha*0.5,
##++           latlon=True,
##++           vmin=vmin, vmax=vmax,
##++           facecolors='none'
##++           )

## # m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
## m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##           edgecolors=(0.0,1.0,0.0),
##           linewidth=1,
##           alpha=m_alpha*0.5,
##           vmin=vmin, vmax=vmax,
##           facecolors='none'
##           )
## 
## #           norm=mpl.colors.Normalize(vmin,vmax),


# basename = os.path.basename(FILE_NAME)
basename = 'Full data + krige, full planet'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
# plt.show()

##### PLOTTING #####

# m_alpha=0.5
m_alpha=1.0

# Draw an equidistant cylindrical projection using the high resolution
# coastline database.
iSubplot=iSubplot+1; icolSubplot = iSubplot % ncolSubplots; irowSubplot = iSubplot/ncolSubplots
iSub = iSubIndex(irowSubplot,icolSubplot,nrowSubplots,ncolSubplots)
if iSubplot % nTotalSubplots == 0:
    fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
    if nTotalSubplots == 1:
        axes = [axes]
print('subplot: ',iSubplot, ", ij= ",irowSubplot,icolSubplot,iSub)
m = Basemap(projection='cyl', resolution='h', ax=axes[iSub] )

#                 llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
#                 )

m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])

mod05 = MODIS_DataField(\
                            datafilename='MOD05_L2.A2015304.1815.006.2015308155414.hdf'\
                            ,datafieldname='Water_Vapor_Infrared'\
                            ,srcdirname='MODIS/'\
                            )
m.pcolormesh(mod05.longitude, mod05.latitude, mod05.data, latlon=True\
                 ,vmin=vmin,vmax=vmax\
                 ,cmap=colormap_x\
                 )

# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
#++ m.scatter(data_x, data_y, c=data_z, s=marker_size, cmap=colormap_x,
#++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
#+++ m.scatter(data_x_basemap, data_y_basemap, c=data_z_basemap
#+++               ,s=marker_size ,cmap=colormap_x
#+++               ,edgecolors=None, linewidth=0
#+++               ,latlon=True, alpha=m_alpha
#+++               ,vmin=vmin, vmax=vmax
#+++                 )
#  norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()


#+++m.scatter(    gridx,gridy,c=gridz
#+++              ,cmap=colormap_x
#+++              ,linewidth=0
#+++              ,alpha=m_alpha
#+++              ,latlon=True
#+++              ,vmin=vmin, vmax=vmax
#+++              ,marker='s'
#+++              ,edgecolors=None
#+++              ,s=marker_size*2
#+++              )

draw_screen_poly( gridx[grid_hull.vertices], gridy[grid_hull.vertices], m )

##++ m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##++           edgecolors=(0.0,1.0,0.0),
##++           linewidth=1,
##++           alpha=m_alpha*0.5,
##++           latlon=True,
##++           vmin=vmin, vmax=vmax,
##++           facecolors='none'
##++           )

## # m.scatter(data_x,data_y,c=data_z, s=marker_size, cmap=colormap_x,
## m.scatter(data_x,data_y, s=marker_size, cmap=colormap_x,
##           edgecolors=(0.0,1.0,0.0),
##           linewidth=1,
##           alpha=m_alpha*0.5,
##           vmin=vmin, vmax=vmax,
##           facecolors='none'
##           )
## 
## #           norm=mpl.colors.Normalize(vmin,vmax),


# basename = os.path.basename(FILE_NAME)
basename = 'Comparison'
# plt.title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
axes[iSub].set_title('{0}\n{1}'.format(basename, DATAFIELD_NAME))
# fig = plt.gcf()
plt.show()
