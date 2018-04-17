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

def center_array(a):
    return 0.5*(np.nanmax(a) + np.nanmin(a))

def span_array(a):
    return (np.nanmax(a) - np.nanmin(a))

# marker_size = 4
marker_size = 16
m_alpha = 1.0
# m_alpha = 0.5

nlat = 100; lat = np.linspace(-90,90,nlat)
nlon = 100; lon = np.linspace(-180,180,nlon)

# lonv,latv = np.meshgrid(lon,lat,sparse=True)
lonv,latv = np.meshgrid(lon,lat)

lonv = np.reshape(lonv,lonv.size)
latv = np.reshape(latv,latv.size)

longitude_orig = lonv; longitude_rad = longitude_orig*rad_from_deg
latitude_orig  = latv; latitude_rad  = latitude_orig *rad_from_deg
data_      = np.zeros(latitude_orig.shape)

lon0 = center_array(longitude_orig); lon0_rad = lon0*rad_from_deg
lat0 = center_array(latitude_orig);  lat0_rad = lat0*rad_from_deg

r2     = np.power(longitude_rad-lon0_rad,2.0) + np.power(latitude_rad-lat0_rad,2.0)
# data_ = np.exp(-4.0*r2)*np.cos(2.0*pi*r2*10.0)
data_orig = np.exp(-4.0*r2)

np.random.seed(120)

# data_ = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,4)
# idx = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,0.0125)

longitude_ = False; latitude_ = False

idx = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,128)
longitude_= longitude_orig[idx]
latitude_ = latitude_orig[idx]
data_     = data_orig[idx]

# print('n(idx) = ',idx[idx].size)

m_lat0   = lat0
m_lon0   = lon0
m_dlon   = 1.5*span_array(longitude_orig)
m_dlat   = 1.5*span_array(latitude_orig)
llcrnrlon = max(m_lon0 - m_dlon/2,-180)
llcrnrlat = max(m_lat0 - m_dlat/2, -90)
urcrnrlon = min(m_lon0 + m_dlon/2, 180)
urcrnrlat = min(m_lat0 + m_dlat/2,  90)

ncolSubplots = 1
nrowSubplots = 1
iSub = 0

fig, axes = plt.subplots(ncolSubplots,nrowSubplots)
axes = [ axes ]

m = Basemap(projection='cyl', resolution='h', ax=axes[iSub],
                llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat, urcrnrlon = urcrnrlon, urcrnrlat = urcrnrlat
                )
m.drawmapboundary(fill_color='grey')
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
# m.scatter(longitude, latitude, c=data_, s=0.1, cmap=plt.cm.jet,
# m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.jet,
m_scatter = m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          latlon=True, alpha=m_alpha)
# norm=mpl.colors.LogNorm(),
# cb = m.colorbar()
# cb.set_label(units_)
# cb.set_alpha(1)
# cb.draw_all()

def adaptive_index(x0,y0,x,y,npts=100,beta0=200,frac=0.1):
    idx_size = 0; idx = False
    beta=beta0
    while(True):
        idx = random_index(x0,y0,x,y,beta)
        idx_size = idx[idx].size
        if idx_size >= npts:
            break
        else:
            beta=beta*(1.0-frac)
            if beta/beta0 < 0.001:
                raise ValueError('Too many iterations, beta too small.')
    return idx

def updatefig(it):
    global m, m_scatter, longitude_rad, latitude_rad, marker_size, m_alpha
    # print('updatefig: ',it)
    try:
        m_scatter.remove()
    except ValueError:
        print('warning: m_scatter.remove()')
    idx_size = 0
    idx=False
    while(idx_size <= 0):
        lon0_rad = np.random.uniform(np.nanmin(longitude_rad),np.nanmax(longitude_rad))
        lat0_rad = np.random.uniform(np.nanmin(latitude_rad),np.nanmax(latitude_rad))
        # idx = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,128)
        # idx = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,96)
        # idx = random_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad,64)
        idx = adaptive_index(lon0_rad,lat0_rad,longitude_rad,latitude_rad)
        idx_size = idx[idx].size
        # sys.stdout.write('.')
    # print('+')
    print('idx_size: ',idx_size)
    longitude_ = False; latitude_ = False; data_ = False
    longitude_= longitude_orig[idx]
    latitude_ = latitude_orig[idx]
    data_     = data_orig[idx]
    try:
        m_scatter = m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.rainbow,
                              edgecolors=None, linewidth=0,
                              latlon=True, alpha=m_alpha)
    except IndexError:
        print('IndexError')
        print('     idx_size: ',idx_size)

ani = animation.FuncAnimation(fig,updatefig,20)
    
plt.show()

# fig, axes = plt.subplots()


