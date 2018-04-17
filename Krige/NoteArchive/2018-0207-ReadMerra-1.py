#!/opt/local/bin/python

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from math import *
import netCDF4 as ntc
from local import ncdump
import datetime as dt


# http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html#code

fileName = 'MERRA2_400.inst1_2d_asm_Nx.20151031.nc4'
ds = Dataset(fileName, 'r')

attrs, dims, vars = ncdump(ds,verbose=False)

lats = ds.variables['lat'][:]
lons = ds.variables['lon'][:]
time = ds.variables['time'][:]
tqv   = ds.variables['TQV'][:]
units = ds.variables['TQV'].units

time_idx = 19
offset = dt.timedelta(hours=48) # ???

for t in time:
    print('t: ',t)

# dt_time = [dt.date(1,1,1) + dt.timedelta(hours=t) - offset for t in time]
# cur_time = dt_time[time_idx]

print("lats.shape: ", lats.shape)
print("lons.shape: ", lons.shape)
print("time.shape: ", time.shape)
print("tqv.shape:  ", tqv.shape)

# Plot of global temperature on our random day
fig = plt.figure()
fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
# Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
# for other projections.
# m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90,\

# m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,\
#             llcrnrlon=-180, urcrnrlon=180, resolution='c', lon_0=0)

m = Basemap(projection='cyl', resolution='h')
m.drawmapboundary(fill_color='grey')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90, 91, 45))
m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
#####

lon2d, lat2d = np.meshgrid(lons,lats)
print('lon2d.shape: ',lon2d.shape)
longitude_ = lon2d
latitude_  = lat2d
data  = tqv[19,:,:].astype(np.double)
print('data.shape: ',data.shape)
# data_ = np.ravel(data)
data_ = data
print('data_.shape: ',data_.shape)
units_ = units
marker_size=64
m_alpha=1

m.scatter(longitude_, latitude_, c=data_, s=marker_size, cmap=plt.cm.rainbow,
          edgecolors=None, linewidth=0,
          latlon=True, alpha=m_alpha)
#  norm=mpl.colors.LogNorm(),
cb = m.colorbar()
cb.set_label(units_)
cb.set_alpha(0.5)
cb.draw_all()
#####


#####
plt.show()
ds.close()
print('done...')


