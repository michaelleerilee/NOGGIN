#!/usr/bin/env python
"""

Graph the source data.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/NOGGIN.

"""

import json
import math
import os
import sys
from time import gmtime, strftime

import noggin
import MODIS_DataField as mdf

import pykrige
import pykrige.variogram_models as vm

# from MODIS_DataField import MODIS_DataField, BoundingBox, mdf.Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.spatial import ConvexHull

import time

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'DailyLoadSketch start'

start_time = time.time()

## 

_drive_OKrige_nlags   = 12
_drive_OKrige_weight  = False
_drive_OKrige_verbose = False
_drive_OKrige_eps     = 1.0e-10
_drive_OKrige_backend = 'vectorized'

_graph_results = True

_plot_variogram      = False
_enable_statistics   = False
##

SRC_DIRECTORY_BASE=mdf.data_src_directory()
SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'

# marker_size = 3.5
# marker_size = 1
marker_size = 0.5
m_alpha = 1.0
colormap_0 = plt.cm.rainbow
colormap_1 = plt.cm.gist_yarg
colormap_2 = plt.cm.plasma
colormap_x = colormap_0
# vmin=2.0; vmax=20.0
# vmin=2.0; vmax=10.0
# vmin=0.5; vmax=8.0
# vmin=0.25; vmax=6.0
# vmin=0.5; vmax=5.0
# vmin=np.nanmin(data1); vmax=np.nanmax(data1)
# For log10
# vmin=-2.0; vmax=1.0
vmin=-2.0; vmax=1.25

def log_map(x):
    return np.log10(x+1.0e-9)

i = "MOD08_D3.A2015304.061.2017323113710.hdf"
# i = "MYD08_D3.A2015304.061.2018054061429.hdf"
# long_name="Atmospheric_Water_Vapor_Mean"

modis_obj = mdf.MODIS_DataField(\
                                datafilename=i\
                                ,datafieldname='Atmospheric_Water_Vapor_Mean'\
                                ,srcdirname=SRC_DIRECTORY\
                                ,hack_branchcut_threshold=200\
)

_plot_modis_obj = False
if _plot_modis_obj:
    # modis_obj.scatterplot(m=m
    modis_obj.scatterplot(m=None\
                          ,plt_show = False\
                          ,vmin=vmin,vmax=vmax\
                          ,cmap=colormap_x\
                          ,marker_size=marker_size*2\
                          ,value_map=log_map
    )
    plt.show()

idx_source = ~modis_obj.data.mask
idx_target =  modis_obj.data.mask

longitude1 = modis_obj.longitude[idx_source]
latitude1  = modis_obj.latitude [idx_source]
data1      = modis_obj.data     [idx_source]

gridx = modis_obj.longitude[idx_target]
gridy = modis_obj.latitude [idx_target]
gridz  = np.zeros(gridx.shape)
gridss = np.zeros(gridx.shape)

krigeBox = mdf.BoundingBox((mdf.Point((np.nanmin(gridx),np.nanmin(gridy)))\
                            ,mdf.Point((np.nanmax(gridx),np.nanmin(gridy)))))
                            
dx = noggin.span_array(gridx)
dy = noggin.span_array(gridy)
dr = math.sqrt(dx*dx+dy*dy)
dg = gridx.size

beta0=1.5*(dr)
lw_scale = 2.5
# lw_scale = 3.0
l=lw_scale*(dx/2)
w=lw_scale*(dy/2)

npts = 2000

nlags=_drive_OKrige_nlags
custom_args = None
vg_model = 'custom'

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


# The 'custom_args' actually used.
variogram_parameters = []

krige_results = []

krige_results.append(\
                     noggin.drive_OKrige(\
                                         grid_stride=dg\
                                         ,random_permute=True\
                                         ,x=gridx,y=gridy\
                                         ,src_x=longitude1\
                                         ,src_y=latitude1\
                                         ,src_z=data1\
                                         ,variogram_model='gamma_rayleigh_nuggetless_variogram_model'\
                                         ,variogram_function=vm.variogram_models['gamma_rayleigh_nuggetless_variogram_model'].function\
                                         ,enable_plotting=_plot_variogram\
                                         ,enable_statistics=_enable_statistics\
                                         ,npts=npts\
                                         ,beta0=beta0\
                                         ,frac=0.0\
                                         ,l=l,w=w\
                                         ,weight=_drive_OKrige_weight\
                                         ,verbose=_drive_OKrige_verbose\
                                         ,eps=_drive_OKrige_eps\
                                         ,backend=_drive_OKrige_backend\
                     ))


if False:
    krige_results.append(\
                         noggin.drive_OKrige(\
                                             grid_stride=dg\
                                             ,random_permute=True\
                                             ,x=gridx,y=gridy\
                                             ,src_x=longitude1\
                                             ,src_y=latitude1\
                                             ,src_z=data1\
                                             ,variogram_model=vg_model\
                                             ,variogram_parameters=custom_args\
                                             ,variogram_function=custom_vg\
                                             ,enable_plotting=_plot_variogram\
                                             ,enable_statistics=_enable_statistics\
                                             ,npts=npts\
                                             ,beta0=beta0\
                                             ,frac=0.0\
                                             ,l=l,w=w\
                                             ,weight=_drive_OKrige_weight\
                                             ,verbose=_drive_OKrige_verbose\
                                             ,eps=_drive_OKrige_eps\
                                             ,backend=_drive_OKrige_backend\
                         ))

krige_results[-1].dbg   = True
krige_results[-1].dbg_x = longitude1
krige_results[-1].dbg_y = latitude1
krige_results[-1].dbg_z = data1
krige_results[-1].sort_on_longitude_dbg_xyz()
                     
end_time = time.time()
print 'calculation wall clock run time (sec) = '+str(end_time-start_time)
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

krigeSketch_results = krige_results

if _graph_results:
    print 'graphing...'
    execfile('2018-0519-KrigeSketchPlot-1.py')

end_time = time.time()
print 'total wall clock run time (sec) = '+str(end_time-start_time)

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'DailyLoadSketch done'
