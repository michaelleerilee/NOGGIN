#!/usr/bin/env python
"""

Graph the source data.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/KRIGE.

"""

import json
import math
import os
import sys
from time import gmtime, strftime

import Krige
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


_drive_OKrige_nlags   = 12
_drive_OKrige_weight  = False
_drive_OKrige_verbose = False
_drive_OKrige_eps     = 1.0e-10
_drive_OKrige_backend = 'vectorized'
_drive_OKrige_plot_variogram      = False
_drive_OKrige_enable_statistics   = False
_drive_OKrige_log_calc            = True

###########################################################################
##
##

SRC_DIRECTORY_BASE=mdf.data_src_directory()
    
###########################################################################
##
##
if False:
    _datafield = 'Atmospheric_Water_Vapor_Mean'
    vmin=-2.0; vmax=1.25
    npts = 2000
    _variogram_model = 'gamma_rayleigh_nuggetless_variogram_model'
    
    SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
    #  MYD08_D3.A2015304.006.2015305223906.hdf
    # i = "MOD08_D3.A2015304.061.2017323113710.hdf"
    # i = "MOD08_D3.A2015305.061.2017323113224.hdf"
    # i = "MOD08_D3.A2015306.061.2017323115110.hdf"
    i = "MYD08_D3.A2015304.061.2018054061429.hdf"
    # long_name="Atmospheric_Water_Vapor_Mean"

###########################################################################
##
## Ozone Burden
if False:
    _datafield = 'Total_Ozone_Mean'
    vmin=1.5; vmax=2.75
    npts = 2000
    _variogram_model = 'linear_variogram_model'
    # _variogram_model = 'gamma_rayleigh_nuggetless_variogram_model'
    _drive_OKrige_plot_variogram      = True
    
    SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
    #  MYD08_D3.A2015304.006.2015305223906.hdf
    # i = "MOD08_D3.A2015304.061.2017323113710.hdf"
    # i = "MOD08_D3.A2015305.061.2017323113224.hdf"
    # i = "MOD08_D3.A2015306.061.2017323115110.hdf"
    i = "MYD08_D3.A2015304.061.2018054061429.hdf"
    # long_name="Atmospheric_Water_Vapor_Mean"

###########################################################################
##
## Ozone Column # But wait, this isn't a MODIS object...
##
if True:
    _datafield = '/HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ColumnAmountO3'
    vmin=1.5; vmax=2.75
    npts = 2000
    # npts = 2000
    # _variogram_model = 'linear_variogram_model'
    # _variogram_model = 'power_variogram_model'
    # _variogram_model = 'gaussian_variogram_model'
    _variogram_model = 'spherical_variogram_model'
    # _variogram_model = 'gamma_rayleigh_nuggetless_variogram_model'
    _drive_OKrige_plot_variogram      = True
    _drive_OKrige_nlags               = 12
    _drive_OKrige_log_calc            = True
    
    SRC_DIRECTORY=SRC_DIRECTORY_BASE+'OMI/'
    i = "OMI-Aura_L3-OMTO3d_2015m1015_v003-2015m1017t024934.he5"

###########################################################################

_graph_results = True

###########################################################################

modis_obj = mdf.MODIS_DataField(\
                                datafilename=i\
                                ,datafieldname=_datafield\
                                ,srcdirname=SRC_DIRECTORY\
                                ,hack_branchcut_threshold=200\
)

_plot_modis_obj = False
if _plot_modis_obj:
    # modis_obj.scatterplot(m=m
    marker_size = 0.5
    m_alpha = 1.0
    colormap_0 = plt.cm.rainbow
    colormap_1 = plt.cm.gist_yarg
    colormap_2 = plt.cm.plasma
    colormap_x = colormap_0
    vmin=-2.0; vmax=1.25
    modis_obj.scatterplot(m=None\
                          ,plt_show = False\
                          ,vmin=vmin,vmax=vmax\
                          ,cmap=colormap_x\
                          ,marker_size=marker_size*2\
                          ,value_map=Krige.log10_map
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

data_mx_in_grid = np.nanmax(data1)

krigeBox = mdf.BoundingBox((mdf.Point((np.nanmin(gridx),np.nanmin(gridy)))\
                            ,mdf.Point((np.nanmax(gridx),np.nanmin(gridy)))))
                            
dx = Krige.span_array(gridx)
dy = Krige.span_array(gridy)
dr = math.sqrt(dx*dx+dy*dy)
dg = gridx.size

beta0=1.5*(dr)
lw_scale = 2.5
# lw_scale = 3.0
l=lw_scale*(dx/2)
w=lw_scale*(dy/2)



# The args actually used.
variogram_parameters = []

krige_results = []

divergence_threshold = 1.5
npts_increase_factor = 1.5

inf_detected = False

max_iter_start = 5
max_iter = max_iter_start
npts_in  = npts

while( True ):
    print 'npts_in( '+str(k)+' ) = '+str(npts_in)
    kr=0    
    kr =\
        Krige.drive_OKrige(\
                           grid_stride         = dg\
                           ,random_permute     = True\
                           ,x                  = gridx\
                           ,y                  = gridy\
                           ,src_x              = longitude1\
                           ,src_y              = latitude1\
                           ,src_z              = data1\
                           ,log_calc           = _drive_OKrige_log_calc\
                           ,variogram_model    = _variogram_model\
                           ,variogram_function = vm.variogram_models[_variogram_model].function\
                           ,enable_plotting    = _drive_OKrige_plot_variogram\
                           ,enable_statistics  = _drive_OKrige_enable_statistics\
                           ,nlags              = _drive_OKrige_nlags
                           ,npts               = npts\
                           ,beta0              = beta0\
                           ,frac               = 0.0\
                           ,l                  = l\
                           ,w                  = w\
                           ,weight             = _drive_OKrige_weight\
                           ,verbose            = _drive_OKrige_verbose\
                           ,eps                = _drive_OKrige_eps\
                           ,backend            = _drive_OKrige_backend\
        )
    kr_mx = np.nanmax(kr.z)
    max_iter = max_iter - 1
    print 'kr_mx,data_mx_in_grid: ',kr_mx,data_mx_in_grid
    if (max_iter < 1) or (kr_mx < divergence_threshold * data_mx_in_grid):
        if (max_iter < 1):
            print '**'
            print '*** kriging diverged, max_iter exceeded, continuing to next tile'
        break
    else:
        print '***'
        print '*** kriging diverged, changing npts, iter: ',max_iter_start-1-max_iter
        if np.inf in kr.z:
            if inf_detected:
                print 'inf detected again, increasing npts'
                npts_in = npts_in*npts_increase_factor
                inf_detected = False
            else:
                print 'inf detected, reducing npts'
                npts_in = npts_in*0.75
                inf_detected = True
        else:
            print 'increasing npts'
            npts_in = npts_in*npts_increase_factor

krige_results.append(kr)

krige_results[-1].title         = modis_obj.datafilename
krige_results[-1].zVariableName = modis_obj.datafieldname
    
krige_results[-1].dbg   = True
krige_results[-1].dbg_x = longitude1
krige_results[-1].dbg_y = latitude1
krige_results[-1].dbg_z = data1
krige_results[-1].sort_on_longitude_dbg_xyz()
                     
end_time = time.time()
print 'calculation wall clock run time (sec) = '+str(end_time-start_time)
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

krigeSketch_results = krige_results

if True:
    print 'mnmx(kr.z):     ',np.nanmin(krige_results[-1].z)     ,np.nanmax(krige_results[-1].z)
    print 'mnmx(kr.dbg_z): ',np.nanmin(krige_results[-1].dbg_z) ,np.nanmax(krige_results[-1].dbg_z)
    print 'mnmx(kr.src_z): ',np.nanmin(krige_results[-1].src_z) ,np.nanmax(krige_results[-1].src_z)

if _graph_results:
    print 'graphing...'
    plot_configuration = Krige.krigePlotConfiguration(marker_size = 1.75\
                                                      ,zVariableName = krigeSketch_results[0].zVariableName\
                                                      ,title         = krigeSketch_results[0].title\
                                                      ,vmap          = Krige.log10_map\
                                                      ,vmin          = vmin\
                                                      ,vmax          = vmax\
    )
    #                                                  ,title         = '.'.join(krigeSketch_results[0].title.split('.',3)[0:2])

    execfile('2018-0519-KrigeSketchPlot-1.py')

end_time = time.time()
print 'total wall clock run time (sec) = '+str(end_time-start_time)

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'DailyLoadSketch done'
