#!/usr/bin/env python
"""

Load the L3 daily data, then try to graph.

ML Rilee, RSTLLC, mike@rilee.net for NASA/ACCESS-15/KRIGE.

"""

import json
import math
import os
import sys
from time import gmtime, strftime

import Krige
import Krige.DataField as df


import pykrige
import pykrige.variogram_models as vm

# from DataField import DataField, BoundingBox, df.Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.spatial import ConvexHull

import time

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'DailyLoadSketch start'

start_time = time.time()

_drive_OKrige_nlags               = 12
_drive_OKrige_weight              = False
_drive_OKrige_verbose             = False
_drive_OKrige_eps                 = 1.0e-10
_drive_OKrige_backend             = 'vectorized'
_drive_OKrige_plot_variogram      = False
_drive_OKrige_enable_statistics   = False
_drive_OKrige_log_calc            = True
_drive_OKrige_random_permute      = True
_drive_OKrige_coordinates_type    = 'geographic'
_drive_OKrige_sampling            = 'adaptive' # or 'decimate' or 'None'

#  Iteration control parameters
#
max_iter = 10
divergence_threshold = 1.5
npts_increase_factor = 1.5
npts_decrease_factor = 0.75

grid_stride_scale = 1.0

frac        = 0.0

# beta0_scale = 2.0
beta0_scale = 1.5

_plot_kriged_data_outline = False


###########################################################################
##
##

SRC_DIRECTORY_BASE=df.data_src_directory()

###########################################################################
##
##

# _DailyLoad_Case = 'MODIS-Water_Vapor_Mean-Case-1'
# _DailyLoad_Case = 'MODIS-Total_Ozone_Burden-Case-1'
_DailyLoad_Case = 'OMI-Total_Ozone-Case-1'

###########################################################################
##
##
if _DailyLoad_Case == 'MODIS-Water_Vapor_Mean-Case-1':
    _datafield = 'Atmospheric_Water_Vapor_Mean'
    vmin=-2.0; vmax=1.25
    npts = 1000
    # npts = 2500
    # npts = 5000
    # npts = 10000
    #+? npts =  50
    # npts =  500
    _variogram_model = 'gamma_rayleigh_nuggetless_variogram_model'
    # _variogram_model = 'spherical'
    # _variogram_model = 'spherical_variogram_model'
    _drive_OKrige_plot_variogram      = False
    _drive_OKrige_weight              = True
    _drive_OKrige_log_calc            = True
    _drive_OKrige_random_permute      = True
    _drive_OKrige_verbose             = False
    _drive_OKrige_coordinates_type    = 'geographic'
    _drive_OKrige_backend             = 'vectorized'
    # _drive_OKrige_sampling            = 'decimate' # uses npts to set stride
    _drive_OKrige_sampling            = 'adaptive' # uses npts to set stride

    _plot_kriged_data_outline = False
    
    max_iter = 10
    npts_increase_factor = 0.8
    npts_decrease_factor = 0.8

    grid_stride_scale = 1
    
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
if _DailyLoad_Case == 'MODIS-Total_Ozone_Burden-Case-1':
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
if _DailyLoad_Case == 'OMI-Total_Ozone-Case-1':
    _datafield = '/HDFEOS/GRIDS/OMI Column Amount O3/Data Fields/ColumnAmountO3'
    vmin=1.5; vmax=2.75
    #+ npts = 260
    # npts = 250 # looks not-terrible
    npts = 300 # looks not-terrible
    # npts = 500 # looks not-terrible
    # npts = 1000
    # npts = 5000
    # npts = 10000 # looks good
    # _variogram_model = 'linear_variogram_model'
    # _variogram_model = 'power_variogram_model'
    # _variogram_model = 'gaussian_variogram_model'
    _variogram_model = 'spherical_variogram_model'
    # _variogram_model = 'gamma_rayleigh_nuggetless_variogram_model'
    _drive_OKrige_plot_variogram      = False
    _drive_OKrige_weight              = True
    _drive_OKrige_nlags               = 6
    _drive_OKrige_log_calc            = True
    _drive_OKrige_random_permute      = True

    _plot_kriged_data_outline = False    

    max_iter = 1
    npts_increase_factor = 0.8
    npts_decrease_factor = 0.8

    grid_stride_scale = 1
    
    SRC_DIRECTORY=SRC_DIRECTORY_BASE+'OMI/'
    i = "OMI-Aura_L3-OMTO3d_2015m1012_v003-2015m1014t023506.he5"
    # i = "OMI-Aura_L3-OMTO3d_2015m1013_v003-2015m1015t013733.he5"
    # i = "OMI-Aura_L3-OMTO3d_2015m1014_v003-2015m1016t021656.he5"
    # i = "OMI-Aura_L3-OMTO3d_2015m1015_v003-2015m1017t024934.he5"

    ## Why are loop and vectorized yielding slightly different stabilities?
    # _drive_OKrige_backend             = 'loop'
    _drive_OKrige_backend               = 'vectorized'
    _drive_OKrige_verbose               = True

###########################################################################

_graph_results = True

###########################################################################

modis_obj = df.DataField(\
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


# idx_source = (~modis_obj.data.mask)
# idx_target = ( modis_obj.data.mask)

# idx_source = np.where((~modis_obj.data.mask) & (modis_obj.latitude > 45.0))
# idx_target = np.where(( modis_obj.data.mask) & (modis_obj.latitude > 45.0))
# idx_source = np.where((~modis_obj.data.mask) & (modis_obj.latitude > 60.0))
# idx_target = np.where(( modis_obj.data.mask) & (modis_obj.latitude > 60.0))
# idx_source = np.where((~modis_obj.data.mask) & (modis_obj.latitude > 0.0))
# idx_target = np.where(( modis_obj.data.mask) & (modis_obj.latitude > 0.0))

# idx_source = np.where((~modis_obj.data.mask) & (modis_obj.latitude > 0.0) & (modis_obj.longitude > 90.0))
# idx_target = np.where(( modis_obj.data.mask) & (modis_obj.latitude > 0.0) & (modis_obj.longitude > 90.0))

###########################################################################
lon0=-180; lat0=-90
dlon=720
dlat=180
# npts=200

###########################################################################
# lon0=120; lat0=20
# mlr broken dlon=20
# mlr ok dlon=15-17
# mlr broken dlon=18
# dlon = 17 # x.size=249
# dlat=dlon

# # ###########################################################################
# lon0=140; lat0=20
# # mlr ok  dlon = 18,20
# # mlr bad dlon = 21,22,23,25
# # mlr post adaptive_index "fix 1"
# # mlr using npts = 350
# # mlr ok  dlon = 21
# # mlr bad dlon = 22,25,30
# # mlr post adaptive_index "fix 2" stop at first slow down in idx_size
# # mlr ok  dlon = 23
# # mlr bad dlon = 24,25,30
# dlon = 24
# dlat=dlon

lon1=lon0+dlon; lat1=lat0+dlat
idx_source = np.where((~modis_obj.data.mask) \
                      & (modis_obj.latitude > lat0) \
                      & (modis_obj.latitude < lat1) \
                      & (modis_obj.longitude > lon0) \
                      & (modis_obj.longitude < lon1))
idx_target = np.where(( modis_obj.data.mask) \
                      & (modis_obj.latitude > lat0) \
                      & (modis_obj.latitude < lat1) \
                      & (modis_obj.longitude > lon0) \
                      & (modis_obj.longitude < lon1))

print('trimming to lat,lon: '+str((lon0,lat0))+'..'+str((lon1,lat1)))

longitude1 = modis_obj.longitude[idx_source]
latitude1  = modis_obj.latitude [idx_source]
data1      = modis_obj.data     [idx_source]

gridx  = modis_obj.longitude[idx_target]
gridy  = modis_obj.latitude [idx_target]
gridz  = np.zeros(gridx.shape)
gridss = np.zeros(gridx.shape)

data_mx_in_grid = np.nanmax(data1)

krigeBox = df.BoundingBox((df.Point((np.nanmin(gridx),np.nanmin(gridy)))\
                            ,df.Point((np.nanmax(gridx),np.nanmin(gridy)))))
                            
# dx = span_array(gridx)
dx = Krige.span_array(gridx)
dy = Krige.span_array(gridy)
dr = math.sqrt(dx*dx+dy*dy)
dg = int(gridx.size*grid_stride_scale)

# beta0=1.5*(dr)
beta0=beta0_scale*(dr)
lw_scale = 2.5
# lw_scale = 3.0
l=lw_scale*(dx/2)
w=lw_scale*(dy/2)

print('l,w=   '+str(l)+','+str(w))
print('beta0= '+str(beta0))
print('frac=  '+str(frac))

# The args actually used.
variogram_parameters = []

krige_results = []

inf_detected = False

# s_history=[npts+1,npts]
# npts_history=[-2,-1]

# max_iter_start = 5
# max_iter = max_iter_start
max_iter_start = max_iter
npts_in        = npts
k=0
i0 = 0
print('-----Start iterating-----')
while( True ):
    print('-----drive_OKrige iteration-----')
    print 'npts_in( '+str(k)+' ) = '+str(npts_in)
    i0 = i0 + 1
    kr=0    
    kr =\
        Krige.drive_OKrige(\
                           grid_stride         = dg\
                           ,random_permute     = _drive_OKrige_random_permute\
                           ,x                  = gridx\
                           ,y                  = gridy\
                           ,src_x              = longitude1\
                           ,src_y              = latitude1\
                           ,src_z              = data1\
                           ,log_calc           = _drive_OKrige_log_calc\
                           ,variogram_model    = _variogram_model\
                           ,variogram_function = vm.variogram_models[_variogram_model].function\
#                           ,variogram_function = vm.spherical_variogram_model\
                           ,enable_plotting    = _drive_OKrige_plot_variogram\
                           ,enable_statistics  = _drive_OKrige_enable_statistics\
                           ,nlags              = _drive_OKrige_nlags
                           ,npts               = npts_in\
                           ,beta0              = beta0\
                           ,frac               = frac\
                           ,l                  = l\
                           ,w                  = w\
                           ,weight             = _drive_OKrige_weight\
                           ,verbose            = _drive_OKrige_verbose\
                           ,eps                = _drive_OKrige_eps\
                           ,backend            = _drive_OKrige_backend\
                           ,coordinates_type   = _drive_OKrige_coordinates_type\
                           ,sampling           = _drive_OKrige_sampling\
        )
    kr_mx    = np.nanmax(kr.z)
    kr_s_mn  = np.nanmin(kr.s)
    kr_s_mx  = np.nanmax(kr.s)
    max_iter = max_iter - 1
    print 'iter:                  ',max_iter_start-1-max_iter
    print 'max_iter:              ',max_iter
    print 'kr_mx,data_mx_in_grid: ',kr_mx,data_mx_in_grid
    print 'kr_s_mnmx:             ',kr_s_mn,kr_s_mx

    ## s_history[0]=s_history[1]
    ## s_history[1]=kr_s_mn
    ## npts_history[0]=npts_history[1]
    ## npts_history[1]=npts_in
    ## npts_guess = npts_history[0]-(s_history[0]*(npts_history[1]-npts_history[0])/(s_history[1]-s_history[0]))
    ## print('npts_guess = '+str(npts_guess))
    ## if npts_guess <= 0:
    ##     npts_guess = npts_in*npts_increase_factor
    ##     print('*** npts_guess = '+str(npts_guess))
        
    
    if (max_iter < 1) or ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
        if (max_iter < 1):
            print '**'
            print '*** max_iter exceeded, continuing to next tile'
        if ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
            print '**'
            print '*** kriging seems to have converged'
        break
    else:
        print '***'
        print '*** kriging diverged, changing npts'
        if np.inf in kr.z:
            if inf_detected:
                inf_detected = False
                print 'inf detected again, increasing npts'
                npts_in = npts_in*npts_increase_factor
            else:
                inf_detected = True
                print 'inf detected, reducing npts'
                npts_in = npts_in*npts_decrease_factor
        else:
            print 'increasing npts'
            npts_in = npts_in*npts_increase_factor
        ## if i0 > 1:
        ##     print '!!!'
        ##     npts_in = npts_guess


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
    print 'mnmx(kr.s):     ',np.nanmin(krige_results[-1].s)     ,np.nanmax(krige_results[-1].s)

if True:
    variable_name   = krige_results[-1].zVariableName
    output_filename = "DailyLoadSketch_"+modis_obj.datafilename+".hdf"
    if '.hdf.hdf' in output_filename[-8:]:
        output_filename = output_filename[:-4]
        
    kHDF = Krige.krigeHDF(\
                          krg_name                 = variable_name+'_krg'\
                          ,krg_units               = modis_obj.units\
                          ,config                  = krige_results[-1].config\
                          ,krg_z                   = krige_results[-1].z \
                          ,krg_s                   = krige_results[-1].s \
                          ,krg_x                   = krige_results[-1].x \
                          ,krg_y                   = krige_results[-1].y \
                          ,orig_name               = modis_obj.datafieldname\
                          ,orig_units              = modis_obj.units\
                          ,orig_z                  = modis_obj.data \
                          ,orig_x                  = modis_obj.longitude \
                          ,orig_y                  = modis_obj.latitude \
                          ,output_filename         = output_filename\
    )
    kHDF.save()

if _graph_results:
    print 'graphing...'
    plot_configuration = Krige.krigePlotConfiguration(marker_size              = 1.75\
                                                      ,zVariableName           = krigeSketch_results[0].zVariableName\
                                                      ,title                   = krigeSketch_results[0].title\
                                                      ,vmap                    = Krige.log10_map\
                                                      ,vmin                    = vmin\
                                                      ,vmax                    = vmax\
                                                      ,kriged                  = True\
                                                      ,source_data             = True\
                                                      ,kriged_data             = True\
                                                      ,kriged_data_outline     = _plot_kriged_data_outline\
                                                      ,source_data_last_sample = False\
    )
    execfile('2018-0519-KrigeSketchPlot-1.py')

if False:
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
