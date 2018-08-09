#!/usr/bin/env python
"""Krige all the given files in a directory to a grid.

2018-0808 ML Rilee, RSTLLC, mike@rilee.net
"""

import argparse
import json
import math
import os
import sys
import time
from time import gmtime, strftime

import Krige
import Krige.DataField as df

# from DataField import DataField, BoundingBox, df.Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.spatial import ConvexHull

import pykrige.variogram_models as vm

###########################################################################

start_time = time.time()

parser = argparse.ArgumentParser(description='Capture krige control parameters.')
parser.add_argument('-s','--sampling_fraction'\
                    ,dest='sampling_fraction'\
#                    ,metavar='samplingFraction'\
                    ,type=float, help='Suggested fraction of data to sample.')

parser.add_argument('-d','--input_directory'\
                    ,dest='inputDirectory'\
                    ,type=str\
                    ,help='The directory to search for input files.')

parser.add_argument('-o','--output_filename'\
                    ,dest='output_filename'\
                    ,type=str\
                    ,help='The output file for the results')
parser.set_defaults(output_filename = 'noggin_krige.hdf')

parser.add_argument('-l','--variogram_bin_number'\
                    ,dest='variogram_nlags'\
                    ,type=int\
                    ,help='The number of bins (of lags) for the variogram calculation.')
parser.set_defaults(variogram_nlags=8)
# parser.set_defaults(variogram_nlags=12)

parser.add_argument('-m','--variogram_model'\
                    ,dest='variogram_model'\
                    ,type=str\
                    ,help='The name of the variogram model to use.')

parser.add_argument('-n','--variable_name'\
                    ,dest='datafieldname'\
                    ,type=str\
                    ,help='The variable or datafield name to load.')

parser.add_argument('-v','--verbose'\
                    ,dest='verbose'\
                    ,action='store_true'\
                    ,help='Toggle verbose printing')

parser.set_defaults(verbose=False)

# parser.add_argument('-a','--adapt_fraction'\
#                     ,dest='adaptFraction'\
#                     ,metavar='adaptFraction'\
#                     ,type=bool, help='Vary the sampling fraction over iterations.')


args=parser.parse_args()

_flag_error_exit=False

if args.datafieldname is None:
    print('noggin_krige: error: variable/datafieldname -n must be specified.')
    _flag_error_exit=True

if args.variogram_nlags is None:
    print('noggin_krige: error: -l variogram_nlags must be specified.')
    _flag_error_exit=True

if args.variogram_model is None:
    print('noggin_krige: error: -m --variogram_model must be specified. e.g. "spherical" or "gamma_rayleigh_nuggetless_variogram_model"')
    _flag_error_exit=True

output_filename = args.output_filename
_verbose = args.verbose

if _flag_error_exit:
    print('noggin_krige: error sys.exit(1)')
    sys.exit(1)
    
_debug = False

_save_index = False

SRC_DIRECTORY=args.inputDirectory
DATAFIELDNAME=args.datafieldname

###########################################################################
#
# TODO Parameters that need to be set/injected
#
_drive_OKrige_enable_statistics = False
_drive_OKrige_nlags             = args.variogram_nlags
_drive_OKrige_variogram         = args.variogram_model
_plot_variogram                 = False
_drive_OKrige_weight            = False
_drive_OKrige_verbose           = args.verbose
_drive_OKrige_eps               = 1.0e-10 # Not well implemented
_drive_OKrige_backend           = 'vectorized'

###########################################################################

_sampling_fraction              = args.sampling_fraction
if _sampling_fraction is None:
    # lores_npts = 2000
    lores_npts = 3500
else:
    lores_npts = None


###########################################################################
# Start with MetaData sketch.
# Construct bounding box index.
src_file_list = [f for f in os.listdir(SRC_DIRECTORY) if (lambda x: '.hdf' in x or '.HDF.' in x)(f)]
if len(src_file_list) == 0:
    print('noggin_krige: eror: no HDF files found in '+SRC_DIRECTORY+', exiting')
    sys.exit(1)
modis_BoundingBoxes = {}
if _save_index:
    modis_BoundingBoxes_json = {}
bb = df.BoundingBox()

for i in src_file_list:
    if _verbose:
        print('loading '+str(i)+' from '+str(SRC_DIRECTORY))
    modis_obj = df.DataField(\
                                datafilename=i\
                                ,datafieldname=DATAFIELDNAME\
                                ,srcdirname=SRC_DIRECTORY\
                                ,hack_branchcut_threshold=200\
                                )
    bb = bb.union(modis_obj.bbox)
    modis_BoundingBoxes[i]=modis_obj.bbox
    if _save_index:
        modis_BoundingBoxes_json[i]=modis_obj.bbox.to_json()

    if _debug:
        print 'xml: ',bb.to_xml()
        print 'json: ',bb.to_json()

if _save_index:
    if _verbose:
        print 'writing to '+INDEX_FILE
        # with open(SRC_DIRECTORY+'modis_BoundingBoxes.json','w') as f:
    with open(INDEX_FILE,'w') as f:
        json.dump(modis_BoundingBoxes_json,f)

###########################################################################
# Move to KrigeSketch conventions
boxes = modis_BoundingBoxes

###########################################################################
# Initialize variables
## TODO: Should we have boxes and tiles in the same structure?
targetBoxes=[]
targetTiles=[]
krigeSketch_results = []
hires_calc = []

# npts_adapt_flag    = True # 
npts_increase_flag = True
# npts_last_kr_s = 0
max_iter_start = 5
k = -1

divergence_threshold = 1.5
npts_increase_factor = 1.5
npts_decrease_factor = 0.75

###########################################################################
# Define full grid (tiles & caps)
# ### FULL with bands and caps
# TODO Inject dependencies to parameterize

###########################################################################
## TODO use bb to limit the following
if False:
    bb_lons,bb_lats = bb.lons_lats()
    lon0 = bb_lons[0]
    lon1 = bb_lons[1]
    lat0 = bb_lats[0]
    lat1 = bb_lats[1]
    dLon = min((lon1-lon0)/3,120)
    dLat = min((lat1-lat0)/3,10)
    # dSearchLon = 0.75*dLon
    # dSearchLat = 0.75*dLat
    dSearchScale = 0.25
    dSearchLon = dSearchScale*dLon
    dSearchLat = dSearchScale*dLat
    hires_npts = lores_npts * 2

###########################################################################
##
if True:
    dLon = 120
    dLat = 10
    # dSearchScale = 0.75
    dSearchScale = 0.25 # Amount beyond the tile to search for granules.
    dSearchLon = dSearchScale*dLon
    dSearchLat = dSearchScale*dLat
    # Full lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  90-dLat
    # 4-box test
    lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  -60-dLat
    # lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 = -60-dLat
    # lores_npts = 2000
    # lores_npts = 3000
    # lores_npts = 3500
    hires_npts = lores_npts * 2
    # hires_npts = 4000
    # hires_calc = [5,11,37,48,60,61,63,67,68]
    # hires_calc = [0,1]
    # hires_calc = [0,1]
    hires_calc = []
    divergence_threshold = 1.5
    npts_increase_factor = 1.5
    npts_decrease_factor = 0.75

    cap_south = df.BoundingBox((df.Point((-180, -90))\
                                ,df.Point((180, -90+dLat))))
    targetBoxes.append(cap_south)
    
    cap_north = df.BoundingBox((df.Point((-180, 90-dLat))\
                                ,df.Point((180, 90))))
    targetBoxes.append(cap_north)

###########################################################################
# Add the other boxes.
for iLon in range(lon0,lon1,dLon):
    for jLat in range(lat0,lat1,dLat):
        krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
                                    ,df.Point((iLon+dLon, jLat+dLat))))
        targetBoxes.append(krigeBox)

###########################################################################


# Now we have boxes. Including the caps.

# Construct the target grid from targetBoxes.
# Make a bounding box covering everything, and then grid that at a given resolution.

cover = df.BoundingBox()
for box in targetBoxes:
    cover = cover.union(box)
tgt_lons,tgt_lats = cover.lons_lats()
if _verbose:
    print('noggin_krige: tgt lons,lats: '+str(tgt_lons)+', '+str(tgt_lats))
tgt_grid_dLon = 0.5
tgt_grid_dLat = 0.5
tgt_grid_full = Krige.rectangular_grid(\
                                       x0 = tgt_lons[0]\
                                       ,x1 = tgt_lons[1]\
                                       ,dx = tgt_grid_dLon\
                                       ,y0 = tgt_lats[0]\
                                       ,y1 = tgt_lats[1]\
                                       ,dy = tgt_grid_dLat\
                                       )

tgt_X1d,tgt_Y1d = tgt_grid_full.gridxy1d()

tgt_lon0 = np.nanmin(tgt_X1d)
tgt_lon1 = np.nanmax(tgt_X1d)
tgt_lat0 = np.nanmin(tgt_Y1d)
tgt_lat1 = np.nanmax(tgt_Y1d)

nx = (tgt_lon1-tgt_lon0)/tgt_grid_dLon
ny = (tgt_lat1-tgt_lat0)/tgt_grid_dLat
if np.abs(nx - int(nx)) > 1.0e-10:
    print('Error: grid_dLon does not evenly divide longitude span.')
if np.abs(ny - int(ny)) > 1.0e-10:
    print('Error: grid_dLat does not evenly divide latitude span.')

###########################################################################

# Construct individual tiles. Hope they line up with tgt_grid_full.
for krigeBox in targetBoxes:
    tile_lons,tile_lats = krigeBox.lons_lats()
    tile = Krige.rectangular_grid(\
                                  x0 = tile_lons[0]\
                                  ,x1 = tile_lons[1]\
                                  ,dx = tgt_grid_dLon\
                                  ,y0 = tile_lats[0]\
                                  ,y1 = tile_lats[1]\
                                  ,dy = tgt_grid_dLat\
    )
    targetTiles.append(tile)

###########################################################################
    
    
### TODO: Verify the size of tgt_?1d match the calculated result.
    
k=-1
for krigeBox in targetBoxes:
    for dummy in [1]:
        k=k+1
        grid = targetTiles[k]
        _calculate = True
        _enable_statistics = _drive_OKrige_enable_statistics
        if _calculate:
            if _verbose:
                print 'working on tile = '+str(k)

            iLon=krigeBox.p0.lon_degrees
            jLat=krigeBox.p0.lat_degrees

            dLon=krigeBox.p1.lon_degrees-krigeBox.p0.lon_degrees
            dLat=krigeBox.p1.lat_degrees-krigeBox.p0.lat_degrees

            if _verbose:
                print 'loading iLon,jLat: '+str(iLon)+','+str(jLat)
                print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
            
            # krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
            #                             ,df.Point((iLon+dLon, jLat+dLat))))
    
            searchBox = df.BoundingBox((df.Point((iLon-dSearchLon,      max(-90, min(jLat-dSearchLat, 90))))\
                                        ,df.Point((iLon+dLon+dSearchLon, max(-90, min(jLat+dLat+dSearchLat, 90))))))
    
            # krigeBox = df.BoundingBox((df.Point((+135.0, 30.0))\
            #                            ,df.Point((+175.0, 45.0))))

            sb_lons,sb_lats = searchBox.lons_lats()
            print('sb lons_lats: '+str(sb_lons)+', '+str(sb_lats))
    
            # Find overlapping granules and load the MODIS objects
            src_data   = {}
            modis_objs = []
            for i,v in boxes.iteritems():
                i = str(i)
                # if i[0:3] in _load_datasets: # Check to see if we're loading something DataField knows about.
                if True:
                    lons,lats = v.lons_lats()
                    print('v lons_lats: '+str(lons)+', '+str(lats))
                    if (np.nanmax(np.abs(lons)) <= 360.0)\
                       and (np.nanmax(np.abs(lats)) <= 90.0):
                        # o = krigeBox.overlap(v)
                        o = searchBox.overlap(v)
                        for _o in o:
                            print('o overlap: '+str(_o.lons_lats()))
                        if len(o) > 0:
                            for kb in o:
                                if not kb.emptyp():
                                    if v not in src_data.values():
                                        # print 'adding '+i+' : '+str(v.lons_lats())
                                        src_data[i] = v
                                        # load the data
                                        if _verbose:
                                            print 'loading '+i+' from '+SRC_DIRECTORY
                                            # print 'type(i): '+str(type(i))
                                        modis_obj = df.DataField(\
                                                                 datafilename=i\
                                                                 ,datafieldname=DATAFIELDNAME\
                                                                 ,srcdirname=SRC_DIRECTORY\
                                                                 ,hack_branchcut_threshold=200\
                                        )
                                        modis_objs.append(modis_obj)
    
            #
            # Get the sizes, allocate the arrays, and then fill them
            sizes_modis_objs      = [ m.data.size for m in modis_objs ]
            total_size_modis_objs = sum(sizes_modis_objs)

            if _verbose:
                print('noggin_krige.py: total number of data points loaded: '+str(total_size_modis_objs))
    
            data      = np.zeros(total_size_modis_objs)
            latitude  = np.zeros(total_size_modis_objs)
            longitude = np.zeros(total_size_modis_objs)
 
            i0=0
            for i in range(len(sizes_modis_objs)):
                i1 = i0 + sizes_modis_objs[i]
                if _verbose:
                    print('adding indexes: ', i0, i1)
                data[i0:i1],latitude[i0:i1],longitude[i0:i1] = modis_objs[i].ravel()
                i0=i1
    
                
            #
            idx = np.where(~np.isnan(data))
            data1      = data[idx]
            latitude1  = latitude[idx]
            longitude1 = longitude[idx]

            #### See start of loop... grid = targetTiles[k]
            gridx,gridy = grid.gridxy()
            in_grid = grid.in_grid(longitude1,latitude1)
            ex_grid = grid.ex_grid(longitude1,latitude1)

            data_mx_in_grid = np.nanmax(data1[in_grid])
        
            # Calculate variogram
    
            nlags=_drive_OKrige_nlags
            #
            dg = gridx.size
            dx = Krige.span_array(gridx)
            dy = Krige.span_array(gridy)
            dr = math.sqrt(dx*dx+dy*dy)

            beta0=1.5*(dr)
            lw_scale = 2.5
            l=lw_scale*(dx/2)
            w=lw_scale*(dy/2)

            # e.g. hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
            if k in hires_calc:
                npts = hires_npts
            else:
                npts = lores_npts

            if _verbose:
                print 'npts( '+str(k)+' ) = '+str(npts)
            
            marker_size = 3.5
            m_alpha = 1.0
            colormap_0 = plt.cm.rainbow
            colormap_1 = plt.cm.gist_yarg
            colormap_2 = plt.cm.plasma
            colormap_x = colormap_0

            #####
            
            inf_detected = False
            max_iter = max_iter_start
            npts_in = npts
            npts_increase_flag = True
            npts_last_kr_s     = 0
            
            while( True ):
                if _verbose:
                    print 'npts_in( '+str(k)+' ) = '+str(npts_in)
                kr=0
                kr=Krige.drive_OKrige(\
                                       grid_stride=dg\
                                       ,random_permute=True\
                                       ,x=gridx,y=gridy\
                                       ,src_x=longitude1\
                                       ,src_y=latitude1\
                                       ,src_z=data1\
                                       ,variogram_model=_drive_OKrige_variogram\
                                       ,variogram_function=vm.variogram_models[_drive_OKrige_variogram].function\
                                       ,nlags=_drive_OKrige_nlags\
                                       ,enable_plotting=_plot_variogram\
                                       ,enable_statistics=_enable_statistics\
                                       ,npts=npts_in\
                                       ,beta0=beta0\
                                       ,frac=0.0\
                                       ,l=l,w=w\
                                       ,weight=_drive_OKrige_weight\
                                       ,verbose=_drive_OKrige_verbose\
                                       ,eps=_drive_OKrige_eps\
                                       ,backend=_drive_OKrige_backend\
                )
                kr.title         = modis_obj.datafilename[0:17]  # TODO Cut up the data filename better
                kr.zVariableName = modis_obj.datafieldname
                kr_mx = np.nanmax(kr.z)
                kr_s_mn = np.nanmin(kr.s)
                kr_s_mx = np.nanmax(kr.s)
                max_iter = max_iter - 1
                print 'kr_s_mn,kr_s_mx,kr_mx,data_mx_in_grid: ',kr_s_mn,kr_s_mx,kr_mx,data_mx_in_grid
                if (max_iter < 1) or ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                    if (max_iter < 1):
                        print '**'
                        if ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                            print '*** max_iter exceeded, continuing to next tile'
                        else:
                            print '*** max_iter exceeded, kriging probably diverged, continuing to next tile'
                    if ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                        print '**'
                        print '*** kriging seems to have converged'
                    break
                else:
                    print '***'
                    print '*** kriging diverged, changing npts, iter: ',max_iter_start-1-max_iter

                    # s_mn should be positive, hopefully small
                    if np.abs(kr_s_mn) > npts_last_kr_s:
                        print('kr_s_mn increased, changing npts change direction')
                        npts_increase_flag = not npts_increase_flag
                    npts_last_kr_s = np.abs(kr_s_mn)
                    if npts_increase_flag:
                        print('increasing npts by '+str(npts_increase_factor))
                        npts_in = npts_in*npts_increase_factor
                    else:
                        print('decreasing npts by '+str(npts_decrease_factor))
                        npts_in = npts_in*npts_decrease_factor
                        
                    # if np.inf in kr.z:
                    #     if inf_detected:
                    #         print 'inf detected again, increasing npts'
                    #         npts_in = npts_in*npts_increase_factor
                    #         inf_detected = False
                    #     else:
                    #         print 'inf detected, reducing npts'
                    #         npts_in = npts_in*0.75
                    #         inf_detected = True
                    # else:
                    #     print 'increasing npts'
                    #     npts_in = npts_in*npts_increase_factor
                                 
            krigeSketch_results.append(kr)

            if _verbose:
                print 'k,mnmx(gridz): '+str(k)\
                    +', ( '+str(np.nanmin(krigeSketch_results[-1].z))\
                    +', '+str(np.nanmax(krigeSketch_results[-1].z))+' )'
    

if True:
    l=0
    for k in krigeSketch_results:
        print 'mnmx(k): '+str(l)+': ( '+str(np.nanmin(k.z))+' '+str(np.nanmax(k.z))+str(' )')
        l=l+1
    l=0
    for k in krigeSketch_results:
        print 'vg_parm: '+str(l)+': '+str(k.vg_parameters)
        l=l+1

# plot_configuration = Krige.krigePlotConfiguration(source_data              = False\
#                                                   ,source_data_last_sample = False\
#                                                   ,title                   = '.'.join(krigeSketch_results[0].title.split('.',3)[0:2])\
#                                                   ,zVariableName           = krigeSketch_results[0].zVariableName\
#                                                   ,vmap                    = Krige.log10_map\
#                                                   ,meridians_and_parallels = False\
# )

# Save an HDF file
# TODO The following is currently broken
# TODO Apparently SWATH does not mean irregular.
if True:
    print('KrigeSketch saving to HDF')
    # Now, we use tgt_X1d and tgt_Y1d
    ny = tgt_Y1d.size
    nx = tgt_X1d.size

    # The following, no.
    x    = np.zeros((ny,nx)); x.fill(np.nan)
    y    = np.zeros((ny,nx)); y.fill(np.nan)

    # Yes, the following.
    z    = np.zeros((ny,nx)); z.fill(np.nan)
    s    = np.zeros((ny,nx)); s.fill(np.nan)
    nans = np.zeros((ny,nx)); nans.fill(np.nan)

    # The following is incorrect.
    i=0
    for kr in krigeSketch_results:
        for k in range(kr.x.size):
            tgt_x_idx = np.where(np.abs(tgt_X1d - kr.x[k]) < 1.0e-10)
            tgt_y_idx = np.where(np.abs(tgt_Y1d - kr.y[k]) < 1.0e-10)

            if (len(tgt_x_idx) != 1) or (len(tgt_y_idx) != 1):
                print('*** tgt_?_idx error. ')
                print('*** tgt_x_idx = '+str(tgt_x_idx))
                print('*** tgt_y_idx = '+str(tgt_y_idx))
                print('*** skipping')
            else:
                x[tgt_y_idx[0],tgt_x_idx[0]] = kr.x[k]
                y[tgt_y_idx[0],tgt_x_idx[0]] = kr.y[k]
                z[tgt_y_idx[0],tgt_x_idx[0]] = kr.z[k]
                s[tgt_y_idx[0],tgt_x_idx[0]] = kr.s[k]
        
    variable_name   = krigeSketch_results[-1].zVariableName

    # Note the config below should be improved. Check that the vars are being saved correctly to HDF.
    kHDF = Krige.krigeHDF(\
                          krg_name                 = variable_name+'_krg'\
                          ,krg_units               = modis_obj.units\
                          ,config                  = krigeSketch_results[-1].config\
                          ,krg_z                   = z\
                          ,krg_s                   = s\
                          ,krg_x                   = tgt_X1d\
                          ,krg_y                   = tgt_Y1d\
                          ,orig_name               = modis_obj.datafieldname\
                          ,orig_units              = modis_obj.units\
                          ,output_filename         = output_filename\
                          ,redimension             = False\
                          ,type_hint               = 'grid'\
    )
    kHDF.save()
    print('KrigeSketch finished saving to HDF')


print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
end_time = time.time()
print 'wall clock run time (sec) = '+str(end_time-start_time)

print 'KrigeSketch done'




###########################################################################
# if __name__ == '__main__':
#     print('---noggin_krige start---')
#     # print('s: ',args.samplingFraction)
# 
# 
#     
#     print('---noggin_krige done---')
