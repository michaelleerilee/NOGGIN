#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Krige all the given files in a directory to a grid.

Examples:

## THE FOLLOWING IS A GOOD FIRST EXAMPLE FOR A SMALLER MACHINE
python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MYD08_D3/ -n Atmospheric_Water_Vapor_Mean -m gamma_rayleigh_nuggetless_variogram_model -v -G

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2/ -n Water_Vapor_Infrared -m gamma_rayleigh_nuggetless_variogram_model -v

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2-1/ -n Water_Vapor_Infrared -m spherical -v -r 0.25 -R

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2-1/ -n Water_Vapor_Infrared -m spherical -v -r 0.25 -R -b -50 -50 -40 -40

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2-1/ -n Water_Vapor_Infrared -m gamma_rayleigh_nuggetless_variogram_model -v -r 0.25 -R -b -50 -50 -40 -40

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2-1/ -n Water_Vapor_Infrared -m spherical -v -r 0.25 -R -b -90 -60 0.0 0.0

# Test a very coarse grid that does not align with the computational tiling.
python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2/ -n Water_Vapor_Infrared -m gamma_rayleigh_nuggetless_variogram_model -r 8.0 -s 0.001 -x -v

python ~/git/NOGGIN-github/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}MODIS-61-MOD05_L2/ -n Water_Vapor_Infrared -m gamma_rayleigh_nuggetless_variogram_model -r 9.0 -s 0.001 -x -v

------

2018-0808 ML Rilee, RSTLLC, mike@rilee.net

Copyright © 2018-2021 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

"""

import argparse
from collections import Counter
import json
import math
import os
import sys
import time
from time import gmtime, strftime

import Krige
import Krige.DataField as df
import DataField as df

# from DataField import DataField, BoundingBox, df.Point, box_covering, Polygon, data_src_directory

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from Krige import fig_generator

import pykrige.variogram_models as vm

###########################################################################

# https://stackoverflow.com/questions/21888406/getting-the-indexes-to-the-duplicate-columns-of-a-numpy-array
def unique_columns2(data):
    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u,uind_fwd,uind,ucounts = np.unique(dataf, return_index=True, return_inverse=True, return_counts=True)
    u = u.view(data.dtype).reshape(-1,data.shape[0]).T
    return (u,uind_fwd,uind,ucounts)

###########################################################################

start_time = time.time()

parser = argparse.ArgumentParser(description='Capture krige control parameters.')

parser.add_argument('-d','--input_directory'\
                    ,dest='inputDirectory'\
                    ,type=str\
                    ,required=True\
                    ,help='The directory to search for input files.')

parser.add_argument('-f','--src_file_list'
                        ,dest='src_file_list'
                        ,required=False
                        ,help='File names and associated geolocation file names')
parser.set_defaults(src_file_list=None)

parser.add_argument('-m','--variogram_model'\
                    ,dest='variogram_model'\
                    ,type=str\
                    ,required=True\
                    ,help='The name of the variogram model to use. [e.g. "spherical","gamma_rayleigh_nuggetless_variogram_model"]')

parser.add_argument('-n','--variable_name'\
                    ,dest='datafieldname'\
                    ,type=str\
                    ,required=True\
                    ,help='The variable or datafield name to load.')

parser.add_argument('-o','--output_filename'\
                    ,dest='output_filename'\
                    ,type=str\
                    ,help='The output file for the results')
parser.set_defaults(output_filename = 'noggin_krige.hdf')

parser.add_argument('-s','--sampling_fraction'\
                    ,dest='sampling_fraction'\
#                    ,metavar='samplingFraction'\
                    ,type=float, help='Suggested fraction of data to sample.')

parser.add_argument('-S','--Sampling_number'\
                    ,dest='lores_npts'\
                    ,type=int\
                    ,help='Set a target for the number of points to be sampled from the source data. Conflicts with sampling_fraction.')

parser.add_argument('-l','--variogram_bin_number'\
                    ,dest='variogram_nlags'\
                    ,type=int\
                    ,help='The number of bins (of lags) for the variogram calculation. Default 8.')
parser.set_defaults(variogram_nlags=8)
# parser.set_defaults(variogram_nlags=12)

parser.add_argument('-v','--verbose'\
                    ,dest='verbose'\
                    ,action='store_true'\
                    ,help='Toggle verbose printing')
parser.set_defaults(verbose=False)

parser.add_argument('-x','--debug'\
                    ,dest='debug'\
                    ,action='store_true'\
                    ,help='Toggle debug printing')
parser.set_defaults(debug=False)

parser.add_argument('-r','--resolution'\
                    ,dest='grid_resolution'\
                    ,type=float\
                    ,help='The resolution for a lat-lon target grid in degrees.')
parser.set_defaults(grid_resolution=1.0)

parser.add_argument('-G','--GapFill'\
                    ,dest='gap_fill'\
                    ,action='store_true'\
                    ,help='Perform a gap filling calculation on one file of Level 3 data.')
parser.set_defaults(gap_fill=False)

parser.add_argument('-R','--Restrict'\
                    ,dest='restrict_to_bounding_box'\
                    ,action='store_true'\
                    ,help='Restricts the target grid to a bounding box. Defaults to a bounding box around source data.')
parser.set_defaults(restrict_to_bounding_box=False)

parser.add_argument('-b','--bounding_box'\
                    ,dest='bounding_box'\
                    ,type=float, nargs=4\
                    ,help='Define a target region of interest, lon0, lat0, lon1, lat1. No effect without --Restrict. NOT TESTED.')

parser.add_argument('-B','--Beta'
                    ,dest='beta0'
                    ,type=float, nargs=1
                    ,help='Variogram scale parameter')
parser.set_defaults(beta0=1)

parser.add_argument('-L','--lw_scale'
                    ,dest='lw_scale'
                    ,type=float, nargs=1
                    ,help='Fractional scale of target dimensions 0.5*(length,width) to set data input window.')
parser.set_defaults(lw_scale=2.5)

# parser.add_argument('-S','--SingleTile'\
#                     ,dest='single_tile'\
#                     ,action='store_true'\
#                     ,help='Krige to a single tile. Useful for Level 3 calculations. Implied by GapFill. NOT IMPLEMENTED.')
# parser.set_defaults(single_tile=False)

# parser.add_argument(''\
#                     )
# parser.set_defaults()

# parser.add_argument('-a','--adapt_fraction'\
#                     ,dest='adaptFraction'\
#                     ,metavar='adaptFraction'\
#                     ,type=bool, help='Vary the sampling fraction over iterations.')



if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
    
args=parser.parse_args()

_flag_error_exit=False

if (args.sampling_fraction is not None) and (args.lores_npts is not None):
    print('noggin_krige: error: sampling fraction and sampling number are both set. One or none should be set.')
    _flag_error_exit=True

if args.datafieldname is None:
    print('noggin_krige: error: variable/datafieldname -n must be specified.')
    _flag_error_exit=True

#if args.variogram_nlags is None:
#    print('noggin_krige: error: -l variogram_nlags must be specified.')
#    _flag_error_exit=True

if args.variogram_model is None:
    print('noggin_krige: error: -m --variogram_model must be specified. e.g. "spherical" or "gamma_rayleigh_nuggetless_variogram_model"')
    _flag_error_exit=True


output_filename = args.output_filename
_verbose = args.verbose
_debug   = args.debug

if _flag_error_exit:
    print('noggin_krige: error sys.exit(1)')
    sys.exit(1)
    
_save_index = False

SRC_DIRECTORY=args.inputDirectory
DATAFIELDNAME=args.datafieldname
SRC_FILE_LIST=args.src_file_list

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
# _drive_OKrige_backend           = 'loop' # Same error pattern as vectorized

# tgt_grid_dLon = 0.5; tgt_grid_dLat = 0.5
# These might conflict with GapFill.
tgt_grid_dLon                   = args.grid_resolution
tgt_grid_dLat                   = args.grid_resolution

beta0                           = args.beta0[0]
lw_scale                        = args.lw_scale[0]
# print('beta0: ',type(lw_scale),lw_scale)
# exit()
###########################################################################

_sampling_fraction              = args.sampling_fraction
if _sampling_fraction is None:
    # lores_npts = 2000
    if args.lores_npts is None:
        lores_npts = 3500
    else:
        lores_npts = args.lores_npts
else:
    lores_npts = None


###########################################################################
# Start with MetaData sketch.
# Construct bounding box index.
src_file_list = []
geo_file_list = []
if SRC_FILE_LIST is None:
    src_file_list = [f for f in os.listdir(SRC_DIRECTORY)
                        if (lambda x:
                                '.hdf' in x
                                or '.HDF.' in x
                                or '.he5' in x
                                or '.h5' in x
                                or '.nc' in x
                                )(f)]
else:
    with open(SRC_FILE_LIST,'r') as f:
        while(True):
            line = f.readline()
            if not line:
                break
            line = line.rstrip()
            try:
                src,geo = line.split(' ')
            except ValueError:
                src = line
                geo = ""
                pass
            src_file_list.append(src)
            geo_file_list.append(geo)
    
if len(src_file_list) == 0:
    print('noggin_krige: error: no HDF files found in '+SRC_DIRECTORY+', exiting')
    sys.exit(1)
modis_BoundingBoxes = {}
if _save_index:
    modis_BoundingBoxes_json = {}
bb = df.BoundingBox()

if _verbose:
    print('noggin_krige: number of files to load: '+str(len(src_file_list)))
geo_k = 0
for i in src_file_list:
    if _verbose:
        print('noggin_krige: loading '+str(i)+' from '+str(SRC_DIRECTORY))

    geofile = ""
    if len(geo_file_list) > 0:
        geofile = geo_file_list[geo_k]
        geo_k+=1

    modis_obj = df.DataField(\
                                 datafilename=i\
                                 ,datafieldname=DATAFIELDNAME\
                                 ,srcdirname=SRC_DIRECTORY\
                                 ,hack_branchcut_threshold=200\
                                 ,geofile=geofile\
                                 )
    bb = bb.union(modis_obj.bbox)
    modis_BoundingBoxes[i]=modis_obj.bbox
    if _save_index:
        modis_BoundingBoxes_json[i]=modis_obj.bbox.to_json()

    if _debug:
        print('noggin_krige:debug: loading '+str(i)+' from '+str(SRC_DIRECTORY))        
        # print 'noggin_krige:debug:xml: ',bb.to_xml()
        print('noggin_krige:debug:json: ',bb.to_json())

if _save_index:
    if _verbose:
        print('writing to '+INDEX_FILE)
        # with open(SRC_DIRECTORY+'modis_BoundingBoxes.json','w') as f:
    with open(INDEX_FILE,'w') as f:
        json.dump(modis_BoundingBoxes_json,f)

###########################################################################
# Move to KrigeSketch conventions
boxes = modis_BoundingBoxes

###########################################################################
# Check on gap fill case.
if args.gap_fill:
    if len(boxes) != 1:
        print('noggin_krige: error: processing more than one level 3 file at a time not implemented')
        sys.exit(1)
    obj_shape = modis_obj.data.shape
    ny = obj_shape[0]; test_dy = 180.0/ny
    nx = obj_shape[1]; test_dx = 360.0/nx
    if test_dy != test_dx:
        print('noggin_krige: error: unexpected data shape aspect ratio.')
        print('noggin_krige: error: nx,ny = '+str(nx)+', '+str(ny))
        print('noggin_krige: error: dx,dy = '+str(test_dx)+', '+str(test_dy))
        sys.exit(1)
    tgt_grid_dLon = test_dx
    tgt_grid_dLat = test_dy
    
# sys.exit(1)
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
# max_iter_start = 5
# max_iter_start = 8
max_iter_start = 3
k = -1

divergence_threshold = 1.5
npts_increase_factor = 1.5
npts_decrease_factor = 0.75

###########################################################################
# Define full grid (tiles & caps)
# ### FULL with bands and caps
# TODO Inject dependencies to parameterize

###########################################################################

# Some defaults for the following...
hires_npts_scale = 2
hires_calc = []

###########################################################################
## TODO use bb to limit the following
# if args.restrict_to_bounding_box:
if args.restrict_to_bounding_box:
    if args.bounding_box is None:
        bb_lons,bb_lats = bb.lons_lats()
    else:
        bb_lons = [args.bounding_box[0],args.bounding_box[2]]; bb_lons.sort()
        bb_lats = [args.bounding_box[1],args.bounding_box[3]]; bb_lats.sort()
    lon0 = bb_lons[0]
    lon1 = bb_lons[1]
    lat0 = bb_lats[0]
    lat1 = bb_lats[1]
    dLon = lon1-lon0
    dLat = lat1-lat0
    #if np.abs(dLon) > 180 or np.abs(dLat) > 90:
    #    print('noggin_krige: error: source data bounding box too big or malformed. exiting.')
    #    sys.exit(1)
    # dSearchLon = 0.75*dLon
    # dSearchLat = 0.75*dLat
    dSearchScale = 0.25
    hires_npts_scale = 2

###########################################################################


# TODO: args.gap_fill may not be orthogonal to restrict_to_bounding_box -- maybe want both...
elif args.gap_fill:
    dLon = 360
    dLat = 180
    lon0 = -180; lon1 = 180; lat0 = -90; lat1 = 90    
    dSearchScale = 0.25 # unnecessary for 1-file level 3 gap filling.
    hires_npts_scale = 2
    hires_calc = []
    divergence_threshold = 1.5
    npts_increase_factor = 1.5
    npts_decrease_factor = 0.75
    
###########################################################################
##
# else or if not args.restrict_to_bounding_box:
# DEFAULT
# TODO: Inject dependencies on these hardcoded parameters
else:
    # default dLon = 120; dLat = 10
    dLon = 120; dLat = 40
    # dSearchScale = 0.75
    dSearchScale = 0.25 # Amount beyond the tile to search for granules.
    # Full
    lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  90-dLat
    # 4-box test    lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  -60-dLat
    # lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 = -60-dLat
    # lores_npts = 2000
    # lores_npts = 3000
    # lores_npts = 3500
    hires_npts_scale = 2
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
                                ,df.Point((180, 90+1.0e-6))))
#                                ,df.Point((180, 90+tgt_grid_dLat))))
    targetBoxes.append(cap_north)

dSearchLon = dSearchScale*dLon
dSearchLat = dSearchScale*dLat

# TODO check lon? lat? etc to see if we can convert to int.
    
###########################################################################
# Add the other boxes.
if type(lon0) is int and type(lon1) is int and type(dLon) is int\
   and type(lat0) is int and type(lat1) is int and type(dLat) is int:
    if _debug:
        print('noggin_krige:debug:box with integer parameters')
    for iLon in range(lon0,lon1,dLon):
        for jLat in range(lat0,lat1,dLat):
            krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
                                       ,df.Point((iLon+dLon, jLat+dLat))))
            if _debug:
                # print('box-latlon: ['+str(iLon)+', '+str(jLat)+'] ['+str(iLon+dLon)+', '+str(jLat+dLat)+']')
                print(krigeBox.to_json())
            targetBoxes.append(krigeBox)
else:
    print('noggin_krige:debug:box with non-integer parameters')
    iLon=lon0
    while iLon < lon1:
        jLat=lat0
        while jLat < lat1:
            krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
                                       ,df.Point((iLon+dLon, jLat+dLat))))
            if _debug:
                # print('box-latlon: ['+str(iLon)+', '+str(jLat)+'] ['+str(iLon+dLon)+', '+str(jLat+dLat)+']')
                print(krigeBox.to_json())
            targetBoxes.append(krigeBox)
            jLat += dLat
        iLon += dLon
# quit()
###########################################################################
if _debug:
    print('lon0,lon1,dLon: ',lon0,lon1,dLon)
    print('lat0,lat1,dLat: ',lat0,lat1,dLat)
    print('noggin_krige: len(targetBoxes) = '+str(len(targetBoxes)))
###########################################################################


# Now we have boxes. Including the caps.

# Construct the target grid from targetBoxes.
# Make a bounding box covering everything, and then grid that at a given resolution.

# TODO: NOTE !!! If we are gap_fill, shouldn't we use the lon-lat from modis_obj?

cover = df.BoundingBox()
for box in targetBoxes:
    cover = cover.union(box)
tgt_lons,tgt_lats = cover.lons_lats()
if _debug:
    print('noggin_krige: tgt lons,lats: '+str(tgt_lons)+', '+str(tgt_lats))
    
if args.gap_fill:
    tgt_X1d = modis_obj.longitude[0,:]
    tgt_Y1d = modis_obj.latitude [:,0]
else:
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

if _debug:
    print('noggin_krige: tgt lon0,lat0: '+str(tgt_lon0)+', '+str(tgt_lat0))
    print('noggin_krige: tgt lon1,lat1: '+str(tgt_lon1)+', '+str(tgt_lat1))
    print('tgt_grid_dLon              : '+str(tgt_grid_dLon))
    print('tgt_grid_dLat              : '+str(tgt_grid_dLat))

nx = (tgt_lon1-tgt_lon0)/tgt_grid_dLon
ny = (tgt_lat1-tgt_lat0)/tgt_grid_dLat
if np.abs(nx - int(nx)) > 1.0e-8:
    print('Error: grid_dLon does not evenly divide longitude span.')
    print('   nx  = ',nx)
    print('tgtxsh = ',tgt_X1d.shape)
    
if np.abs(ny - int(ny)) > 1.0e-8:
    print('Error: grid_dLat does not evenly divide latitude span.')
    print('   ny = ',ny)
    print('tgtysh = ',tgt_Y1d.shape)

###########################################################################

# Construct individual tiles. Hope they line up with tgt_grid_full.
k=0
for krigeBox in targetBoxes:
    kb_lons,kb_lats = krigeBox.lons_lats()
    tile_lons_idx = np.where((kb_lons[0] <= tgt_X1d) & (tgt_X1d <= kb_lons[1]))
    tile_lats_idx = np.where((kb_lats[0] <= tgt_Y1d) & (tgt_Y1d <= kb_lats[1]))   

    # TODO May be inconsistent
    tile = Krige.rectangular_grid(\
                                  x0  = np.nanmin(tgt_X1d[tile_lons_idx])\
                                  ,x1 = np.nanmax(tgt_X1d[tile_lons_idx])\
                                  ,dx = tgt_grid_dLon\
                                  ,x1d= tgt_X1d[tile_lons_idx]\
                                  ,y0 = np.nanmin(tgt_Y1d[tile_lats_idx])\
                                  ,y1 = np.nanmax(tgt_Y1d[tile_lats_idx])\
                                  ,dy = tgt_grid_dLat\
                                  ,y1d= tgt_Y1d[tile_lats_idx]\
    )
    # TODO fix this up so it would have its own class
    # TODO if no target points are in the tile, then don't add it.
    
    if _debug:
        print('noggin_krige: k tile.mnmx: '+str(k)+', '+str(tile.mnmx()))
    targetTiles.append(tile)
    k += 1
    # if k > 18:
    #     break

###########################################################################

# Trying to save the (swath) data for later viewing doesn't work yet.
# concatenate_data=True
concatenate_data=False
if concatenate_data:
    lon_save = None
    lat_save = None
    dat_save = None
    
    
### TODO: Verify the size of tgt_?1d match the calculated result.
    
k=-1
for krigeBox in targetBoxes:
    k=k+1
    print( 'fetching tile = '+str(k))
    if k < len(targetTiles):
    # for dummy in [1]:
# check k for debug        
        grid = targetTiles[k]
        _calculate = True
        _enable_statistics = _drive_OKrige_enable_statistics
        if _calculate:
            if _verbose:
                print( 'working on tile = '+str(k))

            iLon=krigeBox.p0.lon_degrees
            jLat=krigeBox.p0.lat_degrees

            dLon=krigeBox.p1.lon_degrees-krigeBox.p0.lon_degrees
            dLat=krigeBox.p1.lat_degrees-krigeBox.p0.lat_degrees

            if _verbose:
                print( 'loading iLon,jLat: '+str(iLon)+','+str(jLat))
                print( 'loading dLon,dLat: '+str(dLon)+','+str(dLat))
                print( strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
            
            # krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
            #                             ,df.Point((iLon+dLon, jLat+dLat))))
    
            searchBox = df.BoundingBox(( df.Point((iLon-dSearchLon,      max(-90, min(jLat-dSearchLat, 90))))\
                                        ,df.Point((iLon+dLon+dSearchLon, max(-90, min(jLat+dLat+dSearchLat, 90))))))
    
            # krigeBox = df.BoundingBox((df.Point((+135.0, 30.0))\
            #                            ,df.Point((+175.0, 45.0))))

            if _debug:
                sb_lons,sb_lats = searchBox.lons_lats()
                print('sb lons_lats: '+str(sb_lons)+', '+str(sb_lats))
    
            # Find overlapping granules and load the MODIS objects
            src_data   = {}
            modis_objs = []
            # for i,v in boxes.iteritems():
            for i,v in boxes.items():
                i = str(i)
                # if i[0:3] in _load_datasets: # Check to see if we're loading something DataField knows about.
                if True:
                    lons,lats = v.lons_lats()
                    if _debug:
                        print('v lons_lats: '+str(lons)+', '+str(lats))
                    if (np.nanmax(np.abs(lons)) <= 360.0)\
                       and (np.nanmax(np.abs(lats)) <= 90.0):
                        # o = krigeBox.overlap(v)
                        o = searchBox.overlap(v)
                        if _debug:
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
                                            print( 'main: loading '+i+' from '+SRC_DIRECTORY)
                                            # print 'type(i): '+str(type(i))
                                        modis_obj = df.DataField(\
                                                                     datafilename=i\
                                                                     ,datafieldname=DATAFIELDNAME\
                                                                     ,srcdirname=SRC_DIRECTORY\
                                                                     ,hack_branchcut_threshold=200\
                                                                     ,geofile=geofile\
                                                    )
                                        modis_objs.append(modis_obj)

            print('len(modis_objs): ',len(modis_objs))
            #
            # Get the (slice) sizes, allocate the arrays, and then fill them
            # sizes_modis_objs      = [ m.data.size for m in modis_objs ]
            sizes_modis_objs      = [ m.slice_size for m in modis_objs ]
            total_size_modis_objs = sum(sizes_modis_objs)

            if _sampling_fraction is not None:
                lores_npts = _sampling_fraction * total_size_modis_objs

            if _verbose:
                print('noggin_krige.py: total number of data points loaded: '+str(total_size_modis_objs))
    
            data      = np.zeros(total_size_modis_objs)
            latitude  = np.zeros(total_size_modis_objs)
            longitude = np.zeros(total_size_modis_objs)

            print("len(modis_objs[i].data.shape)",len(modis_objs[0].data.shape))
            # exit()
 
            # TODO Set islice via CL parameter
            # islice = 0
            islice = 9
            i0=0
            for i in range(len(sizes_modis_objs)):
                i1 = i0 + sizes_modis_objs[i]
                if _verbose:
                    print('adding indexes: ', i0, i1)
                if len(modis_objs[i].data.shape) == 3:
                    print('extracting data slice: ',islice)
                    data[i0:i1],latitude[i0:i1],longitude[i0:i1] = modis_objs[i].ravel_slice(islice)
                else:
                    print('extracting data')
                    data[i0:i1],latitude[i0:i1],longitude[i0:i1] = modis_objs[i].ravel()
                i0=i1

            # # The data have been loaded. You can search for duplicates now.
            # if False:
            #     dup_lons = [ lon for lon,count in Counter(longitude).iteritems() if count > 1 ]
            #     if len(dup_lons) > 0:
            #         dup_lats = [ lat for lat,count in Counter(latitude).iteritems() if count > 1 ]
            #         if len(dup_lats) > 0:
            #             dup_lons_idx     = np.where(np.in1d(longitude,dup_lons)); # dup_lons_idx.sort()
            #             dup_lats_idx     = np.where(np.in1d(latitude, dup_lats)); # dup_lats_idx.sort()
            #             dup_lonlats_idx  = np.where(np.in1d(dup_lons_idx,dup_lats_idx))
            #             dups_idx = dup_lons_idx[dup_lonlats_idx]:
            #             for ii = range(len(dups_idx)):
            #                 ln0 = longitude[dups_idx[ii]]
            #                 lt0 = latitude [dups_idx[ii]]
            #                 for jj = range(1,len(dups_idx)):
            #                     
            #                 
            #                 sum = 0
            #                 n   = 0
            #                 for dupi in dup_lons_idx[dup_lonlats_idx]:
                            
            #
            if not args.gap_fill:
                idx_source = np.where(~np.isnan(data))
            else:
                idx_source = np.where( ~np.isnan(data)\
                                      & (latitude > lat0) \
                                      & (latitude < lat1) \
                                      & (longitude > lon0) \
                                      & (longitude < lon1))
                idx_target = np.where(  np.isnan(data)\
                                      & (latitude > lat0) \
                                      & (latitude < lat1) \
                                      & (longitude > lon0) \
                                      & (longitude < lon1))

            longitude_tmp = longitude[idx_source]
            latitude_tmp  = latitude[idx_source]
            data_tmp      = data[idx_source]

            #### Average over duplicates... Experimental...
            ulonlat, ulonlat_ind_fwd, ulonlat_ind, ulonlat_count = unique_columns2(np.vstack((longitude_tmp,latitude_tmp)))
            if np.nanmax(ulonlat_count) > 1:
                print('noggin_krige: found at least one duplicate!')
                longitude_tmp1 = ulonlat[0]
                latitude_tmp1  = ulonlat[1]
                data_tmp1 = np.zeros(longitude_tmp1.shape[0])
                for i in range(ulonlat_ind.size):
                    data_tmp1[ulonlat_ind[i]] += data_tmp[i]/ulonlat_count[ulonlat_ind[i]]
            else:
                longitude_tmp1 = longitude_tmp
                latitude_tmp1  = latitude_tmp
                data_tmp1      = data_tmp

            #### And the poles
            longitude_mask = np.arange(longitude_tmp1.shape[0])
            longitude_mask[np.where(latitude_tmp1 == -90)] = -1000
            longitude_mask[np.where(latitude_tmp1 ==  90)] = -2000
            ulonlat, ulonlat_ind_fwd,ulonlat_ind, ulonlat_count = unique_columns2(np.vstack((longitude_mask,latitude_tmp1)))
            #??? ulonlat, ulonlat_ind, ulonlat_count = unique_columns2(latitude_tmp1)???
            if _verbose:
                print('noggin_krige: np.nanmax(ulonlat_count): '+str(np.nanmax(ulonlat_count)))
                print(' where == -90: '+str(len(np.where(latitude_tmp1 == -90))))
                print(' where == 90: '+str(len(np.where(latitude_tmp1 ==  90))))
            if np.nanmax(ulonlat_count) > 1:
                print('noggin_krige: found at least one duplicate at a pole!')
                latitude1  = ulonlat[1]
                longitude1 = longitude_tmp1[ulonlat_ind_fwd]
                data1 = np.zeros(longitude1.shape[0])
                for i in range(ulonlat_ind.size):
                    data1[ulonlat_ind[i]] += data_tmp1[i]/ulonlat_count[ulonlat_ind[i]]
            else:
                longitude1 = longitude_tmp1
                latitude1  = latitude_tmp1
                data1      = data_tmp1
                
            #### See start of loop... grid = targetTiles[k]
            gridx,gridy = grid.gridxy()

            # TODO Fix hijacking of the gridx & gridy
            if args.gap_fill:
                gridx  = longitude[idx_target]
                gridy  = latitude [idx_target]

            # TODO This partis to flag and avoid dry firing.
            in_grid = grid.in_grid(longitude1,latitude1,border=[dSearchLon,dSearchLat])
            # ex_grid = grid.ex_grid(longitude1,latitude1)
            if _debug:
                print('len(in_grid): '+str(len(in_grid)))
                print('in_grid: '+str(in_grid))
            if(len(in_grid[0]) == 0):
                print('noggin_krige: len(in_grid[0]) == 0), no source data. continuing')
                continue
            data_mx_in_grid = np.nanmax(data1[in_grid])
        
            # Calculate variogram
    
            nlags=_drive_OKrige_nlags
            #
            dg = gridx.size
            dx = max(Krige.span_array(gridx),dSearchLon)
            dy = max(Krige.span_array(gridy),dSearchLat)
            dr = math.sqrt(dx*dx+dy*dy)

            if _debug:
                print('noggin_krige:dg,dx,dy,dr: ',dg,dx,dy,dr)

            beta0=beta0*dr # beta0 set in args
            # lw_scale set in args
            l=lw_scale*(dx/2)
            w=lw_scale*(dy/2)

#            # OMI
#            beta0=1.5*(dr) good
#            lw_scale = 2.5
#            l=lw_scale*(dx/2)
#            w=lw_scale*(dy/2)
#
#            # VIIRS
#            beta0=0.04*(dr)
#            lw_scale = 2.5
#            l=lw_scale*(dx/2)
#            w=lw_scale*(dy/2)
 
            # e.g. hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
            if k in hires_calc:
                npts = hires_npts_scale * lores_npts
            else:
                npts = lores_npts

            if _verbose:
                print( 'npts( '+str(k)+' ) = '+str(npts))
            
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

            ####
            if concatenate_data:
                if lon_save is None:
                    lon_save = longitude1.copy()
                    lat_save = latitude1.copy()
                    dat_save = data1.copy()
                else:
                    print('lon_s shape: ',lon_save.shape)
                    print('lon_s type:  ',type(lon_save))
                    print('lon_s dtype: ',lon_save.dtype)
                    print('lon_1 shape: ',longitude1.shape)                    
                    lon_save = np.concatenate((lon_save,longitude1))
                    lat_save = np.concatenate((lat_save,latitude1))
                    dat_save = np.concatenate((dat_save,data1))
            ####
            
            while( True ):
                if _verbose:
                    print( 'npts_in( '+str(k)+' ) = '+str(npts_in))
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

                print('kr.z.shape,strides: ',kr.z.shape,kr.z.strides)
                print('kr.s.shape,strides: ',kr.s.shape,kr.s.strides)
                
                kr.title         = modis_obj.datafilename[0:17]  # TODO Cut up the data filename better
                kr.zVariableName = modis_obj.datafieldname
                kr_x_mn = np.nanmin(kr.x)
                kr_x_mx = np.nanmax(kr.x)
                kr_y_mn = np.nanmin(kr.y)
                kr_y_mx = np.nanmax(kr.y)
                kr_mx = np.nanmax(kr.z)
                kr_s_mn = np.nanmin(kr.s)
                kr_s_mx = np.nanmax(kr.s)
                max_iter = max_iter - 1
                print( 'noggin_krige: kr_s_mn,kr_s_mx,kr_mx,data_mx_in_grid: ',kr_s_mn,kr_s_mx,kr_mx,data_mx_in_grid)
                print( 'noggin_krige: kr mnmx x,y: ',kr_x_mn,kr_x_mx,kr_y_mn,kr_y_mx)
                if (max_iter < 1) or ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                    if (max_iter < 1):
                        print( '**')
                        if ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                            print( '*** max_iter exceeded, continuing to next tile')
                        else:
                            print( '*** max_iter exceeded, kriging probably diverged, continuing to next tile')
                    if ((kr_mx < divergence_threshold * data_mx_in_grid) and (kr_s_mn >= 0)):
                        print( '**')
                        print( '*** kriging seems to have converged')
                    break
                else:
                    print( '***')
                    print( '*** kriging diverged, changing npts, iter: ',max_iter_start-1-max_iter)

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
                    #         print( 'inf detected again, increasing npts')
                    #         npts_in = npts_in*npts_increase_factor
                    #         inf_detected = False
                    #     else:
                    #         print( 'inf detected, reducing npts')
                    #         npts_in = npts_in*0.75
                    #         inf_detected = True
                    # else:
                    #     print( 'increasing npts')
                    #     npts_in = npts_in*npts_increase_factor
                                 
            kr.config.parent_command = sys.argv
            kr.config.iteration      = k
            kr.config.files_loaded   = [obj.datafilename for obj in modis_objs]
            kr.config.datafieldname  = kr.zVariableName
            krigeSketch_results.append(kr)

            if _verbose:
                print( 'k,mnmx(gridz): '+str(k)\
                    +', ( '+str(np.nanmin(krigeSketch_results[-1].z))\
                    +', '+str(np.nanmax(krigeSketch_results[-1].z))+' )')
    

if _verbose:
    l=0
    for k in krigeSketch_results:
        print( 'mnmx(k): '+str(l)+': ( '+str(np.nanmin(k.z))+' '+str(np.nanmax(k.z))+str(' )'))
        l=l+1
    l=0
    for k in krigeSketch_results:
        print( 'vg_parm: '+str(l)+': '+str(k.vg_parameters))
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
    print('noggin_krige saving to HDF')
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
        print('mnmx:lon,lat: ['\
              +str(np.nanmin(kr.x))+', '\
              +str(np.nanmax(kr.x))+']'\
              +'['\
              +str(np.nanmin(kr.y))+', '\
              +str(np.nanmax(kr.y))+'] '\
              +'kr.x.size: '+str(kr.x.size)\
        )
        i=i+1
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

    # TODO: FIX: As a last-minute kluge put all of the files used into the final config.
    # TODO: FIX: Copy items over to a special config to add below.

    all_files_used=[]
    for kr0 in krigeSketch_results:
        all_files_used = all_files_used + kr0.config.files_loaded
    krigeSketch_results[-1].config.files_loaded = list(set(all_files_used))

    if _verbose:
        print('noggin_krige: number of files loaded:     '+str(len(all_files_used)))
        print('noggin_krige: number of individual files: '+str(len(krigeSketch_results[-1].config.files_loaded)))
        if False:
            print('noggin_krige: as json: '+krigeSketch_results[-1].config.as_json())
    
    if not args.gap_fill:
        if concatenate_data:
            # Note the config below should be improved. Check that the vars are being saved correctly to HDF.
            kHDF = Krige.krigeHDF(\
                                krg_name                 = variable_name+'_krg'\
                                ,krg_units               = modis_obj.units\
                                ,config                  = krigeSketch_results[-1].config\
                                ,krg_z                   = z\
                                ,krg_s                   = s\
                                ,krg_x                   = tgt_X1d\
                                ,krg_y                   = tgt_Y1d\
                                ,src_x                   = lon_save\
                                ,src_y                   = lat_save\
                                ,src_z                   = dat_save\
                                ,src_name                = modis_obj.datafieldname\
                                ,src_units               = modis_obj.units\
                                ,orig_name               = modis_obj.datafieldname\
                                ,orig_units              = modis_obj.units\
                                ,output_filename         = output_filename\
                                ,redimension             = False\
                                ,type_hint               = 'swath'\
            )
        else:
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
    else:
        # Note the config below should be improved. Check that the vars are being saved correctly to HDF.
        #
        # This should be a Level 3 case.
        #
        # For args.gap_fill == True, krg_x == orig_x == modis_obj.longitude == tgt_X1d, except for maybe the shape.
        # Here, there should only be one source file and one target file.
        #
        kHDF = Krige.krigeHDF(\
                              krg_name                 = variable_name+'_krg'\
                              ,krg_units               = modis_obj.units\
                              ,config                  = krigeSketch_results[-1].config\
                              ,krg_z                   = z\
                              ,krg_s                   = s\
                              ,krg_x                   = tgt_X1d\
                              ,krg_y                   = tgt_Y1d\
                              ,orig_z                  = modis_obj.data
                              ,orig_x                  = tgt_X1d\
                              ,orig_y                  = tgt_Y1d\
                              ,orig_name               = modis_obj.datafieldname\
                              ,orig_units              = modis_obj.units\
                              ,output_filename         = output_filename\
                              ,redimension             = False\
                              ,type_hint               = 'grid'\
        )

    kHDF.save()
    print('noggin_krige: finished saving to HDF')

#     # Display results
#     fig_gen = fig_generator(1,1)
#     display_obj = df.DataField(\
#                             data = s\
#                             ,longitude = x\
#                             ,latitude  = y\
#                             )
# #                            ,longitude = tgt_X1d\
# #                            ,latitude  = tgt_Y1d\
#     display_obj.scatterplot(title='s',colorbar=True,marker_size=7)
#     plt.show()

print( strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
end_time = time.time()
print( 'wall clock run time (sec) = '+str(end_time-start_time))

print( 'noggin_krige done')




###########################################################################
# if __name__ == '__main__':
#     print('---noggin_krige start---')
#     # print('s: ',args.samplingFraction)
# 
# 
#     
#     print('---noggin_krige done---')
