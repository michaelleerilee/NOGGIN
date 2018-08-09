#!/usr/bin/env python
"""Krige to a box using data catalogued in the metadata file.

2018-0518-1923 ML Rilee, RSTLLC, mike@rilee.net

Important parameters.

* Tile sizes
** The size of each individual tile.
dLon
dLat
** The amount to add to each dimension when selecting data to load.
dSearch


"""

import json
import math
import os
import sys
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

import time

start_time = time.time()

_verbose=True
_debug  =True

_drive_OKrige_weight = True
_drive_OKrige_nlags  = 12
# _drive_OKrige_nlags  = 6
_drive_OKrige_verbose = True
_drive_OKrige_enable_statistics = False
_drive_OKrige_eps = 1.0e-10
# 'vectorized' 'loop' 'C'
_drive_OKrige_backend = 'vectorized'
#bad _drive_OKrige_variogram = 'gamma_rayleigh_variogram_model'
_drive_OKrige_variogram = 'gamma_rayleigh_nuggetless_variogram_model'

_plot_source_data_outside_grid = False
_plot_kriged                   = True
_plot_kriged_outline           = True
_plot_variogram                = False
_plot_meridians_and_parallels   = False

_capture_only      = False
_capture_k         = 17
# _capture_k         = 0
_capture_x         = None
_capture_y         = None
_capture_data1     = None
_capture_x1        = None
_capture_y1        = None
_capture_ex_grid   = None 
_capture_in_grid   = None 
_capture_z         = None
_capture_ss        = None
_catpure_kr        = None
_capture_data_x    = None
_capture_data_y    = None
_capture_data_z    = None

# _load_datasets = ['MYD','MOD']
# _load_datasets = ['MYD']
_load_datasets = ['MOD']

# Choose the source directory for the data and metadata
SRC_DIRECTORY_BASE=df.data_src_directory()
SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
SRC_METADATA=SRC_DIRECTORY+'modis_BoundingBoxes.json'

## class krigeResults(object):
##     def __init__(self\
##                  ,s=None,z=None,x=None,y=None\
##                  ,hull=None\
##                  ,box=None\
##                  ,vg_parameters=None\
##                  ,note='Default note for krigeResults'):
##         self.clear()
##         if x is not None:
##             self.x=x.copy()
##         if y is not None:
##             self.y=y.copy()
##         if z is not None:
##             self.z=z.copy()
##         if s is not None:
##             self.s=s.copy()
##         if hull is not None:
##             self.hull = hull
##         if box is not None:
##             self.box = box.copy()
##         else:
##             self.box = df.BoundingBox()
##         if vg_parameters is not None:
##             self.vg_parameters = vg_parameters
##         self.note = str(note)
##         self.sort_on_longitude()
##         self.construct_hull()
##     def sort_on_longitude(self):
##         """Sort data to avoid bug in basemap. TODO: move into __init__."""
##         if self.x is not None \
##            and self.y is not None \
##            and self.z is not None:
##             idx = self.x.argsort()
##             self.y = self.y[idx[::-1]]
##             self.z = self.z[idx[::-1]]
##             self.x = self.x[idx[::-1]]
##             # self.y = [y for _,y in sorted(zip(self.x,self.y))]
##             # self.z = [z for _,z in sorted(zip(self.x,self.z))]
##             # self.x = sorted(self.x)
##         # else fail silently
##     def clear(self):
##         self.x = None
##         self.y = None
##         self.z = None
##         self.s = None
##         self.hull = None
##         self.box = df.BoundingBox()
##         self.note = 'Default note for krigeResults'
##     def construct_hull(self):
##         """Construct the hull from z,x,y. z is used to get the shape of the data, 
##         so it could be replaced using x and y alone."""
##         if self.z is not None \
##            and self.x is not None \
##            and self.y is not None:
##             xy1 = np.zeros((self.z.shape[0],2))
##             xy1[:,0] = self.x
##             xy1[:,1] = self.y
##             self.hull = ConvexHull(xy1)

print 'KrigeSketch start'
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
# Load the metadata and convert to a dictionary of boxes
print 'loading metadata '+SRC_METADATA
with open(SRC_METADATA,'r') as f:
    boxes_json = json.load(f)
boxes = {}
for i,v in boxes_json.iteritems():
    lons,lats = df.BoundingBox().from_json(v).lons_lats()
    boxes[i] = df.box_covering(lons,lats,hack_branchcut_threshold=180.0)

# Choose box to krige
# Memory Error
# krigeBox = df.BoundingBox((df.Point((+125.0, 30.0))\
#                             ,df.Point((+185.0, 60.0))))
#
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

## TODO: Should we have boxes and tiles in the same structure?
targetBoxes=[]
targetTiles=[]
krigeSketch_results = []
hires_calc = []

# npts_adapt_flag    = True # 
npts_increase_flag = True
# npts_last_kr_s = 0
max_iter_start = 5

divergence_threshold = 1.5
npts_increase_factor = 1.5
npts_decrease_factor = 0.75

# loading_buffer = 5 # for adding some margin to ensure good coverage
dLon = 30
dLat = 30
dSearch = 0.75*dLon
k = -1
#+
### Loop over the tiles.
# ### FULL
# for iLon in range(-180,180,dLon):
#     for jLat in range(-90,90,dLat):
# ### FULL
# ### SMALL CENTER
# for iLon in range(-30,30,dLon):
#     for jLat in range(-30,30,dLat):
# ### SMALL CENTER
# ### SMALL LEFT CORNER
# for iLon in range(-180,-120,dLon):
#     for jLat in range(-90,-30,dLat):
# ### SMALL LEFT CORNER
# for iLon in range(-180,-180+dLon,dLon):
#     for jLat in range(-90,90,dLat):
#+    for jLat in range(-90,90,dLat):
#    for jLat in range(-90,-90+dLat,dLat):
# ### FULL
# lon0 = -180; lon1 = 180; lat0 = -90; lat1 = 90
# lores_npts = 2000
# hires_npts = 16000
# hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
# # ### FULL Baseline 2018-0607 and 0621
# dLon = 30
# dLat = 30
# dSearch = 0.75*dLon
# lon0 = -180; lon1 = 180; lat0 = -90; lat1 = 90
# lores_npts = 2000
# hires_npts = 8000
# # hires_calc = [5,11,37,48,60,61,63,67,68]
# hires_calc = []
# divergence_threshold = 1.5
# npts_increase_factor = 1.5
# 
# # ### FULL Bands
# dLon = 90
# dLat = 10
# dSearch = 0.75*dLon
# lon0 = -180; lon1 = 180; lat0 = -90; lat1 = 90
# lores_npts = 2000
# hires_npts = 8000
# # hires_calc = [5,11,37,48,60,61,63,67,68]
# hires_calc = []
# divergence_threshold = 1.5
# npts_increase_factor = 1.5
# 
# # ### TEST Bands and caps
# dLon = 90
# dLat = 10
# dSearch = 0.75*dLon
# lon0 = -180; lon1 = -90; lat0 = -90+dLat; lat1 = -50
# lores_npts = 2000
# hires_npts = 4000
# # hires_calc = [5,11,37,48,60,61,63,67,68]
# # hires_calc = [0,1]
# hires_calc = []
# divergence_threshold = 1.5
# npts_increase_factor = 1.5
# 
# targetBoxes.append(df.BoundingBox((df.Point((-180, -90))\
#                                     ,df.Point((180, -90+dLat)))))
# 
# targetBoxes.append(df.BoundingBox((df.Point((-180, 90-dLat))\
#                                     ,df.Point((180, 90)))))

# ### FULL with bands and caps
dLon = 120
dLat = 10
dSearch = 0.75*dLon
# Full
lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  90-dLat
# 4-box test lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 =  -60-dLat
# lon0 = -180; lon1 = 180; lat0 = -90+dLat; lat1 = -60-dLat
# lores_npts = 2000
# lores_npts = 3000
lores_npts = 3500
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

# targetBoxes.append(df.BoundingBox((df.Point((iLon, jLat))\
#                                     ,df.Point((iLon+dLon, jLat+dLat)))))

# ### SMALL CENTER
# lon0 = -30; lon1 = 30; lat0 = -30; lat1 = 30
# ### SMALL LEFT CORNER
# lon0 = -180; lon1 = -120; lat0 = -90; lat1 = -30
### SMALL LEFT CORNER+1
# lon0 = -180+15; lon1 = -120+15; lat0 = -90+15; lat1 = -30+15
### 3x3 LEFT CORNER+1
# lores_npts = 2000
# hires_npts = 4000
# hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
# lon0 = -180+15; lon1 = -90+15; lat0 = -90+15; lat1 = -0+15
### A SMALL TEST
# lon0 = -dLon/2; lon1 = +dLon/2; lat0 = -dLat/2; lat1 = +dLat/2
# _capture_k = 0
# lores_npts = 2000
# hires_npts = 4000

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
        _calculate = False
        _enable_statistics = _drive_OKrige_enable_statistics
        if not _capture_only:
            _calculate = True
        else:
            if _capture_k == k:
                _calculate = True
                _enable_statistics = True
        if _calculate:
            print 'working on tile = '+str(k)

            iLon=krigeBox.p0.lon_degrees
            jLat=krigeBox.p0.lat_degrees

            dLon=krigeBox.p1.lon_degrees-krigeBox.p0.lon_degrees
            dLat=krigeBox.p1.lat_degrees-krigeBox.p0.lat_degrees
            
            print 'loading iLon,jLat: '+str(iLon)+','+str(jLat)
            print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
            
            # krigeBox = df.BoundingBox((df.Point((iLon, jLat))\
            #                             ,df.Point((iLon+dLon, jLat+dLat))))
    
            searchBox = df.BoundingBox((df.Point((iLon-dSearch,      max(-90, min(jLat-dSearch, 90))))\
                                        ,df.Point((iLon+dLon+dSearch, max(-90, min(jLat+dLat+dSearch, 90))))))
    
            # krigeBox = df.BoundingBox((df.Point((+135.0, 30.0))\
            #                            ,df.Point((+175.0, 45.0))))
    
    
            # Find overlapping granules and load the MODIS objects
            src_data   = {}
            modis_objs = []
            for i,v in boxes.iteritems():
                i = str(i)
                if i[0:3] in _load_datasets:
                    lons,lats = v.lons_lats()
                    if (np.nanmax(np.abs(lons)) <= 360.0)\
                       and (np.nanmax(np.abs(lats)) <= 90.0):
                        # o = krigeBox.overlap(v)
                        o = searchBox.overlap(v)
                        if len(o) > 0:
                            for kb in o:
                                if not kb.emptyp():
                                    # print 'adding '+i+' : '+str(v.lons_lats())
                                    if v not in src_data.values():
                                        src_data[i] = v
                                        # load the data
                                        print 'loading '+i+' from '+SRC_DIRECTORY
                                        # print 'type(i): '+str(type(i))
                                        modis_obj = df.DataField(\
                                                                 datafilename=i\
                                                                 ,datafieldname='Water_Vapor_Infrared'\
                                                                 ,srcdirname=SRC_DIRECTORY\
                                                                 ,hack_branchcut_threshold=200\
                                        )
                                        modis_objs.append(modis_obj)
    
            #
            # Get the sizes, allocate the arrays, and then fill them
            sizes_modis_objs      = [ m.data.size for m in modis_objs ]
            total_size_modis_objs = sum(sizes_modis_objs)
    
            data      = np.zeros(total_size_modis_objs)
            latitude  = np.zeros(total_size_modis_objs)
            longitude = np.zeros(total_size_modis_objs)
 
            i0=0
            for i in range(len(sizes_modis_objs)):
                i1 = i0 + sizes_modis_objs[i]
                print('adding indexes: ', i0, i1)
                data[i0:i1],latitude[i0:i1],longitude[i0:i1] = modis_objs[i].ravel()
                i0=i1
    
                
            #
            idx = np.where(~np.isnan(data))
            data1      = data[idx]
            latitude1  = latitude[idx]
            longitude1 = longitude[idx]

        
            # modis_obj_1 = DataField(data=data1,latitude=latitude1,longitude=longitude1)
            # modis_obj_1.scatterplot(vmin=1.0,vmax=4.0,title='scatter')
            # modis_obj_1.colormesh(vmin=1.0,vmax=3.0,title='scatter')
            
            #### # Construct the target grid and related arrays for the kriging
            #### kb_lons,kb_lats = krigeBox.lons_lats()
            #### kb_dx = 0.5
            #### kb_dy = 0.5
            #### grid = Krige.rectangular_grid(\
            ####                                x0  = kb_lons[0]\
            ####                                ,x1 = kb_lons[1]\
            ####                                ,dx = kb_dx\
            ####                                ,y0 = kb_lats[0]\
            ####                                ,y1 = kb_lats[1]\
            ####                                ,dy = kb_dy\
            #### )

            #### See start of loop... grid = targetTiles[k]
            gridx,gridy = grid.gridxy()
            in_grid = grid.in_grid(longitude1,latitude1)
            ex_grid = grid.ex_grid(longitude1,latitude1)

            data_mx_in_grid = np.nanmax(data1[in_grid])
        
            # Calculate variogram
    
            nlags=_drive_OKrige_nlags
            ### custom_args = None
            ### 
            ### # A gamma-rayleigh distribution
            ### def custom_vg(params,dist):
            ###     sill    = np.float(params[0])
            ###     falloff = np.float(params[1])
            ###     beta    = np.float(params[2])
            ###     fd      = falloff*dist
            ###     omfd    = 1.0-falloff*dist
            ###     bfd2    = beta*omfd*omfd
            ###     return \
            ###         sill*fd*np.exp(omfd-bfd2)
    
            #
            dg = gridx.size
            dx = Krige.span_array(gridx)
            dy = Krige.span_array(gridy)
            dr = math.sqrt(dx*dx+dy*dy)

            # tile 17 ctr: 105W 75N.
            # (sqrt 1800) -> 42
            # 75-42 = 33 - check
            # 75+42 = 117
    
            # smaller beta => tighter sample distribution
            # beta0=0.5*(dr)
            # beta0=0.75*(dr)
            # beta0=1.0*(dr)
            # beta0=1.25*(dr)
            #+ beta0=1.5*(dr)
            beta0=1.5*(dr)
            # lw_scale = 1.2
            # lw_scale = 1.5
            # lw_scale = 2.0
            # lw_scale = 2.25
            lw_scale = 2.5
            # lw_scale = 3.0
            l=lw_scale*(dx/2)
            w=lw_scale*(dy/2)
            # npts = 1000
            # npts = 1500
            # npts = 2000
            # npts = 2500
            # npts = 3000
            # npts = 4000
            # npts = 5000
            # npts = 6000
            # npts = 7000
            # npts = 8000

            # if k == 11 or k == 17:
            #     # 27b npts = 8000
            #     npts = 10000
            #     # 27c npts = 12000
            # else:
            #     # 27.a-c npts = 4000
            #     npts = 2000

            # 27d hires_calc = [2,5,11,17,49,54,60,71] 12000/2000
            # hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
            # if k in hires_calc:
            #     npts = 16000
            # else:
            #     # 27.a-c npts = 4000
            #     npts = 2000

            # hires_calc = [0,2,5,11,17,20,35,39,49,54,60,71]
            if k in hires_calc:
                npts = hires_npts
            else:
                npts = lores_npts
            
                
            print 'npts( '+str(k)+' ) = '+str(npts)
            
            marker_size = 3.5
            m_alpha = 1.0
            colormap_0 = plt.cm.rainbow
            colormap_1 = plt.cm.gist_yarg
            colormap_2 = plt.cm.plasma
            colormap_x = colormap_0
            
            vmin=1.0; vmax=5.0
            # vmin=np.nanmin(data1); vmax=np.nanmax(data1)

            inf_detected = False
            max_iter = max_iter_start
            npts_in = npts
            npts_increase_flag = True
            npts_last_kr_s     = 0
            
            while( True ):
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
                kr.title         = modis_obj.datafilename
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

            print 'k,mnmx(gridz): '+str(k)\
                +', ( '+str(np.nanmin(krigeSketch_results[-1].z))\
                +', '+str(np.nanmax(krigeSketch_results[-1].z))+' )'
    
            if k == _capture_k:
                print 'capturing k = '+str(k)
                _capture_z         = np.copy(krigeSketch_results[_capture_k].z)
                _capture_ss        = np.copy(krigeSketch_results[_capture_k].s)
                _capture_x         = np.copy(gridx)
                _capture_y         = np.copy(gridy)
                _capture_data1     = np.copy(data1)
                _capture_x1        = np.copy(longitude1)
                _capture_y1        = np.copy(latitude1)
                _capture_ex_grid   = np.copy(ex_grid)
                _capture_in_grid   = np.copy(in_grid)
                _capture_kr        = krigeSketch_results[_capture_k]
                _capture_data_x    = np.copy(krigeSketch_results[_capture_k].src_x)
                _capture_data_y    = np.copy(krigeSketch_results[_capture_k].src_y)
                _capture_data_z    = np.copy(krigeSketch_results[_capture_k].src_z)

            if k >= _capture_k:
                if np.nanmax(_capture_z) != np.nanmax(krigeSketch_results[_capture_k].z):
                    print 'ERROR at k = '+str(k)
                    print 'ERROR _c_k = '+str(_capture_k)
                    print 'ERROR _c_z != ksr[_c_k]: '\
                        +str(np.nanmax(_capture_z))\
                        +' != '+str(np.nanmax(np.nanmax(krigeSketch_results[_capture_k].z)))

if _capture_k < len(krigeSketch_results):
    print '_c_k,mnmx(_capture_z): '\
        +str(_capture_k)+', ( '+str(np.nanmin(_capture_z))\
        +', '+str(np.nanmax(_capture_z))+' )'
    if _capture_only:
        print '_c_k,mnmx(kr[c_k].z): '\
            +str(0)+', ( '+str(np.nanmin(krigeSketch_results[0].z))\
            +', '+str(np.nanmax(krigeSketch_results[0].z))+' )'
    else:
        print '_c_k,mnmx(kr[c_k].z): '\
            +str(_capture_k)+', ( '+str(np.nanmin(krigeSketch_results[_capture_k].z))\
            +', '+str(np.nanmax(krigeSketch_results[_capture_k].z))+' )'

if True:
    l=0
    for k in krigeSketch_results:
        print 'mnmx(k): '+str(l)+': ( '+str(np.nanmin(k.z))+' '+str(np.nanmax(k.z))+str(' )')
        l=l+1
    l=0
    for k in krigeSketch_results:
        print 'vg_parm: '+str(l)+': '+str(k.vg_parameters)
        l=l+1

plot_configuration = Krige.krigePlotConfiguration(source_data              = False\
                                                  ,source_data_last_sample = False\
                                                  ,title                   = '.'.join(krigeSketch_results[0].title.split('.',3)[0:2])\
                                                  ,zVariableName           = krigeSketch_results[0].zVariableName\
                                                  ,vmap                    = Krige.log10_map\
                                                  ,meridians_and_parallels = False\
)

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
    output_filename = "KrigeSketch.hdf"
    if '.hdf.hdf' in output_filename[-8:]:
        output_filename = output_filename[:-4]

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
