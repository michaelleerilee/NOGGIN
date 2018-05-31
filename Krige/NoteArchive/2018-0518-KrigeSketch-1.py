#!/usr/bin/env python
"""Krige to a box using data catalogued in the metadata file.

2018-0518-1923 ML Rilee, RSTLLC, mike@rilee.net

"""

import json
import math
import os
import sys
from time import gmtime, strftime

import noggin
import MODIS_DataField as mdf

# from MODIS_DataField import MODIS_DataField, BoundingBox, mdf.Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.spatial import ConvexHull

_verbose=True
_debug  =True

_plot_source_data_outside_grid = False
_plot_kriged                   = True
_plot_kriged_outline           = True
_plot_variogram                = False

_capture_only      = False
_capture_k         = 31
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
SRC_DIRECTORY_BASE=mdf.data_src_directory()
SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
SRC_METADATA=SRC_DIRECTORY+'modis_BoundingBoxes.json'

class krigeResults(object):
    def __init__(self\
                 ,s=None,z=None,x=None,y=None\
                 ,hull=None\
                 ,box=None\
                 ,note='Default note for krigeResults'):
        self.clear()
        if x is not None:
            self.x=x.copy()
        if y is not None:
            self.y=y.copy()
        if z is not None:
            self.z=z.copy()
        if s is not None:
            self.s=s.copy()
        if hull is not None:
            self.hull = hull
        if box is not None:
            self.box = box.copy()
        else:
            self.box = mdf.BoundingBox()
        self.note = str(note)
        self.sort_on_longitude()
        self.construct_hull()
    def sort_on_longitude(self):
        """Sort data to avoid bug in basemap. TODO: move into __init__."""
        if self.x is not None \
           and self.y is not None \
           and self.z is not None:
            idx = self.x.argsort()
            self.y = self.y[idx[::-1]]
            self.z = self.z[idx[::-1]]
            self.x = self.x[idx[::-1]]
            # self.y = [y for _,y in sorted(zip(self.x,self.y))]
            # self.z = [z for _,z in sorted(zip(self.x,self.z))]
            # self.x = sorted(self.x)
        # else fail silently
    def clear(self):
        self.x = None
        self.y = None
        self.z = None
        self.s = None
        self.hull = None
        self.box = mdf.BoundingBox()
        self.note = 'Default note for krigeResults'
    def construct_hull(self):
        """Construct the hull from z,x,y. z is used to get the shape of the data, 
        so it could be replaced using x and y alone."""
        if self.z is not None \
           and self.x is not None \
           and self.y is not None:
            xy1 = np.zeros((self.z.shape[0],2))
            xy1[:,0] = self.x
            xy1[:,1] = self.y
            self.hull = ConvexHull(xy1)

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
# Load the metadata and convert to a dictionary of boxes
print 'loading metadata '+SRC_METADATA
with open(SRC_METADATA,'r') as f:
    boxes_json = json.load(f)
boxes = {}
for i,v in boxes_json.iteritems():
    lons,lats = mdf.BoundingBox().from_json(v).lons_lats()
    boxes[i] = mdf.box_covering(lons,lats,hack_branchcut_threshold=180.0)


# Choose box to krige
# Memory Error
# krigeBox = mdf.BoundingBox((mdf.Point((+125.0, 30.0))\
#                             ,mdf.Point((+185.0, 60.0))))
#
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

krigeSketch_results = []

# loading_buffer = 5 # for adding some margin to ensure good coverage
dLon = 30
dLat = 30
dSearch = 0.75*dLon
k = -1
#+
### Loop over the tiles.
### 
for iLon in range(-180,180,dLon):
# for iLon in range(-180,-180+dLon,dLon):
#     for jLat in range(-90,90,dLat):
#+    for jLat in range(-90,90,dLat):
#    for jLat in range(-90,-90+dLat,dLat):
    for jLat in range(-90,90,dLat):
        k=k+1
        _calculate = False
        if not _capture_only:
            _calculate = True
        else:
            if _capture_k == k:
                _calculate = True
        if _calculate:
            print 'working on tile = '+str(k)
            print 'loading iLon,jLat: '+str(iLon)+','+str(jLat)
            print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
            
            krigeBox = mdf.BoundingBox((mdf.Point((iLon, jLat))\
                                        ,mdf.Point((iLon+dLon, jLat+dLat))))
    
            searchBox = mdf.BoundingBox((mdf.Point((iLon-dSearch,      max(-90, min(jLat-dSearch, 90))))\
                                        ,mdf.Point((iLon+dLon+dSearch, max(-90, min(jLat+dLat+dSearch, 90))))))
    
            # krigeBox = mdf.BoundingBox((mdf.Point((+135.0, 30.0))\
            #                            ,mdf.Point((+175.0, 45.0))))
    
    
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
                                    src_data[i] = v
                                    # load the data
                                    print 'loading '+i+' from '+SRC_DIRECTORY
                                    # print 'type(i): '+str(type(i))
                                    modis_obj = mdf.MODIS_DataField(\
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
        
            # modis_obj_1 = MODIS_DataField(data=data1,latitude=latitude1,longitude=longitude1)
            # modis_obj_1.scatterplot(vmin=1.0,vmax=4.0,title='scatter')
            # modis_obj_1.colormesh(vmin=1.0,vmax=3.0,title='scatter')
            
            # Construct the target grid and related arrays for the kriging
            kb_lons,kb_lats = krigeBox.lons_lats()
            kb_dx = 0.5
            kb_dy = 0.5
            grid = noggin.rectangular_grid(\
                                           x0  = kb_lons[0]\
                                           ,x1 = kb_lons[1]\
                                           ,dx = kb_dx\
                                           ,y0 = kb_lats[0]\
                                           ,y1 = kb_lats[1]\
                                           ,dy = kb_dy\
            )
            gridx,gridy = grid.gridxy()
            in_grid = grid.in_grid(longitude1,latitude1)
            ex_grid = grid.ex_grid(longitude1,latitude1)
    
            # Target results
            gridz  = np.zeros(gridx.shape)
            gridss = np.zeros(gridx.shape)
    
            # Calculate variogram
    
            nlags=12
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
    
            #
            dg=gridx.size
    
            dx = noggin.span_array(gridx)
            dy = noggin.span_array(gridy)
            dr = math.sqrt(dx*dx+dy*dy)
    
            # smaller beta => tighter sample distribution
            # beta0=0.5*(dr)
            # beta0=0.75*(dr)
            # beta0=1.0*(dr)
            # beta0=1.25*(dr)
            #+ beta0=1.5*(dr)
            # beta0=1.5*(dr)
            beta0=1.5*(dr)
            # lw_scale = 1.2
            # lw_scale = 1.5
            lw_scale = 2.5
            l=lw_scale*(dx/2)
            w=lw_scale*(dy/2)
            # npts = 1000
            npts = 1500
    
            marker_size = 3.5
            m_alpha = 1.0
            colormap_0 = plt.cm.rainbow
            colormap_1 = plt.cm.gist_yarg
            colormap_2 = plt.cm.plasma
            colormap_x = colormap_0
            vmin=1.0; vmax=5.0
            # vmin=np.nanmin(data1); vmax=np.nanmax(data1)
    
            gridz, data_x, data_y, data_z = noggin.drive_OKrige(\
                                                                grid_stride=dg\
                                                                ,random_permute=True\
                                                                ,x=gridx,y=gridy\
                                                                ,src_x=longitude1\
                                                                ,src_y=latitude1\
                                                                ,src_z=data1\
                                                                ,variogram_model='custom'\
                                                                ,variogram_parameters=custom_args\
                                                                ,variogram_function=custom_vg\
                                                                ,enable_plotting=_plot_variogram
                                                                ,npts=npts
                                                                ,beta0=beta0
                                                                ,frac=0.0
                                                                ,l=l,w=w
            )
    
            # krige_result = krigeResults(x=gridx,y=gridy,z=gridz,box=krigeBox)
            # krigeSketch_results.append(krige_result)
            krigeSketch_results.append(krigeResults(x=gridx,y=gridy,z=gridz,box=krigeBox))
            # xy1 = np.zeros((gridz.shape[0],2))
            # xy1[:,0] = gridx
            # xy1[:,1] = gridy
            # grid_hull = ConvexHull(xy1)
            print 'k,mnmx(gridz): '+str(k)+', ( '+str(np.nanmin(gridz))+', '+str(np.nanmax(gridz))+' )'
    
            if k == _capture_k:
                print 'capturing k = '+str(k)
                _capture_z         = np.copy(gridz)
                _capture_ss        = np.copy(gridss)
                _capture_x         = np.copy(gridx)
                _capture_y         = np.copy(gridy)
                _capture_data1     = np.copy(data1)
                _capture_x1        = np.copy(longitude1)
                _capture_y1        = np.copy(latitude1)
                _capture_ex_grid   = np.copy(ex_grid)
                _capture_in_grid   = np.copy(in_grid)
                _capture_kr        = krigeSketch_results[_capture_k]
                _capture_data_x    = np.copy(data_x)
                _capture_data_y    = np.copy(data_y)
                _capture_data_z    = np.copy(data_z)

            if k >= _capture_k:
                if np.nanmax(_capture_z) != np.nanmax(krigeSketch_results[_capture_k].z):
                    print 'ERROR at k = '+str(k)
                    print 'ERROR _c_k = '+str(_capture_k)
                    print 'ERROR _c_z != ksr[_c_k]: '+str(np.nanmax(_capture_z))+' != '+str(np.nanmax(np.nanmax(krigeSketch_results[_capture_k].z)))


print '_c_k,mnmx(_capture_z): '+str(_capture_k)+', ( '+str(np.nanmin(_capture_z))+', '+str(np.nanmax(_capture_z))+' )'
if _capture_only:
    print '_c_k,mnmx(kr[c_k].z): '+str(0)+', ( '+str(np.nanmin(krigeSketch_results[0].z))+', '+str(np.nanmax(krigeSketch_results[0].z))+' )'
else:
    print '_c_k,mnmx(kr[c_k].z): '+str(_capture_k)+', ( '+str(np.nanmin(krigeSketch_results[_capture_k].z))+', '+str(np.nanmax(krigeSketch_results[_capture_k].z))+' )'
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'KrigeSketch done'
