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
        self.construct_hull()
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
print 'loading '+SRC_METADATA
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

krige_results = []

# loading_buffer = 5 # for adding some margin to ensure good coverage
dLon = 20
dLat = 10
for iLon in range(-180,180,dLon):
    for jLat in range(-90,90,dLat):
        print 'loading iLon,jLat: '+str(iLon)+','+str(jLat)
        print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        
        krigeBox = mdf.BoundingBox((mdf.Point((iLon, jLat))\
                                    ,mdf.Point((iLon+dLon, jLat+dLat))))

        # krigeBox = mdf.BoundingBox((mdf.Point((+135.0, 30.0))\
        #                            ,mdf.Point((+175.0, 45.0))))


        # Find overlapping granules and load the MODIS objects
        src_data   = {}
        modis_objs = []
        for i,v in boxes.iteritems():
            i = str(i)
            lons,lats = v.lons_lats()
            if (np.nanmax(np.abs(lons)) <= 360.0)\
               and (np.nanmax(np.abs(lats)) <= 90.0):
                o = krigeBox.overlap(v)
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
        #+ beta0=1.5*(dr)
        # beta0=0.5*(dr)
        beta0=0.75*(dr)
        # beta0=1.0*(dr)
        l=1.2*(dx/2)
        w=1.2*(dy/2)

        marker_size = 3.5
        m_alpha = 1.0
        colormap_0 = plt.cm.rainbow
        colormap_1 = plt.cm.gist_yarg
        colormap_2 = plt.cm.plasma
        colormap_x = colormap_0
        # vmin=1.0; vmax=5.0
        vmin=np.nanmin(data1); vmax=np.nanmax(data1)

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
                                                            ,npts=1000
                                                            ,beta0=beta0
                                                            ,frac=0.0
                                                            ,l=l,w=w
        )

        krige_result = krigeResults(x=gridx,y=gridy,z=gridz,box=krigeBox)
        krige_results.append(krige_result)
        # xy1 = np.zeros((gridz.shape[0],2))
        # xy1[:,0] = gridx
        # xy1[:,1] = gridy
        # grid_hull = ConvexHull(xy1)
        kr = krige_result

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plot results'
#### PLOT RESULTS ####
#
# fig = plt.gcf()
fig_gen = noggin.fig_generator(1,1)
fig,ax = plt.subplots(1,1)
_scale = 2.0*np.pi
wh_scale = [_scale,_scale]
lon_0,lat_0 = krigeBox.centroid().inDegrees()
# m = Basemap(projection='laea', resolution='l', lat_ts=65\
    #            ,width=wh_scale[0]*3000000,height=wh_scale[1]*2500000)
m = Basemap(projection='cyl',resolution='h'\
            ,ax=ax\
            ,lat_0=lat_0, lon_0=lon_0)
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(50., 91., 10.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180, 181., 30), labels=[0, 0, 0, 1])

# if _plot_source_data_outside_grid:
#     modis_obj_2 = mdf.MODIS_DataField(\
#                                       data=data1[ex_grid]\
#                                       ,latitude=latitude1[ex_grid]\
#                                       ,longitude=longitude1[ex_grid])
#     modis_obj_2.scatterplot(m=m\
#                             ,title='scatter'\
#                             ,plt_show = False\
#                             ,vmin=vmin,vmax=vmax\
#                             ,cmap=colormap_x\
#     )
        
if _plot_kriged:
    for kr in krige_results:
        #
        # m = modis_obj_2.get_m()
        m.drawmapboundary(fill_color='dimgrey')
        # m.scatter(gridx,gridy,c=gridz
        m.scatter(kr.x,kr.y,c=kr.z\
                  ,cmap=colormap_x\
                  ,linewidth=0\
                  ,alpha=m_alpha\
                  ,latlon=True\
                  ,vmin=vmin, vmax=vmax\
                  ,edgecolors=None\
                  ,s=marker_size*10\
                  ,marker='s'\
        )
        if _plot_kriged_outline:
            noggin.draw_screen_poly( kr.x[kr.hull.vertices], kr.y[kr.hull.vertices], m )
            
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plt.show'
if True:
    plt.show()

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'done'
