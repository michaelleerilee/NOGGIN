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
_plot_kriged_outline           = False
_plot_variogram                = False

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
vmin=0.25; vmax=6.0
# vmin=0.5; vmax=5.0
# vmin=np.nanmin(data1); vmax=np.nanmax(data1)


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
m.drawparallels(np.arange(-90.0, 91., 10.), labels=[1, 0, 0, 0])
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

m.drawmapboundary(fill_color='dimgrey')

_plot_from_list = False
_plot_list=[144]
_plot_item=True
_plot_kriged_data_index = True

if _plot_kriged:
    k=0
    for kr in krige_results:
        sys.stdout.write('k = '+str(k))
        if _plot_from_list:
            if k in _plot_list:
                _plot_item = True
            else:
                _plot_item = False
                #
        # m = modis_obj_2.get_m()
        # m.scatter(gridx,gridy,c=gridz
        if _plot_item:
            sys.stdout.write(', plotting');
            # y1 = [y for _,y in sorted(zip(kr.x,kr.y))]
            # z1 = [z for _,z in sorted(zip(kr.x,kr.z))]
            # x1 = sorted(kr.x)
            # m.scatter(x1,y1,c=z1
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
            if _plot_kriged_data_index:
                # xt, yt = m( np.sum(kr.x)/kr.x.size, np.sum(kr.y)/kr.y.size )
                xt, yt = m( np.nanmin(kr.x), np.sum(kr.y)/kr.y.size )
                if xt < 0.0:
                    xt = xt+360.0
                # xt, yt = ( np.sum(kr.x)/kr.x.size, np.sum(kr.y)/kr.y.size )
                plt.text(xt,yt,str(k),color='green')
        # sys.stdout.write(', plotting');
        print '.'
        k=k+1
            
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plt.show'
if True:
    plt.show()

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'done'
