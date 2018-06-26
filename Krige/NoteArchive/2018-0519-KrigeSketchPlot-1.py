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

def log_map(x):
    return np.log10(x+1.0e-9)

_verbose=plot_configuration.verbose
_debug  =plot_configuration.debug

_plot_kriged                   = plot_configuration.kriged
# _plot_source_data_outside_grid = False
_plot_source_data              = plot_configuration.source_data
_plot_source_data_last_sample  = plot_configuration.source_data_last_sample
_plot_kriged_data              = plot_configuration.kriged_data
_plot_kriged_outline           = plot_configuration.kriged_outline
_plot_variogram                = plot_configuration.variogram

# marker_size = 3.5
# marker_size = 1
# For 0.5deg map: marker_size = 0.5
# For 1 deg map: marker_size = 1.75
marker_size = plot_configuration.marker_size

m_alpha = plot_configuration.m_alpha
colormap_0 = plt.cm.rainbow
colormap_1 = plt.cm.gist_yarg
colormap_2 = plt.cm.plasma
colormap_x = plot_configuration.colormap
# vmin=2.0; vmax=20.0
# vmin=2.0; vmax=10.0
# vmin=0.5; vmax=8.0
# vmin=0.25; vmax=6.0
# vmin=0.5; vmax=5.0
# vmin=np.nanmin(data1); vmax=np.nanmax(data1)
# For log10
# vmin=-2.0; vmax=1.0
#+ vmin=-2.0; vmax=1.25
vmin = plot_configuration.vmin
vmax = plot_configuration.vmax

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plot results'
#### PLOT RESULTS ####
#
# fig = plt.gcf()

fig_gen = noggin.fig_generator(1,1,figsize=(2560.0/110.0,1300.0/110.0),dpi=110.0)
# fig_gen = noggin.fig_generator(1,1)
# fig,ax = plt.subplots(1,1)
fig,ax = fig_gen.get_fig_axes()
ax = ax[0]

_scale = 2.0*np.pi
wh_scale = [_scale,_scale]
# lon_0,lat_0 = krigeBox.centroid().inDegrees()
lon_0,lat_0 = (0,0)
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
_plot_list=[49]
_plot_item=True
_plot_kriged_data_index = True

if _plot_kriged:
    k=0
    for kr in krigeSketch_results:
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
            # m.scatter(kr.x,kr.y,c=kr.z\
            # m.scatter(kr.x,kr.y,c=np.log10(kr.z+1.0e-9)\
            # m.scatter(kr.x,kr.y,c=kr.z
            #    m.scatter(kr.x,kr.y,c=np.log10(kr.z+1.0e-9)\
            if _plot_kriged_data:
                m.scatter(kr.x,kr.y,c=log_map(kr.z)\
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
                #if xt < 0.0:
                #    xt = xt+360.0
                # xt, yt = ( np.sum(kr.x)/kr.x.size, np.sum(kr.y)/kr.y.size )
                plt.text(xt,yt,str(k),color='green')
            if _plot_source_data and kr.dbg:
                
                #                                  data=np.log10(kr.src_z+1.0e-9)\
                modis_obj_2 = mdf.MODIS_DataField(\
                                                  data=kr.dbg_z\
                                                  ,latitude=kr.dbg_y\
                                                  ,longitude=kr.dbg_x\
                                                  ,title=kr.title\
                                                  ,long_name=kr.zVariableName\
                                                  ,colormesh_title=kr.title\
                )
                
                modis_obj_2.scatterplot(m=m\
                                        ,plt_show = False\
                                        ,vmin=vmin,vmax=vmax\
                                        ,cmap=colormap_x\
                                        ,marker_size=marker_size*10\
                                        ,value_map=log_map\
                )
            if _plot_source_data_last_sample:
                
                #                                  data=np.log10(kr.src_z+1.0e-9)\
                modis_obj_3 = mdf.MODIS_DataField(\
                                                  data=kr.src_z\
                                                  ,latitude=kr.src_y\
                                                  ,longitude=kr.src_x\
                                                  ,title=kr.title\
                                                  ,long_name=kr.zVariableName\
                                                  ,colormesh_title=kr.title\
                )
                modis_obj_3.scatterplot(m=m\
                                        ,plt_show = False\
                                        ,vmin=vmin,vmax=vmax\
                                        ,cmap=colormap_x\
                                        ,marker_size=marker_size*10\
                                        ,value_map=log_map\
                )
                
                
        # sys.stdout.write(', plotting');
        print '.'
        k=k+1
            
print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plt.show'
# basename = os.path.basename(kr.title)
plt.title('{0}\n{1}'.format(plot_configuration.title, plot_configuration.zVariableName))
if True:
    plt.show()

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'done'
