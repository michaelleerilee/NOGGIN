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

# from MODIS_DataField import MODIS_DataField, BoundingBox, mdf.Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.spatial import ConvexHull

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


print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
print 'plotting source data'

## Prepare plot

# fig = plt.gcf()
fig_gen = noggin.fig_generator(1,1)
fig,ax = plt.subplots(1,1)
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

m.drawmapboundary(fill_color='dimgrey')

# _load_datasets = ['MYD','MOD']
# _load_datasets = ['MYD']
_load_datasets = ['MOD']

# Choose the source directory for the data and metadata
SRC_DIRECTORY_BASE=mdf.data_src_directory()
SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
SRC_METADATA=SRC_DIRECTORY+'modis_BoundingBoxes.json'

print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
# Load the metadata and convert to a dictionary of boxes
print 'loading metadata '+SRC_METADATA
with open(SRC_METADATA,'r') as f:
    boxes_json = json.load(f)
boxes = {}
for i,v in boxes_json.iteritems():
    lons,lats = mdf.BoundingBox().from_json(v).lons_lats()
    boxes[i] = mdf.box_covering(lons,lats,hack_branchcut_threshold=180.0)


def log_map(x):
    return np.log10(x+1.0e-9)
    
# k=5
k=len(boxes_json)
boxk=-1
plotk=-1
for i,v in boxes.iteritems():
    i=str(i)
    boxk=boxk+1
    if i[0:3] in _load_datasets:
        if k>0:
            k=k-1
            plotk=plotk+1
            print 'plotk,boxk,file: '+str(plotk)+', '+str(boxk)+', '+i
            modis_obj = mdf.MODIS_DataField(\
                                            datafilename=i\
                                            ,datafieldname='Water_Vapor_Infrared'\
                                            ,srcdirname=SRC_DIRECTORY\
                                            ,hack_branchcut_threshold=200\
            )
            modis_obj.scatterplot(m=m\
                                  ,plt_show = False\
                                  ,vmin=vmin,vmax=vmax\
                                  ,cmap=colormap_x\
                                  ,marker_size=marker_size*2\
                                  ,value_map=log_map
            )
                              

plt.show()
