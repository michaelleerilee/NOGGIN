#!/usr/bin/env python
"""Use the saved metadata to select which files to load and Krige.

2018-0430-1552 ML Rilee, RSTLLC, mike@rilee.net.

"""

import json
import os
import sys
import Krige
from Krige.DataField import DataField, BoundingBox, Point, box_covering, Polygon, data_src_directory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

_verbose=True
_debug=False

#if('NOGGIN_DATA_SRC_DIRECTORY' in os.environ):
#    SRC_DIRECTORY_BASE=os.environ['NOGGIN_DATA_SRC_DIRECTORY']
#else:
#    SRC_DIRECTORY_BASE='./'

SRC_DIRECTORY_BASE=data_src_directory()
SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
SRC_METADATA=SRC_DIRECTORY+'modis_BoundingBoxes.json'

print 'loading '+SRC_METADATA
with open(SRC_METADATA,'r') as f:
    boxes_json = json.load(f)

boxes = {}
for i,v in boxes_json.iteritems():
    lons,lats = BoundingBox().from_json(v).lons_lats()
    boxes[i] = box_covering(lons,lats,hack_branchcut_threshold=180.0)

krigeBox = BoundingBox((Point((-180.0, 30.0))\
                       ,Point((+180.0, 40.0))))

# krigeBox = BoundingBox((Point((-80.0, 30.0))\
#                        ,Point((+80.0, 40.0))))

# krigeBox = BoundingBox((Point((-175, 30.0))\
#                        ,Point(( 175, 40.0))))


src_data = {}
for i,v in boxes.iteritems():
    lons,lats = v.lons_lats()
    if (np.nanmax(np.abs(lons)) <= 360.0)\
       and (np.nanmax(np.abs(lats)) <= 90.0):
        o = krigeBox.overlap(v)
        # print i
        # print o.to_xml()
        # print krigeBox.to_xml()
        # print v.to_xml()
        # print '---'
        if len(o) > 0:
            for kb in o:
                if not kb.emptyp():
                    # not krigeBox.overlap(v).emptyp():
                    print 'adding '+i+' : '+str(v.lons_lats())
                    src_data[i] = v
                    # print '-v-'
                    # print v.to_xml()
                    # print '---'
                    #
    else:
        print 'warning: not adding: '+i+' : '+str(v.lons_lats())

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

from matplotlib.patches import Polygon
def draw_screen_poly( lons, lats, m, facecolor='black', edgecolor='black', fill=False ):
    x, y = m( lons, lats )
    plt.gca().add_patch(Polygon( zip(x,y)\
                                 ,facecolor=facecolor\
                                 ,edgecolor=edgecolor\
                                 ,alpha=0.8, fill=fill))

krig_poly = krigeBox.polygon()

if len(krig_poly) > 0:
    for kp in krig_poly:
        krig_poly_lons,krig_poly_lats = kp.polygon()
        draw_screen_poly( krig_poly_lons,krig_poly_lats,m, facecolor='blue', edgecolor='black', fill=True )

for i,v in src_data.iteritems():
    v_poly = v.polygon()
    if len(v_poly) > 0:
        for vp in v_poly:
            v_lons,v_lats = vp.polygon()
            draw_screen_poly( v_lons, v_lats, m, facecolor='black', edgecolor='red' )

plot_test_box=False
if plot_test_box:
    roi_box = BoundingBox(( Point((  175, -30.0))\
                            ,Point(( -175, -40.0))))
    roi_poly = roi_box.polygon()
    for rp in roi_poly:
        r_lons,r_lats = rp.polygon()
        draw_screen_poly( r_lons, r_lats, m, facecolor='black', edgecolor='blue' )


plt.show()

