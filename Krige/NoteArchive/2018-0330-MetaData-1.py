#!/usr/bin/env python
"""Read datafiles to construct metadata helping comparisons and intersections.

Write bounding box metadata to a file, so it's not necessary to load
all of the HDF files. Currently this routine is destructive,
overwriting any existing metadata file without regard to whether or
not the existing file is useful.

2018-0417-1421-02-EDT ML Rilee, RSTLLC, mike@rilee.net.

"""

import json
import os
import sys
from MODIS_DataField import MODIS_DataField, BoundingBox, Point

_verbose=True
_debug=False

if('NOGGIN_DATA_SRC_DIRECTORY' in os.environ):
    SRC_DIRECTORY_BASE=os.environ['NOGGIN_DATA_SRC_DIRECTORY']
else:
    SRC_DIRECTORY_BASE='./'

SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
    
src_file_list = [f for f in os.listdir(SRC_DIRECTORY) if (lambda x: '.hdf' in x or '.HDF.' in x)(f)]

if _debug:
    src_file_list = src_file_list[0:4]

modis_BoundingBoxes = {}
modis_BoundingBoxes_json = {}
bb = BoundingBox()

for i in src_file_list:
    if _verbose:
        print ('loading ',i)
    modis_obj = MODIS_DataField(\
                                    datafilename=i\
                                    ,datafieldname='Water_Vapor_Infrared'\
                                    ,srcdirname=SRC_DIRECTORY\
                                    )
    bb = bb.union(modis_obj.bbox)
    modis_BoundingBoxes[i]=modis_obj.bbox
    modis_BoundingBoxes_json[i]=modis_obj.bbox.to_json()

    if _debug:
        print 'xml: ',bb.to_xml()
        print 'json: ',bb.to_json()

if _verbose:
    print 'writing to '+SRC_DIRECTORY+'modis_BoundingBoxes.json'
with open(SRC_DIRECTORY+'modis_BoundingBoxes.json','w') as f:
    json.dump(modis_BoundingBoxes_json,f)

if _debug:
    print 'json.dumps'
    print json.dumps(modis_BoundingBoxes_json)

    with open(SRC_DIRECTORY+'modis_BoundingBoxes.json','r') as f:
        boxes_json = json.load(f)
    print 'boxes_json: ',boxes_json

    boxes = {}
    # version 3 -> dict.iter()
    for i,v in boxes_json.iteritems():
        boxes[i] = BoundingBox().from_json(v)

    for j,v0 in modis_BoundingBoxes.iteritems():
        v1 = boxes[j]
        print '--v0--'
        print v0.to_xml()
        print '--v1--'
        print v1.to_xml()
        print 'delta-p0: ',v0.p0.inDegrees()[0],v1.p0.inDegrees()[0],(-v0.p0.inDegrees()[0]+v1.p0.inDegrees()[0])


if _debug:
    print '--metadata overlap empty bug test--'
    krigeBox = BoundingBox((Point((-180.0, -90.0))\
                            ,Point(( 180.0,  90.0))))
    print 'krigeBox: ',krigeBox.to_xml()
    for j,v in boxes.iteritems():
        print 'j:             ',j
        print '--v--'
        print v.to_xml()
        print 'overlap-empty? ',krigeBox.overlap(v).emptyp()
        print krigeBox.overlap(v).to_xml()

if _verbose:
    print 'finished writing MetaData'

