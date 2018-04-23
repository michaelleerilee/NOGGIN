#!/opt/local/bin/python
"""
Read datafiles to construct metadata helping comparisons and intersections.

2018-0417-1421-02-EDT ML Rilee, RSTLLC, mike@rilee.net.
"""

import os
import sys
from MODIS_DataField import MODIS_DataField

if('NOGGIN_DATA_SRC_DIRECTORY' in os.environ):
    SRC_DIRECTORY_BASE=os.environ['NOGGIN_DATA_SRC_DIRECTORY']
else:
    SRC_DIRECTORY_BASE='./'

SRC_DIRECTORY=SRC_DIRECTORY_BASE+'MODIS-61/'
    
src_file_list = [f for f in os.listdir(SRC_DIRECTORY) if (lambda x: '.hdf' in x or '.HDF.' in x)(f)]
src_file_list = src_file_list[0:4]

for i in src_file_list:
    print ('loading ',i)
    modis_obj = MODIS_DataField(\
                                    datafilename=i\
                                    ,datafieldname='Water_Vapor_Infrared'\
                                    ,srcdirname=SRC_DIRECTORY\
                                    )
