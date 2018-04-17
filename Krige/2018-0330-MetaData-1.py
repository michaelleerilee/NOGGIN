#!/opt/local/bin/python

import os
import sys

SRC_DIRECTORY='MODIS-61/'

src_file_list = os.listdir(SRC_DIRECTORY)
src_file_list = src_file_list[0:4]

for i in src_file_list:
    print ('loading ',i)
    modis_obj = MODIS_DataField(\
                                    datafilename=i\
                                    ,datafieldname='Water_Vapor_Infrared'\
                                    ,srcdirname=SRC_DIRECTORY\
                                    )
    
    


