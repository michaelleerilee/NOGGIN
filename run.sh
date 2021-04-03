#!/bin/bash

export PYTHONPATH=/Users/mrilee/git/NOGGIN-PyKrige:/Users/mrilee/git/NOGGIN
export NOGGIN_DATA_SRC_DIRECTORY=/Users/mrilee/Remote/Dropbox/data/NOGGIN/tmp/

python ~/git/NOGGIN/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}/ -n HDFEOS/SWATHS/O3Profile/Data\ Fields/O3 -m gamma_rayleigh_nuggetless_variogram_model -v

# Gapfill on a grid
# python ~/git/NOGGIN/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}/ -n Atmospheric_Water_Vapor_Mean -m gamma_rayleigh_nuggetless_variogram_model -v -G

