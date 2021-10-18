#!/bin/bash

# Example of Kriging L2 data onto a grid
#
# Data from https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMO3PR.003/2021/
#
# Note we interact with files as HDF5 not HDF5-EOS. This just means we interact with the
# data structures at a lower level.
#

usage() {
    echo
    echo "$0"
    echo "  Run noggin_krige over a region of interest. "
    echo
    echo "  Bounding box is required."
    echo
    echo "Example:"
    echo "  $0 lon0 lat0 lon1 lat1 resolution"
    echo
    echo "where (lon0,lat0) and (lon1,lat1) are the lower left"
    echo "and upper right points of the box in  integer degrees."
    echo
    exit
    }

[[ $# -lt 5 ]] && usage

#
export PYTHONPATH=/home/mrilee/git/NOGGIN-PyKrige:/home/mrilee/git/NOGGIN
export NOGGIN_DATA_SRC_DIRECTORY=/home/mrilee/data/VIIRS

## lon0=+20
## lat0=-26
## lon1=+21
## lat1=-25

lon0=$1
lat0=$2
lon1=$3
lat1=$4
resolution=$5

# echo "-b ${lon0} ${lat0} ${lon1} ${lat1}"
# outfile=`printf "noggin_krige_${lon0}_${lat0}_${lon1}_${lat1}.h5"`
echo
echo run-vnp02.sh $1 $2 $3 $4 $5
echo
outfile=`printf "noggin_krige_%04d_%04d_%04d_%04d.h5" ${lon0} ${lat0} ${lon1} ${lat1}`
echo "outfile: ${outfile}"
# exit 0

# Execute the calculation. Krige to a default 1-degree lon-lat grid.
#
# CLI Options
# -d <source data directory>
# -n <variable to extract>  # Note the low-level style of access via HDF-5
# -m <variogram functional model>
# -v # Verbose output
# -l <number of lags in variogram fit>
#
python ~/git/NOGGIN/Krige/noggin_krige.py \
       -f src_file_list \
       -d ${NOGGIN_DATA_SRC_DIRECTORY}/ \
       -n observation_data/l05 \
       -m spherical \
       -R -b ${lon0} ${lat0} ${lon1} ${lat1} \
       -r ${resolution} \
       -S 2000 \
       -l 4 \
       --lw_scale 2 \
       --Beta 3.0 \
       -v -x \
       --output_filename ${outfile}

# Betas...
# 1.5 lots of divergences

# lw_scale
#

# Gapfill on a grid
# python ~/git/NOGGIN/Krige/noggin_krige.py -d ${NOGGIN_DATA_SRC_DIRECTORY}/ -n Atmospheric_Water_Vapor_Mean -m gamma_rayleigh_nuggetless_variogram_model -v -G

# Maybe try -m spherical, might be more robust...
#       -m gamma_rayleigh_nuggetless_variogram_model \
#       -m spherical \
