#!/bin/bash

lonA=19
lonB=29

latA=-27
latB=80

dlon=1
dlat=1

resolution=0.05

for lat0 in `seq $latA $dlat $latB`; do
    for lon0 in `seq $lonA $dlon $lonB`; do
	# echo $dlon $dlat
	(( lon1 = lon0 + dlon ))
	(( lat1 = lat0 + dlat ))
	./run-vnp02.sh $lon0 $lat0 $lon1 $lat1 $resolution
    done
done
