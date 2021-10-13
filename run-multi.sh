#!/bin/bash

lonA=19
lonB=20

latA=-27
latB=-26

dlon=1
dlat=1

for lon0 in `seq $lonA $lonB`; do
    for lat0 in `seq $latA $latB`; do
	# echo $dlon $dlat
	(( lon1 = lon0 + dlon ))
	(( lat1 = lat0 + dlat ))
	./run-vnp02.sh $lon0 $lat0 $lon1 $lat1
    done
done
