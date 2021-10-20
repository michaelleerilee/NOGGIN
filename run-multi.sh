#!/bin/bash

lonA=-140
lonB=-120

# latA=-27
# latA=-15
latA=-88
latB=-68

dlon=2
dlat=2

resolution=0.05

for lat0 in `seq $latA $dlat $latB`; do
    for lon0 in `seq $lonA $dlon $lonB`; do
	# echo $dlon $dlat
	(( lon1 = lon0 + dlon ))
	(( lat1 = lat0 + dlat ))
	./run-vnp02.sh $lon0 $lat0 $lon1 $lat1 $resolution
    done
done
